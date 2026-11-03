from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from astro_project.core.classifier import RealBogusCNN, extract_stamps
from astro_project.core.fits_loader import load_fits_image
from astro_project.core.orbit import (
    compute_tracklet_endpoints,
    estimate_kepler_orbit,
    export_tracklet_mpc80,
)
from astro_project.core.photometry import estimate_background
from astro_project.core.stellarium import slew_stellarium_to_candidate
from astro_project.core.synthetic import generate_demo_asteroid_series
from astro_project.core.wcs_utils import create_synthetic_wcs
from astro_project.gui.components import (
    _plot_comparison_frame,
    _plot_solar_system_orbit,
    _plot_stamp_strip,
    show_parameter_help_dialog,
)
from astro_project.pipeline.kinematics import kinematic_shift_and_stack, mask_stationary_stars
from astro_project.pipeline.orchestration import run_pipeline
from astro_project.pipeline.tracker import predict_boundary_exit


def render_tab_pipeline_grid_search(device: torch.device):
    """Velocity Grid Search & Orbit Analysis"""
    hdr_col1, hdr_col2 = st.columns([3, 1])
    with hdr_col1:
        st.subheader("\U0001f680 Velocity Grid Search & Orbit Analysis")
    with hdr_col2:
        if st.button("\u2753 Parameter-Hilfe \u00f6ffnen", key="help_btn_tab2", use_container_width=True):
            show_parameter_help_dialog()

    series_demo_dir = Path("data/series_demo")
    default_data_dir = Path("data")
    series_demo_dir.mkdir(parents=True, exist_ok=True)
    default_data_dir.mkdir(parents=True, exist_ok=True)

    selected_folder = st.radio(
        "Pipeline-Datenquelle w\u00e4hlen",
        options=["data/series_demo/", "data/"],
        horizontal=True,
        help="W\u00e4hlt den Quellordner f\u00fcr die FITS-Zeitreihe: Synthetische Demo-Serie oder reale Beobachtungsdaten in data/.",
    )

    if selected_folder == "data/":
        uploaded_fits_list = st.file_uploader(
            "\U0001f4c2 Reale FITS-Zeitreihe hochladen (.fits / .fit / .fts)",
            type=["fits", "fit", "fts"],
            accept_multiple_files=True,
            key="pipeline_multi_fits_uploader",
            help="Speichert hochgeladene FITS-Sequenzen im lokalen Ordner data/ f\u00fcr die Pipeline-Analyse.",
        )
        if uploaded_fits_list:
            for up_file in uploaded_fits_list:
                target_path = default_data_dir / Path(up_file.name).name
                target_path.write_bytes(up_file.getvalue())
            st.info(f"{len(uploaded_fits_list)} FITS-Datei(en) in `data/` bereitgestellt.")

    with st.expander("\u2699\ufe0f Parameter fuer synthetische Bildserie (Low-SNR)", expanded=False):
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        orbit_preset_options = {
            "Linear (Konstante Pixelgeschwindigkeit)": "linear",
            "Main-Belt Asteroid (Elliptisch, a=2.65 AU, e=0.12)": "main_belt",
            "Near-Earth Asteroid / NEA (a=1.18 AU, e=0.35)": "nea",
            "Hyperbolisch / Interstellar (a=-1.25 AU, e=1.35)": "hyperbolic",
        }
        telescope_scale_options = {
            "Standard-Survey (1.0 arcsec/px)": 1.0,
            "High-Res Astrograph (0.5 arcsec/px)": 0.5,
            "Wide-Field Survey (2.5 arcsec/px)": 2.5,
            "Deep-Space Wide-Field (15.0 arcsec/px)": 15.0,
        }
        with preset_col1:
            selected_orbit_label = st.selectbox(
                "Bahn-Modell (Kepler / Linear)",
                options=list(orbit_preset_options.keys()),
                index=0,
                help="W\u00e4hlt zwischen linearer Pixelbewegung oder physikalischer 2-K\u00f6rper-Kepler-Ephemeride (Main-Belt, NEA, Hyperbolisch).",
            )
            selected_orbit_preset = orbit_preset_options[selected_orbit_label]
        with preset_col2:
            selected_tel_label = st.selectbox(
                "Teleskop-Aufl\u00f6sung (Pixel-Skala)",
                options=list(telescope_scale_options.keys()),
                index=0,
                help="WCS-Abbildungsma\u00dfstab in Bogensekunden pro Pixel (CDELT).",
            )
            selected_pixel_scale = float(telescope_scale_options[selected_tel_label])
        with preset_col3:
            custom_dt_hours = st.slider(
                "Zeitabstand dt [h / Frame]",
                min_value=0.25,
                max_value=6.0,
                value=1.0,
                step=0.25,
                help="Zeitliche Differenz in Stunden zwischen zwei aufeinanderfolgenden Aufnahmen.",
            )
        st.markdown("**\U0001f3b2 Zufalls-Steuerung pro Parameter:**")
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            rand_frames = st.checkbox(
                "Anzahl Frames zuf\u00e4llig",
                value=False,
                key="rand_frames",
                help="Zieht die Anzahl der Frames zuf\u00e4llig im Bereich [3, 15].",
            )
        with rc2:
            rand_velocity = st.checkbox(
                "vx & vy zuf\u00e4llig",
                value=False,
                key="rand_velocity",
                help="Zieht die Geschwindigkeiten vx und vy zuf\u00e4llig im Bereich [-3.0, 3.0] px/h.",
            )
        with rc3:
            rand_snr = st.checkbox(
                "Asteroiden-SNR zuf\u00e4llig",
                value=False,
                key="rand_snr",
                help="Zieht das Einzelbild-SNR des Asteroiden zuf\u00e4llig im Bereich [1.0, 5.0].",
            )
        with rc4:
            rand_stars = st.checkbox(
                "Feldsterne zuf\u00e4llig",
                value=False,
                key="rand_stars",
                help="Zieht die Anzahl der station\u00e4ren Hintergrundsterne zuf\u00e4llig im Bereich [1, 10].",
            )

        cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
        with cfg_c1:
            custom_n_frames = st.slider(
                "Anzahl Frames",
                min_value=3,
                max_value=15,
                value=5,
                step=1,
                disabled=bool(rand_frames),
                help="Anzahl der zeitlich aufeinanderfolgenden FITS-Aufnahmen in data/series_demo/.",
            )
            custom_n_stars = st.slider(
                "Anzahl Feldsterne",
                min_value=0,
                max_value=10,
                value=5,
                step=1,
                disabled=bool(rand_stars),
                help="Anzahl der hellen station\u00e4ren Hintergrundsterne im Bildfeld.",
            )
        with cfg_c2:
            custom_vx = st.slider(
                "Geschwindigkeit vx [px/h]",
                min_value=-5.0,
                max_value=5.0,
                value=2.0,
                step=0.25,
                disabled=bool(rand_velocity),
                help="Wahre horizontale Eigenbewegung des synthetischen Asteroiden in Pixeln pro Stunde.",
            )
            custom_vy = st.slider(
                "Geschwindigkeit vy [px/h]",
                min_value=-5.0,
                max_value=5.0,
                value=-1.0,
                step=0.25,
                disabled=bool(rand_velocity),
                help="Wahre vertikale Eigenbewegung des synthetischen Asteroiden in Pixeln pro Stunde.",
            )
        with cfg_c3:
            custom_snr = st.slider(
                "Asteroiden-SNR (Einzelbild)",
                min_value=0.8,
                max_value=5.0,
                value=1.5,
                step=0.1,
                disabled=bool(rand_snr),
                help="Signal-zu-Rausch-Verh\u00e4ltnis des Asteroiden pro Einzelaufnahme.",
            )

        if st.button("\U0001f680 Serie generieren", use_container_width=True):
            try:
                rng_gen = np.random.default_rng()
                eff_n_frames = int(rng_gen.integers(3, 16)) if rand_frames else int(custom_n_frames)
                if rand_velocity:
                    eff_vx = float(rng_gen.uniform(-3.0, 3.0))
                    eff_vy = float(rng_gen.uniform(-3.0, 3.0))
                    if abs(eff_vx) < 0.4 and abs(eff_vy) < 0.4:
                        eff_vx = 1.5
                else:
                    eff_vx = float(custom_vx)
                    eff_vy = float(custom_vy)
                eff_snr = float(rng_gen.uniform(1.0, 5.0)) if rand_snr else float(custom_snr)
                eff_n_stars = int(rng_gen.integers(1, 11)) if rand_stars else int(custom_n_stars)
                any_rand = bool(rand_frames or rand_velocity or rand_snr or rand_stars)
                seed_val = int(rng_gen.integers(1, 2**31 - 1)) if any_rand else 42

                with torch.no_grad():
                    paths, times, gt = generate_demo_asteroid_series(
                        output_dir="data/series_demo",
                        n_frames=int(eff_n_frames),
                        height=128,
                        width=128,
                        vx=float(eff_vx),
                        vy=float(eff_vy),
                        asteroid_snr=float(eff_snr),
                        n_stars=int(eff_n_stars),
                        randomize=False,
                        seed=int(seed_val),
                        device=device,
                        pixel_scale_arcsec=float(selected_pixel_scale),
                        dt_hours=float(custom_dt_hours),
                        orbit_preset=str(selected_orbit_preset),
                    )
                    st.session_state["demo_gt"] = gt
                    st.session_state.pop("pipeline_result", None)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                st.success(
                    f"{gt['n_frames']}-Frame FITS-Serie in `data/series_demo` generiert "
                    f"(Modell={gt['orbit_preset']}, Skala={gt['pixel_scale_arcsec']:.2f}\"/px, "
                    f"vx={gt['vx']:.2f} px/h, vy={gt['vy']:.2f} px/h, "
                    f"SNR={gt['asteroid_snr']:.2f}, Start=({gt['x0']:.1f}, {gt['y0']:.1f}))."
                )
            except Exception as exc:
                st.error(f"Fehler beim Generieren der synthetischen Serie: {exc}")

    folder_path = Path(selected_folder)
    fits_files = sorted(
        [
            str(p)
            for p in folder_path.glob("*")
            if p.is_file() and p.suffix.lower() in (".fits", ".fit", ".fts")
        ]
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        vx_min = st.number_input("vx Suchbereich Min [px/h]", value=-2.0, step=0.5, help="Minimale horizontale Suchgeschwindigkeit vx in Pixeln pro Stunde entlang der X-Achse.")
        vx_max = st.number_input("vx Suchbereich Max [px/h]", value=2.0, step=0.5, help="Maximale horizontale Suchgeschwindigkeit vx in Pixeln pro Stunde entlang der X-Achse.")
    with c2:
        vy_min = st.number_input("vy Suchbereich Min [px/h]", value=-2.0, step=0.5, help="Minimale vertikale Suchgeschwindigkeit vy in Pixeln pro Stunde entlang der Y-Achse.")
        vy_max = st.number_input("vy Suchbereich Max [px/h]", value=2.0, step=0.5, help="Maximale vertikale Suchgeschwindigkeit vy in Pixeln pro Stunde entlang der Y-Achse.")
    with c3:
        grid_steps = st.slider("Grid-Schritte (grid_steps)", min_value=3, max_value=25, value=5, help="Anzahl der Gitterpunkte pro Geschwindigkeitsachse (erzeugt grid_steps \u00d7 grid_steps Test-Trajektorien auf der GPU).")
        cnn_threshold = st.slider(
            "RealBogusCNN Score-Schwelle (cnn_threshold)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Mindest-Wahrscheinlichkeit des 3-Kanal RealBogusCNN, ab der ein Kandidat als echtes astronomisches Objekt akzeptiert wird.",
        )

    detection_threshold = st.slider(
        "DAOStarFinder Detektions-Schwelle [ADU \u00fcber Hintergrund]",
        min_value=4.0,
        max_value=200.0,
        value=8.0,
        step=2.0,
        help="Absoluter Schwellenwert in ADU \u00fcber dem Hintergrund im gestackten Summenbild f\u00fcr die DAOStarFinder-Quellendetektion.",
    )

    vx_min = float(vx_min)
    vx_max = float(vx_max)
    vy_min = float(vy_min)
    vy_max = float(vy_max)
    grid_steps = int(grid_steps)
    cnn_threshold = float(cnn_threshold)
    detection_threshold = float(detection_threshold)

    run_clicked = st.button("\U0001f680 GPU-Grid-Search & Pipeline ausf\u00fchren", use_container_width=True)

    if run_clicked or st.session_state.get("pipeline_result"):
        if run_clicked:
            if len(fits_files) == 0:
                st.warning(
                    f"Keine FITS-Dateien in `{selected_folder}` gefunden. Bitte zuerst oben im Expander auf '\U0001f680 Serie generieren' klicken."
                )
                return

            if vx_min > vx_max or vy_min > vy_max:
                st.error("Ung\u00fcltiger Suchbereich: Min-Wert darf nicht gr\u00f6\u00dfer als Max-Wert sein.")
                return

            try:
                with st.spinner(
                    f"F\u00fchre GPU Velocity Grid Search ({grid_steps}\u00d7{grid_steps} Trajektorien) auf {len(fits_files)} FITS-Frames aus..."
                ):
                    with torch.no_grad():
                        mjd_days_list = []
                        raw_frames_np = []
                        cleaned_frames_np = []
                        ref_wcs = None
                        for idx, fpath in enumerate(fits_files):
                            img_t, wcs_obj, hdr = load_fits_image(fpath, device=device, nan_fill=None)
                            if ref_wcs is None and wcs_obj is not None and wcs_obj.is_celestial:
                                ref_wcs = wcs_obj
                            raw_frames_np.append(img_t.detach().cpu().numpy())
                            if hdr is not None and "MJD-OBS" in hdr:
                                mjd_days_list.append(float(hdr["MJD-OBS"]))
                            else:
                                mjd_days_list.append(60000.0 + float(idx) / 24.0)

                            bkg_t = estimate_background(img_t)
                            masked_t = mask_stationary_stars(
                                img_t - bkg_t, threshold=150.0, fwhm=3.0
                            )
                            cleaned_frames_np.append(masked_t.detach().cpu().numpy())

                        times_hours = torch.tensor(
                            [(m - mjd_days_list[0]) * 24.0 for m in mjd_days_list],
                            dtype=torch.float32,
                            device=device,
                        )

                        cnn_model = RealBogusCNN().to(device)
                        cnn_model.eval()

                        catalog = run_pipeline(
                            file_paths=fits_files,
                            times_hours=times_hours,
                            vx_range=(float(vx_min), float(vx_max)),
                            vy_range=(float(vy_min), float(vy_max)),
                            steps=int(grid_steps),
                            detection_threshold=float(detection_threshold),
                            cnn_model=cnn_model,
                            device=device,
                            star_mask_threshold=150.0,
                            cnn_threshold=float(cnn_threshold),
                        )

                        frame0_np = raw_frames_np[0]
                        frames_stack_t = torch.stack(
                            [torch.from_numpy(f) for f in cleaned_frames_np], dim=0
                        ).to(device)

                        if len(catalog) > 0:
                            catalog.sort("peak", reverse=True)
                            best_vx = float(catalog[0]["vx"])
                            best_vy = float(catalog[0]["vy"])
                            target_xy = (
                                float(catalog[0]["x_centroid"]),
                                float(catalog[0]["y_centroid"]),
                            )
                        else:
                            best_vx, best_vy = 0.0, 0.0
                            target_xy = None

                        shifted_stack_t = kinematic_shift_and_stack(
                            frames_stack_t,
                            times_hours=times_hours,
                            vx=best_vx,
                            vy=best_vy,
                        )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                st.session_state["pipeline_result"] = {
                    "catalog": catalog,
                    "best_vx": best_vx,
                    "best_vy": best_vy,
                    "target_xy": target_xy,
                    "raw_frames_np": raw_frames_np,
                    "shifted_stack_t": shifted_stack_t,
                    "times_hours": times_hours,
                    "ref_wcs": ref_wcs,
                    "mjd_days_list": mjd_days_list,
                }
                st.success("Pipeline erfolgreich ausgef\u00fchrt.")
            except Exception as exc:
                st.error(f"Fehler beim Ausf\u00fchren der Pipeline: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return

        res = st.session_state["pipeline_result"]
        catalog = res["catalog"]
        best_vx = res["best_vx"]
        best_vy = res["best_vy"]
        target_xy = res["target_xy"]
        raw_frames_np = res["raw_frames_np"]
        shifted_stack_t = res["shifted_stack_t"]
        times_hours = res["times_hours"]
        ref_wcs = res["ref_wcs"]
        mjd_days_list = res["mjd_days_list"]

        # Display results
        if len(catalog) > 0:
            st.markdown("#### \U0001f4cb Erkannte Kandidaten")
            st.dataframe(catalog.to_pandas(), use_container_width=True, height=280)

            best_candidate = catalog[0]
            st.markdown(
                f"### \U0001f30c Bester Kandidat: vx={best_vx:.2f} px/h, vy={best_vy:.2f} px/h"
            )

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                fig_raw = _plot_comparison_frame(
                    raw_frames_np[0],
                    "Rohbild",
                    target_xy=target_xy,
                    scale_pct=st.session_state.get("plot_scale_pct", 85.0),
                )
                st.pyplot(fig_raw, use_container_width=True)
                plt.close(fig_raw)
            with p_col2:
                fig_shifted = _plot_comparison_frame(
                    shifted_stack_t.detach().cpu().numpy(),
                    "Gestacktes Bild mit Kinematischer Verschiebung",
                    target_xy=target_xy,
                    scale_pct=st.session_state.get("plot_scale_pct", 85.0),
                )
                st.pyplot(fig_shifted, use_container_width=True)
                plt.close(fig_shifted)

            if len(raw_frames_np) > 0 and target_xy is not None:
                st.markdown("#### \U0001f4f8 Stamps")
                frame_stamps_list = [
                    extract_stamps(
                        torch.from_numpy(frame), [target_xy], stamp_size=32
                    )[0, 0].numpy()
                    for frame in raw_frames_np
                ]
                stack_stamp_np = (
                    extract_stamps(
                        shifted_stack_t.detach().cpu(), [target_xy], stamp_size=32
                    )[0, 0].numpy()
                )
                fig_stamps = _plot_stamp_strip(
                    frame_stamps_list,
                    stack_stamp_np,
                    times_hours_list=times_hours.detach().cpu().tolist(),
                    scale_pct=st.session_state.get("plot_scale_pct", 85.0),
                )
                st.pyplot(fig_stamps, use_container_width=True)
                plt.close(fig_stamps)
            st.markdown("#### \U0001fa90 Orbit Estimation & Stellarium Integration")
            orbit_wcs = ref_wcs if ref_wcs is not None else create_synthetic_wcs()
            times_list = times_hours.detach().cpu().tolist()
            base_mjd = mjd_days_list[0] if len(mjd_days_list) > 0 else 60000.0

            nom_epochs = compute_tracklet_endpoints(
                wcs=orbit_wcs,
                x0=target_xy[0],
                y0=target_xy[1],
                vx=best_vx,
                vy=best_vy,
                times_hours=times_list,
                base_mjd=base_mjd,
            )
            nom_orbit = estimate_kepler_orbit(nom_epochs)

            rng = np.random.default_rng(42)
            a_samples: list[float] = []
            e_samples: list[float] = []
            span_h = max(float(times_list[-1] - times_list[0]), 0.25) if len(times_list) > 1 else 1.0
            for _ in range(40):
                dx0, dy0 = rng.normal(0.0, 0.35, size=2)
                dvx, dvy = rng.normal(0.0, 0.25 / span_h, size=2)
                mc_epochs = compute_tracklet_endpoints(
                    wcs=orbit_wcs,
                    x0=target_xy[0] + float(dx0),
                    y0=target_xy[1] + float(dy0),
                    vx=best_vx + float(dvx),
                    vy=best_vy + float(dvy),
                    times_hours=times_list,
                    base_mjd=base_mjd,
                )
                mc_orb = estimate_kepler_orbit(mc_epochs)
                a_samples.append(float(mc_orb["a_au"]))
                e_samples.append(float(mc_orb["e"]))

            o_col1, o_col2, o_col3, o_col4 = st.columns(4)
            o_col1.metric("Semi-major Axis a", f"{nom_orbit['a_au']:.2f} AU")
            o_col2.metric("Eccentricity e", f"{nom_orbit['e']:.2f}")
            o_col3.metric("Inclination i", f"{nom_orbit['i_deg']:.1f}\u00b0")
            o_col4.metric("Orbit Class", str(nom_orbit["orbit_class"]))

            st.markdown("#### \U0001f31e Solar System Orbit & Monte-Carlo Uncertainty")
            fig_orbit = _plot_solar_system_orbit(
                a_nom=float(nom_orbit["a_au"]),
                e_nom=float(nom_orbit["e"]),
                a_samples=a_samples,
                e_samples=e_samples,
                scale_pct=st.session_state.get("plot_scale_pct", 85.0),
                show_all_planets=bool(st.session_state.get("show_all_planets", False)),
            )
            st.pyplot(fig_orbit, use_container_width=True)
            plt.close(fig_orbit)

            img_h, img_w = raw_frames_np[0].shape
            exit_info = predict_boundary_exit(
                wcs=orbit_wcs,
                x0=target_xy[0],
                y0=target_xy[1],
                vx=best_vx,
                vy=best_vy,
                height=int(img_h),
                width=int(img_w),
                base_mjd=base_mjd,
                lookahead_hours=1.0,
            )

            st.markdown("#### \U0001f9ed Nachbarfeld-Tracking & Stellarium RemoteControl")
            tr_col1, tr_col2, tr_col3 = st.columns(3)
            tr_col1.metric("Austritts-Kante", str(exit_info["exit_edge"]).upper())
            tr_col2.metric(
                "Zeit bis Feldaustritt",
                f"{exit_info['time_to_exit_hours']:.2f} h"
                if np.isfinite(exit_info["time_to_exit_hours"])
                else "\u221e (Station\u00e4r)",
            )
            tr_col3.metric(
                "Nachbarfeld (RA / Dec)",
                f"{exit_info['next_field_ra_deg']:.4f}\u00b0, {exit_info['next_field_dec_deg']:.4f}\u00b0",
            )

            cand_ra = float(nom_epochs[0]["ra_deg"])
            cand_dec = float(nom_epochs[0]["dec_deg"])
            cand_mjd = float(nom_epochs[0]["mjd"])

            st_col1, st_col2 = st.columns([1, 2])
            with st_col1:
                if st.button(
                    "\U0001f52d Stellarium Live-Slew auf Kandidat",
                    key="btn_stellarium_slew",
                    use_container_width=True,
                ):
                    slew_res = slew_stellarium_to_candidate(
                        ra_deg=cand_ra,
                        dec_deg=cand_dec,
                        mjd=cand_mjd,
                    )
                    if slew_res.get("ok"):
                        st.success(
                            f"Stellarium zentriert auf RA={cand_ra:.4f}\u00b0, Dec={cand_dec:.4f}\u00b0."
                        )
                    else:
                        st.warning(
                            f"Stellarium RemoteControl nicht erreichbar ({slew_res.get('error')})."
                        )
            with st_col2:
                mpc_report = export_tracklet_mpc80(
                    nom_epochs,
                    designation="K26A00A",
                    mag=20.0,
                    band="V",
                    obs_code="500",
                )
                with st.expander("\U0001f4c4 MPC 80-Column Astrometrie-Export", expanded=False):
                    st.code(mpc_report, language="text")
                    st.download_button(
                        "\U0001f4be MPC80-Report herunterladen (.txt)",
                        data=mpc_report,
                        file_name="mpc80_tracklet.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
        else:
            st.warning("Keine Kandidaten erkannt.")
