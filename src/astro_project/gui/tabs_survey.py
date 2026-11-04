import pandas as pd
import streamlit as st
import torch

from astro_project.pipeline.data_fetcher import (
    SURVEY_TELESCOPE_PRESETS,
    generate_or_fetch_survey_fields,
)
from astro_project.pipeline.survey_engine import run_automated_survey_batch
from astro_project.pipeline.ztf_fetcher import query_ztf_archive, download_ztf_batch


def render_survey_tab(device: str | torch.device):
    """Renders the Automated Sky Survey tab with telescope presets, ledger deduplication, and 3-tier storage."""
    st.markdown("## \U0001f6f0\ufe0f Automated Sky Survey & Tiered Discovery Engine")
    st.caption(
        "Scans multi-field telescope archives on the GPU, filters cosmic rays via temporal coherence gating, "
        "deduplicates via 64-bit Blake2b SQLite ledger, and auto-purges empty staging fields."
    )

    data_source = st.radio(
        "Data Source",
        options=["Synthetic Benchmark (Mock Data)", "ZTF Public Archive (Live)"],
        horizontal=True,
        key="survey_data_source"
    )

    preset_keys = list(SURVEY_TELESCOPE_PRESETS.keys())

    if data_source == "ZTF Public Archive (Live)":
        st.markdown("### ZTF Archive Query Parameters \u2604\ufe0f")
        col_ztf1, col_ztf2, col_ztf3, col_ztf4 = st.columns(4)
        with col_ztf1:
            ztf_ra = st.number_input("RA (deg)", value=180.0, key="ztf_ra")
        with col_ztf2:
            ztf_dec = st.number_input("Dec (deg)", value=25.0, key="ztf_dec")
        with col_ztf3:
            ztf_mjd_start = st.number_input("MJD Start", value=58200.0, key="ztf_mjd_start")
        with col_ztf4:
            ztf_mjd_end = st.number_input("MJD End", value=58200.5, key="ztf_mjd_end")
        n_fields = 1
    else:
        ztf_ra = ztf_dec = ztf_mjd_start = ztf_mjd_end = None

    with st.expander("Teleskop-Leitfaden & Empfehlungen \U0001f52d"):
        st.markdown(
            "- **ZTF Palomar 48-inch**: Weitfeld-Survey, hohe Kadenz (High-Cadence). Ideal fuer die Entdeckung von Near-Earth Asteroids (NEAs) und schnellen Transienten.\n"
            "- **Pan-STARRS PS1**: Hohe Aufloesung (High-Resolution), tiefe Aufnahmen. Perfekt fuer praezise Astrometrie und schwache Objekte im aeusseren Sonnensystem.\n"
            "- **Catalina CSS Schmidt**: Klassisches Weitfeld mit groeberer Pixelskala. Bewaehrt fuer grossflaechige Himmelsdurchmusterungen."
        )

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        telescope = st.selectbox(
            "Survey Telescope Preset",
            options=preset_keys,
            format_func=lambda k: f"{k} ({SURVEY_TELESCOPE_PRESETS[k]['label']})",
            key="survey_telescope_select",
        )
        if data_source == "Synthetic Benchmark (Mock Data)":
            n_fields = st.number_input(
                "Number of Fields (n_fields)",
                min_value=1,
                max_value=50,
                value=5,
                step=1,
                key="survey_n_fields",
            )

    with col_cfg2:
        min_keep_score = st.slider(
            "Min Keep Score (Silver Tier Threshold)",
            min_value=0.0,
            max_value=100.0,
            value=45.0,
            step=1.0,
            key="survey_min_keep_score",
        )
        stack_detection_sigma = st.slider(
            "Stack Detection Threshold (\u03c3)",
            min_value=2.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            key="survey_stack_sigma",
        )

    with col_cfg3:
        preset_info = SURVEY_TELESCOPE_PRESETS[telescope]
        st.markdown(
            f"**Pixel Scale:** `{preset_info['pixel_scale_arcsec']}` arcsec/px  \n"
            f"**Cadence (\u0394t):** `{preset_info['default_dt_hours']}` h  \n"
            f"**MPC Code:** `{preset_info['obs_code']}`"
        )
        auto_purge = st.checkbox(
            "Auto-Purge Empty & Raw Staging FITS",
            value=True,
            key="survey_auto_purge",
        )

    if st.button(
        "\U0001f680 Start Automated Archive Survey",
        type="primary",
        use_container_width=True,
        key="start_automated_survey_btn",
    ):
        with st.status("Staging survey fields & running GPU velocity search...") as status_container:
            if data_source == "ZTF Public Archive (Live)":
                st.info("Querying IRSA database for ZTF epochs...")
                try:
                    epochs = query_ztf_archive(ztf_ra, ztf_dec, ztf_mjd_start, ztf_mjd_end)
                    
                    if not epochs:
                        status_container.update(label="Abgebrochen: Keine Frames gefunden", state="error")
                        st.warning("No ZTF frames found for the given coordinates and time range.")
                        st.stop()
                        
                    st.info(f"Found {len(epochs)} frames. Downloading to staging...")
                    downloaded_files = download_ztf_batch(epochs, dest_dir="data/staging/ztf_live")
                except Exception as e:
                    status_container.update(label="Abgebrochen: Archiv-Fehler", state="error")
                    st.error(f"Archiv-Fehler: {e}")
                    st.info("Versuche es spaeter erneut oder nutze voruebergehend den Synthetic Benchmark Modus.")
                    st.stop()
            else:
                active_indices = [i for i in (1, 3) if i < int(n_fields)]
                if not active_indices and int(n_fields) > 0:
                    active_indices = [0]

                field_batches = generate_or_fetch_survey_fields(
                    base_data_dir="data",
                    telescope=telescope,
                    n_fields=int(n_fields),
                    active_field_indices=active_indices,
                    device=device,
                )

            results = run_automated_survey_batch(
                field_batches=field_batches,
                base_data_dir="data",
                device=device,
                stack_detection_sigma=float(stack_detection_sigma),
                min_keep_score=float(min_keep_score),
                auto_purge=bool(auto_purge),
            )
            status_container.update(label="Survey Batch Complete", state="complete")
            st.session_state["survey_last_results"] = results

    results = st.session_state.get("survey_last_results")
    if results is not None:
        st.markdown("---")
        st.markdown("### \U0001f4ca Survey Batch Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scanned Fields", int(results.get("scanned", 0)))
        m2.metric("Empty/Purged", int(results.get("empty_purged", 0)))
        m3.metric("\u2b50 Gold Candidates", int(results.get("gold_saved", 0)))
        m4.metric("\U0001f948 Silver Candidates", int(results.get("silver_saved", 0)))

        candidates_list = results.get("candidates_list", [])
        if candidates_list:
            st.markdown("### \U0001f3af Discovered Candidates")
            display_cols = ["field_id", "tier", "score", "coherence", "peak"]
            df_cands = pd.DataFrame(candidates_list)
            available_cols = [c for c in display_cols if c in df_cands.columns]
            st.dataframe(df_cands[available_cols], use_container_width=True)
        else:
            st.info("No candidates above the keep score threshold in this batch.")
