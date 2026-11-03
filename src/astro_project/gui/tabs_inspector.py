import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from astropy.utils.data import get_pkg_data_filename
from photutils.detection.daofinder import NoDetectionsWarning

from astro_project.core.fits_loader import load_fits_image
from astro_project.core.photometry import detect_sources, estimate_background
from astro_project.core.wcs_utils import sources_to_skycoords
from astro_project.gui.components import _plot_inspector_frame, show_parameter_help_dialog


def render_tab_fits_inspector(device: torch.device):
    """FITS-Inspector & Photometry"""
    hdr_col1, hdr_col2 = st.columns([3, 1])
    with hdr_col1:
        st.subheader("\U0001f52d FITS-Inspector & Photometry")
    with hdr_col2:
        if st.button("\u2753 Parameter-Hilfe \u00f6ffnen", key="help_btn_tab1", use_container_width=True):
            show_parameter_help_dialog()

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    local_fits = sorted(
        [
            p
            for p in data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in (".fits", ".fit", ".fts")
        ]
    )

    col_src, col_ctrl1, col_ctrl2 = st.columns([2, 1, 1])
    with col_src:
        source_mode = st.radio(
            "FITS Data Source",
            options=[
                "Local data/ directory",
                "Astropy Sample (HorseHead.fits)",
                "Upload FITS file",
            ],
            horizontal=True,
            help="W\u00e4hlt die Quelle der FITS-Bilddaten: Lokales Verzeichnis data/, Astropy-Beispielbild oder eigener Datei-Upload.",
        )
    with col_ctrl1:
        sigma_thresh = st.slider(
            "Detection Threshold (sigma above background)",
            min_value=1.5,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Detektionsschwelle als Vielfaches des Hintergrundrauschens (1\u03c3). Niedrigere Werte finden lichtschw\u00e4chere Quellen.",
        )
    with col_ctrl2:
        fwhm_val = st.slider(
            "DAOStarFinder FWHM [px]",
            min_value=1.5,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Erwartete Halbwertsbreite (FWHM in Pixeln) der Stern-PSF f\u00fcr den DAOStarFinder Gau\u00df-Filter.",
        )

    target_fits_path: str | None = None
    temp_upload_path: Path | None = None

    if source_mode == "Local data/ directory":
        if len(local_fits) == 0:
            st.info(
                "No FITS files found in `data/`. Switch to 'Astropy Sample (HorseHead.fits)', upload a FITS file, or generate the synthetic series in Tab 2."
            )
            return
        selected_rel = st.selectbox(
            "Select FITS file from data/",
            options=[str(p.relative_to(data_dir)) for p in local_fits],
            help="W\u00e4hlt eine lokale FITS-Datei aus dem Verzeichnis data/ zur photometrischen Analyse aus.",
        )
        target_fits_path = str(data_dir / selected_rel)
    elif source_mode == "Astropy Sample (HorseHead.fits)":
        try:
            target_fits_path = get_pkg_data_filename(
                "tutorials/FITS-images/HorseHead.fits"
            )
        except Exception as exc:
            st.error(f"Could not load Astropy sample FITS file: {exc}")
            return
    else:
        uploaded = st.file_uploader(
            "Upload FITS Image (.fits, .fit, .fts)",
            type=["fits", "fit", "fts"],
            help="L\u00e4dt eine eigene FITS-Datei zum Inspizieren hoch.",
        )
        if uploaded is None:
            st.info("Please upload a FITS file to inspect.")
            return
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".fits")
        tmp.write(uploaded.read())
        tmp.flush()
        tmp.close()
        temp_upload_path = Path(tmp.name)
        target_fits_path = str(temp_upload_path)

    try:
        with torch.no_grad():
            image_t, wcs, header = load_fits_image(target_fits_path, device=device, nan_fill=None)
            bkg_t = estimate_background(image_t)
            residual_t = image_t - bkg_t

            height, width = image_t.shape
            bkg_median = float(torch.median(bkg_t).item())
            noise_std = float(torch.std(residual_t).item())
            abs_threshold = max(float(sigma_thresh) * noise_std, 1e-6)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NoDetectionsWarning)
                sources = detect_sources(
                    residual_t, threshold=abs_threshold, fwhm=float(fwhm_val)
                )

            sources_df: pd.DataFrame | None = None
            n_sources = 0
            if sources is not None and len(sources) > 0:
                n_sources = len(sources)
                if wcs is not None and wcs.is_celestial:
                    try:
                        sky_coords = sources_to_skycoords(sources, wcs)
                        sources["ra"] = sky_coords.ra.deg
                        sources["dec"] = sky_coords.dec.deg
                    except Exception:
                        pass
                sources_df = sources.to_pandas()

            raw_np = image_t.detach().cpu().numpy()
            res_np = residual_t.detach().cpu().numpy()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tensor Shape [Y, X]", f"[{height}, {width}]")
        m2.metric("Background Median", f"{bkg_median:.2f} ADU")
        m3.metric("Noise Level (1\u03c3)", f"{noise_std:.2f} ADU")
        m4.metric("Detected Sources", f"{n_sources}")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            fig_raw = _plot_inspector_frame(
                raw_np,
                f"Raw Image [{height}\u00d7{width}]",
                scale_pct=st.session_state.get("plot_scale_pct", 85.0),
            )
            st.pyplot(fig_raw, use_container_width=True)
            plt.close(fig_raw)
        with p_col2:
            fig_res = _plot_inspector_frame(
                res_np,
                f"Background Subtracted + DAOStarFinder ({sigma_thresh:.1f}\u03c3, FWHM={fwhm_val:.1f})",
                sources_df=sources_df,
                scale_pct=st.session_state.get("plot_scale_pct", 85.0),
            )
            st.pyplot(fig_res, use_container_width=True)
            plt.close(fig_res)

        st.markdown("#### \U0001f4cb Detected Sources Catalog")
        if sources_df is not None and len(sources_df) > 0:
            st.dataframe(sources_df, use_container_width=True, height=280)
        else:
            st.info(
                "No sources detected above the selected threshold. Try lowering the Detection Threshold slider."
            )

        with st.expander("\U0001f4c4 FITS Header Metadata"):
            if header is not None:
                hdr_rows = [
                    {
                        "Keyword": str(k),
                        "Value": str(header[k]),
                        "Comment": str(header.comments[k]),
                    }
                    for k in header.keys()
                    if str(k).strip() != ""
                ]
                st.dataframe(pd.DataFrame(hdr_rows), use_container_width=True, height=260)
            else:
                st.write("No FITS header available.")
    except Exception as exc:
        st.error(f"Error inspecting FITS file: {exc}")
    finally:
        if temp_upload_path is not None and temp_upload_path.exists():
            try:
                temp_upload_path.unlink()
            except OSError:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
