import sys
import tempfile
import warnings
from pathlib import Path

# Ensure `src/` is in the Python path if `streamlit run src/astro_project/gui/app.py` is directly started
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import streamlit as st
import torch
from astropy.utils.data import get_pkg_data_filename
from photutils.detection.daofinder import NoDetectionsWarning

from astro_project.core.fits_loader import load_fits_image
from astro_project.core.photometry import detect_sources, estimate_background
from astro_project.core.wcs_utils import sources_to_skycoords


DARK_MODE_CSS = """
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
    .status-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }
</style>
"""


def get_hardware_metrics() -> dict:
    """Reads live metrics for CPU, system RAM, and RTX 4070 / GPU VRAM via psutil and torch.cuda."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    vm = psutil.virtual_memory()
    ram_percent = vm.percent
    ram_used_gb = vm.used / (1024**3)
    ram_total_gb = vm.total / (1024**3)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
        used_bytes = total_bytes - free_bytes
        vram_used_gb = used_bytes / (1024**3)
        vram_total_gb = total_bytes / (1024**3)
        vram_percent = (used_bytes / total_bytes) * 100.0 if total_bytes > 0 else 0.0
    else:
        gpu_name = "RTX 4070 (CUDA inaktiv)"
        vram_used_gb = 0.0
        vram_total_gb = 0.0
        vram_percent = 0.0

    return {
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
        "vram_percent": vram_percent,
    }


def render_sidebar_statusbar() -> torch.device:
    """Renders the hardware statusbar (CPU, RAM, VRAM) and device control in the sidebar."""
    with st.sidebar:
        st.markdown("## ⚡ Hardware Statusbar")
        metrics = get_hardware_metrics()

        st.metric(
            label="CPU-Auslastung",
            value=f"{metrics['cpu_percent']:.1f} %",
        )
        st.progress(min(int(metrics["cpu_percent"]), 100))

        st.metric(
            label="RAM-Auslastung",
            value=f"{metrics['ram_percent']:.1f} %",
            delta=f"{metrics['ram_used_gb']:.1f} / {metrics['ram_total_gb']:.1f} GB",
            delta_color="off",
        )
        st.progress(min(int(metrics["ram_percent"]), 100))

        if metrics["cuda_available"]:
            st.metric(
                label=f"GPU VRAM ({metrics['gpu_name']})",
                value=f"{metrics['vram_used_gb']:.2f} / {metrics['vram_total_gb']:.2f} GB",
                delta=f"{metrics['vram_percent']:.1f} % belegt",
                delta_color="off",
            )
            st.progress(min(int(metrics["vram_percent"]), 100))
        else:
            st.metric(
                label="RTX 4070 VRAM",
                value="0.00 / 0.00 GB",
                delta="CUDA nicht aktiv",
                delta_color="off",
            )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Refresh", width="stretch"):
                st.rerun()
        with col_btn2:
            if st.button("🧹 VRAM Flush", width="stretch"):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                st.rerun()

        st.divider()
        st.markdown("### ⚙️ Compute Device")
        options = ["cuda", "cpu"] if metrics["cuda_available"] else ["cpu"]
        selected_dev = st.selectbox(
            "Ziel-Device",
            options=options,
            index=0,
            help="Wählt das PyTorch-Device für [Y, X] Tensor-Operationen.",
        )
        return torch.device(
            selected_dev if selected_dev == "cuda" and metrics["cuda_available"] else "cpu"
        )


def _plot_fits_frame(
    image_np: np.ndarray,
    title: str,
    sources=None,
) -> plt.Figure:
    """Creates a contrast-enhanced dark-mode plot of a [Y, X] image with optional detections."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0b0f19")
    ax.set_facecolor("#0b0f19")

    vmin, vmax = np.percentile(image_np, [5.0, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1.0

    im = ax.imshow(image_np, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    if sources is not None and len(sources) > 0:
        ax.scatter(
            sources["x_centroid"],
            sources["y_centroid"],
            s=60,
            facecolors="none",
            edgecolors="#38bdf8",
            linewidths=1.2,
            label=f"Detektionen ({len(sources)})",
        )
        ax.legend(
            loc="upper right",
            facecolor="#1e293b",
            edgecolor="#334155",
            labelcolor="#f8fafc",
        )

    ax.set_title(title, color="#f8fafc", fontsize=11)
    ax.set_xlabel("X [Pixel]", color="#94a3b8")
    ax.set_ylabel("Y [Pixel]", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")
    fig.tight_layout()
    return fig


def render_tab_fits_inspector(device: torch.device):
    """Tab 1: FITS Inspector, 2D-Background modeling, DAOStarFinder & WCS-Astrometry."""
    st.subheader("🔭 FITS Inspector & Photometrie ([Y, X] Tensor-Pipeline)")

    col_src, col_params = st.columns([2, 2])
    with col_src:
        data_source = st.radio(
            "Datenquelle wählen",
            options=[
                "Lokales `data/` Verzeichnis / Astropy Sample FITS",
                "Eigene FITS-Datei hochladen",
            ],
            horizontal=True,
        )

        fits_path: str | None = None
        temp_file_path: Path | None = None

        if data_source == "Eigene FITS-Datei hochladen":
            uploaded = st.file_uploader(
                "FITS-Bild auswählen (.fits, .fit, .fts)",
                type=["fits", "fit", "fts"],
            )
            if uploaded is not None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".fits")
                tmp.write(uploaded.read())
                tmp.close()
                temp_file_path = Path(tmp.name)
                fits_path = str(temp_file_path)
        else:
            data_dir = Path("data")
            local_fits = sorted(data_dir.glob("*.fit*")) if data_dir.exists() else []
            options = ["[Astropy Standard] HorseHead.fits"] + [str(p) for p in local_fits]
            choice = st.selectbox("Verfügbare FITS-Dateien", options=options)
            if choice == "[Astropy Standard] HorseHead.fits":
                fits_path = get_pkg_data_filename("tutorials/FITS-images/HorseHead.fits")
            else:
                fits_path = choice

    with col_params:
        c1, c2 = st.columns(2)
        with c1:
            sigma_factor = st.slider(
                "Detektions-Schwelle (σ über Hintergrund)", 2.0, 20.0, 5.0, 0.5
            )
        with c2:
            fwhm = st.slider("DAOStarFinder FWHM [Pixel]", 1.5, 10.0, 3.0, 0.5)

    if not fits_path:
        st.info("Bitte eine FITS-Datei auswählen oder hochladen, um die Analyse zu starten.")
        return

    try:
        with torch.no_grad():
            image_tensor, wcs, header = load_fits_image(fits_path, device=device)
            bkg_tensor = estimate_background(image_tensor)
            residual_tensor = image_tensor - bkg_tensor

            noise_std = float(torch.std(residual_tensor).item())
            abs_threshold = max(sigma_factor * noise_std, 1e-3)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NoDetectionsWarning)
                sources = detect_sources(
                    residual_tensor, threshold=abs_threshold, fwhm=fwhm
                )

            img_np = image_tensor.detach().cpu().numpy()
            res_np = residual_tensor.detach().cpu().numpy()

        # Metric overview
        m1, m2, m3, m4 = st.columns(4)
        height, width = image_tensor.shape
        m1.metric("Tensor Shape [Y, X]", f"[{height}, {width}]")
        m2.metric("Hintergrund-Median", f"{float(torch.median(bkg_tensor).item()):.2f} ADU")
        m3.metric("Rausch-Level (1σ)", f"{noise_std:.2f} ADU")
        num_det = len(sources) if sources is not None else 0
        m4.metric("Gefundene Quellen", f"{num_det}")

        # Visualization: Raw image vs. Background-subtracted image + detections
        p1, p2 = st.columns(2)
        with p1:
            fig_raw = _plot_fits_frame(img_np, "Rohbild [Y, X]")
            st.pyplot(fig_raw, width="stretch")
            plt.close(fig_raw)
        with p2:
            fig_det = _plot_fits_frame(
                res_np, "Hintergrundbereinigt + DAOStarFinder", sources=sources
            )
            st.pyplot(fig_det, width="stretch")
            plt.close(fig_det)

        # WCS-Astrometry & Source table
        st.markdown("#### 🌐 Astrometrische Quellentabelle (WCS Pixel-to-Sky `origin=0`)")
        if sources is not None and len(sources) > 0:
            df = sources.to_pandas()
            if wcs is not None and wcs.is_celestial:
                sky_coords = sources_to_skycoords(sources, wcs)
                df["ra_deg"] = np.asarray(sky_coords.ra.deg, dtype=np.float64)
                df["dec_deg"] = np.asarray(sky_coords.dec.deg, dtype=np.float64)

            preferred_cols = [
                c
                for c in (
                    "id",
                    "x_centroid",
                    "y_centroid",
                    "ra_deg",
                    "dec_deg",
                    "flux",
                    "peak",
                    "sharpness",
                )
                if c in df.columns
            ]
            st.dataframe(df[preferred_cols], width="stretch", height=300)
        else:
            st.warning("Keine Punktquellen oberhalb des gewählten Schwellenwerts detektiert.")

        with st.expander("📄 FITS-Header & WCS-Metadaten anzeigen"):
            header_items = [
                {
                    "Key": str(k),
                    "Value": str(header[k]),
                    "Comment": str(header.comments[k]),
                }
                for k in header.keys()
                if str(k).strip()
            ]
            st.dataframe(pd.DataFrame(header_items), width="stretch", height=260)

    finally:
        if temp_file_path is not None and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def render_tab_pipeline_grid_search(device: torch.device):
    """Tab 2: Placeholder and configuration tab for the GPU Velocity Grid Search & Pipeline."""
    st.subheader("🚀 GPU Shift-and-Stack Pipeline & Velocity Grid Search")
    st.markdown(
        """
        <div class="status-card">
            <h4>Architektur-Bereitschaft: End-to-End Asteroid Discovery Pipeline</h4>
            <p>This tab is prepared for future live executions of the GPU search over FITS time series:</p>
            <ul>
                <li><b>Star Masking:</b> <code>mask_stationary_stars</code> removes static field stars.</li>
                <li><b>Kinematic Stacker:</b> Flux-conserving subpixel shift via <code>F.affine_grid</code> & <code>F.grid_sample(align_corners=False)</code>.</li>
                <li><b>GPU Velocity Search:</b> Parallel search over <code>(vx, vy)</code> under strict <code>torch.no_grad()</code> VRAM protection.</li>
                <li><b>RealBogusCNN:</b> 32×32 stamp classification and export as Astropy catalog with ICRS <code>(RA, Dec)</code> coordinates.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("vx Search Range Min [px/h]", value=-3.0, step=0.5, disabled=True)
        st.number_input("vx Search Range Max [px/h]", value=3.0, step=0.5, disabled=True)
    with c2:
        st.number_input("vy Search Range Min [px/h]", value=-3.0, step=0.5, disabled=True)
        st.number_input("vy Search Range Max [px/h]", value=3.0, step=0.5, disabled=True)
    with c3:
        st.slider("Grid Steps (steps)", min_value=3, max_value=25, value=9, disabled=True)
        st.slider("RealBogusCNN Score Threshold", min_value=0.0, max_value=1.0, value=0.5, disabled=True)

    st.info(
        f"Aktives Compute-Device: **{device}**. Bereit für die Anbindung an `run_pipeline` und `data_fetcher.py`."
    )


def main():
    st.set_page_config(
        page_title="AstroProject | GPU Asteroid Pipeline",
        page_icon="☄️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)

    device = render_sidebar_statusbar()

    st.title("☄️ AstroProject — GPU Shift-and-Stack & Photometry Control Center")
    tab_inspector, tab_pipeline = st.tabs(
        ["FITS Inspector & Photometrie", "Pipeline & Grid Search"]
    )

    with tab_inspector:
        render_tab_fits_inspector(device=device)

    with tab_pipeline:
        render_tab_pipeline_grid_search(device=device)


if __name__ == "__main__":
    main()
