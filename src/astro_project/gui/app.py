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
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from photutils.detection.daofinder import NoDetectionsWarning

from astro_project.core.catalog_match import cross_match_known_objects
from astro_project.core.classifier import RealBogusCNN, extract_stamps
from astro_project.core.fits_loader import load_fits_image
from astro_project.core.hardware import resolve_device
from astro_project.core.orbit import (
    compute_tracklet_endpoints,
    estimate_kepler_orbit,
    export_tracklet_mpc80,
)
from astro_project.core.photometry import detect_sources, estimate_background
from astro_project.core.stellarium import slew_stellarium_to_candidate
from astro_project.core.synthetic import (
    generate_demo_asteroid_series,
    generate_synthetic_sky,
)
from astro_project.core.wcs_utils import create_synthetic_wcs, sources_to_skycoords
from astro_project.pipeline.kinematics import (
    kinematic_shift_and_stack,
    mask_stationary_stars,
)
from astro_project.pipeline.orchestration import run_pipeline
from astro_project.pipeline.tracker import predict_boundary_exit


DARK_MODE_CSS = """
<style>
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.5rem !important;
    }
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 2rem !important;
    }
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
        padding: 10px 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
    .top-statusbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: 34px;
        padding: 0 14px;
        background: linear-gradient(90deg, #111827 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 6px;
        font-size: 0.84rem;
        font-family: 'Segoe UI', Roboto, monospace;
        color: #f8fafc;
        margin-bottom: 0.6rem;
        white-space: nowrap;
        overflow: hidden;
    }
    .top-statusbar span {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .status-badge {
        background-color: #0f172a;
        border: 1px solid #38bdf8;
        color: #38bdf8;
        padding: 1px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    div[data-testid="stImage"] img {
        max-height: 48vh !important;
        width: auto !important;
        max-width: 100% !important;
        margin: 0 auto !important;
        display: block !important;
        object-fit: contain !important;
    }
    div[data-testid="stModal"] {
        background-color: rgba(5, 8, 15, 0.65) !important;
        backdrop-filter: blur(4px) !important;
    }
    div[role="dialog"] {
        background-color: rgba(17, 24, 39, 0.94) !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.7) !important;
    }
</style>
"""

@st.dialog("❓ AstroProject — Parameter- & Physik-Handbuch", width="large")
def show_parameter_help_dialog():
    """Schließbares Dark-Mode-Modal-Fenster mit Erklärungen zu allen Parametern aus Tab 1 und Tab 2."""
    st.markdown(
        """
        ### 🔭 Tab 1: FITS-Inspector & Photometrie
        * **FITS Data Source:** Wählt die Bildquelle aus dem lokalen Ordner `data/`, dem integrierten Astropy-Beispielbild (`HorseHead.fits`) oder einem eigenen Datei-Upload.
        * **Detection Threshold (σ über Hintergrund):** Bestimmt, ab welchem Vielfachen des Hintergrundrauschens ($1\\sigma$) eine Punktquelle durch `DAOStarFinder` als Stern/Objekt akzeptiert wird. Niedrigere Werte finden schwächere Quellen, erhöhen aber Falschdetektionen.
        * **DAOStarFinder FWHM [px]:** Erwartete Halbwertsbreite (Full Width at Half Maximum) der teleskopischen Punktspreizfunktion (PSF) in Pixeln für den Gauß-Matched-Filter.

        ---
        ### 🚀 Tab 2: GPU Shift-and-Stack Pipeline & Velocity Grid Search
        * **Pipeline-Datenquelle:** Wählt zwischen der synthetischen 5-Frame-Demo (`data/series_demo/`) und realen FITS-Sequenzen in `data/`.
        * **vx / vy Suchbereich Min & Max [px/h]:** Definiert den Geschwindigkeitsraum in Pixeln pro Stunde entlang der $[Y, X]$-Bildachsen (`vx` in X-Richtung, `vy` in Y-Richtung), der auf der GPU abgetastet wird.
        * **Grid-Schritte (`grid_steps`):** Anzahl der Stützstellen pro Geschwindigkeitsachse. Ein Wert von $N$ erzeugt ein $N \\times N$ Trajektorien-Gitter.
        * **RealBogusCNN Score-Schwelle (`cnn_threshold`):** Mindest-Klassifikationswahrscheinlichkeit $[0.0, 1.0]$ des 3-Kanal-CNNs (Mean, Median, Variance), ab der ein gestackter Kandidat als echtes bewegtes Objekt (`Real`) eingestuft wird.
        * **DAOStarFinder Detektions-Schwelle [ADU über Hintergrund]:** Absoluter Schwellenwert im Shift-and-Stack-Summenbild. Da das Signal-zu-Rausch-Verhältnis beim Stacking mit $\\sqrt{N_{\\text{Frames}}}$ wächst, treten hier zuvor unsichtbare Sub-Sigma-Asteroiden deutlich hervor.
        * **🎞️ Frame-Navigator & Spur-Overlay:** Erlaubt das Durchblättern der Einzelbilder $0 \\dots N-1$ (Rohbild vs. sternmaskiertes Bild) im direkten Vergleich zum Shift-and-Stack-Summenbild.

        ---
        ### 🪐 Meilenstein 12 & 13: Orbit, MPC-80 & Stellarium
        * **Väisälä-Kepler-Orbit:** Schätzt aus der kurzen Winkelgeschwindigkeit am Himmel die große Halbachse $a\\,[\\text{AU}]$, Exzentrizität $e$, Inklination $i\\,[^\\circ]$ und die Orbit-Klasse (z.B. Main-Belt, NEA).
        * **Nachbarfeld-Tracking:** Berechnet Zeitpunkt und Himmelskoordinate, an denen das Objekt den aktuellen Sensorbereich $[0, W] \\times [0, H]$ verlässt.
        * **Stellarium Live-Slew:** Zentriert die virtuelle Kamera in Stellarium über das RemoteControl-Plugin (`http://localhost:8090/api`) direkt auf die Kandidaten-Koordinate.
        """
    )


def get_hardware_metrics() -> dict:
    """Reads live metrics for CPU, system RAM, and RTX 4070 VRAM fault-tolerantly."""
    try:
        cpu_percent = float(psutil.cpu_percent(interval=0.05))
        vm = psutil.virtual_memory()
        ram_percent = float(vm.percent)
        ram_used_gb = vm.used / (1024**3)
        ram_total_gb = vm.total / (1024**3)
    except Exception:
        cpu_percent = 0.0
        ram_percent = 0.0
        ram_used_gb = 0.0
        ram_total_gb = 0.0

    cuda_available = False
    gpu_name = "RTX 4070 (CUDA inactive)"
    vram_used_gb = 0.0
    vram_reserved_gb = 0.0
    vram_total_gb = 12.0
    vram_percent = 0.0

    try:
        if torch.cuda.is_available():
            torch.cuda.init()
            cuda_available = True
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            total_bytes = props.total_memory
            allocated_bytes = torch.cuda.memory_allocated(0)
            reserved_bytes = torch.cuda.memory_reserved(0)

            free_bytes, _ = torch.cuda.mem_get_info(0)
            driver_used_bytes = max(total_bytes - free_bytes, allocated_bytes)

            vram_used_gb = max(allocated_bytes, driver_used_bytes) / (1024**3)
            vram_reserved_gb = reserved_bytes / (1024**3)
            vram_total_gb = total_bytes / (1024**3)
            vram_percent = (vram_used_gb / vram_total_gb) * 100.0 if vram_total_gb > 0 else 0.0
    except Exception as exc:
        gpu_name = f"CUDA Error ({exc})"

    return {
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_used_gb": vram_used_gb,
        "vram_reserved_gb": vram_reserved_gb,
        "vram_total_gb": vram_total_gb,
        "vram_percent": vram_percent,
    }


def render_sidebar_controls() -> torch.device:
    """Rendert die Device-Auswahl (Standard: CUDA falls verfügbar) und Cache-Steuerung in der Sidebar."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.init()
    with st.sidebar:
        st.markdown("### ⚙️ Compute & VRAM Control")
        options = ["cuda", "cpu"] if cuda_available else ["cpu"]
        selected_dev = st.selectbox(
            "Ziel-Device",
            options=options,
            index=0,
            help="Wählt das PyTorch-Device für [Y, X] Tensor-Operationen (CUDA Standard auf RTX 4070).",
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

        plot_scale_pct = st.slider(
            "🔍 Plot-Größe [%]",
            min_value=50,
            max_value=100,
            value=85,
            step=5,
            help="Skaliert die Darstellungsgröße (figsize) aller Matplotlib-Plots dynamisch für optimales Rein-/Rauszoomen.",
        )
        st.session_state["plot_scale_pct"] = float(plot_scale_pct)

        return resolve_device(
            selected_dev if selected_dev == "cuda" and cuda_available else "cpu"
        )


def render_top_statusbar(device: torch.device):
    """Rendert die ultra-kompakte ~1cm (34px) horizontale Statusbar via st.columns ganz oben im Hauptbereich."""
    metrics = get_hardware_metrics()
    c_cpu, c_ram, c_vram, c_dev = st.columns(4)
    with c_cpu:
        st.markdown(
            f"<div class='top-statusbar'><span>⚡ <b>CPU:</b> {metrics['cpu_percent']:.1f}%</span></div>",
            unsafe_allow_html=True,
        )
    with c_ram:
        st.markdown(
            f"<div class='top-statusbar'><span>🧠 <b>RAM:</b> {metrics['ram_percent']:.1f}% ({metrics['ram_used_gb']:.1f}/{metrics['ram_total_gb']:.0f} GB)</span></div>",
            unsafe_allow_html=True,
        )
    with c_vram:
        st.markdown(
            f"<div class='top-statusbar'><span>🎮 <b>RTX 4070 VRAM:</b> {metrics['vram_used_gb']:.2f} / {metrics['vram_total_gb']:.2f} GB</span></div>",
            unsafe_allow_html=True,
        )
    with c_dev:
        st.markdown(
            f"<div class='top-statusbar'><span>🎯 <b>Device:</b> <code class='status-badge'>{device.type.upper()}</code></span></div>",
            unsafe_allow_html=True,
        )


def _plot_comparison_frame(
    image_np: np.ndarray,
    title: str,
    target_xy: tuple[float, float] | None = None,
    scale_pct: float = 85.0,
    target_label: str = "Asteroid",
) -> plt.Figure:
    """Erzeugt einen Dark-Mode-Plot für den Vorher/Nachher-Vergleich (Einzelbild vs. Shift-and-Stack)."""
    scale = max(float(scale_pct) / 100.0, 0.4)
    fig, ax = plt.subplots(figsize=(6.0 * scale, 5.0 * scale), facecolor="#0b0f19")
    ax.set_facecolor("#0b0f19")

    vmin, vmax = np.percentile(image_np, [5.0, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1.0

    im = ax.imshow(image_np, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    if target_xy is not None:
        tx, ty = target_xy
        ax.scatter(
            [tx],
            [ty],
            s=160,
            facecolors="none",
            edgecolors="#ef4444",
            linewidths=2.0,
            label=f"{target_label} ({tx:.1f}, {ty:.1f})",
        )
        ax.legend(
            loc="upper right",
            facecolor="#1e293b",
            edgecolor="#ef4444",
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


def _plot_inspector_frame(
    image_np: np.ndarray,
    title: str,
    sources_df: pd.DataFrame | None = None,
    scale_pct: float = 85.0,
) -> plt.Figure:
    """Creates a Dark-Mode matplotlib figure for the FITS Inspector (Raw vs. Background-subtracted with detections)."""
    scale = max(float(scale_pct) / 100.0, 0.4)
    fig, ax = plt.subplots(figsize=(6.0 * scale, 5.0 * scale), facecolor="#0b0f19")
    ax.set_facecolor("#0b0f19")

    vmin, vmax = np.percentile(image_np, [5.0, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1.0

    im = ax.imshow(image_np, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    if sources_df is not None and len(sources_df) > 0:
        ax.scatter(
            sources_df["x_centroid"],
            sources_df["y_centroid"],
            s=80,
            facecolors="none",
            edgecolors="#38bdf8",
            linewidths=1.2,
            label=f"Detected ({len(sources_df)})",
        )
        ax.legend(
            loc="upper right",
            facecolor="#1e293b",
            edgecolor="#38bdf8",
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


def _plot_stamp_strip(
    frame_stamps_np: list[np.ndarray],
    stack_stamp_np: np.ndarray,
    times_hours_list: list[float],
    scale_pct: float = 85.0,
) -> plt.Figure:
    """Rendert die 32x32 Stamps für Frame 0..N-1 sowie den Shift-and-Stack Stamp nebeneinander."""
    n_cols = len(frame_stamps_np) + 1
    scale = max(float(scale_pct) / 100.0, 0.4)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(2.2 * n_cols * scale, 2.5 * scale),
        facecolor="#0b0f19",
    )
    if n_cols == 1:
        axes = [axes]

    for idx, stamp in enumerate(frame_stamps_np):
        ax = axes[idx]
        ax.set_facecolor("#0b0f19")
        vmin, vmax = np.percentile(stamp, [5.0, 99.5])
        if vmax <= vmin:
            vmax = vmin + 1.0
        ax.imshow(stamp, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        t_val = times_hours_list[idx] if idx < len(times_hours_list) else float(idx)
        ax.set_title(f"Frame {idx}\n(t={t_val:.2f}h)", color="#f8fafc", fontsize=9)
        ax.tick_params( colors="#94a3b8", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    ax_stack = axes[-1]
    ax_stack.set_facecolor("#0b0f19")
    vmin_s, vmax_s = np.percentile(stack_stamp_np, [5.0, 99.5])
    if vmax_s <= vmin_s:
        vmax_s = vmin_s + 1.0
    ax_stack.imshow(stack_stamp_np, origin="lower", cmap="magma", vmin=vmin_s, vmax=vmax_s)
    ax_stack.set_title("Shift & Stack\n(Summen-Stamp)", color="#38bdf8", fontsize=9, fontweight="bold")
    ax_stack.tick_params(colors="#94a3b8", labelsize=7)
    for spine in ax_stack.spines.values():
        spine.set_color("#38bdf8")

    fig.tight_layout()
    return fig


def render_tab_fits_inspector(device: torch.device):
    """Tab 1: FITS-Inspector & Photometrie"""
    hdr_col1, hdr_col2 = st.columns([3, 1])
    with hdr_col1:
        st.subheader("🔭 FITS-Inspector & Photometrie")
    with hdr_col2:
        if st.button("❓ Parameter-Hilfe öffnen", key="help_btn_tab1", width="stretch"):
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
            help="Wählt die Quelle der FITS-Bilddaten: Lokales Verzeichnis data/, Astropy-Beispielbild oder eigener Datei-Upload.",
        )
    with col_ctrl1:
        sigma_thresh = st.slider(
            "Detection Threshold (sigma above background)",
            min_value=1.5,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Detektionsschwelle als Vielfaches des Hintergrundrauschens (1σ). Niedrigere Werte finden lichtschwächere Quellen.",
        )
    with col_ctrl2:
        fwhm_val = st.slider(
            "DAOStarFinder FWHM [px]",
            min_value=1.5,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Erwartete Halbwertsbreite (FWHM in Pixeln) der Stern-PSF für den DAOStarFinder Gauß-Filter.",
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
            help="Wählt eine lokale FITS-Datei aus dem Verzeichnis data/ zur photometrischen Analyse aus.",
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
            help="Lädt eine eigene FITS-Datei zum Inspectieren hoch.",
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
            image_t, wcs, header = load_fits_image(target_fits_path, device=device)
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
        m3.metric("Noise Level (1σ)", f"{noise_std:.2f} ADU")
        m4.metric("Detected Sources", f"{n_sources}")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            fig_raw = _plot_inspector_frame(
                raw_np,
                f"Raw Image [{height}×{width}]",
                scale_pct=st.session_state.get("plot_scale_pct", 85.0),
            )
            st.pyplot(fig_raw, width="stretch")
            plt.close(fig_raw)
        with p_col2:
            fig_res = _plot_inspector_frame(
                res_np,
                f"Background Subtracted + DAOStarFinder ({sigma_thresh:.1f}σ, FWHM={fwhm_val:.1f})",
                sources_df=sources_df,
                scale_pct=st.session_state.get("plot_scale_pct", 85.0),
            )
            st.pyplot(fig_res, width="stretch")
            plt.close(fig_res)

        st.markdown("#### 📋 Detected Sources Catalog")
        if sources_df is not None and len(sources_df) > 0:
            st.dataframe(sources_df, width="stretch", height=280)
        else:
            st.info(
                "No sources detected above the selected threshold. Try lowering the Detection Threshold slider."
            )

        with st.expander("📄 FITS Header Metadata"):
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
                st.dataframe(pd.DataFrame(hdr_rows), width="stretch", height=260)
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


def render_tab_pipeline_grid_search(device: torch.device):
    """Tab 2: GPU Shift-and-Stack Pipeline & Velocity Grid Search"""
    hdr_col1, hdr_col2 = st.columns([3, 1])
    with hdr_col1:
        st.subheader("🚀 GPU Shift-and-Stack Pipeline & Velocity Grid Search")
    with hdr_col2:
        if st.button("❓ Parameter-Hilfe öffnen", key="help_btn_tab2", width="stretch"):
            show_parameter_help_dialog()

    series_demo_dir = Path("data/series_demo")
    default_data_dir = Path("data")
    series_demo_dir.mkdir(parents=True, exist_ok=True)
    default_data_dir.mkdir(parents=True, exist_ok=True)

    col_src, col_btn = st.columns([2, 2])
    with col_src:
        selected_folder = st.radio(
            "Pipeline-Datenquelle wählen",
            options=["data/series_demo/", "data/"],
            horizontal=True,
            help="Wählt den Quellordner für die FITS-Zeitreihe: Synthetische 5-Frame-Demo oder reale Beobachtungsdaten in data/.",
        )
    with col_btn:
        if st.button(
            "☄️ Synthetische 5-Frame Asteroiden-Serie erzeugen (Low-SNR Demo)",
            width="stretch",
        ):
            try:
                with torch.no_grad():
                    paths, times, gt = generate_demo_asteroid_series(
                        output_dir="data/series_demo",
                        n_frames=5,
                        height=128,
                        width=128,
                        vx=2.0,
                        vy=-1.0,
                        asteroid_snr=1.5,
                        device=device,
                    )
                    st.session_state["demo_gt"] = gt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                st.success(
                    f"5-Frame FITS-Serie erfolgreich in `data/series_demo` erzeugt (Ground-Truth: vx={gt['vx']} px/h, vy={gt['vy']} px/h, Start=({gt['x0']:.1f}, {gt['y0']:.1f}))."
                )
            except Exception as exc:
                st.error(f"Fehler beim Erzeugen der 5-Frame-Serie: {exc}")

    active_dir = Path(selected_folder)
    fits_files = sorted(
        [
            str(p)
            for p in active_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".fits", ".fit", ".fts")
        ]
    )

    st.caption(
        f"Aktiver Ordner `{selected_folder}`: **{len(fits_files)} FITS-Datei(en)** gefunden | Compute-Device: **{device}**"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        vx_min = st.number_input("vx Suchbereich Min [px/h]", value=-2.0, step=0.5, help="Minimale horizontale Suchgeschwindigkeit vx in Pixeln pro Stunde entlang der X-Achse.")
        vx_max = st.number_input("vx Suchbereich Max [px/h]", value=2.0, step=0.5, help="Maximale horizontale Suchgeschwindigkeit vx in Pixeln pro Stunde entlang der X-Achse.")
    with c2:
        vy_min = st.number_input("vy Suchbereich Min [px/h]", value=-2.0, step=0.5, help="Minimale vertikale Suchgeschwindigkeit vy in Pixeln pro Stunde entlang der Y-Achse.")
        vy_max = st.number_input("vy Suchbereich Max [px/h]", value=2.0, step=0.5, help="Maximale vertikale Suchgeschwindigkeit vy in Pixeln pro Stunde entlang der Y-Achse.")
    with c3:
        grid_steps = st.slider("Grid-Schritte (grid_steps)", min_value=3, max_value=25, value=5, help="Anzahl der Gitterpunkte pro Geschwindigkeitsachse (erzeugt grid_steps × grid_steps Test-Trajektorien auf der GPU).")
        cnn_threshold = st.slider(
            "RealBogusCNN Score-Schwelle (cnn_threshold)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Mindest-Wahrscheinlichkeit des 3-Kanal RealBogusCNN, ab der ein Kandidat als echtes astronomisches Objekt akzeptiert wird.",
        )

    detection_threshold = st.slider(
        "DAOStarFinder Detektions-Schwelle [ADU über Hintergrund]",
        min_value=4.0,
        max_value=200.0,
        value=8.0,
        step=2.0,
        help="Absoluter Schwellenwert in ADU über dem Hintergrund im gestackten Summenbild für die DAOStarFinder-Quellendetektion.",
    )

    run_clicked = st.button("🚀 GPU-Grid-Search & Pipeline ausführen", width="stretch")

    if run_clicked:
        if len(fits_files) == 0:
            st.warning(
                f"Keine FITS-Dateien in `{selected_folder}` gefunden. Bitte zuerst oben auf '☄️ Synthetische 5-Frame Asteroiden-Serie erzeugen (Low-SNR Demo)' klicken."
            )
            return

        if vx_min > vx_max or vy_min > vy_max:
            st.error("Ungültiger Suchbereich: Min-Wert darf nicht größer als Max-Wert sein.")
            return

        try:
            with st.spinner(
                f"Führe GPU Velocity Grid Search ({grid_steps}×{grid_steps} Trajektorien) auf {len(fits_files)} FITS-Frames aus..."
            ):
                with torch.no_grad():
                    mjd_list = []
                    raw_frames_np = []
                    cleaned_frames_np = []
                    for fpath in fits_files:
                        img_t, wcs_obj, hdr = load_fits_image(fpath, device=device)
                        raw_frames_np.append(img_t.detach().cpu().numpy())
                        if hdr is not None and "MJD-OBS" in hdr:
                            mjd_list.append(float(hdr["MJD-OBS"]) * 24.0)

                        bkg_t = estimate_background(img_t)
                        masked_t = mask_stationary_stars(
                            img_t - bkg_t, threshold=150.0, fwhm=3.0
                        )
                        cleaned_frames_np.append(masked_t.detach().cpu().numpy())

                    if len(mjd_list) == len(fits_files):
                        times_hours = torch.tensor(
                            [m - mjd_list[0] for m in mjd_list],
                            dtype=torch.float32,
                            device=device,
                        )
                    else:
                        times_hours = torch.arange(
                            len(fits_files), dtype=torch.float32, device=device
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
                    frames_stack_t = torch.stack([torch.from_numpy(f) for f in cleaned_frames_np], dim=0)

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
                        frames=frames_stack_t,
                        times_hours=times_hours,
                        vx=best_vx,
                        vy=best_vy,
                        method="median",
                    )
                    stacked_np = shifted_stack_t.detach().cpu().numpy()

            st.success(
                f"Pipeline abgeschlossen: **{len(catalog)} Kandidat(en)** nach Sternmaskierung, Shift-and-Stack und RealBogusCNN-Filterung gefunden."
            )

            res = {
                "raw_frames_np": raw_frames_np,
                "cleaned_frames_np": cleaned_frames_np,
                "mjd_epochs": mjd_list,
                "times_hours_list": times_hours.detach().cpu().tolist(),
                "wcs_ref": wcs_obj if wcs_obj is not None else create_synthetic_wcs(),
                "stacked_np": stacked_np,
                "catalog_df": catalog.to_pandas(),
                "best_candidate": {
                    "x0": float(catalog[0]["x_centroid"]) if len(catalog) > 0 else 0.0,
                    "y0": float(catalog[0]["y_centroid"]) if len(catalog) > 0 else 0.0,
                    "vx": best_vx,
                    "vy": best_vy,
                    "ra": float(catalog[0]["ra"]) if len(catalog) > 0 else 0.0,
                    "dec": float(catalog[0]["dec"]) if len(catalog) > 0 else 0.0,
                    "peak": float(catalog[0]["peak"]) if len(catalog) > 0 else 0.0,
                    "cnn_score": float(catalog[0]["cnn_score"]) if len(catalog) > 0 else 0.0,
                    "height": frame0_np.shape[0],
                    "width": frame0_np.shape[1],
                    "base_mjd": mjd_list[0] / 24.0 if len(mjd_list) > 0 else 0.0,
                },
                "target_xy": target_xy,
            }
            st.session_state["pipeline_result"] = res

        except Exception as exc:
            st.error(f"Fehler während der Pipeline-Ausführung: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    res = st.session_state.get("pipeline_result")
    if res is not None:
        n_total = len(res["raw_frames_np"])
        nav_c1, nav_c2, nav_c3 = st.columns([2, 2, 2])
        with nav_c1:
            frame_idx = st.slider(
                "🎞️ Original-Frame durchschalten (Frame i)",
                min_value=0,
                max_value=max(0, n_total - 1),
                value=0,
                step=1,
                help="Schaltet interaktiv durch die einzelnen Beobachtungsepochen (Frame 0 bis N-1) der FITS-Serie.",
            )
        with nav_c2:
            view_mode = st.radio(
                "Ansicht Links",
                options=["Rohbild (mit Feldsternen)", "Bereinigt (Feldsterne maskiert)"],
                horizontal=True,
                help="Wechselt zwischen dem unbereinigten Original-Frame und dem Hintergrund-abgezogenen Frame nach Maskierung stationärer Feldsterne.",
            )
        with nav_c3:
            t_h = res["times_hours_list"][frame_idx]
            mjd_i = res["mjd_epochs"][frame_idx]
            st.caption(f"⏱️ **Epoche Frame {frame_idx}:** `t = +{t_h:.2f} h` | `MJD = {mjd_i:.5f}`")
            show_trail = st.checkbox(
                "Asteroiden-Spur im Einzelbild markieren",
                value=True,
                help="Markiert die berechnete Position (x0 + vx·dt, y0 + vy·dt) des besten Kandidaten zum Zeitpunkt des gewählten Einzelbildes.",
            )

        if show_trail and res["target_xy"] is not None:
            dt_i = res["times_hours_list"][frame_idx] - res["times_hours_list"][0]
            frame_marker_xy = (res["target_xy"][0] + res["best_vx"] * dt_i, res["target_xy"][1] + res["best_vy"] * dt_i)
        else:
            frame_marker_xy = None

        left_np = res["raw_frames_np"][frame_idx] if view_mode.startswith("Rohbild") else res["cleaned_frames_np"][frame_idx]

        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            fig_left = _plot_comparison_frame(
                left_np,
                f"Frame {frame_idx} (t={t_h:.2f}h)",
                target_xy=frame_marker_xy,
                scale_pct=st.session_state.get("plot_scale_pct", 85.0),
            )
            st.pyplot(fig_left, width="stretch")
            plt.close(fig_left)
        with col_plot2:
            fig_stack = _plot_comparison_frame(
                res["stacked_np"],
                f"Shift-and-Stack Summenbild (vx={res['best_vx']:.2f}, vy={res['best_vy']:.2f} px/h)",
                target_xy=res["target_xy"],
                scale_pct=st.session_state.get("plot_scale_pct", 85.0),
            )
            st.pyplot(fig_stack, width="stretch")
            plt.close(fig_stack)

        if len(res["catalog_df"]) > 0:
            st.dataframe(res["catalog_df"], width="stretch", height=260)
            selected_cand_idx = st.selectbox(
                "🎯 Kandidat für Orbit-, Stamp- & Stellarium-Analyse wählen",
                options=list(range(len(res["catalog_df"]))),
                index=0,
                help="Wählt einen Kandidaten aus der Detektionskatalogtabelle zur detaillierten Analyse.",
            )
            selected_cand = res["catalog_df"].iloc[selected_cand_idx]
            cand_x0 = float(selected_cand["x_centroid"])
            cand_y0 = float(selected_cand["y_centroid"])
            cand_vx = float(selected_cand["vx"])
            cand_vy = float(selected_cand["vy"])

            frame_stamps_np = []
            for i in range(n_total):
                frame_t = torch.from_numpy(res["raw_frames_np"][i])
                xi = cand_x0 + cand_vx * (res["times_hours_list"][i] - res["times_hours_list"][0])
                yi = cand_y0 + cand_vy * (res["times_hours_list"][i] - res["times_hours_list"][0])
                stamp = extract_stamps(frame_t, [(xi, yi)], stamp_size=32)[0, 0].numpy()
                frame_stamps_np.append(stamp)

            stack_stamp_t = extract_stamps(
                torch.from_numpy(res["stacked_np"]),
                [(cand_x0, cand_y0)],
                stamp_size=32,
            )[0, 0].numpy()

            fig_stamps = _plot_stamp_strip(
                frame_stamps_np,
                stack_stamp_t,
                res["times_hours_list"],
                scale_pct=st.session_state.get("plot_scale_pct", 85.0),
            )
            st.pyplot(fig_stamps, width="stretch")
            plt.close(fig_stamps)

            tracklet = compute_tracklet_endpoints(
                res["wcs_ref"],
                cand_x0,
                cand_y0,
                cand_vx,
                cand_vy,
                res["times_hours_list"],
                base_mjd=res["mjd_epochs"][0] / 24.0,
            )
            orb = estimate_kepler_orbit(tracklet)
            exit_info = predict_boundary_exit(
                res["wcs_ref"],
                cand_x0,
                cand_y0,
                cand_vx,
                cand_vy,
                height=res["best_candidate"]["height"],
                width=res["best_candidate"]["width"],
                base_mjd=res["mjd_epochs"][0] / 24.0,
            )

            known_catalog = None
            if "demo_gt" in st.session_state:
                known_catalog = Table({
                    "x0": [st.session_state["demo_gt"]["x0"]],
                    "y0": [st.session_state["demo_gt"]["y0"]],
                    "vx": [st.session_state["demo_gt"]["vx"]],
                    "vy": [st.session_state["demo_gt"]["vy"]],
                    "ra": [st.session_state["demo_gt"]["ra"]],
                    "dec": [st.session_state["demo_gt"]["dec"]],
                })

            if known_catalog is not None:
                cross_match = cross_match_known_objects(tracklet, known_catalog)
                st.markdown(f"**Cross-Match Status:** {cross_match[0]['match_status']}")
            else:
                st.markdown("**Cross-Match Status:** Kein Referenzkatalog verfügbar.")

            c_orb, c_exit, c_mpc, c_stel = st.columns([2, 2, 2, 2])
            with c_orb:
                st.markdown(f"**Orbit-Klasse:** {orb['orbit_class']}")
                st.markdown(f"**Große Halbachse a:** {orb['a_au']:.2f} AU")
                st.markdown(f"**Exzentrizität e / Inkl. i:** e={orb['e']:.2f} | i={orb['i_deg']:.1f}°")
            with c_exit:
                st.markdown(f"**Feld-Austritt (Tracker):** {exit_info['exit_edge']} in {exit_info['t_exit_hours']:.1f} h")
                st.caption(f"RA_next: {exit_info['RA_next']:.5f}° | Dec_next: {exit_info['Dec_next']:.5f}°")
            with c_mpc:
                mpc_block = export_tracklet_mpc80(tracklet, designation="K26A00A")
                st.code(mpc_block, language="text")
            with c_stel:
                if st.button("🔭 In Stellarium zentrieren (localhost:8090)", key=f"slew_{selected_cand_idx}"):
                    slew_res = slew_stellarium_to_candidate(
                        ra_deg=tracklet[0]["ra_deg"],
                        dec_deg=tracklet[0]["dec_deg"],
                        mjd=tracklet[0]["mjd"],
                    )
                    if slew_res["ok"]:
                        st.success(f"Kandidat in Stellarium zentriert (J2000: {slew_res['j2000_vec']}).")
                    else:
                        st.warning(f"Stellarium nicht erreichbar ({slew_res['error']}). Payload J2000={slew_res['j2000_vec']}")

        else:
            st.info("Keine Kandidaten gefunden. Bitte führen Sie die Pipeline erneut aus.")
    else:
        st.info("Keine Pipeline-Ergebnisse verfügbar. Bitte führen Sie die Pipeline aus.")


def main():
    st.set_page_config(page_title="AstroProject", layout="wide")
    st.markdown(
        """
        <style>
        .css-18ni7ap {
            display: none;
        }
        .css-1d391kg {
            padding-top: 0rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    device = render_sidebar_controls()
    render_top_statusbar(device)

    tabs = st.tabs(["FITS-Inspector", "Pipeline & Analysis"])
    with tabs[0]:
        render_tab_fits_inspector(device)
    with tabs[1]:
        render_tab_pipeline_grid_search(device)


if __name__ == "__main__":
    main()
