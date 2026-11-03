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
        width: 100%;
        box-sizing: border-box;
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

@st.dialog("\u2753 AstroProject \u2014 Parameter- & Physik-Handbuch", width="large")
def show_parameter_help_dialog():
    """Schlie\u00dfbares Dark-Mode-Modal-Fenster mit Erkl\u00e4rungen zu allen Parametern."""
    st.markdown(
        """
        ### \U0001f52d FITS-Inspector & Photometry
        * **FITS Data Source:** W\u00e4hlt die Bildquelle aus dem lokalen Ordner `data/`, dem integrierten Astropy-Beispielbild (`HorseHead.fits`) oder einem eigenen Datei-Upload.
        * **Detection Threshold (\u03c3 \u00fcber Hintergrund):** Bestimmt, ab welchem Vielfachen des Hintergrundrauschens ($1\\sigma$) eine Punktquelle durch `DAOStarFinder` als Stern/Objekt akzeptiert wird. Niedrigere Werte finden schw\u00e4chere Quellen, erh\u00f6hen aber Falschdetektionen.
        * **DAOStarFinder FWHM [px]:** Erwartete Halbwertsbreite (Full Width at Half Maximum) der teleskopischen Punktspreizfunktion (PSF) in Pixeln f\u00fcr den Gau\u00df-Matched-Filter.

        ---
        ### \U0001f680 Velocity Grid Search & Orbit Analysis
        * **Pipeline-Datenquelle:** W\u00e4hlt zwischen der synthetischen Demo (`data/series_demo/`) und realen FITS-Sequenzen in `data/`.
        * **vx / vy Suchbereich Min & Max [px/h]:** Definiert den Geschwindigkeitsraum in Pixeln pro Stunde entlang der $[Y, X]$-Bildachsen (`vx` in X-Richtung, `vy` in Y-Richtung), der auf der GPU abgetastet wird.
        * **Grid-Schritte (`grid_steps`):** Anzahl der St\u00fctzstellen pro Geschwindigkeitsachse. Ein Wert von $N$ erzeugt ein $N \\times N$ Trajektorien-Gitter.
        * **RealBogusCNN Score-Schwelle (`cnn_threshold`):** Mindest-Klassifikationswahrscheinlichkeit $[0.0, 1.0]$ des 3-Kanal-CNNs (Mean, Median, Variance), ab der ein gestackter Kandidat als echtes bewegtes Objekt (`Real`) eingestuft wird.
        * **DAOStarFinder Detektions-Schwelle [ADU \u00fcber Hintergrund]:** Absoluter Schwellenwert im Shift-and-Stack-Summenbild. Da das Signal-zu-Rausch-Verh\u00e4ltnis beim Stacking mit $\\sqrt{N_{\\text{Frames}}}$ w\u00e4chst, treten hier zuvor unsichtbare Sub-Sigma-Asteroiden deutlich hervor.
        * **\U0001f39e\ufe0f Frame-Navigator & Spur-Overlay:** Erlaubt das Durchbl\u00e4ttern der Einzelbilder $0 \\dots N-1$ (Rohbild vs. sternmaskiertes Bild) im direkten Vergleich zum Shift-and-Stack-Summenbild.

        ---
        ### \U0001fa90 Orbit Estimation & Stellarium Integration
        * **V\u00e4is\u00e4l\u00e4-Kepler-Orbit:** Sch\u00e4tzt aus der kurzen Winkelgeschwindigkeit am Himmel die gro\u00dfe Halbachse $a\\,[\\text{AU}]$, Exzentrizit\u00e4t $e$, Inklination $i\\,[^\\circ]$ und die Orbit-Klasse (z.B. Main-Belt, NEA).
        * **Nachbarfeld-Tracking:** Berechnet Zeitpunkt und Himmelskoordinate, an denen das Objekt den aktuellen Sensorbereich $[0, W] \\times [0, H]$ verl\u00e4sst.
        * **Stellarium Live-Slew:** Zentriert die virtuelle Kamera in Stellarium \u00fcber das RemoteControl-Plugin (`http://localhost:8090/api`) direkt auf die Kandidaten-Koordinate.
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
    """Rendert die Device-Auswahl (Standard: CUDA falls verf\u00fcgbar) und Cache-Steuerung in der Sidebar."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.init()
    with st.sidebar:
        st.markdown("### \u2699\ufe0f Compute & VRAM Control")
        options = ["cuda", "cpu"] if cuda_available else ["cpu"]
        selected_dev = st.selectbox(
            "Ziel-Device",
            options=options,
            index=0,
            help="W\u00e4hlt das PyTorch-Device f\u00fcr [Y, X] Tensor-Operationen (CUDA Standard auf RTX 4070).",
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("\U0001f504 Refresh", width="stretch"):
                st.rerun()
        with col_btn2:
            if st.button("\U0001f9f9 VRAM Flush", width="stretch"):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                st.rerun()

        plot_scale_pct = st.slider(
            "\U0001f50d Plot-Gr\u00f6\u00dfe [%]",
            min_value=50,
            max_value=100,
            value=85,
            step=5,
            help="Skaliert die Darstellungsgr\u00f6\u00dfe (figsize) aller Matplotlib-Plots dynamisch f\u00fcr optimales Rein-/Rauszoomen.",
        )
        st.session_state["plot_scale_pct"] = float(plot_scale_pct)

        return resolve_device(
            selected_dev if selected_dev == "cuda" and cuda_available else "cpu"
        )


def render_top_statusbar(device: torch.device):
    """Rendert die ultra-kompakte ~1cm (34px) horizontale Statusbar ueber volle Breite ganz oben im Hauptbereich."""
    metrics = get_hardware_metrics()
    st.markdown(
        f"<div class='top-statusbar'>"
        f"<span>\u26a1 <b>CPU:</b> {metrics['cpu_percent']:.1f}%</span>"
        f"<span>\U0001f9e0 <b>RAM:</b> {metrics['ram_percent']:.1f}% ({metrics['ram_used_gb']:.1f}/{metrics['ram_total_gb']:.0f} GB)</span>"
        f"<span>\U0001f3ae <b>RTX 4070 VRAM:</b> {metrics['vram_used_gb']:.2f} / {metrics['vram_total_gb']:.2f} GB</span>"
        f"<span>\U0001f3af <b>Device:</b> <code class='status-badge'>{device.type.upper()}</code></span>"
        f"</div>",
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


def _plot_solar_system_orbit(
    a_nom: float,
    e_nom: float,
    a_samples: list[float] | np.ndarray,
    e_samples: list[float] | np.ndarray,
    scale_pct: float = 85.0,
) -> plt.Figure:
    """Creates a top-down ecliptic Dark-Mode plot of the nominal orbit and Monte Carlo uncertainty envelope."""
    scale = max(float(scale_pct) / 100.0, 0.4)
    fig, ax = plt.subplots(figsize=(6.0 * scale, 5.5 * scale), facecolor="#0b0f19")
    ax.set_facecolor("#0b0f19")

    theta = np.linspace(0.0, 2.0 * np.pi, 360)

    # Monte Carlo uncertainty envelope (alpha=0.1)
    for a_mc, e_mc in zip(a_samples, e_samples):
        a_val = float(a_mc)
        e_val = float(np.clip(e_mc, 0.0, 0.95))
        if a_val <= 0.0:
            continue
        r_mc = (a_val * (1.0 - e_val**2)) / (1.0 + e_val * np.cos(theta))
        ax.plot(
            r_mc * np.cos(theta),
            r_mc * np.sin(theta),
            color="#38bdf8",
            alpha=0.1,
            linewidth=1.0,
        )

    # Earth orbit at 1 AU
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        color="#94a3b8",
        linestyle="--",
        linewidth=1.3,
        label="Earth Orbit (1.0 AU)",
    )

    # Nominal candidate orbit
    a_nom_val = max(float(a_nom), 0.1)
    e_nom_val = float(np.clip(e_nom, 0.0, 0.95))
    r_nom = (a_nom_val * (1.0 - e_nom_val**2)) / (1.0 + e_nom_val * np.cos(theta))
    ax.plot(
        r_nom * np.cos(theta),
        r_nom * np.sin(theta),
        color="#ef4444",
        linewidth=2.2,
        label=f"Nominal Orbit (a={a_nom_val:.2f} AU, e={e_nom_val:.2f})",
    )

    # Sun at (0, 0)
    ax.scatter(
        [0.0],
        [0.0],
        color="#facc15",
        s=130,
        zorder=5,
        edgecolors="#fef08a",
        label="Sun (0, 0)",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Ecliptic Top-Down Orbit & MC Error Envelope",
        color="#f8fafc",
        fontsize=11,
    )
    ax.set_xlabel("X [AU]", color="#94a3b8")
    ax.set_ylabel("Y [AU]", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, color="#1e293b", linestyle=":", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.legend(
        loc="upper right",
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#f8fafc",
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def render_tab_fits_inspector(device: torch.device):
    """FITS-Inspector & Photometry"""
    hdr_col1, hdr_col2 = st.columns([3, 1])
    with hdr_col1:
        st.subheader("\U0001f52d FITS-Inspector & Photometry")
    with hdr_col2:
        if st.button("\u2753 Parameter-Hilfe \u00f6ffnen", key="help_btn_tab1", width="stretch"):
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
            st.dataframe(sources_df, width="stretch", height=280)
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
    """Velocity Grid Search & Orbit Analysis"""
    hdr_col1, hdr_col2 = st.columns([3, 1])
    with hdr_col1:
        st.subheader("\U0001f680 Velocity Grid Search & Orbit Analysis")
    with hdr_col2:
        if st.button("\u2753 Parameter-Hilfe \u00f6ffnen", key="help_btn_tab2", width="stretch"):
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

    with st.expander("\u2699\ufe0f Parameter fuer synthetische Bildserie (Low-SNR)", expanded=False):
        randomize_series = st.checkbox(
            "\U0001f3b2 Parameter zufaellig generieren",
            value=False,
            help="Zieht Anzahl Frames (3\u201315), Geschwindigkeiten vx/vy, Asteroiden-SNR und Startposition (x0, y0) zuf\u00e4llig.",
        )
        if not randomize_series:
            cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
            with cfg_c1:
                custom_n_frames = st.slider(
                    "Anzahl Frames",
                    min_value=3,
                    max_value=15,
                    value=5,
                    step=1,
                    help="Anzahl der zeitlich aufeinanderfolgenden FITS-Aufnahmen in data/series_demo/.",
                )
                custom_n_stars = st.slider(
                    "Anzahl Feldsterne",
                    min_value=0,
                    max_value=10,
                    value=5,
                    step=1,
                    help="Anzahl der hellen station\u00e4ren Hintergrundsterne im Bildfeld.",
                )
            with cfg_c2:
                custom_vx = st.slider(
                    "Geschwindigkeit vx [px/h]",
                    min_value=-5.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.25,
                    help="Wahre horizontale Eigenbewegung des synthetischen Asteroiden in Pixeln pro Stunde.",
                )
                custom_vy = st.slider(
                    "Geschwindigkeit vy [px/h]",
                    min_value=-5.0,
                    max_value=5.0,
                    value=-1.0,
                    step=0.25,
                    help="Wahre vertikale Eigenbewegung des synthetischen Asteroiden in Pixeln pro Stunde.",
                )
            with cfg_c3:
                custom_snr = st.slider(
                    "Asteroiden-SNR (Einzelbild)",
                    min_value=0.8,
                    max_value=5.0,
                    value=1.5,
                    step=0.1,
                    help="Signal-zu-Rausch-Verh\u00e4ltnis des Asteroiden pro Einzelaufnahme.",
                )
        else:
            custom_n_frames, custom_vx, custom_vy, custom_snr, custom_n_stars = (
                5,
                2.0,
                -1.0,
                1.5,
                5,
            )
            st.caption(
                "Im Zufalls-Modus werden SNR, Geschwindigkeiten, Sterne und Position automatisch vom Algorithmus variiert."
            )

        if st.button("\U0001f680 Serie generieren", width="stretch"):
            try:
                with torch.no_grad():
                    paths, times, gt = generate_demo_asteroid_series(
                        output_dir="data/series_demo",
                        n_frames=int(custom_n_frames),
                        height=128,
                        width=128,
                        vx=float(custom_vx),
                        vy=float(custom_vy),
                        asteroid_snr=float(custom_snr),
                        n_stars=int(custom_n_stars),
                        randomize=bool(randomize_series),
                        device=device,
                    )
                    st.session_state["demo_gt"] = gt
                    st.session_state.pop("pipeline_result", None)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                st.success(
                    f"{gt['n_frames']}-Frame FITS-Serie in `data/series_demo` generiert "
                    f"(Ground-Truth: vx={gt['vx']:.2f} px/h, vy={gt['vy']:.2f} px/h, "
                    f"SNR={gt['asteroid_snr']:.2f}, Sterne={gt['n_stars']}, "
                    f"Start=({gt['x0']:.1f}, {gt['y0']:.1f}))."
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

    run_clicked = st.button("\U0001f680 GPU-Grid-Search & Pipeline ausf\u00fchren", width="stretch")

    if run_clicked:
        if len(fits_files) == 0:
            st.warning(
                f"Keine FITS-Dateien in `{selected_folder}` gefunden. Bitte zuerst oben im Expander auf '\U0001f680 Serie generieren' klicken."
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
                    mjd_days_list = []
                    raw_frames_np = []
                    cleaned_frames_np = []
                    ref_wcs = None
                    for idx, fpath in enumerate(fits_files):
                        img_t, wcs_obj, hdr = load_fits_image(fpath, device=device)
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

            st.success("Pipeline erfolgreich ausgeführt.")

            # Display results
            if len(catalog) > 0:
                st.markdown("#### \U0001f4cb Erkannte Kandidaten")
                st.dataframe(catalog.to_pandas(), width="stretch", height=280)

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
                )
                st.pyplot(fig_orbit, use_container_width=True)
                plt.close(fig_orbit)
            else:
                st.warning("Keine Kandidaten erkannt.")

        except Exception as exc:
            st.error(f"Fehler beim Ausführen der Pipeline: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    st.set_page_config(page_title="AstroProject", page_icon="*", layout="wide")
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
    device = render_sidebar_controls()

    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0.3rem;'>AstroProject</h1>",
        unsafe_allow_html=True,
    )
    render_top_statusbar(device)

    tabs = st.tabs(
        [
            "\U0001f52d FITS-Inspector & Photometry",
            "\U0001f680 Velocity Grid Search & Orbit Analysis",
        ]
    )
    with tabs[0]:
        render_tab_fits_inspector(device)
    with tabs[1]:
        render_tab_pipeline_grid_search(device)


if __name__ == "__main__":
    main()
