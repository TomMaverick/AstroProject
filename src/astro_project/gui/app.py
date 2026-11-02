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

from astro_project.core.classifier import RealBogusCNN
from astro_project.core.fits_loader import load_fits_image
from astro_project.core.photometry import detect_sources, estimate_background
from astro_project.core.synthetic import generate_demo_asteroid_series
from astro_project.core.wcs_utils import sources_to_skycoords
from astro_project.pipeline.kinematics import (
    kinematic_shift_and_stack,
    mask_stationary_stars,
)
from astro_project.pipeline.orchestration import run_pipeline


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
</style>
"""

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

        return torch.device(
            selected_dev if selected_dev == "cuda" and cuda_available else "cpu"
        )


def render_top_statusbar(device: torch.device):
    """Rendert die ultra-kompakte ~1cm (34px) horizontale Statusbar ganz oben im Hauptbereich."""
    metrics = get_hardware_metrics()
    st.markdown(
        f"""
        <div class="top-statusbar">
            <span>⚡ <b>CPU:</b> {metrics['cpu_percent']:.1f}%</span>
            <span>|</span>
            <span>🧠 <b>RAM:</b> {metrics['ram_percent']:.1f}% ({metrics['ram_used_gb']:.1f}/{metrics['ram_total_gb']:.0f} GB)</span>
            <span>|</span>
            <span>🎮 <b>GPU:</b> {metrics['gpu_name']}</span>
            <span>|</span>
            <span>💾 <b>VRAM:</b> {metrics['vram_used_gb']:.2f} / {metrics['vram_total_gb']:.2f} GB</span>
            <span>|</span>
            <span>🎯 <b>Device:</b> <code class="status-badge">{device.type.upper()}</code></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_comparison_frame(
    image_np: np.ndarray,
    title: str,
    target_xy: tuple[float, float] | None = None,
) -> plt.Figure:
    """Erzeugt einen Dark-Mode-Plot für den Vorher/Nachher-Vergleich (Frame 0 vs. Shift-and-Stack)."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0b0f19")
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
            label=f"Asteroid ({tx:.1f}, {ty:.1f})",
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


def render_tab_pipeline_grid_search(device: torch.device):
    """Tab 2: Interaktive Zeitreihen-Demo, GPU Velocity Grid Search & Vorher/Nachher-Visualisierung."""
    st.subheader("🚀 GPU Shift-and-Stack Pipeline & Velocity Grid Search")

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
        )
    with col_btn:
        if st.button(
            "☄️ Synthetische 5-Frame Asteroiden-Serie erzeugen (Low-SNR Demo)",
            width="stretch",
        ):
            try:
                with torch.no_grad():
                    paths, times, gt = generate_demo_asteroid_series(
                        output_dir=series_demo_dir,
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
                    f"5-Frame FITS-Serie erfolgreich in `{series_demo_dir}` erzeugt (Ground-Truth: vx={gt['vx']} px/h, vy={gt['vy']} px/h, Start=({gt['x0']:.1f}, {gt['y0']:.1f}))."
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
        vx_min = st.number_input("vx Suchbereich Min [px/h]", value=-2.0, step=0.5)
        vx_max = st.number_input("vx Suchbereich Max [px/h]", value=2.0, step=0.5)
    with c2:
        vy_min = st.number_input("vy Suchbereich Min [px/h]", value=-2.0, step=0.5)
        vy_max = st.number_input("vy Suchbereich Max [px/h]", value=2.0, step=0.5)
    with c3:
        grid_steps = st.slider("Grid-Schritte (grid_steps)", min_value=3, max_value=25, value=5)
        cnn_threshold = st.slider(
            "RealBogusCNN Score-Schwelle (cnn_threshold)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
        )

    detection_threshold = st.slider(
        "DAOStarFinder Detektions-Schwelle [ADU über Hintergrund]",
        min_value=4.0,
        max_value=200.0,
        value=8.0,
        step=2.0,
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
                    raw_frames = []
                    cleaned_frames = []
                    for fpath in fits_files:
                        img_t, _, hdr = load_fits_image(fpath, device=device)
                        raw_frames.append(img_t)
                        if hdr is not None and "MJD-OBS" in hdr:
                            mjd_list.append(float(hdr["MJD-OBS"]) * 24.0)

                        bkg_t = estimate_background(img_t)
                        masked_t = mask_stationary_stars(
                            img_t - bkg_t, threshold=150.0, fwhm=3.0
                        )
                        cleaned_frames.append(masked_t)

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

                    frame0_np = raw_frames[0].detach().cpu().numpy()
                    frames_stack_t = torch.stack(cleaned_frames, dim=0)

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

            col_plot1, col_plot2 = st.columns(2)
            with col_plot1:
                fig_f0 = _plot_comparison_frame(
                    frame0_np,
                    "Frame 0 (Einzelbild mit Rauschen & Feldsternen)",
                )
                st.pyplot(fig_f0, width="stretch")
                plt.close(fig_f0)
            with col_plot2:
                fig_stack = _plot_comparison_frame(
                    stacked_np,
                    f"Shift-and-Stack Summenbild (vx={best_vx:.2f}, vy={best_vy:.2f} px/h)",
                    target_xy=target_xy,
                )
                st.pyplot(fig_stack, width="stretch")
                plt.close(fig_stack)

            if len(catalog) > 0:
                df_cat = catalog.to_pandas()
                st.dataframe(df_cat, width="stretch", height=320)
            else:
                st.info(
                    "Keine Kandidaten oberhalb der gewählten Detektions- und CNN-Schwellenwerte gefunden."
                )
        except Exception as exc:
            st.error(f"Fehler während der Pipeline-Ausführung: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def render_tab_fits_inspector(device: torch.device):
    """Tab 1: FITS-Inspector & Photometrie"""
    st.subheader("🔭 FITS-Inspector & Photometrie")

    # Placeholder content for the FITS-Inspector tab
    st.write("This is the FITS-Inspector tab. You can add your implementation here.")


def main():
    st.set_page_config(
        page_title="AstroProject | GPU Asteroid Pipeline",
        page_icon="☄️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)

    device = render_sidebar_controls()
    render_top_statusbar(device=device)

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
