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
from astro_project.core.fits_loader import load_fits_image, save_fits_image
from astro_project.core.photometry import detect_sources, estimate_background
from astro_project.core.wcs_utils import create_synthetic_wcs, sources_to_skycoords
from astro_project.pipeline.orchestration import run_pipeline


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
    """Renders the device selection and cache control in the left sidebar."""
    cuda_available = torch.cuda.is_available()
    with st.sidebar:
        st.markdown("### ⚙️ Compute & VRAM Control")
        options = ["cuda", "cpu"] if cuda_available else ["cpu"]
        selected_dev = st.selectbox(
            "Target Device",
            options=options,
            index=0,
            help="Selects the PyTorch device for [Y, X] tensor operations.",
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


def render_header_statusbar():
    """Renders the compact horizontal hardware status bar directly under the main title."""
    metrics = get_hardware_metrics()
    col_cpu, col_ram, col_vram = st.columns(3)

    with col_cpu:
        st.metric(
            label="CPU Utilization",
            value=f"{metrics['cpu_percent']:.1f} %",
            delta="Live System Load",
            delta_color="off",
        )
        st.progress(min(max(int(metrics["cpu_percent"]), 0), 100))

    with col_ram:
        st.metric(
            label="RAM Utilization",
            value=f"{metrics['ram_percent']:.1f} %",
            delta=f"{metrics['ram_used_gb']:.1f} / {metrics['ram_total_gb']:.1f} GB",
            delta_color="off",
        )
        st.progress(min(max(int(metrics["ram_percent"]), 0), 100))

    with col_vram:
        vram_label = (
            f"GPU VRAM ({metrics['gpu_name']})"
            if metrics["cuda_available"]
            else "RTX 4070 VRAM (CUDA inactive)"
        )
        vram_delta = (
            f"{metrics['vram_percent']:.1f} % used (PyTorch Reserved: {metrics['vram_reserved_gb']:.2f} GB)"
            if metrics["cuda_available"]
            else "Fallback: CPU"
        )
        st.metric(
            label=vram_label,
            value=f"{metrics['vram_used_gb']:.2f} / {metrics['vram_total_gb']:.2f} GB",
            delta=vram_delta,
            delta_color="off",
        )
        st.progress(min(max(int(metrics["vram_percent"]), 0), 100))


def _plot_fits_frame(
    image_np: np.ndarray,
    title: str,
    sources=None,
) -> plt.Figure:
    """Generates a contrast-enhanced dark-mode plot of a [Y, X] image with optional detections."""
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
            label=f"Detections ({len(sources)})",
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
    st.subheader("🔭 FITS Inspector & Photometry ([Y, X] Tensor Pipeline)")

    col_src, col_params = st.columns([2, 2])
    with col_src:
        data_source = st.radio(
            "Select data source",
            options=[
                "Local `data/` directory / Astropy Sample FITS",
                "Upload own FITS file",
            ],
            horizontal=True,
        )

        fits_path: str | None = None
        temp_file_path: Path | None = None

        if data_source == "Upload own FITS file":
            uploaded = st.file_uploader(
                "Select FITS image (.fits, .fit, .fts)",
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
            choice = st.selectbox("Available FITS files", options=options)
            if choice == "[Astropy Standard] HorseHead.fits":
                try:
                    fits_path = get_pkg_data_filename("tutorials/FITS-images/HorseHead.fits")
                except Exception as exc:
                    st.error(f"Astropy sample file could not be loaded: {exc}")
            else:
                fits_path = choice

    with col_params:
        c1, c2 = st.columns(2)
        with c1:
            sigma_factor = st.slider(
                "Detection threshold (σ above background)", 2.0, 20.0, 5.0, 0.5
            )
        with c2:
            fwhm = st.slider("DAOStarFinder FWHM [Pixel]", 1.5, 10.0, 3.0, 0.5)

    if not fits_path:
        st.info("Please select or upload a FITS file to start the analysis.")
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

        m1, m2, m3, m4 = st.columns(4)
        height, width = image_tensor.shape
        m1.metric("Tensor Shape [Y, X]", f"[{height}, {width}]")
        m2.metric("Background Median", f"{float(torch.median(bkg_tensor).item()):.2f} ADU")
        m3.metric("Noise Level (1σ)", f"{noise_std:.2f} ADU")
        num_det = len(sources) if sources is not None else 0
        m4.metric("Detected Sources", f"{num_det}")

        p1, p2 = st.columns(2)
        with p1:
            fig_raw = _plot_fits_frame(img_np, "Raw Image [Y, X]")
            st.pyplot(fig_raw, width="stretch")
            plt.close(fig_raw)
        with p2:
            fig_det = _plot_fits_frame(
                res_np, "Background Subtracted + DAOStarFinder", sources=sources
            )
            st.pyplot(fig_det, width="stretch")
            plt.close(fig_det)

        st.markdown("#### 🌐 Astrometric Source Table (WCS Pixel-to-Sky `origin=0`)")
        if sources is not None and len(sources) > 0:
            df = sources.to_pandas()
            if wcs is not None and wcs.is_celestial:
                try:
                    sky_coords = sources_to_skycoords(sources, wcs)
                    df["ra_deg"] = np.asarray(sky_coords.ra.deg, dtype=np.float64)
                    df["dec_deg"] = np.asarray(sky_coords.dec.deg, dtype=np.float64)
                except Exception as wcs_exc:
                    st.warning(f"WCS transformation not possible: {wcs_exc}")

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
            st.warning("No sources detected above the selected threshold.")

        with st.expander("📄 FITS Header & WCS Metadata"):
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

    except Exception as exc:
        st.error(f"Error processing the FITS file: {exc}")
    finally:
        if temp_file_path is not None and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _generate_demo_fits_in_data_dir(data_dir: Path, device: torch.device) -> list[str]:
    """Generates 3 synthetic FITS frames with a moving asteroid in the local `data/` directory."""
    data_dir.mkdir(parents=True, exist_ok=True)
    height, width = 120, 140
    wcs = create_synthetic_wcs(pixel_scale_arcsec=1.0, crval_ra=180.0, crval_dec=45.0)

    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    sigma = 1.5
    x0, y0 = 75.0, 65.0
    vx, vy = 2.0, -1.0
    star_x, star_y = 35.0, 30.0

    created_paths: list[str] = []
    for i in range(3):
        dt = float(i)
        frame = (
            torch.full((height, width), 100.0, dtype=torch.float32, device=device)
            + torch.randn((height, width), dtype=torch.float32, device=device) * 1.5
        )
        frame += 1200.0 * torch.exp(
            -((grid_y - star_y) ** 2 + (grid_x - star_x) ** 2) / (2.0 * sigma**2)
        )
        xi = x0 + vx * dt
        yi = y0 + vy * dt
        frame += 65.0 * torch.exp(
            -((grid_y - yi) ** 2 + (grid_x - xi) ** 2) / (2.0 * sigma**2)
        )
        out_path = data_dir / f"demo_sequence_{i:02d}.fits"
        save_fits_image(frame, str(out_path), wcs=wcs)
        created_paths.append(str(out_path))

    return created_paths


def render_tab_pipeline_grid_search(device: torch.device):
    """Tab 2: Full functional control for the GPU Velocity Grid Search & End-to-End Pipeline."""
    st.subheader("🚀 GPU Shift-and-Stack Pipeline & Velocity Grid Search")

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    fits_files = sorted(
        [
            str(p)
            for p in data_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".fits", ".fit", ".fts")
        ]
    )

    col_info, col_demo = st.columns([3, 1])
    with col_info:
        st.caption(
            f"Local `data/` directory: **{len(fits_files)} FITS file(s)** found | Active Compute Device: **{device}**"
        )
    with col_demo:
        if st.button("✨ Generate Synthetic Test Sequence in `data/`", width="stretch"):
            try:
                with torch.no_grad():
                    _generate_demo_fits_in_data_dir(data_dir, device=device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                st.rerun()
            except Exception as exc:
                st.error(f"Error generating the test sequence: {exc}")

    c1, c2, c3 = st.columns(3)
    with c1:
        vx_min = st.number_input("vx Search Range Min [px/h]", value=-2.0, step=0.5)
        vx_max = st.number_input("vx Search Range Max [px/h]", value=2.0, step=0.5)
    with c2:
        vy_min = st.number_input("vy Search Range Min [px/h]", value=-2.0, step=0.5)
        vy_max = st.number_input("vy Search Range Max [px/h]", value=2.0, step=0.5)
    with c3:
        grid_steps = st.slider("Grid Steps (grid_steps)", min_value=3, max_value=25, value=5)
        cnn_threshold = st.slider(
            "RealBogusCNN Score Threshold (cnn_threshold)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )

    detection_threshold = st.slider(
        "DAOStarFinder Detection Threshold [ADU above background]",
        min_value=5.0,
        max_value=200.0,
        value=20.0,
        step=5.0,
    )

    run_clicked = st.button("🚀 Run GPU Grid Search & Pipeline", width="stretch")

    if run_clicked:
        if len(fits_files) == 0:
            st.warning(
                "No FITS files found in the `data/` directory. Please place FITS files in `data/` or click on 'Generate Synthetic Test Sequence in data/' above."
            )
            return

        if vx_min > vx_max or vy_min > vy_max:
            st.error("Invalid search range: Min value must not be greater than Max value.")
            return

        try:
            with st.spinner(
                f"Running GPU Velocity Grid Search ({grid_steps}×{grid_steps} trajectories) on {len(fits_files)} FITS frames..."
            ):
                with torch.no_grad():
                    mjd_list = []
                    for fpath in fits_files:
                        _, _, hdr = load_fits_image(fpath, device="cpu")
                        if hdr is not None and "MJD-OBS" in hdr:
                            mjd_list.append(float(hdr["MJD-OBS"]) * 24.0)

                    if len(mjd_list) == len(fits_files):
                        times_hours = torch.tensor(
                            mjd_list, dtype=torch.float32, device=device
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
                        cnn_threshold=float(cnn_threshold),
                    )

            st.success(
                f"Pipeline completed: **{len(catalog)} candidate(s)** found after star masking, shift-and-stack, and RealBogusCNN filtering."
            )

            if len(catalog) > 0:
                catalog.sort("peak", reverse=True)
                df_cat = catalog.to_pandas()
                st.dataframe(df_cat, width="stretch", height=360)
            else:
                st.info(
                    "No candidates found above the selected detection and CNN thresholds."
                )
        except Exception as exc:
            st.error(f"Error during pipeline execution: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    st.set_page_config(
        page_title="AstroProject | GPU Asteroid Pipeline",
        page_icon="☄️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)

    device = render_sidebar_controls()

    st.title("☄️ AstroProject — GPU Shift-and-Stack & Photometry Control Center")
    render_header_statusbar()
    st.divider()

    tab_inspector, tab_pipeline = st.tabs(
        ["FITS Inspector & Photometrie", "Pipeline & Grid Search"]
    )

    with tab_inspector:
        render_tab_fits_inspector(device=device)

    with tab_pipeline:
        render_tab_pipeline_grid_search(device=device)


if __name__ == "__main__":
    main()
