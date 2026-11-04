import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import streamlit as st
import torch
import math

from astro_project.core.hardware import resolve_device


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
        * **Nachbarfeld-Tracking:** Berechnet Zeitpunkt und Himmelskoordinate, an denen das Objekt den aktuellen Sensorbereich $[0, W] \\times [H]$ verl\u00e4sst.
        * **Stellarium Live-Slew:** Zentriert die virtuelle Kamera in Stellarium \u00fcber das RemoteControl-Plugin (`http://localhost:8090/api`) direkt auf die Kandidaten-Koordinate.
        """
    )


@st.dialog("System & Einstellungen \u2699\ufe0f")
def show_system_settings_dialog():
    """Modal window for Compute & VRAM Control, Plot Scaling, and Orbit Plot settings."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.init()

    st.markdown("#### Compute & VRAM Control")
    options = ["cuda", "cpu"] if cuda_available else ["cpu"]
    current_dev = st.session_state.get("selected_dev", options[0])
    if current_dev not in options:
        current_dev = options[0]

    selected_dev = st.selectbox(
        "Ziel-Device",
        options=options,
        index=options.index(current_dev),
        help="W\u00e4hlt das PyTorch-Device f\u00fcr [Y, X] Tensor-Operationen (CUDA Standard auf RTX 4070).",
    )
    st.session_state["selected_dev"] = selected_dev

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("\U0001f504 Refresh", use_container_width=True):
            st.rerun()
    with col_btn2:
        if st.button("\U0001f9f9 VRAM Flush", use_container_width=True):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            st.rerun()

    st.markdown("---")
    plot_scale_pct = st.slider(
        "\U0001f50d Plot-Gr\u00f6\u00dfe [%]",
        min_value=50,
        max_value=100,
        value=int(st.session_state.get("plot_scale_pct", 85)),
        step=5,
        help="Skaliert die Darstellungsgr\u00f6\u00dfe (figsize) aller Matplotlib-Plots dynamisch f\u00fcr optimales Rein-/Rauszoomen.",
    )
    st.session_state["plot_scale_pct"] = float(plot_scale_pct)

    show_all_planets = st.checkbox(
        "Alle Hauptplaneten anzeigen",
        value=bool(st.session_state.get("show_all_planets", False)),
        help="Zeichnet im Sonnensystem-Plot die Umlaufbahnen und ungef\u00e4hren Positionen aller Hauptplaneten (Merkur bis Neptun) ein.",
    )
    st.session_state["show_all_planets"] = bool(show_all_planets)


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


def render_sidebar_statusbar(device: torch.device):
    """Renders the hardware statusbar (CPU, RAM, VRAM, Device) at the bottom of the sidebar in a compact 2x2 grid."""
    metrics = get_hardware_metrics()
    with st.sidebar:
        st.markdown(
            "<div class='sidebar-statusbar-wrap'></div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Hardware Status")
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            st.markdown(
                f"<div class='sidebar-hw-card'>"
                f"<span class='sidebar-hw-label'>\u26a1 CPU</span>"
                f"<b>{metrics['cpu_percent']:.1f}%</b>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with r1_c2:
            st.markdown(
                f"<div class='sidebar-hw-card'>"
                f"<span class='sidebar-hw-label'>\U0001f9e0 RAM</span>"
                f"<b>{metrics['ram_percent']:.0f}%</b> ({metrics['ram_used_gb']:.1f}/{metrics['ram_total_gb']:.0f}G)"
                f"</div>",
                unsafe_allow_html=True,
            )

        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            st.markdown(
                f"<div class='sidebar-hw-card'>"
                f"<span class='sidebar-hw-label'>\U0001f3ae VRAM</span>"
                f"<b>{metrics['vram_used_gb']:.2f}</b> / {metrics['vram_total_gb']:.0f} GB"
                f"</div>",
                unsafe_allow_html=True,
            )
        with r2_c2:
            st.markdown(
                f"<div class='sidebar-hw-card'>"
                f"<span class='sidebar-hw-label'>\U0001f3af Device</span>"
                f"<code class='status-badge'>{device.type.upper()}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_sidebar_controls() -> tuple[torch.device, str]:
    """Renders main navigation buttons, the 'System & Einstellungen' modal button, and the 2x2 statusbar in the sidebar."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.init()

    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = "FITS-Inspector & Photometry"
    if "selected_dev" not in st.session_state:
        st.session_state["selected_dev"] = "cuda" if cuda_available else "cpu"
    if "plot_scale_pct" not in st.session_state:
        st.session_state["plot_scale_pct"] = 85.0
    if "show_all_planets" not in st.session_state:
        st.session_state["show_all_planets"] = False

    with st.sidebar:
        st.markdown(
            "<div class='sidebar-title'>\U0001f52d AstroProject</div>",
            unsafe_allow_html=True,
        )
        st.markdown("### Navigation")
        nav_items = [
            "FITS-Inspector & Photometry",
            "Velocity Grid Search & Orbit Analysis",
            "Automated Survey",
            "Candidate Inspector",
        ]
        for item in nav_items:
            is_active = st.session_state["nav_selection"] == item
            if st.button(
                item,
                key=f"nav_btn_{item}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state["nav_selection"] = item
                st.rerun()

        st.markdown("---")
        if st.button(
            "System & Einstellungen \u2699\ufe0f",
            key="open_settings_modal_btn",
            use_container_width=True,
        ):
            show_system_settings_dialog()

    selected_dev = st.session_state.get("selected_dev", "cuda" if cuda_available else "cpu")
    device = resolve_device(
        selected_dev if selected_dev == "cuda" and cuda_available else "cpu"
    )
    render_sidebar_statusbar(device)
    return device, st.session_state["nav_selection"]


def _plot_comparison_frame(
    image_np: np.ndarray,
    title: str,
    target_xy: tuple[float, float] | None = None,
    scale_pct: float = 85.0,
    target_label: str = "Asteroid",
) -> plt.Figure:
    """Creates a Dark-Mode plot for the before/after comparison (single frame vs. shift-and-stack)."""
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
    """Renders the 32x32 stamps for frames 0..N-1 alongside the shift-and-stack stamp."""
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


MAJOR_PLANETS = [
    {"name": "Mercury", "a_au": 0.39, "lon_deg": 252.0, "color": "#9ca3af"},
    {"name": "Venus", "a_au": 0.72, "lon_deg": 181.0, "color": "#fde047"},
    {"name": "Earth", "a_au": 1.00, "lon_deg": 100.0, "color": "#38bdf8"},
    {"name": "Mars", "a_au": 1.52, "lon_deg": 355.0, "color": "#fb7185"},
    {"name": "Jupiter", "a_au": 5.20, "lon_deg": 34.0, "color": "#fb923c"},
    {"name": "Saturn", "a_au": 9.58, "lon_deg": 50.0, "color": "#fcd34d"},
    {"name": "Uranus", "a_au": 19.20, "lon_deg": 314.0, "color": "#22d3ee"},
    {"name": "Neptune", "a_au": 30.05, "lon_deg": 304.0, "color": "#818cf8"},
]


def _plot_solar_system_orbit(
    a_nom: float,
    e_nom: float,
    a_samples: list[float] | np.ndarray,
    e_samples: list[float] | np.ndarray,
    scale_pct: float = 85.0,
    show_all_planets: bool = False,
) -> plt.Figure:
    """Creates a top-down ecliptic Dark-Mode plot of the nominal orbit, planets, and MC uncertainty envelope."""
    scale = max(float(scale_pct) / 100.0, 0.4)
    fig_w = 7.4 * scale if show_all_planets else 6.0 * scale
    fig, ax = plt.subplots(figsize=(fig_w, 5.5 * scale), facecolor="#0b0f19")
    ax.set_facecolor("#0b0f19")

    theta_circle = np.linspace(0.0, 2.0 * np.pi, 360)

    def _orbit_xy(a_val: float, e_val: float) -> tuple[np.ndarray, np.ndarray]:
        if e_val >= 1.0 or a_val < 0.0:
            e_hyp = max(float(e_val), 1.05)
            a_abs = max(abs(float(a_val)), 0.2)
            max_th = math.acos(-1.0 / e_hyp) * 0.82
            th = np.linspace(-max_th, max_th, 240)
            r = (a_abs * (e_hyp**2 - 1.0)) / (1.0 + e_hyp * np.cos(th))
            return r * np.cos(th), r * np.sin(th)
        a_ell = max(float(a_val), 0.1)
        e_ell = float(np.clip(e_val, 0.0, 0.95))
        r = (a_ell * (1.0 - e_ell**2)) / (1.0 + e_ell * np.cos(theta_circle))
        return r * np.cos(theta_circle), r * np.sin(theta_circle)

    # Monte Carlo uncertainty envelope (alpha=0.1)
    for a_mc, e_mc in zip(a_samples, e_samples):
        if float(a_mc) == 0.0:
            continue
        x_mc, y_mc = _orbit_xy(float(a_mc), float(e_mc))
        ax.plot(
            x_mc,
            y_mc,
            color="#38bdf8",
            alpha=0.1,
            linewidth=1.0,
        )

    a_nom_val = float(a_nom) if float(a_nom) != 0.0 else 1.5
    e_nom_val = max(float(e_nom), 0.0)
    ref_dist_au = abs(a_nom_val) * (e_nom_val - 1.0) if (e_nom_val >= 1.0 or a_nom_val < 0.0) else max(a_nom_val, 0.1)

    # Select planets to display: all major planets or closest inner & outer planet
    if show_all_planets:
        selected_planets = MAJOR_PLANETS
    else:
        inner_planets = [p for p in MAJOR_PLANETS if p["a_au"] <= ref_dist_au]
        outer_planets = [p for p in MAJOR_PLANETS if p["a_au"] > ref_dist_au]
        selected_planets = []
        if inner_planets:
            selected_planets.append(max(inner_planets, key=lambda p: p["a_au"]))
        if outer_planets:
            selected_planets.append(min(outer_planets, key=lambda p: p["a_au"]))
        if not selected_planets:
            selected_planets = [MAJOR_PLANETS[2]]

    for planet in selected_planets:
        r_p = float(planet["a_au"])
        lon_rad = np.deg2rad(float(planet["lon_deg"]))
        px = r_p * np.cos(lon_rad)
        py = r_p * np.sin(lon_rad)
        ax.plot(
            r_p * np.cos(theta_circle),
            r_p * np.sin(theta_circle),
            color=planet["color"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            label=f"{planet['name']} ({r_p:.1f} AU)",
        )
        ax.scatter(
            [px],
            [py],
            color=planet["color"],
            s=38,
            zorder=4,
            edgecolors="#0b0f19",
            linewidths=0.8,
        )

    # Nominal candidate orbit
    x_nom, y_nom = _orbit_xy(a_nom_val, e_nom_val)
    ax.plot(
        x_nom,
        y_nom,
        color="#ef4444",
        linewidth=2.2,
        label=f"Nominal (a={a_nom_val:.2f} AU, e={e_nom_val:.2f})",
    )

    # Sun at (0, 0)
    ax.scatter(
        [0.0],
        [0.0],
        color="#facc15",
        s=110,
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

    if show_all_planets:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            facecolor="#1e293b",
            edgecolor="#334155",
            labelcolor="#f8fafc",
            fontsize=7.5,
            framealpha=0.92,
            handlelength=1.4,
            labelspacing=0.35,
        )
    else:
        ax.legend(
            loc="upper right",
            facecolor="#1e293b",
            edgecolor="#334155",
            labelcolor="#f8fafc",
            fontsize=8,
        )
    fig.tight_layout()
    return fig
