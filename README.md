# ☄️ AstroProject — GPU-Accelerated Asteroid Detection Pipeline

A scientifically verified **Shift-and-Stack Pipeline** for detecting faint, moving celestial bodies (asteroids, near-Earth objects, KBOs) in astronomical FITS time series. Developed for GPU acceleration with **PyTorch (CUDA)** on an **NVIDIA RTX 4070 (12 GB VRAM)** under strict adherence to astrophysical flux conservation and astrometry standards.

---

## 🔭 Project Overview

Faint asteroids often disappear in single exposures due to thermal and electronic background noise ($\text{SNR} < 3$). Since they move relative to the background stars, a classical long-exposure blurs the signal into a streak.

**AstroProject** solves this problem through **Kinematic Shift-and-Stack (Synthetic Tracking)** on the GPU:
1. **2D Background Modeling & Star Masking:** Removal of the sky background (`Background2D`) and masking of bright, stationary stars (`DAOStarFinder`).
2. **Subpixel Trajectory Shifting:** Shifting of image sequences along hypothetical velocity vectors $(v_x, v_y)$ using `torch.nn.functional.affine_grid` and `grid_sample(..., align_corners=False)` under strict **Flux Conservation**.
3. **Co-Addition & SNR Scaling:** Constructive stacking along the time axis amplifies the asteroid signal proportional to $\sqrt{N}$ above the noise.
4. **AI Candidate Filtering (`RealBogusCNN`):** A Convolutional Neural Network evaluates $32 \times 32$ pixel stamps and distinguishes true point sources from artifacts.
5. **WCS Pixel-to-Sky Transformation:** Conversion of subpixel accurate centroids (`x_centroid`, `y_centroid`) with `origin=0` into international celestial coordinates (`ICRS` RA/Dec).

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.12+ | Core development & typing |
| **GPU Compute & Deep Learning** | PyTorch (`cuda`) | Subpixel resampling (`grid_sample`), stacking, `RealBogusCNN` |
| **Target Hardware Platform** | NVIDIA RTX 4070 (12 GB VRAM) | Strict VRAM management (`torch.no_grad()`, `empty_cache()`) |
| **FITS I/O & Astrometry** | Astropy (`io.fits`, `wcs`, `coordinates`, `table`) | FITS ingestion, TAN projection, `SkyCoord` matching |
| **Photometry & Detection** | Photutils 3.0+ (`Background2D`, `DAOStarFinder`) | 2D background estimation & subpixel source detection |
| **Archive Integration** | Astroquery (`ipac.irsa.most`) | Ephemeride and FITS cutout query for known asteroids |
| **Web Dashboard** | Streamlit & Psutil | Interactive Dark-Mode Control Center with live VRAM monitoring |
| **Quality Assurance** | Pytest | 100 % mock-free tests against synthetic & real ground-truth matrices |

---

## ✨ Core Features & Architecture Guardrails

- **Strict Axis Discipline:** Image tensors are consistently processed in `[Y, X]` (height, width) or `[N, Y, X]`; Astropy WCS uses `(X, Y)` with `origin=0` to `(RA, Dec)`.
- **Physical Flux Conservation:** All affine GPU transformations enforce `align_corners=False`.
- **12 GB VRAM Protection:** All grid search loops and inference paths run encapsulated under `with torch.no_grad():` and deterministically free GPU caches in teardown.
- **Zero-Mock Verification:** All mathematical and physical transformations are tested against analytical 2D Gaussian PSFs and real telescope data (Palomar / IRSA).

### Project Structure
```text
├── data/                                 # Local FITS data storage (not committed)
├── docs/                                 # Technical references (Astropy WCS, Photutils, PyTorch GPU)
├── src/astro_project/
│   ├── core/
│   │   ├── classifier.py                 # 32x32 stamp extraction & RealBogusCNN
│   │   ├── fits_loader.py                # NaN-safe FITS I/O to target device
│   │   ├── photometry.py                 # Background2D & DAOStarFinder integration
│   │   ├── synthetic.py                  # Synthetic sky & trajectory generation
│   │   └── wcs_utils.py                  # WCS transformations (origin=0) & SkyCoord export
│   ├── pipeline/
│   │   ├── data_fetcher.py               # Astroquery IRSA MOST asteroid fetcher
│   │   ├── kinematics.py                 # Star masking & kinematic shift-and-stack
│   │   ├── orchestration.py              # End-to-end pipeline & Astropy catalog export
│   │   ├── shift_and_stack.py            # Basic GPU shift-and-stack operators
│   │   └── velocity_search.py            # 2D velocity grid search on the GPU
│   └── gui/
│       └── app.py                        # Streamlit Dark-Mode Web GUI & hardware monitor
└── tests/                                # Milestone tests (M1 to M8 + real data ingestion)
```

---

## ⚙️ Installation & Setup

1. **Clone the repository and activate the virtual environment:**
   ```bash
   git clone <repository-url>
   cd astro_project
   python -m venv .venv
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies and package in editable mode:**
   ```bash
   pip install -e .
   pip install streamlit psutil astroquery matplotlib pandas
   ```

---

## 🚀 Launch Commands

### 1. Start the Streamlit Web GUI
Launches the interactive Dark-Mode Control Center with a horizontal live hardware status bar (CPU, RAM, RTX 4070 VRAM), FITS Inspector, and GPU Velocity Grid Search:
```bash
streamlit run src/astro_project/gui/app.py
```

### 2. Run the Automated Test Suite
Executes all synthetic ground-truth tests (Milestones 1–8) as well as the real data ingestion tests:
```bash
pytest -v
```
