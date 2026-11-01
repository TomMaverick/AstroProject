import math
import os
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from astro_project.core.classifier import RealBogusCNN
from astro_project.core.fits_loader import save_fits_image
from astro_project.core.wcs_utils import create_synthetic_wcs, pix_to_world
from astro_project.pipeline.orchestration import run_pipeline


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_synthetic_cnn(device: torch.device) -> RealBogusCNN:
    """Trainiert ein RealBogusCNN kurz auf synthetischen Punktquellen vs. Rauschen (ohne Mocks)."""
    torch.manual_seed(42)
    model = RealBogusCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()

    y = torch.arange(32, dtype=torch.float32, device=device)
    x = torch.arange(32, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    psf = 50.0 * torch.exp(-((gy - 16.0) ** 2 + (gx - 16.0) ** 2) / (2.0 * 1.5**2))

    model.train()
    for _ in range(25):
        pos = psf.unsqueeze(0).unsqueeze(0) + torch.randn((8, 1, 32, 32), device=device) * 1.5
        neg = torch.randn((8, 1, 32, 32), device=device) * 1.5
        batch_x = torch.cat([pos, neg], dim=0)
        batch_y = torch.cat(
            [torch.ones((8, 1), device=device), torch.zeros((8, 1), device=device)], dim=0
        )

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def test_end_to_end_pipeline_with_synthetic_fits(tmp_path, device):
    torch.manual_seed(42)
    n_frames = 3
    height, width = 120, 140
    times_hours = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32, device=device)

    wcs = create_synthetic_wcs(pixel_scale_arcsec=1.0, crval_ra=180.0, crval_dec=45.0)

    # Bekannte Ground-Truth-Parameter für den Asteroiden
    x0, y0 = 80.0, 70.0
    true_vx = 2.0
    true_vy = -1.0
    asteroid_amp = 50.0

    # Heller statischer Feldstern (wird durch Sternmaskierung entfernt)
    star_x, star_y = 35.0, 30.0
    star_amp = 1500.0

    true_ra, true_dec = pix_to_world(wcs, x0, y0)
    true_coord = SkyCoord(ra=float(true_ra) * u.deg, dec=float(true_dec) * u.deg, frame="icrs")

    star_ra, star_dec = pix_to_world(wcs, star_x, star_y)
    star_coord = SkyCoord(ra=float(star_ra) * u.deg, dec=float(star_dec) * u.deg, frame="icrs")

    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    fwhm = 3.0
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    file_paths = []
    for i in range(n_frames):
        dt = float((times_hours[i] - times_hours[0]).item())
        frame = (
            torch.full((height, width), 100.0, dtype=torch.float32, device=device)
            + torch.randn((height, width), dtype=torch.float32, device=device) * 1.5
        )

        # Statischer Feldstern
        frame += star_amp * torch.exp(
            -((grid_y - star_y) ** 2 + (grid_x - star_x) ** 2) / (2.0 * sigma**2)
        )

        # Sich bewegender Asteroid
        xi = x0 + true_vx * dt
        yi = y0 + true_vy * dt
        frame += asteroid_amp * torch.exp(
            -((grid_y - yi) ** 2 + (grid_x - xi) ** 2) / (2.0 * sigma**2)
        )

        fits_path = str(tmp_path / f"synthetic_frame_{i}.fits")
        save_fits_image(frame, fits_path, wcs=wcs)
        file_paths.append(fits_path)

    cnn_model = _train_synthetic_cnn(device)

    try:
        catalog = run_pipeline(
            file_paths=file_paths,
            times_hours=times_hours,
            vx_range=(-2.0, 2.0),
            vy_range=(-2.0, 2.0),
            steps=5,
            detection_threshold=20.0,
            cnn_model=cnn_model,
            device=device,
        )

        # 1. Struktur der Astropy Table verifizieren
        assert isinstance(catalog, Table)
        for col in ("ra", "dec", "vx", "vy", "peak", "cnn_score"):
            assert col in catalog.colnames

        assert len(catalog) >= 1

        # 2. Besten Kandidaten prüfen: muss dem Asteroiden entsprechen
        catalog.sort("peak", reverse=True)
        best = catalog[0]
        assert best["vx"] == pytest.approx(true_vx, abs=1e-4)
        assert best["vy"] == pytest.approx(true_vy, abs=1e-4)
        assert best["cnn_score"] >= 0.5

        det_coord = SkyCoord(ra=best["ra"] * u.deg, dec=best["dec"] * u.deg, frame="icrs")
        assert true_coord.separation(det_coord).arcsecond < 0.5

        # 3. Verifizieren, dass der statische Feldstern herausgefiltert wurde
        for row in catalog:
            row_coord = SkyCoord(ra=row["ra"] * u.deg, dec=row["dec"] * u.deg, frame="icrs")
            assert star_coord.separation(row_coord).arcsecond > 5.0
    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
