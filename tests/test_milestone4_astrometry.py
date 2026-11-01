import math
import pytest
import torch
import astropy.units as u
from astropy.coordinates import SkyCoord

from astro_project.core.photometry import detect_sources
from astro_project.core.wcs_utils import (
    create_synthetic_wcs,
    pix_to_world,
    world_to_pix,
    sources_to_skycoords,
)


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_end_to_end_detection_to_radec(device):
    # 1. Synthetisches WCS-Objekt erzeugen (1.0 arcsec/pixel, crpix=[51, 51] -> origin=0: (50, 50))
    wcs = create_synthetic_wcs(pixel_scale_arcsec=1.0, crval_ra=180.0, crval_dec=45.0)

    # 2. Bekannte Himmelskoordinate definieren und in Pixelkoordinaten (X, Y) mit origin=0 umwandeln
    true_ra = 180.005
    true_dec = 45.004
    true_coord = SkyCoord(ra=true_ra * u.deg, dec=true_dec * u.deg, frame="icrs")

    x_true, y_true = world_to_pix(wcs, true_ra, true_dec)
    x_true = float(x_true)
    y_true = float(y_true)

    # 3. Leeren [Y, X] PyTorch-Tensor erzeugen und künstlichen Stern exakt an (y_true, x_true) platzieren
    height, width = 100, 120
    image = torch.zeros((height, width), dtype=torch.float32, device=device)

    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    fwhm = 3.0
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    image += 1000.0 * torch.exp(
        -((grid_y - y_true) ** 2 + (grid_x - x_true) ** 2) / (2.0 * sigma**2)
    )

    # 4. Stern im Tensor mit detect_sources finden
    sources = detect_sources(image, threshold=50.0, fwhm=fwhm)
    assert sources is not None
    assert len(sources) == 1

    # 5. Gefundene x_centroid und y_centroid über WCS zurück in RA und Dec konvertieren
    x_det = sources["x_centroid"][0]
    y_det = sources["y_centroid"][0]
    ra_det, dec_det = pix_to_world(wcs, x_det, y_det)

    detected_coord = SkyCoord(ra=ra_det * u.deg, dec=dec_det * u.deg, frame="icrs")

    # Ebenfalls Batch-Hilfsfunktion prüfen
    batch_coords = sources_to_skycoords(sources, wcs)
    assert len(batch_coords) == 1

    # 6. Mit SkyCoord.separation() verifizieren, dass der Abstand im Sub-Bogensekunden-Bereich liegt
    sep = true_coord.separation(detected_coord)
    assert sep.arcsecond < 0.1
    assert true_coord.separation(batch_coords[0]).arcsecond < 0.1
