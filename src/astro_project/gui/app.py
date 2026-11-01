import math
import pytest
import torch
import numpy as np
from astro_project.core.photometry import estimate_background, detect_sources

@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_background_estimation(device):
    torch.manual_seed(42)
    height, width = 200, 250
    image = torch.full((height, width), 100.0, device=device, dtype=torch.float32) + torch.randn((height, width), device=device, dtype=torch.float32) * 2.0
    bkg_tensor = estimate_background(image)
    assert bkg_tensor.shape == image.shape
    assert bkg_tensor.device == image.device
    assert bkg_tensor.dtype == torch.float32
    assert torch.median(bkg_tensor).item() == pytest.approx(100.0, abs=0.5)
    assert torch.median(image - bkg_tensor).item() == pytest.approx(0.0, abs=0.5)

def test_source_detection_subpixel_accuracy(device):
    height, width = 120, 150
    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    true_stars = [
        (25.3, 40.7, 1000.0),
        (60.8, 85.2, 1500.0),
        (95.1, 120.4, 800.0),
    ]
    fwhm = 3.0
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    image = torch.zeros((height, width), dtype=torch.float32, device=device)
    for y0, x0, amp in true_stars:
        gauss = amp * torch.exp(-((grid_y - y0) ** 2 + (grid_x - x0) ** 2) / (2.0 * sigma ** 2))
        image += gauss
    sources = detect_sources(image, threshold=50.0, fwhm=fwhm)
    assert sources is not None
    assert len(sources) == 3
    sources.sort("y_centroid")
    true_stars.sort()
    for i, (y0, x0, _) in enumerate(true_stars):
        assert sources["y_centroid"][i] == pytest.approx(y0, abs=0.05)
        assert sources["x_centroid"][i] == pytest.approx(x0, abs=0.05)
