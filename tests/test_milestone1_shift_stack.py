import math
import pytest
import torch

from astro_project.core.synthetic import generate_synthetic_sky
from astro_project.pipeline.shift_and_stack import shift_and_stack


@pytest.fixture(autouse=True)
def vram_cleanup():
    """Ensure VRAM is cleared after every test according to Rule 4."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device(request):
    return torch.device(request.config.getoption("--device"))


def test_synthetic_sky_shape(device):
    """Test 1: Verify the generator outputs a tensor of shape [N, Y, X]."""
    n_frames, height, width = 10, 64, 80
    frames = generate_synthetic_sky(
        n_frames=n_frames,
        height=height,
        width=width,
        x0=20.0,
        y0=15.0,
        vx=0.5,
        vy=0.5,
        flux=1000.0,
        sigma=2.0,
        noise_std=0.0,
        device=device,
    )
    assert frames.shape == (n_frames, height, width)
    assert frames.dtype == torch.float32
    assert frames.device.type == device.type


def test_subpixel_alignment_and_flux_conservation(device):
    """Test 2: Verify subpixel alignment of peak and total flux conservation (< 2% error)."""
    n_frames, height, width = 16, 64, 64
    x0, y0 = 20.0, 25.0
    vx, vy = 0.73, -0.41  # Fractional subpixel velocity
    flux = 5000.0
    sigma = 2.5  # Well-sampled PSF to minimize discretization effects

    frames = generate_synthetic_sky(
        n_frames=n_frames,
        height=height,
        width=width,
        x0=x0,
        y0=y0,
        vx=vx,
        vy=vy,
        flux=flux,
        sigma=sigma,
        noise_std=0.0,
        device=device,
    )

    stacked = shift_and_stack(frames, vx=vx, vy=vy)
    assert stacked.shape == (height, width)

    # Check peak alignment at (y0, x0)
    max_idx = torch.argmax(stacked)
    max_y = (max_idx // width).item()
    max_x = (max_idx % width).item()
    assert max_y == round(y0)
    assert max_x == round(x0)

    # Check subpixel centroid around the peak
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    total_stacked_flux = stacked.sum().item()
    centroid_x = (stacked * x_grid).sum().item() / total_stacked_flux
    centroid_y = (stacked * y_grid).sum().item() / total_stacked_flux

    assert abs(centroid_x - x0) < 0.05
    assert abs(centroid_y - y0) < 0.05

    # Check flux conservation (< 2% deviation)
    rel_flux_error = abs(total_stacked_flux - flux) / flux
    assert rel_flux_error < 0.02


def test_snr_sqrt_n_scaling(device):
    """Test 3: Verify SNR increases according to the sqrt(N) law in noisy sequences."""
    torch.manual_seed(42)
    n_frames, height, width = 36, 128, 128
    x0, y0 = 40.0, 40.0
    vx, vy = 0.6, 0.4
    sigma = 2.0
    noise_std = 10.0

    # Set flux so single-frame peak is ~5 sigma above noise
    peak_signal = 50.0
    flux = peak_signal * (2.0 * math.pi * sigma**2)

    frames = generate_synthetic_sky(
        n_frames=n_frames,
        height=height,
        width=width,
        x0=x0,
        y0=y0,
        vx=vx,
        vy=vy,
        flux=flux,
        sigma=sigma,
        noise_std=noise_std,
        device=device,
    )

    # Estimate single-frame SNR from background region (top-left 25x25 corner)
    single_bg_std = frames[0, :25, :25].std().item()
    single_peak = frames[0, round(y0), round(x0)].item()
    snr_single = single_peak / single_bg_std

    stacked = shift_and_stack(frames, vx=vx, vy=vy)

    # Estimate stacked SNR from background region
    stacked_bg_std = stacked[:25, :25].std().item()
    stacked_peak = stacked[round(y0), round(x0)].item()
    snr_stacked = stacked_peak / stacked_bg_std

    expected_improvement = math.sqrt(n_frames)
    measured_improvement = snr_stacked / snr_single

    # Bilinear interpolation slightly smooths uncorrelated pixel noise (increasing SNR slightly more),
    # Verify that the SNR scales closely with sqrt(N)
    assert measured_improvement >= 0.85 * expected_improvement
    assert stacked_bg_std < single_bg_std / (0.85 * expected_improvement)
