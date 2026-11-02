import math
import pytest
import torch

from astro_project.core.classifier import RealBogusCNN, extract_multichannel_stamps
from astro_project.pipeline.velocity_search import (
    compute_psi_phi_frames,
    trimmed_likelihood_stack,
)


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_psi_phi_matched_filter_maximizes_faint_source_snr(device):
    """
    Test 1 (Psi/Phi Matched Filter): Verify that pre-convolving synthetic frames
    with a Gaussian PSF into Psi and Phi maps is mathematically consistent and
    maximizes the SNR of a faint sub-sigma point source across frames.
    """
    torch.manual_seed(42)
    n_frames = 9
    height, width = 96, 96
    times_hours = torch.linspace(0.0, 4.0, steps=n_frames, dtype=torch.float32, device=device)

    fwhm = 3.0
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    noise_std = 10.0

    # Faint source (~2.2 sigma per frame peak SNR before stacking)
    x0, y0 = 32.0, 40.0
    vx, vy = 3.0, -2.0
    peak_amp = 2.2 * noise_std

    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    frames = []
    for i in range(n_frames):
        dt = float((times_hours[i] - times_hours[0]).item())
        xi = x0 + vx * dt
        yi = y0 + vy * dt
        psf_signal = peak_amp * torch.exp(
            -((grid_y - yi) ** 2 + (grid_x - xi) ** 2) / (2.0 * sigma**2)
        )
        noise = torch.randn((height, width), dtype=torch.float32, device=device) * noise_std
        frames.append(psf_signal + noise)

    frames_tensor = torch.stack(frames, dim=0)  # [N, Y, X]

    # Single-frame raw SNR at (y0, x0) in frame 0
    raw_bg_std = float(torch.std(frames_tensor[0, :20, :20]).item())
    single_frame_snr = float(frames_tensor[0, int(y0), int(x0)].item()) / raw_bg_std

    # Compute Psi and Phi maps on GPU
    psi, phi = compute_psi_phi_frames(frames_tensor, fwhm=fwhm)

    assert psi.shape == frames_tensor.shape
    assert phi.shape == frames_tensor.shape
    assert psi.device == frames_tensor.device
    assert phi.device == frames_tensor.device
    assert torch.all(phi > 0.0)

    # Single-frame matched-filter likelihood map: L_0 = Psi_0 / sqrt(Phi_0)
    likelihood_single = psi[0] / torch.sqrt(phi[0])
    matched_single_snr = float(likelihood_single[int(y0), int(x0)].item())
    assert matched_single_snr > single_frame_snr

    # Multi-frame Trimmed-Likelihood stack along true trajectory
    snr_map = trimmed_likelihood_stack(
        psi=psi,
        phi=phi,
        times_hours=times_hours,
        vx=vx,
        vy=vy,
        trim_fraction=0.2,
    )

    assert snr_map.shape == (height, width)
    stacked_snr = float(snr_map[int(y0), int(x0)].item())

    # Stacked likelihood SNR must significantly exceed single-frame SNR and pass > 5.0 sigma
    assert stacked_snr > matched_single_snr * 1.5
    assert stacked_snr > 5.0


def test_3channel_real_bogus_cnn_forward_pass(device):
    """
    Test 2 (3-Channel RealBogusCNN): Verify that a batch of [N, 3, 32, 32] stamps
    (Channel 0: Mean-Stack, Channel 1: Median-Stack, Channel 2: Variance-Map)
    is processed by RealBogusCNN into [N, 1] probabilities in [0.0, 1.0].
    """
    torch.manual_seed(42)
    n_frames = 5
    height, width = 80, 80
    frames = torch.randn((n_frames, height, width), dtype=torch.float32, device=device)

    positions = [(40.0, 40.0), (10.0, 12.0), (75.0, 70.0), (25.0, 60.0)]
    stamps_3ch = extract_multichannel_stamps(frames, positions, stamp_size=32)

    assert stamps_3ch.shape == (4, 3, 32, 32)
    assert stamps_3ch.device.type == device.type
    # Channel 2 is the variance map and must be non-negative everywhere
    assert torch.all(stamps_3ch[:, 2, :, :] >= 0.0)

    model = RealBogusCNN(in_channels=3).to(device)
    model.eval()

    with torch.no_grad():
        probs = model(stamps_3ch)

    assert probs.shape == (4, 1)
    assert probs.device.type == device.type
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)
