import math
import pytest
import torch

from astro_project.pipeline.kinematics import (
    mask_stationary_stars,
    kinematic_shift_and_stack,
)


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_mask_stationary_stars(device):
    torch.manual_seed(42)
    height, width = 100, 120
    bg_level = 100.0
    noise_std = 2.0

    # [Y, X] Tensor mit Hintergrundrauschen erzeugen
    image = (
        torch.full((height, width), bg_level, dtype=torch.float32, device=device)
        + torch.randn((height, width), dtype=torch.float32, device=device) * noise_std
    )

    # Hellen, statischen Feldstern bei (y=50, x=60) einsetzen
    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    star_y, star_x = 50.0, 60.0
    sigma = 1.5
    star_flux = 1500.0
    image += star_flux * torch.exp(
        -((grid_y - star_y) ** 2 + (grid_x - star_x) ** 2) / (2.0 * sigma**2)
    )

    assert image.max().item() > 1000.0

    # Stern maskieren (Schwellenwert über Hintergrund)
    cleaned = mask_stationary_stars(image, threshold=50.0)

    assert cleaned.shape == image.shape
    assert cleaned.device == image.device
    # Der helle Stern muss entfernt und durch den Median-Hintergrund (~100) ersetzt sein
    star_patch = cleaned[47:54, 57:64]
    assert torch.max(star_patch).item() == pytest.approx(bg_level, abs=5.0 * noise_std)
    assert torch.median(cleaned).item() == pytest.approx(bg_level, abs=1.0)


def test_kinematic_shift_and_stack(device):
    torch.manual_seed(123)
    n_frames, height, width = 3, 100, 120
    times_hours = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32, device=device)

    vx = 2.0   # Pixel / Stunde in X
    vy = -1.0  # Pixel / Stunde in Y
    x0, y0 = 50.0, 60.0

    noise_std = 5.0
    frames = torch.randn((n_frames, height, width), dtype=torch.float32, device=device) * noise_std

    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    sigma = 1.5
    asteroid_amp = 80.0

    for i in range(n_frames):
        dt = (times_hours[i] - times_hours[0]).item()
        xi = x0 + vx * dt
        yi = y0 + vy * dt
        asteroid = asteroid_amp * torch.exp(
            -((grid_y - yi) ** 2 + (grid_x - xi) ** 2) / (2.0 * sigma**2)
        )
        frames[i] += asteroid

    # Kinematisches Shift-and-Stack (Standard: Median sowie Summen-Überlagerung prüfen)
    stacked = kinematic_shift_and_stack(frames, times_hours, vx=vx, vy=vy)
    stacked_sum = kinematic_shift_and_stack(frames, times_hours, vx=vx, vy=vy, method="sum")
    unshifted_median = torch.median(frames, dim=0).values

    assert stacked.shape == (height, width)
    assert stacked.device == device

    # SNR in Einzelbild vs. gestacktem Bild berechnen (Hintergrundbereich abseits des Asteroiden)
    single_bg_std = frames[0, :30, :30].std().item()
    stacked_bg_std = stacked[:30, :30].std().item()

    single_snr = frames[0].max().item() / single_bg_std
    stacked_snr = stacked.max().item() / stacked_bg_std

    # 1. SNR im kinematisch gestackten Bild ist signifikant höher als im Einzelbild
    assert stacked_snr > single_snr * 1.3
    # 2. Maximalwert im kinematischen Median-Stack ist signifikant höher als im unverschobenen Stack
    assert stacked.max().item() > unshifted_median.max().item() * 2.0
    # 3. Bei konstruktiver Überlagerung (Summe) ist der Maximalwert signifikant höher als in jedem Einzelbild
    assert stacked_sum.max().item() > frames.max().item() * 2.0
    # 4. Das Maximum liegt exakt an der Startposition (y0, x0) = (60, 50)
    max_idx = torch.argmax(stacked)
    max_y = (max_idx // width).item()
    max_x = (max_idx % width).item()
    assert abs(max_y - y0) <= 1
    assert abs(max_x - x0) <= 1
