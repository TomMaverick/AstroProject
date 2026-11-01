import pytest
import torch

from astro_project.pipeline.velocity_search import run_velocity_search


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_velocity_grid_search_finds_true_asteroid(device):
    torch.manual_seed(42)
    n_frames, height, width = 4, 100, 120
    times_hours = torch.tensor([0.0, 2.0, 4.0, 6.0], dtype=torch.float32, device=device)

    true_vx = 1.0
    true_vy = -1.0
    x0, y0 = 60.0, 50.0

    noise_std = 2.0
    frames = (
        torch.randn((n_frames, height, width), dtype=torch.float32, device=device)
        * noise_std
    )

    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    sigma = 1.5
    asteroid_amp = 40.0

    for i in range(n_frames):
        dt = (times_hours[i] - times_hours[0]).item()
        xi = x0 + true_vx * dt
        yi = y0 + true_vy * dt
        asteroid = asteroid_amp * torch.exp(
            -((grid_y - yi) ** 2 + (grid_x - xi) ** 2) / (2.0 * sigma**2)
        )
        frames[i] += asteroid

    # Suchgitter deckt [-2.0, -1.0, 0.0, 1.0, 2.0] ab -> enthält (1.0, -1.0) exakt
    candidates = run_velocity_search(
        frames=frames,
        times_hours=times_hours,
        vx_range=(-2.0, 2.0),
        vy_range=(-2.0, 2.0),
        steps=5,
        detection_threshold=20.0,
    )

    assert len(candidates) > 0

    # Prüfen, dass die Ground-Truth-Geschwindigkeit (vx=1.0, vy=-1.0) und Startposition (x0, y0)
    # unter den Detektionen ist und den stärksten Treffer liefert
    matching = [
        c
        for c in candidates
        if c["vx"] == pytest.approx(true_vx, abs=1e-4)
        and c["vy"] == pytest.approx(true_vy, abs=1e-4)
    ]
    assert len(matching) >= 1

    best_match = matching[0]
    assert best_match["x_centroid"] == pytest.approx(x0, abs=0.5)
    assert best_match["y_centroid"] == pytest.approx(y0, abs=0.5)

    # Der beste Kandidat insgesamt (nach peak/flux) muss die wahre Geschwindigkeit haben
    best_overall = max(candidates, key=lambda c: c.get("peak", c.get("flux", 0.0)))
    assert best_overall["vx"] == pytest.approx(true_vx, abs=1e-4)
    assert best_overall["vy"] == pytest.approx(true_vy, abs=1e-4)
