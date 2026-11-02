import pytest
import torch

from astro_project.core.classifier import extract_stamps, RealBogusCNN


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_extract_stamps_shape_and_edge_padding(device):
    height, width = 100, 120
    image = torch.ones((height, width), dtype=torch.float32, device=device) * 10.0

    # Markiere spezifische Pixel an den Zielkoordinaten (x, y) im [Y, X] Bild
    # 1. Zentrum, 2. Sehr nah am oberen/linken Rand, 3. Sehr nah am unteren/rechten Rand
    positions = [
        (60.0, 50.0),
        (2.0, 3.0),
        (118.0, 98.0),
    ]
    image[50, 60] = 500.0
    image[3, 2] = 600.0
    image[98, 118] = 700.0

    stamps = extract_stamps(image, positions, stamp_size=32)

    assert stamps.shape == (3, 1, 32, 32)
    assert stamps.device.type == device.type
    assert stamps.dtype == image.dtype

    # Zentraler Stamp hat kein Zero-Padding (alle Werte >= 10.0) und Peak in der Mitte (16, 16)
    assert stamps[0, 0, 16, 16].item() == pytest.approx(500.0)
    assert torch.min(stamps[0]).item() == pytest.approx(10.0)

    # Rand-Stamps (Index 1 und 2) müssen ohne Absturz durch Zero-Padding aufgefüllt sein
    assert stamps[1, 0, 16, 16].item() == pytest.approx(600.0)
    assert stamps[1, 0, 0, 0].item() == pytest.approx(0.0)

    assert stamps[2, 0, 16, 16].item() == pytest.approx(700.0)
    assert stamps[2, 0, -1, -1].item() == pytest.approx(0.0)


def test_real_bogus_cnn_forward_pass(device):
    torch.manual_seed(42)
    batch_size = 5
    stamps = torch.randn((batch_size, 1, 32, 32), dtype=torch.float32, device=device)

    model = RealBogusCNN().to(device)
    model.eval()

    with torch.no_grad():
        probs = model(stamps)

    assert probs.shape == (batch_size, 1)
    assert probs.device.type == device.type
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)
