import pytest
import torch
import astropy.units as u
from astropy.coordinates import SkyCoord

from astro_project.core.fits_loader import load_fits_image
from astro_project.core.synthetic import generate_demo_asteroid_series
from astro_project.pipeline.orchestration import run_pipeline


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_generate_demo_asteroid_series_and_pipeline_detection(tmp_path, device):
    output_dir = tmp_path / "series_demo"
    n_frames = 5
    height, width = 128, 128
    true_vx = 2.0
    true_vy = -1.0

    file_paths, times_hours, gt = generate_demo_asteroid_series(
        output_dir=output_dir,
        n_frames=n_frames,
        height=height,
        width=width,
        vx=true_vx,
        vy=true_vy,
        asteroid_snr=1.5,
        device=device,
    )

    # 1. Prüfe Anzahl der geschriebenen Dateien, Zeiten und Ground-Truth-Struktur
    assert len(file_paths) == n_frames
    assert len(times_hours) == n_frames
    assert times_hours == sorted(times_hours)
    assert len(set(times_hours)) == n_frames
    assert "x0" in gt and "y0" in gt and "ra0" in gt and "dec0" in gt
    assert len(gt["stars"]) == 5

    # 2. Prüfe FITS-Header (MJD-OBS, WCS) und dass statische Feldsterne an festen [Y, X] Positionen liegen
    mjd_values = []
    star_y0, star_x0 = int(round(gt["stars"][0][0])), int(round(gt["stars"][0][1]))
    for path in file_paths:
        tensor, wcs, header = load_fits_image(path, device=device)
        assert tensor.shape == (height, width)
        assert wcs.is_celestial is True
        assert "MJD-OBS" in header
        mjd_values.append(float(header["MJD-OBS"]))
        # Statischer Feldstern ist an derselben [Y, X] Position in jedem Frame deutlich über Hintergrund (~100)
        assert tensor[star_y0, star_x0].item() > 500.0

    assert mjd_values == sorted(mjd_values)
    assert len(set(mjd_values)) == n_frames

    # 3. Verifiziere, dass `run_pipeline` den schwachen bewegten Asteroiden über die Serie zuverlässig findet
    times_tensor = torch.tensor(times_hours, dtype=torch.float32, device=device)
    catalog = run_pipeline(
        file_paths=file_paths,
        times_hours=times_tensor,
        vx_range=(-2.0, 2.0),
        vy_range=(-2.0, 2.0),
        steps=5,
        detection_threshold=8.0,
        cnn_model=None,
        device=device,
        star_mask_threshold=150.0,
    )

    assert len(catalog) >= 1
    catalog.sort("peak", reverse=True)
    best = catalog[0]

    assert best["vx"] == pytest.approx(true_vx, abs=1e-4)
    assert best["vy"] == pytest.approx(true_vy, abs=1e-4)
    assert best["x_centroid"] == pytest.approx(gt["x0"], abs=1.0)
    assert best["y_centroid"] == pytest.approx(gt["y0"], abs=1.0)

    true_coord = SkyCoord(ra=gt["ra0"] * u.deg, dec=gt["dec0"] * u.deg, frame="icrs")
    det_coord = SkyCoord(ra=best["ra"] * u.deg, dec=best["dec"] * u.deg, frame="icrs")
    assert true_coord.separation(det_coord).arcsecond < 1.0


def test_generate_custom_and_randomized_asteroid_series(tmp_path, device):
    custom_dir = tmp_path / "custom_series"

    # 1. Custom deterministic parameters
    paths_custom, times_custom, gt_custom = generate_demo_asteroid_series(
        output_dir=custom_dir,
        n_frames=4,
        height=128,
        width=128,
        vx=-1.5,
        vy=1.25,
        asteroid_snr=2.5,
        n_stars=3,
        randomize=False,
        device=device,
    )

    assert len(paths_custom) == 4
    assert len(times_custom) == 4
    assert gt_custom["n_frames"] == 4
    assert gt_custom["vx"] == pytest.approx(-1.5)
    assert gt_custom["vy"] == pytest.approx(1.25)
    assert gt_custom["asteroid_snr"] == pytest.approx(2.5)
    assert len(gt_custom["stars"]) == 3

    for p in paths_custom:
        img_t, wcs, hdr = load_fits_image(p, device=device)
        assert img_t.shape == (128, 128)
        assert wcs.is_celestial is True
        assert "MJD-OBS" in hdr

    # 2. Randomize mode into the same directory (verifies old extra frames are cleaned up)
    paths_rand, times_rand, gt_rand = generate_demo_asteroid_series(
        output_dir=custom_dir,
        height=128,
        width=128,
        n_stars=7,
        randomize=True,
        seed=123,
        device=device,
    )

    assert 3 <= len(paths_rand) <= 15
    assert len(paths_rand) == gt_rand["n_frames"]
    assert len(list(custom_dir.glob("*.fits"))) == len(paths_rand)
    assert len(times_rand) == len(paths_rand)
    assert -5.0 <= gt_rand["vx"] <= 5.0
    assert -5.0 <= gt_rand["vy"] <= 5.0
    assert 1.0 <= gt_rand["asteroid_snr"] <= 5.0
    assert 20.0 <= gt_rand["x0"] <= 108.0
    assert 20.0 <= gt_rand["y0"] <= 108.0
    assert len(gt_rand["stars"]) == 7
