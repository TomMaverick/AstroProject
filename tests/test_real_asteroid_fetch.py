from pathlib import Path
import pytest
import requests
import torch
from astropy.utils.data import get_pkg_data_filename

from astro_project.core.fits_loader import load_fits_image
from astro_project.pipeline.data_fetcher import (
    download_asteroid_fits,
    query_asteroid_epochs,
)


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_astroquery_most_query_asteroid_epochs():
    pytest.importorskip("astroquery.ipac.irsa.most")

    try:
        epochs = query_asteroid_epochs(
            target_name="Ceres",
            catalog="wise_merge",
            obs_begin="2010-01-01",
            obs_end="2010-03-01",
        )
    except (
        requests.RequestException,
        ConnectionError,
        TimeoutError,
        OSError,
         Exception,
    ) as exc:
        pytest.skip(f"IRSA MOST service or internet connection unavailable: {exc}")

    assert isinstance(epochs, list)
    if len(epochs) == 0:
        pytest.skip("IRSA MOST returned 0 epochs for the queried time window.")

    first = epochs[0]
    assert isinstance(first, dict)
    for key in ("mjd", "ra", "dec", "fits_url"):
        assert key in first


def test_download_fits_to_data_dir_and_ingest(tmp_path, device):
    """
    Verifiziert den Download/Speicher-Workflow in den lokalen data/-Ordner
    und die direkte Kompatibilität mit `load_fits_image` ([Y, X] Tensor + WCS).
    """
    sample_fits = Path(get_pkg_data_filename("tutorials/FITS-images/HorseHead.fits"))
    file_uri = sample_fits.as_uri()

    target_data_dir = tmp_path / "data"
    epochs = [
        {
            "mjd": 55200.5,
            "ra": 85.25,
            "dec": -2.45,
            "fits_url": file_uri,
        }
    ]

    downloaded_paths = download_asteroid_fits(epochs, data_dir=target_data_dir)
    assert len(downloaded_paths) == 1
    fits_path = Path(downloaded_paths[0])
    assert fits_path.exists()
    assert fits_path.parent == target_data_dir

    # Direkt mit load_fits_image als [Y, X] Tensor auf das Ziel-Device laden
    tensor, wcs, header = load_fits_image(str(fits_path), device=device)
    assert tensor.ndim == 2
    assert tensor.shape == (header["NAXIS2"], header["NAXIS1"])
    assert tensor.device == device
    assert wcs.is_celestial is True
