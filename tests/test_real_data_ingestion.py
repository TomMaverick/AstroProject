import pytest
import torch
from astropy.io import fits
from astropy.time import Time
from astropy.utils.data import get_pkg_data_filename

from astro_project.core.fits_loader import load_fits_image
from astro_project.core.photometry import detect_sources, estimate_background
from astro_project.core.wcs_utils import sources_to_skycoords


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def real_fits_path():
    """
    Bezieht das reale Palomar-Teleskop-FITS-Bild (HorseHead.fits) aus den
    offiziellen Astropy-Paketdaten.
    """
    return get_pkg_data_filename("tutorials/FITS-images/HorseHead.fits")


def test_real_fits_header_and_wcs_ingestion(real_fits_path, device):
    tensor, wcs, header = load_fits_image(real_fits_path, device=device)

    # 1. Tensor liegt auf dem Ziel-Device und ist Float32
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device == device
    assert tensor.dtype == torch.float32

    # 2. Achsen-Konvention [Y, X] = [NAXIS2, NAXIS1] prüfen
    assert tensor.ndim == 2
    assert tensor.shape == (header["NAXIS2"], header["NAXIS1"])

    # 3. Keine NaNs oder Infs im geladenen Tensor
    assert not torch.isnan(tensor).any().item()
    assert not torch.isinf(tensor).any().item()

    # 4. Gültiges astronomisches WCS-Objekt
    assert wcs is not None
    assert wcs.is_celestial is True


def test_real_fits_time_and_metadata_extraction(real_fits_path, device):
    _, _, header = load_fits_image(real_fits_path, device=device)

    assert isinstance(header, fits.Header)

    # Prüfen, dass Beobachtungsdatum / Zeitangaben im Header vorhanden und mit Astropy Time lesbar sind
    date_key = next((k for k in ("DATE-OBS", "DATE", "MJD-OBS") if k in header), None)
    assert date_key is not None
    obs_time = Time(header[date_key])
    assert obs_time.mjd > 0.0

    # Prüfen auf Belichtungszeit- oder Instrumenten-/Teleskop-Metadaten im echten FITS-Header
    meta_keys = ("EXPTIME", "EXPOSURE", "TELESCOP", "INSTRUME", "OBJECT", "ORIGIN")
    present_keys = [k for k in meta_keys if k in header]
    assert len(present_keys) > 0


def test_real_data_photometry_and_source_detection(real_fits_path, device):
    image_tensor, wcs, _ = load_fits_image(real_fits_path, device=device)

    # 1. 2D-Hintergrund auf dem echten Teleskopbild schätzen und abziehen
    bkg_tensor = estimate_background(image_tensor)
    assert bkg_tensor.shape == image_tensor.shape
    assert bkg_tensor.device == device

    residual = image_tensor - bkg_tensor
    noise_std = float(torch.std(residual).item())
    assert noise_std > 0.0

    # 2. Echte Feldsterne im hintergrundbereinigten Bild detektieren (5-Sigma Schwellenwert)
    sources = detect_sources(residual, threshold=5.0 * noise_std, fwhm=3.0)
    assert sources is not None
    assert len(sources) >= 5
    assert "x_centroid" in sources.colnames
    assert "y_centroid" in sources.colnames

    # 3. Gefundene Sterne über das echte WCS in Himmelskoordinaten (RA, Dec) transformieren
    sky_coords = sources_to_skycoords(sources, wcs)
    assert len(sky_coords) == len(sources)
    assert all(0.0 <= ra <= 360.0 for ra in sky_coords.ra.deg)
    assert all(-90.0 <= dec <= 90.0 for dec in sky_coords.dec.deg)
