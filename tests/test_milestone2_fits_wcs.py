import pytest
import torch
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from astro_project.core.wcs_utils import create_synthetic_wcs, pix_to_world, world_to_pix
from astro_project.core.fits_loader import load_fits_image, save_fits_image


@pytest.fixture(autouse=True)
def vram_cleanup():
    """VRAM protection teardown as required by astrophysics verification rules."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_wcs_astrometry_roundtrip():
    """Test 1: Verify WCS pixel <-> world transformation with sub-arcsecond accuracy (origin=0)."""
    wcs = create_synthetic_wcs(pixel_scale_arcsec=1.0, crval_ra=180.0, crval_dec=45.0)

    # Reference pixel in FITS is 1-based [51.0, 51.0], which corresponds to [50.0, 50.0] in 0-based Python indexing
    x_orig = np.array([50.0, 10.25, 85.75], dtype=np.float64)
    y_orig = np.array([50.0, 20.50, 90.10], dtype=np.float64)

    # Forward transformation: Pixel (X, Y) -> World (RA, Dec)
    ra, dec = pix_to_world(wcs, x_orig, y_orig)

    # Check that the reference pixel (50.0, 50.0) maps to CRVAL (180.0, 45.0)
    assert np.isclose(ra[0], 180.0, atol=1e-7)
    assert np.isclose(dec[0], 45.0, atol=1e-7)

    # Backward transformation: World (RA, Dec) -> Pixel (X, Y)
    x_rec, y_rec = world_to_pix(wcs, ra, dec)

    # Verify round-trip in pixel space
    assert np.allclose(x_orig, x_rec, atol=1e-5)
    assert np.allclose(y_orig, y_rec, atol=1e-5)

    # Verify sub-arcsecond accuracy in sky coordinates
    ra_reproj, dec_reproj = pix_to_world(wcs, x_rec, y_rec)
    c_orig = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    c_reproj = SkyCoord(ra=ra_reproj * u.deg, dec=dec_reproj * u.deg, frame="icrs")
    sep_arcsec = c_orig.separation(c_reproj).arcsecond

    assert np.all(sep_arcsec < 1e-3)


def test_fits_loader_axes_and_nan_handling(tmp_path, device):
    """Test 2: Verify mock-free FITS saving/loading preserves [Y, X] shape and replaces NaNs with 0.0."""
    height, width = 64, 128  # Asymmetric [Y, X] dimensions to strictly test axis preservation
    tensor_in = torch.ones((height, width), dtype=torch.float32, device=device)

    # Inject NaN and Inf at specific [Y, X] coordinates
    tensor_in[10, 20] = float("nan")
    tensor_in[30, 40] = float("inf")
    tensor_in[50, 60] = float("-inf")
    tensor_in[5, 100] = 42.5

    wcs_in = create_synthetic_wcs(pixel_scale_arcsec=1.5, crval_ra=120.0, crval_dec=30.0)
    fits_path = tmp_path / "synthetic_test.fits"

    # Save tensor mock-free to disk
    save_fits_image(tensor_in, fits_path, wcs=wcs_in)

    # Load tensor back to target device
    tensor_out, wcs_out, header_out = load_fits_image(fits_path, device=device, nan_fill=0.0)

    # Verify [Y, X] dimensions, dtype, and device
    assert tensor_out.shape == (height, width)
    assert tensor_out.dtype == torch.float32
    assert tensor_out.device.type == device.type

    # Verify NaN and Inf values were replaced by 0.0
    assert not torch.isnan(tensor_out).any()
    assert not torch.isinf(tensor_out).any()
    assert tensor_out[10, 20].item() == 0.0
    assert tensor_out[30, 40].item() == 0.0
    assert tensor_out[50, 60].item() == 0.0

    # Verify valid data and WCS preservation
    assert np.isclose(tensor_out[5, 100].item(), 42.5)
    assert np.allclose(wcs_out.wcs.crval, wcs_in.wcs.crval)
    assert header_out["CREATOR"] == "AstroProject Pipeline"
