import pytest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def test_fits_wcs_loading():
    # Create a synthetic FITS file with WCS
    data = np.random.random((100, 100))
    hdu = fits.PrimaryHDU(data)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50.5, 50.5]
    wcs.wcs.cdelt = np.array([-0.06666667, 0.06666667])
    wcs.wcs.crval = [0, 0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdu.header.update(wcs.to_header())

    # Save the FITS file
    test_fits_file = "test_milestone2.fits"
    hdu.writeto(test_fits_file, overwrite=True)

    # Load the FITS file and check WCS
    with fits.open(test_fits_file) as hdul:
        loaded_wcs = WCS(hdul[0].header)
        assert np.allclose(loaded_wcs.wcs.crpix, wcs.wcs.crpix)
        assert np.allclose(loaded_wcs.wcs.cdelt, wcs.wcs.cdelt)
        assert np.allclose(loaded_wcs.wcs.crval, wcs.wcs.crval)
        assert loaded_wcs.wcs.ctype == wcs.wcs.ctype

    # Clean up
    import os
    os.remove(test_fits_file)
