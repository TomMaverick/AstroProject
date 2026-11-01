import numpy as np
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord


def get_wcs(header) -> WCS:
    """Erstellt ein WCS-Objekt aus einem FITS-Header."""
    return WCS(header)


def create_synthetic_wcs(
    pixel_scale_arcsec: float = 1.0,
    crval_ra: float = 180.0,
    crval_dec: float = 45.0,
) -> WCS:
    """Erstellt ein synthetisches WCS mit bekannter CD-Matrix für Ground-Truth-Tests."""
    w = WCS(naxis=2)
    w.wcs.crpix = [51.0, 51.0]  # Referenzpixel in 1-basierter FITS-Konvention
    w.wcs.crval = [crval_ra, crval_dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale_deg = pixel_scale_arcsec / 3600.0
    w.wcs.cd = [[-scale_deg, 0.0], [0.0, scale_deg]]
    return w


def pix_to_world(wcs: WCS, x_coords, y_coords):
    """Transformiert Pixelkoordinaten (X, Y) zu Himmelskoordinaten (RA, Dec in Grad) mit origin=0."""
    ra, dec = wcs.wcs_pix2world(x_coords, y_coords, 0)
    return ra, dec


def world_to_pix(wcs: WCS, ra, dec):
    """Transformiert Himmelskoordinaten (RA, Dec in Grad) zu Pixelkoordinaten (X, Y) mit origin=0."""
    x_pix, y_pix = wcs.wcs_world2pix(ra, dec, 0)
    return x_pix, y_pix


def sources_to_skycoords(sources, wcs: WCS) -> SkyCoord:
    """Konvertiert eine Astropy-Quellentabelle (x_centroid, y_centroid) per WCS in SkyCoord (ICRS)."""
    x_coords = np.asarray(sources["x_centroid"], dtype=np.float64)
    y_coords = np.asarray(sources["y_centroid"], dtype=np.float64)
    ra, dec = pix_to_world(wcs, x_coords, y_coords)
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
