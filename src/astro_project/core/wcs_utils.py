import numpy as np
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import torch


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


def compute_wcs_frame_offsets(
    wcs_list: list[WCS],
    reference_wcs: WCS,
    height: int,
    width: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Berechnet für jedes WCS in `wcs_list` den translatorischen Pixel-Versatz (dx_wcs, dy_wcs)
    relativ zu `reference_wcs` am Bildzentrum mit `origin=0` als Tensor [N, 2].
    """
    xc = float(width) / 2.0
    yc = float(height) / 2.0

    offsets = []
    for wcs_i in wcs_list:
        if (
            wcs_i is None
            or reference_wcs is None
            or not wcs_i.is_celestial
            or not reference_wcs.is_celestial
        ):
            offsets.append((0.0, 0.0))
            continue
        ra_c, dec_c = pix_to_world(wcs_i, xc, yc)
        x_ref, y_ref = world_to_pix(reference_wcs, ra_c, dec_c)
        dx_wcs = float(np.asarray(x_ref).item()) - xc
        dy_wcs = float(np.asarray(y_ref).item()) - yc
        if not (np.isfinite(dx_wcs) and np.isfinite(dy_wcs)):
            offsets.append((0.0, 0.0))
        else:
            offsets.append((dx_wcs, dy_wcs))

    return torch.tensor(offsets, dtype=torch.float32, device=device)
