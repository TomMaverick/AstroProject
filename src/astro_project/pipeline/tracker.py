import math
from astropy.wcs import WCS

from astro_project.core.wcs_utils import pix_to_world


def predict_boundary_exit(
    wcs: WCS,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    height: int,
    width: int,
    base_mjd: float = 60000.0,
    lookahead_hours: float = 1.0,
) -> dict:
    """
    Predict when and where a moving target exits the image boundaries [0, width] x [0, height],
    and extrapolate sky coordinates for fetching the adjacent neighbor cutout.
    """
    candidates: list[tuple[float, str]] = []

    if vx > 0.0:
        tx = (float(width) - float(x0)) / float(vx)
        candidates.append((tx, "right"))
    elif vx < 0.0:
        tx = (0.0 - float(x0)) / float(vx)
        candidates.append((tx, "left"))

    if vy > 0.0:
        ty = (float(height) - float(y0)) / float(vy)
        candidates.append((ty, "top"))
    elif vy < 0.0:
        ty = (0.0 - float(y0)) / float(vy)
        candidates.append((ty, "bottom"))

    positive = [(t, edge) for t, edge in candidates if t >= 0.0]
    if not positive:
        ra0, dec0 = pix_to_world(wcs, float(x0), float(y0))
        return {
            "t_exit_hours": math.inf,
            "mjd_exit": math.inf,
            "exit_edge": "stationary",
            "x_exit": float(x0),
            "y_exit": float(y0),
            "ra_exit_deg": float(ra0),
            "dec_exit_deg": float(dec0),
            "ra_next_deg": float(ra0),
            "dec_next_deg": float(dec0),
        }

    t_exit_hours, exit_edge = min(positive, key=lambda item: item[0])
    x_exit = float(x0) + float(vx) * t_exit_hours
    y_exit = float(y0) + float(vy) * t_exit_hours

    t_next_hours = t_exit_hours + float(lookahead_hours)
    x_next = float(x0) + float(vx) * t_next_hours
    y_next = float(y0) + float(vy) * t_next_hours

    ra_exit, dec_exit = pix_to_world(wcs, x_exit, y_exit)
    ra_next, dec_next = pix_to_world(wcs, x_next, y_next)

    return {
        "t_exit_hours": float(t_exit_hours),
        "mjd_exit": float(base_mjd) + float(t_exit_hours) / 24.0,
        "exit_edge": exit_edge,
        "x_exit": float(x_exit),
        "y_exit": float(y_exit),
        "ra_exit_deg": float(ra_exit),
        "dec_exit_deg": float(dec_exit),
        "ra_next_deg": float(ra_next),
        "dec_next_deg": float(dec_next),
    }
