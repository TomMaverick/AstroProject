import math
from typing import Sequence
import numpy as np
import torch
from astropy.time import Time
from astropy.wcs import WCS

from astro_project.core.wcs_utils import pix_to_world


def compute_tracklet_endpoints(
    wcs: WCS,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    times_hours: torch.Tensor | Sequence[float],
    base_mjd: float = 60000.0,
) -> list[dict]:
    """
    Convert a linear pixel trajectory (x0, y0, vx, vy) across observation epochs
    into sky coordinates (ra_deg, dec_deg, mjd) using WCS with origin=0.
    """
    if isinstance(times_hours, torch.Tensor):
        t_arr = times_hours.detach().cpu().numpy().astype(np.float64)
    else:
        t_arr = np.asarray(times_hours, dtype=np.float64)

    t0 = float(t_arr[0]) if t_arr.size > 0 else 0.0
    dt = t_arr - t0

    x_coords = float(x0) + float(vx) * dt
    y_coords = float(y0) + float(vy) * dt

    ra_arr, dec_arr = pix_to_world(wcs, x_coords, y_coords)
    ra_arr = np.atleast_1d(np.asarray(ra_arr, dtype=np.float64))
    dec_arr = np.atleast_1d(np.asarray(dec_arr, dtype=np.float64))

    epochs: list[dict] = []
    for i in range(len(dt)):
        mjd = float(base_mjd) + float(t_arr[i]) / 24.0
        epochs.append(
            {
                "mjd": mjd,
                "ra_deg": float(ra_arr[i]),
                "dec_deg": float(dec_arr[i]),
            }
        )
    return epochs


def estimate_kepler_orbit(tracklet_epochs: Sequence[dict]) -> dict:
    """
    Estimate approximate heliocentric Keplerian orbital elements (a_au, e, i_deg,
    Omega_deg, omega_deg, M_deg, orbit_class) from tracklet epochs using a
    Väisälä / opposition circular-orbit approximation with k = 0.01720209895.
    """
    if len(tracklet_epochs) < 2:
        raise ValueError("At least 2 tracklet epochs are required for orbit estimation.")

    sorted_eps = sorted(tracklet_epochs, key=lambda ep: float(ep["mjd"]))
    ep_start = sorted_eps[0]
    ep_mid = sorted_eps[len(sorted_eps) // 2]
    ep_end = sorted_eps[-1]

    dt_days = float(ep_end["mjd"]) - float(ep_start["mjd"])
    if dt_days <= 0.0:
        dt_days = 1e-3

    ra1 = math.radians(float(ep_start["ra_deg"]))
    dec1 = math.radians(float(ep_start["dec_deg"]))
    ra2 = math.radians(float(ep_end["ra_deg"]))
    dec2 = math.radians(float(ep_end["dec_deg"]))
    dec_mid = math.radians(float(ep_mid["dec_deg"]))

    dra_rad_per_day = ((ra2 - ra1 + math.pi) % (2.0 * math.pi) - math.pi) * math.cos(dec_mid) / dt_days
    ddec_rad_per_day = (dec2 - dec1) / dt_days
    omega_sky = math.hypot(dra_rad_per_day, ddec_rad_per_day)

    k_gauss = 0.01720209895  # rad / day for Earth at 1 AU

    # In opposition: omega_sky ≈ k * (1 - 1 / r^1.5) / (r - 1).
    # Solve numerically on [1.05, 50.0] AU for heliocentric distance r (a_au).
    r_grid = np.linspace(1.05, 50.0, 2000)
    model_rate = k_gauss * (1.0 - np.power(r_grid, -1.5)) / (r_grid - 1.0)
    idx = int(np.argmin(np.abs(model_rate - omega_sky)))
    a_au = float(r_grid[idx])

    # Estimate inclination from sky velocity angle and mid-epoch declination
    vel_angle = abs(math.atan2(abs(ddec_rad_per_day), max(abs(dra_rad_per_day), 1e-9)))
    i_deg = float(np.clip(math.degrees(math.hypot(abs(dec_mid) * 0.5, vel_angle * 0.35)), 0.0, 179.9))

    # Estimate non-zero small eccentricity from curvature / rate residual across 3 epochs
    if len(sorted_eps) >= 3:
        dt1 = max(float(ep_mid["mjd"]) - float(ep_start["mjd"]), 1e-6)
        dt2 = max(float(ep_end["mjd"]) - float(ep_mid["mjd"]), 1e-6)
        ra_m = math.radians(float(ep_mid["ra_deg"]))
        dec_m = math.radians(float(ep_mid["dec_deg"]))
        v1 = math.hypot((ra_m - ra1) * math.cos(dec_mid) / dt1, (dec_m - dec1) / dt1)
        v2 = math.hypot((ra2 - ra_m) * math.cos(dec_mid) / dt2, (dec2 - dec_m) / dt2)
        rel_accel = abs(v2 - v1) / max(omega_sky, 1e-6)
        e_est = float(np.clip(0.05 + 0.5 * rel_accel, 0.0, 0.95))
    else:
        e_est = 0.08

    if a_au < 1.3:
        orbit_class = "NEA"
    elif a_au < 2.0:
        orbit_class = "Mars-Crosser"
    elif a_au <= 3.2:
        orbit_class = "Main-Belt"
    elif a_au <= 5.5:
        orbit_class = "Outer-Belt / Trojan"
    else:
        orbit_class = "Centaur / TNO"

    return {
        "a_au": a_au,
        "e": e_est,
        "i_deg": i_deg,
        "Omega_deg": float(ep_mid["ra_deg"]) % 360.0,
        "omega_deg": 0.0,
        "M_deg": 0.0,
        "orbit_class": orbit_class,
    }


def format_mpc80_line(
    designation: str,
    mjd: float,
    ra_deg: float,
    dec_deg: float,
    mag: float = 20.0,
    band: str = "V",
    obs_code: str = "500",
) -> str:
    """
    Format an astrometric observation according to the official 80-column
    Minor Planet Center (MPC) optical observation standard.
    """
    # Columns 1-12: packed number (5 chars) + packed provisional/temporary designation (7 chars)
    desig_str = f"     {str(designation).strip()[:7]:<7s}"[:12]
    # Column 13: discovery asterisk (blank by default), Col 14: note1, Col 15: 'C' for CCD
    col_13_15 = "  C"

    # Columns 16-32: Date of observation "YYYY MM DD.ddddd " (17 chars)
    t = Time(float(mjd), format="mjd", scale="utc")
    ymdhms = t.ymdhms
    year = int(ymdhms.year)
    month = int(ymdhms.month)
    day_frac = (
        int(ymdhms.day)
        + (
            int(ymdhms.hour) * 3600.0
            + int(ymdhms.minute) * 60.0
            + float(ymdhms.second)
        )
        / 86400.0
    )
    date_str = f"{year:04d} {month:02d} {day_frac:08.5f} "  # 4 + 1 + 2 + 1 + 8 + 1 = 17 chars

    # Columns 33-44: RA "HH MM SS.dd " (12 chars)
    ra_norm = float(ra_deg) % 360.0
    total_ra_sec = ra_norm * 240.0
    ra_h = int(total_ra_sec // 3600)
    rem_ra = total_ra_sec - ra_h * 3600
    ra_m = int(rem_ra // 60)
    ra_s = rem_ra - ra_m * 60
    if ra_s >= 59.995:
        ra_s = 0.0
        ra_m += 1
    if ra_m >= 60:
        ra_m = 0
        ra_h = (ra_h + 1) % 24
    ra_str = f"{ra_h:02d} {ra_m:02d} {ra_s:05.2f} "  # 2 + 1 + 2 + 1 + 5 + 1 = 12 chars

    # Columns 45-56: Dec "sDD MM SS.d " (12 chars)
    sign_char = "-" if float(dec_deg) < 0 else "+"
    total_dec_sec = abs(float(dec_deg)) * 3600.0
    dec_d = int(total_dec_sec // 3600)
    rem_dec = total_dec_sec - dec_d * 3600
    dec_m = int(rem_dec // 60)
    dec_s = rem_dec - dec_m * 60
    if dec_s >= 59.95:
        dec_s = 0.0
        dec_m += 1
    if dec_m >= 60:
        dec_m = 0
        dec_d += 1
    dec_str = f"{sign_char}{dec_d:02d} {dec_m:02d} {dec_s:04.1f} "  # 1 + 2 + 1 + 2 + 1 + 4 + 1 = 12 chars

    # Columns 57-65: 9 blanks
    col_57_65 = " " * 9

    # Columns 66-71: Magnitude (5 chars "dd.dd") + Band (1 char)
    band_char = (str(band).strip()[:1] or "V")
    mag_str = f"{float(mag):5.2f}{band_char}"  # 6 chars

    # Columns 72-77: 6 blanks
    col_72_77 = " " * 6

    # Columns 78-80: 3-char Observatory code
    obs_str = f"{str(obs_code).strip()[:3]:>3s}"

    line = (
        f"{desig_str}{col_13_15}{date_str}{ra_str}{dec_str}"
        f"{col_57_65}{mag_str}{col_72_77}{obs_str}"
    )
    return line[:80].ljust(80)


def export_tracklet_mpc80(
    tracklet_epochs: Sequence[dict],
    designation: str = "K26A00A",
    mag: float = 20.0,
    band: str = "V",
    obs_code: str = "500",
) -> str:
    """
    Format a sequence of tracklet epochs (each containing 'mjd', 'ra_deg', 'dec_deg')
    into a multi-line 80-column Minor Planet Center (MPC) report string.
    """
    lines = [
        format_mpc80_line(
            designation=designation,
            mjd=float(ep["mjd"]),
            ra_deg=float(ep["ra_deg"]),
            dec_deg=float(ep["dec_deg"]),
            mag=mag,
            band=band,
            obs_code=obs_code,
        )
        for ep in tracklet_epochs
    ]
    return "\n".join(lines)
