import json
import math
import requests


def radec_to_j2000_vector(ra_deg: float, dec_deg: float) -> list[float]:
    """
    Convert spherical equatorial coordinates (RA, Dec in degrees) into a 3D J2000 unit vector
    [x, y, z] expected by the Stellarium RemoteControl API (/api/main/view).
    """
    ra_rad = math.radians(float(ra_deg))
    dec_rad = math.radians(float(dec_deg))
    cos_dec = math.cos(dec_rad)
    return [
        float(cos_dec * math.cos(ra_rad)),
        float(cos_dec * math.sin(ra_rad)),
        float(math.sin(dec_rad)),
    ]


def slew_stellarium_to_candidate(
    ra_deg: float,
    dec_deg: float,
    mjd: float | None = None,
    base_url: str = "http://localhost:8090/api",
    timeout: float = 1.5,
) -> dict:
    """
    Set observation time (if mjd is provided) and slew the Stellarium camera to (ra_deg, dec_deg)
    via the Stellarium RemoteControl HTTP API. Never raises on connection errors.
    """
    j2000_vec = radec_to_j2000_vector(ra_deg, dec_deg)
    clean_base = base_url.rstrip("/")
    jd = float(mjd) + 2400000.5 if mjd is not None else None

    try:
        if jd is not None:
            resp_time = requests.post(
                f"{clean_base}/main/time",
                data={"time": str(jd), "timerate": "0"},
                timeout=timeout,
            )
            resp_time.raise_for_status()

        resp_view = requests.post(
            f"{clean_base}/main/view",
            data={"j2000": json.dumps(j2000_vec)},
            timeout=timeout,
        )
        resp_view.raise_for_status()

        return {
            "ok": True,
            "error": None,
            "j2000_vec": j2000_vec,
            "jd": jd,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "j2000_vec": j2000_vec,
            "jd": jd,
        }
