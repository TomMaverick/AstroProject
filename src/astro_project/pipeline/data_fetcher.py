from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from astropy.time import Time
from astroquery.ipac.irsa.most import Most

from astro_project.core.synthetic import generate_custom_asteroid_series


SURVEY_TELESCOPE_PRESETS: dict[str, dict] = {
    "ZTF_Palomar_48in": {
        "label": "ZTF Palomar 48-inch Oschin Schmidt",
        "pixel_scale_arcsec": 1.01,
        "default_dt_hours": 0.5,
        "obs_code": "I41",
        "fov_deg": 6.8,
    },
    "PanSTARRS_PS1": {
        "label": "Pan-STARRS1 1.8m Haleakala",
        "pixel_scale_arcsec": 0.258,
        "default_dt_hours": 0.25,
        "obs_code": "F51",
        "fov_deg": 3.0,
    },
    "Catalina_CSS_Schmidt": {
        "label": "Catalina Sky Survey 0.7m Schmidt",
        "pixel_scale_arcsec": 2.50,
        "default_dt_hours": 0.2,
        "obs_code": "703",
        "fov_deg": 4.4,
    },
}


def _extract_value(row, candidates: tuple[str, ...], default=None):
    """Liest den ersten vorhandenen Spaltenwert (case-insensitiv) aus einer Astropy-Table-Row."""
    col_map = {col.lower(): col for col in row.colnames}
    for cand in candidates:
        if cand.lower() in col_map:
            val = row[col_map[cand.lower()]]
            if val is not None:
                return val
    return default


def query_asteroid_epochs(
    target_name: str,
    catalog: str = "wise_merge",
    obs_begin: str = "2010-01-01",
    obs_end: str = "2010-06-01",
) -> list[dict]:
    """
    Fragt über `astroquery.ipac.irsa.most.Most` die Beobachtungsepochen,
    Ephemeriden-Koordinaten und FITS-Bild-/Cutout-URLs für einen Asteroiden ab.
    """
    table = Most.query_object(
        catalog=catalog,
        input_type="name_input",
        obj_name=target_name,
        obs_begin=obs_begin,
        obs_end=obs_end,
        output_mode="Regular",
    )

    if table is None or len(table) == 0:
        return []

    epochs: list[dict] = []
    for row in table:
        raw_mjd = _extract_value(row, ("mjd", "obs_mjd", "date_mjd"))
        raw_date = _extract_value(row, ("obs_date", "date_obs", "datetime", "utc"))
        if raw_mjd is not None:
            mjd_val = float(raw_mjd)
        elif raw_date is not None:
            mjd_val = float(Time(str(raw_date)).mjd)
        else:
            mjd_val = 0.0

        ra_val = float(_extract_value(row, ("ra", "ra_obj", "crval1"), 0.0))
        dec_val = float(_extract_value(row, ("dec", "dec_obj", "crval2"), 0.0))
        fits_url = str(
            _extract_value(
                row,
                ("fits_url", "image_url", "cutout_url", "url", "file_url"),
                "",
            )
        )

        epochs.append(
            {
                "target": target_name,
                "mjd": mjd_val,
                "ra": ra_val,
                "dec": dec_val,
                "fits_url": fits_url,
            }
        )

    return epochs


def download_asteroid_fits(
    epochs: list[dict],
    data_dir: str | Path = "data",
    timeout: float = 30.0,
) -> list[str]:
    """
    Lädt die in `epochs` hinterlegten FITS-Dateien (`fits_url`) in den lokalen
    `data/`-Ordner herunter und gibt die lokalen Dateipfade für `load_fits_image` zurück.
    """
    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[str] = []
    for idx, epoch in enumerate(epochs):
        url = epoch.get("fits_url", "")
        if not url:
            continue

        parsed = urlparse(url)
        filename = Path(parsed.path).name
        if not filename.lower().endswith((".fits", ".fit", ".fts")):
            filename = f"asteroid_epoch_{idx:04d}.fits"
        else:
            filename = f"{idx:04d}_{filename}"

        out_path = target_dir / filename
        with urlopen(url, timeout=timeout) as response:
            out_path.write_bytes(response.read())

        downloaded_files.append(str(out_path))

    return downloaded_files


def generate_or_fetch_survey_fields(
    base_data_dir: str | Path = "data",
    telescope: str = "ZTF_Palomar_48in",
    n_fields: int = 3,
    n_frames: int = 5,
    height: int = 128,
    width: int = 128,
    active_field_indices: Sequence[int] | None = None,
    asteroid_snr: float = 2.5,
    vx: float = 2.0,
    vy: float = -1.0,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> list[dict]:
    """
    Populates `base_data_dir / 'staging'` with N blind-survey fields.
    Only fields whose index is in `active_field_indices` contain a synthetic sub-sigma
    moving object; all other fields contain pure background noise and stationary stars.
    """
    preset = SURVEY_TELESCOPE_PRESETS.get(
        telescope, SURVEY_TELESCOPE_PRESETS["ZTF_Palomar_48in"]
    )
    pix_scale = float(preset["pixel_scale_arcsec"])
    dt_hours = float(preset["default_dt_hours"])

    if active_field_indices is None:
        active_set = {0} if n_fields > 0 else set()
    else:
        active_set = set(int(i) for i in active_field_indices)

    staging_root = Path(base_data_dir) / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)

    field_batches: list[dict] = []
    with torch.no_grad():
        for idx in range(int(n_fields)):
            field_id = f"{telescope.lower()}_field_{idx:03d}"
            field_dir = staging_root / field_id
            has_object = idx in active_set
            snr_val = float(asteroid_snr) if has_object else 0.0

            paths, _, gt = generate_custom_asteroid_series(
                output_dir=field_dir,
                n_frames=int(n_frames),
                height=int(height),
                width=int(width),
                vx=float(vx),
                vy=float(vy),
                asteroid_snr=snr_val,
                n_stars=5,
                randomize=False,
                seed=int(seed) + idx * 17,
                device=device,
                orbit_preset="linear",
                dt_hours=dt_hours,
                pixel_scale_arcsec=pix_scale,
                base_mjd=60000.0 + idx * 0.5,
            )

            # Shift RA/Dec per field in FITS headers so each field has a distinct sky hash
            from astropy.io import fits

            ra_field = 180.0 + idx * 1.25
            dec_field = 25.0 + idx * 0.75
            for fpath in paths:
                with fits.open(fpath, mode="update") as hdul:
                    hdul[0].header["CRVAL1"] = ra_field
                    hdul[0].header["CRVAL2"] = dec_field
                    hdul[0].header["TELESCOP"] = telescope
                    hdul.flush()

            field_batches.append(
                {
                    "field_id": field_id,
                    "telescope": telescope,
                    "file_paths": paths,
                    "has_ground_truth": has_object,
                    "ground_truth": gt,
                }
            )

    return field_batches
