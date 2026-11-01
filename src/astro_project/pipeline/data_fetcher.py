from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from astropy.time import Time
from astroquery.ipac.irsa.most import Most


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
