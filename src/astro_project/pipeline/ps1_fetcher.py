import logging
import time
from io import StringIO
from pathlib import Path
from typing import Sequence
import requests
from astropy.table import Table

logger = logging.getLogger(__name__)

def query_ps1_archive(ra_deg: float, dec_deg: float, mjd_start: float, mjd_end: float, radius_arcmin: float = 15.0, band: str = "r") -> list[dict]:
    """
    Queries the official STScI Pan-STARRS1 API for epochs using:
    1. The ps1filenames.py service (bulk POST) to find observations.
    2. fitscut.cgi to build the direct FITS download URLs.
    """
    url = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    data = {"filters": band, "type": "stack"}
    files = {"file": StringIO(f"{ra_deg} {dec_deg}")}

    try:
        response = requests.post(url, data=data, files=files, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to query PS1 filenames API: {e}")
        return []

    try:
        table = Table.read(response.text, format="ascii")
    except Exception as e:
        logger.error(f"Failed to parse PS1 ASCII response: {e}")
        return []

    results = []

    for row in table:
        try:
            mjd = float(row["mjd"]) if "mjd" in row.colnames else 0.0
        except (ValueError, TypeError):
            mjd = 0.0

        if "mjd" in row.colnames and not (mjd_start <= mjd <= mjd_end):
            continue

        filename = str(row["filename"]) if "filename" in row.colnames else ""
        if not filename:
            continue

        ra_val = float(row["ra"]) if "ra" in row.colnames else ra_deg
        dec_val = float(row["dec"]) if "dec" in row.colnames else dec_deg

        fits_url = f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?size=240&format=fits&ra={ra_val}&dec={dec_val}&red={filename}"

        results.append({
            "target": "PS1_Field",
            "mjd": mjd,
            "ra": ra_val,
            "dec": dec_val,
            "url": fits_url,
            "prod_type": "IMAGE"
        })

    # Sort chronologically by MJD
    results.sort(key=lambda x: x["mjd"])
    logger.info(f"Found {len(results)} PS1 epochs in the specified MJD range.")
    return results

def download_ps1_batch(query_results: list[dict], dest_dir: str | Path = "data/staging", timeout: int = 30) -> list[str]:
    """
    Downloads a batch of PS1 FITS files robustly via fitscut.cgi, skipping existing files.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    downloaded_files = []

    for idx, item in enumerate(query_results):
        url = item.get("url")
        if not url:
            continue

        mjd = item.get("mjd", idx)
        filename = f"ps1_{mjd:.5f}.fits"

        local_file = dest_path / filename

        if local_file.exists():
            logger.info(f"File already exists, skipping: {local_file}")
            downloaded_files.append(str(local_file))
            continue

        try:
            logger.info(f"Downloading PS1 cutout (MJD {mjd})...")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            with open(local_file, "wb") as f:
                f.write(response.content)

            downloaded_files.append(str(local_file))
            time.sleep(1.0)  # Be polite to the STScI server

        except requests.RequestException as e:
            logger.error(f"Failed to download PS1 cutout: {e}")

    return downloaded_files
