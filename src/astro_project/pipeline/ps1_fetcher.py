import json
import logging
import time
from pathlib import Path
from typing import Sequence
import requests

logger = logging.getLogger(__name__)

def query_ps1_archive(ra: float, dec: float, mjd_start: float, mjd_end: float, radius_arcmin: float = 20.0) -> list[dict]:
    """
    Queries the official MAST API for Pan-STARRS1 epochs within a given region and time range.
    """
    url = "https://mast.stsci.edu/api/v0/invoke"
    req_obj = {
        "service": "Mast.Caom.Cone",
        "params": {
            "ra": ra,
            "dec": dec,
            "radius": radius_arcmin / 60.0
        },
        "format": "json",
        "pagesize": 5000,
        "removenullcolumns": True,
        "timeout": 30
    }
    
    try:
        response = requests.post(url, data={"request": json.dumps(req_obj)}, timeout=30)
        response.raise_for_status()
        resp_data = response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to query MAST API: {e}")
        if 'response' in locals() and response is not None:
            logger.error(f"Response text: {response.text}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode MAST API JSON: {e}")
        if 'response' in locals() and response is not None:
            logger.error(f"Response text: {response.text}")
        return []
        
    if resp_data.get("status") != "COMPLETE":
        logger.error(f"MAST API returned non-COMPLETE status: {resp_data.get('status')}")
        return []
        
    data_array = resp_data.get("data", [])
    logger.info(f"Found {len(data_array)} raw entries in MAST API before filtering.")
    
    results = []
    for row in data_array:
        if row.get("obs_collection") != "PS1":
            continue
            
        mjd = row.get("t_min")
        if mjd is None or not (mjd_start <= mjd <= mjd_end):
            continue
            
        data_url = row.get("dataURL")
        if not data_url:
            obs_id = row.get("obs_id", "")
            data_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={obs_id}"
        elif not data_url.startswith("http"):
            data_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={data_url}"
            
        results.append({
            "target": row.get("target_name", "PS1_Field"),
            "mjd": mjd,
            "ra": row.get("s_ra", ra),
            "dec": row.get("s_dec", dec),
            "filter": row.get("filters", "unknown"),
            "fits_url": data_url
        })
        
    # Sort chronologically by MJD
    results.sort(key=lambda x: x["mjd"])
    return results

def download_ps1_batch(epochs: Sequence[dict], dest_dir: str | Path = "data/staging", timeout: int = 60) -> list[str]:
    """
    Downloads a batch of PS1 FITS files robustly, skipping existing files.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    
    for idx, item in enumerate(epochs):
        url = item.get("fits_url")
        if not url:
            continue
            
        # Generate a unique filename based on MJD and filter
        mjd = item.get("mjd", idx)
        filt = item.get("filter", "unk")
        filename = f"ps1_{mjd:.5f}_{filt}.fits"
            
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
