import logging
import time
from pathlib import Path
from typing import Sequence
import requests

logger = logging.getLogger(__name__)

def query_ps1_archive(ra: float, dec: float, mjd_start: float, mjd_end: float, radius_arcmin: float = 20.0) -> list[dict]:
    """
    Queries the official STScI Pan-STARRS1 API for epochs using:
    1. ps1filenames.py to find observations and filenames.
    2. fitscut.cgi to build the direct FITS download URLs.
    """
    # Step 1: Get filenames and metadata
    url = f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra}&dec={dec}&filters=grizy"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
    except requests.RequestException as e:
        logger.error(f"Failed to query PS1 filenames API: {e}")
        return []
        
    if not lines or len(lines) < 2:
        logger.info("No PS1 observations found.")
        return []
        
    header = lines[0].split()
    results = []
    
    for line in lines[1:]:
        row = dict(zip(header, line.split()))
        
        try:
            mjd = float(row.get("mjd", 0.0))
        except ValueError:
            continue
            
        if not (mjd_start <= mjd <= mjd_end):
            continue
            
        filename = row.get("filename", "")
        if not filename:
            continue
            
        filt = row.get("filter", "unknown")
        
        # Step 2: Build fitscut.cgi URL (using 'red' parameter to satisfy color requirement)
        fits_url = f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?ra={ra}&dec={dec}&size=240&format=fits&red={filename}"
        
        results.append({
            "target": "PS1_Field",
            "mjd": mjd,
            "ra": ra,
            "dec": dec,
            "filter": filt,
            "fits_url": fits_url,
            "prod_type": "IMAGE"
        })
        
    # Sort chronologically by MJD
    results.sort(key=lambda x: x["mjd"])
    logger.info(f"Found {len(results)} PS1 epochs in the specified MJD range.")
    return results

def download_ps1_batch(epochs: Sequence[dict], dest_dir: str | Path = "data/staging", timeout: int = 60) -> list[str]:
    """
    Downloads a batch of PS1 FITS files robustly via fitscut.cgi, skipping existing files.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    
    for idx, item in enumerate(epochs):
        url = item.get("fits_url")
        if not url:
            continue
            
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
