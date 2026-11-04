import logging
import time
from pathlib import Path
from typing import Sequence
import requests

logger = logging.getLogger(__name__)

def query_ps1_archive(ra_deg: float, dec_deg: float, mjd_start: float, mjd_end: float, radius_arcmin: float = 20.0, band: str = "r") -> list[dict]:
    """
    Queries the official STScI Pan-STARRS1 API for epochs using:
    1. The ps1filenames.py service to find observations.
    2. fitscut.cgi to build the direct FITS download URLs.
    """
    size_pix = int(radius_arcmin * 60)
    url = f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra_deg}&dec={dec_deg}&size={size_pix}&format=json"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to query PS1 filenames API: {e}")
        return []
         
    results = []
    
    if not isinstance(data, list):
        logger.error("Unexpected JSON format from PS1 filenames API")
        return []
    
    for item in data:
        item_filter = item.get("filter", "")
        if band and item_filter != band:
            continue
            
        try:
            mjd = float(item.get("mjd", 0.0))
        except (ValueError, TypeError):
            continue
              
        if not (mjd_start <= mjd <= mjd_end):
            continue
              
        filename = item.get("filename", "")
        if not filename:
            continue
          
        fits_url = f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?ra={ra_deg}&dec={dec_deg}&size=240&format=fits&red={filename}"
          
        results.append({
            "target": "PS1_Field",
            "mjd": mjd,
            "ra": ra_deg,
            "dec": dec_deg,
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
