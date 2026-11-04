import logging
import time
from pathlib import Path
from typing import Sequence
import requests

logger = logging.getLogger(__name__)

def query_ps1_archive(ra: float, dec: float, mjd_start: float, mjd_end: float, radius_arcmin: float = 20.0) -> list[dict]:
    """
    Queries the official STScI Pan-STARRS Image Cutout API for epochs within a given region and time range.
    """
    # Convert radius from arcmin to pixels (PS1 pixel scale is ~0.25 arcsec/pixel)
    # 20 arcmin = 1200 arcsec = 4800 pixels. We use a reasonable default size.
    size_pixels = int((radius_arcmin * 60) / 0.25)
    
    # The official API endpoint for getting filenames
    url = f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra}&dec={dec}&size={size_pixels}&format=json"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Failed to query PS1 STScI API: {e}")
        return []
        
    logger.info(f"Found {len(data)} raw entries in PS1 API before filtering.")
    
    results = []
    for row in data:
        mjd = row.get("mjd")
        if mjd is None or not (mjd_start <= mjd <= mjd_end):
            continue
            
        # Construct the direct FITS download URL
        # Format: https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?red={filename}&format=fits&size={size}&ra={ra}&dec={dec}
        filename = row.get("filename")
        if not filename:
            continue
            
        download_url = f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?red={filename}&format=fits&size={size_pixels}&ra={ra}&dec={dec}"
            
        results.append({
            "target": "PS1_Field",
            "mjd": mjd,
            "ra": ra,
            "dec": dec,
            "filter": row.get("filter", "unknown"),
            "fits_url": download_url
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
