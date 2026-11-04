# src/astro_project/pipeline/ztf_fetcher.py
import logging
import time
from pathlib import Path
import requests

from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ipac.irsa import Irsa

logger = logging.getLogger(__name__)

def query_ztf_archive(
    ra_deg: float, 
    dec_deg: float, 
    mjd_start: float, 
    mjd_end: float, 
    radius_arcmin: float = 15.0, 
    band: str = "r"
) -> list[dict]:
    """
    Queries the ZTF science archive (ztf_sci) via IRSA for a given region and time range.
    """
    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame='icrs')
    
    try:
        table = Irsa.query_region(coord, catalog="ztf_sci", spatial="Cone", radius=radius_arcmin * u.arcmin)
    except Exception as e:
        logger.error(f"Failed to query IRSA ZTF archive: {e}")
        return []

    target_filter = f"z{band}"
    results = []

    for row in table:
        obsjd = row.get("obsjd")
        filtercode = row.get("filtercode")
        
        if obsjd is None or filtercode is None:
            continue
            
        if not (mjd_start <= obsjd <= mjd_end):
            continue
            
        if filtercode != target_filter and filtercode != band:
            continue

        # Extract URL
        download_url = ""
        if "fileuri" in table.colnames:
            download_url = row["fileuri"]
        elif "url" in table.colnames:
            download_url = row["url"]
        else:
            # Fallback construction if columns are missing but filefracday is present
            fracday = row.get("filefracday", "")
            download_url = f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{fracday}_sciimg.fits"

        results.append({
            "filefracday": row.get("filefracday"),
            "obsjd": obsjd,
            "mjd": obs_mjd,  # Add MJD to result
            "ra": row.get("ra"),
            "dec": row.get("dec"),
            "seeing": row.get("seeing"),
            "url": download_url
        })

    # Sort chronologically by MJD
    results.sort(key=lambda x: x["obsjd"])
    return results

def download_ztf_batch(query_results: list[dict], dest_dir: str | Path, timeout: int = 30) -> list[str]:
    """
    Downloads a batch of ZTF FITS files robustly, skipping existing files.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    
    for item in query_results:
        url = item.get("url")
        if not url:
            continue
            
        filename = url.split("/")[-1]
        if not filename.endswith(".fits"):
            filename += ".fits"
            
        local_file = dest_path / filename
        
        if local_file.exists():
            logger.info(f"File already exists, skipping: {local_file}")
            downloaded_files.append(str(local_file))
            continue
            
        try:
            logger.info(f"Downloading {url}...")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            with open(local_file, "wb") as f:
                f.write(response.content)
                
            downloaded_files.append(str(local_file))
            time.sleep(0.5)  # Be polite to the server
            
        except requests.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            
    return downloaded_files
