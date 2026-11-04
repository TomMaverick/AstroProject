import json
import logging
import time
from pathlib import Path
from typing import Sequence
import requests

logger = logging.getLogger(__name__)

def query_ps1_archive(ra: float, dec: float, mjd_start: float, mjd_end: float, radius_arcmin: float = 20.0) -> list[dict]:
    """
    Queries the official MAST API for Pan-STARRS1 epochs using a two-step workflow:
    1. Mast.Caom.Cone to find observations.
    2. Mast.Caom.Products to get the actual FITS file URIs.
    """
    url = "https://mast.stsci.edu/api/v0/invoke"
    
    # Step 1: Cone Search
    cone_req = {
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
        response = requests.post(url, data={"request": json.dumps(cone_req)}, timeout=30)
        response.raise_for_status()
        cone_data = response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error(f"Failed to query MAST Cone API: {e}")
        return []
        
    if cone_data.get("status") != "COMPLETE":
        logger.error(f"MAST Cone API returned non-COMPLETE status: {cone_data.get('status')}")
        return []
        
    obs_metadata = {}
    for row in cone_data.get("data", []):
        if row.get("obs_collection") != "PS1":
            continue
            
        mjd = row.get("t_min")
        if mjd is None or not (mjd_start <= mjd <= mjd_end):
            continue
            
        obsid = row.get("obsid")
        if obsid:
            obs_metadata[str(obsid)] = {
                "mjd": mjd,
                "ra": row.get("s_ra", ra),
                "dec": row.get("s_dec", dec),
                "filter": row.get("filters", "unknown")
            }
            
    if not obs_metadata:
        logger.info("No PS1 observations found in the given MJD range.")
        return []
        
    # Step 2: Products Search
    obsid_list = list(obs_metadata.keys())
    prod_req = {
        "service": "Mast.Caom.Products",
        "params": {
            "obsid": ",".join(obsid_list)
        },
        "format": "json",
        "pagesize": 5000,
        "removenullcolumns": True,
        "timeout": 30
    }
    
    try:
        response = requests.post(url, data={"request": json.dumps(prod_req)}, timeout=30)
        response.raise_for_status()
        prod_data = response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error(f"Failed to query MAST Products API: {e}")
        return []
        
    results = []
    for prod in prod_data.get("data", []):
        uri = prod.get("dataURI", "")
        prod_type = prod.get("productType", "")
        
        if not uri.lower().endswith(".fits") and prod_type != "IMAGE":
            continue
            
        # Match back to observation metadata
        parent_obsid = str(prod.get("obsID", ""))
        meta = obs_metadata.get(parent_obsid)
        if not meta:
            continue
            
        download_url = f"https://mast.stsci.edu/api/v0/download/file?uri={uri}"
        
        results.append({
            "target": "PS1_Field",
            "mjd": meta["mjd"],
            "ra": meta["ra"],
            "dec": meta["dec"],
            "filter": meta["filter"],
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
