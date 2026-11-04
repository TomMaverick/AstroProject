import logging
from typing import Sequence
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Observations

logger = logging.getLogger(__name__)

def query_ps1_archive(ra: float, dec: float, mjd_start: float, mjd_end: float, radius_arcmin: float = 20.0) -> list[dict]:
    """
    Queries the MAST archive for Pan-STARRS1 epochs within a given region and time range.
    """
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    
    try:
        obs_table = Observations.query_region(coord, radius=radius_arcmin * u.arcmin)
    except Exception as e:
        logger.error(f"Failed to query MAST archive: {e}")
        return []
        
    logger.info(f"Found {len(obs_table)} raw entries in MAST before filtering.")
    
    results = []
    for row in obs_table:
        if row.get("obs_collection") != "PS1":
            continue
            
        mjd = row.get("t_min")
        if mjd is None or not (mjd_start <= mjd <= mjd_end):
            continue
            
        data_url = row.get("dataURL")
        if not data_url:
            data_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={row.get('obs_id', '')}"
            
        results.append({
            "target": row.get("target_name", "PS1_Field"),
            "mjd": mjd,
            "ra": row.get("s_ra", ra),
            "dec": row.get("s_dec", dec),
            "fits_url": data_url
        })
        
    # Sort chronologically by MJD
    results.sort(key=lambda x: x["mjd"])
    return results

def download_ps1_batch(epochs: Sequence[dict], dest_dir: str = "data/staging") -> list[str]:
    """
    Dummy implementation for downloading a batch of PS1 FITS files.
    Replace this with actual logic to download the FITS files from URLs.
    """
    # Placeholder return value
    return [
        f"{dest_dir}/ps1_fits_001.fits",
        f"{dest_dir}/ps1_fits_002.fits"
    ]
