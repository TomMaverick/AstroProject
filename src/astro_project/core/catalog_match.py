import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table


def _find_col(table: Table, candidates: tuple[str, ...]) -> str:
    for col in candidates:
        if col in table.colnames:
            return col
    raise KeyError(f"None of the columns {candidates} found in table {table.colnames}")


def cross_match_known_objects(
    candidates: Table,
    known_catalog: Table,
    max_sep_arcsec: float = 5.0,
) -> Table:
    """
    Cross-match a candidate Astropy Table against a reference catalog of known objects
    using SkyCoord.match_to_catalog_sky. Adds `match_status`, `matched_id`, and `sep_arcsec`.
    """
    result = candidates.copy()

    if len(result) == 0:
        result["match_status"] = np.array([], dtype=str)
        result["matched_id"] = np.array([], dtype=str)
        result["sep_arcsec"] = np.array([], dtype=np.float64)
        return result

    cand_ra_col = _find_col(result, ("ra", "ra_deg", "RA"))
    cand_dec_col = _find_col(result, ("dec", "dec_deg", "DEC"))

    if known_catalog is None or len(known_catalog) == 0:
        result["match_status"] = ["UNKNOWN CANDIDATE"] * len(result)
        result["matched_id"] = [""] * len(result)
        result["sep_arcsec"] = [float("inf")] * len(result)
        return result

    cat_ra_col = _find_col(known_catalog, ("ra", "ra_deg", "RA"))
    cat_dec_col = _find_col(known_catalog, ("dec", "dec_deg", "DEC"))
    id_col = None
    for candidate_id_col in ("id", "name", "designation", "object_id"):
        if candidate_id_col in known_catalog.colnames:
            id_col = candidate_id_col
            break

    cand_coords = SkyCoord(
        ra=np.asarray(result[cand_ra_col], dtype=np.float64) * u.deg,
        dec=np.asarray(result[cand_dec_col], dtype=np.float64) * u.deg,
        frame="icrs",
    )
    cat_coords = SkyCoord(
        ra=np.asarray(known_catalog[cat_ra_col], dtype=np.float64) * u.deg,
        dec=np.asarray(known_catalog[cat_dec_col], dtype=np.float64) * u.deg,
        frame="icrs",
    )

    idx, d2d, _ = cand_coords.match_to_catalog_sky(cat_coords)
    sep_arcsec = d2d.to(u.arcsec).value

    statuses: list[str] = []
    matched_ids: list[str] = []
    for i, sep in enumerate(sep_arcsec):
        if float(sep) <= float(max_sep_arcsec):
            statuses.append("KNOWN")
            if id_col is not None:
                matched_ids.append(str(known_catalog[id_col][int(idx[i])]))
            else:
                matched_ids.append(f"OBJ_{int(idx[i])}")
        else:
            statuses.append("UNKNOWN CANDIDATE")
            matched_ids.append("")

    result["match_status"] = statuses
    result["matched_id"] = matched_ids
    result["sep_arcsec"] = sep_arcsec.astype(np.float64)
    return result
