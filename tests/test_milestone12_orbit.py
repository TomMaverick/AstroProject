import pytest
import torch
from astropy.table import Table

from astro_project.core.hardware import resolve_device
from astro_project.core.wcs_utils import create_synthetic_wcs
from astro_project.core.orbit import (
    compute_tracklet_endpoints,
    estimate_kepler_orbit,
    format_mpc80_line,
)
from astro_project.core.catalog_match import cross_match_known_objects


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return resolve_device("auto")


def test_tracklet_and_kepler_iod(device):
    wcs = create_synthetic_wcs(pixel_scale_arcsec=15.0, crval_ra=180.0, crval_dec=0.0)
    times_hours = torch.tensor([0.0, 1.5, 3.0], dtype=torch.float32, device=device)

    tracklet_epochs = compute_tracklet_endpoints(
        wcs=wcs,
        x0=100.0,
        y0=100.0,
        vx=-2.0,
        vy=0.5,
        times_hours=times_hours,
        base_mjd=60000.0,
    )

    assert len(tracklet_epochs) == 3
    for ep in tracklet_epochs:
        assert "mjd" in ep
        assert "ra_deg" in ep
        assert "dec_deg" in ep

    orbit = estimate_kepler_orbit(tracklet_epochs)
    assert isinstance(orbit, dict)
    for key in ("a_au", "e", "i_deg", "orbit_class"):
        assert key in orbit
    assert orbit["a_au"] > 1.0
    assert 0.0 <= orbit["e"] < 1.0
    assert 0.0 <= orbit["i_deg"] <= 180.0
    assert isinstance(orbit["orbit_class"], str) and len(orbit["orbit_class"]) > 0


def test_cross_match_catalog():
    # Candidate 1 is ~1.0 arcsec away from a known object; Candidate 2 is isolated.
    candidates = Table(
        {
            "ra": [180.0 + (1.0 / 3600.0), 181.0],
            "dec": [45.0, 45.0],
        }
    )
    known_catalog = Table(
        {
            "id": ["(433) Eros", "(1) Ceres"],
            "ra": [180.0, 185.0],
            "dec": [45.0, 46.0],
        }
    )

    matched = cross_match_known_objects(
        candidates=candidates,
        known_catalog=known_catalog,
        max_sep_arcsec=5.0,
    )

    assert "match_status" in matched.colnames
    assert "matched_id" in matched.colnames
    assert "sep_arcsec" in matched.colnames

    assert matched["match_status"][0] == "KNOWN"
    assert matched["matched_id"][0] == "(433) Eros"
    assert matched["sep_arcsec"][0] < 1.5

    assert matched["match_status"][1] == "UNKNOWN CANDIDATE"


def test_mpc80_export():
    line = format_mpc80_line(
        designation="K26A00A",
        mjd=60000.12345,
        ra_deg=180.5,
        dec_deg=45.2,
        mag=20.5,
        obs_code="500",
    )
    assert isinstance(line, str)
    assert len(line) == 80
    assert "K26A00A" in line[:12]
    assert line[14] == "C"
    assert line[77:80] == "500"
