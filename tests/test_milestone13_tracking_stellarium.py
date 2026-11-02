import math
import pytest
import torch

from astro_project.core.hardware import resolve_device
from astro_project.core.wcs_utils import create_synthetic_wcs, pix_to_world
from astro_project.pipeline.tracker import predict_boundary_exit
from astro_project.core.stellarium import (
    radec_to_j2000_vector,
    slew_stellarium_to_candidate,
)


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return resolve_device("auto")


def test_predict_boundary_exit_and_neighbor_field(device):
    wcs = create_synthetic_wcs(
        pixel_scale_arcsec=1.0,
        crval_ra=180.0,
        crval_dec=45.0,
    )
    height, width = 128, 128
    x0, y0 = 110.0, 64.0
    vx, vy = 6.0, 0.0
    base_mjd = 60000.0
    lookahead_hours = 1.0

    result = predict_boundary_exit(
        wcs=wcs,
        x0=x0,
        y0=y0,
        vx=vx,
        vy=vy,
        height=height,
        width=width,
        base_mjd=base_mjd,
        lookahead_hours=lookahead_hours,
    )

    assert result["exit_edge"] == "right"
    # Distance to right boundary (width=128.0) is 18.0 pixels at 6.0 px/h -> 3.0 hours
    assert 0.0 < result["t_exit_hours"] < 4.0
    assert math.isclose(result["t_exit_hours"], (128.0 - 110.0) / 6.0, rel_tol=1e-5)
    assert math.isclose(
        result["mjd_exit"], base_mjd + result["t_exit_hours"] / 24.0, rel_tol=1e-7
    )

    expected_ra_exit, expected_dec_exit = pix_to_world(wcs, 128.0, 64.0)
    assert math.isclose(result["ra_exit_deg"], float(expected_ra_exit), rel_tol=1e-6)
    assert math.isclose(result["dec_exit_deg"], float(expected_dec_exit), rel_tol=1e-6)

    expected_ra_next, expected_dec_next = pix_to_world(
        wcs, 128.0 + vx * lookahead_hours, 64.0
    )
    assert math.isclose(result["ra_next_deg"], float(expected_ra_next), rel_tol=1e-6)
    assert math.isclose(result["dec_next_deg"], float(expected_dec_next), rel_tol=1e-6)


def test_stellarium_bridge_payload_and_fault_tolerance():
    vec = radec_to_j2000_vector(180.0, 45.0)
    assert len(vec) == 3
    norm = math.sqrt(sum(c * c for c in vec))
    assert math.isclose(norm, 1.0, rel_tol=1e-7)
    assert math.isclose(vec[0], -math.cos(math.radians(45.0)), abs_tol=1e-7)
    assert math.isclose(vec[1], 0.0, abs_tol=1e-7)
    assert math.isclose(vec[2], math.sin(math.radians(45.0)), abs_tol=1e-7)

    status = slew_stellarium_to_candidate(
        ra_deg=180.0,
        dec_deg=45.0,
        mjd=60000.0,
        base_url="http://127.0.0.1:59999/api",
        timeout=0.2,
    )
    assert isinstance(status, dict)
    assert status["ok"] is False
    assert "error" in status and status["error"]
    assert "j2000_vec" in status
    assert len(status["j2000_vec"]) == 3
