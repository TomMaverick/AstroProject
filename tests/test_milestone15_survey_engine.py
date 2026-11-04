from pathlib import Path

import numpy as np
import pytest
import torch

from astro_project.core.hardware import resolve_device
from astro_project.pipeline.data_fetcher import (
    SURVEY_TELESCOPE_PRESETS,
    generate_or_fetch_survey_fields,
)
from astro_project.pipeline.survey_engine import (
    SurveyLedger,
    compute_discovery_score,
    estimate_velocity_grid_from_wcs,
    run_automated_survey_batch,
)
from astro_project.core.wcs_utils import create_synthetic_wcs


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return resolve_device("auto")


def test_survey_ledger_deduplication_and_stats(tmp_path: Path):
    """Test 1: Verify 64-bit Blake2b field hash deduplication and summary statistics."""
    ledger_db = tmp_path / "survey_ledger.db"
    ledger = SurveyLedger(ledger_path=ledger_db)

    h1 = ledger.compute_field_hash("ZTF_Palomar_48in", 180.1234, 45.6789, 60000.0, 60000.2)
    h1_same = ledger.compute_field_hash("ZTF_Palomar_48in", 180.1241, 45.6791, 60000.001, 60000.201)
    h2_diff = ledger.compute_field_hash("PanSTARRS_PS1", 180.1234, 45.6789, 60000.0, 60000.2)

    assert isinstance(h1, int)
    assert h1 == h1_same
    assert h1 != h2_diff
    assert not ledger.is_scanned(h1)

    ledger.record_scan(
        field_hash=h1,
        telescope="ZTF_Palomar_48in",
        ra_deg=180.12,
        dec_deg=45.68,
        mjd_start=60000.0,
        mjd_end=60000.2,
        n_frames=5,
        status="gold",
        best_score=82.5,
    )
    ledger.record_scan(
        field_hash=h2_diff,
        telescope="PanSTARRS_PS1",
        ra_deg=180.12,
        dec_deg=45.68,
        mjd_start=60000.0,
        mjd_end=60000.2,
        n_frames=5,
        status="purged",
        best_score=12.0,
    )

    assert ledger.is_scanned(h1)
    assert ledger.is_scanned(h2_diff)

    stats = ledger.get_summary_stats()
    assert stats["total_scanned"] == 2
    assert stats["gold_count"] == 1
    assert stats["empty_purged"] == 1
    assert stats["silver_count"] == 0


def test_compute_discovery_score_coherence_gating():
    """Test 2: Coherent multi-frame sub-sigma track scores much higher than a 1-frame cosmic ray."""
    n_frames, h, w = 5, 32, 32

    # Coherent source: steady flux of 20.0 in the central 5x5 across all 5 frames
    coherent_stamps = np.zeros((n_frames, h, w), dtype=np.float32)
    coherent_stamps[:, 14:19, 14:19] = 20.0
    coherent_stack = np.mean(coherent_stamps, axis=0)

    # Cosmic ray spike: 100.0 in frame 0 only, 0.0 in frames 1..4 (same mean flux = 20.0)
    cosmic_stamps = np.zeros((n_frames, h, w), dtype=np.float32)
    cosmic_stamps[0, 14:19, 14:19] = 100.0
    cosmic_stack = np.mean(cosmic_stamps, axis=0)

    candidate_row = {"peak": 25.0, "cnn_score": 0.90}
    orbit_info = {"a_au": 2.65, "e": 0.15}

    res_coherent = compute_discovery_score(
        candidate_row=candidate_row,
        frame_stamps=coherent_stamps,
        stack_stamp=coherent_stack,
        detection_threshold=10.0,
        orbit_info=orbit_info,
        is_known_object=False,
    )
    res_cosmic = compute_discovery_score(
        candidate_row=candidate_row,
        frame_stamps=cosmic_stamps,
        stack_stamp=cosmic_stack,
        detection_threshold=10.0,
        orbit_info=orbit_info,
        is_known_object=False,
    )

    assert res_coherent["coherence"] >= 0.95
    assert res_cosmic["coherence"] < 0.10
    assert res_coherent["score"] >= 75.0
    assert res_coherent["tier"] == "gold"
    assert res_cosmic["score"] < 45.0
    assert res_cosmic["tier"] == "trash"


def test_automated_survey_batch_tiered_storage_and_auto_purge(tmp_path: Path, device: torch.device):
    """Test 3: End-to-end SGT survey batch with 1 empty field (purged) and 1 asteroid field (saved)."""
    assert "ZTF_Palomar_48in" in SURVEY_TELESCOPE_PRESETS
    wcs_test = create_synthetic_wcs(pixel_scale_arcsec=2.0)
    vx_r, vy_r = estimate_velocity_grid_from_wcs(wcs_test, max_rate_arcsec_hr=36.0)
    assert vx_r == (-18.0, 18.0)
    assert vy_r == (-18.0, 18.0)

    base_dir = tmp_path / "survey_data"
    field_batches = generate_or_fetch_survey_fields(
        base_data_dir=base_dir,
        telescope="ZTF_Palomar_48in",
        n_fields=2,
        n_frames=5,
        height=96,
        width=96,
        active_field_indices=[1],
        asteroid_snr=6.0,
        vx=2.0,
        vy=-1.0,
        seed=42,
        device=device,
    )

    assert len(field_batches) == 2
    field_empty = field_batches[0]
    field_active = field_batches[1]
    assert not field_empty["has_ground_truth"]
    assert field_empty["ground_truth"]["asteroid_snr"] == 0.0
    assert field_empty["ground_truth"]["asteroid_amp"] == 0.0
    assert field_active["has_ground_truth"]
    assert field_active["ground_truth"]["asteroid_snr"] == 6.0

    summary = run_automated_survey_batch(
        field_batches=field_batches,
        base_data_dir=base_dir,
        device=device,
        steps=5,
        single_frame_mask_sigma=8.0,
        stack_detection_sigma=3.0,
        cnn_threshold=0.0,
        min_keep_score=45.0,
        auto_purge=True,
        vx_range=(1.0, 3.0),
        vy_range=(-2.0, 0.0),
    )

    assert summary["scanned"] == 2
    assert summary["skipped_cached"] == 0
    assert summary["empty_purged"] >= 1
    assert (summary["gold_saved"] + summary["silver_saved"]) >= 1

    # Verify Field A (empty) staging FITS files were auto-purged
    for fpath in field_empty["file_paths"]:
        assert not Path(fpath).exists()

    # Verify Field B (asteroid) produced candidate artifacts (.npz stamps + mpc80.txt + metadata.json)
    gold_dir = base_dir / "candidates" / "gold_tier" / field_active["field_id"]
    silver_dir = base_dir / "candidates" / "silver_tier" / field_active["field_id"]
    saved_dir = gold_dir if gold_dir.exists() else silver_dir
    assert saved_dir.exists()
    assert (saved_dir / "stamps.npz").is_file()
    assert (saved_dir / "mpc80.txt").is_file()
    assert (saved_dir / "metadata.json").is_file()

    # Re-running the same batch must skip both fields via SQLite SurveyLedger
    summary_second = run_automated_survey_batch(
        field_batches=field_batches,
        base_data_dir=base_dir,
        device=device,
        steps=5,
        auto_purge=True,
    )
    assert summary_second["skipped_cached"] == 2
    assert summary_second["scanned"] == 0
