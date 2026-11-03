import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import torch
from astropy.wcs.utils import proj_plane_pixel_scales

from astro_project.core.classifier import RealBogusCNN, extract_stamps
from astro_project.core.fits_loader import load_fits_image
from astro_project.core.hardware import resolve_device
from astro_project.core.orbit import (
    compute_tracklet_endpoints,
    estimate_kepler_orbit,
    export_tracklet_mpc80,
)
from astro_project.core.wcs_utils import compute_wcs_frame_offsets, create_synthetic_wcs
from astro_project.pipeline.kinematics import (
    mask_stationary_stars,
    single_pass_wcs_kinematic_stack,
)
from astro_project.pipeline.orchestration import run_pipeline
from astro_project.pipeline.velocity_search import _shift_frames_along_trajectory


class SurveyLedger:
    """Compact SQLite ledger tracking scanned sky fields via 64-bit Blake2b hashes."""

    def __init__(self, ledger_path: str | Path = "data/survey_ledger.db"):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.ledger_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scanned_fields (
                    field_hash INTEGER PRIMARY KEY,
                    telescope TEXT NOT NULL,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                    mjd_start REAL NOT NULL,
                    mjd_end REAL NOT NULL,
                    n_frames INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    best_score REAL NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def compute_field_hash(
        telescope: str,
        ra_deg: float,
        dec_deg: float,
        mjd_start: float,
        mjd_end: float,
    ) -> int:
        """Compute a signed 64-bit integer hash from rounded coordinates (0.01 deg) and MJD (0.01 d)."""
        key_str = (
            f"{str(telescope).strip().upper()}|"
            f"{round(float(ra_deg), 2):.2f}|"
            f"{round(float(dec_deg), 2):.2f}|"
            f"{round(float(mjd_start), 2):.2f}|"
            f"{round(float(mjd_end), 2):.2f}"
        )
        digest = hashlib.blake2b(key_str.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=True)

    def is_scanned(self, field_hash: int) -> bool:
        """Return True if the given 64-bit field hash is already recorded in the ledger."""
        with sqlite3.connect(self.ledger_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM scanned_fields WHERE field_hash = ? LIMIT 1",
                (int(field_hash),),
            ).fetchone()
            return row is not None

    def record_scan(
        self,
        field_hash: int,
        telescope: str,
        ra_deg: float,
        dec_deg: float,
        mjd_start: float,
        mjd_end: float,
        n_frames: int,
        status: str,
        best_score: float,
    ) -> None:
        """Insert or update a field scan record in the SQLite ledger."""
        with sqlite3.connect(self.ledger_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scanned_fields
                (field_hash, telescope, ra_deg, dec_deg, mjd_start, mjd_end, n_frames, status, best_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(field_hash),
                    str(telescope),
                    float(ra_deg),
                    float(dec_deg),
                    float(mjd_start),
                    float(mjd_end),
                    int(n_frames),
                    str(status),
                    float(best_score),
                ),
            )
            conn.commit()

    def get_summary_stats(self) -> dict:
        """Return aggregate ledger statistics across all scanned fields."""
        with sqlite3.connect(self.ledger_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM scanned_fields").fetchone()[0]
            purged = conn.execute(
                "SELECT COUNT(*) FROM scanned_fields WHERE status IN ('purged', 'empty_purged')"
            ).fetchone()[0]
            silver = conn.execute(
                "SELECT COUNT(*) FROM scanned_fields WHERE status = 'silver'"
            ).fetchone()[0]
            gold = conn.execute(
                "SELECT COUNT(*) FROM scanned_fields WHERE status = 'gold'"
            ).fetchone()[0]
        return {
            "total_scanned": int(total),
            "empty_purged": int(purged),
            "silver_count": int(silver),
            "gold_count": int(gold),
        }


def compute_discovery_score(
    candidate_row: dict,
    frame_stamps: np.ndarray,
    stack_stamp: np.ndarray,
    detection_threshold: float,
    orbit_info: dict,
    is_known_object: bool = False,
) -> dict:
    """
    Compute a 0..100 discovery score with temporal coherence gating to reject single-frame
    cosmic rays and hotpixels.
    """
    stamps_arr = np.asarray(frame_stamps, dtype=np.float32).copy()
    stack_arr = np.asarray(stack_stamp, dtype=np.float32).copy()
    is_noise_fluctuation = False

    if stamps_arr.ndim == 3 and stamps_arr.shape[0] > 0:
        n_f, h, w = stamps_arr.shape
        cy, cx = h // 2, w // 2
        y0, y1 = max(0, cy - 2), min(h, cy + 3)
        x0, x1 = max(0, cx - 2), min(w, cx + 3)
        py0, py1 = max(0, cy - 1), min(h, cy + 2)
        px0, px1 = max(0, cx - 1), min(w, cx + 2)

        outer_mask = np.ones((h, w), dtype=bool)
        outer_mask[max(0, cy - 3) : min(h, cy + 4), max(0, cx - 3) : min(w, cx + 4)] = False

        for i in range(n_f):
            bg_med_i = float(np.median(stamps_arr[i][outer_mask]))
            stamps_arr[i] -= bg_med_i

        outer_vals = stamps_arr[:, outer_mask]
        bg_mad = float(np.median(np.abs(outer_vals - np.median(outer_vals))))
        bg_std = 1.4826 * bg_mad

        frame_fluxes = np.mean(stamps_arr[:, py0:py1, px0:px1], axis=(1, 2))
        frame_peaks = np.max(stamps_arr[:, py0:py1, px0:px1], axis=(1, 2))
        pos_fluxes = np.maximum(frame_fluxes, 0.0)
        median_flux = float(np.median(pos_fluxes))
        mean_flux = float(np.mean(pos_fluxes))

        coherence = float(np.clip(median_flux / max(mean_flux, 1e-6), 0.0, 1.0))

        stack_outer = stack_arr[outer_mask]
        stack_bg_med = float(np.median(stack_outer))
        stack_net_peak = float(np.max(stack_arr[y0:y1, x0:x1]) - stack_bg_med)

        if bg_std > 1e-4:
            stack_bg_std = max(
                1.4826 * float(np.median(np.abs(stack_outer - stack_bg_med))),
                bg_std / np.sqrt(max(n_f, 1)),
            )
            stack_snr = stack_net_peak / max(stack_bg_std, 1e-6)

            active_fraction = float(np.mean(frame_fluxes > 0.7 * bg_std))
            median_frame_snr = float(median_flux / max(bg_std, 1e-6))

            if (
                stack_snr < 4.5
                or active_fraction < 0.6
                or median_frame_snr < 0.8
                or mean_flux < 0.8 * bg_std
            ):
                is_noise_fluctuation = True
                coherence *= 0.25
    else:
        coherence = 0.5

    peak_val = float(candidate_row.get("peak", 0.0))
    cnn_prob = float(candidate_row.get("cnn_score", 0.5))

    snr_ratio = peak_val / max(float(detection_threshold), 1e-6)
    if snr_ratio < 1.05:
        is_noise_fluctuation = True

    snr_score = float(np.clip(snr_ratio * 17.5, 0.0, 35.0))
    cnn_score_pts = float(np.clip(cnn_prob * 40.0, 0.0, 40.0))

    a_au = float(orbit_info.get("a_au", float("nan"))) if orbit_info else float("nan")
    if np.isfinite(a_au) and (a_au > 0.3 or a_au < -0.1):
        orbit_pts = 15.0
    else:
        orbit_pts = 5.0

    novelty_pts = 4.0 if is_known_object else 10.0

    raw_score = (snr_score + cnn_score_pts) * (0.3 + 0.7 * coherence) + orbit_pts + novelty_pts
    if coherence < 0.35 or is_noise_fluctuation:
        raw_score = min(raw_score * 0.4, 30.0)

    final_score = float(np.clip(raw_score, 0.0, 100.0))
    if final_score >= 75.0:
        tier = "gold"
    elif final_score >= 45.0:
        tier = "silver"
    else:
        tier = "trash"

    return {
        "score": final_score,
        "tier": tier,
        "coherence": coherence,
    }


def estimate_velocity_grid_from_wcs(
    wcs,
    max_rate_arcsec_hr: float = 36.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Estimate adaptive pixel velocity search bounds ((-v_max, v_max), (-v_max, v_max)) from WCS."""
    pix_scale = 1.0
    if wcs is not None:
        try:
            scales_deg = proj_plane_pixel_scales(wcs.celestial)
            pix_scale = float(np.mean(scales_deg) * 3600.0)
        except Exception:
            pix_scale = 1.0
    if not np.isfinite(pix_scale) or pix_scale <= 0.0:
        pix_scale = 1.0

    v_max = float(np.clip(float(max_rate_arcsec_hr) / pix_scale, 2.0, 45.0))
    return ((-v_max, v_max), (-v_max, v_max))


def _purge_files(file_paths: list[str]) -> None:
    """Delete staging FITS files and remove empty parent directory if applicable."""
    parent_dirs = set()
    for fpath in file_paths:
        p = Path(fpath)
        parent_dirs.add(p.parent)
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass
    for d in parent_dirs:
        if d.is_dir() and not any(d.iterdir()):
            try:
                d.rmdir()
            except OSError:
                pass


def run_automated_survey_batch(
    field_batches: list[dict],
    base_data_dir: str | Path = "data",
    device: str | torch.device = "cpu",
    steps: int = 7,
    single_frame_mask_sigma: float = 4.5,
    stack_detection_sigma: float = 4.0,
    stack_Resource_sigma: float | None = None,
    cnn_threshold: float = 0.2,
    min_keep_score: float = 45.0,
    auto_purge: bool = True,
    vx_range: tuple[float, float] | None = None,
    vy_range: tuple[float, float] | None = None,
) -> dict:
    """
    Process a batch of survey fields with deduplication via SurveyLedger, GPU velocity search,
    coherence-gated discovery scoring, and 3-tier storage management.
    """
    if stack_Resource_sigma is not None:
        stack_detection_sigma = float(stack_Resource_sigma)

    base_path = Path(base_data_dir)
    ledger = SurveyLedger(ledger_path=base_path / "survey_ledger.db")
    resolved_dev = resolve_device(device)

    scanned = 0
    skipped_cached = 0
    empty_purged = 0
    gold_saved = 0
    silver_saved = 0
    candidates_list: list[dict] = []

    with torch.no_grad():
        cnn_model = RealBogusCNN().to(resolved_dev)
        cnn_model.eval()

        for batch in field_batches:
            field_id = str(batch.get("field_id", "unknown_field"))
            telescope = str(batch.get("telescope", "ZTF_Palomar_48in"))
            file_paths = [str(p) for p in batch.get("file_paths", []) if Path(p).is_file()]
            if not file_paths:
                skipped_cached += 1
                continue

            frames_list: list[torch.Tensor] = []
            wcs_list = []
            mjd_list: list[float] = []
            ra_deg = 180.0
            dec_deg = 45.0

            for idx, fpath in enumerate(file_paths):
                img, wcs_obj, hdr = load_fits_image(fpath, device=resolved_dev, nan_fill=0.0)
                frames_list.append(img)
                wcs_list.append(wcs_obj)
                if hdr is not None:
                    if "MJD-OBS" in hdr:
                        mjd_list.append(float(hdr["MJD-OBS"]))
                    else:
                        mjd_list.append(60000.0 + idx * (0.5 / 24.0))
                    if idx == 0:
                        ra_deg = float(hdr.get("CRVAL1", ra_deg))
                        dec_deg = float(hdr.get("CRVAL2", dec_deg))
                else:
                    mjd_list.append(60000.0 + idx * (0.5 / 24.0))

            mjd_start = min(mjd_list)
            mjd_end = max(mjd_list)
            field_hash = ledger.compute_field_hash(
                telescope=telescope,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                mjd_start=mjd_start,
                mjd_end=mjd_end,
            )

            if ledger.is_scanned(field_hash):
                skipped_cached += 1
                continue

            scanned += 1
            frames_tensor = torch.stack(frames_list, dim=0)
            n_frames, height, width = frames_tensor.shape

            # Robust noise estimation via MAD on first frame
            sample_frame = frames_tensor[0]
            med_val = torch.median(sample_frame)
            mad_val = torch.median(torch.abs(sample_frame - med_val))
            noise_std = max(float(mad_val.item()) * 1.4826, 1.0)

            star_mask_thresh = max(float(single_frame_mask_sigma) * noise_std, 150.0)
            det_thresh = (
                float(stack_detection_sigma) * (noise_std / np.sqrt(max(n_frames, 1)))
            )
            pipeline_det_thresh = det_thresh

            ref_wcs = wcs_list[0] if wcs_list[0] is not None else create_synthetic_wcs()
            if vx_range is None or vy_range is None:
                est_vx, est_vy = estimate_velocity_grid_from_wcs(ref_wcs)
                use_vx = vx_range if vx_range is not None else est_vx
                use_vy = vy_range if vy_range is not None else est_vy
            else:
                use_vx = vx_range
                use_vy = vy_range

            times_hours = torch.tensor(
                [(m - mjd_start) * 24.0 for m in mjd_list],
                dtype=torch.float32,
                device=resolved_dev,
            )

            catalog = run_pipeline(
                file_paths=file_paths,
                times_hours=times_hours,
                vx_range=use_vx,
                vy_range=use_vy,
                steps=int(steps),
                detection_threshold=pipeline_det_thresh,
                cnn_model=cnn_model,
                device=resolved_dev,
                star_mask_threshold=star_mask_thresh,
                cnn_threshold=float(cnn_threshold),
            )

            best_cand = None
            best_score = 0.0

            if len(catalog) > 0:
                wcs_offsets = None
                if all(w is not None and w.is_celestial for w in wcs_list):
                    wcs_offsets = compute_wcs_frame_offsets(
                        wcs_list, ref_wcs, height, width, device=resolved_dev
                    )

                masked_frames = mask_stationary_stars(
                    frames_tensor, threshold=star_mask_thresh
                )
                times_list = times_hours.detach().cpu().tolist()
                for cand_idx, row in enumerate(catalog):
                    x0 = float(row["x_centroid"])
                    y0 = float(row["y_centroid"])
                    vx = float(row["vx"])
                    vy = float(row["vy"])
                    peak_val = float(row["peak"])
                    cnn_prob = float(row["cnn_score"])

                    shifted = _shift_frames_along_trajectory(
                        masked_frames,
                        times_hours,
                        vx,
                        vy,
                        wcs_offsets_xy=wcs_offsets,
                    )
                    stack_img = single_pass_wcs_kinematic_stack(
                        masked_frames,
                        times_hours,
                        vx,
                        vy,
                        wcs_offsets_xy=wcs_offsets,
                        method="median",
                    )

                    frame_stamps_list = [
                        extract_stamps(shifted[f_i], [(x0, y0)], stamp_size=32)[0]
                        .detach()
                        .cpu()
                        .numpy()
                        for f_i in range(n_frames)
                    ]
                    frame_stamps_np = np.stack(frame_stamps_list, axis=0)
                    stack_stamp_np = (
                        extract_stamps(stack_img, [(x0, y0)], stamp_size=32)[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    epochs = compute_tracklet_endpoints(
                        wcs=ref_wcs,
                        x0=x0,
                        y0=y0,
                        vx=vx,
                        vy=vy,
                        times_hours=times_list,
                        base_mjd=mjd_start,
                    )
                    orbit_info = estimate_kepler_orbit(epochs)
                    score_info = compute_discovery_score(
                        candidate_row={"peak": peak_val, "cnn_score": cnn_prob},
                        frame_stamps=frame_stamps_np,
                        stack_stamp=stack_stamp_np,
                        detection_threshold=det_thresh,
                        orbit_info=orbit_info,
                        is_known_object=False,
                    )

                    if score_info["score"] > best_score:
                        best_score = score_info["score"]
                        best_cand = {
                            "field_id": field_id,
                            "telescope": telescope,
                            "x0": x0,
                            "y0": y0,
                            "vx": vx,
                            "vy": vy,
                            "peak": peak_val,
                            "cnn_score": cnn_prob,
                            "score": score_info["score"],
                            "tier": score_info["tier"],
                            "coherence": score_info["coherence"],
                            "orbit_info": orbit_info,
                            "epochs": epochs,
                            "frame_stamps_np": frame_stamps_np,
                            "stack_stamp_np": stack_stamp_np,
                        }

            if best_cand is not None and best_cand["score"] >= float(min_keep_score):
                tier_label = "gold" if best_cand["score"] >= 75.0 else "silver"
                tier_folder = "gold_tier" if tier_label == "gold" else "silver_tier"
                dest_dir = base_path / "candidates" / tier_folder / field_id
                dest_dir.mkdir(parents=True, exist_ok=True)

                np.savez_compressed(
                    dest_dir / "stamps.npz",
                    frame_stamps=best_cand["frame_stamps_np"],
                    stack_stamp=best_cand["stack_stamp_np"],
                )

                mpc_txt = export_tracklet_mpc80(
                    best_cand["epochs"],
                    designation="K26S00A",
                    mag=20.5,
                    band="V",
                    obs_code="500",
                )
                (dest_dir / "mpc80.txt").write_text(mpc_txt + "\n", encoding="utf-8")

                meta_payload = {
                    "field_id": field_id,
                    "field_hash": field_hash,
                    "telescope": telescope,
                    "score": best_cand["score"],
                    "tier": tier_label,
                    "coherence": best_cand["coherence"],
                    "x0": best_cand["x0"],
                    "y0": best_cand["y0"],
                    "vx": best_cand["vx"],
                    "vy": best_cand["vy"],
                    "peak": best_cand["peak"],
                    "cnn_score": best_cand["cnn_score"],
                    "orbit": best_cand["orbit_info"],
                }
                (dest_dir / "metadata.json").write_text(
                    json.dumps(meta_payload, indent=2), encoding="utf-8"
                )

                if tier_label == "gold":
                    for fpath in file_paths:
                        src_p = Path(fpath)
                        if src_p.is_file():
                            shutil.copy2(src_p, dest_dir / src_p.name)
                    gold_saved += 1
                else:
                    silver_saved += 1

                if auto_purge:
                    _purge_files(file_paths)

                ledger.record_scan(
                    field_hash=field_hash,
                    telescope=telescope,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    mjd_start=mjd_start,
                    mjd_end=mjd_end,
                    n_frames=n_frames,
                    status=tier_label,
                    best_score=best_cand["score"],
                )
                candidates_list.append(meta_payload)
            else:
                if auto_purge:
                    _purge_files(file_paths)
                empty_purged += 1
                ledger.record_scan(
                    field_hash=field_hash,
                    telescope=telescope,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    mjd_start=mjd_start,
                    mjd_end=mjd_end,
                    n_frames=n_frames,
                    status="empty_purged",
                    best_score=best_score,
                )

    return {
        "scanned": scanned,
        "skipped_cached": skipped_cached,
        "empty_purged": empty_purged,
        "gold_saved": gold_saved,
        "gold_Saved": gold_saved,
        "silver_saved": silver_saved,
        "candidates_list": candidates_list,
        "ledger_stats": ledger.get_summary_stats(),
    }
