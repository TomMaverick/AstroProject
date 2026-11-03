import argparse
from pathlib import Path
from typing import Sequence

import torch

from astro_project.core.classifier import RealBogusCNN
from astro_project.core.fits_loader import load_fits_image
from astro_project.core.hardware import resolve_device
from astro_project.core.orbit import (
    compute_tracklet_endpoints,
    estimate_kepler_orbit,
    export_tracklet_mpc80,
)
from astro_project.core.wcs_utils import create_synthetic_wcs
from astro_project.pipeline.orchestration import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for headless batch execution."""
    parser = argparse.ArgumentParser(
        description="Headless GPU batch pipeline for asteroid detection and MPC80 export."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing sequential FITS frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory where MPC 80-column .txt reports will be saved.",
    )
    parser.add_argument(
        "--vx-min",
        type=float,
        default=-2.0,
        help="Minimum horizontal search velocity vx [px/h].",
    )
    parser.add_argument(
        "--vx-max",
        type=float,
        default=2.0,
        help="Maximum horizontal search velocity vx [px/h].",
    )
    parser.add_argument(
        "--vy-min",
        type=float,
        default=-2.0,
        help="Minimum vertical search velocity vy [px/h].",
    )
    parser.add_argument(
        "--vy-max",
        type=float,
        default=2.0,
        help="Maximum vertical search velocity vy [px/h].",
    )
    parser.add_argument(
        "--grid-steps",
        type=int,
        default=5,
        help="Number of velocity grid steps per axis.",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=8.0,
        help="DAOStarFinder detection threshold in ADU above background.",
    )
    parser.add_argument(
        "--cnn-threshold",
        type=float,
        default=0.2,
        help="Minimum RealBogusCNN probability threshold [0.0, 1.0].",
    )
    return parser


def run_batch_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    vx_min: float = -2.0,
    vx_max: float = 2.0,
    vy_min: float = -2.0,
    vy_max: float = 2.0,
    grid_steps: int = 5,
    detection_threshold: float = 8.0,
    cnn_threshold: float = 0.2,
    dt_hours_fallback: float = 1.0,
) -> dict:
    """Execute the asteroid detection pipeline in headless mode and export MPC80 reports."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fits_files = sorted(
        [
            str(p)
            for p in in_path.glob("*")
            if p.is_file() and p.suffix.lower() in (".fits", ".fit", ".fts")
        ]
    )
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in '{in_path}'.")

    try:
        device = resolve_device("cuda")
    except Exception:
        device = resolve_device("auto")

    print(
        f"\U0001f680 Starting batch pipeline on {len(fits_files)} FITS files "
        f"(device={device})..."
    )

    report_files: list[str] = []
    with torch.no_grad():
        mjd_days_list: list[float] = []
        ref_wcs = None
        for idx, fpath in enumerate(fits_files):
            _, wcs_obj, hdr = load_fits_image(fpath, device=device, nan_fill=None)
            if ref_wcs is None and wcs_obj is not None and wcs_obj.is_celestial:
                ref_wcs = wcs_obj
            if hdr is not None and "MJD-OBS" in hdr:
                mjd_days_list.append(float(hdr["MJD-OBS"]))
            else:
                mjd_days_list.append(60000.0 + (float(idx) * float(dt_hours_fallback)) / 24.0)

        base_mjd = mjd_days_list[0]
        times_hours = torch.tensor(
            [(m - base_mjd) * 24.0 for m in mjd_days_list],
            dtype=torch.float32,
            device=device,
        )

        cnn_model = RealBogusCNN().to(device)
        cnn_model.eval()

        catalog = run_pipeline(
            file_paths=fits_files,
            times_hours=times_hours,
            vx_range=(float(vx_min), float(vx_max)),
            vy_range=(float(vy_min), float(vy_max)),
            steps=int(grid_steps),
            detection_threshold=float(detection_threshold),
            cnn_model=cnn_model,
            device=device,
            star_mask_threshold=150.0,
            cnn_threshold=float(cnn_threshold),
        )

        orbit_wcs = ref_wcs if ref_wcs is not None else create_synthetic_wcs()
        times_list = times_hours.detach().cpu().tolist()

        if len(catalog) > 0:
            catalog.sort("peak", reverse=True)
            for cand_idx, row in enumerate(catalog):
                x0 = float(row["x_centroid"])
                y0 = float(row["y_centroid"])
                vx = float(row["vx"])
                vy = float(row["vy"])
                designation = f"K26A{cand_idx:02d}A"[:7]

                epochs = compute_tracklet_endpoints(
                    wcs=orbit_wcs,
                    x0=x0,
                    y0=y0,
                    vx=vx,
                    vy=vy,
                    times_hours=times_list,
                    base_mjd=base_mjd,
                )
                orbit_info = estimate_kepler_orbit(epochs)
                mpc_txt = export_tracklet_mpc80(
                    epochs,
                    designation=designation,
                    mag=20.0,
                    band="V",
                    obs_code="500",
                )

                out_file = out_path / f"candidate_{cand_idx:03d}_{designation}_mpc80.txt"
                out_file.write_text(mpc_txt + "\n", encoding="utf-8")
                report_files.append(str(out_file))

                if cand_idx == 0:
                    print(
                        f"\u2705 Top candidate {designation}: vx={vx:.2f} px/h, vy={vy:.2f} px/h, "
                        f"a={orbit_info['a_au']:.2f} AU, e={orbit_info['e']:.2f} "
                        f"-> saved to {out_file}"
                    )
        else:
            print("\u26a0\ufe0f No candidates detected above threshold.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"\u2705 Batch pipeline finished: {len(catalog)} candidate(s), "
        f"{len(report_files)} MPC80 file(s) written to '{out_path}'."
    )
    return {
        "catalog": catalog,
        "report_files": report_files,
        "device": device,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for headless batch processing."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    run_batch_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vx_min=args.vx_min,
        vx_max=args.vx_max,
        vy_min=args.vy_min,
        vy_max=args.vy_max,
        grid_steps=args.grid_steps,
        detection_threshold=args.detection_threshold,
        cnn_threshold=args.cnn_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
