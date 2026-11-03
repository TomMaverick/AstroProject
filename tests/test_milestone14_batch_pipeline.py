from pathlib import Path

import pytest
import torch

from astro_project.core.hardware import resolve_device
from astro_project.core.synthetic import generate_demo_asteroid_series
from run_batch_pipeline import main as run_batch_cli
from run_batch_pipeline import run_batch_pipeline


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return resolve_device("auto")


def test_batch_pipeline_generates_mpc80_report(tmp_path: Path, device: torch.device):
    """SGT test: Generate a synthetic FITS series and verify headless batch MPC80 export."""
    input_dir = tmp_path / "fits_input"
    output_dir = tmp_path / "batch_output"

    with torch.no_grad():
        paths, times, gt = generate_demo_asteroid_series(
            output_dir=input_dir,
            n_frames=5,
            height=128,
            width=128,
            vx=2.0,
            vy=-1.0,
            asteroid_snr=4.0,
            n_stars=4,
            randomize=False,
            seed=42,
            device=device,
        )

    assert len(paths) == 5

    result = run_batch_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        vx_min=1.0,
        vx_max=3.0,
        vy_min=-2.0,
        vy_max=0.0,
        grid_steps=5,
        detection_threshold=6.0,
        cnn_threshold=0.0,
    )

    assert len(result["catalog"]) > 0
    assert len(result["report_files"]) >= 1

    report_path = Path(result["report_files"][0])
    assert report_path.is_file()
    assert report_path.suffix == ".txt"

    lines = [line for line in report_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 5
    for line in lines:
        assert len(line) == 80


def test_batch_pipeline_cli_entrypoint(tmp_path: Path, device: torch.device):
    """Verify the CLI argument parser and entrypoint execution end-to-end."""
    input_dir = tmp_path / "cli_input"
    output_dir = tmp_path / "cli_output"

    with torch.no_grad():
        generate_demo_asteroid_series(
            output_dir=input_dir,
            n_frames=5,
            height=128,
            width=128,
            vx=2.0,
            vy=-1.0,
            asteroid_snr=4.0,
            n_stars=3,
            randomize=False,
            seed=42,
            device=device,
        )

    exit_code = run_batch_cli(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--vx-min",
            "1.0",
            "--vx-max",
            "3.0",
            "--vy-min",
            "-2.0",
            "--vy-max",
            "0.0",
            "--grid-steps",
            "5",
            "--detection-threshold",
            "6.0",
            "--cnn-threshold",
            "0.0",
        ]
    )

    assert exit_code == 0
    txt_files = list(output_dir.glob("*.txt"))
    assert len(txt_files) >= 1
    first_lines = [
        line for line in txt_files[0].read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(first_lines) == 5
    assert all(len(line) == 80 for line in first_lines)
