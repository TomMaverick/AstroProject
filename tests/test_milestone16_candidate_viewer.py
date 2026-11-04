"""Tests for Milestone 16: Candidate Viewer utilities (load, stamp rendering, blinking GIF)."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from astro_project.gui.candidate_viewer import (
    create_blinking_gif,
    load_candidate_data,
    render_stamp_to_image,
)


@pytest.fixture(autouse=True)
def vram_cleanup():
    """Ensure CUDA cache is cleared after every test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_load_candidate_data_success_and_missing_files(tmp_path: Path):
    """Verify loading metadata.json and stamps.npz and raising FileNotFoundError when missing."""
    cand_dir = tmp_path / "field_001"
    cand_dir.mkdir(parents=True, exist_ok=True)

    # Missing both metadata.json and stamps.npz
    with pytest.raises(FileNotFoundError):
        load_candidate_data(cand_dir)

    rng = np.random.default_rng(42)
    frame_stamps = rng.normal(loc=100.0, scale=15.0, size=(5, 32, 32)).astype(np.float32)
    stack_stamp = rng.normal(loc=120.0, scale=8.0, size=(32, 32)).astype(np.float32)

    meta_payload = {
        "field_id": "field_001",
        "telescope": "ZTF_Palomar_48in",
        "score": 82.4,
        "tier": "gold",
        "coherence": 0.91,
        "x0": 64.0,
        "y0": 64.0,
        "vx": 1.5,
        "vy": -0.8,
    }
    (cand_dir / "metadata.json").write_text(
        json.dumps(meta_payload, indent=2), encoding="utf-8"
    )

    # Still missing stamps.npz
    with pytest.raises(FileNotFoundError):
        load_candidate_data(cand_dir)

    np.savez_compressed(
        cand_dir / "stamps.npz",
        frame_stamps=frame_stamps,
        stack_stamp=stack_stamp,
    )

    loaded = load_candidate_data(cand_dir)
    assert isinstance(loaded, dict)
    assert loaded["field_id"] == "field_001"
    assert loaded["tier"] == "gold"
    assert loaded["score"] == pytest.approx(82.4)
    assert loaded["metadata"]["telescope"] == "ZTF_Palomar_48in"
    assert loaded["frame_stamps"].shape == (5, 32, 32)
    assert loaded["stack_stamp"].shape == (32, 32)
    assert np.allclose(loaded["frame_stamps"], frame_stamps)
    assert np.allclose(loaded["stack_stamp"], stack_stamp)


def test_render_stamp_to_image_returns_valid_pil_image():
    """Verify 2D and (1, H, W) stamp arrays render to PIL Images and invalid 3D arrays raise ValueError."""
    yy, xx = np.mgrid[0:32, 0:40]
    stamp = (
        100.0 * np.exp(-((xx - 20.0) ** 2 + (yy - 16.0) ** 2) / (2.0 * 2.5**2))
    ).astype(np.float32)

    img = render_stamp_to_image(stamp, vmin_p=2.0, vmax_p=98.0)

    assert isinstance(img, Image.Image)
    # PIL size is (width, height) == (40, 32) for [Y=32, X=40]
    assert img.size == (40, 32)
    assert img.mode == "RGB"

    # Also test (1, 32, 32) singleton-channel input
    stamp_3d_singleton = np.ones((1, 32, 32), dtype=np.float32)
    img_singleton = render_stamp_to_image(stamp_3d_singleton)
    assert isinstance(img_singleton, Image.Image)
    assert img_singleton.size == (32, 32)

    # Multi-frame (5, 32, 32) must raise ValueError
    with pytest.raises(ValueError):
        render_stamp_to_image(np.ones((5, 32, 32), dtype=np.float32))


def test_create_blinking_gif_returns_valid_animated_gif_bytes():
    """Verify 3D frame stamps are encoded into valid animated GIF bytes."""
    n_frames, height, width = 5, 32, 32
    yy, xx = np.mgrid[0:height, 0:width]
    frame_stamps = np.zeros((n_frames, height, width), dtype=np.float32)

    for i in range(n_frames):
        cx = 10.0 + i * 3.0
        cy = 12.0 + i * 2.0
        frame_stamps[i] = 100.0 * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 2.0**2)
        )

    gif_bytes = create_blinking_gif(frame_stamps, duration_ms=200)

    assert isinstance(gif_bytes, bytes)
    assert len(gif_bytes) > 0
    assert gif_bytes[:6] in (b"GIF87a", b"GIF89a")

    with Image.open(BytesIO(gif_bytes)) as gif_img:
        assert gif_img.format == "GIF"
        assert getattr(gif_img, "is_animated", False) is True
        assert getattr(gif_img, "n_frames", 1) == n_frames
        assert gif_img.size == (width, height)
