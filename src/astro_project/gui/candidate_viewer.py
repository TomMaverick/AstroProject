"""Utilities for loading and visualizing saved Gold/Silver tier candidate artifacts."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_candidate_data(candidate_dir: str | Path) -> dict:
    """Loads candidate metadata and stamp arrays from a candidate directory.

    Parameters
    ----------
    candidate_dir : str | Path
        Path to the candidate folder containing ``metadata.json`` and ``stamps.npz``.

    Returns
    -------
    dict
        Dictionary containing ``metadata`` (as well as top-level metadata keys),
        ``frame_stamps``, and ``stack_stamp``.

    Raises
    ------
    FileNotFoundError
        If ``metadata.json`` or ``stamps.npz`` does not exist in ``candidate_dir``.
    """
    cand_path = Path(candidate_dir)
    meta_path = cand_path / "metadata.json"
    stamps_path = cand_path / "stamps.npz"

    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing candidate metadata file: {meta_path}")
    if not stamps_path.is_file():
        raise FileNotFoundError(f"Missing candidate stamps file: {stamps_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with np.load(stamps_path) as data:
        if "frame_stamps" not in data or "stack_stamp" not in data:
            raise KeyError(
                f"Expected 'frame_stamps' and 'stack_stamp' in {stamps_path}"
            )
        frame_stamps = np.asarray(data["frame_stamps"], dtype=np.float32)
        stack_stamp = np.asarray(data["stack_stamp"], dtype=np.float32)

    result = dict(metadata) if isinstance(metadata, dict) else {}
    result["metadata"] = metadata
    result["frame_stamps"] = frame_stamps
    result["stack_stamp"] = stack_stamp
    return result


def render_stamp_to_image(
    stamp_array: np.ndarray,
    vmin_p: float = 2.0,
    vmax_p: float = 98.0,
    cmap_name: str = "viridis",
) -> Image.Image:
    """Renders a 2D stamp array into an RGB PIL Image using percentile scaling.

    Parameters
    ----------
    stamp_array : np.ndarray
        2D numpy array ``[H, W]`` representing the stamp flux.
    vmin_p : float, optional
        Lower percentile for contrast clipping, by default 2.0.
    vmax_p : float, optional
        Upper percentile for contrast clipping, by default 98.0.
    cmap_name : str, optional
        Matplotlib colormap name, by default "viridis".

    Returns
    -------
    PIL.Image.Image
        Rendered RGB PIL Image of size ``(W, H)``.
    """
    arr = np.asarray(stamp_array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D stamp_array [H, W], got shape {arr.shape}")

    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    vmin = float(np.percentile(clean, vmin_p))
    vmax = float(np.percentile(clean, vmax_p))

    if vmax > vmin:
        norm = np.clip((clean - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        norm = np.zeros_like(clean, dtype=np.float32)

    uint8_scaled = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(uint8_scaled)
    rgb_uint8 = np.clip(rgba[..., :3] * 255.0, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(rgb_uint8, mode="RGB")


def create_blinking_gif(
    frame_stamps: np.ndarray,
    duration_ms: int = 200,
    vmin_p: float = 2.0,
    vmax_p: float = 98.0,
    cmap_name: str = "viridis",
) -> bytes:
    """Creates an animated blinking GIF from a 3D stack of frame stamps.

    Parameters
    ----------
    frame_stamps : np.ndarray
        3D numpy array ``[N, H, W]`` of sequential frame stamps.
    duration_ms : int, optional
        Frame duration in milliseconds, by default 200.
    vmin_p : float, optional
        Lower percentile for contrast clipping, by default 2.0.
    vmax_p : float, optional
        Upper percentile for contrast clipping, by default 98.0.
    cmap_name : str, optional
        Matplotlib colormap name, by default "viridis".

    Returns
    -------
    bytes
        Animated GIF file bytes.
    """
    frames_arr = np.asarray(frame_stamps, dtype=np.float32)
    if frames_arr.ndim != 3 or frames_arr.shape[0] == 0:
        raise ValueError(
            f"Expected 3D frame_stamps [N, H, W] with N > 0, got shape {frames_arr.shape}"
        )

    pil_frames = [
        render_stamp_to_image(
            frame, vmin_p=vmin_p, vmax_p=vmax_p, cmap_name=cmap_name
        )
        for frame in frames_arr
    ]

    buffer = BytesIO()
    pil_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(duration_ms),
        loop=0,
    )
    return buffer.getvalue()
