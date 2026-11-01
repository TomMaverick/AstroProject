import torch
import torch.nn.functional as F


def shift_and_stack(frames: torch.Tensor, vx: float, vy: float) -> torch.Tensor:
    """
    Shift a sequence of frames to compensate for linear motion (vx, vy) and stack them via mean.
    Strictly uses align_corners=False for photometric flux conservation.

    Args:
        frames: Tensor of shape [N, Y, X] (N_frames, Height, Width).
        vx: Velocity along the X-axis (width) in pixels/frame.
        vy: Velocity along the Y-axis (height) in pixels/frame.

    Returns:
        Stacked 2D tensor of shape [Y, X].
    """
    if frames.ndim != 3:
        raise ValueError(f"Expected 3D tensor [N, Y, X], got shape {frames.shape}")

    n_frames, height, width = frames.shape
    device = frames.device
    dtype = frames.dtype

    t = torch.arange(n_frames, dtype=dtype, device=device)

    # Shifts in pixel coordinates to map target (t=0) coordinates to source (t) coordinates:
    # A feature at (x0, y0) in the output is located at (x0 + vx*t, y0 + vy*t) in frame t.
    dx = vx * t
    dy = vy * t

    # For align_corners=False, normalized coordinates span [-1, 1] across the full extent W and H,
    # so 1 pixel corresponds to 2 / W (or 2 / H) in normalized coordinates.
    tx = 2.0 * dx / width
    ty = 2.0 * dy / height

    # Construct batch of 2x3 affine transformation matrices [N, 2, 3]
    theta = torch.zeros((n_frames, 2, 3), dtype=dtype, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    # Reshape frames to [N, C, Y, X] for grid_sample
    frames_4d = frames.unsqueeze(1)

    grid = F.affine_grid(theta, size=frames_4d.size(), align_corners=False)
    shifted_frames = F.grid_sample(
        frames_4d,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    # Remove channel dimension -> [N, Y, X] and average over frames -> [Y, X]
    stacked = shifted_frames.squeeze(1).mean(dim=0)
    return stacked
