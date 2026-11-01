import math
from typing import Union
import torch


def generate_synthetic_sky(
    n_frames: int,
    height: int,
    width: int,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    flux: float,
    sigma: float,
    noise_std: float = 0.0,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Generate a synthetic sequence of 2D sky frames with a linearly moving Gaussian point source.
    Follows the [Y, X] (height, width) axis convention.
    """
    device = torch.device(device)

    # Coordinate grids in [Y, X] convention
    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Reshape grids for broadcasting across frames: [1, Y, X]
    grid_y = grid_y.unsqueeze(0)
    grid_x = grid_x.unsqueeze(0)

    # Time steps t = [0, 1, ..., n_frames - 1] shaped as [N, 1, 1]
    t = torch.arange(n_frames, dtype=torch.float32, device=device).view(-1, 1, 1)

    # Source positions at each frame t
    x_t = x0 + vx * t
    y_t = y0 + vy * t

    # Normalized 2D Gaussian PSF
    norm_factor = flux / (2.0 * math.pi * (sigma**2))
    exponent = -((grid_x - x_t) ** 2 + (grid_y - y_t) ** 2) / (2.0 * (sigma**2))
    frames = norm_factor * torch.exp(exponent)

    if noise_std > 0.0:
        noise = torch.randn_like(frames) * noise_std
        frames = frames + noise

    return frames
