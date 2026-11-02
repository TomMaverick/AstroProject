import torch
import torch.nn.functional as F

from astro_project.core.photometry import detect_sources


def _mask_2d_frame(frame: torch.Tensor, threshold: float, fwhm: float = 3.0) -> torch.Tensor:
    """Maskiert helle Quellen in einem einzelnen [Y, X] Tensor und ersetzt sie durch den Median."""
    cleaned = frame.clone()
    bg_median = torch.median(cleaned)
    residual = cleaned - bg_median

    # 1. Photutils DAOStarFinder nutzen, um Quellzentren zu identifizieren
    sources = detect_sources(residual, threshold=threshold, fwhm=fwhm)
    if sources is not None and len(sources) > 0:
        height, width = cleaned.shape
        y_coords = torch.arange(height, dtype=torch.float32, device=cleaned.device)
        x_coords = torch.arange(width, dtype=torch.float32, device=cleaned.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        mask_radius = max(2.0 * fwhm, 5.0)

        for row in sources:
            xc = float(row["x_centroid"])
            yc = float(row["y_centroid"])
            star_mask = ((grid_y - yc) ** 2 + (grid_x - xc) ** 2) <= mask_radius**2
            cleaned[star_mask] = bg_median

    # 2. Verbleibende Pixel über dem Schwellenwert ebenfalls auf den Median setzen
    high_mask = (cleaned - bg_median) > threshold
    cleaned[high_mask] = bg_median
    return cleaned


def mask_stationary_stars(
    frames: torch.Tensor, threshold: float, fwhm: float = 3.0
) -> torch.Tensor:
    """
    Erkennt helle, unbewegte Feldsterne in einem [Y, X] oder [N, Y, X] Tensor
    und ersetzt deren Pixel durch den Median-Hintergrund.
    """
    if frames.ndim == 2:
        return _mask_2d_frame(frames, threshold=threshold, fwhm=fwhm)
    elif frames.ndim == 3:
        cleaned = frames.clone()
        for i in range(cleaned.shape[0]):
            cleaned[i] = _mask_2d_frame(cleaned[i], threshold=threshold, fwhm=fwhm)
        return cleaned
    else:
        raise ValueError(f"Expected 2D [Y, X] or 3D [N, Y, X] tensor, got shape {frames.shape}")


def single_pass_wcs_kinematic_stack(
    frames: torch.Tensor,
    times_hours: torch.Tensor,
    vx: float,
    vy: float,
    wcs_offsets_xy: torch.Tensor | None = None,
    method: str = "median",
) -> torch.Tensor:
    """
    Kombiniert den translatorischen WCS-Drift-Versatz `wcs_offsets_xy` ([N, 2]) und
    die kinematische Rückverschiebung (-vx * dt, -vy * dt) in einem einzigen
    flusserhaltenden F.affine_grid / F.grid_sample (align_corners=False) Aufruf.
    """
    if frames.ndim != 3:
        raise ValueError(f"Expected 3D [N, Y, X] tensor, got shape {frames.shape}")

    n_frames, height, width = frames.shape
    device = frames.device
    dtype = frames.dtype

    t = times_hours.to(device=device, dtype=dtype)
    if t.ndim != 1 or t.shape[0] != n_frames:
        raise ValueError(f"Expected 1D times_hours of length {n_frames}, got shape {t.shape}")

    dt = t - t[0]
    dx = -vx * dt
    dy = -vy * dt

    if wcs_offsets_xy is not None:
        offsets = wcs_offsets_xy.to(device=device, dtype=dtype)
        if offsets.shape != (n_frames, 2):
            raise ValueError(
                f"Expected wcs_offsets_xy of shape ({n_frames}, 2), got {offsets.shape}"
            )
        dx = dx + offsets[:, 0]
        dy = dy + offsets[:, 1]

    tx = -2.0 * dx / float(width)
    ty = -2.0 * dy / float(height)

    theta = torch.zeros((n_frames, 2, 3), dtype=dtype, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    frames_4d = frames.unsqueeze(1)  # [N, 1, Y, X]
    grid = F.affine_grid(theta, size=frames_4d.size(), align_corners=False)
    shifted_4d = F.grid_sample(
        frames_4d,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    shifted = shifted_4d.squeeze(1)  # [N, Y, X]

    if method == "median":
        return torch.median(shifted, dim=0).values
    elif method == "mean":
        return torch.mean(shifted, dim=0)
    elif method == "sum":
        return torch.sum(shifted, dim=0)
    elif method in ("none", None):
        return shifted
    else:
        raise ValueError(f"Unknown stacking method: {method}")


def kinematic_shift_and_stack(
    frames: torch.Tensor,
    times_hours: torch.Tensor,
    vx: float,
    vy: float,
    method: str = "median",
) -> torch.Tensor:
    """
    Verschiebt einen [N, Y, X] Tensor anhand des Zeitvektors `times_hours` und der
    Geschwindigkeit (vx, vy) mittels affine_grid/grid_sample (align_corners=False)
    zurück auf die Position zur Zeit t[0] und stackt die Frames.
    """
    return single_pass_wcs_kinematic_stack(
        frames=frames,
        times_hours=times_hours,
        vx=vx,
        vy=vy,
        wcs_offsets_xy=None,
        method=method,
    )
