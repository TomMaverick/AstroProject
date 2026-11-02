import math
import warnings
import torch
import torch.nn.functional as F
from photutils.detection.daofinder import NoDetectionsWarning

from astro_project.core.photometry import detect_sources
from astro_project.pipeline.kinematics import single_pass_wcs_kinematic_stack


def _build_gaussian_psf_kernel(
    fwhm: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Builds a normalized 2D Gaussian PSF kernel of shape [1, 1, K, K] following [Y, X] convention.
    """
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    radius = max(2, int(math.ceil(3.0 * sigma)))
    size = 2 * radius + 1

    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(gy**2 + gx**2) / (2.0 * sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel.unsqueeze(0).unsqueeze(0)


def compute_psi_phi_frames(
    frames: torch.Tensor,
    fwhm: float = 3.0,
    variances: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-convolves a stack of frames [N, Y, X] with a Gaussian PSF into KBMOD-style
    Psi and Phi images on the GPU:
      Psi_i = (I_i / sigma_i^2) * PSF
      Phi_i = (1 / sigma_i^2) * (PSF^2)
    """
    if frames.ndim != 3:
        raise ValueError(f"Expected 3D tensor [N, Y, X], got shape {frames.shape}")

    device = frames.device
    dtype = frames.dtype
    n_frames, height, width = frames.shape

    with torch.no_grad():
        if variances is None:
            # Robust variance estimation via Median Absolute Deviation (MAD) per frame
            flat = frames.reshape(n_frames, -1)
            med = torch.median(flat, dim=1, keepdim=True).values
            mad = torch.median(torch.abs(flat - med), dim=1).values
            sigma_noise = torch.clamp(1.4826 * mad, min=1e-3)
            var_per_frame = (sigma_noise**2).view(n_frames, 1, 1)
        else:
            var_per_frame = torch.clamp(variances.to(device=device, dtype=dtype), min=1e-6)
            if var_per_frame.ndim == 1:
                var_per_frame = var_per_frame.view(n_frames, 1, 1)

        psf_kernel = _build_gaussian_psf_kernel(fwhm=fwhm, device=device, dtype=dtype)
        psf_sq_kernel = psf_kernel**2
        pad = psf_kernel.shape[-1] // 2

        inv_var_frames = frames / var_per_frame  # [N, Y, X]
        inv_var_map = torch.ones_like(frames) / var_per_frame  # [N, Y, X]

        psi = F.conv2d(inv_var_frames.unsqueeze(1), psf_kernel, padding=pad).squeeze(1)
        phi = F.conv2d(inv_var_map.unsqueeze(1), psf_sq_kernel, padding=pad).squeeze(1)

    return psi, phi


def _shift_frames_along_trajectory(
    frames: torch.Tensor,
    times_hours: torch.Tensor,
    vx: float,
    vy: float,
    wcs_offsets_xy: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Shifts a [N, Y, X] tensor back to t_0 along velocity (vx, vy) using
    align_corners=False subpixel sampling.
    """
    n_frames, height, width = frames.shape
    device = frames.device
    dtype = frames.dtype

    t = times_hours.to(device=device, dtype=dtype)
    dt = t - t[0]

    dx_shift = -vx * dt
    dy_shift = -vy * dt
    if wcs_offsets_xy is not None:
        offsets = wcs_offsets_xy.to(device=device, dtype=dtype)
        dx_shift = dx_shift + offsets[:, 0]
        dy_shift = dy_shift + offsets[:, 1]

    tx = -2.0 * dx_shift / float(width)
    ty = -2.0 * dy_shift / float(height)

    theta = torch.zeros((n_frames, 2, 3), dtype=dtype, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    frames_4d = frames.unsqueeze(1)
    grid = F.affine_grid(theta, size=frames_4d.shape, align_corners=False)
    shifted_4d = F.grid_sample(
        frames_4d,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return shifted_4d.squeeze(1)


def trimmed_likelihood_stack(
    psi: torch.Tensor,
    phi: torch.Tensor,
    times_hours: torch.Tensor,
    vx: float,
    vy: float,
    trim_fraction: float = 0.2,
    wcs_offsets_xy: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Shifts pre-convolved Psi and Phi stacks along (vx, vy) and computes the
    trimmed-likelihood SNR map: sum(Psi) / sqrt(sum(Phi)), rejecting outlier frames.
    """
    with torch.no_grad():
        shifted_psi = _shift_frames_along_trajectory(psi, times_hours, vx=vx, vy=vy, wcs_offsets_xy=wcs_offsets_xy)
        shifted_phi = _shift_frames_along_trajectory(phi, times_hours, vx=vx, vy=vy, wcs_offsets_xy=wcs_offsets_xy)

        n_frames = shifted_psi.shape[0]
        k_trim = int(math.floor(n_frames * trim_fraction)) if trim_fraction > 0.0 else 0

        if n_frames >= 5 and k_trim > 0 and (n_frames - 2 * k_trim) >= 3:
            # Sort per-pixel along the time axis to trim lowest and highest outliers
            sorted_idx = torch.argsort(shifted_psi, dim=0)
            keep_idx = sorted_idx[k_trim : n_frames - k_trim]
            psi_kept = torch.gather(shifted_psi, dim=0, index=keep_idx)
            phi_kept = torch.gather(shifted_phi, dim=0, index=keep_idx)
            psi_sum = torch.sum(psi_kept, dim=0)
            phi_sum = torch.sum(phi_kept, dim=0)
        else:
            psi_sum = torch.sum(shifted_psi, dim=0)
            phi_sum = torch.sum(shifted_phi, dim=0)

        snr_map = psi_sum / torch.sqrt(torch.clamp(phi_sum, min=1e-8))
        return snr_map


def run_velocity_search(
    frames: torch.Tensor,
    times_hours: torch.Tensor,
    vx_range: tuple,
    vy_range: tuple,
    steps: int,
    detection_threshold: float,
    fwhm: float = 3.0,
    use_psi_phi: bool = False,
    trim_fraction: float = 0.2,
    wcs_offsets_xy: torch.Tensor | None = None,
) -> list[dict]:
    """
    Executes a GPU-accelerated velocity grid search (Shift-and-Stack or Psi/Phi
    trimmed-likelihood search) over all (vx, vy) combinations and detects point sources.
    """
    device = frames.device
    vx_values = torch.linspace(vx_range[0], vx_range[1], steps=steps, device=device)
    vy_values = torch.linspace(vy_range[0], vy_range[1], steps=steps, device=device)

    detections: list[dict] = []

    with torch.no_grad():
        psi, phi = (None, None)
        if use_psi_phi:
            psi, phi = compute_psi_phi_frames(frames, fwhm=fwhm)

        for vx_t in vx_values:
            vx = float(vx_t.item())
            for vy_t in vy_values:
                vy = float(vy_t.item())
                if use_psi_phi and psi is not None and phi is not None:
                    stacked = trimmed_likelihood_stack(
                        psi=psi,
                        phi=phi,
                        times_hours=times_hours,
                        vx=vx,
                        vy=vy,
                        trim_fraction=trim_fraction,
                        wcs_offsets_xy=wcs_offsets_xy,
                    )
                else:
                    stacked = single_pass_wcs_kinematic_stack(
                        frames=frames,
                        times_hours=times_hours,
                        vx=vx,
                        vy=vy,
                        wcs_offsets_xy=wcs_offsets_xy,
                        method="median",
                    )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NoDetectionsWarning)
                    sources = detect_sources(
                        stacked,
                        threshold=detection_threshold,
                        fwhm=fwhm,
                    )
                if sources is not None and len(sources) > 0:
                    for row in sources:
                        detections.append(
                            {
                                "x_centroid": float(row["x_centroid"]),
                                "y_centroid": float(row["y_centroid"]),
                                "vx": vx,
                                "vy": vy,
                                "flux": float(row["flux"]) if "flux" in row.colnames else 0.0,
                                "peak": float(row["peak"]) if "peak" in row.colnames else 0.0,
                            }
                        )

    return detections
