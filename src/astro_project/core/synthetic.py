import math
from pathlib import Path
from typing import Union
import torch
from astropy.io import fits

from astro_project.core.fits_loader import save_fits_image
from astro_project.core.hardware import resolve_device
from astro_project.core.orbit import propagate_kepler_ephemeris
from astro_project.core.wcs_utils import create_synthetic_wcs, pix_to_world, world_to_pix


def generate_synthetic_sky(
    n_frames: int,
    height: int,
    width: int,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    flux: float = 1000.0,
    sigma: float = 2.0,
    noise_std: float = 10.0,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Generate a synthetic sequence of 2D sky frames [N, Y, X] containing a moving
    2D Gaussian point source (asteroid) and optional Gaussian background noise.
    """
    device = resolve_device(device)
    with torch.no_grad():
        y_coords = torch.arange(height, dtype=torch.float32, device=device)
        x_coords = torch.arange(width, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        t = torch.arange(n_frames, dtype=torch.float32, device=device).view(-1, 1, 1)
        xt = float(x0) + float(vx) * t
        yt = float(y0) + float(vy) * t

        dy = grid_y.unsqueeze(0) - yt
        dx = grid_x.unsqueeze(0) - xt

        norm = float(flux) / (2.0 * math.pi * (float(sigma) ** 2))
        signal = norm * torch.exp(-(dx**2 + dy**2) / (2.0 * (float(sigma) ** 2)))

        if noise_std > 0.0:
            noise = torch.randn(
                (n_frames, height, width), dtype=torch.float32, device=device
            ) * float(noise_std)
            return signal + noise
        return signal


def generate_demo_asteroid_series(
    output_dir: Union[str, Path],
    n_frames: int = 5,
    height: int = 128,
    width: int = 128,
    vx: float = 2.0,
    vy: float = -1.0,
    asteroid_snr: float = 1.5,
    n_stars: int = 5,
    randomize: bool = False,
    seed: int | None = None,
    device: Union[str, torch.device] = "cpu",
    pixel_scale_arcsec: float = 1.0,
    dt_hours: float = 1.0,
    orbit_preset: str = "linear",
) -> tuple[list[str], list[float], dict]:
    """
    Generates a series of `n_frames` synthetic FITS images ([Y, X] convention)
    with background noise, `n_stars` bright static field stars, and 1 faint,
    linearly moving asteroid. Saves valid FITS files with WCS and ascending
    `MJD-OBS` headers.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)

    # Clean up any leftover series_frame_*.fits from previous runs with more frames
    for old_fits in out_dir.glob("series_frame_*.fits"):
        try:
            old_fits.unlink()
        except OSError:
            pass

    cpu_gen = torch.Generator(device="cpu")
    if seed is not None:
        cpu_gen.manual_seed(int(seed))
    elif not randomize:
        cpu_gen.manual_seed(42)
    else:
        cpu_gen.seed()

    gen = torch.Generator(device=device)
    gen.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=cpu_gen).item()))

    if randomize:
        n_frames = int(torch.randint(3, 16, (1,), generator=cpu_gen).item())
        vx = float((torch.rand((1,), generator=cpu_gen).item() * 6.0) - 3.0)
        vy = float((torch.rand((1,), generator=cpu_gen).item() * 6.0) - 3.0)
        if abs(vx) < 0.4 and abs(vy) < 0.4:
            vx = 1.5
        asteroid_snr = float(1.0 + torch.rand((1,), generator=cpu_gen).item() * 4.0)
        margin_x = max(24.0, abs(vx) * n_frames + 16.0)
        margin_y = max(24.0, abs(vy) * n_frames + 16.0)
        low_x = min(margin_x, width * 0.35)
        high_x = max(width - margin_x, width * 0.65)
        low_y = min(margin_y, height * 0.35)
        high_y = max(height - margin_y, height * 0.65)
        x0 = float(low_x + torch.rand((1,), generator=cpu_gen).item() * (high_x - low_x))
        y0 = float(low_y + torch.rand((1,), generator=cpu_gen).item() * (high_y - low_y))
    else:
        n_frames = int(n_frames)
        vx = float(vx)
        vy = float(vy)
        asteroid_snr = float(asteroid_snr)
        x0 = float(width) * 0.52
        y0 = float(height) * 0.55

    pixel_scale_arcsec = max(float(pixel_scale_arcsec), 0.05)
    dt_hours = max(float(dt_hours), 0.01)
    wcs = create_synthetic_wcs(
        pixel_scale_arcsec=pixel_scale_arcsec,
        crval_ra=180.0,
        crval_dec=45.0,
    )

    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    fwhm = 3.0
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    bg_level = 100.0
    noise_std = 2.0

    asteroid_amp = max(float(asteroid_snr) * noise_std * 5.0, 6.0)

    default_stars = [
        (22.0, 26.0, 1200.0),
        (30.0, 98.0, 1450.0),
        (60.0, 32.0, 1100.0),
        (96.0, 42.0, 1350.0),
        (102.0, 100.0, 1600.0),
        (18.0, 64.0, 1250.0),
        (48.0, 82.0, 1300.0),
        (75.0, 18.0, 1150.0),
        (84.0, 108.0, 1500.0),
        (110.0, 60.0, 1400.0),
    ]

    n_stars = max(0, int(n_stars))
    if not randomize and n_stars <= len(default_stars):
        stars = default_stars[:n_stars]
    else:
        stars = []
        for idx in range(n_stars):
            if not randomize and idx < len(default_stars):
                stars.append(default_stars[idx])
            else:
                sy = float(12.0 + torch.rand((1,), generator=cpu_gen).item() * max(1.0, height - 24.0))
                sx = float(12.0 + torch.rand((1,), generator=cpu_gen).item() * max(1.0, width - 24.0))
                samp = float(1000.0 + torch.rand((1,), generator=cpu_gen).item() * 700.0)
                stars.append((sy, sx, samp))

    ra0, dec0 = pix_to_world(wcs, x0, y0)

    base_mjd = 60000.0
    file_paths: list[str] = []
    times_hours: list[float] = [float(i) * dt_hours for i in range(n_frames)]

    preset_norm = str(orbit_preset).strip().lower()
    true_orbit_info = None
    if preset_norm in ("main_belt", "nea", "hyperbolic"):
        ephem_list, true_orbit_info = propagate_kepler_ephemeris(
            times_hours=times_hours,
            orbit_preset=preset_norm,
            base_mjd=base_mjd,
            ra0_deg=float(ra0),
            dec0_deg=float(dec0),
        )
        ephem_ra = [ep["ra_deg"] for ep in ephem_list]
        ephem_dec = [ep["dec_deg"] for ep in ephem_list]
        px_arr, py_arr = world_to_pix(wcs, ephem_ra, ephem_dec)
        x_traj = [float(v) for v in px_arr]
        y_traj = [float(v) for v in py_arr]
        total_dt = max(times_hours[-1] - times_hours[0], 1e-6) if len(times_hours) > 1 else 1.0
        vx = float((x_traj[-1] - x_traj[0]) / total_dt)
        vy = float((y_traj[-1] - y_traj[0]) / total_dt)
        x0 = float(x_traj[0])
        y0 = float(y_traj[0])
    else:
        x_traj = [x0 + vx * t_hr for t_hr in times_hours]
        y_traj = [y0 + vy * t_hr for t_hr in times_hours]

    with torch.no_grad():
        for i in range(n_frames):
            t_hr = times_hours[i]

            noise = torch.randn(
                (height, width), generator=gen, dtype=torch.float32, device=device
            ) * noise_std
            frame = torch.full((height, width), bg_level, dtype=torch.float32, device=device) + noise

            for sy, sx, samp in stars:
                frame += samp * torch.exp(
                    -((grid_y - sy) ** 2 + (grid_x - sx) ** 2) / (2.0 * sigma**2)
                )

            xi = x_traj[i]
            yi = y_traj[i]
            frame += asteroid_amp * torch.exp(
                -((grid_y - yi) ** 2 + (grid_x - xi) ** 2) / (2.0 * sigma**2)
            )

            fits_path = out_dir / f"series_frame_{i:02d}.fits"
            save_fits_image(frame, str(fits_path), wcs=wcs)

            mjd_obs = base_mjd + (t_hr / 24.0)
            with fits.open(str(fits_path), mode="update", memmap=False) as hdul:
                hdul[0].header["MJD-OBS"] = (mjd_obs, "Modified Julian Date of observation")
                hdul[0].header["EXPTIME"] = (60.0, "Exposure time in seconds")
                hdul[0].header["PIXSCALE"] = (pixel_scale_arcsec, "Pixel scale in arcsec/px")
                hdul[0].header["ORBPRES"] = (preset_norm, "Synthetic orbit preset")
                hdul.flush()

            file_paths.append(str(fits_path))

    ground_truth = {
        "n_frames": int(n_frames),
        "x0": x0,
        "y0": y0,
        "ra0": float(ra0),
        "dec0": float(dec0),
        "vx": float(vx),
        "vy": float(vy),
        "asteroid_snr": float(asteroid_snr),
        "asteroid_amp": asteroid_amp,
        "noise_std": noise_std,
        "n_stars": len(stars),
        "stars": stars,
        "randomized": bool(randomize),
        "pixel_scale_arcsec": float(pixel_scale_arcsec),
        "dt_hours": float(dt_hours),
        "orbit_preset": preset_norm,
        "true_orbit": true_orbit_info,
    }
    return file_paths, times_hours, ground_truth


def generate_custom_asteroid_series(
    output_dir: Union[str, Path],
    n_frames: int = 5,
    height: int = 128,
    width: int = 128,
    vx: float = 2.0,
    vy: float = -1.0,
    asteroid_snr: float = 1.5,
    n_stars: int = 5,
    randomize: bool = False,
    seed: int | None = None,
    device: Union[str, torch.device] = "cpu",
    pixel_scale_arcsec: float = 1.0,
    dt_hours: float = 1.0,
    orbit_preset: str = "linear",
) -> tuple[list[str], list[float], dict]:
    """Wrapper for custom/randomized synthetic asteroid series generation."""
    return generate_demo_asteroid_series(
        output_dir=output_dir,
        n_frames=n_frames,
        height=height,
        width=width,
        vx=vx,
        vy=vy,
        asteroid_snr=asteroid_snr,
        n_stars=n_stars,
        randomize=randomize,
        seed=seed,
        device=device,
        pixel_scale_arcsec=pixel_scale_arcsec,
        dt_hours=dt_hours,
        orbit_preset=orbit_preset,
    )
