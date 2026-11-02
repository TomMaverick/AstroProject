import math
from pathlib import Path
from typing import Union
import torch
from astropy.io import fits

from astro_project.core.fits_loader import save_fits_image
from astro_project.core.wcs_utils import create_synthetic_wcs, pix_to_world


def generate_synthetic_sky(
    n_frames: int, height: int, width: int, x0: float, y0: float, 
    vx: float, vy: float, flux: float, sigma: float, noise_std: float = 0.0, 
    device: Union[str, torch.device] = "cpu"
) -> torch.Tensor:
    device = torch.device(device)
    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_y, grid_x = grid_y.unsqueeze(0), grid_x.unsqueeze(0)
    t = torch.arange(n_frames, dtype=torch.float32, device=device).view(-1, 1, 1)
    x_t, y_t = x0 + vx * t, y0 + vy * t
    norm_factor = flux / (2.0 * math.pi * (sigma**2))
    frames = norm_factor * torch.exp(-((grid_x - x_t)**2 + (grid_y - y_t)**2) / (2.0 * sigma**2))
    if noise_std > 0.0:
        frames += torch.randn_like(frames) * noise_std
    return frames


def generate_demo_asteroid_series(
    output_dir: Union[str, Path],
    n_frames: int = 5,
    height: int = 128,
    width: int = 128,
    vx: float = 2.0,
    vy: float = -1.0,
    asteroid_snr: float = 1.5,
    device: Union[str, torch.device] = "cpu",
) -> tuple[list[str], list[float], dict]:
    """
    Erzeugt eine Serie von `n_frames` synthetischen FITS-Bildern ([Y, X] Konvention)
    mit Hintergrundrauschen, 5 hellen statischen Feldsternen und 1 lichtschwachen,
    sich linear bewegenden Asteroiden. Speichert valide FITS-Dateien mit WCS und
    aufsteigendem `MJD-OBS` Header.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(42)

    wcs = create_synthetic_wcs(pixel_scale_arcsec=1.0, crval_ra=180.0, crval_dec=45.0)

    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    fwhm = 3.0
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    bg_level = 100.0
    noise_std = 2.0

    # Integrierter PSF-Skalierungsfaktor (~8.0 ADU pro SNR-Einheit bei noise_std=2.0),
    # sodass der Asteroid im Einzelbild im Rauschen bleibt, aber im 5-Frame-Stack klar hervortritt.
    asteroid_amp = max(float(asteroid_snr) * noise_std * 5.0, 10.0)

    # 5 feste statische Feldsterne als (y, x, amplitude) Tupel in [Y, X] Konvention
    stars = [
        (22.0, 26.0, 1200.0),
        (30.0, 98.0, 1450.0),
        (60.0, 32.0, 1100.0),
        (96.0, 42.0, 1350.0),
        (102.0, 100.0, 1600.0),
    ]

    x0 = float(width) * 0.52
    y0 = float(height) * 0.55
    ra0, dec0 = pix_to_world(wcs, x0, y0)

    base_mjd = 60000.0
    file_paths: list[str] = []
    times_hours: list[float] = []

    with torch.no_grad():
        for i in range(n_frames):
            t_hr = float(i)
            times_hours.append(t_hr)

            noise = torch.randn(
                (height, width), generator=gen, dtype=torch.float32, device=device
            ) * noise_std
            frame = torch.full((height, width), bg_level, dtype=torch.float32, device=device) + noise

            # Statische Feldsterne addieren
            for sy, sx, samp in stars:
                frame += samp * torch.exp(
                    -((grid_y - sy) ** 2 + (grid_x - sx) ** 2) / (2.0 * sigma**2)
                )

            # Bewegter Asteroid zur Zeit t_hr
            xi = x0 + vx * t_hr
            yi = y0 + vy * t_hr
            frame += asteroid_amp * torch.exp(
                -((grid_y - yi) ** 2 + (grid_x - xi) ** 2) / (2.0 * sigma**2)
            )

            fits_path = out_dir / f"series_frame_{i:02d}.fits"
            save_fits_image(frame, str(fits_path), wcs=wcs)

            # MJD-OBS und EXPTIME in den FITS-Header schreiben
            mjd_obs = base_mjd + (t_hr / 24.0)
            with fits.open(str(fits_path), mode="update", memmap=False) as hdul:
                hdul[0].header["MJD-OBS"] = (mjd_obs, "Modified Julian Date of observation")
                hdul[0].header["EXPTIME"] = (60.0, "Exposure time in seconds")
                hdul.flush()

            file_paths.append(str(fits_path))

    ground_truth = {
        "x0": x0,
        "y0": y0,
        "ra0": float(ra0),
        "dec0": float(dec0),
        "vx": float(vx),
        "vy": float(vy),
        "asteroid_snr": float(asteroid_snr),
        "asteroid_amp": asteroid_amp,
        "noise_std": noise_std,
        "stars": stars,
    }
    return file_paths, times_hours, ground_truth
