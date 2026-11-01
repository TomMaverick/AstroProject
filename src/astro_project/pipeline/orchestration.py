import numpy as np
import torch
import torch.nn as nn
from astropy.table import Table

from astro_project.core.classifier import extract_stamps
from astro_project.core.fits_loader import load_fits_image
from astro_project.core.photometry import estimate_background
from astro_project.core.wcs_utils import sources_to_skycoords
from astro_project.pipeline.kinematics import (
    kinematic_shift_and_stack,
    mask_stationary_stars,
)
from astro_project.pipeline.velocity_search import run_velocity_search


def run_pipeline(
    file_paths: list[str],
    times_hours: torch.Tensor,
    vx_range: tuple,
    vy_range: tuple,
    steps: int,
    detection_threshold: float,
    cnn_model: nn.Module = None,
    device: torch.device | str = None,
    star_mask_threshold: float = 200.0,
    cnn_threshold: float = 0.5,
    fwhm: float = 3.0,
) -> Table:
    """
    End-to-End Pipeline zur Detektion bewegter Objekte (Asteroiden) aus FITS-Zeitserien:
    1. FITS-Dateien inkl. WCS laden
    2. 2D-Hintergrund schätzen und abziehen sowie statische Feldsterne maskieren
    3. GPU Velocity Grid Search (Shift-and-Stack) ausführen
    4. Stamps extrahieren und optional mit RealBogusCNN klassifizieren/filtern
    5. Pixelkoordinaten über WCS in RA/Dec (ICRS) umrechnen und als Astropy Table exportieren
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if len(file_paths) == 0:
        raise ValueError("file_paths must not be empty")

    cleaned_frames = []
    reference_wcs = None
    header_times = []

    # a) & b) FITS-Bilder laden, Hintergrund abziehen und Feldsterne maskieren
    for idx, path in enumerate(file_paths):
        img_tensor, wcs, header = load_fits_image(path, device=device)
        if idx == 0:
            reference_wcs = wcs
        if header is not None and "MJD-OBS" in header:
            header_times.append(float(header["MJD-OBS"]) * 24.0)

        bkg = estimate_background(img_tensor)
        bg_subtracted = img_tensor - bkg
        masked = mask_stationary_stars(
            bg_subtracted, threshold=star_mask_threshold, fwhm=fwhm
        )
        cleaned_frames.append(masked)

    frames = torch.stack(cleaned_frames, dim=0)  # [N, Y, X]

    if times_hours is None:
        if len(header_times) == len(file_paths):
            times_tensor = torch.tensor(header_times, dtype=torch.float32, device=device)
        else:
            raise ValueError("times_hours must be provided if MJD-OBS is absent from FITS headers")
    else:
        times_tensor = times_hours.to(device=device, dtype=torch.float32)

    # c) GPU Velocity Grid Search
    raw_candidates = run_velocity_search(
        frames=frames,
        times_hours=times_tensor,
        vx_range=vx_range,
        vy_range=vy_range,
        steps=steps,
        detection_threshold=detection_threshold,
        fwhm=fwhm,
    )

    colnames = ("x_centroid", "y_centroid", "ra", "dec", "vx", "vy", "flux", "peak", "cnn_score")
    if len(raw_candidates) == 0:
        return Table(
            names=colnames,
            dtype=(float, float, float, float, float, float, float, float, float),
        )

    # d) Stamp-Extraktion und optionale CNN-Klassifikation unter torch.no_grad()
    filtered_candidates = []
    if cnn_model is not None:
        cnn_model = cnn_model.to(device)
        cnn_model.eval()

    with torch.no_grad():
        stack_cache: dict[tuple[float, float], torch.Tensor] = {}
        for cand in raw_candidates:
            vel_key = (cand["vx"], cand["vy"])
            if vel_key not in stack_cache:
                stack_cache[vel_key] = kinematic_shift_and_stack(
                    frames=frames,
                    times_hours=times_tensor,
                    vx=cand["vx"],
                    vy=cand["vy"],
                    method="median",
                )
            stacked_img = stack_cache[vel_key]
            stamps = extract_stamps(
                stacked_img,
                [(cand["x_centroid"], cand["y_centroid"])],
                stamp_size=32,
            )

            if cnn_model is not None:
                score = float(cnn_model(stamps)[0, 0].item())
                if score < cnn_threshold:
                    continue
            else:
                score = 1.0

            cand_with_score = dict(cand)
            cand_with_score["cnn_score"] = score
            filtered_candidates.append(cand_with_score)

    if len(filtered_candidates) == 0:
        return Table(
            names=colnames,
            dtype=(float, float, float, float, float, float, float, float, float),
        )

    # e) & f) Pixel-Zentren über WCS in Himmelskoordinaten (RA, Dec) transformieren und Tabelle bauen
    source_table = Table(
        {
            "x_centroid": np.array([c["x_centroid"] for c in filtered_candidates], dtype=np.float64),
            "y_centroid": np.array([c["y_centroid"] for c in filtered_candidates], dtype=np.float64),
        }
    )
    sky_coords = sources_to_skycoords(source_table, reference_wcs)

    catalog = Table(
        {
            "x_centroid": source_table["x_centroid"],
            "y_centroid": source_table["y_centroid"],
            "ra": np.asarray(sky_coords.ra.deg, dtype=np.float64),
            "dec": np.asarray(sky_coords.dec.deg, dtype=np.float64),
            "vx": np.array([c["vx"] for c in filtered_candidates], dtype=np.float64),
            "vy": np.array([c["vy"] for c in filtered_candidates], dtype=np.float64),
            "flux": np.array([c["flux"] for c in filtered_candidates], dtype=np.float64),
            "peak": np.array([c["peak"] for c in filtered_candidates], dtype=np.float64),
            "cnn_score": np.array([c["cnn_score"] for c in filtered_candidates], dtype=np.float64),
        }
    )
    return catalog
