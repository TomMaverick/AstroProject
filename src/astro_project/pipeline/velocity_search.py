import torch

from astro_project.core.photometry import detect_sources
from astro_project.pipeline.kinematics import kinematic_shift_and_stack


def run_velocity_search(
    frames: torch.Tensor,
    times_hours: torch.Tensor,
    vx_range: tuple,
    vy_range: tuple,
    steps: int,
    detection_threshold: float,
    fwhm: float = 3.0,
) -> list[dict]:
    """
    Führt eine GPU-gestützte Geschwindigkeits-Gittersuche (Shift-and-Stack) über
    alle (vx, vy) Kombinationen im Suchbereich durch und detektiert Punktquellen.
    """
    device = frames.device
    vx_values = torch.linspace(vx_range[0], vx_range[1], steps=steps, device=device)
    vy_values = torch.linspace(vy_range[0], vy_range[1], steps=steps, device=device)

    detections: list[dict] = []

    with torch.no_grad():
        for vx_t in vx_values:
            vx = float(vx_t.item())
            for vy_t in vy_values:
                vy = float(vy_t.item())
                stacked = kinematic_shift_and_stack(
                    frames=frames,
                    times_hours=times_hours,
                    vx=vx,
                    vy=vy,
                    method="median",
                )
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
