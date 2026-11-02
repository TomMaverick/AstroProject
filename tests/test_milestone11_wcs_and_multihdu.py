import numpy as np
import pytest
import torch
from astropy.io import fits

from astro_project.core.fits_loader import load_fits_image
from astro_project.core.synthetic import generate_synthetic_sky
from astro_project.core.wcs_utils import (
    compute_wcs_frame_offsets,
    create_synthetic_wcs,
    pix_to_world,
    world_to_pix,
)
from astro_project.pipeline.kinematics import single_pass_wcs_kinematic_stack


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def test_multi_hdu_and_compressed_fits_loader(tmp_path, device):
    """
    Testet das Laden von Multi-Extension FITS-Dateien (inkl. .fits.fz / CompImageHDU),
    bei denen HDU 0 leer ist und HDU 1 ein 3D-Array (1, Y, X) mit NaNs und WCS enthält.
    """
    height, width = 64, 96
    wcs_gt = create_synthetic_wcs(pixel_scale_arcsec=1.5, crval_ra=210.0, crval_dec=15.0)
    header = wcs_gt.to_header()

    raw_3d = np.ones((1, height, width), dtype=np.float32) * 42.0
    raw_3d[0, 10, 20] = np.nan
    raw_3d[0, 30, 40] = np.inf

    # 1. Multi-Extension FITS mit leerer PrimaryHDU und ImageHDU in Extension 1
    multi_fits_path = tmp_path / "multi_ext.fits"
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=raw_3d, header=header),
        ]
    )
    hdul.writeto(multi_fits_path, overwrite=True)

    tensor, wcs, _ = load_fits_image(multi_fits_path, device=device)
    assert tensor.shape == (height, width)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == device.type
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()
    assert torch.isclose(tensor[10, 20], torch.tensor(0.0, device=device))
    assert torch.isclose(tensor[0, 0], torch.tensor(42.0, device=device))
    assert wcs.is_celestial

    # 2. Komprimierte FITS-Datei (.fits.fz) mit CompImageHDU in Extension 1
    raw_2d = raw_3d[0].copy()
    comp_fits_path = tmp_path / "compressed.fits.fz"
    hdul_comp = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.CompImageHDU(data=raw_2d, header=header),
        ]
    )
    hdul_comp.writeto(comp_fits_path, overwrite=True)

    tensor_fz, wcs_fz, _ = load_fits_image(comp_fits_path, device=device)
    assert tensor_fz.shape == (height, width)
    assert tensor_fz.device.type == device.type
    assert not torch.isnan(tensor_fz).any()
    assert wcs_fz.is_celestial


def test_single_pass_wcs_and_kinematic_stack(device):
    """
    Verifiziert die kombinierte Single-Pass WCS-Drift- und Kinematik-Korrektur
    auf synthetischen Ground-Truth-Daten bzgl. Peak-Zentrierung und Flusserhaltung.
    """
    n_frames = 4
    height, width = 96, 96
    times_hours = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32, device=device)

    x0, y0 = 40.0, 52.0
    vx, vy = 1.25, -0.75
    flux = 2000.0
    sigma = 2.0

    reference_wcs = create_synthetic_wcs(pixel_scale_arcsec=1.0, crval_ra=180.0, crval_dec=45.0)
    wcs_list = []
    frames_list = []

    for i in range(n_frames):
        dt = float(times_hours[i].item())
        dx_drift = 1.5 * i
        dy_drift = -1.0 * i

        # Erzeuge ein gedriftetes WCS für Frame i (CRPIX liegt bei 0-basiertem Index 50.0)
        ra_crval_i, dec_crval_i = pix_to_world(
            reference_wcs, 50.0 - dx_drift, 50.0 - dy_drift
        )
        wcs_i = create_synthetic_wcs(
            pixel_scale_arcsec=1.0,
            crval_ra=float(ra_crval_i),
            crval_dec=float(dec_crval_i),
        )
        wcs_list.append(wcs_i)

        # Wahre Himmelsposition des Asteroiden zur Zeit t_i im Referenz-WCS
        x_true = x0 + vx * dt
        y_true = y0 + vy * dt
        ra_ast, dec_ast = pix_to_world(reference_wcs, x_true, y_true)

        # Gemessene Pixelposition auf dem gedrifteten Sensor i
        x_det, y_det = world_to_pix(wcs_i, ra_ast, dec_ast)

        frame_i = generate_synthetic_sky(
            n_frames=1,
            height=height,
            width=width,
            x0=float(x_det),
            y0=float(y_det),
            vx=0.0,
            vy=0.0,
            flux=flux,
            sigma=sigma,
            noise_std=0.0,
            device=device,
        )[0]
        frames_list.append(frame_i)

    frames = torch.stack(frames_list, dim=0)

    offsets = compute_wcs_frame_offsets(
        wcs_list=wcs_list,
        reference_wcs=reference_wcs,
        height=height,
        width=width,
        device=device,
    )
    assert offsets.shape == (n_frames, 2)
    assert offsets.device.type == device.type
    assert torch.allclose(offsets[0], torch.zeros(2, device=device), atol=1e-4)

    stacked = single_pass_wcs_kinematic_stack(
        frames=frames,
        times_hours=times_hours,
        vx=vx,
        vy=vy,
        wcs_offsets_xy=offsets,
        method="mean",
    )

    assert stacked.shape == (height, width)
    assert stacked.device.type == device.type

    # Peak muss an der ursprünglichen Referenzposition (y0, x0) liegen
    max_idx = torch.argmax(stacked)
    peak_y = int(max_idx // width)
    peak_x = int(max_idx % width)
    assert peak_x == round(x0)
    assert peak_y == round(y0)

    # Flusserhaltung innerhalb < 2%
    total_flux = float(stacked.sum().item())
    rel_flux_error = abs(total_flux - flux) / flux
    assert rel_flux_error < 0.02
