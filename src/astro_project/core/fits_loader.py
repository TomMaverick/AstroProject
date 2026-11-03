import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path
from typing import Union


def _select_image_hdu(hdul, hdu_index: int | None = None):
    """Selects the requested or the first image-bearing HDU (>= 2D) from an HDUList."""
    with np.errstate(invalid="ignore", divide="ignore"):
        if hdu_index is not None:
            hdu = hdul[hdu_index]
            if hdu.data is None:
                raise ValueError(f"HDU at index {hdu_index} contains no image data.")
            return hdu

        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) >= 2:
                return hdu

    raise ValueError("No HDU with >= 2D image data found in FITS file.")


def load_fits_image(
    file_path,
    device: torch.device | str = "cpu",
    nan_fill: float | None = 0.0,
    hdu_index: int | None = None,
):
    """Loads a FITS image (including Multi-HDU / .fits.fz) as a [Y, X] PyTorch tensor onto the target device."""
    with np.errstate(invalid="ignore", divide="ignore"):
        with fits.open(file_path, memmap=False) as hdul:
            hdu = _select_image_hdu(hdul, hdu_index=hdu_index)
            data_np = np.asarray(hdu.data, dtype=np.float32)
            header = hdu.header

    if data_np.ndim > 2:
        data_np = np.squeeze(data_np)
    if data_np.ndim > 2:
        data_np = data_np[(0,) * (data_np.ndim - 2)]
    if data_np.ndim != 2:
        raise ValueError(f"Expected 2D image after squeezing, got shape {data_np.shape}")

    try:
        wcs = WCS(header, naxis=2)
    except Exception:
        wcs = WCS(header)

    finite_mask = np.isfinite(data_np)
    if not np.all(finite_mask):
        if nan_fill is None or nan_fill == "median":
            fill_val = float(np.median(data_np[finite_mask])) if np.any(finite_mask) else 0.0
        else:
            fill_val = float(nan_fill)
        data_np = np.where(finite_mask, data_np, fill_val)

    data_np = np.ascontiguousarray(data_np, dtype=np.float32)

    tensor = torch.from_numpy(data_np).to(device=device)
    return tensor, wcs, header


def load_fits(file_path, hdu_index: int | None = None):
    """Loads raw data and header of a FITS file."""
    with np.errstate(invalid="ignore", divide="ignore"):
        with fits.open(file_path, memmap=False) as hdul:
            hdu = _select_image_hdu(hdul, hdu_index=hdu_index)
            data = np.asarray(hdu.data, dtype=np.float32)
            header = hdu.header
    if data.ndim > 2:
        data = np.squeeze(data)
    return data, header


def save_fits_image(tensor: torch.Tensor, file_path, wcs=None, header=None):
    """Saves a 2D [Y, X] PyTorch tensor mock-free as a FITS file with optional WCS/header."""
    data_np = tensor.detach().cpu().numpy().astype(np.float32)
    out_header = header.copy() if header is not None else fits.Header()
    if wcs is not None:
        out_header.update(wcs.to_header())
    out_header['CREATOR'] = 'AstroProject Pipeline'
    fits.writeto(str(file_path), data_np, header=out_header, overwrite=True)
