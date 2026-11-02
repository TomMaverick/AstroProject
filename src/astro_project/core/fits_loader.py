import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path
from typing import Union


def _select_image_hdu(hdul, hdu_index: int | None = None):
    """Wählt die angeforderte oder die erste bildtragende HDU (mind. 2D) aus einer HDUList."""
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
    nan_fill: float = 0.0,
    hdu_index: int | None = None,
):
    """Lädt ein FITS-Bild (inkl. Multi-HDU / .fits.fz) als [Y, X] PyTorch-Tensor auf das Ziel-Device."""
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

    # NaN / Inf durch nan_fill ersetzen
    data_np = np.nan_to_num(data_np, nan=nan_fill, posinf=nan_fill, neginf=nan_fill)
    data_np = np.ascontiguousarray(data_np, dtype=np.float32)

    # Als PyTorch Tensor auf das Ziel-Device laden
    tensor = torch.from_numpy(data_np).to(device=device)
    return tensor, wcs, header


def load_fits(file_path, hdu_index: int | None = None):
    """Lädt Rohdaten und Header einer FITS-Datei."""
    with fits.open(file_path, memmap=False) as hdul:
        hdu = _select_image_hdu(hdul, hdu_index=hdu_index)
        data = np.asarray(hdu.data, dtype=np.float32)
        header = hdu.header
    if data.ndim > 2:
        data = np.squeeze(data)
    return data, header


def save_fits_image(
    tensor: torch.Tensor,
    file_path: Union[str, Path],
    wcs: WCS | None = None,
    header: fits.Header | None = None,
) -> None:
    """Saves a 2D [Y, X] PyTorch tensor mock-free as a FITS file with optional WCS/header."""
    data_np = tensor.detach().cpu().numpy().astype(np.float32)
    out_header = header.copy() if header is not None else fits.Header()
    if wcs is not None:
        out_header.update(wcs.to_header())
    out_header["CREATOR"] = "AstroProject Pipeline"
    fits.writeto(str(file_path), data_np, header=out_header, overwrite=True)
