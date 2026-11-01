import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS


def load_fits_image(file_path, device: torch.device | str = "cpu", nan_fill: float = 0.0):
    """Lädt ein FITS-Bild als [Y, X] PyTorch-Tensor auf das Ziel-Device."""
    with fits.open(file_path, memmap=False) as hdul:
        hdu = hdul[0]
        data_np = np.asarray(hdu.data, dtype=np.float32)
        header = hdu.header
        wcs = WCS(header)

    # NaN / Inf durch nan_fill ersetzen
    data_np = np.nan_to_num(data_np, nan=nan_fill, posinf=nan_fill, neginf=nan_fill)

    # Als PyTorch Tensor auf das Ziel-Device laden
    tensor = torch.from_numpy(data_np).to(device=device)
    return tensor, wcs, header


def save_fits_image(tensor: torch.Tensor, file_path, wcs: WCS = None):
    """Speichert einen [Y, X] Tensor mock-frei als FITS-Datei (Wichtig für Tests)."""
    data_np = tensor.detach().cpu().numpy().astype(np.float32)
    header = wcs.to_header() if wcs is not None else fits.Header()
    header["CREATOR"] = "AstroProject Pipeline"
    fits.writeto(file_path, data_np, header=header, overwrite=True)


def load_fits(file_path):
    """Lädt Rohdaten und Header einer FITS-Datei."""
    with fits.open(file_path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header
    return data, header
