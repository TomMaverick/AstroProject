import numpy as np
import torch
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder

def estimate_background(image_tensor: torch.Tensor) -> torch.Tensor:
    data_np = image_tensor.detach().cpu().numpy().astype(np.float32)
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(
        data_np,
        (50, 50),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )
    bkg_tensor = torch.from_numpy(bkg.background.astype(np.float32)).to(
        device=image_tensor.device, dtype=image_tensor.dtype
    )
    return bkg_tensor

def detect_sources(image_tensor: torch.Tensor, threshold: float, fwhm: float = 3.0):
    data_np = image_tensor.detach().cpu().numpy().astype(np.float32)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold)
    sources = daofind(data_np)
    return sources
