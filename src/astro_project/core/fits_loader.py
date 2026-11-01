import numpy as np
from astropy.io import fits

def load_fits(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    return data, header
