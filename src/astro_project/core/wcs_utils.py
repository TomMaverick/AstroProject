from astropy.wcs import WCS

def get_wcs(header):
    return WCS(header)
