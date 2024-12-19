import numpy as np

def imager_to_spatial(
        xd : float | np.ndarray, yd : float | np.ndarray,
        scale : float, xs0 : float = 0.6, ys0 : float = 0.6,
    ):
    xs = scale * xd + xs0
    ys = scale * xd + ys0
    return xs, ys


def spatial_to_imager(
        xs : float | np.ndarray, ys : float | np.ndarray, scale : float,
        xs0 : float = 0.6, ys0 : float = 0.6,
    ):
    xd = (xs - xs0) / scale
    yd = (ys - ys0) / scale
    return xd, yd

def sky_to_imager(ra, dec, scale):
    """
    Convert sky coordinates (RA, DEC) relative to the center of the imager
    to detector coordinates (pixels).

    Parameters:
        ra_deg (float): RA offset in degrees relative to the center.
        dec_deg (float): Dec offset in degrees relative to the center.
        scale (float): Plate scale in mas.

    Returns:
        (float, float): Detector coordinates (x_pixel, y_pixel).
    """
    # Convert RA/Dec from degrees to arcsec
    ra_as = ra * 3600
    dec_as = dec * 3600
    
    # Convert to pixel offsets from the center
    x_offset = ra_as / scale
    y_offset = dec_as / scale
    
    # Compute detector coordinates
    nx, ny = 4096, 4096
    x_pixel = nx / 2 - 0.5 + x_offset
    y_pixel = ny / 2 - 0.5 + y_offset
    
    return x_pixel, y_pixel


def ifu_to_spatial(
        xd : float | np.ndarray, yd : float | np.ndarray,
        scale : float, size : tuple [int, int]
    ):
    xs0 = -int(size[1] / 2) * scale
    ys0 = -int(size[0] / 2) * scale
    xs = scale * xd + xs0
    ys = scale * xd + ys0
    return xs, ys