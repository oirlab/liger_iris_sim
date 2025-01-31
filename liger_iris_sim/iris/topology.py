import numpy as np

__all__ = ['imager_to_spatial', 'spatial_to_imager', 'sky_to_imager', 'ifu_to_spatial']

def imager_to_spatial(
        xd : float, yd : float,
        scale : float,
    ) -> tuple[float, float]:
    """
    Convert imager detector coordinates to spatial coordinates.

    Args:
        xd (float): X coordinate in pixels.
        yd (float): Y coordinate in pixels.
        scale (float): Plate scale in arcsec.

    Returns:
        (float, float): Spatial coordinates (xs, ys).
    """
    xs0, ys0 = 0.6, 0.6
    xs = scale * xd + xs0
    ys = scale * yd + ys0
    return xs, ys


def spatial_to_imager(
        xs : float, ys : float,
        scale : float,
    ) -> tuple[float, float]:
    """
    Convert spatial coordinates to imager coordinates.

    Args:
        xs (float): X coordinate in arcsec relative to on-axis.
        ys (float): Y coordinate in arcsec relative to on-axis.
        scale (float): Plate scale in arcsec.

    Returns:
        (float, float): Detector coordinates (x_pixel, y_pixel).
    """
    xs0, ys0 = 0.6, 0.6
    xd = (xs - xs0) / scale
    yd = (ys - ys0) / scale
    return xd, yd


def sky_to_imager(
        ra : float, dec : float,
        scale : float,
        size : tuple[int, int]
    ) -> tuple[float, float]:
    """
    Convert sky coordinates (RA, DEC) relative to the center of the imager
    to detector coordinates (pixels).

    Args:
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
    x_pixel = size[1] / 2 - 0.5 + x_offset
    y_pixel = size[0] / 2 - 0.5 + y_offset
    
    return x_pixel, y_pixel


def ifu_to_spatial(
        xd : float, yd : float,
        scale : float,
        size : tuple [int, int]
    ):
    """
    Convert IFU spaxel coordinates to spatial coordinates.

    Args:
        xd (float): X coordinate in spaxels.
        yd (float): Y coordinate in spaxels.
        scale (float): Plate scale in mas.
        size (tuple[int, int]): Size of the detector in pixels.
    """
    xs0 = -int(size[1] / 2) * scale
    ys0 = -int(size[0] / 2) * scale
    xs = scale * xd + xs0
    ys = scale * yd + ys0
    return xs, ys