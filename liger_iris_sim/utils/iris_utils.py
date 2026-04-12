
__all__ = [
    'iris_imager_to_spatial',
    'iris_skyrel_to_imager',
    'iris_sky_to_imager',
    'iris_ifs_to_spatial',
    'IRIS_PROPS',
]

IRIS_PROPS = {
    'image_plate_scale': 0.004,
    'ifs_lenslet_plate_scales': [0.004, 0.009],
    'ifs_slicer_plate_scales': [0.025, 0.05],
    'tmt_colldiam': 30,
    'tmt_collarea': 630,
    'gain': 1.0,
    'dark_current': 0.025,
    'read_noise': 9,
    'imager_detector_size': (4096, 4096),
    'ifs_detector_size': (4096, 4096),
}


def iris_imager_to_spatial(
    xdet : float, ydet : float,
    scale : float,
) -> tuple[float, float]:
    """
    Convert imager detector coordinates to spatial coordinates.

    Parameters
    ----------
    xdet : float
        X coordinate in pixels.
    ydet : float
        Y coordinate in pixels.
    scale : float
        Plate scale in arcsec.

    Returns
    -------
    xs : float
        X coordinate in arcsec.
    ys : float
        Y coordinate in arcsec.
    """
    xs0, ys0 = 0.6, 0.6
    xs = scale * xdet + xs0
    ys = scale * ydet + ys0
    return xs, ys


def iris_skyrel_to_imager(
    xs : float, ys : float,
    scale : float,
) -> tuple[float, float]:
    """
    Convert relative sky coordinates to imager coordinates.

    Parameters
    ----------
    xs : float
        X coordinate in arcsec relative to on-axis.
    ys : float
        Y coordinate in arcsec relative to on-axis.
    scale : float
        Plate scale in arcsec.

    Returns
    -------
    xdet : float
        X coordinate in pixels.
    ydet : float
        Y coordinate in pixels.
    """
    xs0, ys0 = 0.6, 0.6
    xdet = (xs - xs0) / scale
    ydet = (ys - ys0) / scale
    return xdet, ydet


def iris_sky_to_imager(
    ra_deg : float, dec_deg : float,
    scale : float,
    size : tuple[int, int]
) -> tuple[float, float]:
    """
    Convert sky coordinates (RA, DEC) relative to the center of the imager
    to detector coordinates (pixels).

    Parameters
    ----------
    ra_deg : float
        RA offset in degrees relative to the center.
    dec_deg : float
        Dec offset in degrees relative to the center.
    scale : float
        Plate scale in mas.
    size : tuple[int, int]
        Size of the detector in pixels.

    Returns
    -------
    x_pixel : float
        X coordinate in pixels.
    y_pixel : float
        Y coordinate in pixels.
    """
    # Convert RA/Dec from degrees to arcsec
    ra_as = ra_deg * 3600
    dec_as = dec_deg * 3600
    
    # Convert to pixel offsets from the center
    x_offset = ra_as / scale
    y_offset = dec_as / scale
    
    # Compute detector coordinates
    x_pixel = size[1] / 2 - 0.5 + x_offset
    y_pixel = size[0] / 2 - 0.5 + y_offset
    
    return x_pixel, y_pixel


def iris_ifs_to_spatial(
    ydet : float, xdet : float,
    scale : float,
    size : tuple [int, int]
):
    """
    Convert IFS spaxel coordinates to spatial coordinates.

    Parameters
    ----------
    ydet : float
        The y-coordinate of the spaxel in detector coordinates (pixels).
    xdet : float
        The x-coordinate of the spaxel in detector coordinates (pixels).
    scale : float
        The plate scale in mas.
    size : tuple[int, int]
        The size of the detector in pixels.

    Returns
    -------
    ys : float
        The x-coordinate of the spaxel in spatial coordinates (arcsec).
    xs : float
        The y-coordinate of the spaxel in spatial coordinates (arcsec).
    """
    xs0 = -int(size[1] / 2) * scale
    ys0 = -int(size[0] / 2) * scale
    xs = scale * xdet + xs0
    ys = scale * ydet + ys0
    return ys, xs