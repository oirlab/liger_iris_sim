import numpy as np


# Scale always in arcsec / pixel
# x0, y0 always relative to bottom left of detector


def detector_to_spatial(xd : float | np.dnarray, yd : float | np.dnarray, scale : float, x0 : float = 0.6, y0 : float = 0.6):
    xs = scale * xs + x0
    ys = scale * xs + y0
    return xs, ys


def spatial_to_detector(xs : float | np.dnarray, ys : float | np.dnarray, scale : float, x0 : float = 0.6, y0 : float = 0.6):
    xd = (xs - x0) / scale
    yd = (ys - y0) / scale
    return xd, yd