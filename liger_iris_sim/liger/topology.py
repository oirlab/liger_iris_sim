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


def ifu_to_spatial(
        xd : float | np.ndarray, yd : float | np.ndarray,
        scale : float, size : tuple [int, int]
    ):
    xs0 = -int(size[1] / 2) * scale
    ys0 = -int(size[0] / 2) * scale
    xs = scale * xd + xs0
    ys = scale * xd + ys0
    return xs, ys