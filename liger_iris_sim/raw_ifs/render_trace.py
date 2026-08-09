import numpy as np
from numba import njit

# TODO: Select pixel size based on instrument name if different
from ..utils import LIGER_PROPS
PIXEL_SIZE_UM = LIGER_PROPS['ifs_detector_pixel_size_um']

@njit(nogil=True)
def render_trace_for_lenslet(
    image_out : np.ndarray,
    px_lo : int,
    y_pix : np.ndarray,
    flux : np.ndarray,
    epsf : np.ndarray,
    window_size : int
):
    """
    Add one lenslet's trace to `image_out`, one detector column at a time.

    For column px = px_lo + i, deposits flux[i] distributed over nearby rows
    by the ePSF interpolated at the exact fractional y-offset (py - y_pix[i]).
    dx is zero by construction (evaluated at the column centre), so only
    linear interpolation in y is needed.
    """
    n_pix_y, n_pix_x = image_out.shape
    n_epsf_y = epsf.shape[0]
    n_col = y_pix.size

    # Center of Micropupil PSF
    cy = (epsf.shape[0] - 1) // 2
    cx = (epsf.shape[1] - 1) // 2

    for i in range(n_col):
        px = px_lo + i
        if px < 0 or px >= n_pix_x:
            continue
        if flux[i] == 0.0:
            continue

        y_c = y_pix[i]
        py_lo = int(np.floor(y_c - window_size))
        py_hi = int(np.ceil(y_c + window_size))
        if py_lo < 0:
            py_lo = 0
        if py_hi > n_pix_y - 1:
            py_hi = n_pix_y - 1

        for py in range(py_lo, py_hi + 1):
            fy = (py - y_c) * PIXEL_SIZE_UM + cy
            iy = int(np.floor(fy))
            if iy < 0 or iy + 1 >= n_epsf_y:
                continue
            ty = fy - iy
            image_out[py, px] += flux[i] * (
                (1.0 - ty) * epsf[iy, cx] + ty * epsf[iy + 1, cx]
            )