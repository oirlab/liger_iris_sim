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
    *and* columns, weighted by the 2D ePSF bilinearly interpolated at the
    exact fractional offset (py - y_pix[i], pxn - px) of each neighboring
    pixel from the trace reference point. The reference point sits at the
    exact column centre (dx = 0 relative to px), so the x-offset to any
    neighboring column pxn is always an exact integer number of pixels;
    interpolation there degenerates to an exact lookup whenever
    PIXEL_SIZE_UM is an integer, and stays correct otherwise.

    Parameters
    ----------
    image_out : np.ndarray
        2D array of shape (n_pix_y, n_pix_x) to which the lenslet trace will be added to.
    px_lo : int
        The starting pixel column index for this lenslet's trace.
    y_pix : np.ndarray
        1D array of length n_col (# of columns this lenslet spans) containing the y pixel positions for each column of the lenslet trace. The first index corresponds to the column at px_lo.
    flux : np.ndarray
        1D array of length n_col containing the flux values for each column of the lenslet trace.
    epsf : np.ndarray
        2D array of shape (n_epsf_y, n_epsf_x) representing the ePSF (effective Point Spread Function) for the lenslet. The ePSF is assumed to be centered at (cy, cx) in pixel coordinates.
    window_size : int
        The half-width of the window (in pixels) around each trace point over which to distribute the flux for each column, in both x and y. The function will consider rows from (y_pix[i] - window_size) to (y_pix[i] + window_size), and columns from (px - window_size) to (px + window_size).
    """
    n_pix_y, n_pix_x = image_out.shape
    n_epsf_y, n_epsf_x = epsf.shape
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

        px_lo_w = px - window_size
        px_hi_w = px + window_size
        if px_lo_w < 0:
            px_lo_w = 0
        if px_hi_w > n_pix_x - 1:
            px_hi_w = n_pix_x - 1

        for pxn in range(px_lo_w, px_hi_w + 1):
            fx = (pxn - px) * PIXEL_SIZE_UM + cx
            ix = int(np.floor(fx))
            if ix < 0 or ix + 1 >= n_epsf_x:
                continue
            tx = fx - ix

            for py in range(py_lo, py_hi + 1):
                fy = (py - y_c) * PIXEL_SIZE_UM + cy
                iy = int(np.floor(fy))
                if iy < 0 or iy + 1 >= n_epsf_y:
                    continue
                ty = fy - iy

                w00 = epsf[iy, ix]
                w01 = epsf[iy, ix + 1]
                w10 = epsf[iy + 1, ix]
                w11 = epsf[iy + 1, ix + 1]
                weight = (
                    (1.0 - ty) * (1.0 - tx) * w00
                    + (1.0 - ty) * tx * w01
                    + ty * (1.0 - tx) * w10
                    + ty * tx * w11
                )
                image_out[py, pxn] += flux[i] * weight