import numpy as np
from numba import njit

__all__ = [
    "crop_AO_psf",
    "fix_psf_phase",
]


def crop_AO_psf(
    psf : np.ndarray,
    scale : float,
    wave : float,
    colldiam : float,
    n : int = 100,
):
    """
    Crops a AO PSF parametrized by the telescope diameter and wavelength.
    
    In other words, the PSF is cropped to a size of ``n * wavelength / colldiam``,
    where ``wavelength / colldiam`` is the diffraction limit
    of the telescope at the given wavelength.

    The PSF must have an odd number of rows and columns, and the center of the
    PSF is assumed to be at the center of the array.

    Parameters
    ----------
    psf : np.ndarray
        The PSF to crop.
    scale : float
        The size of a PSF pixel in arcsec.
    wave : float
        The wavelength in microns.
    colldiam : float
        The effective collimating diameter in meters.
    n : int, optional
        The number of lambda / D's to crop by.
        Defaults to 100.

    Returns
    -------
    psf_out: np.ndarray:
        The new PSF.
    """

    if psf.shape[0] % 2 != 1 or psf.shape[1] % 2 != 1:
        raise ValueError(f"PSF must have odd number of rows and columns, got {psf.shape}")
    
    ny, nx = psf.shape

    # lambda / D per pixel
    s = 206265 * wave / (colldiam * 1E6) / scale

    cy, cx = psf.shape[0] // 2, psf.shape[1] // 2

    # Compute the crop size
    # Initial bounds
    w = round(n * s)
    yi = cy - w
    yf = cy + w
    xi = cx - w
    xf = cx + w

    # Check bounds
    yi = max(yi, 0)
    yf = min(yf, ny - 1)
    xi = max(xi, 0)
    xf = min(xf, nx - 1)

    # Ensure odd number of rows and columns
    if (yf - yi) % 2 == 0:
        if yf < ny - 1:
            yf += 1
        else:
            yi -= 1
    if (xf - xi) % 2 == 0:
        if xf < nx - 1:
            xf += 1
        else:
            xi -= 1

    # Slice PSF
    psf_out = psf[yi:yf+1, xi:xf+1].copy()

    # Return
    return psf_out

@njit
def _bilinear_shift(psf, dy, dx):
    h, w = psf.shape
    i = np.arange(h)
    j = np.arange(w)
    i0 = np.clip(np.floor(i - dy).astype(np.int64), 0, h - 2)
    j0 = np.clip(np.floor(j - dx).astype(np.int64), 0, w - 2)

    out = np.zeros_like(psf, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            iy = i0[y]
            ix = j0[x]

            wy = (y - iy - dy)
            wx = (x - ix - dx)

            # Weights and bounds
            w00 = (1 - wy) * (1 - wx)
            w10 = wy * (1 - wx)
            w01 = (1 - wy) * wx
            w11 = wy * wx

            if 0 <= iy < h-1 and 0 <= ix < w-1:
                out[y, x] = (
                    psf[iy, ix]     * w00 +
                    psf[iy+1, ix]   * w10 +
                    psf[iy, ix+1]   * w01 +
                    psf[iy+1, ix+1] * w11
                )
    return out


def fix_psf_phase(psf : np.ndarray) -> np.ndarray:
    """
    If the shape of the PSF has an even number of rows or columns,
    it is shifted by half a pixel in both directions using bilinear
    interpolation to ensure the central peak is centered on a pixel.

    Parameters
    ----------
    psf : np.ndarray
        The PSF to fix.

    Returns
    -------
    psf_out : np.ndarray
        The fixed PSF.
    """
    psf = _bilinear_shift(psf, dy=0.5, dx=0.5)
    psf /= np.sum(psf)
    return psf


def fix_psf_shape(psf : np.ndarray) -> np.ndarray:
    """
    Fix even shaped PSFs with padding by extending the edge pixels.
    """
    ny, nx = psf.shape

    pad_y = 1 if ny % 2 == 0 else 0
    pad_x = 1 if nx % 2 == 0 else 0

    if pad_y == 0 and pad_x == 0:
        return psf

    return np.pad(
        psf,
        ((0, pad_y), (0, pad_x)),
        mode='edge'
    )