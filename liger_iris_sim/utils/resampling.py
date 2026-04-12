import numpy as np
from numba import njit

__all__ = ['rebin_image', 'rebin_spectrum', 'rebin_image_scipy']

from numba import njit
import numpy as np

@njit(cache=True)
def _build_edges(wave: np.ndarray) -> np.ndarray:
    n = len(wave)
    edges = np.empty(n + 1, dtype=np.float64)

    for i in range(1, n):
        edges[i] = 0.5 * (wave[i - 1] + wave[i])

    edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    edges[n] = wave[n - 1] + 0.5 * (wave[n - 1] - wave[n - 2])

    return edges


@njit(cache=True)
def _rebin1d_numba(
    in_edges: np.ndarray,
    in_flux: np.ndarray,
    out_edges: np.ndarray,
    out_flux: np.ndarray,
) -> None:

    n_in = len(in_flux)

    for k in range(n_in):

        i_lo = in_edges[k]
        i_hi = in_edges[k + 1]
        i_width = i_hi - i_lo

        if i_width <= 0.0:
            continue

        for j in range(len(out_flux)):

            o_lo = out_edges[j]
            o_hi = out_edges[j + 1]

            overlap = min(i_hi, o_hi) - max(i_lo, o_lo)

            if overlap > 0.0:
                out_flux[j] += in_flux[k] * (overlap / i_width)


def rebin_spectrum(
    wave: np.ndarray,
    spec: np.ndarray,
    wave_out: np.ndarray,
    return_edges: bool = False
) -> np.ndarray:
    """
    Rebin a 1-D spectrum onto a new wavelength grid.

    Parameters
    ----------
    wave : np.ndarray
        The input wavelength array.
    spec : np.ndarray
        The input spectrum.
    wave_out : np.ndarray
        The output wavelength array.
    return_edges : bool, optional
        If True, also return the input and output wavelength bin edges.
        Default is False.

    Returns
    -------
    spec_out : np.ndarray
        The rebinned spectrum.
    """

    in_edges = _build_edges(wave)
    out_edges = _build_edges(wave_out)

    spec_out = np.zeros(len(wave_out), dtype=np.float64)

    _rebin1d_numba(in_edges, spec, out_edges, spec_out)

    if return_edges:
        return spec_out, in_edges, out_edges
    else:
        return spec_out


@njit(cache=True)
def _rebin2d_numba(
    image  : np.ndarray,
    output_image : np.ndarray,
):
    
    ny_in, nx_in = image.shape
    ny_out, nx_out = output_image.shape

    xbox = np.float32(nx_in) / np.float32(nx_out)
    ybox = np.float32(ny_in) / np.float32(ny_out)

    for i in range(ny_out):
        y0   = i * ybox
        y1   = y0 + ybox
        iy0  = int(y0)
        iy1  = min(int(y1), ny_in - 1)

        for j in range(nx_out):
            x0   = j * xbox
            x1   = x0 + xbox
            ix0  = int(x0)
            ix1  = min(int(x1), nx_in - 1)

            s = np.float32(0.0)
            for ki in range(iy0, iy1 + 1):
                for kj in range(ix0, ix1 + 1):
                    yw = min(ki + 1, y1) - max(ki, y0)
                    xw = min(kj + 1, x1) - max(kj, x0)
                    s += image[ki, kj] * yw * xw

            output_image[i, j] = s

def rebin_image(
    image : np.ndarray,
    scale_in : float | None = None,
    scale_out : float | None = None,
    fix_odd_shape : bool = False
) -> np.ndarray:
    """
    Rebin a 2-D image to a new shape or by a scale factor.

    Parameters
    ----------
    image : np.ndarray
        The 2-D image to rebin.
    new_shape : tuple[int, int] | None, optional
        The target shape.
        Either new_shape or scales must be provided.
    scale_in : float | None, optional
        Size of input pixels in arcsec/pixel.
        Either new_shape or scales must be provided.
    scale_out : float | None, optional
        Size of output pixels in arcsec/pixel.
        Either new_shape or scales must be provided.
    fix_odd_shape : bool, optional
        If True, if the output shape is odd, the last index of each axis is dropped.
        Default is False.

    Returns
    -------
    image_out : np.ndarray
        The rebinned image
    """

    ny_in, nx_in = image.shape
    rel_scale = scale_in / scale_out
    ny_out = max(1, round(ny_in * rel_scale))
    nx_out = max(1, round(nx_in * rel_scale))
    new_shape = (ny_out, nx_out)

    output_image = np.zeros(new_shape, dtype=np.float32)
    _rebin2d_numba(image, output_image)

    if fix_odd_shape:
        if output_image.shape[0] % 2 == 1:
            output_image = output_image[:-1, :]
        if output_image.shape[1] % 2 == 1:
            output_image = output_image[:, :-1]

    return output_image

def rebin_image_scipy(
    image: np.ndarray,
    scale_in: float | None = None,
    scale_out: float | None = None,
    fix_odd_shape: bool = False
) -> np.ndarray:
    
    from scipy.ndimage import zoom

    # import matplotlib
    # matplotlib.use('QTAGG')
    # import matplotlib.pyplot as plt

    ny_in, nx_in = image.shape
    rel_scale = scale_in / scale_out
    ny_out = max(1, round(ny_in * rel_scale))
    nx_out = max(1, round(nx_in * rel_scale))
    new_shape = (ny_out, nx_out)

    #zoom_factors = (new_shape[0] / image.shape[0], new_shape[1] / image.shape[1])
    zoom_factors = (new_shape[0] / image.shape[0], new_shape[1] / image.shape[1])
    output_image = zoom(image, zoom_factors, order=3, mode='nearest', prefilter=True, grid_mode=True)

    if fix_odd_shape:
        if output_image.shape[0] % 2 == 1:
            output_image = output_image[:-1, :]
        if output_image.shape[1] % 2 == 1:
            output_image = output_image[:, :-1]
    
    return output_image