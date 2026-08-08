import numpy as np
from numba import njit

__all__ = ['rebin_image', 'rebin_spectrum', 'rebin_image_scipy']

from numba import njit
import numpy as np

@njit(nogil=True)
def _build_edges(wave: np.ndarray) -> np.ndarray:
    n = len(wave)
    edges = np.empty(n + 1, dtype=np.float64)

    for i in range(1, n):
        edges[i] = 0.5 * (wave[i - 1] + wave[i])

    edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    edges[n] = wave[n - 1] + 0.5 * (wave[n - 1] - wave[n - 2])

    return edges


@njit(nogil=True)
def _quad_basis_integral(xa: float, xb: float, denom: float, c: float, d: float) -> float:
    """Integral over [c, d] of the quadratic Lagrange basis (x-xa)(x-xb)/denom."""
    return (
        (d ** 3 - c ** 3) / 3.0
        - (xa + xb) * (d ** 2 - c ** 2) / 2.0
        + xa * xb * (d - c)
    ) / denom


@njit(nogil=True)
def _rebin1d_numba(
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    in_edges: np.ndarray,
    out_edges: np.ndarray,
    order: int,
    out_flux: np.ndarray,
    out_err_var: np.ndarray,
) -> None:
    """
    Flux-conserving rebin. Assumes a continuous model of the native
    spectrum: piecewise-constant per input bin (order 0), piecewise-linear
    through the (wave, flux) samples (order 1), or piecewise-quadratic
    through the samples using the nearest centered triple of points
    (order 2). That model is integrated over each output bin (defined by
    `out_edges`) and accumulated into `out_flux`. `err`, if not all zero,
    is propagated in quadrature through the same integration weights into
    `out_err_var` (a variance, not yet sqrt'd).
    """
    n = wave.shape[0]
    n_wave = out_flux.shape[0]

    if order == 0:
        j = 0
        for k in range(n):
            i_lo = in_edges[k]
            i_hi = in_edges[k + 1]
            i_width = i_hi - i_lo
            if i_width <= 0.0:
                continue
            fk = flux[k]
            ek = err[k]

            while j < n_wave and out_edges[j + 1] <= i_lo:
                j += 1
            jj = j
            while jj < n_wave and out_edges[jj] < i_hi:
                lo = out_edges[jj]
                hi = out_edges[jj + 1]
                c = i_lo if i_lo > lo else lo
                d = i_hi if i_hi < hi else hi
                if d > c:
                    wgt = (d - c) / i_width
                    out_flux[jj] += fk * wgt
                    out_err_var[jj] += (ek * wgt) ** 2
                jj += 1

    else:
        j = 0
        for k in range(n - 1):
            wk = wave[k]
            wk1 = wave[k + 1]
            h = wk1 - wk
            if h <= 0.0:
                continue

            use_quad = order == 2 and n >= 3
            if use_quad:
                if k >= 1:
                    i0, i1, i2 = k - 1, k, k + 1
                else:
                    i0, i1, i2 = k, k + 1, k + 2
                x0, x1, x2 = wave[i0], wave[i1], wave[i2]
                y0, y1, y2 = flux[i0], flux[i1], flux[i2]
                s0, s1, s2 = err[i0], err[i1], err[i2]
            else:
                fk = flux[k]
                fk1 = flux[k + 1]
                ek = err[k]
                ek1 = err[k + 1]

            while j < n_wave and out_edges[j + 1] <= wk:
                j += 1
            jj = j
            while jj < n_wave and out_edges[jj] < wk1:
                lo = out_edges[jj]
                hi = out_edges[jj + 1]
                c = wk if wk > lo else lo
                d = wk1 if wk1 < hi else hi
                if d > c:
                    if use_quad:
                        w0 = _quad_basis_integral(x1, x2, (x0 - x1) * (x0 - x2), c, d)
                        w1 = _quad_basis_integral(x0, x2, (x1 - x0) * (x1 - x2), c, d)
                        w2 = _quad_basis_integral(x0, x1, (x2 - x0) * (x2 - x1), c, d)
                        out_flux[jj] += w0 * y0 + w1 * y1 + w2 * y2
                        out_err_var[jj] += (w0 * s0) ** 2 + (w1 * s1) ** 2 + (w2 * s2) ** 2
                    else:
                        # Weights of fk, fk1 in the integral of the linear
                        # interpolant through (wk, fk) and (wk1, fk1) over [c, d].
                        alpha = ((wk1 - c) ** 2 - (wk1 - d) ** 2) / (2.0 * h)
                        beta = ((d - wk) ** 2 - (c - wk) ** 2) / (2.0 * h)
                        out_flux[jj] += alpha * fk + beta * fk1
                        out_err_var[jj] += (alpha * ek) ** 2 + (beta * ek1) ** 2
                jj += 1


def rebin_spectrum(
    wave: np.ndarray,
    spec: np.ndarray,
    wave_out: np.ndarray,
    err: np.ndarray | None = None,
    order: int = 0,
    return_edges: bool = False,
):
    """
    Rebin a 1-D spectrum onto a new wavelength grid, conserving flux.

    The native spectrum is modeled as piecewise-constant per input bin
    (order=0, the classic overlap/box rebin), piecewise-linear through the
    (wave, spec) samples (order=1), or piecewise-quadratic through the
    samples using the nearest centered triple of points (order=2). That
    model is integrated over each output bin (edges placed at the
    midpoints between `wave_out` samples, extrapolated at the ends) and
    divided by the output bin width.

    Parameters
    ----------
    wave : np.ndarray
        The input wavelength array.
    spec : np.ndarray
        The input spectrum.
    wave_out : np.ndarray
        The output wavelength array (bin centers).
    err : np.ndarray | None, optional
        1-sigma errors on `spec`, same shape as `spec`. If given, they are
        propagated in quadrature through the same integration weights and
        the rebinned 1-sigma errors are returned alongside the spectrum.
    order : int, optional
        Order of the assumed native spectrum model: 0 (piecewise-constant,
        default, matches prior behavior), 1 (piecewise-linear), or 2
        (piecewise-quadratic).
    return_edges : bool, optional
        If True, also return the input and output wavelength bin edges.
        Default is False.

    Returns
    -------
    spec_out : np.ndarray
        The rebinned spectrum.
    err_out : np.ndarray
        The rebinned 1-sigma error. Only returned if `err` is given.
    in_edges, out_edges : np.ndarray
        Only returned if `return_edges` is True.
    """
    wave = np.asarray(wave, dtype=np.float64)
    spec = np.asarray(spec, dtype=np.float64)
    wave_out = np.asarray(wave_out, dtype=np.float64)
    err_in = np.zeros_like(spec) if err is None else np.asarray(err, dtype=np.float64)

    in_edges = _build_edges(wave)
    out_edges = _build_edges(wave_out)

    spec_out = np.zeros(len(wave_out), dtype=np.float64)
    err_var_out = np.zeros(len(wave_out), dtype=np.float64)

    _rebin1d_numba(wave, spec, err_in, in_edges, out_edges, order, spec_out, err_var_out)

    result = (spec_out,)
    if err is not None:
        result += (np.sqrt(err_var_out),)
    if return_edges:
        result += (in_edges, out_edges)

    return result[0] if len(result) == 1 else result


@njit(nogil=True)
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
    recenter_to_odd_shape : bool = True
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
    recenter_to_odd_shape : bool, optional
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

    if recenter_to_odd_shape:
        from .psf_utils import _recenter_psf_to_odd_shape
        output_image = _recenter_psf_to_odd_shape(output_image)

    return output_image


def rebin_image_scipy(
    image: np.ndarray,
    scale_in: float | None = None,
    scale_out: float | None = None,
    recenter_to_odd_shape: bool = True
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

    if recenter_to_odd_shape:
        from .psf_utils import _recenter_psf_to_odd_shape
        output_image = _recenter_psf_to_odd_shape(output_image)

    return output_image

