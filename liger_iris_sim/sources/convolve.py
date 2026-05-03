import numpy as np
import scipy.interpolate
from astropy.modeling.models import Gaussian1D

from ..utils.psf_utils import shift_psf_phase

__all__ = ['convolve_point_source', 'convolve_spectrum']


def convolve_point_source(
    x : float, y : float,
    flux : np.ndarray,
    psf : np.ndarray,
    image_out : np.ndarray,
    fix_psf_phase : bool = True,
) -> np.ndarray:
    """
    Convolve a point source at [y, x] with a PSF.

    Parameters
    ----------
    x  : float
        The x-coordinate of the point source - second axis.
    y : float
        The y-coordinate of the point source - first axis.
    flux : np.ndarray
        The flux of the point source in any units.
    psf : np.ndarray
        The PSF to convolve with.
    image_out : np.ndarray
        The output image array to write the result into.
    fix_psf_phase : bool
        If True, the PSF is shifted by the subpixel offset of the source.

    Returns
    -------
    image_out : np.ndarray
        An image with the convolved point source in the same units as flux.
    """
    dx = x - np.round(x)
    dy = y - np.round(y)
    if (dx != 0 or dy != 0) and fix_psf_phase:
        psf = shift_psf_phase(psf, dx=dx, dy=dy)

    # Normalize psf
    psf = psf / np.sum(psf)

    H, W = image_out.shape
    ny, nx = psf.shape

    assert ny % 2 == 1 and nx % 2 == 1, "PSF must have odd dimensions"

    cy = ny // 2
    cx = nx // 2

    iy = int(np.round(y))
    ix = int(np.round(x))

    y0 = iy - cy
    x0 = ix - cx
    y1 = y0 + ny
    x1 = x0 + nx

    py0 = max(0, -y0)
    px0 = max(0, -x0)
    py1 = ny - max(0, y1 - H)
    px1 = nx - max(0, x1 - W)

    iy0 = max(0, y0)
    ix0 = max(0, x0)
    iy1 = min(H, y1)
    ix1 = min(W, x1)

    if iy0 >= iy1 or ix0 >= ix1:
        return image_out

    image_out[iy0:iy1, ix0:ix1] += flux * psf[py0:py1, px0:px1]
    
    return image_out

def convolve_spectrum(
    wave : np.ndarray, spectrum : np.ndarray,
    resolution : float,
    n_res : float = 4
) -> np.ndarray:
    """
    Convolve a spectrum with a Gaussian line spread function (LSF) at an average resolution over the bandpass.

    Parameters
    ----------
    wave : np.ndarray
        The wavelength grid. Must be uniformly sampled.
    spectrum : np.ndarray
        The spectrum grid.
    resolution : float
        The desired resolution, R = lambda / fwhm.
    n_res : float, optional
        The number of resolution elements (fwhm) to include in the LSF on each side.
        Defaults to 4.

    Returns
    -------
    spectrum_conv : np.ndarray
        The convolved spectrum, sampled on the same wavelength grid as the input.
    """

    if len(wave) != len(spectrum):
        raise ValueError(f"wave and spectrum must have the same length, "
                     f"got {len(wave)} and {len(spectrum)}")
    
    # Determine number of points for LSF grid
    n_wave = len(wave)
    fwhm = wave[n_wave // 2] / resolution
    stddev = fwhm / (2 * np.sqrt(2 * np.log(2)))
    dl = wave[1] - wave[0]
    n_lsf = round(fwhm / dl * n_res * 2)
    if n_lsf % 2 == 0:
        n_lsf += 1
    
    # Pad size
    n_pad = int(np.floor(n_lsf / 2))

    # Build LSF
    wave_rel = (np.arange(n_lsf) - np.floor(n_lsf / 2)) * dl
    lsf = Gaussian1D(amplitude=1, mean=0, stddev=stddev)(wave_rel)
    lsf /= np.sum(lsf)

    # Pad spectrum
    spectrum_padded = np.pad(spectrum, pad_width=(n_pad, n_pad), mode='edge')

    # Convolve
    spectrum_conv = np.convolve(spectrum_padded, lsf, mode='valid')

    # Return convolved spectrum
    return spectrum_conv