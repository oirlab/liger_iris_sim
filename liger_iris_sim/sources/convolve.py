import numpy as np
import scipy.interpolate
from astropy.modeling.models import Gaussian1D

__all__ = ['convolve_point_source', 'convolve_spectrum']


def convolve_point_source(
    x : float, y : float,
    flux : np.ndarray,
    psf : np.ndarray,
    image_out : np.ndarray,
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

    Returns
    -------
    image_out : np.ndarray
        An image with the convolved point source in the same units as flux.
    """
    psf_height, psf_width = psf.shape
    psf_center_x = np.ceil(psf_width / 2) if psf_width % 2 == 1 else psf_width / 2 - 0.5
    psf_center_y = np.ceil(psf_height / 2) if psf_width % 2 == 1 else psf_height / 2 - 0.5
    xpsf, ypsf = np.arange(psf_width) - psf_center_x, np.arange(psf_height) - psf_center_y
    itp = scipy.interpolate.RegularGridInterpolator(
        (ypsf + y, xpsf + x),
        psf,
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    xarr, yarr = np.arange(image_out.shape[1]), np.arange(image_out.shape[0])
    XARR, YARR = np.meshgrid(xarr, yarr, indexing='ij')
    psf_shifted = itp((XARR, YARR))
    psf_shifted = np.clip(psf_shifted, 0, None)
    psf_shifted /= np.sum(psf_shifted)
    image_out += flux * psf_shifted
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