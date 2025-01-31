import numpy as np
from astropy.modeling.models import Gaussian1D

__all__ = ['convolve_spectrum']

def convolve_spectrum(
        wavelengths : np.ndarray, spectrum : np.ndarray,
        resolution : float, n_res : float = 8
    ) -> np.ndarray:
    """
    Convolve a spectrum with a Gaussian line spread function (LSF).

    Args:
        wavelengths (np.ndarray): The wavelength grid.
        spectrum (np.ndarray): The spectrum grid.
        resolution (float): The desired resolution, R = lambda / fwhm.
        n_res (float): The number of resolution elements (fwhm) to include in the LSF on each side, defaults to 8.
    """
    fwhm = np.mean(wavelengths) / resolution
    stddev = fwhm / (2 * np.sqrt(2 * np.log(2)))
    dl = np.median(np.diff(wavelengths))
    nx = int(fwhm / dl) * n_res * 2
    if nx % 2 == 0:
        nx += 1
    # Pad size
    n_pad = int(np.floor(nx / 2))

    # Build LSF
    wave_rel = (np.arange(nx) - np.floor(nx / 2)) * dl
    lsf = Gaussian1D(amplitude=1, mean=0, stddev=stddev)(wave_rel)
    lsf /= np.sum(lsf)

    # Pad spectrum
    spectrum_padded = np.pad(spectrum, pad_width=(n_pad, n_pad), mode='edge')

    # Convolve
    spectrum_conv = np.convolve(spectrum_padded, lsf, mode='valid')

    # Return
    return spectrum_conv

