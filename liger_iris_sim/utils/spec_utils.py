import numpy as np
import scipy.interpolate
from astropy.modeling.models import Gaussian1D
import scipy.constants
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
import astropy.units as units


# #@njit
# def bin_spectrum(wave, spec, wave_out):
#     spec_out = np.zeros(wave_out.size)
#     for i in range(spec_out.size):
#         #ii = np.searchsorted(wave, wave_out[i], side='left')
#         #jj = np.searchsorted(wave, wave_out[i]+1, side='left')
#         #spec_out[i] = val
#     return spec_out


def convolve_spectrum(
        wavelengths : np.ndarray, spectrum : np.ndarray,
        resolution : float, n_res : float = 8
    ) -> np.ndarray:
    """
    Parameters:
    wavelengths (np.ndarray): The wavelength grid.
    spectrum (np.ndarray): The spectrum grid.
    resolution (float): The desired resolution, R = lambda / fwhm.
    n_res (float): The number of resolution elements (fwhm) to include in the LSF on each side, defaults to 4.
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

