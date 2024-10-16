import numpy as np
import scipy.interpolate
from astropy.modeling.models import Gaussian1D
import scipy.constants
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
import astropy.units as units

def resample_spectrum(
        wavelengths : np.ndarray, spectrum : np.ndarray, wavelengths_out : np.ndarray | None = None,
        phot_flambda : float | None = None, phot_fint : float | None = None,
        resolution : float | None = None
    ):
    """
    Parameters:
    wavelengths (np.ndarray): The wavelength grid in microns.
    spectrum (np.ndarray): The spectrum in arbitrary units.
    wavelengths_out (np.ndarray): The output wavelength grid to sample on.
    phot_flambda (float | None): The photon flux density in photons / sec / micron / m^2.
    phot_fint (float | None): The integrated photon flux in photons / sec / m^2.
    
    Returns:
    wavelengths_out (np.ndarray): The resampled wavelength grid in microns.
    spectrum_out (np.ndarray): The spectrum sampled on wavelengths_out in photons / sec / microns / m^2.
    """

    # Output wavelength grid
    if wavelengths_out is None:
        wavelengths_out = wavelengths.copy()

    # Interpolate onto new grid
    if wavelengths_out is not None:
        # itp = scipy.interpolate.CubicSpline(wavelengths, np.log(spectrum))
        # spectrum_out = np.zeros(wavelengths_out.shape)
        # dws = np.diff(wavelengths_out)
        # dws = np.append(np.diff(wavelengths_out), dws[-1])
        # for i in range(len(wavelengths_out)):
        #     a = wavelengths_out[i] - dws[i] / 2
        #     b = wavelengths_out[i] + dws[i] / 2
        #     val = itp.integrate(a, b, extrapolate=False)
        #     val = np.exp(val)
        #     if np.isfinite(val) and val > 0:
        #         spectrum_out[i] = val
        spectrum_out = FluxConservingResampler()(
            Spectrum1D(
                flux=spectrum * units.ph / (units.s * units.micron * units.m**2),
                spectral_axis=wavelengths * units.micron
            ),
            wavelengths_out * units.micron
        ).data

    else:
        spectrum_out = spectrum.copy()

    # Convolve to resolution
    if resolution is not None:
        spectrum_out = convolve_spectrum(wavelengths_out, spectrum_out, resolution=resolution)

    # Normalize
    spectrum_out /= np.sum(spectrum_out)

    # Scale the spectrum
    dl = np.median(np.diff(wavelengths_out)) # spacing of wave grid in nm
    if phot_flambda is not None:
        spectrum_out *= phot_flambda * (wavelengths_out[-1] - wavelengths_out[0]) / np.sum(spectrum_out * dl)
    elif phot_fint is not None:
        spectrum_out *= phot_fint / np.sum(spectrum_out * dl)
    else:
        raise ValueError("phot_flambda and phot_fint are both None")

    # Return
    return wavelengths_out, spectrum_out


def convolve_spectrum(
        wavelengths : np.ndarray, spectrum : np.ndarray,
        resolution : float, n_res : float = 4
    ):
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
    if nx % 2 != 0:
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

