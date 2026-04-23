from liger_iris_drp_resources import get_model_spectra_dir
from ..sources.convolve import convolve_spectrum
from ..utils import rebin_spectrum

from astropy.io import fits
import numpy as np
import os

def _get_gemini_sky_filepath():
    return os.path.join(get_model_spectra_dir(), 'mk_skybg_zm_16_15_ph.fits')


def _load_gemini_sky():
    """
    Load the Gemini sky spectrum from a FITS file.
    The spectrum is in units of J / (s * m^2 * micron * arcsec^2).

    Returns
    -------
    wave : np.ndarray
        The wavelength array in microns.
    sky_gemini : np.ndarray
        The Gemini sky spectrum in units of J / (s * m^2 * micron * arcsec^2).
    """
    filename = _get_gemini_sky_filepath()
    with fits.open(filename) as hdul:
        header = hdul[0].header
        n = header['NAXIS1']
        w = np.arange(n) * header['CDELT1'] + header['CRVAL1']
        w /= 1E7
        sky_gemini = hdul[0].data
    return w, sky_gemini


def get_gemini_background(
    wave : np.ndarray,
    resolution : float | None = None
) -> np.ndarray:
    """
    Get the Gemini background spectrum resampled onto wave.
    The output spectrum is in units of photons / (s * m^2 * arcsec^2).

    Parameters
    ----------
    wave : np.ndarray
        The wavelength array.

    Returns
    -------
    gem_sky : np.ndarray
        The Gemini background spectrum binned on the input wave grid.
    """
    # Load Gemini file
    # Wave units = microns
    # spec units = photons / (s * m^2 * nm * arcsec^2)
    # ph/sec/arcsec^2/nm/m^2
    gem_wave, gem_sky = _load_gemini_sky()

    # Convolve
    if resolution is not None:
        gem_sky = convolve_spectrum(gem_wave, gem_sky, resolution=resolution)

    gem_sky *= 1E3 # photons / (s * m^2 * micron * arcsec^2)
    dw_in = gem_wave[1] - gem_wave[0]
    gem_sky = rebin_spectrum(gem_wave, gem_sky * dw_in, wave)

    # Return
    return gem_sky

