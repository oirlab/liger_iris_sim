from astropy.modeling.models import Lorentz1D
from liger_iris_drp_resources import get_model_spectra_dir

import numpy as np
import os

import logging
logger = logging.getLogger(__name__)

def _get_ohlines_filepath():
    return os.path.join(get_model_spectra_dir(), 'optical_ir_sky_lines.txt')


def sim_ohlines(
    wave : np.ndarray,
    resolution : float,
) -> np.ndarray:
    """
    Simulate the OH lines.

    Parameters
    ----------
    wave : np.ndarray
        The wavelength array.
    resolution : float
        The spectral resolution.

    Returns
    -------
    ohspec : np.ndarray
        The OH spectrum.
    """

    # OH spectrum in units of photons / (m^2 * s * micron * arcsec^2)
    ohspec = np.zeros(len(wave))

    # read OH line file
    # Units are microns
    ohlines_filepath = _get_ohlines_filepath()
    line_centers, line_strengths = np.loadtxt(ohlines_filepath, unpack=True, comments='#')
    good = np.where(
        (line_centers >= wave[0])
        & (line_centers <= wave[-1])
        & (line_strengths > 0)
    )[0]
    n_good = len(good)

    # Build spectrum
    if n_good > 0:
        line_centers, line_strengths = line_centers[good], line_strengths[good]
        for i in range(n_good):
            ohspec += sim_ohline_lorenztian(
                wave,
                wavecenter=line_centers[i],
                flux=line_strengths[i],
                resolution=resolution
            )
    else:
        logger.warning(f"No OH lines found for this wavelength range ({wave[0]} - {wave[-1]} microns). Returning zero OH spectrum.")

    # Return
    return ohspec


def sim_ohline_lorenztian(
    wave : np.ndarray,
    wavecenter : float,
    flux : float,
    resolution : float,
    intrinsic_resolution : float | None = 100_000.0
):
    """
    Simulate a single OH line with a Lorentzian profile.

    Parameters
    ----------
    wave : np.ndarray
        The wavelength array.
    wavecenter : float
        The center of the line.
    flux : float
        The integrated flux of the line in any units.
    resolution : float
        The spectral resolution.
    intrinsic_resolution : float | None, optional
        The intrinsic spectral resolution of the line. If None, the line is assumed to be a delta function.

    Returns
    -------
    spec : np.ndarray
        The simulated OH line spectrum sampled on wave.
        Output units are photons / (m^2 * s * arcsec^2 * wavebin).
    """
    # flux units: photons / (m^2 * s * arcsec^2)
    fwhm = wavecenter / resolution
    if intrinsic_resolution is not None and np.isfinite(intrinsic_resolution):
        fwhm_intrinsic = wavecenter / intrinsic_resolution
        fwhm = np.sqrt(fwhm**2 + fwhm_intrinsic**2)
    spec = Lorentz1D(amplitude=1, x_0=wavecenter, fwhm=fwhm)(wave)
    tot = np.sum(spec)
    # Out units = photons / (m^2 * s * arcsec^2 * wavebin)
    spec /= tot
    spec *= flux
    return spec