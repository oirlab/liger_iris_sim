import numpy as np
from astropy.io import fits
import scipy.interpolate as interpolate
import scipy.constants
from astropy.modeling.models import Gaussian1D, Lorentz1D
from ..utils import convolve_spectrum
import warnings
import pandas as pd
import os

c = scipy.constants.c  # m/s
h = scipy.constants.h  # J s
k = scipy.constants.k # J / K

def get_maunakea_spectral_sky_transmission(
        wavelengths : np.ndarray,
        resolution : float | None,
        airmass : float = 1,
    ) -> np.ndarray:
    """
    Compute the Maunakea sky transmission spectrum (0, 1).

    Args:
        wavelengths (np.ndarray): The wavelength array.
        tapas_file (str): The path to the TAPAS file.
        resolution (float): The spectral resolution. If None, no convolution is performed.
        airmass (float): The airmass.

    Returns:
        np.ndarray: The convolved and resampled transmission spectrum.
    """
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(module_dir, 'data/sky/TAPAS_Maunakea_NIR.txt')
    df = pd.read_csv(filename, delimiter=',', usecols=(0, 1), names=["wave", "spec"], header=0)
    tapas_wave = np.array(df.wave)
    tapas_spec = np.array(df.spec)
    dw = wavelengths[1] - wavelengths[0]
    good = np.where((tapas_wave >= wavelengths[0] - 10*dw) & (tapas_wave <= wavelengths[-1] + 10*dw))[0]
    tapas_wave, tapas_spec = tapas_wave[good], tapas_spec[good]
    spec = np.interp(wavelengths, tapas_wave, tapas_spec, left=tapas_spec[0], right=tapas_spec[-1])
    if resolution is not None:
        spec = convolve_spectrum(wavelengths, spec, resolution=resolution)
    spec **= airmass
    return spec


def get_maunakea_spectral_sky_emission(
        wavelengths : np.ndarray,
        resolution : float,
        ohlines_file : str | None = None,
        T_tel : float = 275, T_atm : float = 258, T_aos : float = 243, T_zod : float = 5800,
        Em_tel : float = 0.09, Em_atm : float = 0.2, Em_aos : float = 0.01, Em_zod : float = 1.47E-12,
    ) -> dict:
    """
    Compute the Maunakea sky emission spectrum.

    Args:
        wavelengths (np.ndarray): The wavelength array.
        resolution (float): The spectral resolution (taken to be constant across bandpass here).
    """

    sterr = 1 / 206265**2 # rad^2 / arcsec^2
    dw = np.nanmedian(np.diff(wavelengths))

    # BB components
    wavem = wavelengths * 1E-6
    s1 = 2 * h * c**2 / wavem**5
    s2 = h * c / (wavem * k)
    Ephot = h * c / wavem # J / photon
    # flux density: J / (s * m * m^2 * rad^2)
    bbtel = s1 / (np.exp(s2 / T_tel) - 1) # telescope blackbody spectrum
    bbaos = s1 / (np.exp(s2 / T_aos) - 1) # AO blackbody spectrum
    bbatm = s1 / (np.exp(s2 / T_atm) - 1) # Atm blackbody spectrum
    bbzod = s1 / (np.exp(s2 / T_zod) - 1) # Zodiacal light blackbody spectrum
    bbtel /= Ephot # photons / (s * m * m^2 * rad^2)
    bbaos /= Ephot
    bbatm /= Ephot
    bbzod /= Ephot
    bbtel /= 1E6 # photons / (s * micron * m^2 * rad^2)
    bbaos /= 1E6
    bbatm /= 1E6
    bbzod /= 1E6
    bbtel *= sterr # photons / (s * micron * m^2 * arcsec^2)
    bbaos *= sterr
    bbatm *= sterr
    bbzod *= sterr
    bbtel *= dw # photons / (s * m^2 * arcsec^2 * wavebin)
    bbaos *= dw
    bbatm *= dw
    bbzod *= dw
    bbspec = bbtel * Em_tel \
                + bbaos * Em_aos \
                + bbatm * Em_atm

    # OH lines: photons / (s * m^2 * arcsec^2 * wavebin)
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ohlines_file = os.path.join(module_dir, 'data/sky/optical_ir_sky_lines.txt')
    ohspec = sim_ohlines(wavelengths, ohlines_file=ohlines_file, resolution=resolution)

    # Combined sky emission
    # photons / (s * m^2 * arcsec^2 * wavebin)
    sky_emission = bbspec + ohspec
    
    # Results
    out = dict(
        wavelengths=wavelengths, sky_emission=sky_emission,
        bbtel=bbtel, bbaos=bbaos, bbatm=bbatm, bbzod=bbzod, bbspec=bbspec,
        ohspec=ohspec
    )

    return out

def load_gemini_sky():
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(module_dir, 'data/sky/mk_skybg_zm_16_15_ph.fits')
    with fits.open(filename) as hdul:
        header = hdul[0].header
        n = header['NAXIS1']
        w = np.arange(n) * header['CDELT1'] + header['CRVAL1']
        w /= 1E7
        sky_gemini = hdul[0].data
    return w, sky_gemini

# def get_gemini_background(filename : str, wavelengths : np.ndarray):
#     """
#     """
#     # Load Gemini file
#     with fits.open(filename) as f:
#         gemini_spec = f[0].data
#         header = f[0].header

#     # Wave grid for Gemini spectrum
#     cdelt1 = header["CDELT1"] * 1E-4
#     crval1 = header["CRVAL1"] * 1E-4
#     nx = gemini_spec.shape[0]
#     gemini_wave = np.arange(nx) * cdelt1 + crval1

#     # Convolve
#     delt = 2 * (wavelengths[1] - wavelengths[0]) / (gemini_wave[1] - gemini_wave[0])
#     if delt > 1:
#         stddev = delt / 2 * np.sqrt(2 * np.log(2))
#         x = np.arange(4 * int(delt) + 1) - 2 * int(delt)
#         psf = Gaussian1D(amplitude=1, stddev=stddev)(x)
#         psf /= psf.sum()
#         gemini_spec = np.convolve(gemini_spec, psf, mode='same')
    
#     # Interpolate onto our wavelengths grid
#     ohspec = np.interp(wavelengths, gemini_wave, gemini_spec)
    
#     # Return
#     return ohspec


def sim_ohlines(
        wavelengths : np.ndarray,
        resolution : float,
        ohlines_file : str,
    ) -> np.ndarray:
    """
    Simulate the OH lines.

    Args:
        wavelengths (np.ndarray): The wavelength array.
        resolution (float): The spectral resolution.
        ohlines_file (str): The path to the OH lines file.

    Returns:
        np.ndarray: The OH spectrum.
    """

    # OH spectrum in units of photons / (m^2 * s * micron * arcsec^2)
    ohspec = np.zeros(len(wavelengths))

    # read OH line file
    # Units are microns
    line_centers, line_strengths = np.loadtxt(ohlines_file, unpack=True, comments='#')
    good = np.where((line_centers >= wavelengths[0]) & (line_centers <= wavelengths[-1]) & (line_strengths > 0))[0]
    n_good = len(good)

    # Build spectrum
    if n_good > 0:
        line_centers, line_strengths = line_centers[good], line_strengths[good]
        for i in range(n_good):
            ohspec += sim_ohlines_lorenztian(
                wavelengths, wavecenter=line_centers[i],
                flux=line_strengths[i], resolution=resolution
            )
    else:
        warnings.warn("No OH lines found")

    # Return
    return ohspec


def sim_ohlines_lorenztian(
        wavelengths : np.ndarray, wavecenter : float,
        flux : float, resolution : float,
    ):
    """
    Simulate a single OH line with a Lorentzian profile.

    Args:
        wavelengths (np.ndarray): The wavelength array.
        wavecenter (float): The center of the line.
        flux (float): The integrated flux of the line in any units.
        resolution (float): The spectral resolution.
    """
    # flux units: photons / (m^2 * s * arcsec^2)
    fwhm = wavecenter / resolution
    spec = Lorentz1D(amplitude=1, x_0=wavecenter, fwhm=fwhm)(wavelengths)
    tot = np.sum(spec)
    # Out units = photons / (m^2 * s * arcsec^2 * wavebin)
    spec /= tot
    spec *= flux
    return spec