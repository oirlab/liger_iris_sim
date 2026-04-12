import numpy as np
from astropy.io import fits
import scipy.constants
from astropy.modeling.models import Lorentz1D
from ..sources.convolve import convolve_spectrum
from ..utils.resampling import rebin_spectrum

import warnings
import pandas as pd
import os

from liger_iris_drp_resources.model_spectra import download_model_spectra, _get_model_spectra_dir

c = scipy.constants.c  # m/s
h = scipy.constants.h  # J s
k = scipy.constants.k # J / K

__all__ = [
    'get_maunakea_spectral_sky_transmission',
    'get_maunakea_spectral_sky_emission',
]

def _get_tapas_filepath():
    filepath = os.path.join(_get_model_spectra_dir(), 'TAPAS_Maunakea_NIR.txt')
    return filepath

def get_maunakea_spectral_sky_transmission(
    wave : np.ndarray,
    resolution : float | None,
    airmass : float = 1,
) -> np.ndarray:
    """
    Compute the Maunakea sky transmission spectrum (0, 1).

    Parameters
    ----------
    wave : np.ndarray
        The wavelength array in microns to sample the output transmission spectrum on.
    resolution : float | None
        The spectral resolution. If None, no convolution is performed.
        Default is None.
    airmass : float, optional
        The airmass. Default is 1.

    Returns
    -------
    spec : np.ndarray
        The convolved and resampled transmission spectrum.
    """
    tapas_filepth = _get_tapas_filepath()
    df = pd.read_csv(tapas_filepth, delimiter=',', usecols=(0, 1), names=["wave", "spec"], header=0)
    tapas_wave = np.array(df.wave)
    tapas_spec = np.array(df.spec)
    dw = wave[1] - wave[0]
    good = np.where((tapas_wave >= wave[0] - 10*dw) & (tapas_wave <= wave[-1] + 10*dw))[0]
    # TODO: Bin, don't convolve this onto wavegrid.
    tapas_wave = tapas_wave[good]
    tapas_spec = tapas_spec[good]
    #tapas_wave_uniform = np.linspace(tapas_wave[0], tapas_wave[-1], len(tapas_wave))
    #tapas_spec_uniform = np.interp(tapas_wave_uniform, tapas_wave, tapas_spec)
    #dw = tapas_wave_uniform[1] - tapas_wave_uniform[0]
    #flux_scale = dw / (wave[1] - wave[0])
    #spec = rebin_spectrum(tapas_wave_uniform, tapas_spec_uniform * flux_scale, wave)
    spec, *edges = rebin_spectrum(tapas_wave, tapas_spec * np.gradient(tapas_wave), wave, return_edges=True)
    spec /= (edges[1][1:] - edges[1][:-1])
    # import matplotlib
    # matplotlib.use('QTAGG')
    # import matplotlib.pyplot as plt
    # plt.plot(tapas_wave, tapas_spec)
    # plt.plot(wave, spec)
    # plt.show()
    # breakpoint()
    if resolution is not None:
        spec = convolve_spectrum(wave, spec, resolution=resolution)
    spec **= airmass
    out = dict( # NOTE: Keep dict when we eventually return species separately
        wave=wave,
        sky_trans=spec,
    )
    return out


def get_maunakea_spectral_sky_emission(
    wave : np.ndarray,
    resolution : float,
    T_tel : float = 275, T_atm : float = 258, T_aos : float = 243, T_zod : float = 5800,
    Em_tel : float = 0.09, Em_atm : float = 0.2, Em_aos : float = 0.01, # Em_zod : float = 1.47E-12,
    ohsim : bool = True,
) -> dict:
    """
    Compute the Maunakea sky emission spectrum.

    Parameters
    ----------
    wave : np.ndarray
        The wavelength array.
    resolution : float
        The spectral resolution (taken to be constant across bandpass here).
    T_tel : float, optional
        The telescope temperature. Defaults to 275.
    T_atm : float, optional
        The atmospheric temperature. Defaults to 258.
    T_aos : float, optional
        The AO temperature. Defaults to 243.
    T_zod : float, optional
        The zodiacal light temperature. Defaults to 5800.
    Em_tel : float, optional
        The telescope emission coefficient. Defaults to 0.09.
    Em_atm : float, optional
        The atmospheric emission coefficient. Defaults to 0.2.
    Em_aos : float, optional
        The AO emission coefficient. Defaults to 0.01.
    ohsim : bool, optional
        If True, include the simulated OH lines. Defaults to True.

    Returns
    -------
    sky_data : dict
        A dictionary containing the sky emission spectrum and its components:
        - 'wave': The wavelength array in microns.
        - 'sky_em': The total sky emission spectrum (photons / (s * m^2 * arcsec^2 * wavebin)).
        - 'bbtel': The telescope blackbody spectrum component (photons / (s * m^2 * arcsec^2 * wavebin)).
        - 'bbaos': The AO blackbody spectrum component (photons / (s * m^2 * arcsec^2 * wavebin)).
        - 'bbatm': The atmospheric blackbody spectrum component (photons / (s * m^2 * arcsec^2 * wavebin)).
        - 'bbzod': The zodiacal light blackbody spectrum component (photons / (s * m^2 * arcsec^2 * wavebin)).
    """

    sterr = 1 / 206265**2 # rad^2 / arcsec^2
    dw = np.nanmedian(np.diff(wave))

    # BB components
    wavem = wave * 1E-6
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
    if ohsim:
        ohspec = sim_ohlines(wave, resolution=resolution)
    else:
        ohspec = None

    # Combined sky emission
    # photons / (s * m^2 * arcsec^2 * wavebin)
    sky_em = bbspec + ohspec
    
    # Results
    sky_data = dict(
        wave=wave, sky_em=sky_em,
        bbtel=bbtel, bbaos=bbaos, bbatm=bbatm, bbzod=bbzod, bbspec=bbspec,
        ohspec=ohspec
    )

    return sky_data

def _get_gemini_sky_filepath():
    return os.path.join(_get_model_spectra_dir(), 'mk_skybg_zm_16_15_ph.fits')

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


def _get_ohlines_filepath():
    return os.path.join(_get_model_spectra_dir(), 'optical_ir_sky_lines.txt')

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
        warnings.warn(f"No OH lines found for this wavelength range ({wave[0]} - {wave[-1]} microns). Returning zero OH spectrum.")

    # Return
    return ohspec


def sim_ohline_lorenztian(
    wave : np.ndarray,
    wavecenter : float,
    flux : float,
    resolution : float,
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

    Returns
    -------
    spec : np.ndarray
        The simulated OH line spectrum sampled on wave.
        Output units are photons / (m^2 * s * arcsec^2 * wavebin).
    """
    # flux units: photons / (m^2 * s * arcsec^2)
    fwhm = wavecenter / resolution
    spec = Lorentz1D(amplitude=1, x_0=wavecenter, fwhm=fwhm)(wave)
    tot = np.sum(spec)
    # Out units = photons / (m^2 * s * arcsec^2 * wavebin)
    spec /= tot
    spec *= flux
    return spec