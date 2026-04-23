from ..sources.convolve import convolve_spectrum
from ..utils import rebin_spectrum, generate_wave_grid_for_filter
from .ohlines import sim_ohlines

import numpy as np
import scipy.constants
import pandas as pd
import os

from liger_iris_drp_resources import get_model_spectra_dir, load_filters_summary

c = scipy.constants.c  # m/s
h = scipy.constants.h  # J s
k = scipy.constants.k # J / K

__all__ = [
    'get_maunakea_sky_background',
    'get_maunakea_spectral_sky_transmission',
    'get_maunakea_spectral_sky_emission',
]

def _get_tapas_filepath():
    filepath = os.path.join(get_model_spectra_dir(), 'TAPAS_Maunakea_NIR.txt')
    return filepath

def get_maunakea_sky_background(
    wave : np.ndarray | None = None,
    filter_name : str | None = None,
    filter_info : dict | None = None,
    resolution : float | None = None,
    T_tel : float = 275, T_atm : float = 258, T_aos : float = 243,
    Em_tel : float = 0.09, Em_atm : float = 0.2, Em_aos : float = 0.01,
    ohsim : bool = True,
    airmass : float = 1,
    plate_scale : float | None = None
):
    
    if resolution is None:
        resolution = 10_000
    
    if wave is None:
        if filter_info is None:
            if filter_name is None:
                raise ValueError("Must provide either wave, filter_info, or filter_name.")
            filter_info = load_filters_summary(filter_name)

        wave = generate_wave_grid_for_filter(filter_info, resolution)
    
    sky_em = get_maunakea_spectral_sky_emission(
        wave=wave, resolution=resolution,
        T_tel=T_tel, T_atm=T_atm, T_aos=T_aos,
        Em_tel=Em_tel, Em_atm=Em_atm, Em_aos=Em_aos,
        ohsim=ohsim, plate_scale=plate_scale
    )
    sky_trans = get_maunakea_spectral_sky_transmission(
        wave=wave, resolution=resolution,
        airmass=airmass
    )

    #sky_photon_flux = np.sum(sky_em['sky_em'] * sky_trans['sky_trans'])
    sky_em_tot = np.sum(sky_em['sky_em'])
    sky_trans_mean = np.mean(sky_trans['sky_trans'])

    return {
        'sky_em_rate_bandpass_tot': sky_em_tot,
        'sky_trans_bandpass_mean': sky_trans_mean,
        **sky_em,
        **sky_trans,  # overwrites duplicates from sky_em
    }


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
    resolution : float = None,
    T_tel : float = 275, T_atm : float = 258, T_aos : float = 243,
    Em_tel : float = 0.09, Em_atm : float = 0.2, Em_aos : float = 0.01,
    ohsim : bool = True,
    plate_scale : float | None = None
) -> dict:
    """
    Compute the Maunakea sky emission spectrum.

    Parameters
    ----------
    wave : np.ndarray
        The wavelength array.
    resolution : float | None
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
    plate_scale : float, optional.
        Plate scale in arcsec / pixel (or spaxel). If provided, apply the plate scale correction to convert from per arcsec^2 to per pixel.
        Default is None.

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
    #bbzod = s1 / (np.exp(s2 / T_zod) - 1) # Zodiacal light blackbody spectrum
    bbtel /= Ephot # photons / (s * m * m^2 * rad^2)
    bbaos /= Ephot
    bbatm /= Ephot
    #bbzod /= Ephot
    bbtel /= 1E6 # photons / (s * micron * m^2 * rad^2)
    bbaos /= 1E6
    bbatm /= 1E6
    #bbzod /= 1E6
    bbtel *= sterr # photons / (s * micron * m^2 * arcsec^2)
    bbaos *= sterr
    bbatm *= sterr
    #bbzod *= sterr
    bbtel *= dw # photons / (s * m^2 * arcsec^2 * wavebin)
    bbaos *= dw
    bbatm *= dw
    #bbzod *= dw
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

    if plate_scale is not None:
        sky_em *= plate_scale**2 # Integrate over 2D pixel (photons / sec / m^2)
        bbtel *= plate_scale**2
        bbaos *= plate_scale**2
        bbatm *= plate_scale**2
        #bbzod *= plate_scale**2
        bbspec *= plate_scale**2
        ohspec *= plate_scale**2
    
    # Results
    sky_data = dict(
        wave=wave, sky_em=sky_em,
        bbtel=bbtel, bbaos=bbaos, bbatm=bbatm, bbspec=bbspec,
        ohspec=ohspec
    )

    return sky_data
