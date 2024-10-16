import numpy as np
from astropy.io import fits
import scipy.interpolate as interpolate
import scipy.constants
from astropy.modeling.models import Gaussian1D, Lorentz1D
import warnings

c = scipy.constants.c  # m/s
h = scipy.constants.h  # J s
k = scipy.constants.k # J / K


def get_maunakea_spectral_sky_background(
        wavelengths : np.ndarray, resolution : float, ohsim : bool = True,
        gemini_file : str | None = None,
        ohlines_file : str | None = None,
        T_tel : float = 275, T_atm : float = 258, T_aos : float = 243, T_zod : float = 5800,
        backmag : float = 22.0, imagmag : float = 20.0, zp : float = 25.0, zpphot : float = 25.0,
        Em_tel : float = 0.09, Em_atm : float = 0.2, Em_aos : float = 0.01, Em_zod : float = 1.47E-12,
    ):
    """
    Parameters:
    wavelengths (np.ndarray): blah.
    resolution (float): The spectral resolution (taken to be constant across bandpass here).
    """

    # Generate arrays
    bbtel = np.zeros(wavelengths.size)  # telescope blackbody spectrum
    bbaos = np.zeros(wavelengths.size)  # AO blackbody spectrum
    bbatm = np.zeros(wavelengths.size)  # Atm blackbody spectrum
    bbzod = np.zeros(wavelengths.size)  # Zodiacal light blackbody spectrum
    bbspec = np.zeros(wavelengths.size)  # Total blackbody spectrum

    sterr2 = (1 / 206265**2)

    # Loop over the first and last pixel of the complete spectrum
    for i in range(wavelengths.size):

        # Wavelength in microns and meters
        wave = wavelengths[i]
        wavem = wave * 1E-6
      
        # Thermal Blackbodies in units J / (s * m * m^2 * rad^2)
        s1 = 2 * h * c**2 / wavem**5
        s2 = h * c / (wavem * k)
        bbtel[i] = s1 / (np.exp(s2 / T_tel) - 1)
        bbaos[i] = s1 / (np.exp(s2 / T_aos) - 1)
        bbatm[i] = s1 / (np.exp(s2 / T_atm) - 1)
        bbzod[i] = s1 / (np.exp(s2 / T_zod) - 1)
        
        # Convert to photons / (s * micron * m^2 * rad^2) for Vega conversion below
        # Ephot = hc/lambda
        Ephot = h * c / wavem # J / photon
        bbtel[i] /= Ephot # photons / (s * m * m^2 * rad^2)
        bbaos[i] /= Ephot
        bbatm[i] /= Ephot
        bbzod[i] /= Ephot
        bbtel[i] /= 1E6 # photons / (s * micron * m^2 * rad^2)
        bbaos[i] /= 1E6
        bbatm[i] /= 1E6
        bbzod[i] /= 1E6
        bbtel[i] *= sterr2 # photons / (s * micron * m^2 * arcsec^2)
        bbaos[i] *= sterr2
        bbatm[i] *= sterr2
        bbzod[i] *= sterr2

        # Total BB together with emissivities from each component
        # in units photons / (s * m^2 * micron * arcsecond)
        # Only use the BB for the AO system and the telescope since 
        # the Gemini observations already includes the atmosphere
        bbspec[i] = bbtel[i] * Em_tel \
                  + bbaos[i] * Em_aos

    # OH lines
    if ohsim:
        ohspec = sim_ohlines(wavelengths, ohlines_file=ohlines_file, resolution=resolution)
    else:

        # Load Gemini file
        with fits.open(gemini_file) as f:
            gemini_spec = f[0].data
            header = f[0].header

        # Wave grid for Gemini spectrum
        cdelt1 = header["CDELT1"] * 1E-4
        crval1 = header["CRVAL1"] * 1E-4
        nx = gemini_spec.shape[0]
        gemini_wave = np.arange(nx) * cdelt1 + crval1

        # Convolve 
        delt = 2 * (wavelengths[1] - wavelengths[0]) / (gemini_wave[1] - gemini_wave[0])
        if delt > 1:
            stddev = delt / 2 * np.sqrt(2 * np.log(2))
            x = np.arange(4 * int(delt) + 1) - 2 * int(delt)
            psf = Gaussian1D(amplitude=1, stddev=stddev)(x)
            psf /= psf.sum()
            gemini_spec = np.convolve(gemini_spec, psf, mode='same')
        
        # Interpolate onto our wavelengths grid
        ohspec = np.interp(wavelengths, gemini_wave, gemini_spec)


    # Combined background
    background = bbspec + ohspec
    
    # Results
    out = dict(
        wavelengths=wavelengths, background=background,
        bbtel=bbtel, bbaos=bbaos, bbatm=bbatm, bbzod=bbzod, bbspec=bbspec,
        ohspec=ohspec
    )

    return out


def sim_ohlines(
        wavelengths : np.ndarray,
        resolution : float,
        ohlines_file : str,
    ):

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
    # flux units: photons / (m^2 * s * arcsec^2)
    fwhm = wavecenter / resolution
    gamma = fwhm / 2
    dl = np.median(np.diff(wavelengths))
    spec = Lorentz1D(amplitude=1, x_0=wavecenter, fwhm=fwhm)(wavelengths)
    spec /= gamma * np.pi
    spec *= dl * flux
    return spec