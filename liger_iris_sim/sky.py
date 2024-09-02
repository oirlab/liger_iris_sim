import numpy as np
from astropy.io import fits
import scipy.interpolate as interpolate
import scipy.constants
from astropy.modeling.models import Gaussian1D

c = scipy.constants.c  # m/s
h = scipy.constants.h  # J s
k = scipy.constants.k # J / K


def get_maunakea_sky_background(
        wavelengths : np.ndarray, ohsim : bool = True,
        gemini_file : str | None = None,
        T_tel : float = 275, T_atm : float = 258, T_aos : float = 243, T_zod : float = 5800,
        backmag : float = 22.0, imagmag : float = 20.0, zp : float = 25.0, zpphot : float = 25.0,
        Em_tel : float = 0.09, Em_atm : float = 0.2, Em_aos : float = 0.01, Em_zod : float = 1.47E-12,
    ):
    
    # sterradians / arcsecond^2
    sterrad = 2.35E-11

    # Generate arrays
    bbtel = np.zeros(wavelengths.size)  # telescope blackbody spectrum
    bbaos = np.zeros(wavelengths.size)  # AO blackbody spectrum
    bbatm = np.zeros(wavelengths.size)  # Atm blackbody spectrum
    bbzod = np.zeros(wavelengths.size)  # Zodiacal light blackbody spectrum
    bbspec = np.zeros(wavelengths.size)  # Total blackbody spectrum
    contspec = np.zeros(wavelengths.size)  # continuum of sky

    # Loop over the first and last pixel of the complete spectrum
    for i in range(wavelengths.size):

        # Wavelength
        wave = wavelengths[i]
      
        # Thermal Blackbodies (Tel, AO system, atmosphere) in erg s^-1 cm^-2 cm^-1 sr^-1
        wm = wave * 1E-9
        s1 = 2 * h * c**2 / wm**5
        s2 = h * c / (wm * k)
        bbtel[i] = s1 / (np.exp(s2 / T_tel) - 1)
        bbaos[i] = s1 / (np.exp(s2 / T_aos) - 1)
        bbatm[i] = s1 / (np.exp(s2 / T_atm) - 1)
        bbzod[i] = s1 / (np.exp(s2 / T_zod) - 1)

        # Convert to photon s^-1 m^-2 nm^-1 sr-2 for Vega conversion below
        # 5.03e7 * lambda photons/erg (where labmda is in nm)
        bbtel[i] *= 5.03E7 * wave / 10
        bbaos[i] *= 5.03E7 * wave / 10
        bbatm[i] *= 5.03E7 * wave / 10
        bbzod[i] *= 5.03E7 * wave / 10

        # Total BB together with emissivities from each component in photons s^-1 m^-2 um^-1 arcsecond^-2
        # Only use the BB for the AO system and the telescope since 
        # the Gemini observations already includes the atmosphere
        # bbspec[i] = sterrad * (bbtel[i] * Em_tel + bbaos[i] * Em_aos + bbatm[i] * Em_atm + bbzod[i] * Em_zod)
        bbspec[i] = sterrad * (bbtel[i] * Em_tel + bbaos[i] * Em_aos)


    # OH lines
    if ohsim:
        ohspec = sim_ohlines(wavelengths, n_lines=30_000, background=1E-8, simdir=simdir)
    else:

        # Load Gemini file
        with fits.open(gemini_file) as f:
            gemini_spec = f[0].data
            header = f[0].header

        # Wave grid for Gemini spectrum
        cdelt1 = header["cdelt1"]
        crval1 = header["crval1"]
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

    # Results
    out = {
        "wavelengths" : wavelengths,
        "bbtel" : bbtel,
        "bbaos" : bbaos,
        "bbatm" : bbatm,
        "bbzod" : bbzod,
        "bbspec" : bbspec,
        "oh" : ohspec
    }

    return out


def sim_ohlines(wavelengths : np.ndarray, n_lines : int = 30_000, background : float | None = 1E-8, oh_lines_file : str | None = None):

    # OH spectrum
    ohspec = np.zeros(len(wavelengths))

    # read OH line file
    # Units are microns
    line_centers, line_strengths = np.loadtxt(oh_lines_file, unpack=True, comments='#')
    line_centers *= 1E3 # microns to nm
    good = np.where((line_centers >= wavelengths[0]) & (line_centers <= wavelengths[-1]) & (line_strengths > 0))[0]
    n_good = len(good)

    # Build spectrum
    if n_good > 0:
        line_centers, line_strengths = line_centers[good], line_strengths[good]
        for i in range(n_good):
           ohspec += sim_ohlines_lorenztian(wavelengths, line_centers[i], n_lines, background, strength = line_strengths[i])

    # Return
    return ohspec


def sim_ohlines_lorenztian(wavelengths : np.ndarray, wavecenter : float, n_lines : int, background : np.ndarray, strength : float):
    omega = wavecenter / (n_lines * np.pi * np.sqrt(2))
    spec = omega**2 / ((wavelengths - wavecenter)**2 + omega**2) + background
    spec *= strength / (omega * np.pi)
    return spec