import numpy as np
from astropy.io import fits
import scipy.interpolate as interpolate
import scipy.constants
from astropy.modeling.models import Gaussian1D

c = scipy.constants.c  # m/s
h = scipy.constants.h  # J s
k = scipy.constants.k # J / K

from filters import get_iris_filter_data


def get_sky_background(
        filter : str, collarea : float, resolution : float | None = None, ohsim : bool = True,
        T_tel : float = 275, T_atm : float = 258, T_aos : float = 243, T_zod : float = 5800,
        Em_tel : float = 0.09, Em_atm : float = 0.2, Em_aos : float = 0.01, Em_zod : float = 1.47E-12,
        airmass : str = '10', vapor : str = '15',
        simdir : str = '/data/group/data/iris/sim/',
    ):

    # Filter info
    filter_filename = f'{simdir}/info/iris_filter_{filter}.dat'
    filter_data = get_iris_filter_data(filter_filename, filter)
    backmag = filter_data['backmag']
    wavemin = filter_data['wavemin']
    wavemax = filter_data['wavemax']

    # Determine the length of the cube, dependent on filter
    nx_spectrum = int(np.ceil(np.log10(wavemax / wavemin) / np.log10(1 + 1 / resolution)))
    dx_spectrum = (wavemax - wavemin) / nx_spectrum # nm / channel
    
    # Wavelength grid for final sky background spectrum
    waves = wavemin + dx_spectrum * np.arange(nx_spectrum)
    
    # sterradians / arcsecond^2
    sterrad = 2.35E-11

    # Generate arrays
    bbtel = np.zeros(nx_spectrum)  # telescope blackbody spectrum
    bbaos = np.zeros(nx_spectrum)  # AO blackbody spectrum
    bbatm = np.zeros(nx_spectrum)  # Atm blackbody spectrum
    bbzod = np.zeros(nx_spectrum)  # Zodiacal light blackbody spectrum
    bbspec = np.zeros(nx_spectrum)  # Total blackbody spectrum
    contspec = np.zeros(nx_spectrum)  # continuum of sky

    # Loop over the first and last pixel of the complete spectrum
    for i in range(nx_spectrum):

        # Wavelength
        wave = waves[i]
      
        # Thermal Blackbodies (Tel, AO system, atmosphere) in erg s^-1 cm^-2 cm^-1 sr^-1
        wave_meters = wave * 1E-9
        s1 = 2 * h * c**2 / wave_meters**5
        bbtel[i] = s1 / (np.exp(h * c / (wave_meters * k * T_tel)) - 1)
        bbaos[i] = s1 / (np.exp(h * c / (wave_meters * k * T_aos)) - 1)
        bbatm[i] = s1 / (np.exp(h * c / (wave_meters * k * T_atm)) - 1)
        bbzod[i] = s1 / (np.exp(h * c / (wave_meters * k * T_zod)) - 1)

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
        ohspec = sim_ohlines(waves, n_lines=30_000, background=1E-8, simdir=simdir)
    else:

        # Load Gemini file
        gemini_file = simdir + 'skyspectra/mk_skybg_zm_16_15_ph.fits'
        with fits.open(gemini_file) as f:
            gemini_spec = f[0].data
            header = f[0].header

        # Wave grid for Gemini spectrum
        cdelt1 = header["cdelt1"]
        crval1 = header["crval1"]
        nx = gemini_spec.shape[0]
        gemini_wave = np.arange(nx) * cdelt1 + crval1

        # Convolve 
        delt = 2 * (waves[1] - waves[0]) / (gemini_wave[1] - gemini_wave[0])
        if delt > 1:
            stddev = delt / 2 * np.sqrt(2 * np.log(2))
            x = np.arange(4 * int(delt) + 1) - 2 * int(delt)
            psf = Gaussian1D(amplitude=1, stddev=stddev)(x)
            psf /= psf.sum()
            gemini_spec = np.convolve(gemini_spec, psf, mode='same')
        
        # Interpolate onto our wave grid
        ohspec = np.interp(waves, gemini_wave, gemini_spec)

    # Results
    out = {
        "waves" : waves,
        "bbtel" : bbtel,
        "bbaos" : bbaos,
        "bbatm" : bbatm,
        "bbzod" : bbzod,
        "bbspec" : bbspec,
        "oh" : ohspec
    }

    return out


def sim_ohlines(waves, n_lines=30_000, background=1E-8, simdir='/data/group/data/iris/sim/'):

    # File of OH lines
    oh_lines_file = simdir + '/info/optical_ir_sky_lines.dat'

    # OH spectrum
    ohspec = np.zeros(len(waves))

    # read OH line file
    # Units are microns
    line_centers, line_strengths = np.loadtxt(oh_lines_file, unpack=True, comments='#')
    line_centers *= 1E3 # microns to nm
    wi, wf = waves[0], waves[-1]
    good = np.where((line_centers >= waves[0]) & (line_centers <= waves[-1]) & (line_strengths > 0))[0]
    n_good = len(good)
     
    # Build spectrum
    if n_good > 0:
        line_centers, line_strengths = line_centers[good], line_strengths[good]
        for i in range(n_good):
           ohspec += sim_ohlines_lorenztian(waves, line_centers[i], n_lines, background, strength = line_strengths[i])

    # Return
    return ohspec


def sim_ohlines_lorenztian(waves, wavecenter, n_lines, background, strength):
    omega = wavecenter / (n_lines * np.pi * np.sqrt(2))
    spec = omega**2 / ((waves - wavecenter)**2 + omega**2) + background
    spec *= strength / (omega * np.pi)
    return spec


def sim_inst_scatter(waves, wavecenter, n_lines=30_000, background=1E-8, strength=None):
    omega = wavecenter / (n_lines * np.pi * np.sqrt(2))
    spec = omega ** 2 / ((waves - wavecenter) ** 2.0 + omega ** 2.0) + background
    spec *= strength / (omega * np.pi)
    return spec