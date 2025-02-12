import numpy as np
from astropy import units as u
from synphot import SourceSpectrum
import os

__all__ = ['compute_filter_zeropoint', 'compute_filter_mag', 'load_filter_data', 'load_filter_transmission_curve']


def compute_filter_zeropoint(filter_wave : np.ndarray, filter_trans : np.ndarray) -> float:
    """
    Compute the zero point of the filter in phot/s/m^2.

    Args:
        filter_wave (np.ndarray): The filter curve wave grid (microns).
        filter_trans (np.ndarray): The filter curve transmission (0-1).

    Returns:
        float: The zero point in phot/s/m^2.
    """

    # Load Vega spectrum from synphot
    vega_spectrum = SourceSpectrum.from_vega()

    # Convert wavelength to microns
    vega_wave = vega_spectrum.waveset.to(u.micron).value  # microns

    # Get Vega photon flux density
    vega_flux_photlam = vega_spectrum(vega_wave * u.micron).value # phot/s/cm^2/Ang
    vega_flux_photlam *= 1E4  # phot/s/cm^2/micron
    vega_flux_photlam *= 100**2 # phot/s/m^2/micron

    # Interpolate Vega flux to filter wavelengths
    vega_flux_interp = np.interp(filter_wave, vega_wave, vega_flux_photlam, left=0, right=0)

    # Integral of flux over bandpass (photons/s/m^2)
    zp = np.trapz(vega_flux_interp * filter_trans, filter_wave)

    # Return zp in phot/s/m^2
    return zp

def compute_filter_mag(photon_flux : float, zp : float) -> float:
    """
    Compute the magnitudes of the filter curve.

    Args:
        photon_flux (float): The integrated photon flux across the bandpass in phot/s/m^2.
        zp (float): The zero point of the filter in phot/s/m^2.

    Returns:
        float: The magnitude
    """
    mag = -2.5 * np.log10(photon_flux / zp)
    return mag


def load_filter_data():
    """
    Loads the filter summary file.

    Returns:
        dict: The filter data. Keys are filter names.
            Values are also dicts with basic info for the filter.
    """
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(module_dir, 'data/filters/filters_summary.txt')
    data = np.genfromtxt(filename, dtype=None, names=True, delimiter=',', encoding='utf-8')
    out = {}
    for i, filt in enumerate(data['filter']):
        out[filt] = {key : data[key][i] for key in data.dtype.names}
    return out


def load_filter_transmission_curve(filter_name : str):
    """
    Load the transmission curve for a filter.

    Args:
        filter_name (str): The filter name.

    Returns:
        np.ndarray: The wavelength grid (microns).
        np.ndarray: The transmission curve (0-1).
    """
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(module_dir, f'data/filters/iris_filter_{filter_name}.txt')
    filter_wave, filter_trans = np.loadtxt(filename, delimiter=',', unpack=True)
    return filter_wave, filter_trans