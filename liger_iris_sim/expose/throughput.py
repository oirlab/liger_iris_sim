import numpy as np
from liger_iris_drp_resources.throughput import load_throughputs

__all__ = [
    'compute_liger_throughput',
    'compute_iris_throughput',
]


def compute_liger_throughput(
    mode : str,
    wave : float,
    ifs_mode : str | None = None,
    tel : float = 0.8, ao : float = 0.65, filt : float = 0.9
):
    """
    Compute the total throughput for the given mode and wavelength.

    Parameters
    ----------
    mode : str
        The mode ('imager', 'ifs').
    wave : float
        The wavelength in microns.
    ifs_mode : str | None
        The IFS mode ('slicer', 'lenslet') if mode is 'ifs'.
        Must be provided if mode is 'ifs'.
    tel : float
        The telescope throughput. Default is 0.8.
    ao : float
        The AO throughput. Default is 0.65.
    filt : float
        The filter throughput. Default is 0.9.

    Returns
    -------
    tput_tot : float
        The total throughput.
    """
    waves_tput, inst_tput = load_throughputs(instrument='Liger', mode=mode, ifs_mode=ifs_mode)
    _inst_tput = np.interp(wave, waves_tput, inst_tput)
    tput_tot = tel * ao * filt * _inst_tput
    return tput_tot



def compute_iris_throughput(
    mode : str,
    wave : float,
    ifs_mode : str | None = None,
    tel : float = 0.8, ao : float = 0.65, filt : float = 0.9
):
    """
    Compute the total throughput for the given mode and wavelength.

    Parameters
    ----------
    mode : str
        The mode ('imager', 'ifs').
    wave : float
        The wavelength in microns.
    ifs_mode : str | None
        The IFS mode ('slicer', 'lenslet') if mode is 'ifs'.
        Must be provided if mode is 'ifs'.
    tel : float
        The telescope throughput. Default is 0.8.
    ao : float
        The AO throughput. Default is 0.65.
    filt : float
        The filter throughput. Default is 0.9.

    Returns
    -------
    tput_tot : float
        The total throughput.
    """
    waves_tput, inst_tput = load_throughputs(instrument='IRIS', mode=mode, ifs_mode=ifs_mode)
    _inst_tput = np.interp(wave, waves_tput, inst_tput)
    tput_tot = tel * ao * filt * _inst_tput
    return tput_tot