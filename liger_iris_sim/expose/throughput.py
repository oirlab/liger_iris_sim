import numpy as np
from liger_iris_drp_resources.throughput import load_throughputs

from numbers import Number

__all__ = [
    'compute_throughput',
]

def _compute_throughput(
    inst : float = 1.0, tel : float = 1.0, ao : float = 1.0, filt : float = 1.0
) -> float:
    return tel * ao * filt * inst


def compute_throughput(
    instrument_name : str,
    instrument_mode : str,
    wave : float,
    ifs_mode : str | None = None,
    instrument_only : bool = False,
    tel : float = 0.8, ao : float = 0.65, filt : float | str = None,
) -> float:
    """
    Compute the total throughput for the given mode and wavelength.

    Parameters
    ----------
    instrument_name : str
        The instrument name ('Liger' or 'IRIS').
    instrument_mode : str
        The mode ('img', 'ifs').
    wave : float
        The wavelength in microns.
    ifs_mode : str | None
        The IFS mode ('slicer', 'lenslet') if mode is 'ifs'.
        Must be provided if mode is 'ifs'.
    instrument_only : bool
        If True, return only the instrument throughput. Default is False.
    tel : float
        The telescope throughput. Default is 0.8.
    ao : float
        The AO throughput. Default is 0.65.
    filt : float | str
        The filter throughput. Default is the maximum filter throughput.
    """
    waves_tput, inst_tput = load_throughputs(
        instrument_name=instrument_name,
        instrument_mode=instrument_mode,
        ifs_mode=ifs_mode
    )
    _inst_tput = np.interp(wave, waves_tput, inst_tput)

    # Get the filter throughput at this wavelength if not provided
    if isinstance(filt, str):
        _, filt = get_filter_throughput(filter_name=filt, wave=wave)
    elif filt is None:
        raise ValueError("Filter throughput must be provided as a float or filter name string")

    # Instrument-only throughput or total throughput
    if instrument_only:
        return _inst_tput
    else:
        return _compute_throughput(inst=_inst_tput, tel=tel, ao=ao, filt=filt)


def get_filter_throughput(
    filter_name : str,
    wave : float | None = None
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Get the filter throughput at a given wavelength or array of wavelengths.

    Parameters
    ----------
    filter_name : str
        The name of the filter.
    wave : float | np.ndarray, optional
        The wavelength(s) in microns at which to evaluate the filter throughput.
        If None, the wavelength at which the filter transmission is maximum will be used.
        Default is None.

    Returns
    -------
    wave : float | np.ndarray
        The wavelength in microns at which the filter throughput is evaluated.
    tput_filt : float | np.ndarray
        The filter throughput at the specified wavelength(s).
    """
    
    # Load the filter transmission curve
    from liger_iris_drp_resources import load_filter_transmission_curve
    filt_wave, filt_trans = load_filter_transmission_curve(filter_name)

    if wave is None:
        k = np.argmax(filt_trans)
        wave = filt_wave[k]
        tput_filt = filt_trans[k]
        return wave, tput_filt
    if isinstance(wave, (Number, np.ndarray)):
        tput_filt = np.interp(wave, filt_wave, filt_trans)
        return wave, tput_filt