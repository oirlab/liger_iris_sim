import numpy as np
from ..utils import _resolve_mode

def compute_eff(
        mode : str,
        wavelength : float,
        tel : float = 0.91, ao : float = 0.8, filt : float = 0.9
    ) -> float:
    """
    Compute the total efficiency for the given mode and wavelength.

    Args:
        mode (str): The mode ('img', 'slicer', 'lenslet').
        wavelength (float): The wavelength in nm.
        tel (float): The telescope efficiency.
        ao (float): The AO efficiency.
        filt (float): The filter efficiency.

    """
    mode = _resolve_mode(mode)
    if mode == 'img':
        return _compute_imager_eff(wavelength, tel=tel, ao=ao, filt=filt)
    elif mode in ('slicer', 'lenslet'):
        return _compute_ifu_eff(wavelength, tel=tel, ao=ao, filt=filt)

def _compute_imager_eff(
        wavelength : float, tel : float = 0.91, ao : float = 0.8, filt : float = 0.9
    ):
    waves_eff = np.array([830, 900, 2000, 2200, 2300, 2412])
    imager_eff = np.array([0.631, 0.772, 0.772, 0.813, 0.763, 0.728])
    imager_eff = np.interp(wavelength, waves_eff, imager_eff)
    efftot = tel * ao * filt * imager_eff
    return efftot


def _compute_ifu_eff(
        wavelength : float, tel : float = 0.91, ao : float = 0.8, filt : float = 0.9
    ):
    waves_eff = np.array([830, 900, 2000, 2200, 2300, 2412])
    ifu_eff = np.array([0.631, 0.772, 0.772, 0.813, 0.763, 0.728])
    ifu_eff = np.interp(wavelength, waves_eff, ifu_eff)
    efftot = tel * ao * filt * ifu_eff
    return efftot