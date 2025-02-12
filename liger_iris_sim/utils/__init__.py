from .binning import *
from .spec_utils import *
from .filter_utils import *
from .grating_utils import *

__all__ = [
    'rebin', 'frebin', 'bin_image', 'crop_AO_psf',
    'compute_filter_zeropoint',
    'convolve_spectrum',
    '_resolve_mode',
    'load_filter_data',
    'load_filter_transmission_curve',
    'load_grating_data',
]

def _resolve_mode(mode : str):
    if mode.lower() in ('imager', 'img'):
        return 'img'
    elif mode.lower() == 'slicer':
        return 'slicer'
    elif mode.lower() == 'lenslet':
        return 'lenslet'
    else:
        raise ValueError(f"Unknown mode: {mode}")