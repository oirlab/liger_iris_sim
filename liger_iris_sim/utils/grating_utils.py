import numpy as np
import os

__all__ = ['load_grating_data']


def load_grating_data():
    """
    Loads the grating summary data.

    Returns:
        dict: The grating data. Keys are grating names.
            Values are also dicts with basic info for the grating.
    """
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(module_dir, 'data/gratings/gratings_summary.txt')
    data = np.genfromtxt(filename, dtype=None, names=True, delimiter=',', encoding='utf-8')
    out = {}
    for i, filt in enumerate(data['grating']):
        out[filt] = {key : data[key][i] for key in data.dtype.names}
    return out