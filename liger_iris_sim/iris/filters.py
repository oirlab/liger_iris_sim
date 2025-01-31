import numpy as np
import os

__all__ = ['get_filter_data']


def get_filter_data():
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