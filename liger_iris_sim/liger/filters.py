import numpy as np


def read_filter_info(filename : str):

    # Load filter data
    info = np.genfromtxt(filename, dtype=str)

    # Convert to dict with these keys
    keys = [
        "filter", "wavemin", "wavemax", "wavecenter",
        "bandwidth", "backmag", "imagmag", "zp", "zpphot",
        "psfname", "psfsamp", "psfsize", "filterfiles"
    ]
    filter_data = dict(zip(keys, info.T))

    # Convert to numbers
    for key in filter_data:
        if filter_data[key][0][0].isdigit():
            filter_data[key] = filter_data[key].astype(float)

    # Convert wavelengths to microns
    filter_data['wavemin'] /= 1E4
    filter_data['wavecenter'] /= 1E4
    filter_data['wavemax'] /= 1E4
    filter_data['bandwidth'] /= 1E4

    # Return
    return filter_data


def get_filter_data(filename : str, filt : str | None = None, bb : bool | None = False):
    """
    Gets the Liger filters
    
    Parameters:
    filename (str): The filename containing the info for all filters.
    """

    # Read filter data
    filter_data_raw = read_filter_info(filename)

    # Transpose so that filter names are the keys
    filter_data = {}
    for i, f in enumerate(filter_data_raw['filter']):
        filter_data[f] = dict(
            zip(
                filter_data_raw.keys(),
                [filter_data_raw[key][i] for key in filter_data_raw]
            )
        )

    # What filters to return
    if filt is not None:
        return filter_data[filt]
    elif bb:
        return {
            key : filter_data[key] for key in filter_data.keys() if key.lower().endswith('bb')
        }
    else:
        return filter_data