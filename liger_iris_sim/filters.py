import numpy as np


def read_iris_filter_info(filename : str):

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

    # Return
    return filter_data


def get_iris_filter_data(filename : str, filter : str):

    # Read filter data
    filter_data = read_iris_filter_info(filename)
    filters_lowercase = np.array([f.lower() for f in filter_data['filter']])

    # Match
    index = np.where(filter.lower().strip() == filters_lowercase)[0][0]
    
    # Get this row
    filter_data = dict(
        zip(
            filter_data.keys(),
            [filter_data[key][index] for key in filter_data]
        )
    )

    # Return
    return filter_data