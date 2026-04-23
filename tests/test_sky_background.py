from liger_iris_sim.sky import get_maunakea_sky_background

import os
import numpy as np

def test_sky_background():
    
    wave = np.linspace(1.1, 1.15, 1000) # microns
    res = 4000
    
    sky_data = get_maunakea_sky_background(
        wave=wave,
        filter_info=None,
        resolution=res
    )

    assert len(sky_data['sky_trans']) == len(wave)
    assert len(sky_data['sky_em']) == len(wave)

