from liger_iris_drp_resources.model_spectra import download_model_spectra
from liger_iris_sim.sky import get_maunakea_spectral_sky_emission, get_maunakea_spectral_sky_transmission

import os
import numpy as np

def test_sky_background():
    
    download_model_spectra()
    
    wave = np.linspace(1.1, 1.15, 1000) # microns
    res = 4000
    
    # Emission
    sky_em = get_maunakea_spectral_sky_emission(wave=wave, resolution=res)

    assert len(sky_em['sky_em']) == len(wave)
    
    # Transmission
    sky_trans = get_maunakea_spectral_sky_transmission(wave=wave, resolution=res)

    assert len(sky_trans['sky_trans']) == len(wave)