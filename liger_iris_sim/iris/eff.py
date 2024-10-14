import numpy as np

def compute_imager_eff(
        wave : float, tel : float = 0.91, ao : float = 0.8, filt : float = 0.9
    ):
    waves_eff = np.array([830, 900, 2000, 2200, 2300, 2412])
    imager_eff = np.array([0.631, 0.772, 0.772, 0.813, 0.763, 0.728])
    imager_eff = np.interp(wave, waves_eff, imager_eff)
    efftot = tel * ao * filt * imager_eff
    return efftot