import numpy as np

def compute_imager_eff(wave : float, tel=0.91, ao=0.8, filt=0.9):
    waves_eff = np.array([830, 900, 2000, 2200, 2300, 2412])
    imager_eff = np.array([0.631, 0.772, 0.772, 0.813, 0.763, 0.728] )
    efftot = tel * ao * filt * imager_eff
    efftot = np.interp(wave, waves_eff, efftot)
    return efftot


def compute_ifu_eff(wave : float, tel=0.91, ao=0.8, filt=0.9):
    waves_eff = np.array([830, 900, 2000, 2200, 2300, 2412])
    ifu_eff = np.array([0.631, 0.772, 0.772, 0.813, 0.763, 0.728] )
    efftot = tel * ao * filt * ifu_eff
    efftot = np.interp(wave, waves_eff, efftot)
    return efftot