import numpy as np


# def compute_imager_eff(wave : float, tel=0.8, ao=0.65, filt=0.9):
#     waves_eff = np.array([830, 900, 2000, 2200, 2300, 2412])
#     imager_eff = np.array([0.631, 0.772, 0.772, 0.813, 0.763, 0.728] )
#     efftot = tel * ao * filt * imager_eff
#     efftot = np.interp(wave, waves_eff, efftot)
#     return efftot


# def compute_ifu_eff(wave : float, tel=0.8, ao=0.65, filt=0.9):
#     waves_eff = np.array([830, 900, 2000, 2200, 2300, 2412])
#     ifu_eff = np.array([0.631, 0.772, 0.772, 0.813, 0.763, 0.728])
#     efftot = tel * ao * filt * ifu_eff
#     efftot = np.interp(wave, waves_eff, efftot)
#     return efftot


def compute_eff(
        wave : float,
        mode : str,
        inst : float | None = None,
        tel : float = 0.8, ao : float = 0.65, filt : float = 0.9
    ):
    if mode.lower() == 'imager' and inst is None:
        waves_eff = np.array([0.81, 0.839275, 0.86785, 0.896425, 0.93, 0.96, 0.995, 1.02, 1.045, 1.1, 1.15, 1.207, 1.248, 1.289, 1.34, 1.47, 1.559, 1.633, 1.71, 1.79, 1.99, 2.1, 2.2, 2.3, 2.41])
        inst_eff = np.array([0.6209, 0.6226, 0.6589, 0.66, 0.7289, 0.7296, 0.7312, 0.7294, 0.7282, 0.7296, 0.7335, 0.7342, 0.7337, 0.7337, 0.7314, 0.7305, 0.7336, 0.7345, 0.7362, 0.7354, 0.7284, 0.7598, 0.7614, 0.7621, 0.762])
    elif mode.lower() == 'slicer' and inst is None:
        waves_eff = np.array([0.81, 0.851225, 0.89175, 0.932275, 0.987, 0.99, 1.042, 1.0965, 1.135, 1.207, 1.248, 1.289, 1.41, 1.47, 1.559, 1.633, 1.71, 1.82, 1.96, 2.1, 2.2, 2.3, 2.42])
        inst_eff = np.array([0.307, 0.3277, 0.3399, 0.3797, 0.3869, 0.3869, 0.3788, 0.3825, 0.3867, 0.395, 0.3939, 0.3939, 0.3836, 0.3855, 0.3936, 0.3958, 0.4012, 0.3957, 0.3782, 0.4225, 0.4239, 0.4261, 0.4216])
    elif mode.lower() == 'lenslet' and inst is None:
        waves_eff = np.array([0.81, 0.851225, 0.89175, 0.932275, 0.987, 0.99, 1.042, 1.0965, 1.135, 1.207, 1.248, 1.289, 1.41, 1.47, 1.559, 1.633, 1.71, 1.82, 1.96, 2.1, 2.2, 2.3, 2.42])
        inst_eff = np.array([0.3612, 0.3846, 0.3972, 0.4425, 0.45, 0.45, 0.4425, 0.4459, 0.4498, 0.4575, 0.4568, 0.4568, 0.4472, 0.449, 0.4565, 0.4586, 0.4642, 0.4594, 0.4419, 0.4754, 0.4776, 0.4795, 0.4775])  
    else:
        raise ValueError(f"Invalid mode for Liger {mode=}")
    _inst_eff = np.interp(wave, waves_eff, inst_eff)
    efftot = tel * ao * filt * _inst_eff
    return efftot