__all__ = ['LIGER_PROPS']

LIGER_PROPS = {
    'imager_plate_scale': 0.01, # mas
    'ifs_lenslet_plate_scales': [0.01, 0.02], # mas
    'ifs_slicer_plate_scales': [0.025, 0.05], # mas
    'keck_colldiam': 10, # m
    'keck_collarea': 76, # m^2
    'gain': 1.0,
    'dark_current': 0.025, # e- / s
    'read_noise': 9, # e-
    'imager_detector_size': (2048, 2048),
    'ifs_detector_size': (4096, 4096),
}