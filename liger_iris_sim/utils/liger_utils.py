__all__ = ['LIGER_PROPS']

LIGER_PROPS = {
    'imager_plate_scale': 0.01, # arcsec
    'ifs_lenslet_plate_scales': [0.01, 0.02], # arcsec
    'ifs_slicer_plate_scales': [0.025, 0.05], # arcsec
    'colldiam': 10.949, # m
    'collarea': 76, # m^2 (effective 9.84 m diameter)
    'gain': 1.0,
    'dark_current': 0.025, # e- / s
    'read_noise': 9, # e-
    'imager_detector_size': (2048, 2048),
    'ifs_detector_size': (4096, 4096),
}