
# Detector size (one detector)
imager_detector_size = (4096, 4096)
ifu_detector_size = (4096, 4096)

# plate scaling (only option for IRIS imager)
imager_plate_scale = 0.004
ifu_lenslet_plate_scales = [0.004, 0.009]
ifu_slicer_plate_scales = [0.025, 0.05]

# Bottom left corner of detector in spatial coords
imager_offset_coords = (0.6, 0.6)

# Dark current
imager_dark_current = 0.002 # e- / s
ifu_dark_current = 0.002 # e- / s

# Read noise
imager_read_noise = 5 # e-
ifu_read_noise = 5 # e-

# TMT coll area
tmt_collarea = 630 # m^2

# TMT coll diameter
tmt_colldiam = 30 # m

# Gain
imager_gain = 3.04 # e- / ADU
ifu_gain = 3.04 # e- / ADU

# BB filters
bb_filters = ['Zbb', 'Ybb', 'Jbb', 'Hbb', 'Kbb']