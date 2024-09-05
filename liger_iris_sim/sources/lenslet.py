import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.constants

c = scipy.constants.c # m/s
h = scipy.constants.h # J s



# Positions in detector coordinates
def make_point_sources_ifu_lenslet_cube(
        xdet : np.ndarray, ydet : np.ndarray, templates : list[tuple[np.ndarray, np.ndarray]],
        size : tuple[int, int],
        psfs : list[np.ndarray],
        cube_out : np.ndarray | None = None,
    ):

    # Number of sources
    assert len(xdet) == len(ydet) == len(templates) == len(psfs)
    n_sources = len(xdet)
    
    # Number of wavelengths
    n_wavelengths = len(templates[0][0])

    # Output image in units phot / s / m^2
    if cube_out is None:
        cube_out = np.zeros(shape=(size[0], size[1], len(templates[0][0])))

    # Loop over point sources
    for i in range(n_sources):
        for j in range(n_wavelengths):

            # Print
            print(f"Processing source {i + 1} : X = {xdet[i]}, Y = {ydet[i]}, Wavelength = {templates[i][0][j]} nm")

            # Flux
            #flux = 
        
            flux = 100

            # Convolve with PSF
            image_ij = convolve_point_source(xdet[i], ydet[i], flux, psfs[i], size=size)

            # Inject into image
            cube_out[:, :, j] += image_ij
        
    return cube_out