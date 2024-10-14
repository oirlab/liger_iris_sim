import numpy as np
import matplotlib.pyplot as plt
from .convolve import convolve_point_source

# Positions in detector coordinates
def make_point_sources_ifu_cube(
        xdet : np.ndarray, ydet : np.ndarray,
        templates : list[np.ndarray],
        psfs : list[np.ndarray],
        size : tuple[int, int],
        cube_out : np.ndarray | None = None,
    ) -> np.ndarray:
    """
    Parameters:
    xdet (np.ndarray): The x positions in detector coords.
    ydet (np.ndarray): The x positions in detector coords.
    templates (list[np.ndarray]): A list of templates for each source. Column [:, 0] is wavelength in nm, [:, 1] is flux in photons / sec / m^2 for each wavelength bin.
    psfs (list[np.ndarray])

    Returns:
    np.ndarray: The image cube in photons / sec / m^2 for each wavelength bin.
    """

    # Number of sources
    assert len(xdet) == len(ydet) == len(templates) == len(psfs)
    n_sources = len(xdet)
    
    # Number of wavelengths
    n_wavelengths = len(templates[0][0])

    # Output cube in units phot / s / nm / m^2
    if cube_out is None:
        cube_out = np.zeros(shape=(size[0], size[1], n_wavelengths))

    # Loop over point sources
    for i in range(n_sources):
        for j in range(n_wavelengths):

            # Print
            print(f"Processing source {i + 1} : X = {xdet[i]}, Y = {ydet[i]}, Wavelength = {templates[i][0][j]} microns")

            # Convolve with PSF
            image_ij = convolve_point_source(xdet[i], ydet[i], templates[i][1][j], psfs[i], size=size)

            # Inject into image
            cube_out[:, :, j] += image_ij
        
    # Return the cube
    return cube_out