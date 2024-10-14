import numpy as np
import matplotlib.pyplot as plt
from .convolve import convolve_point_source


# Positions in detector coordinates
def make_point_sources_imager(
        xdet : np.ndarray, ydet : np.ndarray, photon_fluxes : np.ndarray,
        psfs : list[np.ndarray],
        size : tuple[int, int],
        image_out : np.ndarray | None = None,
    ) -> np.ndarray:
    """
    Parameters:
    xdet (np.ndarray): The x (horizontal) positions of each source in units of detector pixels.
    xdet (np.ndarray): The y (vertical) positions of each source in units of detector pixels.
    photon_fluxes (np.ndarray): The photon fluxes of each source in units of photons / sec / m^2.
    psfs (list[np.ndarray]): The PSF image for each source. The PSF can be of arbitrary size but must be on the correct scale, and is assumed to be centered in the image.
    size (tuple[int, int]): The output image shape.
    image_out (np.ndarray | None): If provided, sources are added into this image, optional.

    Returns:
    np.ndarray: The output image in units of photons / sec / m^2.
    """

    # Number of sources
    assert len(xdet) == len(ydet) == len(photon_fluxes) == len(psfs)
    n_sources = len(xdet)

    # Output image in units phot / s / m^2
    if image_out is None:
        image_out = np.zeros(shape=size)

    # Loop over point sources
    for i in range(n_sources):

        # Print
        print(f"Convolving source {i + 1}:")
        print(f"  Xdet = {xdet[i]}, Ydet = {ydet[i]}, Flux = {photon_fluxes[i]} photons / sec / m^2")

        # Convolve with PSF
        image_i = convolve_point_source(xdet[i], ydet[i], photon_fluxes[i], psfs[i], size=size)

        # Inject into image
        image_out += image_i

    # Return
    return image_out

