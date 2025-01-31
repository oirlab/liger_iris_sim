import numpy as np
import matplotlib.pyplot as plt
from .convolve import convolve_point_source


# Positions in detector coordinates
def make_point_source_imager(
        xdet : float, ydet : float,
        photon_flux : float,
        psf : np.ndarray,
        size : tuple[int, int],
        image_out : np.ndarray | None = None,
    ) -> np.ndarray:
    """
    Create an image with a point source at the given detector coordinates.


    Args:
        xdet (float): The x (horizontal, second axis) position of the source in detector pixels.
        ydet (float): The y (vertical, first axis) position of the source in detector pixels.
        photon_flux (float): The photon fluxe in photons / sec / m^2.
        psf (np.ndarray): The PSF image for each source.
            The PSF can be of arbitrary size but must be on the correct scale,
            and is assumed to be centered in the image.
        size (tuple[int, int]): The output image shape.
    Returns:
        np.ndarray: The output image in units of photons / sec / m^2.
    """

    # Output image in units phot / s / m^2
    if image_out is None:
        image_out = np.zeros(shape=size)

    # Print
    print(f"Creating point source for {xdet=}, {ydet=}, photon_flux={photon_flux} phot / sec / m^2")

    # Convolve with PSF
    image_out = convolve_point_source(xdet, ydet, photon_flux, psf, size=size)

    # Return
    return image_out

