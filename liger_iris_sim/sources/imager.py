import numpy as np
import matplotlib.pyplot as plt
from .convolve import convolve_point_source


# Positions in detector coordinates
def make_point_sources_imager(
        xdet : np.ndarray, ydet : np.ndarray, fluxes : np.ndarray,
        size : tuple[int, int],
        psfs : list[np.ndarray],
        image_out : np.ndarray | None = None,
    ):

    # Number of sources
    assert len(xdet) == len(ydet) == len(fluxes) == len(psfs)
    n_sources = len(xdet)

    # Output image in units phot / s / m^2
    if image_out is None:
        image_out = np.zeros(shape=size)

    # Loop over point sources
    for i in range(n_sources):

        # Print
        print(f"Convolving source {i + 1}:")
        print(f"  Xdet = {xdet[i]}, Ydet = {ydet[i]}, Flux = {fluxes[i]}")

        # Convolve with PSF
        image_i = convolve_point_source(xdet[i], ydet[i], fluxes[i], psfs[i], size=size)

        # Inject into image
        image_out += image_i

    # Return
    return image_out

