import numpy as np
from .convolve import convolve_point_source

__all__ = ['make_point_source_ifu_cube']

# Positions in detector coordinates
def make_point_source_ifu_cube(
        xdet : float, ydet : float,
        template : tuple[np.ndarray, np.ndarray],
        psf : np.ndarray,
        size : tuple[int, int],
        cube_out : np.ndarray | None = None,
    ) -> np.ndarray:
    """
    Args:
        xdet (np.ndarray): The x positions in detector coords.
        ydet (np.ndarray): The x positions in detector coords.
        template (list[tuple[np.ndarray, np.ndarray]): A spectral template (wave [microns], flux [photons/sec/m^2]).
        psfs (list[np.ndarray]): The psfs for each source (2D arrays).

    Returns:
        np.ndarray: The image cube in photons / sec / m^2 for each wavelength*spaxel bin.
    """

    # Number of wavelengths
    n_wavelengths = len(template[0])

    print(f"Creating IFU cube with point source {xdet=}, {ydet=}, {template[0][0] - template[0][1]} microns, {n_wavelengths=}")

    # Output cube in units phot / s / nm / m^2
    if cube_out is None:
        cube_out = np.zeros(shape=(size[0], size[1], n_wavelengths), dtype=float)

    # Relative to template[1][0]
    image0 = convolve_point_source(xdet, ydet, template[1][0], psf, size=size)

    # Loop over point sources
    for i in range(n_wavelengths):
        image_i = image0 * template[1][i] / template[1][0]
        cube_out[:, :, i] += image_i
        
    # Return the cube
    return cube_out