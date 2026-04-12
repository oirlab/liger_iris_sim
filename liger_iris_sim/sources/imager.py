import numpy as np
from .convolve import convolve_point_source

__all__ = ['make_point_source_image']

# Positions in detector coordinates
def make_point_source_image(
    xdet : float, ydet : float,
    photon_flux : float,
    psf : np.ndarray,
    size : tuple[int, int] | None = None,
    image_out : np.ndarray | None = None,
) -> np.ndarray:
    """
    Create an image with a point source at the given detector coordinates.

    Parameters
    ----------
    xdet : float
        The x (horizontal, second axis) position of the source in detector pixels.
    ydet : float
        The y (vertical, first axis) position of the source in detector pixels.
    photon_flux : float
        The photon flux in photons / sec / m^2.
    psf : np.ndarray
        The PSF image for each source.
        The PSF can be of arbitrary size but must be on the correct scale,
        and is assumed to be centered in the image.
    size : tuple[int, int] | None = None
        The output image shape. Either size or image_out must be provided.
    image_out : np.ndarray | None
        An optional output array to write the image into.
        If None, a new array will be created according to ``size``.
        Default is None.
    
    Returns
    -------
    np.ndarray
        The output image in units of photons / sec / m^2 for each pixel.
    """

    # Output image in units phot / s / m^2
    if image_out is None:
        image_out = np.zeros(shape=size)
    else:
        size = image_out.shape

    # Print
    print(f"Creating point source for {xdet=}, {ydet=}, photon_flux={photon_flux} phot / sec / m^2")

    # Convolve with PSF
    convolve_point_source(xdet, ydet, photon_flux, psf, image_out=image_out)

    # Return
    return image_out

