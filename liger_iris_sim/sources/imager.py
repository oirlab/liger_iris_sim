import numpy as np
from .convolve import convolve_point_source
from numbers import Number

import logging
logger = logging.getLogger(__name__)

__all__ = ['make_point_source_image']

# Positions in detector coordinates
def make_point_source_image(
    xdet : float | np.ndarray, ydet : float | np.ndarray,
    flux : float | np.ndarray,
    psf : np.ndarray,
    size : tuple[int, int] | None = None,
    image_out : np.ndarray | None = None,
    peak_flux : bool = False,
) -> np.ndarray:
    """
    Create an image with a point source at the given detector coordinates.

    Parameters
    ----------
    xdet : float | np.ndarray
        The x (horizontal, second axis) position of the source in detector pixels.
    ydet : float | np.ndarray
        The y (vertical, first axis) position of the source in detector pixels.
    flux : float | np.ndarray
        The source flux in any units.
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
    peak_flux : bool
        If True, the flux is interpreted as the peak flux of the point source.
        The value of the maximum pixel will only match the input flux value if the PSF phase is zero in x and y (i.e., the source is centered on a pixel; xdet and ydet are integers).
        If False, the flux is interpreted as the total flux of the point source.
        Default is False.
    
    Returns
    -------
    np.ndarray
        The output image in units of photons / sec / m^2 for each pixel.
    """

    if image_out is None:
        if size is None:
            raise ValueError("Either size or image_out must be provided")
        image_out = np.zeros(size, dtype=np.float32)
    else:
        size = image_out.shape

    psf /= np.sum(psf)
    psf_max = np.max(psf)

    if peak_flux:
        flux_tot = flux / psf_max
    else:
        flux_tot = flux

    def to_arry(x):
        if isinstance(x, Number):
            return np.array([x])
        else:
            return np.asarray(x)
        
    xdet = to_arry(xdet)
    ydet = to_arry(ydet)
    flux_tot = to_arry(flux_tot)

    n_sources = len(xdet)
    for i in range(n_sources):
        convolve_point_source(
            xdet[i],
            ydet[i],
            flux_tot[i],
            psf,
            image_out=image_out,
        )

    return image_out

