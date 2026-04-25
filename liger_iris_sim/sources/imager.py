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
    psf : np.ndarray | list[np.ndarray],
    psf_indices : np.ndarray | None = None,
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
    psf : np.ndarray | list[np.ndarray]
        The PSF image for each source.
        The PSF can be of arbitrary size but must be on the correct scale,
        and is assumed to be centered in the image.
        Optionally, a list of PSFs can be provided, in which case each source will be convolved with the corresponding PSF in the list.
    psf_indices : np.ndarray | None, optional
        If psf is a list of PSFs, this array gives the index of each point source in the psf list.
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

    def to_array(x):
        if isinstance(x, Number):
            return np.array([x])
        else:
            return np.asarray(x)
        
    if isinstance(psf, list):
        if psf_indices is None:
            raise ValueError("psf_indices must be provided if psf is a list")
        
    xdet = to_array(xdet)
    ydet = to_array(ydet)
    flux = to_array(flux)

    n_sources = len(xdet)
    for i in range(n_sources):
        if isinstance(psf, list):
            _psf = psf[psf_indices[i]]
        else:
            _psf = psf

        if peak_flux:
            flux_tot = flux[i] / np.max(_psf)
        else:
            flux_tot = flux[i]

        convolve_point_source(
            xdet[i],
            ydet[i],
            flux_tot,
            _psf,
            image_out=image_out,
        )

    return image_out

