import numpy as np
from .convolve import convolve_point_source

import logging
logger = logging.getLogger(__name__)

__all__ = ['make_point_source_ifs_cube']

# Positions in detector coordinates
def make_point_source_ifs_cube(
    xdet : float | np.ndarray, ydet : float | np.ndarray,
    template : tuple[np.ndarray, np.ndarray],
    psf : np.ndarray,
    size : tuple[int, int] | None = None,
    cube_out : np.ndarray | None = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    xdet : float | np.ndarray
        The x-coordinate of the point source in detector coordinates.
    ydet : float | np.ndarray
        The y-coordinate of the point source in detector coordinates.
    template : tuple[np.ndarray, np.ndarray]
        A tuple of (wavelengths, fluxes) for each point source.
        The wavelengths are in microns and
        The fluxes are in photons / sec / m^2.
    psf : np.ndarray
        The PSF image to convolve with.
    size : tuple[int, int]
        The size of the output cube in (y, x).
    cube_out : np.ndarray | None, optional
        An optional pre-allocated output cube to write into.
        If None, a new cube will be created.

    Returns
    -------
    cube_out : np.ndarray
        The output cube in photons / sec / m^2 for each voxel.
    """

    # Number of wavelengths
    template_wave, template_flux = template
    nw = len(template_wave)

    # Output cube in units phot / s / m^2
    if cube_out is None:
        if size is None:
            raise ValueError("Either size or cube_out must be provided")
        cube_out = np.zeros((nw, *size), dtype=np.float32)
    else:
        size = cube_out.shape[1:]

    logger.info(f"Creating IFS cube with point source {xdet=}, {ydet=}")
    
    # Relative to template_flux[0]
    convolve_point_source(
        xdet, ydet, template_flux[0],
        psf, image_out=cube_out[0]
    )

    # Loop over point sources
    for i in range(1, nw):
        image_i = cube_out[0] * template_flux[i] / template_flux[0]
        cube_out[i, :, :] += image_i
        
    # Return the cube
    return cube_out