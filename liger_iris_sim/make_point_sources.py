import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.constants

c = scipy.constants.c # m/s
h = scipy.constants.h # J s


# Positions in detector coordinates
def make_point_source_image(
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
        print(f"Processing source {i + 1} : X = {xdet[i]}, Y = {ydet[i]}, Flux = {fluxes[i]}")

        # Convolve with PSF
        image_i = convolve_point_source(xdet[i], ydet[i], fluxes[i], psfs[i], size=size)

        # Inject into image
        image_out += image_i
        
    return image_out



# Positions in detector coordinates
def make_point_source_ifu_lenselt_cube(
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


# # Positions in detector coordinates
# def make_point_source_ifu_slicer_cube(
#         xdet : np.ndarray, ydet : np.ndarray, templates : list[np.ndarray],
#         size : tuple[int, int],
#         psfs : list[np.ndarray],
#         cube_out : np.ndarray | None = None,
#     ):
#     pass


def convolve_point_source(x : float, y : float, flux : np.ndarray, psf : np.ndarray, size : tuple):
    psf_height, psf_width = psf.shape
    psf_center_x = np.ceil(psf_width / 2) if psf_width % 2 == 1 else psf_width / 2 - 0.5
    psf_center_y = np.ceil(psf_height / 2) if psf_width % 2 == 1 else psf_height / 2 - 0.5
    xpsf, ypsf = np.arange(psf_width) - psf_center_x, np.arange(psf_height) - psf_center_y
    itp = interp.RectBivariateSpline(ypsf + y, xpsf + x, psf, kx=1, ky=1, s=0)
    xarr, yarr = np.arange(size[1]), np.arange(size[0])
    psf_shifted = itp(yarr, xarr)
    psf_shifted = np.clip(psf_shifted, 0, None)
    psf_shifted /= np.sum(psf_shifted)
    image_out = flux * psf_shifted
    return image_out


