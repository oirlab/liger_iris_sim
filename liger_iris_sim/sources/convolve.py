import numpy as np
import scipy.interpolate


def convolve_point_source(x : float, y : float, flux : np.ndarray, psf : np.ndarray, size : tuple):
    psf_height, psf_width = psf.shape
    psf_center_x = np.ceil(psf_width / 2) if psf_width % 2 == 1 else psf_width / 2 - 0.5
    psf_center_y = np.ceil(psf_height / 2) if psf_width % 2 == 1 else psf_height / 2 - 0.5
    xpsf, ypsf = np.arange(psf_width) - psf_center_x, np.arange(psf_height) - psf_center_y
    itp = scipy.interpolate.RegularGridInterpolator((ypsf + y, xpsf + x), psf, method='linear', bounds_error=False, fill_value=0)
    xarr, yarr = np.arange(size[1]), np.arange(size[0])
    XARR, YARR = np.meshgrid(xarr, yarr, indexing='ij')
    psf_shifted = itp((XARR, YARR))
    psf_shifted = np.clip(psf_shifted, 0, None)
    psf_shifted /= np.sum(psf_shifted)
    image_out = flux * psf_shifted
    return image_out