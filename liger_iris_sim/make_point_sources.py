import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.constants
c = scipy.constants.c # m/s
h = scipy.constants.h # J s

import utils
from filters import get_iris_filter_data
from psf import get_iris_imager_psf


# Positions in arcsec
def make_iris_point_source_image(
        xs : np.ndarray, ys : np.ndarray, magnitudes : np.ndarray,
        scale : float, filter : str,
        psf_files : list[str], psfs : list[np.ndarray] | None = None,
        resolution : float = None,
        size : tuple[int, int] = (4096, 4096),
        x0 : float = 0.6, y0 : float = 0.6,
        image_out : np.ndarray | None = None,
        psf_dir : str = "/data/group/data/iris/sim/psfs/",
        itime : float = None, atm : str = '50', zenith : str = '45',
        simdir : str = '/data/group/data/iris/sim/',
    ):
    
    # Read in filter info
    filter_filename = f'{simdir}/info/iris_filter_{filter}.dat'
    filter_data = get_iris_filter_data(filter_filename, filter)

    # Number of sources
    n_sources = len(xs)
    assert n_sources == len(ys) == len(magnitudes)

    # Output image in units phot / s / m^2
    if image_out is None:
        image_out = np.zeros(shape=size)

    # Loop over point sources
    for i in range(n_sources):

        # Location of this source relative to on-axis
        x, y = xs[i], ys[i]

        # Magnitude of this source at this wavelength
        mag = magnitudes[i]

        # Convert spatial position to detector coordinates (top right)
        xd, yd = utils.spatialonaxis_to_detector(x, y, scale, x0, y0)

        # Get the PSF for this location, wavelength, etc.
        psf, psf_info = get_iris_imager_psf(
           filter_data["wavecenter"], x, y,
           zenith=zenith, itime=itime, atm=atm, psf_dir=psf_dir
        )
        psf_shape = psf.shape

        # Convert magnitude to flux (integrated over filter bandpass)
        flux = filter_data["zp"] * 10**(-mag / 2.5) # phot / s / m^2
        energy_per_photon = h * c / filter_data["wavecenter"] # J / photon
        flux_photons = flux / energy_per_photon # photons / s / m^2

        # Convolve with PSF
        image_i = utils.convolve_point_source(xd, yd, flux_photons, psf, shape_out = size)

        # Inject into image
        image_out += image_i
        
    return image_out


def convolve_point_source(x, y, flux, psf : np.ndarray, shape_out : tuple):
    image_out = np.zeros(shape_out)
    psf_height, psf_width = psf.shape
    psf_center_x, psf_center_y = int(np.floor(psf_width / 2)), int(np.floor(psf_height / 2))
    xint = int(np.floor(x))
    yint = int(np.floor(y))
    dx = x - xint
    dy = y - yint
    xi = xint - psf_center_x
    xf = xint + psf_center_x
    yi = yint - psf_center_y
    yf = yint + psf_center_y
    yi = np.maximum(0, yi)
    yf = np.minimum(yf, shape_out[1] - 1)
    xi = np.maximum(0, xi)
    xf = np.minimum(xf, shape_out[2] - 1)
    psf_shifted = scipy.ndimage.shift(psf, (dx, dy), mode='constant', cval=0.0)
    psf_shifted = np.clip(psf_shifted, 0, None)
    psf_shifted /= np.sum(psf_shifted)
    image_out[yi:yf+1, xi:xf+1] = flux * psf_shifted
    return image_out