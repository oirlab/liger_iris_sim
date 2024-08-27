import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.constants
c = scipy.constants.c # m/s
h = scipy.constants.h # J s

from . import utils
from .filters import get_iris_filter_data
from .psf import get_iris_imager_psf


# Positions in arcsec
def make_point_source_iris_imager(
        xd : np.ndarray, yd : np.ndarray, magnitudes : np.ndarray,
        scale : float, filt : str,
        psf_files : list[str] | None = None, psfs : list[np.ndarray] | None = None,
        size : tuple[int, int] = (4096, 4096),
        image_out : np.ndarray | None = None,
        itime : float = 300, atm : str = '50', zenith : str = '45',
        simdir : str = '/data/group/data/iris/sim/',
    ):
    
    # Read in filter info
    filter_filename = f'{simdir}info/filter_info.dat'
    filter_data = get_iris_filter_data(filter_filename, filt)

    # Number of sources
    n_sources = len(xd)
    assert n_sources == len(yd) == len(magnitudes)

    # Output image in units phot / s / m^2
    if image_out is None:
        image_out = np.zeros(shape=size)

    if psfs is None:
        psfs_out = []
    else:
        psfs_out = psfs.copy()

    # Loop over point sources
    for i in range(n_sources):

        # Location of this source relative to on-axis
        x, y = xd[i], yd[i]

        # Magnitude of this source at this wavelength
        mag = magnitudes[i]

        # Get the PSF for this location, wavelength, etc.
        if psfs is not None:
            psf = psfs[i]
            psf_info = None
        else:

            # Get PSF
            psf, psf_info = get_iris_imager_psf(
                filter_data["wavecenter"], x, y,
                zenith=zenith, itime=itime, atm=atm, simdir=simdir
            )

            print(f"Source Location: x={x}, y={y} pix. Using PSF:\n{psf_info}")

            # Bin PSF
            bin_factor = scale / psf_info['psf_sampling']
            if bin_factor > 1:
                shape_out = (int(np.round(psf.shape[0] / bin_factor)), int(np.round(psf.shape[1] / bin_factor)))
                psf = utils.frebin(psf, shape=shape_out)
                #psf_info['psf_sampling_binned'] = 
            elif bin_factor < 1:
                pass

        # Add PSF to list
        psfs_out.append((psf, psf_info))

        # Shape
        psf_shape = psf.shape

        # Convert magnitude to flux (integrated over filter bandpass)
        flux = filter_data["zp"] * 10**(-mag / 2.5) # phot / s / m^2
        energy_per_photon = h * c / (1E-9 * filter_data["wavecenter"]) # J / photon
        flux_photons = flux / energy_per_photon # photons / s / m^2

        # Convolve with PSF
        breakpoint()
        image_i = convolve_point_source(x, y, flux_photons, psf, size = size)

        # Inject into image
        image_out += image_i
        
    return image_out, psfs_out


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