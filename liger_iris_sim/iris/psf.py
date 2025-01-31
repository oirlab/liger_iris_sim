import numpy as np
import os
import re
from astropy.io import fits
from ..utils import _resolve_mode

__all__ = ['get_psf', 'get_psf_filename', 'read_psf']


def get_psf(
        mode : str,
        wavelength : float,
        xs : float = 0, ys : float = 0,
        itime : float = 300,
        zenith : str = '45',
        atm : str = '50',
        psfdir : str = '/data/group/data/iris/sim/psfs/',
    ):
    """
    Get the filename PSF for the imager or IFU.
    """
    filename = get_psf_filename(mode, xs=xs, ys=ys, itime=itime, zenith=zenith, atm=atm, psfdir=psfdir)
    psf, info = read_psf(filename, wavelength=wavelength)
    info['mode'] = mode
    return psf, info
    

def get_psf_filename(
        mode : str,
        xs : float = 0, ys : float = 0,
        itime : float = 300,
        zenith : str = '45', atm : str = '50',
        psfdir : str = '/data/group/data/iris/sim/psfs/'
    ) -> str:
    """
    Gets the filename of the PSF for the imager or IFU.

    Args:
        mode (str): The mode ('img', 'slicer', 'lenslet').
        xs (float): The x offset in arcsec from on-axis.
        ys (float): The y offset in arcsec from on-axis.
        itime (float): The integration time in seconds.
        zenith (str): The zenith angle in degrees.
        atm (str): The atmosphere in percentile.
        psfdir (str): The directory where the PSFs are stored.
    """
    mode = _resolve_mode(mode)
    itimes = np.array([1.4, 300])
    k = np.argmin(np.abs(itimes - itime))
    itime = itimes[k]
    if itime == int(itime):
        itime = int(itime)
    zenith = int(zenith)
    if mode == 'img':
        xs_ao = np.array([0.6, 4.7, 8.8, 12.9, 17])
        ys_ao = np.array([0.6, 4.7, 8.8, 12.9, 17])
        xs = xs_ao[np.argmin(np.abs(xs_ao - xs))]
        ys = xs_ao[np.argmin(np.abs(ys_ao - ys))]
        if xs == int(xs):
            xs = int(xs)
        if ys == int(ys):
            ys = int(ys)
        filename = psfdir + f"za{zenith}_{int(atm)}p_im_{itime}s{os.sep}evlpsfcl_1_x{xs}_y{ys}_2mas.fits"
    elif mode in ('slicer', 'lenslet'):
        filename = psfdir + f"za{zenith}_{int(atm)}p_ifu_{itime}s{os.sep}evlpsfcl_1_x0_y0_2mas.fits"
    else:
        raise ValueError(f"Unknown mode '{mode}'.")
    return filename


def read_psf(
        filename : str,
        hdunum : int | None = None,
        wavelength : float | None = None
    ) -> tuple[np.ndarray, dict]:
    """
    Read a PSF file and return the PSF and header info.
    """
    if hdunum is None:
        hdunum = get_psf_hdu_for_wavelength(filename, wavelength)
    with fits.open(filename) as hdulist:
        psf = hdulist[hdunum].data
        info = parse_psf_header(hdulist[hdunum].header)
        info['filename'] = filename
        info['hdunum'] = hdunum
        info['atm'] = filename.split('/')[-2][2:4]
        info['weather'] = filename.split('/')[-2][5:7]
    return psf, info


def get_psf_hdu_for_wavelength(filename : str, wavelength : float) -> int:
    """
    Get the HDU number for a given wavelength in microns.
    """
    with fits.open(filename) as hdulist:
        waves = np.full(len(hdulist), np.nan)
        for i in range(len(hdulist)):
            header = hdulist[i].header
            info = parse_psf_header(header)
            waves[i] = info['wavelength']
        hdunum = np.argmin(np.abs(waves - wavelength))
    return hdunum


def parse_psf_header(header : fits.Header) -> dict:
    """
    Parse the header of a PSF file.
    """

    # Split into a list
    comments = ""
    for comment in header['COMMENT']:
        comments += comment.strip()
    comments = comments.split(';')
    # Result
    info = {}

    # Position
    match = re.search(r'Science PSF at \(([\d.]+),\s*([\d.]+)\)\s*arcsec', comments[0])
    info['x'] = match[1]
    info['y'] = match[2]

    # r0
    match = re.search(r'r0=([\d.]+)', comments[1])
    info['r0'] = float(match[1])

    # l0
    match = re.search(r'l0=([\d.]+)', comments[1])
    info['l0'] = float(match[1])

    # wavelength
    match = re.search(r'Wavelength:\s*([\d.eE+-]+)m', comments[2])
    info['wavelength'] = 1E6 * float(match[1]) # convert meters to microns

    # OPD sampling
    match = re.search(r'OPD Sampling:\s*([\d.]+)m', comments[3])
    info['opd_sampling'] = 1E6 * float(match[1]) # convert meters to microns

    # fft grid
    match = re.search(r'FFT Grid:\s*(\d+)x(\d+)', comments[4])
    info['fft_grid'] = (int(match[1]), int(match[2]))

    # psf sampling
    match = re.search(r'PSF Sampling:\s*([\d.eE+-]+)\s*arcsec', comments[5])
    info['psf_sampling'] = float(match[1]) # arcsec

    # Sum
    match = re.search(r'PSF Sum to:\s*([\d.eE+-]+)', comments[6])
    info['sum'] = float(match[1])

    # itime
    match = re.search(r'Exposure:\s*(\d+)s', comments[7])
    info['itime'] = float(match[1])
    
    return info
