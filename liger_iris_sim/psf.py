from astropy.io import fits
import numpy as np
import re
import os
from utils import frebin


def get_iris_imager_psf(wavelength, x, y, zenith='45', itime=300, atm='50', psf_dir='/data/group/data/iris/sim/psfs/'):
    itimes = np.array([1.4, 300])
    k = np.argmin(itimes - itime)
    if itime == int(itime):
        itime = int(itime)
    xs = np.array([0.6, 4.7, 8.8, 12.9, 17])
    ys = np.array([0.6, 4.7, 8.8, 12.9, 17])
    x = xs[np.argmin(np.abs(xs - x))]
    y = xs[np.argmin(np.abs(xs - x))]
    if x == int(x):
        x = int(x)
    if y == int(y):
        y = int(y)
    filename = psf_dir + f"za{zenith}_{int(atm)}p_im_{itime}s{os.sep}evlpsfcl_1_x{x}_y{y}_2mas.fits"
    psf, info = read_iris_psf(filename, wavelength=wavelength)
    return psf, info


def bin_psf(psf : np.ndarray, scale_in : tuple, scale_out : tuple):
    shape_in = psf.shape
    shape_out = (int(shape_in[0] * scale_in / scale_out), int(shape_in[1] * scale_in / scale_out))
    return frebin(psf, shape=shape_out, total=True)


def parse_iris_psf_loc(filename : str):
    x, y = filename.split('/')[-1].split('_')[2:4]
    x, y = x[1:], y[1:]
    x, y = float(x), float(y)
    return x, y


def read_iris_psf(filename : str, hdunum=None, wavelength=None):
    with fits.open(filename) as hdulist:
        if hdunum is None:
            waves = np.full(len(hdulist), np.nan)
            for i in range(len(hdulist)):
                header = hdulist[i].header
                info = parse_iris_psf_info(header)
                waves[i] = info['wavelength']
        k = np.argmin(np.abs(waves - wavelength))
        psf = hdulist[k].data
        info = {key.lower() : val for key, val in info.items()}
        info['filename'] = filename
    return psf, info


def parse_iris_psf_info(header):

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
    info['wavelength'] = 1E9 * float(match[1]) # convert meters to nm

    # OPD sampling
    match = re.search(r'OPD Sampling:\s*([\d.]+)m', comments[3])
    info['opd_sampling'] = 1E9 * float(match[1]) # convert meters to nm

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