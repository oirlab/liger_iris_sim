import numpy as np
import os
import re
from astropy.io import fits

def get_psf(
        wavelength : float, xs : float, ys : float,
        psfdir : str = '/data/group/data/kapa/liger/sim/PSFs_LTAO_11_09_19/ltao_7x7_YJHK/'
    ):
    xs_ao = np.array([-15, -10, -5, 0, 5, 10, 15])
    ys_ao = np.array([-15, -10, -5, 0, 5, 10, 15])
    xs = xs_ao[np.argmin(np.abs(xs_ao - xs))]
    ys = xs_ao[np.argmin(np.abs(ys_ao - ys))]
    liger_psf_filters = ['Y', 'J', 'H', 'K']
    liger_psf_filter_waves = np.array([1020, 1248, 1650, 2124])
    filt = liger_psf_filters[np.argmin(np.abs(liger_psf_filter_waves - wavelength))]
    if filt == 'Y':
        filename = psfdir + f"ltao_7_7_hy/evlpsfcl_1_x{xs}_y{ys}.fits"
        hdunum = 1
    elif filt == 'J':
        filename = psfdir + f"ltao_7_7_kj/evlpsfcl_1_x{xs}_y{ys}.fits"
        hdunum = 1
    elif filt == 'H':
        filename = psfdir + f"ltao_7_7_hy/evlpsfcl_1_x{xs}_y{ys}.fits"
        hdunum = 0
    elif filt == 'K':
        filename = psfdir + f"ltao_7_7_kj/evlpsfcl_1_x{xs}_y{ys}.fits"
        hdunum = 0
    psf, info = read_psf(filename, hdunum)
    return psf, info


def parse_psf_loc(filename : str):
    match = re.search(r"_x([-+]?\d+)_y([-+]?\d+)", os.path.basename(filename))
    x, y = match.group(1), match.group(2)
    x, y = int(x), int(y)
    return x, y


def read_psf(
        filename : str, hdunum : int | None = None,
    ):
    with fits.open(filename) as hdulist:
        psf = hdulist[hdunum].data
        info = parse_psf_header(hdulist[hdunum].header)
        info['filename'] = filename
        info['hdunum'] = hdunum
        info['position'] = parse_psf_loc(filename)
    return psf, info


def parse_psf_header(header : fits.Header):

    # Result
    info = {}

    # r0
    info['r0'] = header['R0'] * 1E6 # meters -> microns

    # l0 (outer scale)
    info['l0'] = header['L0'] * 1E6 # meters -> microns

    # wavelength
    info['wavelength'] = header['WVL'] * 1E6 # meters -> microns

    # OPD sampling
    info['opd_sampling'] = header['DT'] * 1E6 # meters -> microns

    # fft grid
    info['fft_grid'] = (int(header['NFFT'].real), int(header['NFFT'].imag))

    # psf sampling (arcsec)
    info['psf_sampling'] = header['DP']

    # Sum
    info['sum'] = header['SUM']

    # itime
    info['itime'] = header['DT']

    # Theta
    info['theta'] = (int(header['THETA'].real), int(header['THETA'].imag))
    
    return info