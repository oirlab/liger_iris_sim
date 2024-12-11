from .filters import *
from .eff import *
from .psf import *
from .saturation import *
from .config import *


def imager_to_spatial(
        xd : float | np.ndarray, yd : float | np.ndarray,
        scale : float, xs0 : float = 0.6, ys0 : float = 0.6,
    ):
    xs = scale * xd + xs0
    ys = scale * xd + ys0
    return xs, ys


def spatial_to_imager(
        xs : float | np.ndarray, ys : float | np.ndarray, scale : float,
        xs0 : float = 0.6, ys0 : float = 0.6,
    ):
    xd = (xs - xs0) / scale
    yd = (ys - ys0) / scale
    return xd, yd


def ifu_to_spatial(
        xd : float | np.ndarray, yd : float | np.ndarray,
        scale : float, size : tuple [int, int]
    ):
    xs0 = -int(size[1] / 2) * scale
    ys0 = -int(size[0] / 2) * scale
    xs = scale * xd + xs0
    ys = scale * xd + ys0
    return xs, ys

def generate_filename(
        obsid : str ,
        detector : str, obstype : str, level : int | str = 0,
        exp : int | str = '0001', subarray : int | str | None = None,
        dir : str = '',
    ):
    # Format: OBSID-Liger-DET[X]-OBSTYPLevel-EXP-SUB.fits
    if isinstance(exp, int):
        exp = str(exp).zfill(4)
    if type(subarray) in (int, str):
        subarray = '-' + str(subarray).zfill(2)
    else:
        subarray = ''
    return f"{dir}{obsid}_Liger_{detector.upper()}_{obstype}_LVL{int(level)}_{exp}{subarray}.fits"


DEFAULT_HEADER = fits.Header()
DEFAULT_HEADER['OBSTYPE'] = (None, 'Observation Type')
DEFAULT_HEADER['ITIME'] = (None, 'Exposure Time (sec)')
DEFAULT_HEADER['JDUTCST'] = (None, 'UTC start time of exposure')
DEFAULT_HEADER['JDUTCEND'] = (None, 'UTC end time of exposure')
DEFAULT_HEADER['DATE-OBS'] = (None, 'ISO 8601 start time of exposure')
DEFAULT_HEADER['NFRAMES'] = (None, 'Number of frames')
DEFAULT_HEADER['FILTER'] = (None, 'Filter')
DEFAULT_HEADER['SCALEIMG'] = (0.004, 'Average plate scale for imager (arcsec)')
DEFAULT_HEADER['SCALEIFU'] = (None, 'Average plate scale for IFU (arcsec)')
DEFAULT_HEADER['SPECRES'] = (None, 'Average spectral resolution')
DEFAULT_HEADER['TARGNAME'] = (None, 'Target name')
DEFAULT_HEADER['TARGRA'] = (None, 'Target RA (deg)')
DEFAULT_HEADER['TARGDEC'] = (None, 'Target Decl (deg)')
DEFAULT_HEADER['READNOIS'] = (imager_read_noise, 'Detector read noise (e- RMS)')
DEFAULT_HEADER['GAIN'] = (imager_gain, 'Detector gain (e-/ADU)')
DEFAULT_HEADER['TELESCOP'] = ('KeckI', 'Telescope name')
DEFAULT_HEADER['INSTRUME'] = ('Liger', 'Instrument name')
DEFAULT_HEADER['DETECTOR'] = ('IMG', 'Detector name')
DEFAULT_HEADER['DATAMODL'] = (None, 'stpipe data model')
DEFAULT_HEADER['DRPVER'] = ('0.0.1', 'Version of the DRP')


def generate_primary_header(metadata : dict):

    # Create header
    header = DEFAULT_HEADER.copy()

    # Update with new metadata
    header.update(metadata)

    # Check on special keys
    if DEFAULT_HEADER['DATE-OBS'] is None:
        DEFAULT_HEADER['DATE-OBS'] = DEFAULT_HEADER['JDUTCST']
    
    # Return
    return header