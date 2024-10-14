from .filters import *
from .eff import *
from .psf import *
from .saturation import *
from .config import *


def generate_filename(
        obsid : str ,
        detector : str, typ : str, level : int | str = 0,
        exp : int | str = '0001', subarray : int | str | None = None,
        path : str = ''
    ):
    # Format: OBSID-Liger-DET[X]-TYPLevel-EXP-SUB.fits
    if isinstance(exp, int):
        exp = str(exp).zfill(4)
    if type(subarray) in (int, str):
        subarray = '-' + str(subarray).zfill(2)
    else:
        subarray = ''
    return f"{path}{obsid}-Liger-{detector.upper()}-{typ.upper()}{int(level)}-{exp}{subarray}.fits"