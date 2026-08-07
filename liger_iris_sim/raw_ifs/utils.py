import numpy as np

# TODO: Select pixel size based on instrument name if different
from ..utils import LIGER_PROPS
PIXEL_SIZE_UM = LIGER_PROPS['ifs_detector_pixel_size_um']

def check_window_size(
    epsf : np.ndarray,
    cy : int, cx : int,
    window_size : int,
    n_phase : int = 8
) -> float:
    """
    How much light a (2*window_size + 1)^2 pixel stamp actually captures.

    window_size is no longer derived automatically, so this is the check that
    it is big enough.  Returns the WORST fraction over a grid of sub-pixel
    phases; anything below ~0.999 means flux is being silently thrown away.
    """
    n_ey, n_ex = epsf.shape
    worst = 1.0
    for py_ph in np.linspace(0.0, 1.0, n_phase, endpoint=False):
        for px_ph in np.linspace(0.0, 1.0, n_phase, endpoint=False):
            tot = 0.0
            for ky in range(-window_size, window_size + 1):
                for kx in range(-window_size, window_size + 1):
                    fy = (ky - py_ph) * PIXEL_SIZE_UM + cy
                    fx = (kx - px_ph) * PIXEL_SIZE_UM + cx
                    iy, ix = int(np.floor(fy)), int(np.floor(fx))
                    if iy < 0 or iy + 1 >= n_ey or ix < 0 or ix + 1 >= n_ex:
                        continue
                    ty, tx = fy - iy, fx - ix
                    tot += ((1-ty)*(1-tx)*epsf[iy, ix] + (1-ty)*tx*epsf[iy, ix+1]
                            + ty*(1-tx)*epsf[iy+1, ix] + ty*tx*epsf[iy+1, ix+1])
            worst = min(worst, tot)
    return worst