import numpy as np

from ..utils import LIGER_PROPS
PIXEL_SIZE_UM = LIGER_PROPS['ifs_detector_pixel_size_um']

from liger_iris_drp_resources import load_micropupil_for_filter

__all__ = ["get_effective_psf"]

def get_effective_psf(filter_name : str) -> np.ndarray:
    """
    Load the micropupil PSF for the given filter and convolve it with the detector pixel response.

    Parameters
    ----------
    filter_name : str
        The name of the filter for which to load the micropupil PSF.

    Returns
    -------
    epsf : ndarray
        The effective PSF lookup array, sampled at 1 micron.
    """
    mpupil_psf = load_micropupil_for_filter(filter_name)
    epsf = build_epsf(mpupil_psf)
    return epsf

def build_epsf(mpupil_psf : np.ndarray):
    """
    Turn a micropupil image into the effective PSF with original sampling.

    Parameters
    ----------
    mpupil_psf : ndarray
        The oversampled (1 micron) micropupil image.

    Returns
    -------
    epsf : ndarray
        The effective PSF lookup array, sampled at 1 micron.
    """

    mpupil_psf = mpupil_psf / mpupil_psf.sum()
    epsf = convolve_pixel_response(
        mpupil_psf,
        input_pixel_size=1,
        output_pixel_size=PIXEL_SIZE_UM
    )
    assert epsf.shape[0] % 2 == 1 and epsf.shape[1] % 2 == 1, f"EPSF shape must be odd, got {epsf.shape}"
    return epsf

def convolve_pixel_response(
    image : np.ndarray,
    input_pixel_size : int,
    output_pixel_size : int,
) -> np.ndarray:
    """
    Convolve the input array with a box filter of the specified size.

    Parameters
    ----------
    image : np.ndarray
        The input array to be convolved.
    input_pixel_size : int
        The size of the input pixels (in microns).
    output_pixel_size : int, optional
        The size of the output pixels (in microns). Default is PIXEL_SIZE_UM.

    Returns
    -------
    image_out : np.ndarray
        The convolved array, with the same shape as the input array.
    """

    w = int(output_pixel_size // input_pixel_size)
    assert w >= 1, f"Window size must be >= 1, got w = output_pixel_size ({output_pixel_size}) // input_pixel_size ({input_pixel_size}) = {w}"
    pad = w // 2
    image_padded = np.pad(image, (pad, pad), mode='edge')
    c = image_padded.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))   # zeros — defines the exclusive convention
    ny, nx = image.shape
    lo = slice(0, ny), slice(0, nx)
    hi = slice(w, w + ny), slice(w, w + nx)
    image_out = c[hi[0], hi[1]] - c[lo[0], hi[1]] - c[hi[0], lo[1]] + c[lo[0], lo[1]]
    return image_out