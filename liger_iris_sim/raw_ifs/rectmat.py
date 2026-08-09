import os
import numpy as np
from astropy.io import fits

from .trace_geometry import get_trace_geometry
from .micropupil_psf import get_effective_psf
from .render_trace import render_trace_for_lenslet

from ..utils import LIGER_PROPS
DETECTOR_SHAPE = LIGER_PROPS['ifs_detector_size']

from liger_iris_drp_resources import (
    load_filters_summary,
    load_ifs_array_mask,
    load_ifs_trace_geometry,
)

__all__ = ["make_rectmat", "save_rectmat_to_fits"]


def _trace_pixel_span(
    x_pts : np.ndarray, y_pts : np.ndarray, wave_pts : np.ndarray,
    tracepos_deg : int, wavesol_deg : int,
):
    """
    Detector column range and trace geometry for one lenslet, clipped to the detector. Reuses `get_trace_geometry`. so no cube-specific end-padding is applied — there is no input cube here, just the raw trace) so the pixel span always matches the renderer's own definition of the trace.
    """
    x_lo, x_hi, y_of_x, wave_of_x = get_trace_geometry(
        x_pts, y_pts, wave_pts,
        tracepos_deg=tracepos_deg, wavesol_deg=wavesol_deg,
    )
    pix_lo = max(int(np.ceil(x_lo - 0.5)), 0)
    pix_hi = min(int(np.floor(x_hi + 0.5)), DETECTOR_SHAPE[1] - 1)
    return pix_lo, pix_hi, y_of_x, wave_of_x


def _measure_rectmat_dims(
    arr_mask : np.ndarray,
    x_pix : np.ndarray, y_pix : np.ndarray,
    wave_pts : np.ndarray,
    tracepos_deg : int, wavesol_deg : int,
    window_size : int,
) -> tuple:
    """
    Pass 1: measure the (max_trace_len, n_row_window) needed to hold every
    lenslet's trace before allocating the rectification matrix.
    """
    n_lens_y, n_lens_x = arr_mask.shape
    n_lenslets = len(x_pix)
    max_trace_len = 0
    max_row_extent = 0

    for ly in range(n_lens_y):
        for lx in range(n_lens_x):
            idx = int(arr_mask[ly, lx])
            if idx < 0 or idx >= n_lenslets:
                continue
            pix_lo, pix_hi, y_of_x, _ = _trace_pixel_span(
                x_pix[idx].astype(float), y_pix[idx].astype(float), wave_pts,
                tracepos_deg, wavesol_deg,
            )
            if pix_lo > pix_hi:
                continue
            max_trace_len = max(max_trace_len, pix_hi - pix_lo + 1)

            x_cols = np.arange(pix_lo, pix_hi + 1, dtype=float)
            y_centers = y_of_x(x_cols)
            row_lo = int(np.floor(np.min(y_centers))) - window_size
            row_hi = int(np.ceil(np.max(y_centers))) + window_size
            max_row_extent = max(max_row_extent, row_hi - row_lo + 1)

    return max_trace_len, max_row_extent


def _build_rectmat(
    arr_mask : np.ndarray,
    x_pix : np.ndarray, y_pix : np.ndarray,
    wave_pts : np.ndarray,
    epsf : np.ndarray,
    tracepos_deg : int, wavesol_deg : int,
    window_size : int,
    max_trace_len : int, n_row_window : int,
) -> tuple:
    """
    Pass 2: fill the rectification matrix, offsets, and wavelength
    solution for every valid lenslet.

    rectmat[r, tc, ly, lx] is the weight for detector pixel
    (offsets[1, ly, lx] + r, offsets[0, ly, lx] + tc).
    """
    n_lens_y, n_lens_x = arr_mask.shape
    n_lenslets = len(x_pix)

    rectmat = np.zeros((n_row_window, max_trace_len, n_lens_y, n_lens_x), dtype=np.float32)
    offsets = np.full((2, n_lens_y, n_lens_x), -1, dtype=np.int32)
    wavesol = np.zeros((max_trace_len, n_lens_y, n_lens_x), dtype=np.float32)

    for ly in range(n_lens_y):
        for lx in range(n_lens_x):
            idx = int(arr_mask[ly, lx])
            if idx < 0 or idx >= n_lenslets:
                continue

            pix_lo, pix_hi, y_of_x, wave_of_x = _trace_pixel_span(
                x_pix[idx].astype(float), y_pix[idx].astype(float), wave_pts,
                tracepos_deg, wavesol_deg,
            )
            if pix_lo > pix_hi:
                continue

            x_cols = np.arange(pix_lo, pix_hi + 1, dtype=float)
            y_centers = y_of_x(x_cols)

            row_start = max(int(np.floor(np.min(y_centers))) - window_size, 0)
            offsets[0, ly, lx] = pix_lo
            offsets[1, ly, lx] = row_start
            wavesol[:len(x_cols), ly, lx] = wave_of_x(x_cols).astype(np.float32)

            # Illuminate this lenslet alone with a flat (all-ones) spectrum and
            # render it with the same 2D ePSF kernel used by the forward
            # simulator, so the rectmat stays exactly consistent with it.
            avail_rows = min(n_row_window, DETECTOR_SHAPE[0] - row_start)
            render_trace_for_lenslet(
                rectmat[:avail_rows, :, ly, lx],
                px_lo=0,
                y_pix=y_centers - row_start,
                flux=np.ones(len(x_cols), dtype=np.float32),
                epsf=epsf,
                window_size=window_size,
            )

    return rectmat, offsets, wavesol


def save_rectmat_to_fits(
    rectmat : np.ndarray, offsets : np.ndarray, wavesol : np.ndarray,
    output_path : str,
) -> None:
    """
    Save a rectification matrix to a FITS file.

    Parameters
    ----------
    rectmat : ndarray
        Shape (n_row_window, max_trace_len, n_lens_y, n_lens_x).
    offsets : ndarray
        [col_start, row_start] per lenslet, shape (2, n_lens_y, n_lens_x).
    wavesol : ndarray
        Wavelength solution per lenslet, shape
        (max_trace_len, n_lens_y, n_lens_x), microns.
    output_path : str
        Path to write the FITS file. Extensions: "RECTMAT", "OFFSETS",
        "WAVESOL". Parent directory is created if needed.
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(rectmat, name="RECTMAT"),
        fits.ImageHDU(offsets, name="OFFSETS"),
        fits.ImageHDU(wavesol, name="WAVESOL"),
    ])
    hdul.writeto(output_path, overwrite=True)


# Main function user calls to build a rectification matrix for their IFS mode/filter/resolution
def make_rectmat(
    ifs_mode : str,
    filter_name : str,
    resolution : str,
    window_size : int = 2,
    tracepos_deg : int = 1,
    wavesol_deg : int = 1,
    output_path : str = None,
) -> dict:
    """
    Main function user calls to build a rectification matrix: the
    per-lenslet linear map from raw detector pixels to trace-column flux,
    for a given IFS mode/filter/resolution. This is the extraction
    operator consumed by spectral extraction (e.g. `extract_ifs_lenslet`)
    to invert a raw detector frame back into a lenslet cube.

    Uses the same trace geometry (`get_trace_geometry`) and effective PSF
    (`get_effective_psf`) as `simulate_raw_ifs_frame`, so the extraction
    operator built here stays consistent with the forward renderer.

    Parameters
    ----------
    ifs_mode : str
        IFS mode, e.g. "lenslet".
    filter_name : str
        Name of the filter.
    resolution : str
        Spectral resolution.
    window_size : int, optional
        Half-width in detector rows and columns of the ePSF footprint
        captured around each trace point. Default 2.
    tracepos_deg : int, optional
        Degree of polynomial fit to the trace position y(x). Default 1.
    wavesol_deg : int, optional
        Degree of polynomial fit to the wavelength solution lambda(x).
        Default 1.
    output_path : str, optional
        If given, save the rectification matrix (RECTMAT, OFFSETS,
        WAVESOL extensions) here.

    Returns
    -------
    out : dict
        "rectmat" : ndarray, shape (n_row_window, max_trace_len, n_lens_y, n_lens_x).
            rectmat[r, tc, ly, lx] is the weight for detector pixel
            (offsets[1, ly, lx] + r, offsets[0, ly, lx] + tc).
        "offsets" : ndarray, shape (2, n_lens_y, n_lens_x): [col_start, row_start] per lenslet.
            col_start < 0 marks an invalid/unused lenslet.
        "wavesol" : ndarray, shape (max_trace_len, n_lens_y, n_lens_x), microns.
        "filepath" : str or None.
    """
    filter_info = load_filters_summary(filter_name)
    wave_pts = np.linspace(filter_info["wavemin"], filter_info["wavemax"], 5)

    arr_mask = load_ifs_array_mask()
    x_pix, y_pix = load_ifs_trace_geometry(ifs_mode, filter_name, resolution)

    epsf = get_effective_psf(filter_name)

    max_trace_len, n_row_window = _measure_rectmat_dims(
        arr_mask, x_pix, y_pix, wave_pts, tracepos_deg, wavesol_deg, window_size,
    )

    rectmat, offsets, wavesol = _build_rectmat(
        arr_mask, x_pix, y_pix, wave_pts, epsf,
        tracepos_deg, wavesol_deg, window_size,
        max_trace_len, n_row_window,
    )

    out = {
        "rectmat": rectmat,
        "offsets": offsets,
        "wavesol": wavesol,
        "filepath": output_path,
    }

    if output_path is not None:
        save_rectmat_to_fits(rectmat, offsets, wavesol, output_path)

    return out
