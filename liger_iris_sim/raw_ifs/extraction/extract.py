import os
import numpy as np
from astropy.io import fits

from .column_extraction_ols import extract_columns
from .resample import resample_to_common_grid

__all__ = ["extract_ifs_lenslet"]


def _load_rectmat(rectmat_path : str) -> tuple:
    """
    Load a rectification matrix, e.g. as produced by `make_rectmats.py`.

    Returns
    -------
    rectmat : ndarray
        Shape (n_row_window, max_trace_len, n_lens_y, n_lens_x).
    offsets : ndarray
        [col_start, row_start] per lenslet, shape (2, n_lens_y, n_lens_x).
    wavesol : ndarray
        Wavelength solution per lenslet, shape
        (max_trace_len, n_lens_y, n_lens_x), microns.
    """
    with fits.open(rectmat_path) as hdul:
        rectmat = hdul["RECTMAT"].data
        offsets = hdul["OFFSETS"].data
        wavesol = hdul["WAVESOL"].data
    return rectmat, offsets, wavesol


def save_extracted_cube_to_fits(
    wave_out : np.ndarray,
    cube_flux : np.ndarray,
    cube_err : np.ndarray,
    output_path : str,
) -> None:
    """
    Save an extracted IFS cube to a FITS file.

    Parameters
    ----------
    wave_out : ndarray
        Wavelength solution, microns. Shape (n_wave,) if resampled onto a
        common grid, or (max_trace_len, n_lens_y, n_lens_x) if native
        per-lenslet (see `extract_ifs_lenslet`).
    cube_flux, cube_err : ndarray
        Flux and its 1-sigma error, same leading/trailing shape convention
        as `wave_out`: (n_wave, n_lens_y, n_lens_x) or
        (max_trace_len, n_lens_y, n_lens_x).
    output_path : str
        Path to write the FITS file. Extensions: "FLUX", "ERR", "WAVE".
        Parent directory is created if needed.
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    wave_hdr = fits.Header()

    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(cube_flux.astype(np.float32), name="FLUX"),
        fits.ImageHDU(cube_err.astype(np.float32), name="ERR"),
        fits.ImageHDU(wave_out.astype(np.float32), name="WAVE", header=wave_hdr),
    ])
    hdul.writeto(output_path, overwrite=True)


# Main function user calls to extract a lenslet IFS cube from a raw detector frame
def extract_ifs_lenslet(
    data : np.ndarray,
    rectmat_path : str,
    output_wavesol : np.ndarray | None = None,
    interp_order : int = 1,
    error : np.ndarray = None,
    output_path : str = None,
) -> dict:
    """
    Main function user calls to extract a lenslet IFS cube from a raw
    detector frame, given a precomputed rectification matrix.

    Extraction is a column-by-column linear (weighted) least-squares solve
    against the rectification matrix: at each detector column, every
    lenslet trace passing through it is identified, and traces whose PSF
    footprints overlap on the detector are solved for jointly (deconvolving
    each trace's flux from its neighbours' contribution to shared pixels),
    while non-overlapping traces solve independently. Per-lenslet spectra
    are then resampled onto the caller-provided common wavelength grid
    `output_wavesol`. See `column_extraction_ols.extract_columns` and
    `resample._resample_to_common_grid` for details.

    Parameters
    ----------
    data : ndarray
        Raw detector image, shape (n_rows, n_cols).
    rectmat_path : str
        Path to the rectification matrix FITS file, with RECTMAT,
        OFFSETS, and WAVESOL extensions.
    error : ndarray, optional
        Per-pixel 1-sigma error for `data`, same shape. If given, a
        weighted least-squares solve is used and the returned "err" is
        populated. If None (default), an unweighted least-squares solve
        is used and "err" is NaN — see `column_extraction_ols.extract_columns`.
    output_wavesol : ndarray, optional
        Common wavelength grid to resample every lenslet onto, shape
        (n_wave,), microns. Must be monotonically increasing. If None
        (default), no resampling is done and each lenslet's native
        (per-lenslet) wavelength solution and trace-column flux are
        returned instead — see "wave"/"flux"/"err" below.
    interp_order : int, optional
        Order of the polynomial for interpolation used to resample each lenslet's spectrum onto the common wavelength grid.
    output_path : str, optional
        If given, save the extracted cube (FLUX, ERR, WAVE extensions) here.

    Returns
    -------
    out : dict
        "wave" : ndarray, microns. Shape (n_wave,) if `output_wavesol` was
            given; otherwise shape (max_trace_len, n_lens_y, n_lens_x),
            the native per-lenslet wavelength solution.
        "flux" : ndarray, shape (n_wave, n_lens_y, n_lens_x) or
            (max_trace_len, n_lens_y, n_lens_x), matching "wave".
        "err"  : ndarray, same shape as "flux".
        "filepath" : str or None.
    """
    rectmat, offsets, wavesol = _load_rectmat(rectmat_path)

    n_valid_off = int(np.sum(offsets[0] >= 0))
    if n_valid_off == 0:
        raise RuntimeError("Rectmat has no valid lenslets — re-run make_rectmats.py first.")

    flux_tc, err_tc = extract_columns(
        data, rectmat, offsets, wavesol, err=error,
    )

    if output_wavesol is not None:
        cube_flux, cube_err = resample_to_common_grid(
            flux_tc, err_tc,
            offsets, wavesol,
            output_wavesol,
            order=interp_order,
        )
    else:
        # No common grid: fall back to each lenslet's native wavelength
        # solution and trace-column flux. Already wave-first
        # (max_trace_len, n_lens_y, n_lens_x), matching the resampled case.
        output_wavesol = wavesol
        cube_flux = flux_tc
        cube_err = err_tc

    out = {
        "wave": output_wavesol,
        "flux": cube_flux,
        "err": cube_err,
        "filepath": output_path,
    }

    if output_path is not None:
        save_extracted_cube_to_fits(output_wavesol, cube_flux, cube_err, output_path)

    return out
