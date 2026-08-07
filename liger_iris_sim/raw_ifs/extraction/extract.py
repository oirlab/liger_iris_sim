import os
import numpy as np
from astropy.io import fits

from .column_extraction_ols import extract_columns
from .resample import _resample_to_common_grid

__all__ = ["extract_ifs_lenslet"]


def _load_rectmat(rectmat_path : str) -> tuple:
    """
    Load a rectification matrix, e.g. as produced by `make_rectmats.py`.

    Returns
    -------
    rectmat : ndarray
        Shape (n_lens_y, n_lens_x, n_row_window, max_trace_len).
    offsets : ndarray
        [col_start, row_start] per lenslet, shape (n_lens_y, n_lens_x, 2).
    wavesol : ndarray
        Wavelength solution per lenslet, shape
        (n_lens_y, n_lens_x, max_trace_len), microns.
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
        Common wavelength grid, shape (n_wave,), microns.
    cube_flux, cube_err : ndarray
        Flux and its 1-sigma error, shape (n_wave, n_lens_y, n_lens_x).
    output_path : str
        Path to write the FITS file. Extensions: "FLUX", "ERR", "WAVE".
        Parent directory is created if needed.
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    wave_hdr = fits.Header()
    wave_hdr["BUNIT"] = "micron"
    wave_hdr["WAVEMIN"] = float(wave_out[0])
    wave_hdr["WAVEMAX"] = float(wave_out[-1])

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
    output_wavesol : np.ndarray,
    error : np.ndarray = None,
    output_path : str = None,
    density : bool = False,
) -> dict:
    """
    Main function user calls to extract a lenslet IFS cube from a raw
    detector frame, given a precomputed rectification matrix.

    Extraction is a column-by-column linear (weighted) least-squares solve
    against the rectification matrix: at each detector column, every
    lenslet trace passing through it is solved for independently (traces
    are assumed non-overlapping in the row direction, so the design matrix
    is block-diagonal). Per-lenslet spectra are then resampled onto the
    caller-provided common wavelength grid `output_wavesol`. See
    `column_extraction_ols.extract_columns` and
    `resample._resample_to_common_grid` for details.

    Parameters
    ----------
    data : ndarray
        Raw detector image, shape (n_rows, n_cols).
    rectmat_path : str
        Path to the rectification matrix FITS file, with RECTMAT,
        OFFSETS, and WAVESOL extensions.
    output_wavesol : ndarray
        Common wavelength grid to resample every lenslet onto, shape
        (n_wave,), microns. Must be monotonically increasing.
    error : ndarray, optional
        Per-pixel 1-sigma error for `data`, same shape. If given, a
        weighted least-squares solve is used and the returned "err" is
        populated. If None (default), an unweighted least-squares solve
        is used and "err" is NaN — see `column_extraction_ols.extract_columns`.
    output_path : str, optional
        If given, save the extracted cube (FLUX, ERR, WAVE extensions) here.
    density : bool, optional
        If True, convert the output flux from phot/s (per trace-column) to
        phot/s/micron by dividing by the local wavelength bin width.
        Default False.

    Returns
    -------
    out : dict
        "wave" : ndarray, shape (n_wave,), microns.
        "flux" : ndarray, shape (n_wave, n_lens_y, n_lens_x).
        "err"  : ndarray, shape (n_wave, n_lens_y, n_lens_x).
        "filepath" : str or None.
    """
    rectmat, offsets, wavesol = _load_rectmat(rectmat_path)

    n_valid_off = int(np.sum(offsets[:, :, 0] >= 0))
    if n_valid_off == 0:
        raise RuntimeError("Rectmat has no valid lenslets — re-run make_rectmats.py first.")

    flux_tc, err_tc = extract_columns(
        data, rectmat, offsets, wavesol, err=error,
    )

    cube_flux, cube_err = _resample_to_common_grid(
        flux_tc, err_tc, offsets, wavesol, output_wavesol, density=density,
    )

    out = {
        "wave": output_wavesol,
        "flux": cube_flux,
        "err": cube_err,
        "filepath": output_path,
    }

    if output_path is not None:
        save_extracted_cube_to_fits(output_wavesol, cube_flux, cube_err, output_path)

    return out
