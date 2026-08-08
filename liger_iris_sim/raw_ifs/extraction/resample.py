import numpy as np

from ...utils.resampling import rebin_spectrum, _build_edges

__all__ = ["resample_to_common_grid"]

def resample_to_common_grid(
    flux_tc : np.ndarray,
    err_tc : np.ndarray,
    offsets : np.ndarray,
    wavesol : np.ndarray,
    output_wavesol : np.ndarray,
    order : int = 1,
) -> tuple:
    """
    Resample every lenslet's per-trace-column flux onto a single common
    wavelength grid, provided by the caller (must be monotonically
    increasing), via `rebin_spectrum`
    (see `liger_iris_sim.utils.resampling`).

    Each lenslet's native (wavesol, flux) samples are modeled as
    piecewise-constant per native bin (order=0), piecewise-linear through
    the samples (order=1, default), or piecewise-quadratic through the
    samples (order=2). That model is integrated over each output bin --
    edges placed at the midpoints between `output_wavesol` samples,
    extrapolated at the ends -- giving a flux-conserving bin sum rather
    than a point sample. This avoids aliasing when the native and output
    grids have different (and possibly locally varying) spectral
    sampling. Errors are propagated in quadrature through the same
    integration weights, assuming independent errors at each native
    sample. An output bin is NaN unless it is fully spanned by the
    lenslet's native wavelength coverage.

    Parameters
    ----------
    flux_tc, err_tc : ndarray
        Output of `extract_columns`, shape
        (max_trace_len, n_lens_y, n_lens_x). Units of phot/s.
    offsets : ndarray
        [col_start, row_start] per lenslet, shape (2, n_lens_y, n_lens_x).
    wavesol : ndarray
        Wavelength solution per lenslet, shape
        (max_trace_len, n_lens_y, n_lens_x), microns.
    output_wavesol : ndarray
        Common wavelength grid to resample onto (bin centers), microns.
        Must be monotonically increasing.
    order : int, optional
        Order of the assumed native spectrum model, passed through to
        `rebin_spectrum`: 0 (piecewise-constant), 1 (piecewise-linear,
        default), or 2 (piecewise-quadratic).

    Returns
    -------
    cube_flux : ndarray
        Resampled cube flux, shape (n_wave, n_lens_y, n_lens_x), phot/s.
    cube_err : ndarray
        Resampled 1-sigma error, shape (n_wave, n_lens_y, n_lens_x), phot/s.
    """
    _, n_lens_y, n_lens_x = wavesol.shape
    trace_lengths = np.sum(wavesol > 0, axis=0)
    n_wave = len(output_wavesol)
    out_edges = _build_edges(np.asarray(output_wavesol, dtype=np.float64))

    cube_flux = np.full((n_wave, n_lens_y, n_lens_x), np.nan, dtype=np.float32)
    cube_err = np.full((n_wave, n_lens_y, n_lens_x), np.nan, dtype=np.float32)

    for ly in range(n_lens_y):
        for lx in range(n_lens_x):
            if offsets[0, ly, lx] < 0:
                continue
            tlen = int(trace_lengths[ly, lx])
            if tlen < 2:
                continue
            w = wavesol[:tlen, ly, lx]
            f = flux_tc[:tlen, ly, lx]
            e = err_tc[:tlen, ly, lx]
            good = np.isfinite(f) & (w > 0)
            if good.sum() < 2:
                continue

            # wavesol may run blue->red or red->blue depending on trace
            # orientation, so sort explicitly to get an increasing grid.
            sort_idx = np.argsort(w[good])
            w_sorted = w[good][sort_idx]
            f_sorted = f[good][sort_idx]
            e_sorted = e[good][sort_idx]

            if order == 0:
                # flux_tc is already the extensive (phot/s) value
                # integrated over its own native bin -- exactly what the
                # box-rebin (order=0) expects, no conversion needed.
                f_in = f_sorted
                e_in = e_sorted
            else:
                # order=1/2 treat each sample as a point value of a
                # continuous phot/s/micron density, connected by straight
                # lines (order=1) or local quadratics (order=2). flux_tc is
                # extensive, so divide out the native per-column bin width
                # to recover a density before handing it to rebin_spectrum.
                native_bin_width = np.diff(_build_edges(np.asarray(w_sorted, dtype=np.float64)))
                f_in = f_sorted / native_bin_width
                e_in = e_sorted / native_bin_width

            # rebin_spectrum returns the bin-integrated (extensive) phot/s
            # value directly -- no division by output bin width, since the
            # input flux_tc's are already extensive (phot/s per native bin).
            flux_bin, err_bin = rebin_spectrum(
                w_sorted, f_in, output_wavesol, err=e_in, order=order
            )

            covered = (out_edges[:-1] >= w_sorted[0]) & (out_edges[1:] <= w_sorted[-1])
            cube_flux[:, ly, lx] = np.where(covered, flux_bin, np.nan)
            cube_err[:, ly, lx] = np.where(covered, err_bin, np.nan)

    return cube_flux, cube_err
