import numpy as np

def _resample_to_common_grid(
    flux_tc : np.ndarray,
    err_tc : np.ndarray,
    offsets : np.ndarray,
    wavesol : np.ndarray,
    output_wavesol : np.ndarray,
    density : bool = False,
) -> tuple:
    """
    Resample every lenslet's per-trace-column flux onto a single common
    wavelength grid, provided by the caller (must be monotonically
    increasing).

    Parameters
    ----------
    flux_tc, err_tc : ndarray
        Output of `extract_columns`, shape
        (n_lens_y, n_lens_x, max_trace_len).
    offsets : ndarray
        [col_start, row_start] per lenslet, shape (n_lens_y, n_lens_x, 2).
    wavesol : ndarray
        Wavelength solution per lenslet, shape
        (n_lens_y, n_lens_x, max_trace_len), microns.
    output_wavesol : ndarray
        Common wavelength grid to resample onto, microns. Must be
        monotonically increasing.
    density : bool, optional
        If True, convert flux from phot/s (per trace-column) to
        phot/s/micron by dividing by the local wavelength bin width
        (and rescale err to match). Default False.

    Returns
    -------
    cube_flux : ndarray
        Resampled cube flux, shape (n_wave, n_lens_y, n_lens_x).
    cube_err : ndarray
        Resampled 1-sigma error, shape (n_wave, n_lens_y, n_lens_x).
    """
    n_lens_y, n_lens_x, _ = wavesol.shape
    trace_lengths = np.sum(wavesol > 0, axis=-1)
    n_wave = len(output_wavesol)

    cube_flux = np.full((n_wave, n_lens_y, n_lens_x), np.nan, dtype=np.float32)
    cube_err = np.full((n_wave, n_lens_y, n_lens_x), np.nan, dtype=np.float32)

    for ly in range(n_lens_y):
        for lx in range(n_lens_x):
            if offsets[ly, lx, 0] < 0:
                continue
            tlen = int(trace_lengths[ly, lx])
            if tlen < 2:
                continue
            w = wavesol[ly, lx, :tlen]
            f = flux_tc[ly, lx, :tlen]
            e = err_tc[ly, lx, :tlen]
            good = np.isfinite(f) & (w > 0)
            if good.sum() < 2:
                continue

            # np.interp requires xp increasing; wavesol may run blue->red
            # or red->blue depending on trace orientation, so sort explicitly.
            order = np.argsort(w[good])
            w_sorted = w[good][order]
            cube_flux[:, ly, lx] = np.interp(
                output_wavesol, w_sorted, f[good][order],
                left=np.nan, right=np.nan
            )
            cube_err[:, ly, lx] = np.interp(
                output_wavesol, w_sorted, e[good][order],
                left=np.nan, right=np.nan
            )

    if density:
        dwave = np.abs(np.gradient(output_wavesol)).astype(np.float32)[:, None, None]
        cube_flux = cube_flux / dwave
        cube_err = cube_err / dwave

    return cube_flux, cube_err
