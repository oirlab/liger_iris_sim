import numpy as np

__all__ = ['extract_columns']


def extract_columns(
    detector : np.ndarray,
    rectmat : np.ndarray,
    offsets : np.ndarray,
    wavesol : np.ndarray,
    err : np.ndarray = None,
    verbose : bool = True,
) -> tuple:
    """
    Extract flux one detector column at a time via a linear (weighted)
    least-squares solve against the rectification matrix.

    At each detector column, every lenslet trace that passes through it is
    identified and its 1-D flux is solved for directly:

        flux[tc] = Σ_r (P_r · d_r) / Σ_r (P_r²)                  [unweighted, err=None]
        flux[tc] = Σ_r (P_r · d_r / σ²_r) / Σ_r (P_r² / σ²_r)     [WLS, err given]

    Traces are assumed non-overlapping in the row direction, so the design
    matrix is block-diagonal and every lenslet active in a column can be
    solved independently (and vectorized together) rather than as one
    large linear system.

    Parameters
    ----------
    detector : ndarray
        Raw detector image, shape (n_rows, n_cols).
    rectmat : ndarray
        Rectification matrix, shape
        (n_lens_y, n_lens_x, n_row_window, max_trace_len). rectmat[ly, lx, r, tc]
        is the weight for detector pixel
        (offsets[ly, lx, 1] + r, offsets[ly, lx, 0] + tc).
    offsets : ndarray
        [col_start, row_start] per lenslet, shape (n_lens_y, n_lens_x, 2).
        col_start < 0 marks an invalid/unused lenslet.
    wavesol : ndarray
        Wavelength solution per lenslet, shape
        (n_lens_y, n_lens_x, max_trace_len), microns. Zero-padded beyond
        the trace's actual length.
    err : ndarray, optional
        Per-pixel 1-sigma error for `detector`, same shape. If given, a
        weighted least-squares solve is used (weights = 1/err**2) and
        `err_tc` is populated with the resulting flux uncertainty. If None
        (default), an unweighted least-squares solve is used and `err_tc`
        is left as NaN — with no noise model there is nothing meaningful
        to report, so it is not fabricated from the data.
    verbose : bool, optional
        Print per-column progress. Default True.

    Returns
    -------
    flux_tc : ndarray
        Extracted flux per lenslet per trace-column, shape
        (n_lens_y, n_lens_x, max_trace_len). NaN where unfilled.
    err_tc : ndarray
        1-sigma error of `flux_tc`, same shape. NaN where unfilled, and
        entirely NaN if `err` is None.
    """
    n_lens_y, n_lens_x, n_row_window, max_trace_len = rectmat.shape
    det_n_rows, det_n_cols = detector.shape

    trace_lengths = np.sum(wavesol > 0, axis=-1).astype(np.int32)
    col_starts = offsets[:, :, 0]
    col_ends = np.where(col_starts >= 0, col_starts + trace_lengths, -1)
    row_starts = offsets[:, :, 1]

    flux_tc = np.full((n_lens_y, n_lens_x, max_trace_len), np.nan, dtype=np.float32)
    err_tc = np.full((n_lens_y, n_lens_x, max_trace_len), np.nan, dtype=np.float32)

    for col in range(det_n_cols):
        # Which lenslets are active at this detector column?
        active = (col_starts >= 0) & (col >= col_starts) & (col < col_ends)
        if not active.any():
            continue

        lys, lxs = np.where(active)               # active lenslet coords
        tcs = col - col_starts[lys, lxs]           # trace-local column index
        r0s = row_starts[lys, lxs]                 # first row of strip

        # Row-index matrix: (n_active, n_row_window)
        row_idx = np.clip(r0s[:, None] + np.arange(n_row_window)[None, :],
                           0, det_n_rows - 1)

        # Detector column slice for each active lenslet: (n_active, n_row_window)
        d_stack = detector[row_idx, col]

        # PSF weights from rectmat: (n_active, n_row_window)
        rm_sub = rectmat[lys, lxs]                          # (n_active, n_row_window, max_tc)
        P_stack = rm_sub[np.arange(len(tcs)), :, tcs]       # (n_active, n_row_window)

        if err is None:
            num = np.sum(P_stack * d_stack, axis=1)         # (n_active,)
            denom = np.sum(P_stack * P_stack, axis=1)
        else:
            e_stack = err[row_idx, col]
            var = e_stack ** 2
            p_div_v = P_stack / var
            num = np.sum(p_div_v * d_stack, axis=1)
            denom = np.sum(p_div_v * P_stack, axis=1)

        good = denom > 0
        flux_tc[lys[good], lxs[good], tcs[good]] = num[good] / denom[good]
        if err is not None:
            err_tc[lys[good], lxs[good], tcs[good]] = 1.0 / np.sqrt(denom[good])

        if verbose and col % 200 == 0:
            print(f"\r  col {col}/{det_n_cols}  active={active.sum()}", end="", flush=True)

    if verbose:
        print(f"\r  col {det_n_cols}/{det_n_cols}  done                  ")

    return flux_tc, err_tc
