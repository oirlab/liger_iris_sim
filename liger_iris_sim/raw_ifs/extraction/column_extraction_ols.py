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

    Neighbouring lenslet traces are packed closely enough in the
    cross-dispersion direction that their PSF footprints genuinely overlap
    on the detector -- this is what the rectification matrix is for. So
    within a detector column, traces are not solved independently: for the
    full set of traces active at that column, the design matrix

        A[row, k] = PSF weight of trace k at detector row `row`

    (mostly zero -- each trace only touches a handful of rows) is
    block-diagonal, since two traces only couple through AᵀWA if they
    share a detector row. Traces are grouped into these blocks by row-span
    overlap (an interval-graph connected-components problem, solved directly
    via a sort + sweep rather than a general graph routine, since the
    "edges" here are just 1D window overlaps), and each block's flux is the
    weighted least-squares solution of its own (small, dense) A f = d:

        f = (AᵀWA)⁻¹ Aᵀ W d

    with W = I (unweighted, err=None) or diag(1/sigma**2) (WLS, err
    given). This reduces exactly to the single-trace formula for isolated
    traces (the common case, handled directly without building a matrix)
    and jointly deconvolves overlapping ones from each other's
    contribution to their shared pixels. The per-trace flux uncertainty is
    sqrt of the corresponding diagonal of each block's (AᵀWA)⁻¹.

    Parameters
    ----------
    detector : ndarray
        Raw detector image, shape (n_rows, n_cols).
    rectmat : ndarray
        Rectification matrix, shape
        (n_row_window, max_trace_len, n_lens_y, n_lens_x). rectmat[r, tc, ly, lx]
        is the weight for detector pixel
        (offsets[1, ly, lx] + r, offsets[0, ly, lx] + tc).
    offsets : ndarray
        [col_start, row_start] per lenslet, shape (2, n_lens_y, n_lens_x).
        col_start < 0 marks an invalid/unused lenslet.
    wavesol : ndarray
        Wavelength solution per lenslet, shape
        (max_trace_len, n_lens_y, n_lens_x), microns. Zero-padded beyond
        the trace's actual length.
    err : ndarray, optional
        Per-pixel 1-sigma error for `detector`, same shape. If given, a
        weighted least-squares solve is used (weights = 1/err**2, with
        non-positive err treated as an infinitely-noisy/masked pixel and
        given zero weight) and `err_tc` is populated with the resulting
        flux uncertainty. If None (default), an unweighted least-squares
        solve is used and `err_tc` is left as NaN -- with no noise model
        there is nothing meaningful to report, so it is not fabricated
        from the data.
    verbose : bool, optional
        Print per-column progress. Default True.

    Returns
    -------
    flux_tc : ndarray
        Extracted flux per lenslet per trace-column, shape
        (max_trace_len, n_lens_y, n_lens_x). NaN where unfilled, or where
        a block's system was singular and could not be solved.
    err_tc : ndarray
        1-sigma error of `flux_tc`, same shape. NaN where unfilled, and
        entirely NaN if `err` is None.
    """
    n_row_window, max_trace_len, n_lens_y, n_lens_x = rectmat.shape
    det_n_rows, det_n_cols = detector.shape

    trace_lengths = np.sum(wavesol > 0, axis=0).astype(np.int32)
    col_starts = offsets[0]
    col_ends = np.where(col_starts >= 0, col_starts + trace_lengths, -1)
    row_starts = offsets[1]
    row_offsets = np.arange(n_row_window)

    flux_tc = np.full((max_trace_len, n_lens_y, n_lens_x), np.nan, dtype=np.float32)
    err_tc = np.full((max_trace_len, n_lens_y, n_lens_x), np.nan, dtype=np.float32)

    for col in range(det_n_cols):
        # Which lenslets are active at this detector column?
        active = (col_starts >= 0) & (col >= col_starts) & (col < col_ends)
        if not active.any():
            continue

        lys, lxs = np.where(active)               # active lenslet coords
        tcs = col - col_starts[lys, lxs]           # trace-local column index
        r0s = row_starts[lys, lxs]                 # first row of each trace's rectmat buffer

        P_stack = rectmat[:, tcs, lys, lxs].T.astype(np.float64)      # (n_active, n_row_window)
        rows_idx = np.clip(r0s[:, None] + row_offsets[None, :], 0, det_n_rows - 1)

        d_rows = detector[rows_idx, col].astype(np.float64)
        if err is None:
            w_rows = np.ones_like(d_rows)
        else:
            var_rows = err[rows_idx, col].astype(np.float64) ** 2
            w_rows = np.where(var_rows > 0, 1.0 / var_rows, 0.0)

        # Every trace's own (unmixed) normal-equation terms, Σ w P² and
        # Σ w P d over its own window -- the single-trace closed form,
        # correct as-is for isolated traces and reused as the RHS for
        # overlapping ones below.
        diag_self = np.sum(P_stack ** 2 * w_rows, axis=1)
        AtWd_self = np.sum(P_stack * w_rows * d_rows, axis=1)

        # Each trace's rectmat buffer (r0s .. r0s+n_row_window-1) spans its
        # whole diagonal excursion across the trace, not just this column --
        # the PSF actually only has nonzero weight over a handful of rows
        # near its centre at this particular column. Use that true nonzero
        # footprint (not the buffer bounds) to decide overlap, or nearly
        # every trace sharing a buffer row range would look connected.
        nz = P_stack != 0
        has_nz = nz.any(axis=1)
        first_nz = np.argmax(nz, axis=1)
        last_nz = n_row_window - 1 - np.argmax(nz[:, ::-1], axis=1)
        row_lo = r0s + first_nz
        row_hi = r0s + last_nz

        # Group traces by true-footprint row overlap (interval-graph
        # connected components via sort + sweep): two traces only share
        # detector pixels, and so only need a joint solve, if their
        # footprints overlap directly or transitively through a chain of
        # neighbours.
        active_idx = np.where(has_nz)[0]
        order = active_idx[np.argsort(row_lo[active_idx], kind="stable")]
        running_max_end = np.maximum.accumulate(row_hi[order])
        new_group = np.concatenate(([True], row_lo[order][1:] > running_max_end[:-1]))
        group_bounds = np.concatenate((np.where(new_group)[0], [order.size]))

        for g in range(len(group_bounds) - 1):
            members = order[group_bounds[g]:group_bounds[g + 1]]

            if members.size == 1:
                m = members[0]
                if diag_self[m] > 0:
                    flux_tc[tcs[m], lys[m], lxs[m]] = AtWd_self[m] / diag_self[m]
                    if err is not None:
                        err_tc[tcs[m], lys[m], lxs[m]] = 1.0 / np.sqrt(diag_self[m])
                continue

            # Overlapping group: solve the joint weighted least-squares
            # system on its own small, dense design matrix, built from each
            # member's true nonzero footprint only (not its full buffer).
            n_m = members.size
            g_row_start = int(row_lo[members].min())
            n_rows_g = int(row_hi[members].max()) - g_row_start + 1

            row_idx_g_raw = r0s[members][:, None] + row_offsets[None, :] - g_row_start
            col_idx_full = np.repeat(np.arange(n_m), n_row_window).reshape(n_m, n_row_window)
            mask = nz[members]
            A = np.zeros((n_rows_g, n_m), dtype=np.float64)
            A[row_idx_g_raw[mask], col_idx_full[mask]] = P_stack[members][mask]

            rows_abs = np.clip(g_row_start + np.arange(n_rows_g), 0, det_n_rows - 1)
            d = detector[rows_abs, col].astype(np.float64)
            if err is None:
                w = np.ones(n_rows_g, dtype=np.float64)
            else:
                var = err[rows_abs, col].astype(np.float64) ** 2
                w = np.where(var > 0, 1.0 / var, 0.0)

            Aw = A * w[:, None]
            AtWA = A.T @ Aw
            AtWd = Aw.T @ d

            try:
                cov = np.linalg.inv(AtWA)
            except np.linalg.LinAlgError:
                continue    # leave this group's flux/err as NaN

            flux = cov @ AtWd
            flux_tc[tcs[members], lys[members], lxs[members]] = flux
            if err is not None:
                err_tc[tcs[members], lys[members], lxs[members]] = np.sqrt(
                    np.clip(np.diag(cov), 0.0, None)
                )

        if verbose and col % 200 == 0:
            print(f"\r  col {col}/{det_n_cols}", end="", flush=True)

    if verbose:
        print(f"\r  col {det_n_cols}/{det_n_cols}  done                  ")

    return flux_tc, err_tc
