import numpy as np
from .integration import _get_channel_edges

def fit_trace(
    x_pts : np.ndarray, y_pts : np.ndarray,
    wave_pts : np.ndarray,
    tracepos_deg : int,
    wavesol_deg : int
):
    """
    Fit y(x) and lambda(x) through the Zemax points.
    `x_pts` and `y_pts` are in detector pixels.
    `wave_pts` is in microns.

    Returns
    -------
    y_of_x : callable
        The function y(x) giving the trace's row (sub-pixel) in pixels for any column x.
    wave_of_x : callable
        The function lambda(x) giving the trace's wavelength in microns for any column x.
    """
    x0 = float(x_pts.mean())
    scale = max(float(np.ptp(x_pts)) / 2.0, 1.0)   # condition the fit
    u = (x_pts - x0) / scale

    coef_y = np.polyfit(u, y_pts, tracepos_deg)
    coef_w = np.polyfit(u, wave_pts, wavesol_deg)

    def y_of_x(x : np.ndarray) -> np.ndarray:
        return np.polyval(coef_y, (x - x0) / scale)

    def wave_of_x(x : np.ndarray) -> np.ndarray:
        return np.polyval(coef_w, (x - x0) / scale)

    return y_of_x, wave_of_x


def get_trace_geometry(
    x_pix_pts : np.ndarray,
    y_pix_pts : np.ndarray,
    wave_pts : np.ndarray,
    wave : np.ndarray,
    tracepos_deg : int,
    wavesol_deg : int,
    density : bool,
    pad_ends : bool
):
    """
    The single definition of "where does this trace start and stop, and what
    are y(x) and lambda(x) along it".  Used by the renderer and any flux check,
    so the two cannot drift apart.

    Returns
    -------
    x_lo, x_hi : float
        First and last pixel column the trace spans (fractional).
    y_of_x : callable
        The function y(x) giving the trace's row (sub-pixel) in pixels for any column x.
    wave_of_x : callable
        The function lambda(x) giving the trace's wavelength in microns for any column x.
    """

    # Start and stop of the trace in detector pixels
    x_lo = float(np.min(x_pix_pts))
    x_hi = float(np.max(x_pix_pts))
    y_of_x, wave_of_x = fit_trace(
        x_pix_pts, y_pix_pts, wave_pts,
        tracepos_deg=tracepos_deg,
        wavesol_deg=wavesol_deg
    )

    if pad_ends and not density and x_hi > x_lo:
        # The Zemax points sit at channel CENTRES, so the outer half-channel of
        # the cube falls beyond them and would be clipped -- a silent 1/n_wave
        # flux leak.  Extend just far enough to catch it, capped at 2 pixels so
        # a bad fit cannot run away.
        edges_w = _get_channel_edges(wave)
        half = 0.5 * max(edges_w[1] - edges_w[0], edges_w[-1] - edges_w[-2])
        disp = min(
            abs(wave_of_x(x_lo + 1.0) - wave_of_x(x_lo)),
            abs(wave_of_x(x_hi) - wave_of_x(x_hi - 1.0))
        )
        if disp > 0:
            pad = min(half / disp, 2.0)
            x_lo -= pad
            x_hi += pad

    return x_lo, x_hi, y_of_x, wave_of_x
