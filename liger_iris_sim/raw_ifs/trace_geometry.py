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
    tracepos_deg : int,
    wavesol_deg : int,
):
    """
    Determine the start and stop columns of the trace for this lenslet.
    Determines the trace position y(x) and wavelength solution lambda(x).

    All units are in detector pixels (x, y) and microns (lambda).

    Parameters
    ----------
    x_pix_pts : np.ndarray
        Zemax trace x-pixel points (detector columns).
    y_pix_pts : np.ndarray
        Zemax trace y-pixel points (detector rows).
    wave_pts : np.ndarray
        Zemax trace wavelength points (microns).
    tracepos_deg : int
        Degree of polynomial to fit the trace position y(x).
    wavesol_deg : int
        Degree of polynomial to fit the wavelength solution lambda(x).

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

    return x_lo, x_hi, y_of_x, wave_of_x
