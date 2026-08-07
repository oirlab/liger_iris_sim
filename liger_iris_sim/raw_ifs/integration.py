import numpy as np

__all__ = ["integrate_spectrum"]

def _integral_to(wave : np.ndarray, flux : np.ndarray, x : np.ndarray) -> np.ndarray:
    """
    Integral of `flux` d(lambda) from wave[0] up to each x.
    The input spectrum is linearly interpolated between its samples.
    Values outside [wave[0], wave[-1]] are 0.
    """
    x = np.clip(np.asarray(x, dtype=np.float64), wave[0], wave[-1])
    dw = np.diff(wave)
    slope = np.diff(flux) / dw
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (flux[:-1] + flux[1:]) * dw)))
    k = np.clip(np.searchsorted(wave, x, side="right") - 1, 0, wave.size - 2)
    d = x - wave[k]
    return cum[k] + flux[k] * d + 0.5 * slope[k] * d * d


def _get_channel_edges(wave : np.ndarray) -> np.ndarray:
    """Wavelength bin edges for a spectrum sampled at `wave`."""
    mid = 0.5 * (wave[:-1] + wave[1:])
    return np.concatenate(([2 * wave[0] - mid[0]], mid, [2 * wave[-1] - mid[-1]]))


def integrate_spectrum(
    wave : np.ndarray, flux : np.ndarray,
    wave_edges : np.ndarray,
    density : bool,
) -> np.ndarray:
    """
    Integrate a spectrum over wavelength bins.

    Parameters
    ----------
    wave : ndarray
        The wavelength samples of the input spectrum.
    flux : ndarray
        The flux samples of the input spectrum.
    wave_edges : ndarray
        The wavelength bin edges to integrate over.
    density : bool
        If True, the input flux is a density (per unit wavelength) and the output is the integral over each bin.
        If False, the input flux is already integrated over each bin and the output is the integral over each bin.

    Returns
    -------
    flux_out : ndarray
        The integrated flux over each wavelength bin defined by `wave_edges`.
    """
    if density:
        c = _integral_to(wave, flux, wave_edges)
    else:
        cum = np.concatenate(([0.0], np.cumsum(flux)))
        c = np.interp(wave_edges, _get_channel_edges(wave), cum)
    flux_out = np.abs(np.diff(c))
    return flux_out
