import numpy as np

from .convolve import convolve_point_source, convolve_spectrum
from ..utils.resampling import rebin_spectrum
from ..utils.psf_utils import shift_psf_phase

from numbers import Number

import logging
logger = logging.getLogger(__name__)

__all__ = ['make_point_source_ifs_cube']

def make_point_source_ifs_cube(
    xdet : float | np.ndarray, ydet : float | np.ndarray,
    wave : np.ndarray,
    psf : np.ndarray | list[np.ndarray],
    template : tuple[np.ndarray, np.ndarray] | np.ndarray,
    flux_int : float | np.ndarray | None = None,
    #template_units : str | None = None,
    psf_indices : np.ndarray | None = None,
    resolution : float | None = None,
    z : float | np.ndarray | None = None,
    size : tuple[int, int] | None = None,
    cube_out : np.ndarray | None = None,
) -> np.ndarray:
    """
    Render one or more point sources into an IFS data cube.

    Parameters
    ----------
    xdet : float | np.ndarray
        X-coordinate(s) of the point source(s) in detector pixels (second axis).
    ydet : float | np.ndarray
        Y-coordinate(s) of the point source(s) in detector pixels (first axis).
    flux_int : float | np.ndarray | None
        Integrated flux over the bandpass for each source in photons / sec / m^2.
        If provided, the template is scaled so that its sum over the output wave
        grid matches this value. If None, the template is used as-is (and must
        already be in physical units).
    wave : np.ndarray
        Output wavelength bin centres, in microns.
    template : tuple[np.ndarray, np.ndarray] | np.ndarray
        Spectral template. Two forms are accepted:

        - ``(wave_t, flux_t)`` — a tuple of 1-D arrays giving the template
          wavelengths (microns) and flux density (photons / sec / m^2 / micron).
          The template is integrated over each output bin via ``rebin_spectrum``.
        - ``np.ndarray`` — a 1-D array of length ``len(wave)`` already binned
          onto the output grid in photons / sec / m^2 per bin. No rebinning is
          applied unless a redshift is provided.
    psf : np.ndarray | list[np.ndarray]
        PSF image, or list of PSF images (for a spatially-dependent PSF).
    psf_indices : np.ndarray | None, optional
        When ``psf`` is a list, the integer index into that list for each
        source. Must have the same length as ``xdet``.
    resolution : float | None, optional
        If given, the flux-density template is convolved to this spectral
        resolution before rebinning.
    z : float | np.ndarray | None, optional
        Redshift(s) for each source. The template wavelengths are multiplied by
        ``(1 + z)`` before rebinning onto the output grid.
    size : tuple[int, int] | None, optional
        Spatial size of the output cube as ``(ny, nx)``. Required when
        ``cube_out`` is None.
    cube_out : np.ndarray | None, optional
        Pre-allocated output array of shape ``(len(wave), ny, nx)``. If None a
        new zero-filled float32 array is created.

    Returns
    -------
    cube_out : np.ndarray
        Data cube of shape ``(len(wave), ny, nx)`` in photons / sec / m^2 per
        spaxel per wavelength bin.
    """

    nw = len(wave)

    if cube_out is None:
        if size is None:
            raise ValueError("Either size or cube_out must be provided")
        cube_out = np.zeros((nw, *size), dtype=np.float32)
    else:
        size = cube_out.shape[1:]

    def to_array(x):
        return np.array([x]) if isinstance(x, Number) else np.asarray(x)

    xdet = to_array(xdet)
    ydet = to_array(ydet)
    n_sources = len(xdet)

    if z is not None:
        z = to_array(z)

    if flux_int is not None:
        flux_int = to_array(flux_int)

    if isinstance(template, (tuple, list)):
        t_wave, t_spec = template
        flux_density = True
    else:
        t_wave = wave
        t_spec = template
        flux_density = False

    if resolution is not None:
        t_spec = convolve_spectrum(t_wave, t_spec, resolution=resolution, n_res=4)

    for i in range(n_sources):

        _psf = psf[psf_indices[i]] if psf_indices is not None else psf

        # Apply redshift to template wavelengths
        _t_wave = t_wave * (1 + z[i]) if z is not None else t_wave

        # Rebin onto output wave grid
        if flux_density:
            # Convert flux density -> flux per bin, then rebin
            t_spec_z = rebin_spectrum(_t_wave, t_spec * np.gradient(_t_wave), wave)
        elif z is not None:
            # Already per-bin but redshifted off the output grid
            t_spec_z = rebin_spectrum(_t_wave, t_spec, wave)
        else:
            # Already binned on wave — use directly
            t_spec_z = t_spec

        # Scale to integrated flux if provided
        if flux_int is not None:
            total = t_spec_z.sum()
            if total > 0:
                t_spec_z = t_spec_z * (flux_int[i] / total)

        k = np.argmax(t_spec_z)
        if not (t_spec_z[k] > 0):
            logger.warning(
                f"Source {i} has zero flux in the output cube wavelength grid "
                "after rebinning. Skipping."
            )
            continue

        # Render the reference slice
        image_k = np.zeros(size, dtype=np.float32)
        
        convolve_point_source(
            xdet[i], ydet[i],
            t_spec_z[k],
            _psf, image_out=image_k,
            fix_psf_phase=True,
        )

        # Scale every slice by the spectral ratio relative to slice k
        for j in range(nw):
            cube_out[j] += image_k * (t_spec_z[j] / t_spec_z[k])

    return cube_out


def normalize_spectrum_to_snr(
    wave : np.ndarray,
    flux : np.ndarray,
    resolution : float,
    itime : float,
    snr : float,
) -> np.ndarray:
    """
    Rescale a spectrum's overall amplitude so its median wavelength bin
    reaches a target Poisson (shot-noise-limited) SNR, assuming Nyquist
    sampling of the spectral resolution element (2 bins per resolution
    element). Only Poisson noise is considered; read noise and other
    detector effects are ignored.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength grid, microns.
    flux : np.ndarray
        Flux density spectrum (e.g. photons/s/micron), sampled on `wave`.
        Any consistent rate-density units work; the result scales the same way.
    resolution : float
        Spectral resolution, R = lambda / fwhm.
    itime : float
        Integration time, seconds.
    snr : float
        Target SNR per wavelength bin.

    Returns
    -------
    flux_scaled : np.ndarray
        `flux` rescaled so its median bin reaches `snr`, on the same `wave` grid.
        Other bins scale with the local flux level, so brighter/fainter
        regions of the spectrum end up above/below `snr` accordingly.
    """
    fwhm = wave / resolution
    dlambda = fwhm / 2  # Nyquist sampling: 2 bins per resolution element

    counts = flux * dlambda * itime
    ref_counts = np.median(counts)

    scale = snr ** 2 / ref_counts
    return flux * scale
