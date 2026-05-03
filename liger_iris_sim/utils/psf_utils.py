import numpy as np

from liger_iris_drp_resources import load_liger_psf, load_iris_psf
from liger_iris_drp_resources.psfs import _get_liger_psf_filename, _get_iris_psf_filename
from .resampling import rebin_image

__all__ = [
    "get_psfs",
    "get_psf",
    #"get_psf_interp",
    "crop_AO_psf",
    "shift_psf_phase",
    "extend_psf_powerlaw",
]

# PSF grid constants (from liger_iris_drp_resources)
#_LIGER_PSF_WAVE_GRID = np.array([1.02, 1.248, 1.65, 2.124])           # µm  (Y/J/H/K)
#_LIGER_PSF_SPAT_GRID = np.array([-15., -10., -5., 0., 5., 10., 15.])  # arcsec
#_IRIS_PSF_SPAT_GRID_IMG = np.array([0.6, 4.7, 8.8, 12.9, 17.0])       # arcsec

def get_psfs(
    instrument_name : str,
    instrument_mode : str | None = None,
    wave : float | None = None,
    xs : float | None = None, ys : float | None = None,
    xdet : float | None = None, ydet : float | None = None,
    output_plate_scale : float | None = None,
    recenter_to_odd_shape : bool = True,
    extend_powerlaw : bool | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Get the PSF for a given instrument and mode.

    Parameters
    ----------
    instrument_name : str
        The name of the instrument (e.g. 'Liger', 'Iris').
    instrument_mode : str, optional
        The instrument mode (e.g. 'img', 'ifs').
        Not used for Liger.
    wave : float, optional
        The wavelength for which to retrieve the PSF.
    xs, ys : np.ndarray, optional
        The spatial offset in arcseconds from the PSF center. Defaults to (0, 0).
    xdet, ydet : np.ndarray, optional
        The detector offset in pixels from the PSF center. Defaults to (0, 0).
    output_plate_scale : float, optional
        If provided, the PSF will be resampled to this plate scale in arcsec/pixel.
    recenter_to_odd_shape : bool, optional
        If True, the output PSF will be recentered to have an odd number of rows and columns. Default is True.
    extend_powerlaw : bool | None, optional
        If True, the PSF will be extended with a power-law tail. If None, the PSF will be extended if the instrument is Liger in imaging mode. Default is None.
    """
    
    # Load the PSF
    inst_name = instrument_name.lower()
    inst_mode = instrument_mode.lower() if instrument_mode is not None else None
    psfs = []
    infos = []
    indices = np.empty(len(xdet), dtype=int)
    key_to_index = {}
    if inst_name == 'liger':
        for i in range(len(xdet)):
            filename, hdunum = _get_liger_psf_filename(
                instrument_mode=inst_mode,
                wave=wave,
                xs=xs, ys=ys,
                xdet=xdet[i], ydet=ydet[i],
            )
            key = (filename, hdunum)
            if key not in key_to_index:
                idx = len(psfs)
                psf, info = get_psf(
                    instrument_name=instrument_name,
                    instrument_mode=instrument_mode,
                    wave=wave,
                    xs=xs, ys=ys,
                    xdet=xdet[i], ydet=ydet[i],
                    recenter_to_odd_shape=recenter_to_odd_shape,
                    extend_powerlaw=extend_powerlaw,
                    output_plate_scale=output_plate_scale,
                )

                key_to_index[key] = idx
                psfs.append(psf)
                infos.append(info)

            else:
                idx = key_to_index[key]

            indices[i] = idx
    elif inst_name == 'iris':
        for i in range(len(xdet)):
            filename, hdunum = _get_iris_psf_filename(
                instrument_mode=inst_mode,
                wave=wave,
                xs=xs, ys=ys,
                xdet=xdet[i], ydet=ydet[i],
            )

            key = (filename, hdunum)

            if key not in key_to_index:
                idx = len(psfs)

                psf, info = get_psf(
                    instrument_name=instrument_name,
                    instrument_mode=instrument_mode,
                    wave=wave,
                    xs=xs, ys=ys,
                    xdet=xdet[i], ydet=ydet[i],
                    recenter_to_odd_shape=recenter_to_odd_shape,
                    extend_powerlaw=extend_powerlaw,
                    output_plate_scale=output_plate_scale,
                )

                key_to_index[key] = idx
                psfs.append(psf)
                infos.append(info)
            else:
                idx = key_to_index[key]

            indices[i] = idx
        
    else:
        raise ValueError(f"Invalid instrument name: {instrument_name}")
    
    # # Rebin image to output plate scale
    # input_scale = info['psf_sampling']
    # if output_plate_scale is not None and input_scale != output_plate_scale:
    #     psf = rebin_image(
    #         psf,
    #         scale_in=input_scale,
    #         scale_out=output_plate_scale,
    #         recenter_to_odd_shape=False
    #     )
    #     info['psf_sampling'] = output_plate_scale

    return psfs, infos, indices


def get_psf(
    instrument_name : str,
    instrument_mode : str | None = None,
    wave : float | None = None,
    xs : float | None = None, ys : float | None = None,
    xdet : float | None = None, ydet : float | None = None,
    output_plate_scale : float | None = None,
    recenter_to_odd_shape : bool = True,
    extend_powerlaw : bool | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Get the PSF for a given instrument and mode.

    Parameters
    ----------
    instrument_name : str
        The name of the instrument (e.g. 'Liger', 'Iris').
    instrument_mode : str, optional
        The instrument mode (e.g. 'img', 'ifs').
        Not used for Liger.
    wave : float, optional
        The wavelength for which to retrieve the PSF.
    xs, ys : float, optional
        The spatial offset in arcseconds from the PSF center. Defaults to (0, 0).
    xdet, ydet : float, optional
        The detector offset in pixels from the PSF center. Defaults to (0, 0).
    output_plate_scale : float, optional
        If provided, the PSF will be resampled to this plate scale in arcsec/pixel.
    recenter_to_odd_shape : bool, optional
        If True, the output PSF will be recentered to have an odd number of rows and columns. Default is True.
    extend_powerlaw : bool | None, optional
        If True, the PSF will be extended with a power-law tail. If None, the PSF will be extended if the instrument is Liger in imaging mode. Default is None.
    """
    
    # Load the PSF
    inst_name = instrument_name.lower()
    inst_mode = instrument_mode.lower() if instrument_mode is not None else None
    if inst_name == 'liger':
        psf, info = load_liger_psf(
            instrument_mode=instrument_mode,
            wave=wave,
            xs=xs, ys=ys,
            xdet=xdet, ydet=ydet,
        )
    elif inst_name == 'iris':
        psf, info = load_iris_psf(
            instrument_mode=instrument_mode,
            wave=wave,
            xs=xs, ys=ys,
            xdet=xdet, ydet=ydet,
        )
    else:
        raise ValueError(f"Invalid instrument name: {instrument_name}")
    
    # Rebin image to output plate scale
    input_scale = info['psf_sampling']
    if output_plate_scale is not None and input_scale != output_plate_scale:
        psf = rebin_image(
            psf,
            scale_in=input_scale,
            scale_out=output_plate_scale,
            recenter_to_odd_shape=recenter_to_odd_shape,
        )
        info['psf_sampling'] = output_plate_scale

    # Optionally extend the PSF with a power-law tail for Liger
    if extend_powerlaw is None:
        extend_powerlaw = inst_name == 'liger' and inst_mode == 'img'
    if extend_powerlaw:
        if psf.shape[0] > 100:
            rmin = 50
        elif psf.shape[0] > 50:
            rmin = 25
        else:
            rmin = int(0.6 * min(psf.shape) // 2)
        psf = extend_psf_powerlaw(
            psf,
            rmin=rmin,
            extend_factor=1.5,
            alpha_min=2.0
        )
        #import matplotlib
        #matplotlib.use("QTAGG")
        #import matplotlib.pyplot as plt
        #breakpoint()
        # psf1 = psf.copy()
        # psf2 = extend_psf_powerlaw(
        #     psf,
        #     rmin=20,
        #     extend_factor=1.5,
        #     alpha_min=2.0
        # )
        # x1 = np.arange(psf.shape[1]) - psf.shape[1] // 2
        # x2 = np.arange(psf2.shape[1]) - psf2.shape[1] // 2
        # plt.plot(x1, np.log(psf1[:, psf1.shape[1]//2]), label='Original PSF')
        # plt.plot(x2, np.log(psf2[:, psf2.shape[1]//2]), label='Extended PSF')
        # plt.show()

    # Ensure shape is odd
    if recenter_to_odd_shape:
        psf = _recenter_psf_to_odd_shape(psf)

    return psf, info


# def get_psf_interp(
#     instrument_name: str,
#     instrument_mode: str | None = None,
#     wave: float | None = None,
#     xs: float | None = None, ys: float | None = None,
#     xdet: float | None = None, ydet: float | None = None,
#     output_plate_scale: float | None = None,
#     recenter_to_odd_shape: bool = True,
#     extend_powerlaw: bool | None = None,
#     interp_wave: bool = True,
#     interp_spat: bool = True,
# ) -> tuple[np.ndarray, dict]:
#     """
#     Get an interpolated PSF for a given instrument and mode.

#     Unlike ``get_psf()``, which snaps to the nearest grid point, this function
#     linearly interpolates in wavelength (``interp_wave=True``) and/or
#     bilinearly in the spatial dimension (``interp_spat=True``) between the
#     available PSF grid points.  Spatial interpolation is a no-op for IFS mode
#     (single on-axis PSF).

#     Parameters
#     ----------
#     instrument_name : str
#         'Liger' or 'Iris'.
#     instrument_mode : str, optional
#         'img' or 'ifs'.
#     wave : float, optional
#         Wavelength in µm.
#     xs, ys : float, optional
#         Sky offset in arcseconds.
#     xdet, ydet : float, optional
#         Detector pixel offset (converted to arcsec internally).
#     output_plate_scale : float
#         Resample all grid PSFs to this plate scale in arcsec/pixel before
#         interpolating. Required (cannot be None).
#     recenter_to_odd_shape : bool
#         Recenter output to an odd shape. Default True.
#     extend_powerlaw : bool | None
#         Extend the PSF with a power-law halo (see ``get_psf`` for details).
#     interp_wave : bool
#         Interpolate linearly in wavelength between grid points. Default True.
#     interp_spat : bool
#         Interpolate bilinearly in (xs, ys) between grid points. Default True.

#     Returns
#     -------
#     psf : np.ndarray
#         Interpolated PSF (float32), normalised to unit sum.
#     info : dict
#         Metadata from the dominant contributing grid PSF.
#     """
#     if output_plate_scale is None:
#         raise ValueError("output_plate_scale is required for get_psf_interp.")

#     inst_name = instrument_name.lower()
#     inst_mode = instrument_mode.lower() if instrument_mode is not None else None

#     # ------------------------------------------------------------------
#     # Resolve xs/ys
#     # ------------------------------------------------------------------
#     if inst_mode == 'ifs':
#         xs, ys = 0.0, 0.0
#     else:
#         if xdet is not None and ydet is not None and xs is None:
#             det_scale = 0.01 if inst_name == 'liger' else 0.004  # arcsec/pixel
#             xs = det_scale * float(xdet)
#             ys = det_scale * float(ydet)
#         if xs is None:
#             xs, ys = 0.0, 0.0

#     # ------------------------------------------------------------------
#     # Instrument-specific grids
#     # ------------------------------------------------------------------
#     if inst_name == 'liger':
#         wave_grid = _LIGER_PSF_WAVE_GRID
#         spat_grid = _LIGER_PSF_SPAT_GRID
#     elif inst_name == 'iris':
#         spat_grid = _IRIS_PSF_SPAT_GRID_IMG if inst_mode == 'img' else None
#         wave_grid = None  # read from FITS HDUs below
#     else:
#         raise ValueError(f"Invalid instrument name: {instrument_name}")

#     # For IRIS, derive the wavelength grid from the FITS file's HDU headers.
#     if inst_name == 'iris' and interp_wave and wave is not None:
#         from liger_iris_drp_resources.psfs import (
#             _get_iris_psf_dir,
#             _get_iris_psf_filename,
#             _parse_iris_psf_header,
#         )
#         from astropy.io import fits as _fits
#         import os as _os
#         xs_snap = (
#             spat_grid[np.argmin(np.abs(spat_grid - xs))]
#             if inst_mode == 'img' else 0.0
#         )
#         ys_snap = (
#             spat_grid[np.argmin(np.abs(spat_grid - ys))]
#             if inst_mode == 'img' else 0.0
#         )
#         iris_fp = _os.path.join(
#             _get_iris_psf_dir(),
#             _get_iris_psf_filename(instrument_mode=inst_mode, xs=xs_snap, ys=ys_snap),
#         )
#         with _fits.open(iris_fp) as hdul:
#             wave_grid = np.array([
#                 _parse_iris_psf_header(hdul[i].header)['wavelength']
#                 for i in range(len(hdul))
#             ])

#     # ------------------------------------------------------------------
#     # Compute 1-D linear interpolation weights
#     # ------------------------------------------------------------------
#     def _linear_weights(grid, val):
#         """Return [(grid_value, weight), ...] for linear interpolation."""
#         if grid is None or val is None:
#             return [(val, 1.0)]
#         val = float(val)
#         if val <= grid[0]:
#             return [(grid[0], 1.0)]
#         if val >= grid[-1]:
#             return [(grid[-1], 1.0)]
#         idx = int(np.searchsorted(grid, val))
#         g0, g1 = grid[idx - 1], grid[idx]
#         t = (val - g0) / (g1 - g0)
#         return [(g0, 1.0 - t), (g1, t)]

#     wave_w = _linear_weights(wave_grid, wave) if interp_wave else [(wave, 1.0)]
#     if interp_spat and inst_mode != 'ifs' and spat_grid is not None:
#         xs_w = _linear_weights(spat_grid, xs)
#         ys_w = _linear_weights(spat_grid, ys)
#     else:
#         xs_w = [(xs, 1.0)]
#         ys_w = [(ys, 1.0)]

#     # ------------------------------------------------------------------
#     # Load required PSFs (deduplicated)
#     # If output_plate_scale is not given, the first loaded PSF's native
#     # psf_sampling is used as the common scale for all subsequent loads,
#     # so all PSFs are on the same pixel grid before blending.
#     # ------------------------------------------------------------------
#     psf_cache: dict = {}

#     def _load(xi, yi, wi):
#         key = (
#             round(float(xi), 6) if xi is not None else None,
#             round(float(yi), 6) if yi is not None else None,
#             round(float(wi), 8) if wi is not None else None,
#         )
#         if key not in psf_cache:
#             psf_cache[key] = get_psf(
#                 instrument_name=instrument_name,
#                 instrument_mode=instrument_mode,
#                 wave=wi,
#                 xs=xi, ys=yi,
#                 output_plate_scale=output_plate_scale,
#                 recenter_to_odd_shape=recenter_to_odd_shape,
#                 extend_powerlaw=extend_powerlaw,
#             )
#         return psf_cache[key]

#     # ------------------------------------------------------------------
#     # Accumulate weighted PSFs
#     # ------------------------------------------------------------------
#     contributions: list[tuple[float, np.ndarray, dict]] = []
#     for xi, wx in xs_w:
#         for yi, wy in ys_w:
#             for wi, ww in wave_w:
#                 weight = wx * wy * ww
#                 if weight < 1e-14:
#                     continue
#                 psf_i, info_i = _load(xi, yi, wi)
#                 contributions.append((weight, psf_i, info_i))

#     # ------------------------------------------------------------------
#     # Pad all PSFs to a common (largest) odd shape, then blend
#     # ------------------------------------------------------------------
#     max_ny = max(p.shape[0] for _, p, _ in contributions)
#     max_nx = max(p.shape[1] for _, p, _ in contributions)
#     if max_ny % 2 == 0:
#         max_ny += 1
#     if max_nx % 2 == 0:
#         max_nx += 1

#     psf_out = np.zeros((max_ny, max_nx), dtype=np.float64)
#     for weight, psf_i, _ in contributions:
#         psf_out += weight * _pad_psf_to_shape(psf_i, (max_ny, max_nx))

#     s = psf_out.sum()
#     if s > 0:
#         psf_out /= s

#     ref_info = max(contributions, key=lambda t: t[0])[2]

#     if recenter_to_odd_shape:
#         psf_out = _recenter_psf_to_odd_shape(psf_out)

#     return psf_out.astype(np.float32), ref_info


# def shift_psf_phase(
#     psf : np.ndarray,
#     dx : float,
#     dy : float
# ) -> np.ndarray:
#     """
#     Shift the phase of a PSF image.
#     """
#     from scipy.ndimage import fourier_shift
#     f = np.fft.fftn(psf)
#     f_shifted = fourier_shift(f, shift=(dy, dx))
#     axes = tuple(range(psf.ndim))
#     psf_shifted = np.fft.ifftn(f_shifted).real
#     psf_shifted = np.clip(psf_shifted, 0, None)
#     psf_shifted /= psf_shifted.sum()
#     return psf_shifted

def shift_psf_phase(psf : np.ndarray, dx : float = 0.0, dy : float = 0.0, order : int = 1) -> np.ndarray:
    from scipy.ndimage import shift
    shifted = shift(psf, shift=(dy, dx), order=order, mode='nearest')
    shifted = np.clip(shifted, 0, None)
    return shifted

def crop_AO_psf(
    psf : np.ndarray,
    scale : float,
    wave : float,
    colldiam : float,
    n : int = 100,
):
    """
    Crops a AO PSF parametrized by the telescope diameter and wavelength.
    
    In other words, the PSF is cropped to a size of ``n * wavelength / colldiam``,
    where ``wavelength / colldiam`` is the diffraction limit
    of the telescope at the given wavelength.

    The PSF must have an odd number of rows and columns, and the center of the
    PSF is assumed to be at the center of the array.

    Parameters
    ----------
    psf : np.ndarray
        The PSF to crop.
    scale : float
        The size of a PSF pixel in arcsec.
    wave : float
        The wavelength in microns.
    colldiam : float
        The effective collimating diameter in meters.
    n : int, optional
        The number of lambda / D's to crop by.
        Defaults to 100.

    Returns
    -------
    psf_out: np.ndarray:
        The new PSF.
    """

    if psf.shape[0] % 2 != 1 or psf.shape[1] % 2 != 1:
        raise ValueError(f"PSF must have odd number of rows and columns, got {psf.shape}")
    
    ny, nx = psf.shape

    # lambda / D per pixel
    s = 206265 * wave / (colldiam * 1E6) / scale

    cy, cx = psf.shape[0] // 2, psf.shape[1] // 2

    # Compute the crop size
    # Initial bounds
    w = round(n * s)
    yi = cy - w
    yf = cy + w
    xi = cx - w
    xf = cx + w

    # Check bounds
    yi = max(yi, 0)
    yf = min(yf, ny - 1)
    xi = max(xi, 0)
    xf = min(xf, nx - 1)

    # Ensure odd number of rows and columns
    if (yf - yi) % 2 == 0:
        if yf < ny - 1:
            yf += 1
        else:
            yi -= 1
    if (xf - xi) % 2 == 0:
        if xf < nx - 1:
            xf += 1
        else:
            xi -= 1

    # Slice PSF
    psf_out = psf[yi:yf+1, xi:xf+1].copy()

    # Return
    return psf_out


def _recenter_psf_to_odd_shape(psf: np.ndarray) -> np.ndarray:
    ny, nx = psf.shape

    dx = -0.5 if nx % 2 == 0 else 0.0
    dy = -0.5 if ny % 2 == 0 else 0.0

    if dx != 0.0 or dy != 0.0:
        psf = shift_psf_phase(psf, dx=dx, dy=dy)

    ny_new = ny if ny % 2 == 1 else ny - 1
    nx_new = nx if nx % 2 == 1 else nx - 1

    y0 = (ny - ny_new) // 2
    x0 = (nx - nx_new) // 2

    return psf[y0:y0 + ny_new, x0:x0 + nx_new]


def _pad_psf_to_shape(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Zero-pad a PSF to *shape*, keeping it centred."""
    out = np.zeros(shape, dtype=np.float64)
    y_off = (shape[0] - psf.shape[0]) // 2
    x_off = (shape[1] - psf.shape[1]) // 2
    out[y_off:y_off + psf.shape[0], x_off:x_off + psf.shape[1]] = psf
    return out


def extend_psf_powerlaw(
    psf: np.ndarray,
    rmin: float,
    extend_factor: float = 1.5,
    blend_width: float = 10.0,
    alpha_min: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Extend a PSF by fitting a power-law to its wings and padding the array.

    Parameters
    ----------
    psf : ndarray (ny, nx)
        Input PSF, assumed centred.
    rmin : float
        Radius (px) at which the power-law fit begins and the blend is
        centred. Should be in the wing regime, beyond the AO-corrected core.
    extend_factor : float
        Linear scale factor for the output array (must be > 1).
    blend_width : float
        Full radial width (px) of the cosine blend transition centred on
        rmin. The PSF is used purely inside rmin-blend_width/2; the power
        law purely outside rmin+blend_width/2; a smooth cosine ramp joins
        them. Increase if the transition looks abrupt.
    eps : float
        Small floor to avoid log(0).
    """
    psf = np.asarray(psf, dtype=np.float64)
    ny, nx = psf.shape

    if extend_factor <= 1:
        return psf.copy()

    Ny = int(np.ceil(ny * extend_factor))
    Nx = int(np.ceil(nx * extend_factor))
    if Ny % 2 == 0:
        Ny += 1
    if Nx % 2 == 0:
        Nx += 1

    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0

    # --- radial profile ---
    y, x = np.indices(psf.shape)
    r = np.hypot(x - cx, y - cy)
    r_int = r.astype(np.int32)
    r_max = r_int.max()

    sums   = np.bincount(r_int.ravel(), weights=psf.ravel(), minlength=r_max + 1)
    counts = np.bincount(r_int.ravel(), minlength=r_max + 1)
    radial = sums / np.maximum(counts, 1)
    radial = np.maximum(radial, eps)

    # --- fit power-law slope from rmin outward ---
    rmin_int = int(round(rmin))
    # r_fit = np.arange(rmin_int, r_max + 1)
    # r_fit = r_fit[r_fit > 0]

    r_edge = min(cy, cx)  # largest fully-sampled circle inside the array

    r_fit = np.arange(rmin_int, int(r_edge) + 1)
    r_fit = r_fit[r_fit > 0]

    if len(r_fit) < 2:
        raise ValueError(
            f"Too few radial samples to fit: rmin={rmin} (→ {rmin_int}), "
            f"r_max={r_max}. Reduce rmin or use a larger PSF array."
        )

    log_r = np.log(r_fit.astype(float))
    log_I = np.log(radial[r_fit])

    # sigma-clip to exclude bright rings/bumps before fitting
    residuals = log_I - np.polyfit(log_r, log_I, 1)[0] * log_r
    mask = np.abs(residuals - residuals.mean()) < 2.5 * residuals.std()

    coeffs = np.polyfit(log_r[mask], log_I[mask], 1)
    alpha = -coeffs[0]

    # Anchor I0 to the actual PSF at rmin so tail(rmin) == PSF(rmin)
    ring = psf[r_int == rmin_int]
    I0 = np.median(ring) if ring.size > 0 else radial[rmin_int]

    # --- build extended grid ---
    Y, X = np.indices((Ny, Nx))
    Cy = (Ny - 1) / 2.0
    Cx = (Nx - 1) / 2.0
    R = np.hypot(X - Cx, Y - Cy)

    tail = I0 * (np.maximum(R, rmin) / rmin) ** (-alpha)

    # --- paste and blend ---
    out = tail.copy()
    y0 = (Ny - ny) // 2
    x0 = (Nx - nx) // 2

    sub_R = R[y0:y0 + ny, x0:x0 + nx]

    # Cosine blend centred on rmin over blend_width:
    #   w=1 (pure PSF)   for r < rmin - blend_width/2
    #   w=0 (pure tail)  for r > rmin + blend_width/2
    #   smooth cosine ramp in between
    r_lo = rmin - blend_width / 2.0
    r_hi = rmin + blend_width / 2.0

    w = np.ones_like(sub_R)
    ramp = (sub_R >= r_lo) & (sub_R <= r_hi)
    w[ramp] = 0.5 * (1.0 + np.cos(np.pi * (sub_R[ramp] - r_lo) / blend_width))
    w[sub_R > r_hi] = 0.0

    out[y0:y0 + ny, x0:x0 + nx] = w * psf + (1.0 - w) * tail[y0:y0 + ny, x0:x0 + nx]

    # Flux normalisation
    out *= psf.sum() / np.maximum(out.sum(), eps)

    return out