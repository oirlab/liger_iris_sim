from matplotlib import axes
import numpy as np
from numba import njit

from liger_iris_drp_resources import load_liger_psf, load_iris_psf
from liger_iris_drp_resources.psfs import _get_liger_psf_filename, _get_iris_psf_filename
from .resampling import rebin_image

__all__ = [
    "get_psfs",
    "get_psf",
    "crop_AO_psf",
    "shift_psf_phase",
]

def get_psfs(
    instrument_name : str,
    instrument_mode : str | None = None,
    wave : float | None = None,
    xs : float | None = None, ys : float | None = None,
    xdet : float | None = None, ydet : float | None = None,
    output_plate_scale : float | None = None,
    crop_to_odd_shape : bool = True,
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
    crop_to_odd_shape : bool, optional
        If True, the output PSF will be cropped to have an odd number of rows and columns. Default is True.
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
                    crop_to_odd_shape=crop_to_odd_shape,
                    extend_powerlaw=extend_powerlaw,
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
                    crop_to_odd_shape=crop_to_odd_shape,
                    extend_powerlaw=extend_powerlaw,
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
    #         crop_to_odd_shape=False
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
    crop_to_odd_shape : bool = True,
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
    crop_to_odd_shape : bool, optional
        If True, the output PSF will be cropped to have an odd number of rows and columns. Default is True.
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
            crop_to_odd_shape=False
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
    if crop_to_odd_shape:
        psf = _crop_to_odd_shape(psf)

    return psf, info


def shift_psf_phase(
    psf : np.ndarray,
    dx : float,
    dy : float
) -> np.ndarray:
    """
    Shift the phase of a PSF image.
    """
    from scipy.ndimage import fourier_shift
    f = np.fft.fftn(psf)
    f_shifted = fourier_shift(f, shift=(dy, dx))
    axes = tuple(range(psf.ndim))
    psf_shifted = np.fft.irfftn(f_shifted, s=psf.shape, axes=axes).real
    psf_shifted = np.clip(psf_shifted, 0, None)
    psf_shifted /= psf_shifted.sum()
    return psf_shifted


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


def _crop_to_odd_shape(psf: np.ndarray) -> np.ndarray:
    ny, nx = psf.shape
    return psf[:ny - ((ny + 1) % 2), :nx - ((nx + 1) % 2)]


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
    alpha_min : float
        Minimum power-law exponent (I ~ r^{-alpha}). AO halos are typically
        r^{-2} to r^{-3}, so 2.0 is a safe lower bound.
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