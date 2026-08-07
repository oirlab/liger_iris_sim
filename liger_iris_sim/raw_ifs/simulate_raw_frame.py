import numpy as np
from astropy.io import fits

from .trace_geometry import get_trace_geometry
from .integration import integrate_spectrum
from .render_trace import render_trace_for_lenslet
from .micropupil_psf import get_effective_psf

from ..utils import LIGER_PROPS
DETECTOR_SHAPE = LIGER_PROPS['ifs_detector_size']

from liger_iris_drp_resources import (
    load_filters_summary,
    load_ifs_array_mask,
    load_ifs_trace_geometry,
)

__all__ = ["simulate_raw_ifs_frame"]

def _simulate_lenslet_raw_frame(
    input_cube : np.ndarray,
    input_wave : np.ndarray,
    arr_mask : np.ndarray,
    x_pix : np.ndarray,
    y_pix : np.ndarray,
    wave_pts : np.ndarray,
    epsf : np.ndarray,
    window_size : int = 2,
    tracepos_deg : int = 1,
    wavesol_deg : int = 1,
    density : bool = False,
    pad_ends : bool = True,
) -> np.ndarray:
    """
    Render a raw 2D IFS image from a cube of lenslet spectra.
    This function is not intended to be called by the user directly.
    Use `simulate_lenslet_raw_frame` instead.

    Parameters
    ----------
    input_cube : ndarray
        Input IFS cube of shape (n_wave, n_lenslets_y, n_lenslets_x).
        Can be in photons/sec or photons/sec/micron if density=True.
    input_wave : ndarray
        Common wavelength grid for `input_cube`, microns.
    arr_mask : ndarray
        Helper array mapping the 2D lenslet indices to a 1D index for the trace arrays x_pix and y_pix.
    x_pix, y_pix : ndarray
        Detector pixel positions of the trace points.
        x_pix are detector columns, y_pix are detector rows.
        Shape=(n_lenslets, 5)
    wave_pts : ndarray
        Wavelengths corresponding to those 5 points.
    epsf : ndarray
        Effective PSF of the micropupil (oversampled).
    window_size : int
        Half-width of the box around the trace to render, in pixels.
    tracepos_deg : int
        Degree of polynomial to fit the trace position y(x).
    wavesol_deg : int
        Degree of polynomial to fit the wavelength solution lambda(x).
    density : bool
        If True, the input_cube is in photons/sec/micron.
        If False, the input_cube is in photons/sec.
    pad_ends : bool
        If True, extend the trace beyond the Zemax points by half a channel on each end.

    Returns
    -------
    image_out : ndarray
        Rendered raw 2D IFS image of shape (n_pix_y, n_pix_x).
    """

    n_lens_y, n_lens_x = arr_mask.shape
    image_out = np.zeros(DETECTOR_SHAPE, dtype=np.float32)
    n_lenslets = len(x_pix)

    for ly in range(n_lens_y):
        for lx in range(n_lens_x):

            # Spectrum for this lenslet, shape=(n_wave,)
            input_spec = input_cube[:, ly, lx].astype(np.float32)
            if not input_spec.any():
                continue

            # 1D index in x_pts and y_pts for this lenslet.
            index_1d = int(arr_mask[ly, lx])
            if index_1d < 0 or index_1d >= n_lenslets:
                continue

            # Trace position sample points for this lenslet.
            # x = cols (axis=1), y = rows (axis=0)
            x_pix_pts = x_pix[index_1d]
            y_pix_pts = y_pix[index_1d]

            # Get the trace geometry for this lenslet.
            x_lo, x_hi, y_of_x, wave_of_x = get_trace_geometry(
                x_pix_pts=x_pix_pts,
                y_pix_pts=y_pix_pts,
                wave_pts=wave_pts,
                wave=input_wave,
                tracepos_deg=tracepos_deg,
                wavesol_deg=wavesol_deg,
                density=density,
                pad_ends=pad_ends
            )

            # Pixel columns to render
            pix_lo = int(np.ceil(x_lo - 0.5))
            pix_hi = int(np.floor(x_hi + 0.5))
            x_pixels = np.arange(pix_lo, pix_hi + 1, dtype=np.float32)

            # Corresponding y positions for each column
            y_pixel_centers = y_of_x(x_pixels)

            # Integrate the spectrum over the wavelength range of each column
            col_edges = wave_of_x(np.append(x_pixels - 0.5, x_pixels[-1] + 0.5))
            flux = integrate_spectrum(
                input_wave,
                input_spec,
                col_edges,
                density=density
            )

            # Render the lenslet.
            render_trace_for_lenslet(
                image_out,
                px_lo=pix_lo,
                y_pix=y_pixel_centers,
                flux=flux,
                epsf=epsf,
                window_size=window_size
            )

        if n_lens_y > 1:
            print(f"\r  Rendered row {ly + 1}/{n_lens_y}", end="", flush=True)

    return image_out

# Main function user calls to render a raw 2D IFS image from a cube of lenslet spectra
def simulate_raw_ifs_frame(
    input_cube : np.ndarray,
    input_wave : np.ndarray,
    ifs_mode : str,
    filter_name : str,
    resolution : str,
    window_size : int = 2,
    tracepos_deg : int = 1,
    wavesol_deg : int = 1,
    density : bool = False,
    pad_ends : bool = True,
    itime : float = None,
    n_frames : int = 1,
    poisson : bool = True,
    dark_current : float = 0.002,
    read_noise : float = 5.0,
    output_path : str = None,
) -> dict:
    """
    Main function user calls to render a raw 2D IFS image from a cube of lenslet spectra.

    Parameters
    ----------
    input_cube : ndarray
        Input IFS cube of shape (n_wave, n_lenslets_y, n_lenslets_x).
        Can be in photons/sec or photons/sec/micron if density=True.
    input_wave : ndarray
        Common wavelength grid for `input_cube`, microns.
    filter_name : str
        Name of the filter used.
    resolution : str
        Resolution of the filter used.
    itime : float, optional
        Integration time in seconds. If None, no noise is added.
    n_frames : int, optional
        Number of frames to simulate. Default is 1.
    include_poisson : bool, optional
        Whether to include Poisson noise. Default is True.
    """

    # Load the filter info
    filter_info = load_filters_summary(filter_name)

    # Load the array mask
    arr_mask = load_ifs_array_mask()

    # Load the trace geometry files
    x_pix, y_pix = load_ifs_trace_geometry(
        ifs_mode=ifs_mode,
        filter_name=filter_name,
        resolution=resolution,
    )

    # Pre compute the effective PSF (psf convolved by detector pixel response = tophat)
    epsf = get_effective_psf(filter_name)

    # Define the wavelength sample points for the trace geometry.
    wave_pts = np.linspace(filter_info["wavemin"], filter_info["wavemax"], 5)

    # Simulate the raw frame (no noise)
    sim = _simulate_lenslet_raw_frame(
        input_cube=input_cube,
        input_wave=input_wave,
        arr_mask=arr_mask,
        x_pix=x_pix,
        y_pix=y_pix,
        wave_pts=wave_pts,
        epsf=epsf,
        window_size=window_size,
        tracepos_deg=tracepos_deg,
        wavesol_deg=wavesol_deg,
        density=density,
        pad_ends=pad_ends,
    )

    # Add noise
    if itime is not None and itime > 0:
        data_noise, error = add_noise(
            sim,
            itime=itime,
            n_frames=n_frames,
            poisson=poisson,
            dark_current=dark_current,
            read_noise=read_noise
        )
    else:
        data = np.copy(sim)
        error = np.zeros_like(data)

    out = {"sim": sim, "data": data, "error": error, "filepath": output_path}

    if output_path is not None:
        save_raw_frame_to_fits(out, output_path)

    return out


def save_raw_frame_to_fits(out, output_path : str):
    """
    Save the raw frame and noise to a FITS file.

    Parameters
    ----------
    out : dict
        Result returned from `simulate_raw_ifs_frame`.
        Output FITS file has extensions:
        - "PRIMARY" for the primary header (no data)
        - "DATA"
        - "ERR" for the simulated data and noise, respectively.
    output_path : str
        Path to save the FITS file.
    """
    # TODO: Add metadata
    if output_path is not None:
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(out['data'], name="DATA"),
        ])
        if out['data_noise'] is not None:
            hdul.append(fits.ImageHDU(out['data_noise'], name="ERR"))
        hdul.writeto(output_path, overwrite=True)


def add_noise(
    input_image_rate : np.ndarray,
    itime : float,
    read_noise : float,
    dark_current : float,
    n_frames : int = 1,
    poisson : bool = True,
) -> np.ndarray:
    """
    Add noise sources to the 2D image.

    Parameters
    ----------
    input_image_rate : ndarray
        The input image to which noise will be added.
    itime : float
        Integration time in seconds.
    n_frames : int, optional
        Number of frames to simulate. Default is 1.
    read_noise : float, optional
        Read noise in electrons. Default is 10.
    dark_current : float, optional
        Dark current in electrons/second. Default is 0.002.
    poisson : bool, optional
        Whether to include Poisson noise. Default is True.

    Returns
    -------
    noisy : ndarray
        The image with added noise.
    error : ndarray
        The error associated with the noisy image.
    """

    var = np.zeros_like(input_image_rate, dtype=np.float32)

    # Total counts including dark current
    total = (input_image_rate + dark_current) * itime * n_frames

    # Add Poisson noise
    if poisson:
        image_noise = np.random.poisson(np.clip(total, 0.0, None)).astype(np.float32)
        var_tot = np.abs(image_noise)
    else:
        image_noise = total.copy()
        var_tot = np.zeros_like(image_noise, dtype=np.float32)

    # Add read noise
    if read_noise:
        image_noise += np.random.normal(0.0, read_noise * np.sqrt(n_frames), total.shape).astype(np.float32)
        var_tot += (read_noise**2) * n_frames

    # Rescale back to rate units
    image_noise /= (itime * n_frames)
    var_tot /= (itime * n_frames)**2
    err = np.sqrt(var_tot).astype(np.float32)

    return image_noise, err
