import numpy as np
import scipy.constants as constants
from astropy.convolution import convolve_fft
from astropy.io import fits
import matplotlib.pyplot as plt

from liger_iris_sim.utils import rebin_image_scipy, rebin_spectrum
from liger_iris_sim.sky import get_maunakea_sky_background
from liger_iris_sim.expose import expose_ifs, compute_throughput
from liger_iris_sim.utils import generate_wave_grid_for_filter, LIGER_PROPS, get_psf
from liger_iris_sim.sources import convolve_spectrum

from liger_iris_drp_resources.filters import load_filters_summary

gal_input_scale = 0.025   # arcsec / pixel
line_center_obs = 1.3     # μm — average observed H-alpha wavelength at z~1

with fits.open('id11978_Halpha_cube_wl.fits') as f:
    gal_wave = f[0].data.astype(np.float64)  # μm

with fits.open('id11978_Halpha_cube.fits') as f:
    # units: erg / (s * cm^2 * pixel), plate scale 0.025 arcsec/pixel
    Ephot = (constants.h * constants.c / (line_center_obs * 1e-6)) * 1e7  # erg / photon
    gal_cube = f[0].data.astype(np.float32)           # erg / (s * cm^2 * pixel)
    gal_cube = np.permute_dims(gal_cube, (2, 0, 1))   # → (wave, y, x)
    gal_cube = gal_cube / Ephot * 100**2               # photons / (s * m^2 * pixel)
    gal_cube = gal_cube[:, 25:-25, 25:-25]             # crop border
    gal_cube = np.clip(gal_cube, 0, np.inf)

nw_in, ny_in, nx_in = gal_cube.shape
print(f"Input cube shape  : ({nw_in}, {ny_in}, {nx_in})  [wave, y, x]")
print(f"Input wave range  : {gal_wave[0]:.4f} – {gal_wave[-1]:.4f} μm")


instrument_name = 'Liger'
instrument_mode = 'ifs'
ifs_mode = 'slicer'   # 'lenslet' or 'slicer'
filt = 'JN3'
scale = 0.075      # arcsec / spaxel
read_noise = 9      # e- RMS
dark_current = 0.025  # e- / sec / pixel
itime = 900.0  # sec (15 min per frame)
n_frames = 16      # total = 4 hr
resolution = 4_000  # R = λ / Δλ
collarea = LIGER_PROPS['keck_collarea']  # m^2

# Output FOV matches the input galaxy FOV
# In practice, tHis requires dithers
size = (
    int(gal_input_scale / scale * ny_in),
    int(gal_input_scale / scale * nx_in),
)
print(f"Output IFS size: {size[0]} × {size[1]} spaxels")

filter_info = load_filters_summary(filter_name=filt)

wave = generate_wave_grid_for_filter(filter_info, resolution=resolution)
print(f"Wavelength grid: {wave[0]:.4f} – {wave[-1]:.4f} μm, {len(wave)} channels")


tput = compute_throughput(
    instrument_name=instrument_name,
    instrument_mode=instrument_mode,
    ifs_mode=ifs_mode,
    wave=line_center_obs,
    tel=0.91, ao=0.8, filt=0.9,
)
print(f"Total throughput at {line_center_obs} μm ({ifs_mode}): {tput:.3f}")

psf, psf_info = get_psf(
    instrument_name=instrument_name,
    instrument_mode=instrument_mode,
    wave=filter_info['wavecenter'],
    xs=0, ys=0,
    output_plate_scale=scale,
)
print(f"PSF shape after rebinning: {psf.shape}")



sky_data = get_maunakea_sky_background(
    wave, resolution=resolution,
    T_tel=275, T_atm=258, T_aos=243,
    Em_tel=0.09, Em_atm=0.2, Em_aos=0.01,
    plate_scale=scale,
)

sky_em_rate = sky_data['sky_em']
sky_trans = sky_data['sky_trans']


def resample_galaxy_cube(
    gal_wave, gal_cube, gal_scale,
    wave_out, scale_out, size_out,
    resolution, psf,
):
    """
    Resample a galaxy IFS cube onto the instrument wavelength and spatial grid.

    Parameters
    ----------
    gal_wave : ndarray
        Input wavelength array (μm).
    gal_cube : ndarray, shape (nw, ny, nx)
        Input cube in photons / (s·m²·pixel).
    gal_scale : float
        Input plate scale in arcsec/pixel.
    wave_out : ndarray
        Output wavelength grid (μm).
    scale_out : float
        Output spaxel scale in arcsec/spaxel.
    size_out : tuple (ny_out, nx_out)
        Output spatial dimensions in spaxels.
    resolution : float or None
        Target spectral resolution R. Pass None to skip spectral convolution.
    psf : ndarray or None
        PSF kernel for spatial convolution.

    Returns
    -------
    gal_cube_out : ndarray, shape (nw_out, ny_out, nx_out)
        Resampled cube in photons / (s·m²·spaxel).
    """
    nw_in, ny_in, nx_in = gal_cube.shape
    ny_out, nx_out = size_out
    nw_out = len(wave_out)

    # --- spectral resampling ---
    gal_cube_specresampled = np.zeros((nw_out, ny_in, nx_in), dtype=float)
    for i in range(ny_in):
        print(f"Spectral resampling row {i + 1}/{ny_in}", end='\r')
        for j in range(nx_in):
            spec = gal_cube[:, i, j]
            if resolution is not None:
                spec = convolve_spectrum(gal_wave, spec, resolution=resolution)
            gal_cube_specresampled[:, i, j] = rebin_spectrum(gal_wave, spec, wave_out)

    # --- spatial resampling + PSF convolution ---
    gal_cube_out = np.zeros((nw_out, ny_out, nx_out), dtype=float)
    for i in range(nw_out):
        print(f"Spatially resampling and convolving slice {i + 1}/{nw_out}", end='\r')
        image = rebin_image_scipy(
            gal_cube_specresampled[i],
            scale_in=gal_scale,
            scale_out=scale_out,
        )
        if psf is not None:
            image = convolve_fft(image, psf, normalize_kernel=True)
        gal_cube_out[i] = image
    print()

    return np.clip(gal_cube_out, 0, np.inf)


source_rate = resample_galaxy_cube(
    gal_wave, gal_cube, gal_input_scale,
    wave_out=wave, scale_out=scale,
    size_out=size, resolution=resolution,
    psf=psf,
)
print(f"Resampled source cube shape: {source_rate.shape}  [wave, y, x]")


sim = expose_ifs(
    source_rate,
    itime=itime, n_frames=n_frames, collarea=collarea,
    sky_emission_rate=sky_em_rate,
    sky_transmission=sky_trans,
    tput=tput, read_noise=read_noise, dark_current=dark_current,
)

print(f"Output cube shape : {sim['observed_tot'].shape}")
print(f"Peak SNR (per voxel): {np.nanmax(sim['snr']):.1f}")


snr_map = np.nansum(sim['snr'] ** 2, axis=0) ** 0.5
extent = [0, size[1] * scale, 0, size[0] * scale]  # arcsec

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

im0 = axes[0].imshow(
    np.nansum(sim['source_rate'], axis=0),
    origin='lower', extent=extent, cmap='inferno',
)
plt.colorbar(im0, ax=axes[0], label='Source rate (e-/s)')
axes[0].set_title('Source (noiseless, collapsed)')
axes[0].set_xlabel('x (arcsec)')
axes[0].set_ylabel('y (arcsec)')

im1 = axes[1].imshow(snr_map, origin='lower', extent=extent, cmap='viridis', vmin=0)
plt.colorbar(im1, ax=axes[1], label='SNR')
axes[1].set_title('SNR integrated over bandpass')
axes[1].set_xlabel('x (arcsec)')

plt.tight_layout()
plt.show()