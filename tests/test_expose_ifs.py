
from liger_iris_sim.sources import make_point_source_ifs_cube
from liger_iris_sim.expose import compute_liger_throughput, expose_ifs
from liger_iris_sim.sources.ifs import make_point_source_ifs_cube
from liger_iris_sim.sky import get_maunakea_spectral_sky_emission, get_maunakea_spectral_sky_transmission
from liger_iris_sim.utils import LIGER_PROPS, rebin_image, compute_filter_photon_flux, generate_wave_grid_for_filter

from liger_iris_drp_resources.filters import load_filters_summary
from liger_iris_drp_resources.psfs import get_liger_psf, download_liger_psfs
from liger_iris_drp_resources.model_spectra import download_model_spectra

import numpy as np

def test_expose_ifs():

    np.random.seed(1)

    download_liger_psfs()
    download_model_spectra()

    # Atmosphere, instrument, and exposure params
    mode = 'ifs' # img, ifs
    ifs_mode = 'lenslet'
    filt = 'J'
    size = (128, 128)
    read_noise = 9 # e- RMS
    dark_current = 0.025 # e-/sec
    scale = 0.014 # arcsec / pixel
    itime = 1000 # 10 sec
    n_frames = 1 # 1 coadd
    resolution = 4000
    #tel_tput = 0.91 # Default
    #ao_tput = 0.8 # Default
    #filt_tput = 0.9 # Default
    collarea = LIGER_PROPS['keck_collarea'] # m^2

    # Load filter data
    filter_info = load_filters_summary(filter_name=filt)

    # Calculate total throughput
    tput = compute_liger_throughput(
        mode=mode,
        wave=filter_info['wavecenter'],
        ifs_mode=ifs_mode,
    )
    print(f"Total throughput for {ifs_mode} mode: {tput:.3f}")

    # Load on axis PSF at this wavelength and bin to match pixel scale
    psf, psf_info = get_liger_psf(filter_info['wavecenter'], xs=0, ys=0)
    psf = rebin_image(
        psf,
        scale_in=psf_info['psf_sampling'],
        scale_out=scale
    )

    # NQ sampled wave grid for this filter
    wave = generate_wave_grid_for_filter(filter_info, resolution=resolution)

    # Point source params
    xpix = size[1] // 2
    ypix = size[0] // 2
    mag = 15.5
    photon_flux = compute_filter_photon_flux(mag, zp=filter_info['zpphot'])

    # Template of each star is flat
    base_template = np.ones(wave.size, dtype=np.float32)

    # Sky emission
    sky_em = get_maunakea_spectral_sky_emission(
        wave,
        resolution=resolution,
        T_tel=275, T_atm=258, T_aos=243, T_zod=5800,
        Em_tel=0.09, Em_atm=0.2, Em_aos=0.01,
    )
    
    # Integrate over sky pixel
    sky_em_rate = sky_em['sky_em'] * scale**2 # Integrate over pixel: photons / (s * m^2 * wavebin)

    # Sky transmission
    sky_trans = get_maunakea_spectral_sky_transmission(
        wave,
        resolution=resolution,
        airmass=1
    )

    # Input cube of one star
    input_cube_rate = np.zeros((len(wave), *size), dtype=np.float32)
    template_spec = base_template / np.sum(base_template) * photon_flux # photons / sec / m^2
    template = (wave, template_spec)
    make_point_source_ifs_cube(
        xpix, ypix, template,
        psf=psf,
        cube_out=input_cube_rate,
    )

    # Expose
    sim = expose_ifs(
        input_cube_rate,
        itime=itime, n_frames=n_frames, collarea=collarea,
        sky_emission_rate=sky_em_rate,
        sky_transmission=sky_trans['sky_trans'],
        tput=tput, read_noise=read_noise, dark_current=dark_current,
    )

    snr_peak = np.nanmax(sim['snr'])
    assert 35 < snr_peak < 45, f"Expected SNR between 35 and 45, got {snr_peak:.3f}"