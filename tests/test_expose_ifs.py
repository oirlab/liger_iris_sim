
from liger_iris_sim.sources import make_point_source_ifs_cube
from liger_iris_sim.expose import compute_throughput, expose_ifs
from liger_iris_sim.sky import get_maunakea_sky_background
from liger_iris_sim.utils import LIGER_PROPS, compute_filter_photon_flux, generate_wave_grid_for_filter, get_psf

from liger_iris_drp_resources.filters import load_filters_summary

import numpy as np

def test_expose_ifs():

    np.random.seed(1)

    # Atmosphere, instrument, and exposure params
    instrument_name = 'Liger'
    instrument_mode = 'ifs' # img, ifs
    ifs_mode = 'lenslet'
    filter_name = 'J'
    size = (128, 128)
    read_noise = 9 # e- RMS
    dark_current = 0.025 # e-/sec
    scale = 0.014 # arcsec / pixel
    itime = 1000 # 1000 sec
    n_frames = 1 # 1 coadd
    resolution = 4000
    #tel_tput = 0.91 # Default
    #ao_tput = 0.8 # Default
    #filt_tput = 0.9 # Default
    collarea = LIGER_PROPS['keck_collarea'] # m^2

    # Load filter data
    filter_info = load_filters_summary(filter_name=filter_name)

    # Calculate total throughput
    tput = compute_throughput(
        instrument_name=instrument_name,
        instrument_mode=instrument_mode,
        wave=filter_info['wavecenter'],
        ifs_mode=ifs_mode,
        filt=filter_name,
    )
    print(f"Total throughput for {ifs_mode} mode: {tput:.3f}")

    # Load on axis PSF at this wavelength and bin to match pixel scale
    psf, psf_info = get_psf(
        instrument_name=instrument_name,
        instrument_mode=instrument_mode,
        wave=filter_info['wavecenter'],
        xs=0, ys=0,
        output_plate_scale=scale,
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

    # Compute background sky emission and transmission
    sky_data = get_maunakea_sky_background(
        wave=wave,
        filter_info=filter_info,
        resolution=resolution,
        T_tel=275, T_atm=258, T_aos=243, # Default values
        Em_tel=0.09, Em_atm=0.2, Em_aos=0.01, # Default values
        airmass=1.4, # Typical value
        plate_scale=scale,
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
        sky_emission_rate=sky_data['sky_em'],
        sky_transmission=sky_data['sky_trans'],
        tput=tput, read_noise=read_noise, dark_current=dark_current,
    )

    # NOTE: Keep for debugging
    # import matplotlib
    # matplotlib.use("QTAGG")
    # import matplotlib.pyplot as plt
    # breakpoint()
    # plt.plot(np.sum(sim['observed_rate'][:, 60:69, 60:69], axis=(1, 2))); plt.show()

    snr_peak = np.nanmax(sim['snr'])
    assert 35 < snr_peak < 45, f"Expected SNR between 35 and 45, got {snr_peak:.3f}"