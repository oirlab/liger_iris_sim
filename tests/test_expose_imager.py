
from liger_iris_sim.sources import make_point_source_image
from liger_iris_sim.expose import compute_throughput, expose_imager
from liger_iris_sim.utils import LIGER_PROPS, compute_filter_photon_flux

from liger_iris_sim.sky import get_maunakea_sky_background
from liger_iris_drp_resources.filters import load_filters_summary
from liger_iris_sim.utils import get_psf

import numpy as np

def test_expose_imager():

    np.random.seed(1)

    # Atmosphere, instrument, and exposure params
    instrument_name = 'Liger'
    instrument_mode = 'IMG' # img, lenslet, slicer
    filter_name = 'J'
    size = (512, 512) # Smaller for testing
    read_noise = 9 # e- RMS
    dark_current = 0.025 # e-/sec
    scale = 0.01 # arcsec / pixel
    itime = 10 # 10 sec
    n_frames = 1 # 1 coadd
    #tel_tput = 0.91 # Default
    #ao_tput = 0.8 # Default
    #filt_tput = 0.9 # Default
    collarea = LIGER_PROPS['collarea'] # m^2

    # Load filter data
    filter_info = load_filters_summary(filter_name=filter_name)
    wavecenter = filter_info['wavecenter']

    # Calculate total throughput
    tput = compute_throughput(
        instrument_name=instrument_name,
        instrument_mode=instrument_mode,
        wave=wavecenter,
        filt=filter_name,
    )
    print(f"Total throughput {tput:.3f}")

    # Load on axis PSF at this wavelength and bin to match pixel scale
    psf, psf_info = get_psf(
        instrument_name=instrument_name,
        instrument_mode=instrument_mode,
        wave=wavecenter,
        xs=0, ys=0,
        output_plate_scale=scale,
    )

    # Point source params
    xpix = size[1] // 2
    ypix = size[0] // 2
    mag = 15.5 # Vega mag
    photon_flux = compute_filter_photon_flux(mag, zp=filter_info['zpphot'])
    
    source_rate = np.zeros(size, dtype=np.float32)
    make_point_source_image(
        xpix, ypix, photon_flux,
        psf=psf,
        image_out=source_rate,
    )

    # Compute background sky emission and transmission
    sky_data = get_maunakea_sky_background(
        resolution=10_000,
        filter_info=filter_info,
        T_tel=275, T_atm=258, T_aos=243, # Default values
        Em_tel=0.09, Em_atm=0.2, Em_aos=0.01, # Default values
        airmass=1.4, # Typical value
        plate_scale=scale,
    )

    # Total background for imager
    sky_emission_rate = sky_data['sky_em_rate_bandpass_tot']

    # Expose
    sim = expose_imager(
        source_rate,
        itime=itime, n_frames=n_frames, collarea=collarea,
        sky_emission_rate=sky_emission_rate,
        tput=tput, read_noise=read_noise, dark_current=dark_current,
    )
    
    snr_peak = np.nanmax(sim['snr'])
    assert 140 < snr_peak < 150, f"Expected SNR between 140 and 150, got {snr_peak:.3f}"