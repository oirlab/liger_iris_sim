
from liger_iris_sim.sources import make_point_source_image
from liger_iris_sim.expose import compute_liger_throughput, expose_imager
from liger_iris_sim.utils import LIGER_PROPS, rebin_image, compute_filter_photon_flux

from liger_iris_drp_resources.filters import load_filters_summary
from liger_iris_drp_resources.psfs import get_liger_psf, download_liger_psfs
from liger_iris_drp_resources.model_spectra import download_model_spectra

import numpy as np

def test_expose_imager():

    np.random.seed(1)

    download_liger_psfs()
    download_model_spectra()

    # Atmosphere, instrument, and exposure params
    mode = 'img' # img, lenslet, slicer
    filt = 'J'
    size = (512, 512) # Smaller for testing
    read_noise = 9 # e- RMS
    dark_current = 0.025 # e-/sec
    scale = 0.01 # arcsec / pixel
    itime = 10 # 10 sec
    n_frames = 1 # 1 coadd
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
    )
    print(f"Total throughput for {mode} mode: {tput:.3f}")

    # Load on axis PSF at this wavelength and bin to match pixel scale
    psf, psf_info = get_liger_psf(filter_info['wavecenter'], xs=0, ys=0)
    psf = rebin_image(
        psf,
        scale_in=psf_info['psf_sampling'],
        scale_out=scale
    )

    # Point source params
    xpix = size[1] // 2
    ypix = size[0] // 2
    mag = 15.5
    photon_flux = compute_filter_photon_flux(mag, zp=filter_info['zpphot'])
    
    source_rate = np.zeros(size, dtype=np.float32)
    make_point_source_image(
        xpix, ypix, photon_flux,
        psf=psf,
        image_out=source_rate,
    )

    # Total sky
    # photons / sec / arcsec^2 / m^2
    sky_em_rate = compute_filter_photon_flux(filter_info['backmag'], zp=filter_info['zpphot'])
    # Integrate over 2D pixel (photons / sec / m^2)
    sky_em_rate *= scale**2

    # Expose
    sim = expose_imager(
        source_rate,
        itime=itime, n_frames=n_frames, collarea=collarea,
        sky_emission_rate=sky_em_rate,
        tput=tput, read_noise=read_noise, dark_current=dark_current,
    )
    
    snr_peak = np.nanmax(sim['snr'])
    assert 130 < snr_peak < 150, f"Expected SNR between 130 and 150, got {snr_peak:.3f}"