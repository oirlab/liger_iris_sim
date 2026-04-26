Quickstart
==========

Ensure you have followed the :doc:`installation instructions <../installation>` before proceeding.

Ensure that the data resources have been downloaded.

PSFs
----

.. code-block:: python

    from liger_iris_sim.utils import get_psf

    psf, psfinfo = get_psf(
        instrument_name='Liger',
        instrument_mode='IMG',
        wave=1.25,  # microns
        xs=0, ys=0, # arcsec offsets from field center
    )


Source Creation
---------------

.. code-block:: python

    from liger_iris_sim.sources import make_point_source_image
    from liger_iris_sim.utils import compute_filter_photon_flux
    from liger_iris_drp_resources.filters import load_filters_summary
    from liger_iris_sim.utils import get_psf
    import numpy as np

    filt = 'J'
    scale = 0.01  # arcsec / pixel
    size = (512, 512)

    filter_info = load_filters_summary(filter_name=filt)
    psf, psf_info = get_psf(
        instrument_name='Liger',
        instrument_mode='IMG',
        wave=filter_info['wavecenter'],
        xs=0, ys=0
    )

    mag = 15.5
    photon_flux = compute_filter_photon_flux(mag, zp=filter_info['zpphot'])

    source_rate = np.zeros(size, dtype=np.float32)
    xpix, ypix = size[1] // 2, size[0] // 2
    make_point_source_image(
        xpix, ypix, photon_flux,
        psf=psf,
        image_out=source_rate,
    )


Sky Background
--------------

.. code-block:: python

    from liger_iris_sim.sky import get_maunakea_sky_background
    from liger_iris_sim.utils import generate_wave_grid_for_filter
    from liger_iris_drp_resources.filters import load_filters_summary

    filter_info = load_filters_summary(filter_name='J')
    wave = generate_wave_grid_for_filter(filter_info, resolution=4000)

    sky_data = get_maunakea_sky_background(
        wave, resolution=4000,
        T_tel=275, T_atm=258, T_aos=243, T_zod=5800,
        Em_tel=0.09, Em_atm=0.2, Em_aos=0.01,
        airmass=1.4,
        plate_scale=scale,
    )


Throughput
----------

.. code-block:: python

    from liger_iris_sim.expose import compute_throughput
    from liger_iris_drp_resources.filters import load_filters_summary

    # Load filter info for J band
    filter_info = load_filters_summary(filter_name='J')
    wave = filter_info['wavecenter'] # Central wavelength in microns

    # Total Liger Imager throughput
    tput_img = compute_throughput(
        instrument_name='Liger',
        instrument_mode='IMG',
        wave=wave,
        tel=0.8, ao=0.65, filt=0.9, # Default values
    )

    # Total Liger Imager throughput
    tput_ifs = compute_throughput(
        instrument_name='Liger',
        instrument_mode='IMG',
        ifs_mode='LENSLET',
        wave=wave,
        tel=0.8, ao=0.65, filt=0.9, # Default values
    )

    # Instrument-only throughput
    # Total Liger Imager throughput
    tput_inst = compute_throughput(
        instrument_name='Liger',
        instrument_mode='IMG',
        ifs_mode='LENSLET',
        wave=wave,
        instrument_only=True,
    )


Expose
------

.. code-block:: python

    from liger_iris_sim.expose import expose_imager
    from liger_iris_sim.utils import LIGER_PROPS, compute_filter_photon_flux
    from liger_iris_sim.sky import get_maunakea_sky_background
    from liger_iris_drp_resources import load_filters_summary

    filter_info = load_filters_summary(filter_name='J')
    scale = 0.01
    sky_em_rate = get_maunakea_sky_background(
        wave, resolution=4000,
        T_tel=275, T_atm=258, T_aos=243, T_zod=5800,
        Em_tel=0.09, Em_atm=0.2, Em_aos=0.01,
        airmass=1,
        plate_scale=scale,
    )

    source_rate = np.ones((2048, 2048), dtype=np.float32)

    sim = expose_imager(
        source_rate,
        itime=10, n_frames=1,
        collarea=LIGER_PROPS['collarea'],
        sky_emission_rate=sky_em_rate,
        tput=tput_img,
        read_noise=9, dark_current=0.025,
    )

    print(f"Peak SNR: {sim['snr'].max():.1f}")