Overview
========

Ensure you have followed the :doc:`installation instructions <../installation>` before proceeding with this quickstart guide.

If any example below requires data from the `Link to Google Drive folder <https://drive.google.com/drive/folders/15vSPi9QRine2F2zhZ7xeSXJF7fMXZdoe?usp=drive_link>`_, it will automatically download the necessary files to the local resources directory.

Examples below assume the user has set the environment variable **LIGER_IRIS_DRP_RESOURCE_DIR** to a valid directory path, or is using the default resources directory.


Source Creation
---------------

Use :py:func:`~liger_iris_sim.sources.imager.make_point_source_image` to create an imager source image, or
:py:func:`~liger_iris_sim.sources.ifs.make_point_source_ifs_cube` to create a source cube for IFS simulations.
Both functions place a point source at the given detector pixel coordinates, convolved with a PSF,
and return a rate image or cube in units of photons / sec / m².

.. code-block:: python

    from liger_iris_sim.sources import make_point_source_image
    from liger_iris_sim.utils import compute_filter_photon_flux
    from liger_iris_drp_resources.filters import load_filters_summary
    from liger_iris_drp_resources.psfs import get_liger_psf, download_liger_psfs
    from liger_iris_sim.utils import rebin_image
    import numpy as np

    download_liger_psfs()

    filt = 'J'
    scale = 0.01  # arcsec / pixel
    size = (512, 512)

    filter_info = load_filters_summary(filter_name=filt)
    psf, psf_info = get_liger_psf(filter_info['wavecenter'], xs=0, ys=0)
    psf = rebin_image(psf, scale=psf_info['psf_sampling'] / scale)

    mag = 15.5
    photon_flux = compute_filter_photon_flux(mag, zp=filter_info['zpphot'])

    source_rate = np.zeros(size, dtype=np.float32)
    make_point_source_image(
        size[1] // 2, size[0] // 2, photon_flux,
        psf=psf,
        image_out=source_rate,
    )


Sky Background
--------------

Use :py:func:`~liger_iris_sim.sky.background_sky.get_maunakea_spectral_sky_emission` and
:py:func:`~liger_iris_sim.sky.background_sky.get_maunakea_spectral_sky_transmission` to obtain
the Maunakea sky emission and transmission spectra sampled on a user-supplied wavelength grid.
For broadband imager simulations, a single per-pixel sky rate can be derived from the filter
background magnitude using :py:func:`~liger_iris_sim.utils.filter_utils.compute_filter_photon_flux`.

.. code-block:: python

    from liger_iris_sim.sky import get_maunakea_spectral_sky_emission, get_maunakea_spectral_sky_transmission
    from liger_iris_sim.utils import generate_wave_grid_for_filter
    from liger_iris_drp_resources.filters import load_filters_summary
    from liger_iris_drp_resources.model_spectra import download_model_spectra

    download_model_spectra()

    filter_info = load_filters_summary(filter_name='J')
    wave = generate_wave_grid_for_filter(filter_info, resolution=4000)

    sky_em = get_maunakea_spectral_sky_emission(
        wave, resolution=4000,
        T_tel=275, T_atm=258, T_aos=243, T_zod=5800,
        Em_tel=0.09, Em_atm=0.2, Em_aos=0.01,
    )

    sky_trans = get_maunakea_spectral_sky_transmission(wave, resolution=4000, airmass=1)


Throughput
----------

Use :py:func:`~liger_iris_sim.expose.throughput.compute_liger_throughput` to compute the total system throughput
(telescope * AO * filter * instrument) at a given wavelength for either the imager or IFS mode.

.. code-block:: python

    from liger_iris_sim.expose import compute_liger_throughput
    from liger_iris_drp_resources.filters import load_filters_summary

    filter_info = load_filters_summary(filter_name='J')

    # Imager throughput
    tput_img = compute_liger_throughput(mode='img', wave=filter_info['wavecenter'])

    # IFS lenslet throughput
    tput_ifs = compute_liger_throughput(
        mode='ifs',
        wave=filter_info['wavecenter'],
        ifs_mode='lenslet',
    )


Expose
------

Use :py:func:`~liger_iris_sim.expose.imager.expose_imager` or :py:func:`~liger_iris_sim.expose.ifs.expose_ifs` to
simulate a detector exposure from a source rate image or cube. Both functions return a dict containing
noise-free and noise-added images/cubes, individual noise components, and per-pixel SNR.

.. code-block:: python

    from liger_iris_sim.expose import expose_imager
    from liger_iris_sim.utils import LIGER_PROPS, compute_filter_photon_flux
    from liger_iris_drp_resources.filters import load_filters_summary

    filter_info = load_filters_summary(filter_name='J')
    scale = 0.01        # arcsec / pixel
    sky_em_rate = compute_filter_photon_flux(filter_info['backmag'], zp=filter_info['zpphot']) * scale**2

    sim = expose_imager(
        source_rate,                          # 2-D array from make_point_source_image
        itime=10, n_frames=1,
        collarea=LIGER_PROPS['keck_collarea'],
        sky_emission_rate=sky_em_rate,
        tput=tput_img,
        read_noise=9, dark_current=0.025,
    )

    print(f"Peak SNR: {sim['snr'].max():.1f}")