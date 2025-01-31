import numpy as np

# SNR
def expose_ifu(
        source_cube : np.ndarray,
        itime : float, n_frames : int,
        collarea : float,
        sky_emission : np.ndarray,
        sky_transmission : np.ndarray,
        efftot : float,
        read_noise : float, dark_current : float, num_detector_pixels : float = 1.0
    ) -> dict:
    """
    Args:
        source_cube (np.ndarray): Source cube (y, x, wave). Units are photons / sec / m^2 / wavebin.
        itime (float): Integration time (sec).
        collarea (np.ndarray): Collimating area (m^2)
        sky_emission (np.ndarray): The sky background emission spectrum sampled on wavebins
            in units of photons / sec / m^2 / wavebin. Sky emission is NOT modulated by sky_transmission.
        sky_transmission (np.ndarray): The sky background transmission for each spectrum
            normalized to [0, 1] for each wavebin. Only affects the source spectrum.
        efftot (np.ndarray): Total efficiciency (convert photons -> e-).
        num_detector_pixels (float): The average number of detector pixels that correspond to a 3D IFU pixel.
    
    Returns:
        dict: The exposure components. Keys are:
            observed (np.ndarray): Combined exposure (e- / s / wavebin)
            source (np.ndarray): Source signal (e- / s / wavebin)
            background (np.ndarray): Background signal (e- / s / wavebin)
            snr (np.ndarray): SNR
            noise (np.ndarray): Noise (e- / s / wavebin)
            read_noise (np.ndarray): Effective read noise (e- RMS wavebin)
    """

    # Multiply by tellurics (photons / sec / m^2 / wavebin)
    source_rate = source_cube * sky_transmission

    # Integrate over telescope aperture (photons / sec / wavebin)
    source_rate *= collarea
    sky_emission_rate = sky_emission * collarea

    # Efficiency (effectively converts to e- / s / wavebin)
    source_rate *= efftot
    sky_emission_rate *= efftot

    # Dark rate (e- / s / wavebin)
    dark_rate = dark_current * num_detector_pixels

    # Dark tot (e- / wavebin)
    dark_tot = dark_rate * itime * n_frames

    # Sky emission tot (e- / wavebin)
    sky_emission_tot = sky_emission_rate * itime * n_frames

    # Total background (e- / wavebin)
    background_tot = sky_emission_tot + dark_tot

    # Source tot (e- / wavebin)
    source_tot = source_rate * itime * n_frames

    # Total read noise noise contribution over all frames, just make 2D (e-)
    read_noise_tot = np.random.normal(
        loc=0,
        scale=read_noise * np.sqrt(n_frames) * np.sqrt(num_detector_pixels),
        size=source_cube[:, :, 0].shape
    )

    # Final simulated image over all frames (e-)
    sim_tot = source_tot + background_tot

    # Simulated noise
    noise_tot = np.sqrt(sim_tot + (read_noise * np.sqrt(n_frames))**2)

    # Add poisson noise to final image (e-)
    sim_tot_noisy = np.random.poisson(lam=sim_tot, size=source_cube.shape).astype(float)

    # Add read noise to final image (broadcast to each wavelength slice)
    sim_tot_noisy = sim_tot_noisy + read_noise_tot[:, :, None]

    # SNR
    snr = source_tot / noise_tot

    # Outputs in e-
    out = dict(
        observed=sim_tot_noisy,
        source=source_tot,
        sky_emission=sky_emission_tot,
        background=background_tot,
        sky_transmission=sky_transmission,
        snr=snr, noise=noise_tot,
        read_noise=read_noise_tot,
        dark=dark_tot,
    )
      
    # Return
    return out