import numpy as np
import matplotlib.pyplot as plt

def expose_imager(
        source_image : np.ndarray,
        itime : float, n_frames : int,
        collarea : float, sky_emission_rate : float, efftot : float,
        read_noise : float, dark_current : float, num_detector_pixels : float = 1.0
    ) -> dict:
    """
    Parameters:
    source_image (np.ndarray): The source image with the correct shape and scale as the final image, in units of photons / sec / m^2.
    itime (float): The exposure time in seconds.
    n_frames (float): The total number of frames to coadd, each inducing a read noise.
    collarea (float): The telescope collimating area in units of m^2.
    sky_emission_rate (float): The background sky emission rate in units of photons / sec / m^2 / pixel.
    efftot (float): The total efficiency of the system (top of atmosphere -> detector).
    read_noise (float): The read noise in units of e- RMS.
    dark_current (float): The dark current rate in units of ADU / sec / pixel.
    num_detector_pixels (float): The average number of detector pixels that correspond to a 2D imager pixel.

    Returns:
    dict: The exposure components. Keys are:
        observed (np.ndarray): Combined exposure (e- / s)
        source (np.ndarray): Source signal (e- / s)
        background (np.ndarray): Background signal (e- / s)
        snr (np.ndarray): SNR
        noise (np.ndarray): Noise (e- / s)
        read_noise (np.ndarray): Effective read noise (e- / s)
    """

    # Integrate over telescope aperture (photons / sec)
    source_rate = source_image * collarea
    sky_emission_rate = sky_emission_rate * collarea

    # Efficiency (effectively converts to e- / s)
    source_rate *= efftot
    sky_emission_rate *= efftot

    # Dark rate (e- / s)
    dark_rate = dark_current

    # Integrate source and background over itime and frames (e-)
    source_tot = source_rate * itime * n_frames
    dark_tot = dark_rate * itime * n_frames
    sky_emission_tot = sky_emission_rate * itime * n_frames

    # Final simulated image over all frames (e-)
    sim_tot = source_tot + dark_tot + sky_emission_tot

    # Add poisson noise to final image (e-)
    observed_tot = np.random.poisson(lam=sim_tot, size=sim_tot.shape)

    # Total read noise noise contribution over all frames (e-)
    read_noise_tot = np.random.normal(
        loc=0,
        scale=read_noise * np.sqrt(n_frames) * np.sqrt(num_detector_pixels),
        size=observed_tot.shape
    )

    # Add read noise to final image
    observed_tot = observed_tot + read_noise_tot

    # Simulated noise
    noise_tot = np.sqrt(sim_tot + (read_noise * np.sqrt(n_frames))**2)

    # SNR
    snr = source_tot / noise_tot

    # Convert back to e-/s
    sim_rate = sim_tot / (n_frames * itime)
    observed_rate = observed_tot / (n_frames * itime)
    source_rate = source_tot / (n_frames * itime)
    noise_rate = noise_tot / (n_frames * itime)

    # Outputs
    out = dict(
        sim_rate=sim_rate, sim_tot=sim_tot,
        observed_rate=observed_rate, observed_tot=observed_tot,
        source_rate=source_rate, source_tot=source_tot,
        dark_rate=dark_rate, dark_tot=dark_tot,
        sky_emission_rate=sky_emission_rate, sky_emission_tot=sky_emission_tot,
        snr=snr,
        noise_rate=noise_rate, noise_tot=noise_tot,
        read_noise_tot=read_noise_tot,
    )
      
    # Return
    return out