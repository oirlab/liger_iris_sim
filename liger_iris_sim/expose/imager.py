import numpy as np
import matplotlib.pyplot as plt

def expose_imager(
        source_image : np.ndarray,
        itime : float, n_frames : int,
        collarea : float, sky_background_rate : float, efftot : float,
        gain : float, read_noise : float, dark_current : float
    ) -> dict:
    """
    Parameters:
    source_image (np.ndarray): The source image with the correct shape as the final image, in units of photons / sec / m^2.
    itime (float): The exposure time in seconds.
    n_frames (float): The total number of frames to coadd, each inducing a read noise.
    collarea (float): The telescope collimating area in units of m^2.
    sky_background_rate (float): The sky background rate in units of photons / sec / m^2.
    efftot (float): The total efficiency of the system (top of atmosphere -> detector).
    gain (float): The gain in units of e- / ADU.
    read_noise (float): The read noise in units of e- RMS.
    dark_current (float): The dark current rate in units of ADU / s

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
    sky_background_rate = sky_background_rate * collarea

    # Efficiency (effectively converts to e- / s)
    source_rate *= efftot
    sky_background_rate *= efftot

    # Dark rate (e- / s)
    dark_rate = dark_current * gain

    # Total background rate (e- / s)
    background_rate = sky_background_rate + dark_rate

    # Integrate source and background over itime and frames (e-)
    source_tot = source_rate * itime * n_frames
    background_tot = background_rate * itime * n_frames

    # Final simulated image over all frames (e-)
    sim_tot = source_tot + background_tot

    # Add poisson noise to final image (e-)
    sim_tot_noisy = np.random.poisson(lam=sim_tot, size=source_image.shape)

    # Total read noise noise contribution over all frames (e-)
    read_noise_tot = np.random.normal(loc=0, scale=read_noise * np.sqrt(n_frames), size=source_image.shape)

    # Add read noise to final image
    sim_tot_noisy = sim_tot_noisy + read_noise_tot

    # Simulated noise
    noise_tot = np.sqrt(sim_tot + (read_noise * np.sqrt(n_frames))**2)

    # SNR
    snr = source_tot / noise_tot

    # Convert back to e-/s
    sim_tot_noisy /= (n_frames * itime)
    source_tot /= (n_frames * itime)
    background_tot /= (n_frames * itime)
    noise_tot /= (n_frames * itime)

    # Outputs
    out = dict(
        observed=sim_tot_noisy,
        source=source_tot, background=background_tot,
        snr=snr, noise=noise_tot,
        read_noise=read_noise_tot,
    )
      
    # Return
    return out