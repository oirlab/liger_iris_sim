import numpy as np
import matplotlib.pyplot as plt

# SNR
# SNR equation:
# S/N = S*sqrt(T)/sqrt(S + npix(B + D + R^2/t))
# t = itime per frame
# T = sqrt(Nframes*itime)
# S, B, D -> electrons per second
# R -> read noise (electrons)

def expose_ifu(
        source_cube : np.ndarray,
        wavebins : np.ndarray,
        itime : float, n_frames : int,
        collarea : float,
        sky_background : np.ndarray,
        efftot : float,
        gain : float, read_noise : float, dark_current : float
    ) -> dict:
    """
    Parameters:
    source_cube (np.ndarray): Source cube (y, x, wave). Units are photons / sec / m^2 / wavebin.
    wavebins (np.ndarray): Wavelength grid (microns).
    itime (float): Integration time (sec).
    collarea (np.ndarray): Collimating area (m^2)
    sky_background (np.ndarray): The sky background spectrum sampled on wavebins
        in units of photons / sec / m^2 / wavebin.
    efftot (np.ndarray): Total efficiciency (convert photons -> e-)
    gain (float): Detector gain (e-)
    
    Returns:
    dict: The exposure components. Keys are:
        observed (np.ndarray): Combined exposure (e- / s / wavebin)
        source (np.ndarray): Source signal (e- / s / wavebin)
        background (np.ndarray): Background signal (e- / s / wavebin)
        snr (np.ndarray): SNR
        noise (np.ndarray): Noise (e- / s / wavebin)
        read_noise (np.ndarray): Effective read noise (e- RMS wavebin)
    """

    # Integrate over telescope aperture (photons / sec / wavebin)
    source_rate = source_cube * collarea
    sky_background_rate = sky_background * collarea

    # Efficiency (effectively converts to e- / s / wavebin)
    source_rate *= efftot
    sky_background_rate *= efftot

    # Dark rate (e- / s)
    dark_rate = dark_current * gain

    # Total background rate (e- / s / wavebin)
    background_rate = sky_background_rate + dark_rate

    # Integrate source and background over itime and frames (e- / wavebin)
    source_tot = source_rate * itime * n_frames
    background_tot = background_rate * itime * n_frames

    # Total read noise noise contribution over all frames, just make 2D (e-)
    read_noise_tot = np.random.normal(loc=0, scale=read_noise * np.sqrt(n_frames), size=source_cube[:, :, 0].shape)

    # Final simulated image over all frames (e-)
    sim_tot = source_tot + background_tot

    # Simulated noise
    noise_tot = np.sqrt(sim_tot + (read_noise * np.sqrt(n_frames))**2)

    # Add poisson noise to final image (e-)
    sim_tot_noisy = np.random.poisson(lam=sim_tot, size=source_cube.shape).astype(float)
    #source_tot_noisy = np.random.poisson(lam=source_tot, size=source_cube.shape).astype(float)
    #background_tot_noisy = np.random.poisson(lam=background_tot, size=background_tot.shape).astype(float)
    #sim_tot_noisy = source_tot_noisy + background_tot_noisy
    #sim_tot_noisy = sim_tot.copy()

    #sim_tot_noisy = sim_tot + np.random.normal(loc=0, scale=noise_tot)

    # Add read noise to final image (broadcast to each wavelength slice)
    sim_tot_noisy = sim_tot_noisy + read_noise_tot[:, :, None]

    # SNR
    snr = source_tot / noise_tot

    # Convert back to e-/s
    sim_tot_noisy /= (n_frames * itime)
    source_tot /= (n_frames * itime)
    #source_tot_noisy /= (n_frames * itime)
    background_tot /= (n_frames * itime)
    #background_tot_noisy /= (n_frames * itime)
    noise_tot /= (n_frames * itime)

    # Outputs
    out = dict(
        observed=sim_tot_noisy,
        source=source_tot,
        #source_observed=source_tot_noisy,
        background=background_tot,
        #background_observed=background_tot_noisy,
        snr=snr, noise=noise_tot,
        read_noise=read_noise_tot,
    )
      
    # Return
    return out