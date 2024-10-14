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
        itime : float, n_frames : int,
        collarea : float,
        sky_background : np.ndarray,
        efftot : float,
        gain : float, read_noise : float, dark_current : float
    ) -> dict:
    """
    Parameters:
    source_cube (np.ndarray): Source cube (y, x, wave). Each wave bin in units of photons / sec / m^2 / micron.
    itime (np.ndarray): Integration time (sec).
    collarea (np.ndarray): Collimating area (m^2)
    sky_background (np.ndarray): The sky background spectrum sampled on the same
        wave bins as source_cube. Each bin in sky_background in units of photons / sec / m^2 / micron.
    efftot (np.ndarray): Total efficiciency (convert photons -> e-)
    gain (float): Detector gain (e-)
    
    Returns:
    dict: The exposure components. Keys are:
        observed (np.ndarray): Combined exposure (e- / s / micron)
        source (np.ndarray): Source signal (e- / s / micron)
        background (np.ndarray): Background signal (e- / s / mircon)
        snr (np.ndarray): SNR
        noise (np.ndarray): Noise (e- / s / micron)
        read_noise (np.ndarray): Effective read noise (e- / s)
    """

    # Integrate over telescope aperture (photons / sec / micron)
    source_rate = source_cube * collarea
    sky_background_rate = sky_background * collarea

    # Efficiency (effectively converts to e- / s / micron)
    source_rate *= efftot
    sky_background_rate *= efftot

    # Dark rate (e- / s / micron?)
    dark_rate = dark_current * gain

    # Total background rate (e- / s / micron)
    background_rate = sky_background_rate + dark_rate

    # Integrate source and background over itime and frames (e- / micron)
    source_tot = source_rate * itime * n_frames
    background_tot = background_rate * itime * n_frames

    # Total read noise noise contribution over all frames, just make 2D (e-)
    read_noise_tot = np.random.normal(loc=0, scale=read_noise * np.sqrt(n_frames), size=source_cube[:, :, 0].shape)

    # Final simulated image over all frames (e-)
    sim_tot = source_tot + background_tot

    # Add poisson noise to final image (e-)
    sim_tot_noisy = np.random.poisson(lam=sim_tot, size=source_cube.shape)

    # Add read noise to final image
    sim_tot_noisy = sim_tot_noisy + read_noise_tot[:, :, None]

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