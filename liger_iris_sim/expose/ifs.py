import numpy as np

__all__ = ['expose_ifs']

# SNR
def expose_ifs(
    source_cube : np.ndarray,
    itime : float, n_frames : int,
    collarea : float,
    sky_emission_rate : np.ndarray,
    sky_transmission : np.ndarray,
    tput : float,
    read_noise : float, dark_current : float, num_detector_pixels : float = 2.0
) -> dict:
    """
    Parameters
    ----------
    source_cube : np.ndarray
        Source cube (wave, y, x). Units are photons / sec / m^2.
    itime : float
        Integration time (sec).
    collarea : float
        Collimating area (m^2)
    sky_emission_rate : np.ndarray
        The sky background emission spectrum sampled on the same wavebins as source_cube
        in units of photons / sec / m^2.
        Sky emission is NOT modulated by sky_transmission.
    sky_transmission : np.ndarray
        The sky background transmission for each spectrum
        normalized to [0, 1] for each wavebin. Only affects the source spectrum.
    tput : float
        Total throughput (convert photons -> e-).
    num_detector_pixels : float
        The average number of detector pixels that correspond to an IFS voxel.

    Returns
    -------
    output : dict
        The final simulation and intermediate/ancillary products. Entries are:
            - sim_rate (np.ndarray): The simulated rate in e- / sec / wavebin.
            - sim_tot (np.ndarray): The simulated total in e- / wavebin.
            - observed_rate (np.ndarray): The observed rate in e- / sec / wavebin.
            - observed_tot (np.ndarray): The observed total in e- / wavebin.
            - source_rate (np.ndarray): The source rate in e- / sec / wavebin.
            - source_tot (np.ndarray): The source total in e- / wavebin.
            - dark_rate (float): The dark rate in e- / sec.
            - dark_tot (float): The dark total in e-.
            - sky_emission_rate (np.ndarray): The sky emission rate in e- / sec / wavebin.
            - sky_emission_tot (np.ndarray): The sky emission total in e- / wavebin.
            - sky_transmission (np.ndarray): The normalized sky transmission.
            - snr (np.ndarray): The SNR of the simulation.
            - noise_rate (np.ndarray): The noise rate in e- / sec / wavebin.
            - noise_tot (np.ndarray): The noise total in e- / wavebin.
            - read_noise_tot (float): The read noise contribution in e-.
    """

    # Shape
    nw, ny, nx = source_cube.shape

    # Integrate over telescope aperture (photons / sec)
    source_cube = source_cube * collarea
    sky_emission_rate = sky_emission_rate * collarea

    # Efficiency (effectively converts from photons to photoelectrons, i.e to: e- / s)
    source_cube *= tput
    sky_emission_rate *= tput

    # Dark rate (e- / s)
    dark_rate = dark_current * num_detector_pixels

    # Integrate source and background over itime and frames (e-)
    source_tot = source_cube * itime * n_frames
    dark_tot = dark_rate * itime * n_frames
    sky_emission_tot = sky_emission_rate * itime * n_frames

    # Sky transmission (e-)
    source_tot = source_tot * sky_transmission[:, None, None]

    # Final simulated image over all frames (e-)
    sim_tot = source_tot + dark_tot + sky_emission_tot[:, None, None]

    # Add poisson noise to final image (e-)
    observed_tot = np.random.poisson(lam=sim_tot, size=sim_tot.shape)

    # Total read noise noise contribution over all frames (e-)
    read_noise_tot = np.random.normal(
        loc=0,
        scale=read_noise * np.sqrt(n_frames) * np.sqrt(num_detector_pixels),
        size=(ny, nx)
    )

    # Add read noise to final image
    observed_tot = observed_tot + read_noise_tot[None, :, :]

    # Simulated noise
    noise_tot = np.sqrt(sim_tot + (read_noise * np.sqrt(n_frames))**2)

    # SNR
    snr = source_tot / noise_tot

    # Convert back to e-/s
    sim_rate = sim_tot / (n_frames * itime)
    observed_rate = observed_tot / (n_frames * itime)
    source_rate = source_tot / (n_frames * itime)
    noise_rate = noise_tot / (n_frames * itime)

    # Outputs in e-
    out = dict(
        sim_rate=sim_rate, sim_tot=sim_tot,
        observed_rate=observed_rate, observed_tot=observed_tot,
        source_rate=source_rate, source_tot=source_tot,
        dark_rate=dark_rate, dark_tot=dark_tot,
        sky_emission_rate=sky_emission_rate, sky_emission_tot=sky_emission_tot,
        sky_transmission=sky_transmission,
        snr=snr,
        noise_rate=noise_rate, noise_tot=noise_tot,
        read_noise_tot=read_noise_tot,
    )
      
    # Return
    return out