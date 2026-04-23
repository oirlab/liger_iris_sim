import numpy as np

__all__ = ['expose_ifs']

# SNR
def expose_ifs(
    source_cube : np.ndarray,
    itime : float = 1.0,
    n_frames : int = 1,
    collarea : float = 1.0,
    sky_emission_rate : np.ndarray | float = 0.0,
    sky_transmission : np.ndarray | float = 1.0,
    tput : float = 1.0,
    read_noise : float = 0.0,
    dark_current : float = 0.0,
    num_detector_pixels : float = 1.0,
) -> dict:
    """
    Parameters
    ----------
    source_cube : np.ndarray
        Source cube (wave, y, x). Units are photons / sec / m^2.
    itime : float
        The exposure time in seconds.
        Default is 1.0 sec.
    n_frames : int
        The total number of frames to coadd, each inducing a read noise.
        Default is 1.
    collarea : float
        The telescope collimating area in units of m^2.
        Default is 1.0 m^2.
    sky_emission_rate : float
        The background sky emission rate in units of photons / sec / m^2 / pixel.
        Default is 0.0.
    tput : float
        The total throughput of the system. Default is 1.0.
    read_noise : float
        The read noise in units of e- RMS. Default is 0.0.
    dark_current : float
        The dark current rate in units of e- / sec / pixel.
        Default is 0.0.

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
    if isinstance(sky_transmission, np.ndarray):
        if sky_transmission.ndim == 1:
            source_tot = source_tot * sky_transmission[:, None, None]
        elif sky_transmission.ndim == 3:
            source_tot = source_tot * sky_transmission
    else:
        source_tot = source_tot * sky_transmission

    # Final simulated image over all frames (e-)
    if isinstance(sky_emission_tot, np.ndarray) and sky_emission_tot.ndim == 1:
        sky_emission_tot = sky_emission_tot[:, None, None]

    sim_tot = source_tot + dark_tot + sky_emission_tot

    # Add poisson noise to final image (e-)
    observed_tot = np.random.poisson(lam=sim_tot, size=sim_tot.shape)

    # Total read noise noise contribution over all frames (e-)
    if read_noise > 0:
        read_noise_tot = np.random.normal(
            loc=0,
            scale=read_noise * np.sqrt(n_frames) * np.sqrt(num_detector_pixels),
            size=(ny, nx)
        )
    else:
        read_noise_tot = np.zeros((ny, nx), dtype=np.float32)

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