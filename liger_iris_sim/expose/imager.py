import numpy as np

__all__ = ['expose_imager']

def expose_imager(
    source_image : np.ndarray,
    itime : float, n_frames : int,
    collarea : float, sky_emission_rate : float, tput : float,
    read_noise : float, dark_current : float,
) -> dict:
    """
    Parameters
    ----------
    source_image : np.ndarray
        The source image with the correct shape and scale as the final image, in units of photons / sec / m^2.
    itime : float
        The exposure time in seconds.
    n_frames : int
        The total number of frames to coadd, each inducing a read noise.
    collarea : float
        The telescope collimating area in units of m^2.
    sky_emission_rate : float
        The background sky emission rate in units of photons / sec / m^2 / pixel.
    tput : float
        The total throughput of the system (top of atmosphere -> detector).
    read_noise : float
        The read noise in units of e- RMS.
    dark_current : float
        The dark current rate in units of e- / sec / pixel.

    Returns
    -------
    output : dict
        The exposure components. Keys are:
            - 'sim_rate': 2D numpy array of the simulated image rate, noise free (e- / s)
            - 'sim_tot': 2D numpy array of the simulated image total, noise free (e-)
            - 'observed_rate': 2D numpy array of the observed image rate, with noise (e- / s)
            - 'observed_tot': 2D numpy array of the observed image total, with noise (e-)
            - 'source_rate': 2D numpy array of the source signal rate, noise free (e- / s)
            - 'source_tot': 2D numpy array of the source signal total, noise free (e-)
            - 'dark_rate': Value (scalar) of the dark current rate (e- / s)
            - 'dark_tot': Value (scalar) of the dark current total (e-)
            - 'sky_em_rate': Value (scalar) of the sky emission rate (e- / s)
            - 'sky_em_tot': Value (scalar) of the sky emission total (e-)
            - 'snr': 2D numpy array of the SNR for each pixel
            - 'noise_rate': 2D numpy array of the noise rate (e- / s)
            - 'noise_tot': 2D numpy array of the noise total (e-)
            - 'read_noise_tot': 2D numpy array of the read noise total (e-)

    """

    # Integrate over telescope aperture (photons / sec)
    source_rate = source_image * collarea
    sky_em_rate = sky_emission_rate * collarea

    # Throughput (effectively converts from photons / sec to e- / s)
    source_rate *= tput
    sky_em_rate *= tput

    # Dark rate (e- / s)
    dark_rate = dark_current

    # Integrate source and background over itime and frames (e-)
    source_tot = source_rate * itime * n_frames
    dark_tot = dark_rate * itime * n_frames
    sky_em_tot = sky_em_rate * itime * n_frames

    # Final simulated image over all frames (e-)
    sim_tot = source_tot + dark_tot + sky_em_tot

    # Add poisson noise to final image (e-)
    observed_tot = np.random.poisson(lam=sim_tot, size=sim_tot.shape)

    # Total read noise contribution over all frames (e-)
    read_noise_tot = np.random.normal(
        loc=0,
        scale=read_noise * np.sqrt(n_frames),
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

        # Full sim (perfect noiseless image)
        sim_rate=sim_rate, # e- / s
        sim_tot=sim_tot, # e-

        # Full sim with noise
        observed_rate=observed_rate, # e- / s
        observed_tot=observed_tot, # e-

        # Recorded source rate at detector
        source_rate=source_rate, # e- / s
        source_tot=source_tot, # e-

        # Dark contribution
        dark_rate=dark_rate, # e- / s
        dark_tot=dark_tot, # e-

        # Sky contribution
        sky_em_rate=sky_em_rate, # e- / s
        sky_em_tot=sky_em_tot, # e-
        
        # Source SNR
        snr=snr,

        # Photon and read noise contributions
        noise_rate=noise_rate,
        noise_tot=noise_tot,
        read_noise_tot=read_noise_tot,
    )

    return out