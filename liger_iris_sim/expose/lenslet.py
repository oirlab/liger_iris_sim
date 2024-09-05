import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.constants
c = scipy.constants.c # m/s
h = scipy.constants.h # J s

# SNR
# SNR equation:
# S/N = S*sqrt(T)/sqrt(S + npix(B + D + R^2/t))
# t = itime per frame
# T = sqrt(Nframes*itime)
# S, B, D -> electrons per second
# R -> read noise (electrons)

def expose_ifu_lenslet(
        source_cube : np.ndarray,
        itime : float, scale : float = 0.004, n_frames : int = 1,
        collarea : float = 630, sky_background_rate_m2 : float = 0, efftot : float | None = None,
        gain : float = 3.04, read_noise : float = 5, dark_current : float = 0.002
    ):

    # Pixel size on sky
    pixsize = scale**2 # arcsec^2 / spaxial

    # Integrate background over telescope aperture and include efficiency
    sky_background_rate = sky_background_rate_m2 * collarea * efftot # photons / second

    # Distribute over spaxial
    sky_background_rate *= pixsize # photons arcsec^2 / second / spaxial

    # Integrate source image over telescope aperture and include efficiency
    source_rate = source_cube * collarea * efftot # photons / second

    # Integrate source over itime
    source_total = source_rate * itime
    
    # Combine source + sky background
    source_sky_rate = source_rate + sky_background_rate

    # Dark current rate
    dark_rate = dark_current

    # Combine sky background + dark current
    background_rate = sky_background_rate + dark_rate
    
    # Observed image integrated over itime
    observed_total_per_frame = itime * (source_rate + background_rate)

    # Observed noise per frame
    observed_noise_per_frame = np.sqrt(observed_total_per_frame + read_noise**2)

    # Total snr where s is source, noise is everything else
    observed_snr_per_frame = source_rate * itime / observed_noise_per_frame

    # Now add Poisson noise to the total observed image
    observed_total_per_frame_with_noise = np.random.poisson(
        lam=observed_total_per_frame,
        size=observed_total_per_frame.shape
    ).astype("float64")

    # Outputs
    out = {
        "source_sky_rate" : source_sky_rate,
        "dark_rate" : dark_rate,
        "observed_total_per_frame" : observed_total_per_frame,
        "observed_total_per_frame_with_noise" : observed_total_per_frame_with_noise,
        "observed_snr_per_frame" : observed_snr_per_frame,
        "observed_noise_per_frame" : observed_noise_per_frame
    }

    # Return
    return out