import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.constants
c = scipy.constants.c # m/s
h = scipy.constants.h # J s

import utils
from filters import get_iris_filter_data
from psf import get_iris_imager_psf

# SNR
# SNR equation:
# S/N = S*sqrt(T)/sqrt(S + npix(B + D + R^2/t))
# t = itime per frame
# T = sqrt(Nframes*itime)
# S, B, D -> electrons per second
# R -> read noise (electrons)

def iris_sim_imager(
        source_image : np.ndarray, filter : str, scale : float,
        itime : float, n_frames : int,
        collarea : float, bgmag : float | None = None, efftot : float | None = None,
        gain : float = 1.0, read_noise : float = 1.0, dark_current : float = 0,
        saturation : bool = True, zenith : str = '45', simdir : str = '/data/group/data/iris/sim/',
    ):

    # Filter info
    filter_data = get_iris_filter_data(filter, simdir=simdir)
    wi, wc, wf = filter_data['wavemin'], filter_data['wavecenter'], filter_data['wavemax']

    # Calculate throughput
    if efftot is None:
        waves_eff = np.array([830, 900, 2000, 2200, 2300, 2412])
        imager_eff = np.array([0.631, 0.772, 0.772, 0.813, 0.763, 0.728] )
        tmt_eff = 0.91
        nf_eff = 0.8
        filter_eff = 0.9
        efftot = filter_eff * tmt_eff * nf_eff * imager_eff
        efftot = np.interp(np.array([wi, wf]), waves_eff, efftot)
        efftot = np.mean(efftot)

    # background magnitude integrated over this filter
    if bgmag is None:
        bgmag = filter_data['imagmag']

    # Pixel size on sky
    pixsize = scale**2 # arcsec^2 / pixel^2

    # Convert background to photons / s / meter^2
    sky_background_rate_m2 = filter_data['zp'] * 10**(-0.4 * bgmag)

    # Integrate background over telescope aperture and include efficiency
    sky_background_rate = sky_background_rate_m2 * collarea * efftot # photons / second

    # Integrate source image over telescope aperture and include efficiency
    source_rate = source_image * collarea * efftot # photons / second

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