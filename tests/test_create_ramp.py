from liger_iris_sim.utr.create_ramp import create_ramp
import numpy as np


def test_create_ramp():

    np.random.seed(1)

    shape = (32, 32)

    gain_map = np.ones(shape)
    detflat_map = np.ones(shape)
    dark_map = np.zeros(shape)
    bias_map = np.zeros(shape)
    rn_map = np.ones(shape) * 9
    readtime = 2.7
    n_reads = 10

    input_rate = 1000
    electron_rate_map = np.ones(shape) * input_rate # e-/s

    ramp_data = create_ramp(
        electron_rate_map,  # in e-/s
        readtime=readtime, n_reads=n_reads,
        nonlin_coeffs=None,
        gain=gain_map,  # in e-/s
        flat=detflat_map,
        dark=dark_map,  # in e-/s
        bias=bias_map,  # in e-/s
        kTC_noise=50,  # in e-/s
        poisson_noise=True, read_noise=rn_map,  # in e-/s
        convert_to_uint16=True, clip_ramps=True,
        max_cores=1,
        n_channels=1
    )

    t = np.linspace(0, readtime * (n_reads - 1), n_reads)
    y = ramp_data['data'][:, 16, 16]
    pfit = np.polyfit(t, y, deg=1)
    assert abs(pfit[0] - input_rate) / input_rate < 0.01