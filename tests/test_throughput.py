from liger_iris_sim.expose.throughput import compute_throughput, get_filter_throughput
import numpy as np

def test_throughput(tmp_path):
    
    waves = [0.8, 1.25, 1.65, 2.2]
    instrument_names = ['Liger', 'IRIS']
    filter_name = 'J'
    inst_mode = 'IMG'
    ifs_mode = None
    for inst_name in instrument_names:
        for wave in waves:
            tput = compute_throughput(
                instrument_name=inst_name,
                instrument_mode=inst_mode,
                wave=wave,
                ifs_mode=ifs_mode,
                filt=0.9,
            )
            assert 0 < tput < 1, f'Throughput out of bounds: {tput}'


    inst_mode = 'IFS'
    ifs_modes = ['SLICER', 'LENSLET']
    for inst_name in instrument_names:
        for wave in waves:
            for ifs_mode in ifs_modes:
                tput = compute_throughput(
                    instrument_name=inst_name,
                    instrument_mode=inst_mode,
                    wave=wave,
                    ifs_mode=ifs_mode,
                    filt=0.9,
                )
                assert 0 < tput < 1, f'Throughput out of bounds: {tput}'


    # Filter throughput at max wavelength
    wave, tput_filt = get_filter_throughput(
        filter_name=filter_name, wave=None
    )
    assert 0 < tput_filt < 1, f'Filter throughput out of bounds: {tput_filt}'

    # Filter throughput at specified wavelength
    wave, tput_filt = get_filter_throughput(
        filter_name=filter_name, wave=1.25
    )
    assert 0 < tput_filt < 1, f'Filter throughput out of bounds: {tput_filt}'

    # At array of wavelengths
    wave = np.linspace(1.24, 1.26, 5)
    wave, tput_filt = get_filter_throughput(
        filter_name='J', wave=wave
    )
    assert isinstance(tput_filt, np.ndarray), 'Wavelength output should be an array'
    assert np.all(tput_filt > 0) and np.all(tput_filt < 1), f'Filter throughput out of bounds: {tput_filt}'