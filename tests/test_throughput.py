from liger_iris_sim.expose.throughput import compute_liger_throughput, compute_iris_throughput


def test_throughput(tmp_path):
    waves = [0.8, 1.25, 1.65, 2.2]
    for wave in waves:
        tput = compute_liger_throughput(
            mode='img',
            wave=wave,
            ifs_mode=None
        )
        assert 0 < tput < 1, f'Liger throughput at {wave} microns is out of bounds: {tput}'

        tput = compute_liger_throughput(
            mode='ifs',
            wave=wave,
            ifs_mode='slicer',
        )
        assert 0 < tput < 1, f'Liger throughput at {wave} microns is out of bounds: {tput}'

        tput = compute_liger_throughput(
            mode='ifs',
            wave=wave,
            ifs_mode='lenslet',
        )
        assert 0 < tput < 1, f'Liger throughput at {wave} microns is out of bounds: {tput}'