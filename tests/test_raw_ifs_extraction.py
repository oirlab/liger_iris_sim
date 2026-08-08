"""
Round-trip tests for the raw IFS simulate -> extract pipeline, built on the
same calls as the `sim_raw_frame_liger_ifs` / `extract_raw_frame_liger_ifs`
example notebooks.

Each test illuminates exactly one lenslet at a time (every other lenslet's
input spectrum is zero) and checks that the extracted spectrum for that
lenslet matches what was put in. Isolating a single lenslet like this rules
out cross-talk with neighbouring traces as an explanation for any mismatch --
what's left is purely the render + extract round trip for that lenslet's own
trace geometry, which is what actually varies from lenslet to lenslet across
the array.
"""
import numpy as np
import pytest

from liger_iris_sim.raw_ifs import simulate_raw_ifs_frame
from liger_iris_sim.raw_ifs.rectmat import make_rectmat
from liger_iris_sim.raw_ifs.extraction import extract_ifs_lenslet
from liger_iris_sim.utils import LIGER_PROPS, generate_wave_grid_for_filter
from liger_iris_drp_resources import (
    load_filters_summary,
    load_ifs_array_mask,
    load_ifs_trace_geometry,
)

FILTER_NAME = "KN2"
RESOLUTION = 4000
ARRAY_SIZE = 128


@pytest.fixture(scope="module")
def rectmat(tmp_path_factory):
    """Built once and shared by every test in this module -- it only depends
    on the instrument mode/filter/resolution, not on any particular cube."""
    path = tmp_path_factory.mktemp("rectmats") / f"rectmat_{FILTER_NAME}_{RESOLUTION}.fits"
    out = make_rectmat(
        ifs_mode="lenslet",
        filter_name=FILTER_NAME,
        resolution=RESOLUTION,
        output_path=str(path),
    )
    return out


@pytest.fixture(scope="module")
def wave_grid_and_flat_spectrum():
    """A flat (constant-density) input spectrum, so any recovered deviation
    from 1.0 reflects an instrumental/pipeline systematic rather than
    aliasing against spectral structure."""
    filter_info = load_filters_summary(FILTER_NAME)
    wave_grid = np.linspace(filter_info["wavemin"], filter_info["wavemax"], 500)
    flux = np.full_like(wave_grid, 1e6, dtype=np.float64)
    wave_out = generate_wave_grid_for_filter(filter_info, resolution=RESOLUTION, sampling_factor=2)
    return wave_grid, flux, wave_out


@pytest.fixture(scope="module")
def fully_onboard_mask():
    """
    Which of the nominal 128x128 lenslets land *entirely* on the physical
    4096x4096 detector (all 5 Zemax trace points in bounds), computed
    directly from the trace geometry rather than from rectmat validity.

    This matters because rectmat's `offsets[0] >= 0` is a weaker condition:
    `_trace_pixel_span` clamps the column range to the detector width
    before checking pix_lo <= pix_hi, so a lenslet whose trace only
    partially overlaps the detector (e.g. most of it off-chip in Y) can
    still be marked "valid" there while actually extracting to all-NaN --
    see test_partially_off_detector_lenslet_yields_no_flux below. Fully
    on-detector is the right criterion for "this lenslet should recover a
    real spectrum".
    """
    det_shape = LIGER_PROPS["ifs_detector_size"]
    mask = load_ifs_array_mask()
    x_pix, y_pix = load_ifs_trace_geometry(
        ifs_mode="lenslet", filter_name=FILTER_NAME, resolution=RESOLUTION
    )
    n_y, n_x = mask.shape
    onboard = np.zeros((n_y, n_x), dtype=bool)
    for ly in range(n_y):
        for lx in range(n_x):
            idx = int(mask[ly, lx])
            xs = x_pix[idx]
            ys = y_pix[idx]
            onboard[ly, lx] = (
                (xs >= 0).all() and (xs < det_shape[1]).all()
                and (ys >= 0).all() and (ys < det_shape[0]).all()
            )
    return onboard


@pytest.fixture(scope="module")
def representative_lenslets(fully_onboard_mask):
    """
    A handful of lenslets spanning the real, fully-on-detector footprint
    (only ~44% of the nominal 128x128 lenslet grid lands entirely on the
    physical detector -- the rest disperse off-chip or straddle the edge).
    Includes the top/bottom/left/right extremes of that footprint (most
    likely to show edge-of-coverage artifacts), the array center, and a
    few deterministic interior points.
    """
    ys, xs = np.where(fully_onboard_mask)

    targets = {
        (int(ys[np.argmin(ys)]), int(xs[np.argmin(ys)])),  # topmost
        (int(ys[np.argmax(ys)]), int(xs[np.argmax(ys)])),  # bottommost
        (int(ys[np.argmin(xs)]), int(xs[np.argmin(xs)])),  # leftmost
        (int(ys[np.argmax(xs)]), int(xs[np.argmax(xs)])),  # rightmost
        (64, 64),                                          # array center
    }
    rng = np.random.default_rng(0)
    for _ in range(3):
        i = rng.integers(0, len(ys))
        targets.add((int(ys[i]), int(xs[i])))
    return sorted(targets)


def _illuminate_and_extract(ly, lx, rectmat, wave_grid, flux, wave_out):
    cube = np.zeros((len(wave_grid), ARRAY_SIZE, ARRAY_SIZE), dtype=np.float32)
    cube[:, ly, lx] = flux
    sim = simulate_raw_ifs_frame(
        input_cube=cube,
        input_wave=wave_grid,
        ifs_mode="lenslet",
        filter_name=FILTER_NAME,
        resolution=RESOLUTION,
        itime=0,
        read_noise=0,
        tracepos_deg=1,
        wavesol_deg=1,
        density=True,
    )
    out = extract_ifs_lenslet(
        sim["sim"], rectmat["filepath"],
        error=None,
        output_wavesol=wave_out,
        interp_order=1,
    )
    return out["flux"][:, ly, lx]


class TestIlluminateEachLenslet:

    @pytest.mark.parametrize("ly,lx", [(64, 64)], ids=["center"])
    def test_center_lenslet_recovers_flat_spectrum_tightly(
        self, ly, lx, rectmat, wave_grid_and_flat_spectrum
    ):
        """The array-center lenslet has no edge-of-coverage effects, so it
        should recover the input spectrum almost exactly."""
        wave_grid, flux, wave_out = wave_grid_and_flat_spectrum
        true_at_out = np.interp(wave_out, wave_grid, flux)

        extracted = _illuminate_and_extract(ly, lx, rectmat, wave_grid, flux, wave_out)
        good = np.isfinite(extracted)
        ratio = extracted[good] / true_at_out[good]

        assert good.sum() / len(extracted) > 0.9
        assert np.mean(ratio) == pytest.approx(1.0, abs=0.01)
        assert np.std(ratio) < 0.01

    def test_recovers_flat_spectrum_across_the_array(
        self, rectmat, representative_lenslets, wave_grid_and_flat_spectrum
    ):
        """
        Sweep lenslets across the real on-detector footprint (not just the
        center) and check each recovers the flat input spectrum it alone
        was illuminated with.

        Tolerance is a bit looser than the center-lenslet test and uses the
        median rather than the mean: footprint-edge lenslets can have a
        single output channel land right at the boundary of their native
        wavelength coverage, which the resampling step (rightly) treats
        differently from interior channels -- that's a real, separate,
        small edge-of-coverage effect, not the multi-percent-scale,
        many-channel systematic this test is guarding against.
        """
        wave_grid, flux, wave_out = wave_grid_and_flat_spectrum
        true_at_out = np.interp(wave_out, wave_grid, flux)

        failures = []
        for ly, lx in representative_lenslets:
            extracted = _illuminate_and_extract(ly, lx, rectmat, wave_grid, flux, wave_out)
            good = np.isfinite(extracted)
            if good.sum() == 0:
                failures.append(f"({ly},{lx}): no finite output")
                continue

            ratio = extracted[good] / true_at_out[good]
            median_ratio = np.median(ratio)
            frac_within_2pct = np.mean(np.abs(ratio - 1.0) < 0.02)

            if not (abs(median_ratio - 1.0) < 0.01 and frac_within_2pct > 0.98):
                failures.append(
                    f"({ly},{lx}): median_ratio={median_ratio:.4f} "
                    f"frac_within_2pct={frac_within_2pct:.3f}"
                )

        assert not failures, "lenslets failing flat-spectrum recovery:\n" + "\n".join(failures)

    def test_off_detector_lenslet_yields_no_flux(self, rectmat, wave_grid_and_flat_spectrum):
        """Lenslets whose Zemax trace points fall off the physical detector
        (offsets[0] < 0 in the rectmat) must not be silently assigned a
        spectrum -- extraction should leave them entirely NaN rather than
        fabricate a value."""
        offsets = rectmat["offsets"]
        invalid = np.argwhere(offsets[0] < 0)
        assert len(invalid) > 0, "expected at least one off-detector lenslet in this footprint"
        ly, lx = (int(v) for v in invalid[0])

        wave_grid, flux, wave_out = wave_grid_and_flat_spectrum
        extracted = _illuminate_and_extract(ly, lx, rectmat, wave_grid, flux, wave_out)
        assert np.all(~np.isfinite(extracted))

    def test_partially_off_detector_lenslet_yields_no_flux(
        self, rectmat, fully_onboard_mask, wave_grid_and_flat_spectrum
    ):
        """
        Documents a discovered edge case, distinct from the fully-off-detector
        case above: `_trace_pixel_span` (rectmat.py) clamps a lenslet's
        column range to the detector width *before* checking pix_lo <=
        pix_hi, so a lenslet whose trace only partially overlaps the
        detector (e.g. mostly off-chip in Y, but its clamped X-column range
        still overlaps) gets offsets[0] >= 0 ("valid") even though its
        actual PSF footprint lands entirely outside the detector rows and
        extraction. As implemented today that still degrades gracefully to
        all-NaN rather than a fabricated value, which is what this test
        pins down -- but the "valid" offset is a mildly misleading signal
        to any caller that uses it as a stand-in for "has a usable
        spectrum". Flagged for a design decision rather than "fixed" here.
        """
        offsets = rectmat["offsets"]
        marked_valid = offsets[0] >= 0
        partially_off = marked_valid & ~fully_onboard_mask
        candidates = np.argwhere(partially_off)
        if len(candidates) == 0:
            pytest.skip("no rectmat-valid-but-partially-off-detector lenslet in this footprint")
        ly, lx = (int(v) for v in candidates[0])

        wave_grid, flux, wave_out = wave_grid_and_flat_spectrum
        extracted = _illuminate_and_extract(ly, lx, rectmat, wave_grid, flux, wave_out)
        assert np.all(~np.isfinite(extracted))
