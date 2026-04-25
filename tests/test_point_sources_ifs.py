import numpy as np
import pytest
from liger_iris_sim.sources import make_point_source_ifs_cube
from liger_iris_sim.utils import generate_wave_grid_for_filter
from liger_iris_sim.utils.psf_utils import shift_psf_phase
from liger_iris_sim.sources.convolve import convolve_point_source
from liger_iris_drp_resources.filters import load_filters_summary


@pytest.fixture
def wave_and_filter():
    filter_info = load_filters_summary(filter_name='J')
    wave = generate_wave_grid_for_filter(filter_info, resolution=4000)
    return wave, filter_info


@pytest.fixture
def simple_psf():
    psf = np.zeros((149, 149), dtype=np.float64)
    psf[74, 74] = 1.0
    return psf


@pytest.fixture
def gaussian_psf():
    y, x = np.mgrid[-74:75, -74:75]
    psf = np.exp(-(x**2 + y**2) / (2 * 5**2)).astype(np.float64)
    psf /= psf.sum()
    return psf


# --- shift_psf_phase ---

class TestShiftPsfPhase:

    def test_preserves_shape(self, gaussian_psf):
        shifted = shift_psf_phase(gaussian_psf, dx=0.3, dy=-0.2)
        assert shifted.shape == gaussian_psf.shape

    def test_preserves_normalization(self, gaussian_psf):
        shifted = shift_psf_phase(gaussian_psf, dx=0.3, dy=-0.2)
        assert shifted.sum() == pytest.approx(1.0, rel=1e-5)

    def test_no_negative_values(self, gaussian_psf):
        shifted = shift_psf_phase(gaussian_psf, dx=0.5, dy=0.5)
        assert np.all(shifted >= 0)

    def test_zero_shift_is_identity(self, gaussian_psf):
        shifted = shift_psf_phase(gaussian_psf, dx=0.0, dy=0.0)
        np.testing.assert_allclose(shifted, gaussian_psf, atol=1e-6)

    def test_subpixel_shift_moves_centroid(self, gaussian_psf):
        dx, dy = 0.4, 0.3
        shifted = shift_psf_phase(gaussian_psf, dx=dx, dy=dy)
        y, x = np.mgrid[:149, :149]
        cx0 = np.sum(x * gaussian_psf)
        cy0 = np.sum(y * gaussian_psf)
        cx1 = np.sum(x * shifted)
        cy1 = np.sum(y * shifted)
        assert cx1 - cx0 == pytest.approx(dx, abs=0.01)
        assert cy1 - cy0 == pytest.approx(dy, abs=0.01)


# --- convolve_point_source ---

class TestConvolvePointSource:

    def test_flux_conservation(self, gaussian_psf):
        image = np.zeros((64, 64), dtype=np.float32)
        flux = 100.0
        convolve_point_source(32.0, 32.0, flux, gaussian_psf, image_out=image)
        assert image.sum() == pytest.approx(flux, rel=1e-4)

    def test_source_at_center(self, simple_psf):
        image = np.zeros((149, 149), dtype=np.float32)
        convolve_point_source(74.0, 74.0, 1.0, simple_psf, image_out=image, fix_psf_phase=False)
        assert image[74, 74] == pytest.approx(1.0, rel=1e-5)

    def test_out_of_bounds_source_ignored(self, gaussian_psf):
        image = np.zeros((64, 64), dtype=np.float32)
        convolve_point_source(500.0, 500.0, 1.0, gaussian_psf, image_out=image)
        assert image.sum() == pytest.approx(0.0)

    def test_does_not_mutate_psf(self, gaussian_psf):
        original = gaussian_psf.copy()
        image = np.zeros((64, 64), dtype=np.float32)
        convolve_point_source(32.0, 32.0, 1.0, gaussian_psf, image_out=image)
        np.testing.assert_array_equal(gaussian_psf, original)

    def test_accumulates_into_existing_image(self, gaussian_psf):
        image = np.zeros((64, 64), dtype=np.float32)
        convolve_point_source(32.0, 32.0, 1.0, gaussian_psf, image_out=image)
        convolve_point_source(32.0, 32.0, 1.0, gaussian_psf, image_out=image)
        assert image.sum() == pytest.approx(2.0, rel=1e-4)


# --- make_point_source_ifs_cube ---

class TestMakePointSourceIfsCube:

    def test_output_shape(self, wave_and_filter, gaussian_psf):
        wave, filter_info = wave_and_filter
        size = (64, 64)
        spec = np.exp(-(wave - filter_info['wavecenter'])**2 / (2 * 0.05**2))
        template = (wave, spec / spec.sum())
        cube = make_point_source_ifs_cube(
            xdet=32.0, ydet=32.0, wave=wave,
            template=template, flux_int=100.0,
            psf=gaussian_psf, size=size,
        )
        assert cube.shape == (len(wave), *size)

    def test_flux_conservation(self, wave_and_filter, gaussian_psf):
        wave, filter_info = wave_and_filter
        flux = 500.0
        spec = np.exp(-(wave - filter_info['wavecenter'])**2 / (2 * 0.05**2))
        template = (wave, spec / spec.sum())
        cube = make_point_source_ifs_cube(
            xdet=32.0, ydet=32.0, wave=wave,
            template=template, flux_int=flux,
            psf=gaussian_psf, size=(64, 64),
        )
        assert cube.sum() == pytest.approx(flux, rel=1e-3)

    def test_redshift_shifts_centroid_wavelength(self, wave_and_filter, gaussian_psf):
        wave, filter_info = wave_and_filter
        wc = filter_info['wavecenter']
        z = 0.002
        spec = np.exp(-(wave - wc)**2 / (2 * 0.02**2))
        template = (wave, spec / spec.sum())
        cube = make_point_source_ifs_cube(
            xdet=32.0, ydet=32.0, wave=wave,
            template=template, flux_int=100.0,
            psf=gaussian_psf, size=(64, 64), z=z,
        )
        spectrum = cube[:, 32, 32]
        wc_rec = np.sum(wave * spectrum) / np.sum(spectrum)
        z_rec = (wc_rec - wc) / wc
        assert z_rec == pytest.approx(z, rel=0.05)

    def test_zero_flux_source_skipped(self, wave_and_filter, gaussian_psf):
        wave, _ = wave_and_filter
        # Template entirely outside the wave grid — will rebin to zero
        t_wave = np.linspace(3.0, 4.0, 100)
        t_spec = np.ones(100)
        cube = make_point_source_ifs_cube(
            xdet=32.0, ydet=32.0, wave=wave,
            template=(t_wave, t_spec), flux_int=100.0,
            psf=gaussian_psf, size=(64, 64),
        )
        assert cube.sum() == pytest.approx(0.0)

    def test_multiple_sources(self, wave_and_filter, gaussian_psf):
        wave, filter_info = wave_and_filter
        spec = np.exp(-(wave - filter_info['wavecenter'])**2 / (2 * 0.05**2))
        template = (wave, spec / spec.sum())
        fluxes = np.array([100.0, 200.0])
        cube = make_point_source_ifs_cube(
            xdet=np.array([20.0, 40.0]),
            ydet=np.array([20.0, 40.0]),
            wave=wave, template=template,
            flux_int=fluxes, psf=gaussian_psf, size=(64, 64),
        )
        assert cube.sum() == pytest.approx(fluxes.sum(), rel=1e-3)

    def test_prebinned_template_no_rebin(self, wave_and_filter, gaussian_psf):
        wave, filter_info = wave_and_filter
        spec = np.exp(-(wave - filter_info['wavecenter'])**2 / (2 * 0.05**2))
        spec = spec / spec.sum() * 100.0  # already in flux per bin
        cube = make_point_source_ifs_cube(
            xdet=32.0, ydet=32.0, wave=wave,
            template=spec, flux_int=None,
            psf=gaussian_psf, size=(64, 64),
        )
        assert cube.sum() == pytest.approx(100.0, rel=1e-3)