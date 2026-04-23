# conftest.py
import pytest

from liger_iris_drp_resources import download

@pytest.fixture(scope="session", autouse=True)
def prepare_test_data():
    download(
        model_spectra=True,
        liger_psfs=True,
        iris_psfs=False,
        filter_trans=True,
        skip_if_exists=True
    )