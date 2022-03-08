import pytest, os
from netCDF4 import Dataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="session")
def glitch_model_args():
    with Dataset(os.path.join(ROOT_DIR, "tests", "test_data.nc")) as root:
        model = root['model']
        nu_max = model['nu_max'][()].data
        delta_nu = model['delta_nu'][()].data
    return (nu_max, delta_nu)
