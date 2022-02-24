import pytest
import h5py

@pytest.fixture
def glitch_model_args():
    with h5py.File('tests/test_data.hdf5') as file:
        model = file['model']
        nu_max = model['nu_max'][()]
        delta_nu = model['delta_nu'][()]
        n = model['n'][()]
    return (n, nu_max, delta_nu)
