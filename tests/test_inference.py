import h5py
import pytest
import numpy as np

from asterion.inference import Inference
from asterion.models import GlitchModel

@pytest.fixture
def glitch_truths():
    with h5py.File('tests/test_data.hdf5') as file:
        truths = file['truths']
        out = {}
        for k, v in truths.items():
            out[k] = v[()]
    return out

@pytest.fixture
def glitch_inference_kwargs():
    with h5py.File('tests/test_data.hdf5') as file:
        obs = file['observed']
        nu = obs['nu'][()]
        nu_err = obs['nu_err'][()]
    return {'nu': nu, 'nu_err': nu_err}


class TestGlitchInference:
    def test_inference(self, glitch_model_args, glitch_inference_kwargs,
                       glitch_truths):
        model = GlitchModel(*glitch_model_args)
        infer = Inference(model, **glitch_inference_kwargs)        
        infer.sample()
        infer.posterior_predictive()
        
        data = infer.get_data()
        for key, truth in glitch_truths.items():
            samples = data.posterior.get(key, None)
            if samples is None:
                samples = data.posterior_predictive[key]
            dim = ('draw', 'chain')
            mean = samples.mean(dim=dim)
            std = samples.std(dim=dim)
            assert np.all(mean + 3*std > truth)
            assert np.all(mean - 3*std < truth)
