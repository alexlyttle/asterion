import pytest, os
import numpy as np

from asterion.inference import Inference
from asterion.models import GlitchModel
from netCDF4 import Dataset

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="module")
def glitch_truths():
    with Dataset(os.path.join(TEST_DIR, "test_data.nc")) as file:
        truths = file['truths']
        out = {}
        for k, v in truths.variables.items():
            out[k] = v[()].data
    return out

@pytest.fixture(scope="module")
def glitch_inference_kwargs():
    with Dataset(os.path.join(TEST_DIR, "test_data.nc")) as file:
        obs = file['observed']
        n = obs["n"][()].data
        nu = obs['nu'][()].data
        nu_err = obs['nu_err'][()].data
    return {'n': n, 'nu': nu, 'nu_err': nu_err}


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
