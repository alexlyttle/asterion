import numpy as np
from asterion.models import GlitchModel


class TestGlitchModel:
    def test_init_required(self, glitch_model_args):
        model = GlitchModel(*glitch_model_args)
        
        # Check model n is same as input
        assert np.all(model.n == np.array(glitch_model_args[0]))
    
    def test_init_optional(self, glitch_model_args):
        teff = (5000., 100.)
        eps = (0.3, 0.1)
        model = GlitchModel(*glitch_model_args, teff=teff, epsilon=eps,
                            num_pred=100, seed=11)

    def test_eval_repr(self, glitch_model_args):
        model = GlitchModel(*glitch_model_args)
        globals = {'array': np.array, 'GlitchModel': GlitchModel}
        new_model = eval(repr(model), globals)
        assert repr(model) == repr(new_model)
