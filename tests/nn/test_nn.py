import os
import numpy as np

from jax import random
from asterion.nn import TrainedBayesianNN, BayesianNN
from asterion.utils import PACKAGE_DIR

FILENAME = os.path.join(PACKAGE_DIR, "data", "tau_prior.nc")


class TestIOBayesionNN:
    
    bnn = BayesianNN.from_file(FILENAME)
    
    def test_dims(self):
        """Test BNN dimensions loaded correctly."""
        # Update right-hand values if they change
        assert self.bnn.x_dim == 2
        assert self.bnn.y_dim == 2
        assert self.bnn.hidden_dim == 10
        
    def test_ref_params(self):
        """Test reference parameter shapes."""
        params = self.bnn.ref_params
        assert params["sigma"].shape == ()
        assert params["w0"].shape == (self.bnn.x_dim, self.bnn.hidden_dim)
        assert params["w1"].shape == (self.bnn.hidden_dim, self.bnn.hidden_dim)
        assert params["w2"].shape == (self.bnn.hidden_dim, self.bnn.y_dim)

    def test_training_data(self):
        """Test training data shapes."""
        length = 174559  # <-- update this if changes
        assert self.bnn.x_train.shape == (length, self.bnn.x_dim)
        assert self.bnn.y_train.shape == (length, self.bnn.y_dim)

    def _compare_bnn(self, bnn, ignore=None):
        """Compare new BNN with reference BNN."""
        if ignore is None:
            ignore = []
        d1 = self.bnn.__dict__
        d2 = bnn.__dict__
        for key in d1.keys():
            if key in ignore:
                continue
            assert key in d2.keys()  # <-- check key in bnn attributes
            if isinstance(d1[key], dict):
                for v1, v2 in zip(d1[key].values(), d2[key].values()):
                    assert np.all(v1 == v2)
            else:
                assert np.all(d1[key] == d2[key])

    def test_to_file(self, tmp_path):
        """Test saving BNN to file and compare to reference."""
        filename = tmp_path / "tau_prior.nc"
        self.bnn.to_file(filename)
        bnn = BayesianNN.from_file(filename)
        assert isinstance(bnn, BayesianNN)  # <-- check is correct instance
        self._compare_bnn(bnn)
    
    def test_to_file_trained(self, tmp_path):
        """Test saving BNN to file as trained and compare to reference."""
        filename = tmp_path / "tau_prior_trained.nc"
        self.bnn.to_file(filename, trained=True)
        bnn = BayesianNN.from_file(filename)
        assert isinstance(bnn, TrainedBayesianNN)
        self._compare_bnn(bnn, ignore=["_x_train", "_y_train"])

class TestBayesianNN:
    
    bnn = BayesianNN.from_file(FILENAME)
    
    def test_optimize(self):
        """Test optimizing the BNN."""
        key = random.PRNGKey(0)
        ref_params = self.bnn.ref_params.copy()
        _ = self.bnn.optimize(key, 100, subsample_size=10)
        assert self.bnn.ref_params["sigma"] != ref_params["sigma"]
        self.bnn.ref_params = ref_params # <-- reset ref params

    def test_predict_optimized(self):
        """Test making predictions with the optimized BNN."""
        key = random.PRNGKey(0)
        num_samples = 10
        x = np.zeros((5, 2))
        samples = self.bnn.predict(key, x, kind='optimized', num_samples=num_samples)
        assert samples["y"].shape == (num_samples,) + x.shape
