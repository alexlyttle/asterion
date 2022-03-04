from __future__ import annotations
from logging import warning

import numpyro, warnings
import numpyro.distributions as dist
import jax.numpy as jnp

from jax.nn import sigmoid
from numpyro.infer import SVI, MCMC, NUTS, HMCECS, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adam
from typing import Dict, Optional
from numpy.typing import ArrayLike
from netCDF4 import Dataset


class TrainedBayesianNN:
    """Trained Bayesian neural network class.

    This class is the base class for BayesianNN, without the capability for
    training. It takes x and y scale parameters, trained samples, and optimized
    reference parameters as inputs.
    """

    def __init__(
        self,
        x_loc: ArrayLike,
        x_scale: ArrayLike,
        x_dim: int,
        y_loc: ArrayLike,
        y_scale: ArrayLike,
        y_dim: int,
        hidden_dim: int = 5,
        samples: Dict[str, ArrayLike] = None,
        ref_params: Dict[str, ArrayLike] = None,
    ):
        self._deprecation_warning()
        self.x_loc = x_loc
        self.x_scale = x_scale
        self.x_dim = x_dim

        self.y_loc = y_loc
        self.y_scale = y_scale
        self.y_dim = y_dim

        self.hidden_dim = hidden_dim
        self.samples = samples
        self.ref_params = ref_params

    def _deprecation_warning(self):
        warnings.warn(
            f"Class '{self.__class__.__name__}' is deprecated and " +
            "no longer supported."
        )

    def model(
        self,
        x: ArrayLike,
        y: Optional[ArrayLike] = None,
        subsample_size: Optional[int] = None,
    ):
        """Bayesian neural network model.

        Takes input `x` and transforms it to output `y` with noise `sigma`
        through weights layers `w0`, `w1` and `w2` each containing a hidden
        dimension of size `hidden_dim`. Optional subsampling may be done
        for faster performance.
        """
        num_obs = x.shape[0]
        x = (x - self.x_loc) / self.x_scale

        # prior on the observation noise
        sigma = numpyro.sample("sigma", dist.Gamma(1.0))

        # sample first layer (we put unit normal priors on all weights)
        shape = (self.x_dim, self.hidden_dim)
        w0 = numpyro.sample(
            "w0", dist.Normal(jnp.zeros(shape), jnp.ones(shape))
        )

        # sample second layer
        shape = (self.hidden_dim, self.hidden_dim)
        w1 = numpyro.sample(
            "w1", dist.Normal(jnp.zeros(shape), jnp.ones(shape))
        )

        # sample final layer of weights and neural network output
        shape = (self.hidden_dim, self.y_dim)
        w2 = numpyro.sample(
            "w2", dist.Normal(jnp.zeros(shape), jnp.ones(shape))
        )

        # observe data with optional subsampling along second rightmost dim
        with numpyro.plate(
            "obs", num_obs, subsample_size=subsample_size, dim=-2
        ):
            x = numpyro.subsample(x, event_dim=0)

            # Propagate forward through neural network
            z = sigmoid(x @ w0)
            z = sigmoid(z @ w1)
            z = z @ w2

            if y is not None:
                y = numpyro.subsample(y, event_dim=0)
                y = (y - self.y_loc) / self.y_scale  # Scale y during training
                assert z.shape == y.shape

            y_offset = numpyro.sample("y_scaled", dist.Normal(z, sigma), obs=y)
            if y is None:
                # We rescale y when making predictions
                numpyro.deterministic(
                    "y", self.y_loc + self.y_scale * y_offset
                )

    def predict(self, rng_key, x, kind="trained", num_samples=None):
        """Make predictions from the trained or optimized neural network.

        If kind == 'optimized' then `num_samples` must be passed.
        """
        if kind == "trained":
            predictive = Predictive(self.model, posterior_samples=self.samples)
        elif kind == "optimized":
            if num_samples is None:
                raise ValueError("Argument 'num_samples' is None.")
            model = numpyro.handlers.condition(
                self.model, data=self.ref_params
            )
            predictive = Predictive(model, num_samples=num_samples)
        else:
            raise ValueError(
                "Argument 'kind' must be one of 'trained' "
                + " or 'optimized'."
            )
        samples = predictive(rng_key, x)
        return samples

    @staticmethod
    def _data_from_file(file):
        samples = params = None
        if "samples" in file.groups:
            samples = {}
            for k, v in file["samples"].variables.items():
                samples[k] = jnp.array(v[()])
        if "ref_params" in file.groups:
            params = {}
            for k, v in file["ref_params"].variables.items():
                params[k] = jnp.array(v[()])
        return samples, params

    @classmethod
    def _from_root(cls, root):
        x_loc = root["training/x"].getncattr("loc")
        x_scale = root["training/x"].getncattr("scale")
        x_dim = root.dimensions["features"].size
        
        y_loc = root["training/y"].getncattr("loc")
        y_scale = root["training/y"].getncattr("scale")
        y_dim = root.dimensions["outputs"].size
        
        hidden_dim = root.dimensions["hidden"].size
        samples, params = cls._data_from_file(root)
        
        bnn = cls(
            x_loc,
            x_scale,
            x_dim,
            y_loc,
            y_scale,
            y_dim,
            hidden_dim=hidden_dim,
            samples=samples,
            ref_params=params,
        )
        return bnn

    @classmethod
    def from_file(cls, filename):
        
        with Dataset(filename, "r") as root:
            return cls._from_root(root)


class BayesianNN(TrainedBayesianNN):
    """Bayesian neural network class.

    Class for optimizing and/or training a simple Bayesian neural network.

    Args:
        x_train (:term:`array_like`): Input training data (features).
        y_train (:term:`array_like`): Output training data (targets).
        hidden_dim (int): Size of hidden layer dimension.
        samples (dict of :term:`array_like`): Dictionary mapping sample sites
            to samples.
        ref_params (dict of :term:`array_like`): Dictionary mapping sample
            sites to reference params (e.g. MAP or other optimization).
    """

    def __init__(
        self, x_train, y_train, hidden_dim=5, samples=None, ref_params=None
    ):
        self._deprecation_warning()
        self.x_train = x_train
        self.y_train = y_train

        self.hidden_dim = hidden_dim
        self.samples = samples
        self.ref_params = ref_params

    @property
    def x_train(self):
        return self._x_train

    @x_train.setter
    def x_train(self, value):
        self._x_train = value
        self.x_loc = value.mean(axis=0)
        self.x_scale = value.std(axis=0)
        self.x_dim = value.shape[1]

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        self._y_train = value
        self.y_loc = value.mean(axis=0)
        self.y_scale = value.std(axis=0)
        self.y_dim = value.shape[1]

    def optimize(
        self, rng_key, num_steps=5000, step_size=1e-2, subsample_size=100
    ):
        """Optimize the model with SVI using a delta function guide to obtain
        the MAP.
        
        Args:
            rng_key (jax.random.PRNGKey): Random key for training.
            num_steps (int): Number of steps (or epochs):
            step_size (float): Step size for the Adam optimizer.
            subsample_size (int): Size of training data subsamples (or
                batches).
        
        Returns:
            numpyro.infer.SVIRunResult: Resulting SVI run result object.
        """
        optimizer = Adam(step_size=step_size)
        guide = AutoDelta(self.model)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(
            rng_key,
            num_steps,
            self.x_train,
            y=self.y_train,
            subsample_size=subsample_size,
        )
        params = svi_result.params
        ref_params = {
            "sigma": params["sigma_auto_loc"],
        }
        for i in range(3):
            key = f"w{i}"
            ref_params[key] = params[f"{key}_auto_loc"]

        self.ref_params = ref_params
        return svi_result

    def train(
        self,
        rng_key,
        num_warmup=1000,
        num_samples=1000,
        num_chains=1,
        target_accept_prob=0.8,
        init_strategy=None,
        subsample_size=None,
        num_blocks=None,
    ):
        """Train the model using NUTS, or HMCECS if subsample_size is not None.
        
        Warning:
            This method is not yet tested, use with caution.
        
        Args:
            rng_key (jax.random.PRNGKey): Random key for training.
            num_warmup (int): Number of MCMC warmup steps.
            num_samples (int): Number of MCMC samples.
            num_chains (int): Number of parallel MCMC chains.
            target_accept_prob (float): Target MCMC acceptance probability.
            init_strategy (callable, optional): Initialization strategy.
                Default is :func:`numpyro.infer.init_to_median`.
            subsample_size (int, optional): Subsample size for HMCECS. Default
                is None to use the NUTS sampler.
            num_blocks (int, optional): Number of blocks for the HMCECS
                sampler. Default is 10 if subsample_size is not None.
        
        Returns:
            numpyro.infer.MCMC: Trained MCMC object.
        """
        if init_strategy is None:
            init_strategy = numpyro.infer.init_to_median

        sampler = NUTS(
            self.model,
            target_accept_prob=target_accept_prob,
            init_strategy=init_strategy,
        )

        if subsample_size is not None:
            if num_blocks is None:
                num_blocks = 10
            proxy = None
            if self.ref_params is not None:
                # taylor proxy estimates log likelihood (ll) by expansion about
                # reference params
                proxy = HMCECS.taylor_proxy(self.ref_params)
            sampler = HMCECS(sampler, num_blocks=num_blocks, proxy=proxy)

        mcmc = MCMC(
            sampler,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )
        mcmc.run(
            rng_key,
            self.x_train,
            y=self.y_train,
            subsample_size=subsample_size,
        )
        self.samples = mcmc.get_samples()
        return mcmc

    def to_file(self, filename, trained=False):
        """Save to NetCDF4 file.

        Args:
            filename (str or file_like):  File path or IO buffer.
            trained (bool): Whether neural network is considered trained. If
                False, the training data is saved, otherwise just the metadata
                such as scale parameters and dimension sizes are saved.
        """
        with Dataset(filename, 'w') as root:
            _ = root.createDimension("features", self.x_train.shape[1])
            _ = root.createDimension("outputs", self.y_train.shape[1])
            _ = root.createDimension("hidden", self.hidden_dim)

            training = root.createGroup("training")
            _ = training.createDimension("length", self.x_train.shape[0])
            x = training.createVariable("x", self.x_train.dtype, ("length", "features"))
            y = training.createVariable('y', self.y_train.dtype, ("length", "outputs"))
            x.setncattr('loc', self.x_loc)
            x.setncattr('scale', self.x_scale)
            y.setncattr('loc', self.y_loc)
            y.setncattr('scale', self.y_scale)

            if not trained:
                x[:] = self.x_train
                y[:] = self.y_train

            if self.samples is not None:
                samples = root.createGroup('samples')
                _ = samples.createDimension("draw", self.samples["sigma"].shape[0])
                sigma = samples.createVariable('sigma', self.samples['sigma'].dtype, ("draw",))
                w0 = samples.createVariable('w0', self.samples['w0'].dtype, ("draw", "features", "hidden"))
                w1 = samples.createVariable('w1', self.samples['w1'].dtype, ("draw", "hidden", "hidden"))
                w2 = samples.createVariable('w2', self.samples['w2'].dtype, ("draw", "hidden", "outputs"))
                w0[:] = self.samples['w0']
                w1[:] = self.samples['w1']
                w2[:] = self.samples['w2']
                sigma[:] = self.samples['sigma']

            if self.ref_params is not None:
                params = root.createGroup('ref_params')
                sigma = params.createVariable('sigma', self.ref_params['sigma'].dtype)
                w0 = params.createVariable('w0', self.ref_params['w0'].dtype, ("features", "hidden"))
                w1 = params.createVariable('w1', self.ref_params['w1'].dtype, ("hidden", "hidden"))
                w2 = params.createVariable('w2', self.ref_params['w2'].dtype, ("hidden", "outputs"))
                w0[:] = self.ref_params['w0']
                w1[:] = self.ref_params['w1']
                w2[:] = self.ref_params['w2']
                sigma[:] = self.ref_params['sigma']

    @classmethod
    def from_file(cls, filename):
        with Dataset(filename, "r") as root:
            x = root["training/x"][()]
            y = root["training/y"][()]
            x_train = jnp.array(x)
            y_train = jnp.array(y)
            
            if jnp.all(x.fill_value == x_train) or jnp.all(y.fill_value == y_train):
                # If no training data available:
                return TrainedBayesianNN._from_root(root)

            hidden_dim = root.dimensions["hidden"].size
            samples, params = cls._data_from_file(root)

        bnn = cls(
            x_train,
            y_train,
            hidden_dim=hidden_dim,
            samples=samples,
            ref_params=params,
        )
        return bnn
