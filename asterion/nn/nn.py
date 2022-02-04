import numpyro, h5py
import numpyro.distributions as dist
import jax.numpy as jnp

from jax.nn import sigmoid
from numpyro.infer import SVI, MCMC, NUTS, HMCECS, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adam
from typing import Dict, Optional
from numpy.typing import ArrayLike


class TrainedBayesianNN:
    """Trained Bayesian neural network class.

    This class is the base class for BayesianNN, without the capability for
    training. It takes x and y scale parameters, trained samples, and optimized
    reference parameters as inputs.
    """
    def __init__(self, x_loc: ArrayLike, x_scale: ArrayLike, x_dim: int, 
                 y_loc: ArrayLike, y_scale: ArrayLike, y_dim: int, 
                 hidden_dim: int=5, samples: Dict[str, ArrayLike]=None,
                 ref_params: Dict[str, ArrayLike]=None):
        self.x_loc = x_loc
        self.x_scale = x_scale
        self.x_dim = x_dim
        
        self.y_loc = y_loc
        self.y_scale = y_scale
        self.y_dim = y_dim
        
        self.hidden_dim = hidden_dim
        self.samples = samples
        self.ref_params = ref_params

    def model(self, x: ArrayLike, y: Optional[ArrayLike]=None,
              subsample_size: Optional[int]=None):
        """Bayesian neural network model. 
        
        Takes input `x` and transforms it to output `y` with noise `sigma`
        through weights layers `w0`, `w1` and `w2` each containing a hidden
        dimension of size `hidden_dim`. Optional subsampling may be done
        for faster performance.
        """
        num_obs = x.shape[0]
        x = (x - self.x_loc) / self.x_scale
                
        # we put a prior on the observation noise
        sigma = numpyro.sample("sigma", dist.Gamma(1.0))

        # sample first layer (we put unit normal priors on all weights)
        shape = (self.x_dim, self.hidden_dim)
        w0 = numpyro.sample("w0", dist.Normal(jnp.zeros(shape), 
                                              jnp.ones(shape)))

        # sample second layer
        shape = (self.hidden_dim, self.hidden_dim)
        w1 = numpyro.sample("w1", dist.Normal(jnp.zeros(shape), 
                                              jnp.ones(shape)))

        # sample final layer of weights and neural network output
        shape = (self.hidden_dim, self.y_dim)
        w2 = numpyro.sample("w2", dist.Normal(jnp.zeros(shape), 
                                              jnp.ones(shape)))

        # observe data with optional subsampling along second rightmost dim
        with numpyro.plate("obs", num_obs, subsample_size=subsample_size, 
                           dim=-2):
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
                numpyro.deterministic("y", 
                                      self.y_loc + self.y_scale * y_offset)
    
    def predict(self, rng_key, x, kind='trained', num_samples=None):
        """Make predictions from the trained or optimized neural network.
        
        If kind == 'optimized' then `num_samples` must be passed.
        """
        if kind == 'trained':
            predictive = Predictive(self.model, posterior_samples=self.samples)
        elif kind == 'optimized':
            if num_samples is None:
                raise ValueError('Argument \'num_samples\' is None.')
            model = numpyro.handlers.condition(self.model, 
                                               data=self.ref_params)
            predictive = Predictive(model, num_samples=num_samples)
        else:
            raise ValueError('Argument \'kind\' must be one of \'trained\' ' +\
                             ' or \'optimized\'.')
        samples = predictive(rng_key, x)
        return samples
    
    @staticmethod
    def _data_from_file(file):
        samples = params = None
        hidden_dim = file['hidden_dim'][()]
        if 'samples' in file.keys():
            samples = {}
            for key, value in file['samples'].items():
                samples[key] = value[()]
        if 'ref_params' in file.keys():
            params = {}
            for key, value in file['ref_params'].items():
                params[key] = value[()]    
        
        return hidden_dim, samples, params

    @classmethod
    def from_file(cls, filename):
        with h5py.File(filename, 'r') as file:
            
            # Metadata
            x_loc = file['x_train'].attrs['loc']
            x_scale = file['x_train'].attrs['scale']
            x_dim = file['x_train'].attrs['dim']

            y_loc = file['y_train'].attrs['loc']
            y_scale = file['y_train'].attrs['scale']
            y_dim = file['y_train'].attrs['dim']                
            
            hidden_dim, samples, params = cls._data_from_file(file)
        
        bnn = cls(x_loc, x_scale, x_dim, y_loc, y_scale, y_dim,
                  hidden_dim=hidden_dim, samples=samples, ref_params=params)
        return bnn


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
    def __init__(self, x_train, y_train, hidden_dim=5, samples=None,
                 ref_params=None):
        
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
    
    def optimize(self, rng_key, num_steps=5000, step_size=1e-2,
                 subsample_size=100):
        """Optimize the model with SVI."""
        optimizer = Adam(step_size=step_size)
        guide = AutoDelta(self.model)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(rng_key, num_steps, self.x_train, y=self.y_train, 
                             subsample_size=subsample_size)
        params = svi_result.params
        ref_params = {
            "sigma": params["sigma_auto_loc"],
        }
        for i in range(3):
            key = f'w{i}'
            ref_params[key] = params[f"{key}_auto_loc"]

        self.ref_params = ref_params
        return svi_result

    def train(self, rng_key, num_warmup=1000, num_samples=1000, num_chains=1, 
              target_accept_prob=0.8, init_strategy=None, subsample_size=None,
              num_blocks=None):
        "Train the model using NUTS, or HMCECS if sample_size is not None."
        if init_strategy is None:
            init_strategy = numpyro.infer.init_to_median

        sampler = NUTS(self.model, target_accept_prob=target_accept_prob, 
                       init_strategy=init_strategy)
        
        if subsample_size is not None:
            if num_blocks is None:
                num_blocks = 10
            proxy = None
            if self.ref_params is not None:
                # taylor proxy estimates log likelihood (ll) by expansion about
                # reference params
                proxy = HMCECS.taylor_proxy(self.ref_params)
            sampler = HMCECS(sampler, num_blocks=num_blocks, proxy=proxy)
        
        mcmc = MCMC(sampler, num_warmup=num_warmup, num_samples=num_samples, 
                    num_chains=num_chains)
        mcmc.run(rng_key, self.x_train, y=self.y_train,
                 subsample_size=subsample_size)
        self.samples = mcmc.get_samples()
        return mcmc
    
    def to_file(self, filename, trained=False):
        """Save to HDF5 file. 
        
        Args:
            filename (str or file_like):  File path or IO buffer.
            trained (bool): Whether neural network is considered trained. If
                False, the training data is saved, otherwise just the metadata
                such as scale parameters and dimension sizes are saved.
        """
        with h5py.File(filename, 'w') as file:
            if trained:
                x_train = file.create_dataset(
                    'x_train',
                    data=h5py.Empty("f"),
                )
            else:
                x_train = file.create_dataset(
                    'x_train',
                    data=self.x_train,
                    chunks=True,
                    compression='gzip'
                )

            x_train.attrs['loc'] = self.x_loc
            x_train.attrs['scale'] = self.x_scale
            x_train.attrs['dim'] = self.x_dim

            if trained:
                y_train = file.create_dataset(
                    'y_train',
                    data=h5py.Empty("f"),
                )
            else:
                y_train = file.create_dataset(
                    'y_train',
                    data=self.y_train,
                    chunks=True,
                    compression='gzip'
                )

            y_train.attrs['loc'] = self.y_loc
            y_train.attrs['scale'] = self.y_scale
            y_train.attrs['dim'] = self.y_dim
            
            file.create_dataset('hidden_dim', data=self.hidden_dim)
            if self.samples is not None:
                samples = file.create_group('samples')
                for key, value in self.samples.items():
                    samples.create_dataset(key, data=value)
            if self.ref_params is not None:
                params = file.create_group('ref_params')
                for key, value in self.ref_params.items():
                    params.create_dataset(key, data=value)

    @classmethod
    def from_file(cls, filename):
        with h5py.File(filename, 'r') as file:
            trained = file['x_train'].shape is None or \
                file['y_train'].shape is None

        if trained:
            # I.e. doesn't contain training data
            return TrainedBayesianNN.from_file(filename)
  
        with h5py.File(filename, 'r') as file:
            x_train = file['x_train'][()]
            y_train = file['y_train'][()]
            
            hidden_dim, samples, params = cls._data_from_file(file)
        
        bnn = cls(x_train, y_train, hidden_dim=hidden_dim, samples=samples,
                  ref_params=params)
        return bnn
