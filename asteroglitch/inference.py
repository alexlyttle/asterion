import math

import numpy as np

import jax
import jax.numpy as jnp

from jax import random

import numpyro
import numpyro.distributions as dist

from numpyro.infer.reparam import CircularReparam, LocScaleReparam
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, init_to_median, Predictive
from numpyro.distributions import constraints

from collections import namedtuple

import warnings

import arviz as az


class Data:
    def __init__(self, observed=None, coords=None, dims=None):
        self.observed = observed
        self.coords = coords
        self.dims = dims
        # self.nu = None
        # self.nu_err = None
        # self.n = None
        # self.constant = None
        self.prior = None
        self.prior_sample_stats = None
        self.prior_predictive = None
        self.posterior = None
        self.posterior_sample_stats = None
        self.posterior_predictive = None
    
    def to_arviz(self):
        data = az.from_dict(
            observed_data=self.observed,
            # constant_data=self.constant,
            prior=self.prior,
            sample_stats_prior=self.prior_sample_stats,
            prior_predictive=self.prior_predictive,
            posterior=self.posterior,
            sample_stats=self.posterior_sample_stats,
            posterior_predictive=self.posterior_predictive,
            coords=self.coords,
            dims=self.dims,
        )
        return data


class Inference:
    def __init__(self, model, num_warmup=1000, num_samples=1000, num_chains=5, *, seed):
        self._rng_key = random.PRNGKey(seed)

        self.model = model
        observed = {'nu': self.model.nu, 'nu_err': self.model.nu_err}
        coords = {k: v.coords for k, v in self.model.dimensions.items()}
        dims = {}
        trace = self.model.get_posterior_trace(self._rng_key)
        for k, v in trace.items():
            dims[k] = [dim.name for dim in v["cond_indep_stack"][::-1]]

        self.data = Data(observed=observed, coords=coords, dims=dims)
        
        self._num_warmup = num_warmup
        self._num_samples = num_samples
        self._num_chains = num_chains
    
    def _sample(self, model, extra_fields=(), init_params=None, 
                kernel_kwargs={}, mcmc_kwargs={}):
        
        target_accept_prob = kernel_kwargs.pop('target_accept_prob', 0.99)
        init_strategy = kernel_kwargs.pop('init_strategy', lambda site=None: init_to_median(site=site, num_samples=1000))
        step_size = kernel_kwargs.pop('step_size', 0.1)
        
        forbidden_keys = ['num_warmup', 'num_samples', 'num_chains']
        forbidden = [k for k in forbidden_keys if k in mcmc_kwargs.keys()]
        if len(forbidden) > 0:
            raise KeyError(f"Keys {forbidden} found in 'mcmc_kwargs' are already defined.")

        kernel = NUTS(model, target_accept_prob=target_accept_prob, init_strategy=init_strategy, 
                      step_size=step_size, **kernel_kwargs)

        mcmc = MCMC(kernel, num_warmup=self._num_warmup, num_samples=self._num_samples,
                    num_chains=self._num_chains, **mcmc_kwargs)

        rng_key, self._rng_key = random.split(self._rng_key)
        mcmc.run(rng_key, extra_fields=extra_fields, init_params=init_params)

        # TODO: warn diverging (and rhat if num_chains > 1)
        # samples = mcmc.get_samples(group_by_chain=True)
        # sample_stats = mcmc.get_extra_fields(group_by_chain=True)
        return mcmc
    
    def _get_samples(self, mcmc, group_by_chain=True):
        samples = mcmc.get_samples(group_by_chain=True)
        sample_stats = mcmc.get_extra_fields(group_by_chain=True) 
        return samples, sample_stats

    def sample_prior(self, *args, **kwargs):
        model = handlers.reparam(self.model.prior, self.model.prior_reparam)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r'/bnumpyro/b')
            mcmc = self._sample(model, *args, **kwargs)

        samples, sample_stats = self._get_samples(mcmc)

        self.data.prior = samples
        self.data.prior_sample_stats = sample_stats

    def sample_posterior(self, *args, **kwargs):
        model = handlers.reparam(self.model.posterior, self.model.posterior_reparam)
        # model = self.neural_transport.reparam(model)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r'/bnumpyro/b')
            mcmc = self._sample(model, *args, **kwargs)

        samples, sample_stats = self._get_samples(mcmc)

        self.data.posterior = samples
        self.data.posterior_sample_stats = sample_stats

    def _predictive(self, model, predictive_kwargs={}):
        posterior_samples = predictive_kwargs.pop('posterior_samples', None)
        num_samples = predictive_kwargs.pop('num_samples', None)
        batch_ndims = predictive_kwargs.pop('batch_ndims', 2)

        if posterior_samples is None and num_samples is None:
            num_samples = self._num_chains * self._num_samples

        predictive = Predictive(model, posterior_samples=posterior_samples, 
                                num_samples=num_samples, batch_ndims=batch_ndims,
                                **predictive_kwargs)

        rng_key, self._rng_key = random.split(self._rng_key)
        samples = predictive(rng_key)
        return samples
    
    def prior_predictive(self, *args, **kwargs):
        """
        args and kwargs for model.prior
        """
        samples = self._predictive(self.model.prior, *args, **kwargs)
        self.data.prior_predictive = samples
    
    def posterior_predictive(self, *args, **kwargs):
        """
        args and kwargs for model.posterior
        """
        predictive_kwargs = {'posterior_samples': self.data.posterior}
        samples = self._predictive(self.model.posterior, *args, predictive_kwargs=predictive_kwargs, **kwargs)
        self.data.posterior_predictive = samples
