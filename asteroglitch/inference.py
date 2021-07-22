import math

import numpy as np

import jax
import jax.numpy as jnp

from jax import random

import numpyro
import numpyro.distributions as dist

from numpyro.infer.reparam import CircularReparam, LocScaleReparam
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.distributions import constraints

from collections import namedtuple

import warnings

import arviz as az


class Inference:
    def __init__(self, model):
        self.model = model
        self.data = az.from_dict(
            observed_data=model.observed_data,
            constant_data=model.constant_data,
            predictions_constant_data=model.predictions_constant_data
        )
    
    def _sample(self, model, num_warmup=1000, num_samples=1000, num_chains=5, 
                seed=0, kernel_kwargs={}, mcmc_kwargs={},
                model_args=(), model_kwargs={}):
        
        target_accept_prob = kernel_kwargs.pop('target_accept_prob', 0.99)
        init_strategy = kernel_kwargs.pop('init_strategy', lambda site=None: init_to_median(site=site, num_samples=1000))
        step_size = kernel_kwargs.pop('step_size', 0.1)

        kernel = NUTS(self.model, target_accept_prob=target_accept_prob, init_strategy=init_strategy, 
                    step_size=step_size, **kernel_kwargs)

        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, **mcmc_kwargs)

        rng_key = random.PRNGKey(seed)
        mcmc.run(rng_key, *model_args, **model_kwargs)

        # TODO: warn diverging (and rhat if num_chains > 1)
        samples = mcmc.get_samples(group_by_chain=True)
        sample_stats = mcmc.get_extra_fields(group_by_chain=True)
        return mcmc
    
    def _get_samples(self, mcmc, group_by_chain=True):
        samples = mcmc.get_samples(group_by_chain=True)
        sample_stats = mcmc.get_extra_fields(group_by_chain=True) 
        return samples, sample_stats

    def sample_prior(self, *args, **kwargs):
        model = handlers.reparam(self.model.prior, self.model.prior_reparam)
        mcmc = self._sample(model, *args, **kwargs)

        samples, sample_stats = self._get_samples(mcmc)

        self.data.add_groups(
            {
                'prior': samples,
                'prior_sample_stats': sample_stats
            }
        ) 
        return mcmc

    def sample_posterior(self, *args, **kwargs):
        model = handlers.reparam(self.model.posterior, self.model.posterior_reparam)
        model = self.neural_transport.reparam(model)

        mcmc = self._sample(model, *args, **kwargs)

        samples, sample_stats = self._get_samples(mcmc)

        self.data.add_groups(
            {
                'posterior': samples,
                'sample_stats': sample_stats
            }
        )
        return mcmc

    def _predictive(self):
        pass

