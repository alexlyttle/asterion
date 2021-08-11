"""[summary]
"""
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

from .data import Data
from .models import Model

from typing import Callable


__all__ = [
    "Inference"
]


class Inference:
    """[summary]

    Note:
        [note]

    Args:
        model: [description]
        seed: [description]
        num_warmup: [description], by default 1000.
        num_samples: [description], by default 1000.
        num_chains: [description], by default 5.

    See Also:
        [see also]
    
    Warning:
        [warning]

    Hint:
        [hint]
    """ 
    def __init__(self, model: Model, num_warmup: int=1000,
                 num_samples: int=1000, num_chains: int=5, *, seed: int):       
        self._rng_key = random.PRNGKey(seed)        
        self.model: Model = model  #: [description]
        
        observed = {"nu": self.model.nu}
        constant = {"nu_err": self.model.nu_err, "obs_mask": self.model.obs_mask,
                    "circ_var_names": self.model.circ_var_names}
        coords = {k: v.coords for k, v in self.model.dimensions.items()}
        dims = {}
        trace = self.model.get_posterior_trace(self._rng_key)
        for k, v in trace.items():
            dims[k] = [dim.name for dim in v["cond_indep_stack"][::-1]]

        self._data: Data = Data(
            observed=observed, 
            constant=constant, 
            coords=coords, 
            dims=dims
        )  #: [description]

        self._num_warmup = num_warmup
        self._num_samples = num_samples
        self._num_chains = num_chains
    
    def _sample(self, model: Callable, extra_fields: tuple=(),
                init_params: dict=None, kernel_kwargs: dict={},
                mcmc_kwargs: dict={}) -> numpyro.infer.MCMC:
        """[summary]

        Args:
            model: [description]
            extra_fields: [description]. Defaults to ().
            init_params: [description]. Defaults to None.
            kernel_kwargs: [description]. Defaults to {}.
            mcmc_kwargs: [description]. Defaults to {}.

        Raises:
            KeyError: [description]

        Returns:
            [description]
        """                           
        target_accept_prob = kernel_kwargs.pop("target_accept_prob", 0.99)
        init_strategy = kernel_kwargs.pop("init_strategy", lambda site=None: \
            init_to_median(site=site, num_samples=1000))
        step_size = kernel_kwargs.pop("step_size", 0.1)
        
        forbidden_keys = ["num_warmup", "num_samples", "num_chains"]
        forbidden = [k for k in forbidden_keys if k in mcmc_kwargs.keys()]
        if len(forbidden) > 0:
            raise KeyError(
                f"Keys {forbidden} found in 'mcmc_kwargs' are already defined."
            )

        kernel = NUTS(model, target_accept_prob=target_accept_prob,
                      init_strategy=init_strategy, step_size=step_size,
                      **kernel_kwargs)

        mcmc = MCMC(kernel, num_warmup=self._num_warmup,
                    num_samples=self._num_samples, num_chains=self._num_chains,
                    **mcmc_kwargs)

        rng_key, self._rng_key = random.split(self._rng_key)
        mcmc.run(rng_key, extra_fields=extra_fields, init_params=init_params)

        # TODO: warn diverging (and rhat if num_chains > 1)
        # samples = mcmc.get_samples(group_by_chain=True)
        # sample_stats = mcmc.get_extra_fields(group_by_chain=True)
        return mcmc
    
    def _get_samples(self, mcmc: numpyro.infer.MCMC, group_by_chain: bool=True):
        samples = mcmc.get_samples(group_by_chain=True)
        sample_stats = mcmc.get_extra_fields(group_by_chain=True) 
        return samples, sample_stats

    def sample(self, *args, **kwargs):
        """[summary]

        Args:
            extra_fields (Optional[tuple]): [description], by default ().
            init_params (Optional[dict]): [description], by default None.
            kernel_kwargs (Optional[dict]): [description], by default {}.
            mcmc_kwargs (Optional[dict]): [description], by default {}.
        """  
        model = handlers.reparam(self.model.posterior, self.model.reparam)
        # model = self.neural_transport.reparam(model)

        # with warnings.catch_warnings():
            # warnings.filterwarnings("ignore", category=UserWarning, module=r"\bnumpyro\b")
        mcmc = self._sample(model, *args, **kwargs)

        samples, sample_stats = self._get_samples(mcmc)

        self._data.posterior = samples
        self._data.sample_stats = sample_stats

    def _predictive(self, model: Callable, predictive_kwargs: dict={}):
        """[summary]

        Args:
            model: [description]
            predictive_kwargs: [description], by default {}.
        Returns:
            [description]
        """
        posterior_samples = predictive_kwargs.pop("posterior_samples", None)
        num_samples = predictive_kwargs.pop("num_samples", None)
        batch_ndims = predictive_kwargs.pop("batch_ndims", 2)

        if posterior_samples is None and num_samples is None:
            num_samples = self._num_chains * self._num_samples

        predictive = Predictive(model, posterior_samples=posterior_samples, 
                                num_samples=num_samples, batch_ndims=batch_ndims,
                                **predictive_kwargs)

        rng_key, self._rng_key = random.split(self._rng_key)
        samples = predictive(rng_key)
        return samples
    
    def prior_predictive(self):
        """[summary]
        """
        samples = self._predictive(self.model.prior)
        self._data.prior_predictive = samples
    
    def posterior_predictive(self):
        """[summary]
        """
        predictive_kwargs = {"posterior_samples": self._data.posterior}
        samples = self._predictive(self.model.posterior, predictive_kwargs=predictive_kwargs)
        self._data.posterior_predictive = samples

    def predictions(self):
        predictive_kwargs = {"posterior_samples": self._data.posterior}
        samples = self._predictive(self.model.predictions, predictive_kwargs=predictive_kwargs)
        self._data.predictions = samples

        # Pred coords and dims - this should be a function
        coords = {k: v.coords for k, v in self.model.dimensions.items()}
        coords["chain"] = np.arange(self._num_chains)
        coords["draw"] = np.arange(self._num_samples)
        dims = {}
        trace = self.model.get_predictions_trace(self._rng_key)
        for k, v in trace.items():
            dims[k] = ["chain", "draw"]
            dims[k] += [dim.name for dim in v["cond_indep_stack"][::-1]]
        self._data.pred_coords = coords
        self._data.pred_dims = dims

    def summary(self, group="posterior"):
        stat_funcs = {
            "16th": lambda x: np.quantile(x, .16),
            "50th": np.median,
            "84th": lambda x: np.quantile(x, .84),
        }
        return az.summary(self.data, group=group, fmt="xarray", round_to="none", 
            stat_funcs=stat_funcs, circ_var_names=self.model.circ_var_names)

    @property
    def data(self) -> az.InferenceData:
        return self._data.to_arviz()