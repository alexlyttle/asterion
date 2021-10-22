"""[summary]
"""
import math

import numpy as np

import jax
import jax.numpy as jnp

from jax import random

import numpyro
import numpyro.distributions as dist

from numpyro.infer.reparam import Reparam
from numpyro import handlers as hdl
from numpyro.infer import MCMC, NUTS, init_to_median, Predictive
from numpyro.distributions import constraints

from collections import namedtuple, OrderedDict

import warnings

import arviz as az

from .data import Data
from .models import Model

from typing import Callable, Optional

from .data import ModifiedNumPyroConverter

import warnings
import xarray
from numpyro.infer.reparam import CircularReparam


__all__ = [
    "Inference"
]


# class Inference:
#     """[summary]

#     Note:
#         [note]

#     Args:
#         model: [description]
#         seed: [description]
#         num_warmup: [description], by default 1000.
#         num_samples: [description], by default 1000.
#         num_chains: [description], by default 5.

#     See Also:
#         [see also]
    
#     Warning:
#         [warning]

#     Hint:
#         [hint]
#     """ 
#     def __init__(self, model: Model, num_warmup: int=1000,
#                  num_samples: int=1000, num_chains: int=5, *, seed: int):       
#         self._rng_key = random.PRNGKey(seed)        
#         self.model: Model = model  #: [description]
        
#         observed = {"nu": self.model.nu}
#         constant = {"nu_err": self.model.nu_err, "obs_mask": self.model.obs_mask,
#                     "circ_var_names": self.model.circ_var_names}
#         coords = {k: v.coords for k, v in self.model.dimensions.items()}
#         dims = {}
#         trace = self.model.get_posterior_trace(self._rng_key)
#         for k, v in trace.items():
#             dims[k] = [dim.name for dim in v["cond_indep_stack"][::-1]]

#         self._data: Data = Data(
#             observed=observed, 
#             constant=constant, 
#             coords=coords, 
#             dims=dims
#         )  #: [description]

#         self._num_warmup = num_warmup
#         self._num_samples = num_samples
#         self._num_chains = num_chains
    
#     def _sample(self, model: Callable, extra_fields: tuple=(),
#                 init_params: dict=None, kernel_kwargs: dict={},
#                 mcmc_kwargs: dict={}) -> numpyro.infer.MCMC:
#         """[summary]

#         Args:
#             model: [description]
#             extra_fields: [description]. Defaults to ().
#             init_params: [description]. Defaults to None.
#             kernel_kwargs: [description]. Defaults to {}.
#             mcmc_kwargs: [description]. Defaults to {}.

#         Raises:
#             KeyError: [description]

#         Returns:
#             [description]
#         """                           
#         target_accept_prob = kernel_kwargs.pop("target_accept_prob", 0.99)
#         init_strategy = kernel_kwargs.pop("init_strategy", lambda site=None: \
#             init_to_median(site=site, num_samples=1000))
#         step_size = kernel_kwargs.pop("step_size", 0.1)
        
#         forbidden_keys = ["num_warmup", "num_samples", "num_chains"]
#         forbidden = [k for k in forbidden_keys if k in mcmc_kwargs.keys()]
#         if len(forbidden) > 0:
#             raise KeyError(
#                 f"Keys {forbidden} found in 'mcmc_kwargs' are already defined."
#             )

#         kernel = NUTS(model, target_accept_prob=target_accept_prob,
#                       init_strategy=init_strategy, step_size=step_size,
#                       **kernel_kwargs)

#         mcmc = MCMC(kernel, num_warmup=self._num_warmup,
#                     num_samples=self._num_samples, num_chains=self._num_chains,
#                     **mcmc_kwargs)

#         rng_key, self._rng_key = random.split(self._rng_key)
#         mcmc.run(rng_key, extra_fields=extra_fields, init_params=init_params)

#         # TODO: warn diverging (and rhat if num_chains > 1)
#         # samples = mcmc.get_samples(group_by_chain=True)
#         # sample_stats = mcmc.get_extra_fields(group_by_chain=True)
#         return mcmc
    
#     def _get_samples(self, mcmc: numpyro.infer.MCMC, group_by_chain: bool=True):
#         samples = mcmc.get_samples(group_by_chain=True)
#         sample_stats = mcmc.get_extra_fields(group_by_chain=True) 
#         return samples, sample_stats

#     def sample(self, *args, **kwargs):
#         """[summary]

#         Args:
#             extra_fields (Optional[tuple]): [description], by default ().
#             init_params (Optional[dict]): [description], by default None.
#             kernel_kwargs (Optional[dict]): [description], by default {}.
#             mcmc_kwargs (Optional[dict]): [description], by default {}.
#         """  
#         model = hdl.reparam(self.model.posterior, self.model.reparam)
#         # model = self.neural_transport.reparam(model)

#         # with warnings.catch_warnings():
#             # warnings.filterwarnings("ignore", category=UserWarning, module=r"\bnumpyro\b")
#         mcmc = self._sample(model, *args, **kwargs)

#         samples, sample_stats = self._get_samples(mcmc)

#         self._data.posterior = samples
#         self._data.sample_stats = sample_stats

#     def _predictive(self, model: Callable, predictive_kwargs: dict={}):
#         """[summary]

#         Args:
#             model: [description]
#             predictive_kwargs: [description], by default {}.
#         Returns:
#             [description]
#         """
#         posterior_samples = predictive_kwargs.pop("posterior_samples", None)
#         num_samples = predictive_kwargs.pop("num_samples", None)
#         batch_ndims = predictive_kwargs.pop("batch_ndims", 2)

#         if posterior_samples is None and num_samples is None:
#             num_samples = self._num_chains * self._num_samples

#         predictive = Predictive(model, posterior_samples=posterior_samples, 
#                                 num_samples=num_samples, batch_ndims=batch_ndims,
#                                 **predictive_kwargs)

#         rng_key, self._rng_key = random.split(self._rng_key)
#         samples = predictive(rng_key)
#         return samples
    
#     def prior_predictive(self):
#         """[summary]
#         """
#         samples = self._predictive(self.model.prior)
#         self._data.prior_predictive = samples
    
#     def posterior_predictive(self):
#         """[summary]
#         """
#         predictive_kwargs = {"posterior_samples": self._data.posterior}
#         samples = self._predictive(self.model.posterior, predictive_kwargs=predictive_kwargs)
#         self._data.posterior_predictive = samples

#     def predictions(self):
#         predictive_kwargs = {"posterior_samples": self._data.posterior}
#         samples = self._predictive(self.model.predictions, predictive_kwargs=predictive_kwargs)
#         self._data.predictions = samples

#         # Pred coords and dims - this should be a function
#         coords = {k: v.coords for k, v in self.model.dimensions.items()}
#         coords["chain"] = np.arange(self._num_chains)
#         coords["draw"] = np.arange(self._num_samples)
#         dims = {}
#         trace = self.model.get_predictions_trace(self._rng_key)
#         for k, v in trace.items():
#             dims[k] = ["chain", "draw"]
#             dims[k] += [dim.name for dim in v["cond_indep_stack"][::-1]]
#         self._data.pred_coords = coords
#         self._data.pred_dims = dims

#     def summary(self, group="posterior"):
#         stat_funcs = {
#             "16th": lambda x: np.quantile(x, .16),
#             "50th": np.median,
#             "84th": lambda x: np.quantile(x, .84),
#         }
#         return az.summary(self.data, group=group, fmt="xarray", round_to="none", 
#             stat_funcs=stat_funcs, circ_var_names=self.model.circ_var_names)

#     @property
#     def data(self) -> az.InferenceData:
#         return self._data.to_arviz()


class Inference:
    """[summary]

    Note:
        [note]

    Args:
        model: [description]
        seed: [description]

    See Also:
        [see also]
    
    Warning:
        [warning]

    Hint:
        [hint]
    """ 
    def __init__(self, model: Model, *, seed: int):       
        self._rng_key = random.PRNGKey(seed)        
        self.model: Model = model  #: [description]
        self.samples = None
        self.sample_stats = None
        self.prior_predictive_samples = None
        self.predictive_samples = None
        self.mcmc = None
        self._model_args = ()
        self._model_kwargs = {}

    def _get_dims(self, trace):
        coords = {}
        dims = {}
        for k, v in trace.items():
            name = v["name"]
            if v["type"] == "dimension":
                coords[name] = v["value"]
            elif "dims" in v.keys():
                dims[name] = v["dims"]
        return dims, coords

    def get_trace(self, model_args: tuple=(), model_kwargs: dict={}) -> OrderedDict:
        """[summary]

        Args:
            model_args: [description]. Defaults to ().
            model_kwargs: [description]. Defaults to {}.

        Returns:
            [description]
        """
        rng_key, self._rng_key = random.split(self._rng_key)
        model = hdl.trace(
            hdl.seed(
                self.model, rng_key
            )
        )
        trace = model.get_trace(*model_args, **model_kwargs)
        return trace

    def get_circ_var_names(self) -> list:
        """[summary]

        Returns:
            Circular variable names in the model
        """
        var_names = []
        trace = self.get_trace(self._model_args, self._model_kwargs)
        for key, value in trace.items():
            if value['type'] == 'sample':
                if value['fn'].support is constraints.circular:
                    var_names.append(value['name'])
        return var_names
    
    def auto_reparam(self) -> numpyro.handlers.reparam:
        var_names = self.get_circ_var_names()
        return hdl.reparam(config={k: CircularReparam() for k in var_names})

    def get_sampler(self, method: str='NUTS', handlers: list=[], 
                    reparam='auto', **kwargs):
        """[summary]

        Args:
            method: Sampling method, choose from ['NUTS']. Defaults to 'NUTS'.
            handlers: List of handlers to apply to the model. Defaults to [].
            reparam: If 'auto', automatically reparameterises model. If 'none',
                no reparameterisation is done. If numpyro.handlers.reparam,
                this is applied instead.
            **kwargs: Keyword arguments to pass to sample initialisation.

        Raises:
            NotImplementedError: If a sampling method other than 'NUTS' is
                chosen.

        Returns:
            MCMC sampler.
        """
        if method != 'NUTS':
            raise NotImplementedError(f"Method '{method}' not implemented")

        target_accept_prob = kwargs.pop("target_accept_prob", 0.99)
        init_strategy = kwargs.pop("init_strategy", lambda site=None: \
            init_to_median(site=site, num_samples=100))
        step_size = kwargs.pop("step_size", 0.1)
        
        if reparam == 'auto':
            handlers.append(self.auto_reparam())
        elif reparam == 'none':
            pass
        else:
            handlers.append(reparam)

        model = self.model
        for h in handlers:
            model = h(model)

        sampler = NUTS(model, target_accept_prob=target_accept_prob,
                       init_strategy=init_strategy, step_size=step_size,
                       **kwargs)
        return sampler

    def init_mcmc(self, sampler: numpyro.infer.mcmc.MCMCKernel, num_warmup: int=1000,
                  num_samples: int=1000, num_chains: int=1, **kwargs):
        """Initialises the MCMC.

        Args:
            sampler: [description]
            num_warmup: [description]. Defaults to 1000.
            num_samples: [description]. Defaults to 1000.
            num_chains: [description]. Defaults to 1.
        """
        self.mcmc = MCMC(sampler, num_warmup=num_warmup,
                    num_samples=num_samples, num_chains=num_chains,
                    **kwargs)
    
    def _update_args_kwargs(self, model_args: tuple, model_kwargs: dict):
        self._model_args = model_args
        self._model_kwargs.update(model_kwargs)

    def run_mcmc(self, model_args: tuple=(), model_kwargs: dict={}, 
                 extra_fields: tuple=(), init_params: dict=None) -> tuple:
        """Runs MCMC for a given set of model arguments.

        Args:
            model_args: Positional arguments to pass to model. Defaults to ().
            model_kwargs: Keyword arguments to pass to model. Defaults to {}.
            extra_fields: Extra fields to report in sample_stats. Defaults to ().
            init_params: Initial parameter values prior to sampling. Defaults to None.

        Raises:
            RuntimeError: If
            UserWarning: if numpyro throws a UserWarning

        Returns:
            [description]
        """
        self._update_args_kwargs(model_args, model_kwargs)
        rng_key, self._rng_key = random.split(self._rng_key)

        # Filter UserWarnings from numpyro as errors
        module = r'\bnumpyro\b'
        category = UserWarning
        warnings.filterwarnings("error", module=module, category=category)
        try:
            self.mcmc.run(rng_key, *model_args, extra_fields=extra_fields,
                          init_params=init_params, **model_kwargs)
        except UserWarning as w:
            msg = w.args[0]
            if "CircularReparam" in msg:
                # TODO: make these more helpful
                msg = 'Add prior reparameteriser to the list of handlers.'
                raise RuntimeError(msg)
            raise w
        # Reset warnings filter
        warnings.filterwarnings("default", module=module, category=category)

        samples = self.mcmc.get_samples(group_by_chain=True)
        sample_stats = self.mcmc.get_extra_fields(group_by_chain=True)
        return samples, sample_stats

    def sample(self, num_warmup: int=1000, num_samples: int=1000,
                num_chains: int=1, model_args: tuple=(), model_kwargs: dict={}, 
                method: str='NUTS', handlers: list=[], extra_fields: tuple=(),
                init_params: dict=None, sampler_kwargs: dict={},
                mcmc_kwargs: dict={}) -> numpyro.infer.MCMC:
        """[summary]

        Args:
            *args: [description]
            extra_fields: [description]. Extra fields to track in sample
                statistics. Defaults to ().
            init_params: [description]. Initial values for the parameters.
                Defaults to None.
            sampler_kwargs: Kwargs passed to sample init. Defaults to {}.
            mcmc_kwargs: Kwargs to pass to MCMC init. Defaults to {}.

        Raises:
            KeyError: [description]

        Returns:
            [description]
        """                           
        sampler = self.get_sampler(method=method, handlers=handlers,
                                   **sampler_kwargs)

        self.init_mcmc(sampler, num_warmup=num_warmup, num_samples=num_samples,
                       num_chains=num_chains, **mcmc_kwargs)

        self.samples, self.sample_stats = self.run_mcmc(
            model_args=model_args, model_kwargs=model_kwargs,
            extra_fields=extra_fields, init_params=init_params,
        )

        # TODO: warn diverging (and rhat if num_chains > 1)
        # samples = mcmc.get_samples(group_by_chain=True)
        # sample_stats = mcmc.get_extra_fields(group_by_chain=True)
        return self.get_summary()

    def predictive(self, model_args=(), model_kwargs={}, **kwargs) -> dict:
        """[summary]

        Args:
            model_args: Model arguments.
            model_kwargs: Model keyword arguments.
            **kwargs: Kwargs to pass to Predictive.
        Returns:
            [description]
        """
        self._update_args_kwargs(model_args, model_kwargs)
        posterior_samples = kwargs.pop("posterior_samples", None)
        num_samples = kwargs.pop("num_samples", None)
        # batch_ndims = kwargs.pop("batch_ndims", 2)

        predictive = Predictive(self.model, posterior_samples=posterior_samples, 
                                num_samples=num_samples, **kwargs)

        rng_key, self._rng_key = random.split(self._rng_key)
        samples = predictive(rng_key, *model_args, **model_kwargs)
        return samples
    
    def prior_predictive(self, num_samples: int=1000, model_args: tuple=(),
                         model_kwargs: dict={}, **kwargs):
        """[summary]

        Args:
            num_samples: [description]. Defaults to 1000.
            model_args: [description]. Defaults to ().
            model_kwargs: [description]. Defaults to {}.
        """
        self.prior_predictive_samples = self.predictive(
            model_args=model_args, model_kwargs=model_kwargs,
            num_samples=num_samples, **kwargs
        )

    def posterior_predictive(self, model_args: tuple=(), model_kwargs: dict={},
                             **kwargs):
        """[summary]

        Args:
            model_args: [description]. Defaults to ().
            model_kwargs: [description]. Defaults to {}.
        """
        batch_ndims = 1
        if self.mcmc.num_chains > 1:
            batch_ndims = 2
        self.predictive_samples = self.predictive(
            model_args=model_args, model_kwargs=model_kwargs,
            posterior_samples=self.samples, batch_ndims=batch_ndims, **kwargs
        )

    def get_summary(self, group: str="posterior", var_names: Optional[list]=None,
                    **kwargs) -> xarray.Dataset:
        """Get a summary of the inference for a chosen group.

        Args:
            group: [description]. Defaults to "posterior".
            var_names: [description]. Defaults to None (all variable names)
            **kwargs: Keyword arguments to pass to az.summary.
        Returns:
            Summary of inference data.
        """
        data = self.get_data()
        fmt = kwargs.pop('fmt', 'xarray')
        round_to = kwargs.pop('round_to', 'none')
        stat_funcs = {
            "16th": lambda x: np.quantile(x, .16),
            "50th": np.median,
            "84th": lambda x: np.quantile(x, .84),
        }
        stat_funcs = kwargs.pop('stat_func', stat_funcs)
        circ_var_names = kwargs.pop(
            'circ_var_names',
            self.get_circ_var_names()
        )
        return az.summary(data, group=group, var_names=var_names, fmt=fmt,
            round_to=round_to, stat_funcs=stat_funcs,
            circ_var_names=circ_var_names, **kwargs)

    def get_data(self) -> az.InferenceData:
        """[summary]

        Returns:
            Inference data.
        """
        trace = self.get_trace(self._model_args, self._model_kwargs)
        constant_data = {k: v for k, v in self._model_kwargs.items() if k not in trace.keys() and v is not None}

        dims, coords = self._get_dims(trace)

        data = ModifiedNumPyroConverter(posterior=self.mcmc, prior=self.prior_predictive_samples,
                        posterior_predictive=self.predictive_samples, constant_data=constant_data,
                        coords=coords, dims=dims).to_inference_data()
        # data = az.from_numpyro(posterior=self.mcmc, prior=self.prior_predictive_samples,
        #                 posterior_predictive=self.predictive_samples, constant_data=constant_data,
        #                 coords=coords, dims=dims)
        return data
