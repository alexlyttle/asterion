"""inference.py
"""
from __future__ import annotations

import os

import numpy as np

from jax import random
import json
import numpyro

from numpyro import handlers as hdl
from numpyro.infer import MCMC, NUTS, init_to_median, Predictive, SVI
from numpyro.distributions import constraints
from numpyro.infer import svi

from collections import OrderedDict

import arviz as az
from numpyro.infer.svi import SVIRunResult

from .models import Model

from typing import Optional, Sequence, Dict, Union, List, Tuple

from .io import ModifiedNumPyroConverter

import warnings
import xarray
from numpyro.infer.reparam import CircularReparam

from astropy.table import Table
import astropy.units as u
import pandas as pd

__all__ = [
    "Results",
    "Inference",
]


class Results:
    """[summary]

    Args:
        data (arviz.InferenceData): description
        units (dict, optional): description
        circ_var_names (list, optional): 
    """
    def __init__(self, data: az.InferenceData, units: Dict[str, u.Unit]=None,
                 circ_var_names: Sequence[str]=None):
#         self._tables = tables
        self.data: az.InferenceData = data  #: Inference data.
        # self.summary = summary
        self.units: Dict[str, u.Unit] = units  #: Astropy units.
        self.circ_var_names: Sequence[str] = circ_var_names
        """: Circular variable names for calculating summary statistics."""

#         dim_vars = {}
#         for group in ['posterior', 'posterior_predictive']:
#             if group not in self.data.groups():
#                 continue
#             for k in self.data[group].data_vars.keys():
#                 dims = self.data[group][k].dims[2:]
#                 if dims not in dim_vars.keys():
#                     dim_vars[dims] = []
#                 if k not in dim_vars[dims]:
#                     dim_vars[dims].append(k)
#         self._dim_vars = dim_vars

    def _get_dim_vars(self, group: str='posterior') -> Dict[Tuple[str], List[str]]:
        """Get the dimensions and their variable names
        """
        dim_vars = {}
        for k in self.data[group].data_vars.keys():
            ns = 2  # Don't count chains and draws
            if group in ['observed_data', 'constant_data']:
                ns = 0  # These groups don't have chains and draws
            dims = self.data[group][k].dims[ns:]
            if dims not in dim_vars.keys():
                dim_vars[dims] = []
            if k not in dim_vars[dims]:
                dim_vars[dims].append(k)
        return dim_vars

    def get_dims(self, group: str='posterior') -> List[Tuple[str]]:
        """Get available dimension groups for a given inference data group.

        Args:
            group (str): Inference data group.

        Returns:
            list: [description]
        """
        # return list(self._dim_vars.keys())
        dim_vars = self._get_dim_vars(group=group)
        return list(dim_vars.keys())
    
    def get_var_names(self, group: str='posterior',
                      dims: Union[str, Tuple[str]]='all') -> List[str]:
        """Get var names for a given group and dimensions.
        
        Args:
            group (str): Inference data group.
            dims (str, or tuple of str): Dimensions by which to group variables. If 'all', returns
                variable names for all model dimensions. If a tuple of
                dimension names, returns variable names in that dimension
                group.
        
        Returns:
            list: Variable names for a given group and dimensions.
        """
        if dims == 'all':
            var_names = list(self.data[group].data_vars.keys())
        else:
            dim_vars = self._get_dim_vars(group=group)
            var_names = list(dim_vars[dims])
#             var_names = list(
#                 set(self.data[group].data_vars.keys()).intersection(set(self._dim_vars[dims]))
#             )
        return var_names

    def _validate_var_names(self, group: str='posterior',
                            var_names: Optional[List[str]]=None,
                            dims: Union[str, Tuple[str]]='all') -> List[str]:
        """Validate variable names."""
        available_vars = self.get_var_names(group=group, dims=dims)
        
        if var_names is None:
#             var_names = list(self.data[group].data_vars.keys())
            var_names = available_vars
            if len(var_names) == 0:
                if dims == 'all':
                    msg = f'No variables exist in group \'{group}\'.'
                else:
                    msg = f'No variables exist for dims {dims}' + \
                          f' in group \'{group}\'.'
                raise ValueError(msg)
        elif dims != 'all':
            subset = set(var_names)
            available = set(available_vars)
            if not available.intersection(subset) == subset:
                diff = subset.difference(available)
                raise ValueError(f'Variable name(s) {diff} not available' +
                                 f' for dims {dims} in group \'{group}\'.')
        return var_names

    def get_summary(self, group: str="posterior",
                    var_names: Optional[List[str]]=None,
                    **kwargs) -> Union[xarray.Dataset, pd.DataFrame]:
        """Get a summary of the inference data for a chosen group.

        Args:
            group (str): [description]. Defaults to 'posterior'.
            var_names (list, optional): [description]. Defaults to None (all variable names)
            **kwargs: Keyword arguments to pass to :func:`az.summary`.

        Returns:
            xarray.Dataset, or pandas.DataFrame: Summary of inference data.
        """
        fmt = kwargs.pop('fmt', 'xarray')
        round_to = kwargs.pop('round_to', 'none')
        stat_funcs = {
            "16th": lambda x: np.quantile(x, .16),
            "50th": np.median,
            "84th": lambda x: np.quantile(x, .84),
        }
        stat_funcs = kwargs.pop('stat_func', stat_funcs)
        
        var_names = self._validate_var_names(group=group, var_names=var_names)
        circ_var_names = [i for i in self.circ_var_names if i in var_names]
        circ_var_names = kwargs.pop(
            'circ_var_names',
            circ_var_names
        )
        return az.summary(self.data, group=group, var_names=var_names, fmt=fmt,
            round_to=round_to, stat_funcs=stat_funcs,
            circ_var_names=circ_var_names, **kwargs)

    def get_table(self, dims: Tuple[str], group: str='posterior',
                  var_names: Optional[List[str]]=None,
                  metrics: Optional[List[str]]=None, fmt: str='pandas',
                  round_to: Union[str, int]='auto',
                  **kwargs) -> Union[pd.DataFrame, Table]:
        """[summary]

        Args:
            dims (tuple of str): [description]
            group (str, optional): [description]. Defaults to 'posterior'.
            var_names (list of str, optional): [description]. Defaults to None.
            metrics (list of str, optional): [description]. Defaults to None.
            fmt (str, optional): [description]. Defaults to 'pandas'.
            round_to (str, or int, optional): [description]. Defaults to 'auto'.

        Returns:
            pandas.DataFrame, or astropy.table.Table]: [description]
        """

        var_names = self._validate_var_names(group=group, var_names=var_names, dims=dims)
        summary = self.get_summary(group=group, var_names=var_names, **kwargs)

        if metrics is None:
            metrics = ['mean', 'sd', '16th', '50th', '84th']

        keep_mcse = False
        if 'mcse_mean' in metrics:
            keep_mcse = True
        else:
            metrics.append('mcse_mean')

#         dim_vars = [i for i in self._dim_vars[dims] if i in var_names]
        table = summary[var_names].sel({'metric': metrics}).to_dataframe()
        if round_to == 'auto':
            # Rounds to the mcse_mean. I.e. if mcse_mean is in range (0.01, 0.1] then the
            # metrics are rounded to 2 decimal places
            level = None
            precision = np.log10(table.loc['mcse_mean']).astype(int) - 1
            if isinstance(table.index, pd.MultiIndex):
                level = 'metric'
                precision = precision.min()  # Choose min precision = max decimal precision
            if not keep_mcse:
                table = table.drop(index='mcse_mean', level=level)
            table = table.round(-precision)

        elif round_to != 'none':
            table = table.round(round_to)
        
        if fmt == 'astropy':
            units = {k: v for k, v in self.units.items() if k in var_names}
            table = Table.from_pandas(table.reset_index(), units=units)
        return table

    def save(self, path: str, **kwargs):
        """Save the results.

        Args:
            path (str): Path to directory in which to save the results.
        """
        self.data.to_netcdf(os.path.join(path, 'data.nc'), **kwargs)
        save_dict = {
            'units': {k: v.to_string() for k, v in self.units.items()},
            'circ_var_names': self.circ_var_names,
        }
        with open(os.path.join(path, 'config.json'), 'w') as fp:
            json.dump(save_dict, fp, indent=2)

    @staticmethod
    def load(path: str, **kwargs):
        """Load results.

        Args:
            path (str): Directory from which to load results.

        Returns:
            Results: Results object loaded from path.
        """
        data = az.from_netcdf(os.path.join(path, 'data.nc'), **kwargs)
        with open(os.path.join(path, 'config.json'), 'r') as fp:
            load_dict = json.load(fp)
        load_dict['units'] = {k: u.Unit(v) for k, v in load_dict['units'].items()}
        return Results(data, **load_dict)


class Inference:
    """[summary]

    Args:
        model (Model): [description]
        seed (int): [description]
    """ 
    def __init__(self, model: Model, *, seed: int):       
        self._rng_key = random.PRNGKey(seed)        
        self.model: Model = model  #:Model: Model with which to perform inference.
        self.samples: Optional[dict] = None  #:dict: Posterior MCMC samples.
        self.sample_stats: Optional[dict] = None  #:dict: Posterior MCMC sample stats.
        self.prior_predictive_samples: Optional[dict] = None  #:dict: Prior predictive samples.
        self.predictive_samples: Optional[dict] = None  #:dict: Posterior predictive samples.
        self.mcmc: Optional[MCMC] = None  #:numpyro.infer.MCMC: MCMC inference object.
        self._model_args: tuple = ()
        self._model_kwargs: dict = {}
        self.map_svi: Optional[SVI] = None  #:numpyro.infer.SVI: MAP inference object.
        self.map_result: Optional[svi.SVIRunResult] = None  #:numpyro.infer.svi.SVIRunResult: MAP result.

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

    def get_trace(self, model_args: tuple=(),
                  model_kwargs: dict={}) -> OrderedDict:
        """[summary]

        Args:
            model_args (tuple): [description]. Defaults to ().
            model_kwargs (dict): [description]. Defaults to {}.

        Returns:
            OrderedDict: [description]
        """
        rng_key, self._rng_key = random.split(self._rng_key)
        model = hdl.trace(
            hdl.seed(
                self.model, rng_key
            )
        )
        trace = model.get_trace(*model_args, **model_kwargs)
        return trace

    def get_circ_var_names(self) -> List[str]:
        """[summary]

        Returns:
            list: Circular variable names in the model.
        """
        var_names = []
        trace = self.get_trace(self._model_args, self._model_kwargs)
        for key, value in trace.items():
            if value['type'] == 'sample':
                if value['fn'].support is constraints.circular:
                    var_names.append(value['name'])
        return var_names
    
    def _auto_reparam(self) -> numpyro.handlers.reparam:
        var_names = self.get_circ_var_names()
        return hdl.reparam(config={k: CircularReparam() for k in var_names})

    def _init_handlers(self, handlers: list,
                       reparam: Union[str, hdl.reparam]='auto') -> list:
        # handlers = handlers.copy()
        if handlers is None:
            handlers = []
        if reparam == 'auto':
            handlers.append(self._auto_reparam())
        elif reparam == 'none':
            pass
        else:
            handlers.append(reparam)
        return handlers

    def get_sampler(self, method: str='NUTS', handlers: Optional[list]=None, 
                    reparam: Union[str, hdl.reparam]='auto', 
                    **kwargs) -> numpyro.infer.mcmc.MCMCKernel:
        """[summary]

        Args:
            method (str): Sampling method, choose from ['NUTS']. Defaults to 'NUTS'.
            handlers (list): List of handlers to apply to the model. Defaults to [].
            reparam (str, or numpyro.handlers.reparam): If 'auto', automatically reparameterises model. If 'none',
                no reparameterisation is done. If numpyro.handlers.reparam,
                this is applied instead.
            **kwargs: Keyword arguments to pass to sample initialisation.

        Raises:
            NotImplementedError: If a sampling method other than 'NUTS' is
                chosen.

        Returns:
            numpyro.infer.mcmc.MCMCKernel: MCMC sampler.
        """
        if method != 'NUTS':
            raise NotImplementedError(f"Method '{method}' not implemented")

        target_accept_prob = kwargs.pop("target_accept_prob", 0.99)
        init_strategy = kwargs.pop("init_strategy", lambda site=None: \
            init_to_median(site=site, num_samples=100))
        step_size = kwargs.pop("step_size", 0.1)
        
        # if reparam == 'auto':
        #     handlers.append(self._auto_reparam())
        # elif reparam == 'none':
        #     pass
        # else:
        #     handlers.append(reparam)
        handlers = self._init_handlers(handlers, reparam=reparam)

        model = self.model
        for h in handlers:
            model = h(model)

        sampler = NUTS(model, target_accept_prob=target_accept_prob,
                       init_strategy=init_strategy, step_size=step_size,
                       **kwargs)
        return sampler

    def init_mcmc(self, sampler: numpyro.infer.mcmc.MCMCKernel,
                  num_warmup: int=1000, num_samples: int=1000,
                  num_chains: int=1, **kwargs):
        """Initialises the MCMC.

        Args:
            sampler (numpyro.infer.mcmc.MCMCKernel): [description]
            num_warmup (int): [description]. Defaults to 1000.
            num_samples (int): [description]. Defaults to 1000.
            num_chains (int): [description]. Defaults to 1.
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
            model_args (tuple): Positional arguments to pass to model. Defaults to ().
            model_kwargs (dict): Keyword arguments to pass to model. Defaults to {}.
            extra_fields (tuple): Extra fields to report in sample_stats. Defaults to ().
            init_params (dict): Initial parameter values prior to sampling. Defaults to None.

        Returns:
            tuple: [description]
        """
        rng_key, self._rng_key = random.split(self._rng_key)
        self.mcmc.run(rng_key, *model_args, extra_fields=extra_fields,
                        init_params=init_params, **model_kwargs)
        self._update_args_kwargs(model_args, model_kwargs)
        # # Filter UserWarnings from numpyro as errors
        # module = r'\bnumpyro\b'
        # category = UserWarning
        # warnings.filterwarnings("error", module=module, category=category)
        # try:
        #     self.mcmc.run(rng_key, *model_args, extra_fields=extra_fields,
        #                   init_params=init_params, **model_kwargs)
        # except UserWarning as w:
        #     msg = w.args[0]
        #     if "CircularReparam" in msg:
        #         # TODO: make these more helpful
        #         msg = 'Add prior reparameteriser to the list of handlers.'
        #         raise RuntimeError(msg)
        #     raise w
        # # Reset warnings filter
        # warnings.filterwarnings("default", module=module, category=category)

        samples = self.mcmc.get_samples(group_by_chain=True)
        sample_stats = self.mcmc.get_extra_fields(group_by_chain=True)
        return samples, sample_stats

    def sample(self, num_warmup: int=1000, num_samples: int=1000,
                num_chains: int=1, model_args: tuple=(), model_kwargs: dict={}, 
                method: str='NUTS', handlers: Optional[list]=None,
                extra_fields: tuple=(), init_params: dict=None,
                sampler_kwargs: dict={}, mcmc_kwargs: dict={}):
        """[summary]

        Args:
            num_warmup (int): Number of warmup steps.
            num_samples (int): Number of samples after warmup.
            num_chains (int): Number of MCMC chains. If > 1, sampler will attempt
                to run in parallel on your device.
            model_args (tuple): Positional arguments to pass to the model callable.
            model_kwargs (dict): Keyword arguments to pass to the model callable.
            method (str): Sampler method, choose from ['NUTS'].
            handlers (list, optional): Handlers to apply to the model during inference.
            extra_fields (tuple): Extra fields to track in sample
                statistics. Defaults to ().
            init_params (dict, optional): Initial values for the parameters.
            sampler_kwargs (dict): Kwargs passed to :meth:`get_sampler`.
            mcmc_kwargs (dict): Kwargs to pass to :meth:`init_mcmc`.
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

    def find_map(self, num_steps: int=10000, model_args: tuple=(),
                 model_kwargs: dict={}, handlers: Optional[list]=None,
                 reparam: Union[str, hdl.reparam]='auto',
                 svi_kwargs: dict={}):
        """EXPERIMENTAL: find MAP.

        Args:
            num_steps (int): [description]. Defaults to 10000.
            model_args (tuple): [description]. Defaults to ().
            model_kwargs (dict): [description]. Defaults to {}.
            handlers (list, optional): [description]. Defaults to None.
            reparam (str, or numpyro.handlers.reparam): [description]. Defaults to 'auto'.
            svi_kwargs (dict): [description]. Defaults to {}.
        """
        handlers = self._init_handlers(handlers, reparam=reparam)
        model = self.model
        for h in handlers:
            model = h(model)

        guide = numpyro.infer.autoguide.AutoDelta(model)

        optim = svi_kwargs.pop('optim', numpyro.optim.Minimize())
        loss = svi_kwargs.pop('loss', numpyro.infer.Trace_ELBO())
        self.map_svi = SVI(model, guide, optim, loss=loss, **svi_kwargs)
        
        rng_key, self._rng_key = random.split(self._rng_key)
        self.map_result = self.map_svi.run(rng_key, num_steps, *model_args, **model_kwargs)
        self._update_args_kwargs(model_args, model_kwargs)

    def predictive(self, model_args: tuple=(), model_kwargs: dict={},
                   **kwargs) -> dict:
        """[summary]

        Args:
            model_args (tuple): Positional arguments to pass to the model callable.
            model_kwargs (dict): Keyword arguments to pass to the model callable.
            **kwargs: Kwargs to pass to Predictive.

        Returns:
            dict: [description]
        """
        posterior_samples = kwargs.pop("posterior_samples", {})
        num_samples = kwargs.pop("num_samples", None)
        # batch_ndims = kwargs.pop("batch_ndims", 2)
        return_sites = kwargs.pop("return_sites", None)
        if return_sites is None:
            trace = self.get_trace(model_args, model_kwargs)
            return_sites = []
            for k, site in trace.items():
                # Only return non-observed sample sites not in samples and
                # all deterministic sites.
                if (site["type"] == "sample"):
                    if not site["is_observed"] and k not in posterior_samples:
                        return_sites.append(k)
                elif (site["type"] == "deterministic"):
                    return_sites.append(k)
                
        predictive = Predictive(self.model, posterior_samples=posterior_samples, 
                                num_samples=num_samples, return_sites=return_sites, 
                                **kwargs)
        
        if predictive.batch_ndims == 0:
            # Fix bug in Predictive for computing batch shape
            predictive._batch_shape = ()

        rng_key, self._rng_key = random.split(self._rng_key)
        samples = predictive(rng_key, *model_args, **model_kwargs)
        self._update_args_kwargs(model_args, model_kwargs)
        return samples
    
    def prior_predictive(self, num_samples: int=1000, model_args: tuple=(),
                         model_kwargs: dict={}, **kwargs):
        """[summary]

        Args:
            num_samples (int): Number of samples to take from the prior.
            model_args (tuple): Positional arguments to pass to the model callable.
            model_kwargs (dict): Keyword arguments to pass to the model callable.
        """
        self.prior_predictive_samples = self.predictive(
            model_args=model_args, model_kwargs=model_kwargs,
            num_samples=num_samples, **kwargs
        )

    def posterior_predictive(self, model_args: tuple=(), model_kwargs: dict={},
                             **kwargs):
        """[summary]

        Args:
            model_args (tuple): Positional arguments to pass to the model callable.
            model_kwargs (dict): Keyword arguments to pass to the model callable.
        """
        batch_ndims = 1
        if self.mcmc.num_chains > 1:
            batch_ndims = 2
        self.predictive_samples = self.predictive(
            model_args=model_args, model_kwargs=model_kwargs,
            posterior_samples=self.samples, batch_ndims=batch_ndims, **kwargs
        )

    def map_predictive(self, model_args: tuple=(), model_kwargs: dict={},
                       **kwargs) -> xarray.Dataset:
        """EXPERIMENTAL: Get predictive from MAP.

        Args:
            model_args (tuple): [description]. Defaults to ().
            model_kwargs (dict): [description]. Defaults to {}.

        Returns:
            xarray.Dataset: [description]
        """
        batch_ndims = 0
        guide = self.map_svi.guide
        params = self.map_result.params
        map_pred = self.predictive(
            model_args=model_args, model_kwargs=model_kwargs,
            guide=guide, params=params, num_samples=1, batch_ndims=batch_ndims,
            **kwargs
        )
        trace = self.get_trace(self._model_args, self._model_kwargs)
        dims, coords = self._get_dims(trace)
        ds = {}
        for k, v in map_pred.items():
            ds[k] = xarray.DataArray(v, dims=dims.get(k))
        return xarray.Dataset(ds, coords=coords)

    def get_results(self) -> Results:
        """Get inference results.

        Returns:
            Results: Inference results.
        """
        data = self.get_data()
        circ_var_names = self.get_circ_var_names()
        return Results(data, units=self.model.units, circ_var_names=circ_var_names)
 
    def get_data(self) -> az.InferenceData:
        """Get inference data.

        Returns:
            arviz.InferenceData: Inference data.
        """
        trace = self.get_trace(self._model_args, self._model_kwargs)
        dims, coords = self._get_dims(trace)
        data = ModifiedNumPyroConverter(posterior=self.mcmc, prior=self.prior_predictive_samples,
                        posterior_predictive=self.predictive_samples,
                        coords=coords, dims=dims).to_inference_data()
        # data = az.from_numpyro(posterior=self.mcmc, prior=self.prior_predictive_samples,
        #                 posterior_predictive=self.predictive_samples, constant_data=constant_data,
        #                 coords=coords, dims=dims)
        if 'observed_data' not in data:
            # Check for observed data in trace
            observed_data = {}
            for k, site in trace.items():
                if site.get('is_observed', False):
                    observed_data[k] = site['value']
            if len(observed_data) > 0:
                data.add_group(observed_data=observed_data)
        return data
