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
from numpyro.contrib.nested_sampling import NestedSampler

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

def _get_dim_vars(data, group: str='posterior') -> Dict[Tuple[str], List[str]]:
    """Get the dimensions and their variable names
    """
    dim_vars = {}
    for k in data[group].data_vars.keys():
        ns = 2  # Don't count chains and draws
        if group in ['observed_data', 'constant_data']:
            ns = 0  # These groups don't have chains and draws
        dims = data[group][k].dims[ns:]
        if dims not in dim_vars.keys():
            dim_vars[dims] = []
        if k not in dim_vars[dims]:
            dim_vars[dims].append(k)
    return dim_vars

def get_dims(data, group: str='posterior') -> List[Tuple[str]]:
    """Get available dimension groups for a given inference data group.

    Args:
        group (str): Inference data group.

    Returns:
        list: [description]
    """
    # return list(self._dim_vars.keys())
    dim_vars = _get_dim_vars(data, group=group)
    return list(dim_vars.keys())

def get_var_names(data, group: str='posterior',
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
        var_names = list(data[group].data_vars.keys())
    else:
        dim_vars = _get_dim_vars(data, group=group)
        var_names = list(dim_vars[dims])
#             var_names = list(
#                 set(self.data[group].data_vars.keys()).intersection(set(self._dim_vars[dims]))
#             )
    return var_names

def _validate_var_names(data, group: str='posterior',
                        var_names: Optional[List[str]]=None,
                        dims: Union[str, Tuple[str]]='all') -> List[str]:
    """Validate variable names."""
    available_vars = get_var_names(data, group=group, dims=dims)
    
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

def get_summary(data, group: str="posterior",
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
    kind = kwargs.pop('kind', 'stats')  # default just stats, no diagnostics
    
    var_names = _validate_var_names(data, group=group, var_names=var_names)

    # self.data[group]
    circ_var_names = [k for k in var_names if data[group][k].attrs.get('is_circular', 0) == 1]
    # circ_var_names = [i for i in self.circ_var_names if i in var_names]
    circ_var_names = kwargs.pop(
        'circ_var_names',
        circ_var_names
    )
    return az.summary(data, group=group, var_names=var_names, fmt=fmt,
        round_to=round_to, stat_funcs=stat_funcs,
        circ_var_names=circ_var_names, kind=kind, **kwargs)

def get_table(data, dims: Tuple[str], group: str='posterior',
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

    var_names = _validate_var_names(data, group=group, var_names=var_names, dims=dims)
    summary = get_summary(data, group=group, var_names=var_names, **kwargs)

    if metrics is None:
        metrics = ['mean', 'sd', '16th', '50th', '84th']

    # keep_mcse = False
    # if 'mcse_mean' in metrics:
    #     keep_mcse = True
    # else:
    #     metrics.append('mcse_mean')

#         dim_vars = [i for i in self._dim_vars[dims] if i in var_names]
    table = summary[var_names].sel({'metric': metrics}).to_dataframe()
    if round_to == 'auto':
        # Rounds to the mcse_mean. I.e. if mean_err is in range (0.01, 0.1] then the
        # metrics are rounded to 2 decimal places
        # level = None
        if 'sd' not in metrics:
            raise ValueError('Automatic rounding requires \'sd\' in \'metrics\'.')
        mean_err = table.loc['sd'] / np.sqrt(data[group].draw.size)
        precision = np.log10(mean_err).astype(int) - 1
        if isinstance(table.index, pd.MultiIndex):
            # level = 'metric'
            precision = precision.min()  # Choose min precision = max decimal precision
        # if not keep_mcse:
            # table = table.drop(index='mcse_mean', level=level)
        table = table.round(-precision)

    elif round_to != 'none':
        table = table.round(round_to)
    
    if fmt == 'astropy':
        # units = {k: v for k, v in self.units.items() if k in var_names}
        units = {k: u.Unit(data[group][k].attrs.get('unit', '')) for k in var_names}
        table = Table.from_pandas(table.reset_index(), units=units)
    return table
