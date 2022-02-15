"""The results module contains functions for inspecting, summarising, and 
tabulating inference data.
"""
from __future__ import annotations

import xarray
import numpy as np
import arviz as az
import astropy.units as u
import pandas as pd

from typing import Optional, Dict, Union, List, Tuple
from astropy.table import Table

def _get_dim_vars(data, group: str='posterior') -> Dict[Tuple[str], List[str]]:
    """Get the dimensions and their variable names.
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
        data (arviz.InferenceData): Inference data object.
        group (str): Inference data group.

    Returns:
        list of tuple: [description]
    """
    # return list(self._dim_vars.keys())
    
    dim_vars = _get_dim_vars(data, group=group)
    return list(dim_vars.keys())

def get_var_names(data, group: str='posterior',
                    dims: Union[str, Tuple[str]]='all') -> List[str]:
    """Get var names for a given group and dimensions.
    
    Args:
        data (arviz.InferenceData): Inference data object.
        group (str): Inference data group.
        dims (str, or tuple of str): Dimensions by which to group variables. 
            If 'all', returns variable names for all model dimensions. If a
            tuple of dimension names, returns variable names in that dimension
            group.
    
    Returns:
        list of str: Variable names for a given group and dimensions.
    """
    if dims == 'all':
        var_names = list(data[group].data_vars.keys())
    else:
        dim_vars = _get_dim_vars(data, group=group)
        var_names = list(dim_vars[dims])
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
        data (arviz.InferenceData): Inference data object.
        group (str, optional): [description]. Defaults to 'posterior'.
        var_names (list, optional): [description]. Defaults to None (all 
            variable names)
        **kwargs: Keyword arguments to pass to :func:`arviz.summary`.

    Returns:
        xarray.Dataset, or pandas.DataFrame: Summary of inference data.
    
    See Also:
        :func:`arviz.summary`: The function for which this wraps.
    """
    fmt = kwargs.pop('fmt', 'xarray')
    round_to = kwargs.pop('round_to', 'none')
    stat_funcs = {
        'mean': np.mean,
        'sd': lambda x: np.std(x, ddof=1),
        "16th": lambda x: np.quantile(x, .16),
        "50th": np.median,
        "84th": lambda x: np.quantile(x, .84),
    }
    stat_funcs = kwargs.pop('stat_funcs', stat_funcs)
    extend = kwargs.pop('extend', False)
    kind = kwargs.pop('kind', 'stats')  # default just stats, no diagnostics
    
    var_names = _validate_var_names(data, group=group, var_names=var_names)

    # self.data[group]
    circ_var_names = [
        k for k in var_names if data[group][k].attrs.get('is_circular', 0) == 1
    ]
    # circ_var_names = [i for i in self.circ_var_names if i in var_names]
    circ_var_names = kwargs.pop(
        'circ_var_names',
        circ_var_names
    )
    summary = az.summary(data, group=group, var_names=var_names, fmt=fmt,
        round_to=round_to, stat_funcs=stat_funcs, extend=extend,
        circ_var_names=circ_var_names, kind=kind, **kwargs)
    
    # Check for duplicated metric names which are a pain to deal with.
    unique, counts = np.unique(summary.metric, return_counts=True)
    is_dup = (counts > 1)
    if is_dup.any():
        dup = list(unique[is_dup])
        raise ValueError(f"Metric names {dup} are duplicated.")
    return summary

def get_table(data, *, dims: Tuple[str], group: str='posterior',
                var_names: Optional[List[str]]=None, fmt: str='pandas',
                round_to: Union[str, int]='auto',
                **kwargs) -> Union[pd.DataFrame, Table]:
    """Get a table of results for parameters in data corresponding to a chosen
    model dimension. Two-dimensional tables 

    Args:
        data (arviz.InferenceData): Inference data object.
        dims (tuple of str): The parameter dimensions for the table. E.g. pass
            () to return a table of 0-dimensional parameters in data, or pass 
            ('n',) for 1-dimensional parameters along dimension 'n'.
        group (str, optional): Group in data to tabulate. Defaults to 
            'posterior'.
        var_names (list of str, optional): Variable names in data to show in
            table. By default all variables along the chosen dim are shown.
            Defaults to None.
        fmt (str, optional): Table format, one of ['pandas', 'astropy']. 
            Defaults to 'pandas'.
        round_to (str, or int, optional): Precision of table data. Defaults to 
            'auto' which chooses the precision for each variable based on the
            error on the mean.
        **kwargs: Keyword arguments to pass to :func:`get_summary`.

    Returns:
        pandas.DataFrame, or astropy.table.Table]: [description]
    """

    var_names = _validate_var_names(data, group=group, var_names=var_names, 
                                    dims=dims)
    summary = get_summary(data, group=group, var_names=var_names, **kwargs)
    table = summary[var_names].to_dataframe()
    if round_to == 'auto':
        # Rounds to the error on the mean. I.e. if mean_err is in range 
        # (0.01, 0.1] then the metrics are rounded to 2 decimal places
        if 'sd' not in summary.metric:
            raise ValueError('Automatic rounding requires the standard ' + \
                'deviation \'sd\'.')
        mean_err = table.loc['sd'] / np.sqrt(data[group].draw.size)
        precision = np.log10(mean_err).astype(int) - 1
        if isinstance(table.index, pd.MultiIndex):
            precision = precision.min()  # Choose min precision = max decimal precision
        table = table.round(-precision)

    elif round_to != 'none':
        table = table.round(round_to)
    
    if fmt == 'astropy':
        # units = {k: v for k, v in self.units.items() if k in var_names}
        units = {
            k: u.Unit(data[group][k].attrs.get('unit', '')) for k in var_names
        }
        table = Table.from_pandas(table.reset_index(), units=units)
    return table
