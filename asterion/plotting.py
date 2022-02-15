"""The plotting module contains functions for plotting inference data."""
from __future__ import annotations

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import astropy.units as u

from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from corner import corner
from arviz.labels import MapLabeller
from typing import List, Optional, Dict, Union

from .annotations import DistLike

def plot_glitch(data: az.InferenceData, group='posterior', kind: str='He',
                quantiles: Optional[List[float]]=None, 
                ax: plt.Axes=None) -> plt.Axes:
    """Plot the glitch from either the prior or posterior predictive contained
    in inference data. 

    Args:
        data (arviz.InferenceData): Inference data object.
        group (str): One of ['posterior', 'prior'].
        kind (str): Kind of glitch to plot. One of ['He', 'CZ'].
        quantiles (iterable, optional): Quantiles to plot as confidence
            intervals. If None, defaults to the 68% confidence interval. Pass
            an empty list to plot no confidence intervals.
        ax (matplotlib.axes.Axes): Axis on which to plot the glitch.

    Returns:
        matplotlib.axes.Axes: Axis on which the glitch is plot.
    """
    if group == 'posterior':
        predictive = data.posterior_predictive
    elif group == 'prior':
        predictive = data.prior_predictive
    else:
        raise ValueError(f"Group '{group}' is not one of ['posterior', 'prior'].")

    if quantiles is None:
        quantiles = [.16, .84]

    nu = data.observed_data.nu
    nu_err = data.constant_data.nu_err
    n = predictive.n
    n_pred = predictive.n_pred
    
    if ax is None:
        ax = plt.gca()

    kindl = kind.lower()
    dnu_key = 'dnu_'+kindl
    dim = ('chain', 'draw')  # dim over which to take stats
    dnu = predictive[dnu_key]

    if group != 'prior':
        # Plot observed - prior predictive should be independent of obs
        res = nu - predictive['nu']
        dnu_obs = dnu + res
        
        # TODO: should we show model error on dnu_obs here?
        ax.errorbar(n, dnu_obs.median(dim=dim),
                    yerr=nu_err, color='C0', marker='o',
                    linestyle='none', label='observed')
    
    dnu_pred = predictive[dnu_key+'_pred']
    dnu_med = dnu_pred.median(dim=dim)
    ax.plot(n_pred, dnu_med, label='median', color='C1')

    # Fill quantiles with alpha decreasing away from the median
    dnu_quant = dnu_pred.quantile(quantiles, dim=dim)
    num_quant = len(quantiles)
    alphas = np.linspace(0.1, 0.5, num_quant*2+1)
    for i in range(num_quant):
        delta = quantiles[-i-1] - quantiles[i]
        ax.fill_between(n_pred, dnu_quant[i], dnu_quant[-i-1],
                        color='C1', alpha=alphas[2*i+1],
                        label=f'{delta:.1%} CI')
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(r'$n$')
    
    ylabel = [dnu.attrs.get('symbol', dnu_key)]

    unit = u.Unit(dnu.attrs.get('unit', ''))
    if str(unit) != '':
        ylabel.append(unit.to_string(format='latex_inline'))

    ax.set_ylabel('/'.join(ylabel))
    ax.legend()
    return ax

def get_labeller(data: az.InferenceData, group: str='posterior', 
                 var_names: Optional[List[str]]=None) -> MapLabeller:
    """Get labeller for use with arviz plotting. This automatically searches
    variable metadata (contained in their attrs dictionary) for its 'symbol'
    and 'unit' if available.
    
    Args:
        data (arviz.InferenceData): Inference data object.
        group (str, optional): Inference data group for which to map labels.
        var_names (list of str, optional): Variable names for which to map
            labels.
        
    Returns:
        arviz.labels.MapLabeller: Label map
    """
    if var_names is None:
        var_names = list(data[group].keys())  # Use all variables (dangerous)

    var_name_map = {}
    for key in var_names:
        # Loop through var names and extract units where available
        # SYMBOL
        sym = data[group][key].attrs.get('symbol', '')
        
        if sym == '':
            sym = key
        
        L = [sym]
        
        # UNITS
        unit = u.Unit(data[group][key].attrs.get('unit', ''))
        if isinstance(unit, u.LogUnit):
            # LogUnit doesn't support latex_inline
            unit = unit.physical_unit

        if str(unit) != '':
            L.append(f'{unit.to_string("latex_inline")}')

        var_name_map[key] = '/'.join(L)
        
    return MapLabeller(var_name_map=var_name_map)

def plot_corner(data: az.InferenceData, group: str='posterior',
                var_names: Optional[List[str]]=None, 
                quantiles: Optional[List[float]]=None, 
                labeller: Union[str, MapLabeller]='auto', 
                **kwargs) -> Figure:
    """A wrapper for :func:`corner.corner` with automatic labelling and
    custom default arguments specified below.

    Args:
        data (arviz.InferenceData): Inference data object.
        group (str, optional): Inference data group from which to take samples.
            Defaults to 'posterior'.
        var_names (list of str, optional): Variable names to plot. 
            Defaults to plotting all available variables.
        quantiles (iterable, optional): Quantiles to plot as dashed lines in
            the marginals. If None, defaults to the 68% confidence interval. 
            Pass an empty list to plot no confidence intervals.
        labeller (str or MapLabeller, optional): Labeller which maps variable
            names to their axis labels. Defaults to 'auto'.
        **kwargs: Keyword arguments to pass to :func:`corner.corner`.

    Returns:
        matplotlib.figure.Figure: Figure object.
    
    See Also:
        :func:`corner.corner`: The function for which this wraps.
    """
    if quantiles is None:
        quantiles = [.16, .84]
    if labeller == 'auto':
        labeller = get_labeller(data, group=group, var_names=var_names)
    
    show_titles = kwargs.pop('show_titles', True)
    smooth = kwargs.pop('smooth', 1.)
    
    fig = corner(
        data,
        group=group,
        var_names=var_names, 
        quantiles=quantiles,
        labeller=labeller,
        show_titles=show_titles,
        smooth=smooth,
        **kwargs
    )
    return fig
