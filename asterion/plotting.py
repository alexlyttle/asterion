from __future__ import annotations

import numpy as np

from typing import Callable, Optional, Dict
from .annotations import DistLike

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import astropy.units as u
import arviz as az
from arviz.labels import MapLabeller
    
def plot_glitch(data: az.InferenceData, group='posterior', kind: str='He',
                quantiles: Optional[list]=[.16, .84], 
                ax: plt.Axes=None) -> plt.Axes:
    """Plot the glitch.

    Args:
        data (arviz.InferenceData): Inference class
        group (str): One of ['posterior', 'prior'].
        kind (str): Kind of glitch to plot. One of ['He', 'CZ'].
        quantiles (iterable, optional): Quantiles to plot as confidence
            intervals. Defaults to the 68% confidence interval.
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

    nu = data.observed_data.nu
    nu_err = data.constant_data.nu_err
    n = predictive.n
    n_pred = predictive.n_pred
    
    if ax is None:
        ax = plt.gca()

    kindl = kind.lower()

    dim = ('chain', 'draw')  # dim over which to take stats
    dnu = predictive['dnu_'+kindl]

    if group != 'prior':
        res = nu - predictive['nu']
        dnu_obs = dnu + res
        
        # TODO: good way to show model error on dnu_obs here??
        ax.errorbar(n, dnu_obs.median(dim=dim),
                    yerr=nu_err, color='C0', marker='o',
                    linestyle='none', label='observed')
    
    dnu_pred = predictive['dnu_'+kindl+'_pred']
    dnu_med = dnu_pred.median(dim=dim)
    ax.plot(n_pred, dnu_med, label='median', color='C1')

    if quantiles is not None:
        # Fill quantiles shaded away from the median
        dnu_quant = dnu_pred.quantile(quantiles, dim=dim)
        num_quant = len(quantiles)//2
        alphas = np.linspace(0.1, 0.5, num_quant*2+1)
        for i in range(num_quant):
            delta = quantiles[-i-1] - quantiles[i]
            ax.fill_between(n_pred, dnu_quant[i], dnu_quant[-i-1],
                            color='C1', alpha=alphas[2*i+1],
                            label=f'{delta:.1%} CI')
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(r'$n$')
    L = [dnu.attrs['symbol']]

    unit = u.Unit(dnu.attrs['unit'])
    if str(unit) != '':
        L.append(unit.to_string(format='latex_inline'))

    ax.set_ylabel('/'.join(L))
    ax.legend()
    return ax

def get_labeller(data: az.InferenceData, group='posterior', var_names=None):
    """Get labeller for use with arviz plotting."""
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

def plot_corner(data, group='posterior', var_names=None, quantiles=[.16, .84], 
                labeller='auto', **kwargs):
    import corner
    # if var_names is None:
    #     var_names = list(data[group].keys())  # Use all variables (dangerous)

    # if labeller == 'auto':
    #     var_name_map = {}
    #     for key in var_names:
    #         # Loop through var names and extract units where available
    #         # SYMBOL
    #         sym = data[group][key].attrs.get('symbol', '')
            
    #         if sym == '':
    #             sym = key
            
    #         L = [sym]
            
    #         # UNITS
    #         unit = u.Unit(data[group][key].attrs.get('unit', ''))
    #         if isinstance(unit, u.LogUnit):
    #             # LogUnit doesn't support latex_inline
    #             unit = unit.physical_unit

    #         if str(unit) != '':
    #             L.append(f'{unit.to_string("latex_inline")}')

    #         var_name_map[key] = '/'.join(L)
            
    #     labeller = MapLabeller(var_name_map=var_name_map)
    if labeller == 'auto':
        labeller = get_labeller(data, group=group, var_names=var_names)
    
    show_titles = kwargs.pop('show_titles', True)
    smooth = kwargs.pop('smooth', 1.)
    
    fig = corner.corner(
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
