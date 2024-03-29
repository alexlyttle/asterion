"""The plotting module contains functions for plotting inference data."""
from __future__ import annotations

import os, colorsys
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import astropy.units as u

from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from matplotlib.pyplot import style
from corner import corner
from arviz.labels import MapLabeller
from typing import List, Optional, Union
from .utils import PACKAGE_DIR

__all__ = [
    "plot_glitch",
    "plot_corner",
    "get_labeller",
]

_asterion_style_path = os.path.join(PACKAGE_DIR, "stylelib")
style.core.USER_LIBRARY_PATHS.append(_asterion_style_path)
style.core.reload_library()


def _lighten_color(color, amount):
    """
    Modified from source: https://stackoverflow.com/a/49601444

    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def _validate_predictive_group(data: az.InferenceData, group: str):
    """Validate the predictive groups in data.

    Args:
        data (arviz.InferenceData): Inference data object.
        group (str): One of ['posterior', 'prior'].

    Raises:
        ValueError: If group is not valid.
        KeyError: If predictive is not in data, gives helpful suggestion.

    Returns:
        xarray.Dataset: Dataset corresponding to the predictive of group.
    """
    if group == "posterior":
        key = "posterior_predictive"
        predictive = data.get(key, None)
    elif group == "prior":
        key = "prior_predictive"
        predictive = data.get(key, None)
    else:
        raise ValueError(
            f"Group '{group}' is not one of ['posterior', 'prior']."
        )

    if predictive is None:
        raise KeyError(
            f"Group '{key}' not in data. Consider using method "
            + "'Inference.{key}()' to sample the predictive."
        )
    return predictive


def plot_glitch(
    data: az.InferenceData,
    group: str = "posterior",
    kind: str = "full",
    x_var: str = "n",
    quantiles: Optional[List[float]] = None,
    observed: Union[bool, str] = "auto",
    use_alpha: bool = True,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot the glitch from either the prior or posterior predictive contained
    in inference data.

    Args:
        data (arviz.InferenceData): Inference data object.
        group (str): One of ['posterior', 'prior'].
        kind (str): Kind of glitch to plot. One of ['full', 'helium', 'BCZ'].
        x_var (str): Variable name for x-axis. One of ['n', 'nu']. If 'nu', the
            median value of 'nu' in ``data['group']`` is used.
        quantiles (iterable, optional): Quantiles to plot as confidence
            intervals. If None, defaults to the 68% confidence interval. Pass
            an empty list to plot no confidence intervals.
        observed (bool or str): Whether to plot observed data. Default is
            "auto" which will plot observed data when group is "posterior".
        use_alpha (bool): Whether to use alpha channel for transparency. If
            False, will shade with lightened solid color.
        ax (matplotlib.axes.Axes): Axis on which to plot the glitch.
        **kwargs: Keyword arguments to pass to :func:`matplotlib.pyplot.plot`.

    Raises:
        ValueError: If kind is not valid.

    Returns:
        matplotlib.axes.Axes: Axis on which the glitch is plot.
    """
    dim = ("chain", "draw")  # dim over which to take stats
    predictive = _validate_predictive_group(data, group)

    if quantiles is None:
        quantiles = [0.16, 0.84]

    if observed == "auto":
        observed = group == "posterior"

    nu = data.observed_data.nu
    nu_err = data.constant_data.nu_err

    if x_var == "n":
        x = predictive.n
        x_pred = predictive.n_pred
        xlabel = "n"
    elif x_var == "nu":
        x = predictive.nu.median(dim=dim)
        x_pred = predictive.nu_pred.median(dim=dim)
        xlabel = r"$\nu$ ($\mathrm{\mu Hz}$)"

    if ax is None:
        _, ax = plt.subplots()

    kindl = kind.lower()
    if kindl == "full":
        dnu = predictive["dnu_he"] + predictive["dnu_cz"]
        dnu_pred = predictive["dnu_he_pred"] + predictive["dnu_cz_pred"]
    else:
        if kindl in {"helium", "he"}:
            kind = "helium"  # Set to helium for consistency in legend label
            dnu_key = "dnu_he"
        elif kindl in {"bcz", "cz"}:
            kind = "BCZ"  # Set to BCZ for consistency in legend label
            dnu_key = "dnu_cz"
        else:
            # Raise error if kind is not valid
            raise ValueError(
                f"Kind '{kindl}' is not one of " + "['full', 'helium', 'BCZ']."
            )
        dnu = predictive[dnu_key]
        dnu_pred = predictive[dnu_key + "_pred"]
        # label = dnu.attrs.get("symbol", r"$\delta\nu_{" + kind + "}$")
    label = f"{kind} glitch model"

    if observed:
        # Plot observed - prior predictive should be independent of obs
        res = nu - predictive["nu"]
        dnu_obs = dnu + res
        # glitch = label
        # if "+" in label:
        #     glitch = "$($" + label + "$)$"
        # TODO: should we show model error on dnu_obs here?
        ax.errorbar(
            x,
            dnu_obs.median(dim=dim),
            yerr=nu_err,
            color="k",
            marker="o",
            linestyle="none",
            label=r"observed",
        )

    dnu_med = dnu_pred.median(dim=dim)
    label = kwargs.pop("label", label)
    (line,) = ax.plot(x_pred, dnu_med, label=label, **kwargs)

    # Fill quantiles with alpha decreasing away from the median
    dnu_quant = dnu_pred.quantile(quantiles, dim=dim)
    num_quant = len(quantiles) // 2
    num_alphas = num_quant * 2 + 1
    alphas = np.linspace(0.1, 0.5, num_alphas)
    base_color = line.get_color()

    if use_alpha:
        colors = [base_color] * num_alphas
    else:
        # Mimic alpha by lightening the base color
        colors = [_lighten_color(base_color, 1.5 * a) for a in alphas]
        alphas = [None] * num_alphas  # reset alphas to None

    for i in range(num_quant):
        delta = quantiles[-i - 1] - quantiles[i]
        ax.fill_between(
            x_pred,
            dnu_quant[i],
            dnu_quant[-i - 1],
            color=colors[2 * i + 1],  # <-- same as model line color
            alpha=alphas[2 * i + 1],
            label=f"{delta:.1%} CI",
        )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # integer x-ticks
    ax.set_xlabel(xlabel)
    # ax.set_xlabel("radial order")

    # ylabel = [dnu.attrs.get("symbol", r"$\delta\nu$")]
    # ylabel = [r"$\delta\nu$"]
    unit = u.Unit(dnu.attrs.get("unit", "uHz"))
    # if str(unit) != "":
    # ylabel.append(unit.to_string(format="latex_inline"))

    # ax.set_ylabel("/".join(ylabel))
    ax.set_ylabel(
        r"$\delta\nu$ " + f"({unit.to_string(format='latex_inline')})"
    )
    ax.legend()
    return ax


def plot_echelle(
    data: az.InferenceData,
    group="posterior",
    kind: str = "full",
    delta_nu: Optional[float] = None,
    quantiles: Optional[List[float]] = None,
    observed: Union[bool, str] = "auto",
    use_alpha: bool = True,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot an echelle diagram of the data.

    Choose to plot the full mode, background model or glitchless model. This is
    compatible with data from inference on models like :class:`GlitchModel`.

    Args:
        data (az.InferenceData): Inference data object.
        group (str): On of ['posterior', 'prior']. Defaults to 'posterior'.
        kind (str): One of ['full', 'glitchless', 'background']. Defaults to
            'full' which plots the full model for nu. Use 'glitchless' to plot
            the model without the glitch components. Use 'background' to plot
            the background component of the model.
        delta_nu (float, optional): Large frequency separation to modulo by.
            If None, the median value from ``data['group']`` is used.
        quantiles (iterable, optional): Quantiles to plot as confidence
            intervals. If None, defaults to the 68% confidence interval. Pass
            an empty list to plot no confidence intervals.
        observed (bool or str): Whether to plot observed data. Default is
            "auto" which will plot observed data when group is "posterior".
        use_alpha (bool): Whether to use alpha channel for transparency. If
            False, will shade with lightened solid color.
        ax (matplotlib.axes.Axes): Axis on which to plot the echelle.
        **kwargs: Keyword arguments to pass to :func:`matplotlib.pyplot.plot`.

    Raises:
        ValueError: If kind is not valid.

    Returns:
        matplotlib.axes.Axes: Axis on which the echelle is plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    if quantiles is None:
        quantiles = [0.16, 0.84]

    if observed == "auto":
        observed = group == "posterior"

    predictive = _validate_predictive_group(data, group)
    dim = ("chain", "draw")  # dim over which to take stats

    if delta_nu is None:
        if group == "prior":  # <-- currently no prior group
            delta_nu = predictive["delta_nu"].median().to_numpy()
        else:
            delta_nu = data[group]["delta_nu"].median().to_numpy()

    nu = data.observed_data.nu
    nu_err = data.constant_data.nu_err
    n_pred = predictive.n_pred

    if observed:
        # Plot observed - prior predictive should be independent of obs
        ax.errorbar(
            nu % delta_nu,
            nu,
            xerr=nu_err,
            color="k",
            marker="o",
            linestyle="none",
            label="observed",
        )

    # All mean function components for GP
    # full_mu = [
    # predictive["nu_bkg"].attrs.get("symbol", r"$\nu_\mathrm{bkg}$"),
    # predictive["dnu_he"].attrs.get("symbol", r"$\delta\nu_{He}$"),
    # predictive["dnu_cz"].attrs.get("symbol", r"$\delta\nu_{BCZ}$"),
    # ]
    kindl = kind.lower()
    if kindl == "full":
        y = predictive["nu_pred"]
        # label = r"$\mathrm{GP}($" + " + ".join(full_mu) + r"$,\,K)$"
    elif kindl == "background":
        y = predictive["nu_bkg_pred"]
        # label = full_mu[0]  # <-- just the background, no GP
    elif kindl == "glitchless":
        y = (
            predictive["nu_pred"]
            - predictive.get("dnu_he_pred", 0.0)
            - predictive.get("dnu_cz_pred", 0.0)
        )
        y.attrs["unit"] = predictive["nu_pred"].attrs["unit"]
        # label = r"$\mathrm{GP}($" + full_mu[0] + r"$,\,K)$"
    else:
        raise ValueError(
            f"Kind '{kindl}' is not one of "
            + "['full', 'background', 'glitchless']."
        )
    label = f"{kindl} model"

    y_mod = (y - n_pred * delta_nu) % delta_nu
    y_med = y.median(dim=dim)
    label = kwargs.pop("label", label)
    (line,) = ax.plot(
        y_mod.median(dim=dim),
        y_med,
        label=label,
        **kwargs,
    )

    y_mod_quant = y_mod.quantile(quantiles, dim=dim)
    num_quant = len(quantiles) // 2
    num_alphas = num_quant * 2 + 1
    alphas = np.linspace(0.1, 0.5, num_alphas)
    base_color = line.get_color()

    if use_alpha:
        colors = [base_color] * num_alphas
    else:
        # Mimic alpha by lightening the base color
        colors = [_lighten_color(base_color, 1.5 * a) for a in alphas]
        alphas = [None] * num_alphas  # reset alphas to None

    for i in range(num_quant):
        delta = quantiles[-i - 1] - quantiles[i]
        ax.fill_betweenx(
            y_med,
            y_mod_quant[i],
            y_mod_quant[-i - 1],
            color=colors[2 * i + 1],
            alpha=alphas[2 * i + 1],
            label=f"{delta:.1%} CI",
        )

    # xlabel = [r"$\nu\,\mathrm{mod}.\,{" + f"{delta_nu:.2f}" + "}$"]
    unit = u.Unit(y.attrs.get("unit", ""))
    # if str(unit) != "":
    # xlabel.append(unit.to_string(format="latex_inline"))
    # ax.set_xlabel("/".join(xlabel))

    ax.set_xlabel(
        r"$\nu$ modulo "
        + f"{delta_nu:.2f} "
        + f"({unit.to_string(format='latex_inline')})"
    )
    # ylabel = [r"$\nu$"]
    unit = u.Unit(nu.attrs.get("unit", "uHz"))
    # if str(unit) != "":
    # ylabel.append(unit.to_string(format="latex_inline"))
    # ax.set_ylabel("/".join(ylabel))

    ax.set_ylabel(r"$\nu$ " + f"({unit.to_string(format='latex_inline')})")

    ax.legend()

    return ax


def get_labeller(
    data: az.InferenceData,
    group: str = "posterior",
    var_names: Optional[List[str]] = None,
) -> MapLabeller:
    """Get labeller for use with arviz plotting. This automatically searches
    variable metadata (contained in their attrs dictionary) for its 'symbol'
    and 'unit' if available.

    Args:
        data (arviz.InferenceData): Inference data object.
        group (str): Inference data group for which to map labels.
        var_names (list[str], optional): Variable names for
            which to map labels.

    Returns:
        arviz.labels.MapLabeller: Label map.
    """
    if var_names is None:
        var_names = list(data[group].keys())  # Use all variables (dangerous)

    var_name_map = {}
    for key in var_names:
        # Loop through var names and extract units where available
        # SYMBOL
        sym = data[group][key].attrs.get("symbol", "")

        if sym == "":
            sym = key

        L = [sym]

        # UNITS
        unit = u.Unit(data[group][key].attrs.get("unit", ""))
        if isinstance(unit, u.LogUnit):
            # LogUnit doesn't support latex_inline
            unit = unit.physical_unit

        if str(unit) != "":
            L.append(f'{unit.to_string("latex_inline")}')

        var_name_map[key] = "/".join(L)

    return MapLabeller(var_name_map=var_name_map)


def plot_corner(
    data: az.InferenceData,
    group: str = "posterior",
    var_names: Optional[List[str]] = None,
    quantiles: Optional[List[float]] = None,
    labeller: Union[str, MapLabeller] = "auto",
    **kwargs,
) -> Figure:
    """A wrapper for :func:`corner.corner` with automatic labelling and
    custom default arguments specified below.

    Args:
        data (arviz.InferenceData): Inference data object.
        group (str): Inference data group from which to take samples.
            Defaults to 'posterior'.
        var_names (List[str], optional): Variable names to plot.
            Defaults to plotting all available variables.
        quantiles (iterable, optional): Quantiles to plot as dashed lines in
            the marginals. If None, defaults to the 68% confidence interval.
            Pass an empty list to plot no confidence intervals.
        labeller (str, or MapLabeller): Labeller which maps variable
            names to their axis labels. Defaults to 'auto'.
        **kwargs: Keyword arguments to pass to :func:`corner.corner`.

    Returns:
        matplotlib.figure.Figure: Figure object.

    See Also:
        :func:`corner.corner`: The function for which this wraps.
    """
    if quantiles is None:
        quantiles = [0.16, 0.84]
    if labeller == "auto":
        labeller = get_labeller(data, group=group, var_names=var_names)

    show_titles = kwargs.pop("show_titles", True)
    smooth = kwargs.pop("smooth", 1.0)

    fig = corner(
        data,
        group=group,
        var_names=var_names,
        quantiles=quantiles,
        labeller=labeller,
        show_titles=show_titles,
        smooth=smooth,
        **kwargs,
    )
    return fig
