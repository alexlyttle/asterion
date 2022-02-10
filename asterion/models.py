"""Probabilistic models for asteroseismic oscillation mode frequencies.
"""
from __future__ import annotations

import numpy as np

import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.distribution import Distribution

from numpyro.primitives import Messenger, plate, apply_stack

from typing import Callable, Optional, Dict
from .annotations import DistLike

from .gp import GP, SquaredExponential

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import warnings

import astropy.units as u

from collections.abc import Iterable
from numpy.typing import ArrayLike

import arviz as az

__all__ = [
    "distribution",
    "dimension",
    "Model",
    "ZerosFunction",
    "AsyFunction",
    "HeGlitchFunction",
    "CZGlitchFunction",
    "GlitchModel",
]


def distribution(
    value: DistLike,
    default_dist: Optional[dist.Distribution]=None,
) -> dist.Distribution:
    """Return a numpyro distribition.

    If value is not a distribution, this returns the default distribution,
    unpacking value as its arguments.

    Args:
        value (:term:`dist_like`): Iterable of args to pass to default_dist, 
            or a Distribution.
        default_dist (numpyro.distributions.distribution.Distribution, \
optional): Default distribution. Defaults to dist.Normal if None.

    Returns:
        numpyro.distributions.distribution.Distribution: [description]
    """
    if default_dist is None:
        default_dist = dist.Normal
    if not isinstance(value, dist.Distribution):
        if not isinstance(value, Iterable):
            value = (value,)
        value = default_dist(*value)
    return value


class dimension(Messenger):
    """Context manager for a model dimension.

    Args:
        name (str): Name of the dimension.
        size (int): Size of the dimension.
        coords (:term:`array_like`, optional): Coordinates for points in the
            dimension. Defaults to :code:`np.arange(size)`.
        dim (int, optional): Where to place the dimension. Defaults to
            :code:`-1` which corresponds to the rightmost dimension. Must be
            negative.
    """

    def __init__(self, name: str, size: int, coords: Optional[ArrayLike]=None,
                 dim: Optional[ArrayLike]=None):
        self.name: str = name  #:str: Name of the dimension.
        self.size: int = size  #:int: Size of the dimension.
        self.dim: int = -1 if dim is None else dim
        """:int: Location in which to insert the dimension."""
        
        assert self.dim < 0
        if coords is None:
            coords = np.arange(self.size)
        self.coords: np.ndarray = np.array(coords)
        """:numpy.ndarray: Coordinates for the dimension."""

        msg = self._get_message()
        apply_stack(msg)
        super().__init__()
    
    def _get_message(self) -> dict:
        msg = {
            "name": self.name,
            "type": "dimension",
            "dim": self.dim,
            "value": self.coords,
        }
        return msg
        
    def __enter__(self) -> dict:
        super().__enter__()
        return self._get_message()

    def process_message(self, msg: dict):
        """Process the message.

        Args:
            msg (dict): Message.

        Raises:
            ValueError: If the corresponding dimension of the site is of
                incorrect size.
        """
        if msg["type"] not in ("param", "sample", "deterministic"):
            # We don't add dimensions to dimensions
            return

        if msg["value"] is None:
            shape = ()
            if "fn" in msg.keys():
                sample_shape = msg["kwargs"].get("sample_shape", ())
                shape = msg["fn"].shape(sample_shape)
        else:
            shape = msg["value"].shape

        if "dims" not in msg.keys():
            dims = [f"{msg['name']}_dim_{i}" for i in range(len(shape))]
            msg["dims"] = dims
        
        if "dim_stack" not in msg.keys():
            msg["dim_stack"] = []
        
        dim = self.dim
        while dim in msg["dim_stack"]:
            dim -= 1

        msg["dim_stack"].append(dim)
        msg["dims"][dim] = self.name
        
        if shape[dim] != self.size:
            raise ValueError(f"Dimension {dim} of site \'{msg['name']}\' should have length {self.size}")


class Model:
    """Base Model class.
    
    A model is a probabilistic object which may be given to Inference. It does
    not need to return anything during inference, but should have at least
    one observed sample sites.

    A prior is a model which returns a parameter or function when called and
    has no observed sample sites.
    """
    units: Dict[str, u.Unit] = {}
    symbols: Dict[str, str] = {}
    """:dict: Astropy units corresponding to each model parameter."""

    def predict(self, *args, **kwargs):
        """Model predictions. By default this calls the model.
        """
        return self(*args, **kwargs)

    def __call__(self):
        """Call the model during inference.

        Raises:
            NotImplementedError: This is an abstract base class and cannot be
                called.
        """
        raise NotImplementedError


class ZerosFunction(Model):
    """A prior on the zeros function :math:`f` where
    :math:`f(\\boldsymbol{x}) = \\boldsymbol{0}`.
    """
    def __call__(self) -> Callable:
        """Samples the prior for the zeros function.

        Returns:
            function: The function :math:`f`.
        """
        return lambda x: jnp.zeros(x.shape)


class AsyFunction(Model):
    """Prior on the linear asymptotic function :math:`f`, where
    :math:`f(n) = \\Delta\\nu (n + \\epsilon)`.
    
    Args:
        delta_nu (:term:`dist_like`): Prior for the large frequency separation
            :math:`\\Delta\\nu`. Pass either the arguments of
            :class:`dist.Normal` or a :class:`dist.Distribution`.
        epsilon (:term:`dist_like`): Prior for the phase term
            :math:`\\epsilon`. Pass either the arguments of :class:`dist.Gamma`
            or a :class:`dist.Distribution`. Defaults to :code:`(14., 10.)`.
    """
    def __init__(self, delta_nu: DistLike, epsilon: DistLike=None):
        self.delta_nu: dist.Distribution = distribution(delta_nu)
        """:numpyro.distributions.distribution.Distribution: The distribution for
        :math:`\\Delta\\nu`."""

        if epsilon is None:
            epsilon = (np.log(1.4), 0.4)
        self.epsilon: dist.Distribution = distribution(epsilon, dist.LogNormal)
        """:numpyro.distributions.distribution.Distribution: The distribution 
        for :math:`\\epsilon`."""

        self.units = {
            'delta_nu': u.microhertz,
            'epsilon': u.dimensionless_unscaled,
        }
        self.symbols = {
            'delta_nu': r'$\Delta\nu$',
            'epsilon': r'$\epsilon$',
        }

    def __call__(self) -> Callable:
        """Samples the prior for the linear asymptotic function.

        Returns:
            function: The function :math:`f`.
        """
        delta_nu = numpyro.sample('delta_nu', self.delta_nu)
        epsilon = numpyro.sample('epsilon', self.epsilon)
        
        def fn(n):
            return delta_nu * (n + epsilon)
        return fn


class _GlitchFunction(Model):
    """Prior on the glitch oscillation function :math:`f`, where
    :math:`f(nu) = \\sin(4\\pi\\tau\\nu + \\phi)`.

    Args:
        log_tau (:term:`dist_like`): The prior for the acoustic depth of the
            glitch, :math:`\\tau`. Pass either the arguments of
            :class:`dist.Normal` or a :class:`dist.Distribution`.
        phi (:term:`dist_like`): The prior for the phase of the glitch,
            :math:`\\phi`. Pass either the arguments of :class:`dist.VonMises`
            or a :class:`dist.Distribution`.
    """

    def __init__(self, log_tau: DistLike, phi: DistLike=None):
        self.log_tau = distribution(log_tau)
        """:numpyro.distributions.distribution.Distribution: The distribution 
        for :math:`\\tau`."""
        
        if phi is None:
            phi = (-np.pi, np.pi)
        self.phi = distribution(phi, dist.Uniform)
        """:numpyro.distributions.distribution.Distribution: The distribution
        for :math:`\\phi`."""

    @staticmethod
    def oscillation(nu: ArrayLike, tau: ArrayLike,
                    phi: ArrayLike) -> jnp.ndarray:
        r"""Oscillatory component of the acoustic glitch.

        Args:
            nu (:term:`array_like`): Mode frequency, :math:`\\nu`.
            tau (:term:`array_like`): Acoustic depth, :math:`\\tau`.
            phi (:term:`array_like`): Glitch phase, :math:`\\phi`.

        Returns:
            jax.numpy.ndarray: Glitch oscillation.
        """
        return jnp.sin(4 * jnp.pi * tau * nu + phi)

    def __call__(self) -> Callable:
        """Samples the prior for a generic glitch oscillation function.

        Returns:
            function: The function :math:`f`.
        """
        log_tau = numpyro.sample('log_tau', self.log_tau)
        tau = numpyro.deterministic('tau', 10**log_tau)
        phi = numpyro.sample('phi', self.phi)
        def fn(nu):
            return self.oscillation(nu, tau, phi)
        return fn


class HeGlitchFunction(_GlitchFunction):
    """Prior on the second ionisation of helium glitch function :math:`f`,
    where :math:`f(\\nu) = a_\\mathrm{He} \\nu \\exp(-b_\\mathrm{He} \\nu^2) 
    \\sin(4\\pi\\tau_\\mathrm{He}\\nu + \\phi_\\mathrm{He})`.

    The priors for the glitch parameters 
    :math:`a_\\mathrm{He},b_\\mathrm{He},\\tau_\\mathrm{He}` are inferred
    from that of the frequency at maximum power, :math:`\\nu_\\max` using
    scaling relations derived from stellar models (Lyttle et al. in prep.).

    Args:
        nu_max (:term:`dist_like`): The prior for the frequency at maximum
            power, :math:`\\nu_\\max`. Pass either the arguments of
            :class:`dist.Normal` or a :class:`dist.Distribution`.
    """
    def __init__(self, nu_max: DistLike, log_tau: DistLike, phi: DistLike=None):
        self.log_a: dist.Distribution
        """:numpyro.distributions.distribution.Distribution: The distribition
        for the glitch phase parameter phi_he."""
        
        self.log_b: dist.Distribution
        """:numpyro.distributions.distribution.Distribution: The distribution
        for log base-10 of the glitch parameter b_he."""
        
        self.log_tau: dist.Distribution
        """:numpyro.distributions.distribution.Distribution: The distribution
        for log base-10 of the glitch parameter (acoustic depth) tau_he."""
        
        log_numax = jnp.log10(distribution(nu_max).mean)
        # Attempt rough guess of glitch params
        self.log_a = dist.Normal(-1.10 - 0.35*log_numax, 0.8)
        self.log_b = dist.Normal(0.719 - 2.14*log_numax, 0.8)
        # self.log_tau = dist.Normal(0.44 - 1.03*log_numax, 0.1)
        
        super().__init__(log_tau, phi=phi)
        
        # self.phi: dist.Distribution = dist.VonMises(0.0, 0.1)
        # """:numpyro.distributions.distribution.Distribution: The distribution
        # for the phase parameter."""
        
        self.units = {
            'a_he': u.dimensionless_unscaled,
            'b_he': u.megasecond**2,
            'tau_he': u.megasecond,
            'phi_he': u.rad,
        }
        self.symbols = {
            'a_he': r'$a_\mathrm{He}$',
            'b_he': r'$b_\mathrm{He}$',
            'tau_he': r'$\tau_\mathrm{He}$',
            'phi_he': r'$\phi_\mathrm{He}$',
        }
        # log units
        for k in ['a_he', 'b_he', 'tau_he']:
            log_k = f'log_{k}'
            self.units[log_k] = u.LogUnit(self.units[k])
            self.symbols[log_k] = r'$\log\,' + self.symbols[k][1:]

    # @property
    # def nu_max(self) -> dist.Distribution:
    #     """The distribution of the frequency at maximum power,
    #     :math:`\\nu_\\max`.
    #     """
    #     return self._nu_max

    # @nu_max.setter
    # def nu_max(self, value: DistLike):
    #     """Resets the priors for the glitch parameters.

    #     Args:
    #         value (:term:`dist_like`): The prior for the frequency at maximum
    #             power, :math:`\\nu_\\max`. Pass either the arguments of
    #             :class:`dist.Normal` or a :class:`dist.Distribution`.
    #     """
    #     self._nu_max = distribution(value)
    #     log_numax = jnp.log10(self._nu_max.mean)
    #     self.log_a = dist.Normal(-1.10 - 0.35*log_numax, 0.7)
    #     self.log_b = dist.Normal(0.719 - 2.14*log_numax, 0.7)
    #     self.log_tau = dist.Normal(0.44 - 1.03*log_numax, 0.1)

    @staticmethod
    def amplitude(nu: ArrayLike, a: ArrayLike, b: ArrayLike) -> jnp.ndarray:
        r"""The amplitude of the glitch,
        :math:`a_\\mathrm{He} \\nu \\exp(-b_\\mathrm{He} \\nu^2)`.

        Args:
            nu (:term:`array_like`): Mode frequency, :math:`\\nu`.
            a (:term:`array_like`): Amplitude, :math:`a_\\mathrm{He}`.
            b (:term:`array_like`): Decay, :math:`b_\\mathrm{He}`.

        Returns:
            jax.numpy.ndarray: Helium glitch amplitude.
        """
        return a * nu * jnp.exp(- b * nu**2)
    
    def __call__(self) -> Callable:
        """Samples the helium glitch function.

        Returns:
            function: The function :math:`f`.
        """
        log_a = numpyro.sample('log_a_he', self.log_a)
        log_b = numpyro.sample('log_b_he', self.log_b)
        log_tau = numpyro.sample('log_tau_he', self.log_tau)
        
        a = numpyro.deterministic('a_he', 10**log_a)
        b = numpyro.deterministic('b_he', 10**log_b)
        tau = numpyro.deterministic('tau_he', 10**log_tau)
        phi = numpyro.sample('phi_he', self.phi)
        
        def fn(nu):
            return self.amplitude(nu, a, b) * self.oscillation(nu, tau, phi)
        return fn


class CZGlitchFunction(_GlitchFunction):
    """Prior on the base of the convective zone glitch function :math:`f`,
    where :math:`f(\\nu) = a_\\mathrm{CZ} \\nu^{-2}
    \\sin(4\\pi\\tau_\\mathrm{CZ}\\nu + \\phi_\\mathrm{CZ})`

    The priors for the glitch parameters 
    :math:`a_\\mathrm{He},b_\\mathrm{He},\\tau_\\mathrm{He}` are inferred
    from that of the frequency at maximum power, :math:`\\nu_\\max` using
    scaling relations derived from stellar models (Lyttle et al. in prep.).

    Args:
        nu_max (:term:`dist_like`): The prior for the frequency at maximum
            power, :math:`\\nu_\\max`. Pass either the arguments of
            :class:`dist.Normal` or a :class:`dist.Distribution`.
    """
    def __init__(self, nu_max: DistLike, log_tau: DistLike, phi: DistLike=None):
        self.log_a: dist.Distribution
        """:numpyro.distributions.distribution.Distribution: The distribition
        for the glitch phase parameter phi_cz."""
        
        self.log_tau: dist.Distribution
        """:numpyro.distributions.distribution.Distribution: The distribution
        for log base-10 of the acoustic depth tau_cz."""
        
        log_numax = jnp.log10(distribution(nu_max).mean)
        # Rough guess of glitch params
        self.log_a = dist.Normal(2*log_numax - 1.0, 0.8)
        # self.log_tau = dist.Normal(0.77 - 0.99*log_numax, 0.1)
        
        super().__init__(log_tau, phi=phi)
        # self.phi: dist.Distribution = dist.VonMises(0.0, 0.1)
        # """:numpyro.distributions.distribution.Distribution: The distribition
        # for the glitch phase parameter phi_cz."""

        self.units = {
            'a_cz': u.microhertz**3,
            'tau_cz': u.megasecond,
            'phi_cz': u.rad,
        }
        self.symbols = {
            'a_cz': r'$a_\mathrm{BCZ}$',
            'tau_cz': r'$\tau_\mathrm{BCZ}$',
            'phi_cz': r'$\phi_\mathrm{BCZ}$',
        }

        # log units
        for k in ['a_cz', 'tau_cz']:
            log_k = f'log_{k}'
            self.units[log_k] = u.LogUnit(self.units[k])
            self.symbols[log_k] = r'$\log\,' + self.symbols[k][1:]
    
    # @property
    # def nu_max(self) -> dist.Distribution:
    #     """The distribution of the frequency at maximum power,
    #     :math:`\\nu_\\max`.
    #     """
    #     return self._nu_max

    # @nu_max.setter
    # def nu_max(self, value: DistLike):
    #     """Resets the priors for the glitch parameters.

    #     Args:
    #         value (:term:`dist_like`): The prior for the frequency at maximum
    #             power, :math:`\\nu_\\max`. Pass either the arguments of
    #             :class:`dist.Normal` or a :class:`dist.Distribution`.
    #     """
    #     self._nu_max = distribution(value)
    #     log_numax = jnp.log10(self._nu_max.mean)
    #     self.log_a = dist.Normal(2*log_numax - 1.0, 0.7)
    #     self.log_tau = dist.Normal(0.77 - 0.99*log_numax, 0.1)
    
    @staticmethod
    def amplitude(nu: ArrayLike, a: ArrayLike) -> jnp.ndarray:
        """The amplitude of the glitch,
        :math:`a_\\mathrm{CZ} / \\nu^{-2}`.

        Args:
            nu (:term:`array_like`): Mode frequency, :math:`\\nu`.
            a (:term:`array_like`): Amplitude, :math:`a_\\mathrm{CZ}`.

        Returns:
            jax.numpy.ndarray: Base of the convective zone glitch amplitude.
        """
        return jnp.divide(a, nu**2)
    
    def __call__(self) -> Callable:
        """Samples the convective zone glitch function.

        Returns:
            function: The function :math:`f`.
        """
        log_a = numpyro.sample('log_a_cz', self.log_a)
        log_tau = numpyro.sample('log_tau_cz', self.log_tau)
        
        a = numpyro.deterministic('a_cz', 10**log_a)
        tau = numpyro.deterministic('tau_cz', 10**log_tau)
        phi = numpyro.sample('phi_cz', self.phi)
        
        def fn(nu):
            return self.amplitude(nu, a) * self.oscillation(nu, tau, phi)
        return fn


class GlitchModel(Model):
    r"""Asteroseismic glitch model.

    .. math::
    
        \nu_\mathrm{obs} \sim \mathcal{GP}(m(n), k(n, n') + 
        \sigma^2\mathcal{I})

    Where the mean function is,

    .. math::

        m(n) &= \nu_\mathrm{bkg} + \delta_\mathrm{He} + \delta_\mathrm{CZ},\\
        \nu_\mathrm{bkg} &= f_\mathrm{bkg}(n),\\
        \delta_\mathrm{He} &= f_\mathrm{He}(\nu_\mathrm{bkg}),\\
        \delta_\mathrm{CZ} &= f_\mathrm{CZ}(\nu_\mathrm{bkg}),
    
    and the kernel function is,

    .. math::

        k(n, n') = \sigma_k^2 \exp\left( - \frac{(n' - n)^2}{l^2} \right).

    Args:
        background (Model): Background prior model which, when called, returns
            a function :math:`f_\mathrm{bkg}` describing the smoothly varying
            (non-glitch) component of the oscillation modes.
        he_glitch (Model): Glitch prior model which, when called, returns a
            function :math:`f_\mathrm{He}` describing the contribution to the
            modes from the glitch due to the second ionisation of helium in the
            stellar convective envelope.
        cz_glitch (Model): Convective zone glitch prior model which, when
            called, returns a function :math:`f_\mathrm{He}` describing the
            contribution to the modes from the glitch due to the base of the
            convection zone.

    """
    def __init__(
        self,
        n: ArrayLike,
        background: Model,
        he_glitch: Optional[Model]=None,
        cz_glitch: Optional[Model]=None,
        num_pred: int=250,
    ):
        
        self.n = np.asarray(n)
        """:numpy.ndarray: Radial order of observations."""
        self.n_pred = np.linspace(n[0], n[-1], num_pred)
        """:numpy.ndarray: Radial order of predictions."""
        self.background: Model = background
        """:Model: Background function prior."""
        if he_glitch is None:
            he_glitch = ZerosFunction()
        if cz_glitch is None:
            cz_glitch = ZerosFunction()
        self.he_glitch: Model = he_glitch
        """:Model: Helium glitch function prior."""

        self.cz_glitch: Model = cz_glitch
        """:Model: Convective zone glitch function prior."""

        self.units = {
            'nu_obs': u.microhertz,
            'nu': u.microhertz,
            'nu_bkg': u.microhertz,
            'dnu_he': u.microhertz,
            'dnu_cz': u.microhertz,
            'noise': u.microhertz,
        }

        self.symbols = {
            'nu_obs': r'$\nu_\mathrm{obs}$',
            'nu': r'$\nu$',
            'nu_bkg': r'$\nu_\mathrm{bkg}$',
            'dnu_he': r'$\delta\nu_\mathrm{He}$',
            'dnu_cz': r'$\delta\nu_\mathrm{BCZ}$',
            'noise': r'$\sigma_\mathrm{WN}$',
        }

        for model in [self.background, self.he_glitch, self.cz_glitch]:
            self.units.update(model.units)
            self.symbols.update(model.symbols)

    # def plot_glitch(self, samples: dict, nu: ArrayLike=None, 
    #                 nu_err: ArrayLike=None, kind: str='He',
    #                 quantiles: Optional[list]=[.16, .84], 
    #                 ax: plt.Axes=None) -> plt.Axes:
    #     """Plot the glitch.

    #     Args:
    #         samples (dict): Predictive samples (e.g. from MCMC or Predictive).
    #         nu (:term:`array_like`): Observed mode frequencies.
    #         nu_err (:term:`array_like`): Observed mode frequency uncertainty.
    #         kind (str): Kind of glitch to plot. One of ['He', 'CZ'].
    #         group (str): Inference data group to plot. One of ['prior',
    #             'posterior'] is supported.
    #         quantiles (iterable, optional): Quantiles to plot as confidence
    #             intervals. Defaults to no confidence intervals drawn.
    #         ax (matplotlib.axes.Axes): Axis on which to plot the glitch.

    #     Returns:
    #         matplotlib.axes.Axes: Axis on which the glitch is plot.
    #     """
    #     if ax is None:
    #         ax = plt.gca()

    #     kindl = kind.lower()
        
    #     # Account for case where first dimension is num_chains
    #     shape = samples['nu'].shape
    #     assert len(shape) > 1
    #     num_chains = 1
    #     if len(shape) == 3:
    #         num_chains = shape[0]
    #     new_shape = (num_chains * shape[-2], shape[-1])

    #     dnu = samples['dnu_'+kindl].reshape(new_shape)

    #     if nu is not None:
    #         res = nu - samples['nu'].reshape(new_shape)
    #         dnu_obs = dnu + res 
            
    #         # TODO: good way to show model error on dnu_obs here??
    #         ax.errorbar(self.n, np.median(dnu_obs, axis=0),
    #                     yerr=nu_err, color='C0', marker='o',
    #                     linestyle='none', label='observed')
        
    #     # TODO: if not pred, then just show the model dnu with errorbars
    #     # according to quantiles
    #     if 'nu_pred' in samples.keys():
    #         shape = samples['nu_pred'].shape
    #         new_shape = (num_chains * shape[-2], shape[-1])
    #         dnu_pred = samples['dnu_'+kindl+'_pred'].reshape(new_shape)
    #         dnu_med = np.median(dnu_pred, axis=0)
    #         ax.plot(self.n_pred, dnu_med, label='median', color='C1')

    #         if quantiles is not None:
    #             dnu_quant = np.quantile(dnu_pred, quantiles, axis=0)
    #             num_quant = len(quantiles)//2
    #             alphas = np.linspace(0.1, 0.5, num_quant*2+1)
    #             for i in range(num_quant):
    #                 delta = quantiles[-i-1] - quantiles[i]
    #                 ax.fill_between(self.n_pred, dnu_quant[i], dnu_quant[-i-1],
    #                                 color='C1', alpha=alphas[2*i+1],
    #                                 label=f'{delta:.1%} CI')
        
    #     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #     ax.set_xlabel(r'$n$')
    #     var = r'$\delta\nu_\mathrm{\,'+kind+r'}$'
    #     unit = f"({self.units[f'dnu_{kindl}'].to_string(format='latex_inline')})"
    #     ax.set_ylabel(' '.join([var, unit]))
    #     ax.legend()
    #     return ax

    def predict(self, nu: ArrayLike=None, nu_err: ArrayLike=None):
        return self(nu=nu, nu_err=nu_err, pred=True)

    def __call__(self, nu: ArrayLike=None,
                 nu_err: ArrayLike=None, pred: bool=False):
        """Sample the model for given observables.

        Args:
            nu (:term:`array_like`, optional): Observed radial mode frequencies.
            nu_err (:term:`array_like`, optional): Gaussian observational uncertainties (sigma) for nu.
            pred (bool): If True, make predictions nu and nu_pred from n and num_pred.
            num_pred (int): Number of predictions in the range n.min() to n.max().
        """
        # TODO it would be more general for all models to take an obs dict as
        # argument and every parameter to do obs.get('name', None)
        n = self.n
        n_pred = self.n_pred
        bkg_func = self.background()
        he_glitch_func = self.he_glitch()
        cz_glitch_func = self.cz_glitch()

        def mean(n):
            nu_bkg = bkg_func(n)
            return nu_bkg + he_glitch_func(nu_bkg) + cz_glitch_func(nu_bkg)

        var = numpyro.param('kernel_var', 10.0)
        # length = numpyro.sample('kernel_length', dist.Gamma(5.0))
        length = numpyro.param('kernel_length', 4.0)

        noise = numpyro.sample('noise', dist.HalfNormal(0.1))
        if nu_err is not None:
            noise += nu_err
            # noise = jnp.sqrt(noise**2 + nu_err**2)
        kernel = SquaredExponential(var, length)
        gp = GP(kernel, mean=mean)
        
        with dimension('n', n.shape[-1], coords=n):
            gp.sample('nu_obs', n, noise=noise, obs=nu)
            
            if pred:
                gp.predict('nu', n)

            nu_bkg = numpyro.deterministic('nu_bkg', bkg_func(n))
            numpyro.deterministic('dnu_he', he_glitch_func(nu_bkg))
            numpyro.deterministic('dnu_cz', cz_glitch_func(nu_bkg))

        if pred:
            with dimension('n_pred', n_pred.shape[-1], coords=n_pred):
                gp.predict('nu_pred', n_pred)
                nu_bkg = numpyro.deterministic('nu_bkg_pred', bkg_func(n_pred))
                numpyro.deterministic('dnu_he_pred', he_glitch_func(nu_bkg))
                numpyro.deterministic('dnu_cz_pred', cz_glitch_func(nu_bkg))
