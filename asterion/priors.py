from __future__ import annotations

import os
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import astropy.units as u

from jax import random
from typing import Callable, Dict, Iterable, Optional
from numpy.typing import ArrayLike
from pprint import pformat

from . import PACKAGE_DIR
from .nn import BayesianNN
from .annotations import DistLike

__all__ = [
    "distribution",
    "Prior",
    "ZerosFunction",
    "AsyFunction",
    "HeGlitchFunction",
    "CZGlitchFunction",
    "TauPrior",
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


class Prior:
    """Prior class.
    
    A prior is a model which returns a parameter or function when called and
    has no observed sample sites.
    
    Args:
        symbols (dict, optional): Dictionary mapping model variable names to 
            their mathematical symbols.
        units (dict, optional): Dictionary mapping model variable names to
            their units.
 
    Attributes:
        symbols (dict): Dictionary mapping model variable names to their
            mathematical symbols.
        units (dict): Dictionary mapping model variable names to their units.
    """
    def __init__(self, *args, **kwargs):
        self._init_arguments(*args, **kwargs)
        self.symbols: Dict[str, str]
        self.units: Dict[str, u.Unit]
    
    def _init_arguments(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs

    @staticmethod
    def _indent(s):
        return ' ' + '\n '.join(s.splitlines())

    def __repr__(self) -> str:
        name = self.__class__.__name__
        args = ',\n'.join([repr(a) for a in self._init_args])
        kwargs = ['='.join([k, repr(v)]) for k, v in self._init_kwargs.items()]
        kwargs = ',\n'.join(kwargs)
        s = ',\n'.join([args, kwargs])
        return f'{name}(\n{self._indent(s)}\n)'
   
    def __call__(self) -> ...:
        """Call the model during inference.

        Raises:
            NotImplementedError: This is an abstract base class and cannot be
                called.
        """
        raise NotImplementedError


class ZerosFunction(Prior):
    """A prior on the zeros function :math:`f` where
    :math:`f(\\boldsymbol{x}) = \\boldsymbol{0}`.
    """
    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __call__(self) -> Callable:
        """Samples the prior for the zeros function.

        Returns:
            function: The function :math:`f`.
        """
        return lambda x: jnp.zeros(x.shape)


class AsyFunction(Prior):
    """Prior on the linear asymptotic function :math:`f`, where
    :math:`f(n) = \\Delta\\nu (n + \\epsilon)`.
    
    Args:
        delta_nu (:term:`dist_like`): Prior for the large frequency separation
            :math:`\\Delta\\nu`. Pass either the arguments of
            :class:`dist.Normal` or a :class:`dist.Distribution`.
        epsilon (:term:`dist_like`): Prior for the phase term
            :math:`\\epsilon`. Pass either the arguments of :class:`dist.Gamma`
            or a :class:`dist.Distribution`. Defaults to :code:`(14., 10.)`.
    
    Attributes:
        delta_nu (numpyro.distributions.distribution.Distribution): The
            distribution for :math:`\\Delta\\nu`.
        epsilon (numpyro.distributions.distribution.Distribution): The
            distribution for :math:`\\epsilon`. 
    """
    def __init__(self, delta_nu: DistLike, epsilon: DistLike=None):
        super().__init__(delta_nu, epsilon=epsilon)

        self.units = {
            'delta_nu': u.microhertz,
            'epsilon': u.dimensionless_unscaled,
        }
        self.symbols = {
            'delta_nu': r'$\Delta\nu$',
            'epsilon': r'$\epsilon$',
        }

        self.delta_nu: dist.Distribution = distribution(delta_nu)

        if epsilon is None:
            epsilon = (np.log(1.4), 0.4)
        self.epsilon: dist.Distribution = distribution(epsilon, dist.LogNormal)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        
        s = f'{repr(self.delta_nu)},\n' + \
            f'epsilon=\\\n {self._indent(repr(self.epsilon))}'
        return f'{name}(\n {self._indent(s)}\n)'

    def __call__(self) -> Callable:
        """Samples the prior for the linear asymptotic function.

        Returns:
            function: The function :math:`f`.
        """
        self._delta_nu = numpyro.sample('delta_nu', self.delta_nu)
        self._epsilon = numpyro.sample('epsilon', self.epsilon)
        
        def fn(n):
            return self._delta_nu * (n + self._epsilon)
        return fn


class _GlitchFunction(Prior):
    """Prior on the glitch oscillation function :math:`f`, where
    :math:`f(nu) = \\sin(4\\pi\\tau\\nu + \\phi)`.

    Args:
        log_tau (:term:`dist_like`): The prior for the acoustic depth of the
            glitch, :math:`\\tau`. Pass either the arguments of
            :class:`dist.Normal` or a :class:`dist.Distribution`.
        phi (:term:`dist_like`): The prior for the phase of the glitch,
            :math:`\\phi`. Pass either the arguments of :class:`dist.Uniform`
            or a :class:`dist.Distribution`.
    
    Attributes:
        log_tau (numpyro.distributions.distribution.Distribution): The
            distribution for :math:`\\log\\tau`.
        phi (numpyro.distributions.distribution.Distribution): The distribution
            for :math:`\\phi`.
    """

    def __init__(self, *args, log_tau: DistLike, phi: DistLike=None, **kwargs):
        super().__init__(*args, log_tau=log_tau, phi=phi, **kwargs)
        self.log_tau: dist.Distribution = distribution(log_tau)
        
        if phi is None:
            phi = (-np.pi, np.pi)
        self.phi: dist.Distribution = distribution(phi, dist.Uniform)

    # @staticmethod
    # def oscillation(nu: ArrayLike, tau: ArrayLike,
    #                 phi: ArrayLike) -> jnp.ndarray:
    #     r"""Oscillatory component of the acoustic glitch.

    #     Args:
    #         nu (:term:`array_like`): Mode frequency, :math:`\\nu`.
    #         tau (:term:`array_like`): Acoustic depth, :math:`\\tau`.
    #         phi (:term:`array_like`): Glitch phase, :math:`\\phi`.

    #     Returns:
    #         jax.numpy.ndarray: Glitch oscillation.
    #     """
    #     return jnp.sin(4 * jnp.pi * tau * nu + phi)

    def oscillation(self, nu: ArrayLike) -> jnp.ndarray:
        r"""Oscillatory component of the acoustic glitch.

        Args:
            nu (:term:`array_like`): Mode frequency, :math:`\\nu`.

        Returns:
            jax.numpy.ndarray: Glitch oscillation.
        """
        return jnp.sin(4 * jnp.pi * self._tau * nu + self._phi)

    def __call__(self) -> Callable:
        """Samples the prior for a generic glitch oscillation function.

        Returns:
            function: The function :math:`f`.
        """
        log_tau = numpyro.sample('log_tau', self.log_tau)
        self._tau = numpyro.deterministic('tau', 10**log_tau)
        self._phi = numpyro.sample('phi', self.phi)
        def fn(nu):
            return self.oscillation(nu)
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
        log_tau (:term:`dist_like`): The prior for the acoustic depth of the
            glitch, :math:`\\tau_\\mathrm{He}`. Pass either the arguments of
            :class:`dist.Normal` or a :class:`dist.Distribution`.
        phi (:term:`dist_like`): The prior for the phase of the glitch,
            :math:`\\phi_\\mathrm{He}`. Pass either the arguments of
            :class:`dist.Uniform` or a :class:`dist.Distribution`.
    
    Attributes:
        log_a (numpyro.distributions.distribution.Distribution): The
            distribution for the glitch amplitude parameter
            :math:`\\log a_\\mathrm{He}`.
        log_b (numpyro.distributions.distribution.Distribution): The
            distribution for the glitch decay parameter
            :math:`\\log b_\\mathrm{He}`.
        log_tau (numpyro.distributions.distribution.Distribution): The
            distribution for :math:`\\log\\tau_\\mathrm{He}`.
        phi (numpyro.distributions.distribution.Distribution): The distribution
            for :math:`\\phi_\\mathrm{He}`.
    """
    def __init__(self, nu_max: DistLike, log_tau: DistLike, phi: DistLike=None):
        super().__init__(nu_max, log_tau=log_tau, phi=phi)
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

        log_numax = jnp.log10(distribution(nu_max).mean)
        # Attempt rough guess of glitch params
        # self.log_a: dist.Distribution = dist.Normal(-1.10 - 0.35*log_numax, 0.8)
        self.log_a: dist.Distribution = dist.Normal(-log_numax, 1.0)
        
        # self.log_b: dist.Distribution = dist.Normal(0.719 - 2.14*log_numax, 0.8)
        self.log_b: dist.Distribution = dist.Normal(-6.4, 1.0)
        # self.log_tau = dist.Normal(0.44 - 1.03*log_numax, 0.1)

    # @staticmethod
    # def amplitude(nu: ArrayLike, a: ArrayLike, b: ArrayLike) -> jnp.ndarray:
    #     r"""The amplitude of the glitch,
    #     :math:`a_\\mathrm{He} \\nu \\exp(-b_\\mathrm{He} \\nu^2)`.

    #     Args:
    #         nu (:term:`array_like`): Mode frequency, :math:`\\nu`.
    #         a (:term:`array_like`): Amplitude, :math:`a_\\mathrm{He}`.
    #         b (:term:`array_like`): Decay, :math:`b_\\mathrm{He}`.

    #     Returns:
    #         jax.numpy.ndarray: Helium glitch amplitude.
    #     """
    #     return a * nu * jnp.exp(- b * nu**2)
    
    def amplitude(self, nu: ArrayLike) -> jnp.ndarray:
        r"""The amplitude of the glitch,
        :math:`a_\\mathrm{He} \\nu \\exp(-b_\\mathrm{He} \\nu^2)`.

        Args:
            nu (:term:`array_like`): Mode frequency, :math:`\\nu`.

        Returns:
            jax.numpy.ndarray: Helium glitch amplitude.
        """
        return self._a * nu * jnp.exp(- self._b * nu**2)

    def _average_amplitude(self, low: ArrayLike, high: ArrayLike):
        """The average amplitude over the glitch for a given frequency range.
        """
        return self._a * \
            (jnp.exp(- self._b * low**2) - jnp.exp(- self._b * high**2)) / \
            (2 * self._b * (high - low))

    def __call__(self) -> Callable:
        """Samples the helium glitch function.

        Returns:
            function: The function :math:`f`.
        """
        log_a = numpyro.sample('log_a_he', self.log_a)
        log_b = numpyro.sample('log_b_he', self.log_b)
        log_tau = numpyro.sample('log_tau_he', self.log_tau)
        
        self._a = numpyro.deterministic('a_he', 10**log_a)
        self._b = numpyro.deterministic('b_he', 10**log_b)
        self._tau = numpyro.deterministic('tau_he', 10**log_tau)
        self._phi = numpyro.sample('phi_he', self.phi)
        
        # def fn(nu):
        #     return self.amplitude(nu, a, b) * self.oscillation(nu, tau, phi)
        def fn(nu):
            return self.amplitude(nu) * self.oscillation(nu)
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
        log_tau (:term:`dist_like`): The prior for the acoustic depth of the
            glitch, :math:`\\tau_\\mathrm{BCZ}`. Pass either the arguments of
            :class:`dist.Normal` or a :class:`dist.Distribution`.
        phi (:term:`dist_like`): The prior for the phase of the glitch,
            :math:`\\phi_\\mathrm{BCZ}`. Pass either the arguments of
            :class:`dist.Uniform` or a :class:`dist.Distribution`.
    
    Attributes:
        log_a (numpyro.distributions.distribution.Distribution): The
            distribution for the glitch amplitude parameter
            :math:`\\log a_\\mathrm{BCZ}`
        log_tau (numpyro.distributions.distribution.Distribution): The
            distribution for :math:`\\log\\tau_\\mathrm{BCZ}`.
        phi (numpyro.distributions.distribution.Distribution): The distribution
            for :math:`\\phi_\\mathrm{BCZ}`.
    """
    def __init__(self, nu_max: DistLike, log_tau: DistLike, phi: DistLike=None):        
        super().__init__(nu_max, log_tau=log_tau, phi=phi)

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
        
        log_numax = jnp.log10(distribution(nu_max).mean)
        # Rough guess of glitch params
        self.log_a: dist.Distribution = dist.Normal(2.0 * log_numax - 1.5, 1.0)

        # self.log_tau = dist.Normal(0.77 - 0.99*log_numax, 0.1)
    
    # @staticmethod
    # def amplitude(nu: ArrayLike, a: ArrayLike) -> jnp.ndarray:
    #     """The amplitude of the glitch,
    #     :math:`a_\\mathrm{CZ} / \\nu^{-2}`.

    #     Args:
    #         nu (:term:`array_like`): Mode frequency, :math:`\\nu`.
    #         a (:term:`array_like`): Amplitude, :math:`a_\\mathrm{CZ}`.

    #     Returns:
    #         jax.numpy.ndarray: Base of the convective zone glitch amplitude.
    #     """
    #     return jnp.divide(a, nu**2)

    def amplitude(self, nu: ArrayLike) -> jnp.ndarray:
        """The amplitude of the glitch,
        :math:`a_\\mathrm{CZ} / \\nu^{-2}`.

        Args:
            nu (:term:`array_like`): Mode frequency, :math:`\\nu`.

        Returns:
            jax.numpy.ndarray: Base of the convective zone glitch amplitude.
        """
        return jnp.divide(self._a, nu**2)

    def __call__(self) -> Callable:
        """Samples the convective zone glitch function.

        Returns:
            function: The function :math:`f`.
        """
        log_a = numpyro.sample('log_a_cz', self.log_a)
        log_tau = numpyro.sample('log_tau_cz', self.log_tau)
        
        self._a = numpyro.deterministic('a_cz', 10**log_a)
        self._tau = numpyro.deterministic('tau_cz', 10**log_tau)
        self._phi = numpyro.sample('phi_cz', self.phi)
        
        def fn(nu):
            return self.amplitude(nu) * self.oscillation(nu)
        return fn


class TauPrior(Prior):
    """[summary]

    Args:
        nu_max (:term:`dist_like`): [description]
        teff (:term:`dist_like`): [description]
    """
    def __init__(self, nu_max: DistLike, teff: DistLike=None) -> None:
        super().__init__(nu_max, teff=teff)
        self.nu_max = distribution(nu_max)
        
        if teff is None:
            teff = (5000., 700.)
        self.teff = distribution(teff)
        self.log_tau_he = None
        self.log_tau_cz = None

    def condition(self, rng_key, num_obs=1000, kind='trained', num_samples=None):
        rng_key, key1, key2 = random.split(rng_key, 3)
        sample_shape = (num_obs,)
        log_numax = jnp.log10(
            self.nu_max.sample(key1, sample_shape=sample_shape)
        )
        teff = self.teff.sample(key2, sample_shape=sample_shape)
        x = jnp.stack([log_numax, teff], axis=-1)
            
        prior = BayesianNN.from_file(
            os.path.join(PACKAGE_DIR, 'data', 'tau_prior.hdf5')
        )

        rng_key, key = random.split(rng_key)
        samples = prior.predict(key, x, kind=kind, num_samples=num_samples)
        tau_he = samples['y'][..., 0] - 6  # log mega-seconds
        tau_cz = samples['y'][..., 1] - 6  # log mega-seconds
        log_tau_he = distribution((tau_he.mean(), tau_he.std(ddof=1)))
        log_tau_cz = distribution((tau_cz.mean(), tau_cz.std(ddof=1)))
        return log_tau_he, log_tau_cz
