"""Prior models."""
from __future__ import annotations

import os, jax
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import astropy.units as u

from jax import random
from typing import Callable, Dict, Iterable, Optional
from numpy.typing import ArrayLike

from .utils import PACKAGE_DIR, distribution
from .typing import DistLike

__all__ = [
    "Prior",
    "ZerosFunction",
    "AsyFunction",
    "HeGlitchFunction",
    "CZGlitchFunction",
    "TauPrior",
]


class Prior:
    """Prior class.

    A prior is a model which returns a parameter or function when called and
    has no observed sample sites.

    Args:
        *args: Positional arguments to display in the object representation.
        **kwargs: Keyword arguments to display in the object representation.

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
        return " " + "\n ".join(s.splitlines())

    def __repr__(self) -> str:
        name = self.__class__.__name__
        args = ",\n".join([repr(a) for a in self._init_args])
        kwargs = ["=".join([k, repr(v)]) for k, v in self._init_kwargs.items()]
        kwargs = ",\n".join(kwargs)
        s = ",\n".join([args, kwargs])
        return f"{name}(\n{self._indent(s)}\n)"

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
        return f"{self.__class__.__name__}()"

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
            :class:`dist.Normal`, or a :class:`dist.Distribution`.
        epsilon (:term:`dist_like`): Prior for the phase term
            :math:`\\epsilon`. Pass either the arguments of
            :class:`dist.LogNormal`, or a :class:`dist.Distribution`. Defaults
            to :code:`(np.log(1.4), 0.4)`.

    Attributes:
        delta_nu (numpyro.distributions.distribution.Distribution): The
            distribution for :math:`\\Delta\\nu`.
        epsilon (numpyro.distributions.distribution.Distribution): The
            distribution for :math:`\\epsilon`.
    """

    def __init__(self, delta_nu: DistLike, epsilon: DistLike = None):
        super().__init__(delta_nu, epsilon=epsilon)

        self.units = {
            "delta_nu": u.microhertz,
            "epsilon": u.dimensionless_unscaled,
        }
        self.symbols = {
            "delta_nu": r"$\Delta\nu$",
            "epsilon": r"$\epsilon$",
        }

        self.delta_nu: dist.Distribution = distribution(delta_nu)

        if epsilon is None:
            epsilon = (np.log(1.4), 0.4)
        self.epsilon: dist.Distribution = distribution(epsilon, dist.LogNormal)

    def __repr__(self) -> str:
        name = self.__class__.__name__

        s = (
            f"{repr(self.delta_nu)},\n"
            + f"epsilon=\\\n {self._indent(repr(self.epsilon))}"
        )
        return f"{name}(\n {self._indent(s)}\n)"

    def __call__(self) -> Callable:
        """Samples the prior for the linear asymptotic function.

        Returns:
            function: The function :math:`f`.
        """
        self._delta_nu = numpyro.sample("delta_nu", self.delta_nu)
        self._epsilon = numpyro.sample("epsilon", self.epsilon)

        def fn(n):
            return self._delta_nu * (n + self._epsilon)

        return fn


class _GlitchFunction(Prior):
    """Prior on the glitch oscillation function :math:`f`, where
    :math:`f(nu) = \\sin(4\\pi\\tau\\nu + \\phi)`.

    Args:
        log_tau (:term:`dist_like`): The prior for the acoustic depth of the
            glitch, :math:`\\tau`. Pass either the arguments of
            :class:`dist.Normal`, or a :class:`dist.Distribution`.
        phi (:term:`dist_like`): The prior for the phase of the glitch,
            :math:`\\phi`. Pass either the arguments of :class:`dist.Uniform`
            or a :class:`dist.Distribution`.

    Attributes:
        log_tau (numpyro.distributions.distribution.Distribution): The
            distribution for :math:`\\log\\tau`.
        phi (numpyro.distributions.distribution.Distribution): The distribution
            for :math:`\\phi`.
    """

    def __init__(
        self, *args, log_tau: DistLike, phi: DistLike = None, **kwargs
    ):
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
        log_tau = numpyro.sample("log_tau", self.log_tau)
        self._tau = numpyro.deterministic("tau", 10**log_tau)
        self._phi = numpyro.sample("phi", self.phi)

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
            :class:`dist.Normal`, or a :class:`dist.Distribution`.
        log_tau (:term:`dist_like`): The prior for the acoustic depth of the
            glitch, :math:`\\tau_\\mathrm{He}`. Pass either the arguments of
            :class:`dist.Normal`, or a :class:`dist.Distribution`.
        phi (:term:`dist_like`): The prior for the phase of the glitch,
            :math:`\\phi_\\mathrm{He}`. Pass either the arguments of
            :class:`dist.Uniform`, or a :class:`dist.Distribution`.

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

    def __init__(
        self, nu_max: DistLike, log_tau: DistLike, phi: DistLike = None
    ):
        super().__init__(nu_max, log_tau=log_tau, phi=phi)
        self.units = {
            "a_he": u.dimensionless_unscaled,
            "b_he": u.megasecond**2,
            "tau_he": u.megasecond,
            "phi_he": u.rad,
        }
        self.symbols = {
            "a_he": r"$a_\mathrm{He}$",
            "b_he": r"$b_\mathrm{He}$",
            "tau_he": r"$\tau_\mathrm{He}$",
            "phi_he": r"$\phi_\mathrm{He}$",
        }
        # log units
        for k in ["a_he", "b_he", "tau_he"]:
            log_k = f"log_{k}"
            self.units[log_k] = u.LogUnit(self.units[k])
            self.symbols[log_k] = r"$\log\," + self.symbols[k][1:]

        log_numax = jnp.log10(distribution(nu_max).mean)
        # Attempt rough guess of glitch params
        self.log_a: dist.Distribution = dist.Normal(-log_numax, 1.0)
        self.log_b: dist.Distribution = dist.Normal(-7.0, 2.0)

    def amplitude(self, nu: ArrayLike) -> jnp.ndarray:
        r"""The amplitude of the glitch,
        :math:`a_\\mathrm{He} \\nu \\exp(-b_\\mathrm{He} \\nu^2)`.

        Args:
            nu (:term:`array_like`): Mode frequency, :math:`\\nu`.

        Returns:
            jax.numpy.ndarray: Helium glitch amplitude.
        """
        return self._a * nu * jnp.exp(-self._b * nu**2)

    def _average_amplitude(self, low: ArrayLike, high: ArrayLike):
        """The average amplitude over the glitch for a given frequency range."""
        return (
            self._a
            * (jnp.exp(-self._b * low**2) - jnp.exp(-self._b * high**2))
            / (2 * self._b * (high - low))
        )

    def __call__(self) -> Callable:
        """Samples the helium glitch function.

        Returns:
            function: The function :math:`f`.
        """
        log_a = numpyro.sample("log_a_he", self.log_a)
        log_b = numpyro.sample("log_b_he", self.log_b)
        log_tau = numpyro.sample("log_tau_he", self.log_tau)

        self._a = numpyro.deterministic("a_he", 10**log_a)
        self._b = numpyro.deterministic("b_he", 10**log_b)
        self._tau = numpyro.deterministic("tau_he", 10**log_tau)
        self._phi = numpyro.sample("phi_he", self.phi)

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
            :class:`dist.Normal`, or a :class:`dist.Distribution`.
        phi (:term:`dist_like`): The prior for the phase of the glitch,
            :math:`\\phi_\\mathrm{BCZ}`. Pass either the arguments of
            :class:`dist.Uniform`, or a :class:`dist.Distribution`.

    Attributes:
        log_a (numpyro.distributions.distribution.Distribution): The
            distribution for the glitch amplitude parameter
            :math:`\\log a_\\mathrm{BCZ}`
        log_tau (numpyro.distributions.distribution.Distribution): The
            distribution for :math:`\\log\\tau_\\mathrm{BCZ}`.
        phi (numpyro.distributions.distribution.Distribution): The distribution
            for :math:`\\phi_\\mathrm{BCZ}`.
    """

    def __init__(
        self, nu_max: DistLike, log_tau: DistLike, phi: DistLike = None
    ):
        super().__init__(nu_max, log_tau=log_tau, phi=phi)

        self.units = {
            "a_cz": u.microhertz**3,
            "tau_cz": u.megasecond,
            "phi_cz": u.rad,
        }
        self.symbols = {
            "a_cz": r"$a_\mathrm{BCZ}$",
            "tau_cz": r"$\tau_\mathrm{BCZ}$",
            "phi_cz": r"$\phi_\mathrm{BCZ}$",
        }

        # log units
        for k in ["a_cz", "tau_cz"]:
            log_k = f"log_{k}"
            self.units[log_k] = u.LogUnit(self.units[k])
            self.symbols[log_k] = r"$\log\," + self.symbols[k][1:]

        log_numax = jnp.log10(distribution(nu_max).mean)
        # Rough guess of glitch params
        self.log_a: dist.Distribution = dist.Normal(2.0 * log_numax - 1.5, 1.0)

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
        log_a = numpyro.sample("log_a_cz", self.log_a)
        log_tau = numpyro.sample("log_tau_cz", self.log_tau)

        self._a = numpyro.deterministic("a_cz", 10**log_a)
        self._tau = numpyro.deterministic("tau_cz", 10**log_tau)
        self._phi = numpyro.sample("phi_cz", self.phi)

        def fn(nu):
            return self.amplitude(nu) * self.oscillation(nu)

        return fn


class TauPrior(Prior):
    """Prior on the acoustic depths of helium and BCZ glitches.

    Args:
        nu_max (:term:`dist_like`): Frequency at maximum power (uHz).
        teff (:term:`dist_like`): Effective temperature (K).
    """

    def __init__(
            self,
            nu_max: DistLike,
            teff: DistLike = None,
            noise: float = 0.005
        ) -> None:
        super().__init__(nu_max, teff=teff)
        self.nu_max = distribution(nu_max)

        if teff is None:
            teff = (5000.0, 800.0)  # Uninformative prior
        self.teff = distribution(teff)
        
        # Load weights, loc and cov
        from .data import tau_prior
        self.loc = jnp.array(tau_prior["loc"])
        self.cov = jnp.array(tau_prior["cov"])
        self.weights = jnp.array(tau_prior["weights"])
        self.noise = 0.005

    def __call__(self):
        assignment = numpyro.sample("assignment", dist.Categorical(self.weights))
        
        loc = self.loc[assignment]
        cov = self.cov[assignment]

        nu_max = numpyro.sample("nu_max", self.nu_max)
        log_nu_max = jnp.log10(nu_max)

        teff = numpyro.sample("teff", self.teff)

        loc0101 = loc[0:2]
        cov0101 = jnp.array([
            [cov[0, 0], cov[0, 1]],
            [cov[1, 0], cov[1, 1]]
        ])
                                        
        L = jax.scipy.linalg.cho_factor(cov0101, lower=True)
        A = jax.scipy.linalg.cho_solve(L, jnp.array([log_nu_max, teff]) - loc0101)
        
        loc2323 = loc[2:]
        cov2323 = jnp.array([
            [cov[2, 2], cov[2, 3]],
            [cov[3, 2], cov[3, 3]]
        ])  
        
        cov0123 = jnp.array([
            [cov[0, 2], cov[1, 2]],
            [cov[0, 3], cov[1, 3]]
        ])
        v = jax.scipy.linalg.cho_solve(L, cov0123.T)
        
        cond_loc = loc2323 + jnp.dot(cov0123, A)
        cond_cov = (
            cov2323 - jnp.dot(cov0123, v) 
            + self.noise * jnp.eye(2)  # Add white noise
        )
        numpyro.sample("log_tau", dist.MultivariateNormal(cond_loc, cond_cov))
