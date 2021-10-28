from __future__ import annotations

import math

import numpy as np

import jax
import jax.numpy as jnp

from jax import random

import numpyro
import numpyro.distributions as dist

from numpyro.infer.reparam import CircularReparam, Reparam
from numpyro import handlers
from numpyro.primitives import Messenger, plate

from typing import Union, Optional, Callable, Dict, ClassVar, Any
from .annotations import Array1D, Array2D, Array3D

from .gp import GP, SquaredExponential

from collections.abc import Iterable
import astropy.units as u

class Prior:
    """Base Prior class.
    """
    reparam = None
    units = {}
    def _init_prior(self, value, default_dist=dist.Normal):
        if not isinstance(value, dist.Distribution):
            if not isinstance(value, Iterable):
                value = (value,)
            value = default_dist(*value)
        return value
    def __call__(self):
        """Samples the prior.
        """
        raise NotImplementedError

class ZerosPrior(Prior):
    """Zeros prior.
    """
    def __call__(self):
        return lambda x: jnp.zeros(x.shape)


class AsyPrior(Prior):
    """Prior on the linear asymptotic function f, where
        f(n) = delta_nu * (n + epsilon)
    """
    def __init__(self, delta_nu, epsilon=None):
        self.delta_nu = self._init_prior(delta_nu)
        if epsilon is None:
            epsilon = (14., 10.)
        self.epsilon = self._init_prior(epsilon, dist.Gamma)
    
    def __call__(self):
        delta_nu = numpyro.sample('delta_nu', self.delta_nu)
        epsilon = numpyro.sample('epsilon', self.epsilon)
        
        def fn(n):
            return delta_nu * (n + epsilon)
        return fn


class GlitchPrior(Prior):
    """Prior on the glitch oscillation function f, where,
        f(nu) = sin(4*pi*tau*nu + phi)

    When performing inference, use the following reparameterisation
    """
    reparam = numpyro.handlers.reparam(config={'phi': CircularReparam()})

    def __init__(self, tau, phi):
        self.tau = self._init_prior(tau)
        self.phi = self._init_prior(phi, dist.VonMises)

    @staticmethod
    def oscillation(nu, tau, phi):
        return jnp.sin(4 * jnp.pi * tau * nu + phi)

    def __call__(self):
        tau = numpyro.sample('tau', self.tau)
        phi = numpyro.sample('phi', self.phi)
        def fn(nu):
            return self.oscillation(nu, tau, phi)
        return fn


class HeGlitchPrior(GlitchPrior):
    """Prior on the 2nd ionisation of helium glitch function f, where
        f(nu) = A(nu) * sin(4*pi*tau*nu + phi)
        A(nu) = a * nu * exp(-b * nu**2)

    When performing inference, reperameterise using

    he_glitch = HeGlitchPrior(...)
    ...
    model = ...(..., he_glitch, ...)
    handlers = [..., he_glitch.reparam, ...]
    infer = Inference(model, ...)
    infer.sample(..., sampler_kwargs={'handlers': handlers})

    Args:
        nu_max: [description]

    """
    reparam = numpyro.handlers.reparam(config={'phi_he': CircularReparam()})
    units = {
        'a_he': u.dimensionless_unscaled,
        'b_he': u.megasecond**2,
        'tau_he': u.megasecond,
        'phi_he': u.rad,
    }
    def __init__(self, nu_max):
        self.nu_max = nu_max
        self.phi = dist.VonMises(0.0, 0.1)

    @property
    def nu_max(self):
        return self._nu_max

    @nu_max.setter
    def nu_max(self, value):
        self._nu_max = self._init_prior(value)
        log_numax = jnp.log10(self._nu_max.mean)
        self.log_a = dist.Normal(-1.10 - 0.35*log_numax, 0.7)
        self.log_b = dist.Normal(0.719 - 2.14*log_numax, 0.7)
        self.log_tau = dist.Normal(0.44 - 1.03*log_numax, 0.1)
    
    @staticmethod
    def amplitude(nu, a, b):
        return a * nu * jnp.exp(- b * nu**2)
    
    def __call__(self):
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


class CZGlitchPrior(GlitchPrior):
    """Prior on the base of the convective zone glitch function f, where
        f(nu) = a / nu**2 * sin(4*pi*tau*nu + phi)

    When performing inference, reperameterise using

    cz_glitch = CZGlitchPrior(...)
    ...
    model = ...(..., cz_glitch, ...)
    handlers = [..., cz_glitch.reparam, ...]
    infer = Inference(model, ...)
    infer.sample(..., sampler_kwargs={'handlers': handlers})

    Args:
        nu_max: [description]

    """
    reparam = numpyro.handlers.reparam(config={'phi_cz': CircularReparam()})
    units = {
        'a_cz': u.microhertz**3,
        'tau_cz': u.megasecond,
        'phi_cz': u.rad,
    }
    def __init__(self, nu_max):
        self.nu_max = nu_max
        self.phi = dist.VonMises(0.0, 0.1)

    @property
    def nu_max(self):
        return self._nu_max

    @nu_max.setter
    def nu_max(self, value):
        self._nu_max = self._init_prior(value)
        log_numax = jnp.log10(self._nu_max.mean)
        self.log_a = dist.Normal(2*log_numax - 1.0, 0.7)
        self.log_tau = dist.Normal(0.77 - 0.99*log_numax, 0.1)
    
    @staticmethod
    def amplitude(nu, a):
        return a / nu**2
    
    def __call__(self):
        log_a = numpyro.sample('log_a_cz', self.log_a)
        log_tau = numpyro.sample('log_tau_cz', self.log_tau)
        
        a = numpyro.deterministic('a_cz', 10**log_a)
        tau = numpyro.deterministic('tau_cz', 10**log_tau)
        phi = numpyro.sample('phi_cz', self.phi)
        
        def fn(nu):
            return self.amplitude(nu, a) * self.oscillation(nu, tau, phi)
        return fn
