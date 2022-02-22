"""Probabilistic models for asteroseismic oscillation mode frequencies.
"""
from __future__ import annotations

import numpyro
import numpy as np
import astropy.units as u

from numpy.typing import ArrayLike
from typing import Optional
from jax import random

from .annotations import DistLike
from .gp import GP
from .gp.kernels import SquaredExponential
from .priors import (AsyFunction, CZGlitchFunction, HeGlitchFunction, TauPrior, 
                     Prior)
from .messengers import dimension
from .utils import distribution

__all__ = [
    "Model",
    "GlitchModel",
]


class Model(Prior):
    """Model class.
    
    A model is a probabilistic object which may be given to Inference. It does
    not need to return anything during inference, but should have at least
    one observed sample sites.
    """
    def predict(self, *args, **kwargs):
        """Model predictions. By default this calls the model.
        """
        return self(*args, **kwargs)

    def __call__(self, nu=None, nu_err=None):
        """Call the model during inference.

        Args:
            nu (:term:`array_like`, optional): Observed radial mode frequencies.
            nu_err (:term:`array_like`, optional): Gaussian observational uncertainties (sigma) for nu.
        
        Raises:
            NotImplementedError: This is an abstract class and cannot be 
                called.
        """
        raise NotImplementedError


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
        n (:term:`array_like`): Radial order of model observations.
        nu_max (:term:`dist_like`): Prior on the frequency at maximum power.
        delta_nu (:term:`dist_like`): Prior on the large frequency separation.
        teff (:term:`dist_like`, optional): Prior on the effective temperature.
            This is used for estimating a prior on the glitch acoustic depths.
            If None (default), a prior of Normal(5000, 700) is assumed.
        epsilon (:term:`dist_like`, optional): Prior on the asymptotic phase
            parameter.
        num_pred (int): The number of points in radial order for
            which to make predictions.

    Attributes:
        n (numpy.ndarray): Radial order of model observations.
        n_pred (numpy.ndarray): Radial order of model predictions.
        background (Prior): Prior on the background function.
        he_glitch (Prior): Prior on the helium glitch function.
        cz_glitch (Prior): Prior on the base of convective zone glitch
            function.
    """
    def __init__(
        self,
        n: ArrayLike,
        nu_max: DistLike,
        delta_nu: DistLike,
        teff: Optional[DistLike]=None,
        epsilon: Optional[DistLike]=None,
        # background: Prior,
        # he_glitch: Optional[Prior]=None,
        # cz_glitch: Optional[Prior]=None,
        num_pred: int=250,
        seed: int=0,
    ):  
        super().__init__(n, nu_max, delta_nu, teff=teff, epsilon=epsilon,
                         num_pred=num_pred, seed=seed)
        self.n = np.asarray(n)
        self.n_pred = np.linspace(n[0], n[-1], num_pred)

        # background (Prior): Background prior model which, when called, returns
        #     a function :math:`f_\mathrm{bkg}` describing the smoothly varying
        #     (non-glitch) component of the oscillation modes.
        # he_glitch (Prior): Glitch prior model which, when called, returns a
        #     function :math:`f_\mathrm{He}` describing the contribution to the
        #     modes from the glitch due to the second ionisation of helium in the
        #     stellar convective envelope.
        # cz_glitch (Prior): Convective zone glitch prior model which, when
        #     called, returns a function :math:`f_\mathrm{He}` describing the
        #     contribution to the modes from the glitch due to the base of the
        #     convection zone.

        self.background: Prior = AsyFunction(delta_nu, epsilon=epsilon)
        
        key = random.PRNGKey(seed)
        prior = TauPrior(nu_max, teff)
        log_tau_he, log_tau_cz = prior.condition(key, kind='optimized', 
                                                 num_samples=1000)
        
        self.he_glitch: Prior = HeGlitchFunction(nu_max, log_tau=log_tau_he)
        self.cz_glitch: Prior = CZGlitchFunction(nu_max, log_tau=log_tau_cz)

        self._nu_max = distribution(nu_max)
        
        self.units = {
            'nu_obs': u.microhertz,
            'nu': u.microhertz,
            'nu_bkg': u.microhertz,
            'dnu_he': u.microhertz,
            'dnu_cz': u.microhertz,
            'he_nu_max': u.microhertz,
            # 'cz_nu_max': u.microhertz,
            'he_amplitude': u.microhertz,
            # 'noise': u.microhertz,
        }

        self.symbols = {
            'nu_obs': r'$\nu_\mathrm{obs}$',
            'nu': r'$\nu$',
            'nu_bkg': r'$\nu_\mathrm{bkg}$',
            'dnu_he': r'$\delta\nu_\mathrm{He}$',
            'dnu_cz': r'$\delta\nu_\mathrm{BCZ}$',
            'he_nu_max': r'$A_\mathrm{He}(\nu_\max)$',
            # 'cz_nu_max': r'$A_\mathrm{BCZ}(\nu_\max)$',
            'he_amplitude': r'$\langle A_\mathrm{He} \rangle$',
            # 'noise': r'$\sigma_\mathrm{WN}$',
        }

        for prior in [self.background, self.he_glitch, self.cz_glitch]:
            # Inherit units from priors.
            self.units.update(prior.units)
            self.symbols.update(prior.symbols)

        self._kernel_var = 0.1 * self.background.delta_nu.mean
        # super().__init__(symbols=symbols, units=units)
        # self._init_arguments(n, nu_max, delta_nu, teff=teff, epsilon=epsilon,
        #                      num_pred=num_pred, seed=seed)

    def predict(self, nu: ArrayLike=None, nu_err: ArrayLike=None):
        # In some models we may not want to pass nu to make predictions.
        # The predict method allows for control over this.
        return self(nu=nu, nu_err=nu_err, pred=True)
        
    def __call__(self, nu: ArrayLike=None,
                 nu_err: ArrayLike=None, pred: bool=False):
        """Sample the model for given observables.

        Args:
            nu (:term:`array_like`, optional): Observed radial mode frequencies.
            nu_err (:term:`array_like`, optional): Gaussian observational uncertainties (sigma) for nu.
            pred (bool): If True, make predictions nu and nu_pred from n and num_pred.
        """
        # TODO it would be more general for all models to take an obs dict as
        # argument and every parameter to do obs.get('name', None)
        n = self.n
        n_pred = self.n_pred
        bkg_func = self.background()
        he_glitch_func = self.he_glitch()
        cz_glitch_func = self.cz_glitch()
        
        _nu_max = numpyro.sample('_nu_max', self._nu_max)
        numpyro.deterministic('he_nu_max', self.he_glitch.amplitude(_nu_max))
        # numpyro.deterministic('cz_nu_max', self.cz_glitch.amplitude(_nu_max))
        
        low = _nu_max - 5 * self.background._delta_nu
        high = _nu_max + 5 * self.background._delta_nu
        numpyro.deterministic('he_amplitude',
                              self.he_glitch._average_amplitude(low, high))

        def mean(n):
            nu_bkg = bkg_func(n)
            return nu_bkg + he_glitch_func(nu_bkg) + cz_glitch_func(nu_bkg)

        var = numpyro.param('kernel_var', self._kernel_var)
        length = numpyro.param('kernel_length', 5.0)

        # noise = numpyro.sample('noise', dist.HalfNormal(0.1))
        # if nu_err is not None:
        #     noise += nu_err
        #     # noise = jnp.sqrt(noise**2 + nu_err**2)
        kernel = SquaredExponential(var, length)
        gp = GP(kernel, mean=mean)
        
        with dimension('n', n.shape[-1], coords=n):
            gp.sample('nu_obs', n, noise=nu_err, obs=nu)
            
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
