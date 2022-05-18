"""Probabilistic models for asteroseismic oscillation mode frequencies.
"""
from __future__ import annotations

import numpyro
import numpy as np
import astropy.units as u
import jax.numpy as jnp

from numpy.typing import ArrayLike
from typing import Optional, Union
from jax import random
from numpyro.infer import Predictive
from tinygp import kernels, GaussianProcess

from .typing import DistLike
from .priors import (
    AsyFunction,
    CZGlitchFunction,
    HeGlitchFunction,
    TauPrior,
    Prior,
)
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

    def __call__(self, n, nu=None, nu_err=None, n_pred=None):
        """Call the model during inference.

        Args:
            nu (:term:`array_like`, optional): Observed radial mode
                frequencies.
            nu_err (:term:`array_like`, optional): Gaussian observational
                uncertainties (sigma) for nu.

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
        seed (int): The seed used to generate samples from the prior on the
            glitch periods (acoustic depths) tau_he and tau_cz.
        window_width (float): The number of delta_nu either side of nu_max over
            which to average the helium glitch amplitude for the parameter
            'he_amplitude'.

    Attributes:
        n (numpy.ndarray): Radial order of model observations.
        n_pred (numpy.ndarray): Radial order of model predictions.
        background (Prior): Prior on the background function.
        he_glitch (Prior): Prior on the helium glitch function.
        cz_glitch (Prior): Prior on the base of convective zone glitch
            function.
        window_width (str or float): The number of delta_nu either side of
            nu_max over which to average the glitch amplitudes. If string,
            'full', the window is chosen over the entire range in frequency.
    """

    def __init__(
        self,
        nu_max: DistLike,
        delta_nu: DistLike,
        teff: Optional[DistLike] = None,
        epsilon: Optional[DistLike] = None,
        seed: int = 0,
        window_width: Union[str, float] = "full",
    ):
        super().__init__(
            nu_max,
            delta_nu,
            teff=teff,
            epsilon=epsilon,
            seed=seed,
        )
        self.background: Prior = AsyFunction(delta_nu, epsilon=epsilon)

        key = random.PRNGKey(seed)
        
        # tau_prior = TauPrior(nu_max, teff)
        # log_tau_he, log_tau_cz = self._init_tau(key, tau_prior)
        def logistic(x, x0, k):
            return 1 / (1 + np.exp(-k * (x - x0)))
        
        log_numax = jnp.log10(nu_max[0])
        
        mu_he = - log_numax + 0.3
        sigma_he = 0.08 + 0.05 * logistic(log_numax, 2.8, 20.0)
        log_tau_he = (mu_he, sigma_he)
        
        x0 = 2.9
        mu_cz = - 0.225 * logistic(log_numax, x0, 20.0) - 0.8 * log_numax + 0.35
        sigma_cz = 0.08 + 0.1 * np.exp(- 0.5 * (log_numax - x0)**2/0.1**2)
        log_tau_cz = (mu_cz, sigma_cz)
        
        self.he_glitch: Prior = HeGlitchFunction(nu_max, log_tau=log_tau_he)
        self.cz_glitch: Prior = CZGlitchFunction(nu_max, log_tau=log_tau_cz)

        self._nu_max = distribution(nu_max)

        self._kernel_var = 0.1 * self.background.delta_nu.mean
        self._kernel_length = 5.0
        self.window_width = window_width

        self.units = {
            "nu_obs": u.microhertz,
            "nu": u.microhertz,
            "nu_bkg": u.microhertz,
            "dnu_he": u.microhertz,
            "dnu_cz": u.microhertz,
            "he_nu_max": u.microhertz,
            "cz_nu_max": u.microhertz,
            "he_amplitude": u.microhertz,
            "cz_amplitude": u.microhertz,
            "nu_max": u.microhertz,
        }

        self.symbols = {
            "nu_obs": r"$\nu_\mathrm{obs}$",
            "nu": r"$\nu$",
            "nu_bkg": r"$\nu_\mathrm{bkg}$",
            "dnu_he": r"$\delta\nu_\mathrm{He}$",
            "dnu_cz": r"$\delta\nu_\mathrm{BCZ}$",
            "he_nu_max": r"$A_\mathrm{He}(\nu_\max)$",
            "cz_nu_max": r"$A_\mathrm{BCZ}(\nu_\max)$",
            "he_amplitude": r"$\langle A_\mathrm{He} \rangle$",
            "cz_amplitude": r"$\langle A_\mathrm{BCZ} \rangle$",
            "nu_max": r"$\nu_\mathrm{max}$",
        }

        for prior in [self.background, self.he_glitch, self.cz_glitch]:
            # Inherit units from priors.
            self.units.update(prior.units)
            self.symbols.update(prior.symbols)

    def _init_tau(self, rng_key, tau_prior, num_samples=5000):
        predictive = Predictive(tau_prior, num_samples=num_samples)
        pred = predictive(rng_key)
        log_tau = pred["log_tau"] - 6  # Convert from seconds to mega seconds
        loc = log_tau.mean(axis=0)
        scale = log_tau.std(axis=0, ddof=1)
        return (
            distribution((loc[0], scale[0])),  # tau_he
            distribution((loc[1], scale[1])),  # tau_cz
        )

    def _glitch_amplitudes(self, nu):
        nu_max = numpyro.sample("nu_max", self._nu_max)
        numpyro.deterministic("he_nu_max", self.he_glitch.amplitude(nu_max))
        numpyro.deterministic("cz_nu_max", self.cz_glitch.amplitude(nu_max))

        if self.window_width == "full":
            low, high = nu.min(), nu.max()
        else:
            low = nu_max - self.window_width * self.background._delta_nu
            high = nu_max + self.window_width * self.background._delta_nu

        he_amp = numpyro.deterministic(
            "he_amplitude", self.he_glitch._average_amplitude(low, high)
        )
        cz_amp = numpyro.deterministic(
            "cz_amplitude", self.cz_glitch._average_amplitude(low, high)
        )
        return he_amp, cz_amp

    def _amplitude_prior(self, he_amp, cz_amp):
        # Prior that log(he_amp) == log(cz_amp) is a 2-sigma event
        # and the He amplitude is ~ 4 times the BCZ amplitude (log10(4) ~ 0.6)
        delta = 2.0 * (jnp.log10(he_amp) - jnp.log10(cz_amp) - 0.6) / 0.6
        logp = numpyro.distributions.Normal().log_prob(delta)
        numpyro.factor("amp", logp)

    def __call__(
        self,
        n: ArrayLike,
        nu: Optional[ArrayLike] = None,
        nu_err: Optional[ArrayLike] = None,
        n_pred: Optional[ArrayLike] = None,
    ):
        """Sample the model for given observables.

        Args:
            nu (:term:`array_like`, optional): Observed radial mode
                frequencies.
            nu_err (:term:`array_like`, optional): Gaussian observational
                uncertainties (sigma) for nu.
            pred (bool): If True, make predictions nu and nu_pred from n and
                num_pred.
        """
        # TODO it may be more general for all models to take an obs dict as
        # argument and every parameter to do obs.get('name', None)
        bkg_func = self.background()
        he_glitch_func = self.he_glitch()
        cz_glitch_func = self.cz_glitch()

        # The mean function for the GP
        def mean(n):
            nu_bkg = bkg_func(n)[0]  # shape of bkg_func is (1, num_orders)
            return nu_bkg + he_glitch_func(nu_bkg) + cz_glitch_func(nu_bkg)

        var = numpyro.param("kernel_var", self._kernel_var)
        length = numpyro.param("kernel_length", self._kernel_length)

        # kernel = SquaredExponential(var, length)
        kernel = var * kernels.ExpSquared(length)
        diag = 1e-6 if nu_err is None else nu_err**2  # No need for jitter
        gp = GaussianProcess(kernel, n, mean=mean, diag=diag)
        # gp = GP(kernel, mean=mean)

        with dimension("n", n.shape[-1], coords=n):
            nu = numpyro.sample("nu_obs", gp.numpyro_dist(), obs=nu)  # new nu!
            # nu = gp.sample("nu_obs", n, noise=nu_err, obs=nu)

            # if n_pred is not None:
                # gp.predict("nu", n)  # prediction without noise

            nu_bkg = numpyro.deterministic("nu_bkg", bkg_func(n))
            numpyro.deterministic("dnu_he", he_glitch_func(nu_bkg))
            numpyro.deterministic("dnu_cz", cz_glitch_func(nu_bkg))

        if n_pred is not None:
            with dimension("n", n.shape[-1], coords=n):
                numpyro.sample("nu", gp.condition(nu, n).gp.numpyro_dist())
            
            with dimension("n_pred", n_pred.shape[-1], coords=n_pred):
                # gp.predict("nu_pred", n_pred)
                numpyro.sample(
                    "nu_pred",
                    gp.condition(nu, n_pred).gp.numpyro_dist()
                )
                nu_bkg = numpyro.deterministic("nu_bkg_pred", bkg_func(n_pred))
                numpyro.deterministic("dnu_he_pred", he_glitch_func(nu_bkg))
                numpyro.deterministic("dnu_cz_pred", cz_glitch_func(nu_bkg))

        # Other deterministics
        self._amplitude_prior(*self._glitch_amplitudes(nu))


class GlitchModelComparison(GlitchModel):
    r"""Asteroseismic glitch model comparison. Compare the glitch model with
    a glitchless model. The frequencies are modelled using a GP with the same
    kernel function but different mean functions.

    The glitch model is the same as :class:`GlitchModel`. The glitcheless model
    is the same except that the mean function is,

    .. math::

        m_0(n) = f_\mathrm{bkg}(n),

    The two models are compared using the Bayes' factor,

    .. math::

        K = \frac{p(\nu_\mathrm{obs} \mid \mathcal{GP}_1)}
        {p(\nu_\mathrm{obs} \mid \mathcal{GP}_0)}

    where :math:`\mathcal{GP}_0` is the glitchless model and
    :math:`\mathcal{GP}_1` is the glitch model.

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
        seed (int): The seed used to generate samples from the prior on the
            glitch periods (acoustic depths) tau_he and tau_cz.
        window_width (float): The number of delta_nu either side of nu_max over
            which to average the helium glitch amplitude for the parameter
            'he_amplitude'.

    Attributes:
        n (numpy.ndarray): Radial order of model observations.
        n_pred (numpy.ndarray): Radial order of model predictions.
        background (Prior): Prior on the background function.
        he_glitch (Prior): Prior on the helium glitch function.
        cz_glitch (Prior): Prior on the base of convective zone glitch
            function.
        window_width (float): The number of delta_nu either side of nu_max over
            which to average the helium glitch amplitude for the parameter
            'he_amplitude'.
    """

    def __init__(
        self,
        nu_max: DistLike,
        delta_nu: DistLike,
        teff: Optional[DistLike] = None,
        epsilon: Optional[DistLike] = None,
        seed: int = 0,
        window_width: Union[str, float] = "full",
    ):
        super().__init__(nu_max, delta_nu, teff, epsilon, seed, window_width)

        self._prefix = "null"
        self._divider = "."

        units = {
            "log_k": u.LogUnit(u.dimensionless_unscaled),
        }

        symbols = {"log_k": r"$\log(k)$"}

        null_vars = ["nu", "nu_obs", "nu_bkg"]
        for var_name in null_vars:
            key = self._divider.join([self._prefix, var_name])
            units[key] = self.units[var_name]
            symbols[key] = self.symbols[var_name]

        self.units.update(units)
        self.symbols.update(symbols)

    def __call__(
        self,
        n: ArrayLike,
        nu: Optional[ArrayLike] = None,
        nu_err: Optional[ArrayLike] = None,
        n_pred: Optional[ArrayLike] = None,
    ):
        """Sample the model for given observables.

        Args:
            nu (:term:`array_like`, optional): Observed radial mode
                frequencies.
            nu_err (:term:`array_like`, optional): Gaussian observational
                uncertainties (sigma) for nu.
            pred (bool): If True, make predictions nu and nu_pred from n and
                num_pred.
        """
        # Same kernel function for both models
        var = numpyro.param("kernel_var", self._kernel_var)
        length = numpyro.param("kernel_length", self._kernel_length)
        kernel = var * kernels.ExpSquared(length)

        diag = 1e-6 if nu_err is None else nu_err**2  # No need for jitter

        args = ("models", 2)
        with dimension(*args):
            with numpyro.plate(*args):
                # Broadcast background function to both models
                bkg_func = self.background()

        # MODEL 0
        with numpyro.handlers.scope(prefix=self._prefix, divider="."):
            # Contain null model parameters in the null scope
            def mean0(n):
                return jnp.squeeze(bkg_func(n)[0])

            # gp0 = GP(kernel, mean=mean0)
            # dist0 = gp0.distribution(n, noise=nu_err)
            gp0 = GaussianProcess(kernel, n, mean=mean0, diag=diag)
            dist0 = gp0.numpyro_dist()

            with dimension("n", n.shape[-1], coords=n):
                nu0 = numpyro.sample("nu_obs", dist0, obs=nu)
                numpyro.deterministic("nu_bkg", bkg_func(n)[0])

            if n_pred is not None:
                with dimension("n", n.shape[-1], coords=n):
                    # gp0.predict("nu", n)
                    numpyro.sample(
                        "nu",
                        gp0.condition(nu0, n).gp.numpyro_dist(),
                    )
                with dimension("n_pred", n_pred.shape[-1], coords=n_pred):
                    # gp0.predict("nu_pred", n_pred)
                    numpyro.sample(
                        "nu_pred",
                        gp0.condition(nu0, n_pred).gp.numpyro_dist()
                    )
                    numpyro.deterministic("nu_bkg_pred", bkg_func(n_pred)[0])

        # MODEL 1
        he_glitch_func = self.he_glitch()
        cz_glitch_func = self.cz_glitch()

        def mean(n):
            nu_bkg = jnp.squeeze(bkg_func(n)[1])
            return nu_bkg + he_glitch_func(nu_bkg) + cz_glitch_func(nu_bkg)

        # gp = GP(kernel, mean=mean)
        # dist = gp.distribution(n, noise=nu_err)
        gp = GaussianProcess(kernel, n, mean=mean, diag=diag)
        dist = gp.numpyro_dist()

        with dimension("n", n.shape[-1], coords=n):

            nu = numpyro.sample("nu_obs", dist, obs=nu)  # redefines nu!
            nu_bkg = numpyro.deterministic("nu_bkg", bkg_func(n)[1])
            numpyro.deterministic("dnu_he", he_glitch_func(nu_bkg))
            numpyro.deterministic("dnu_cz", cz_glitch_func(nu_bkg))

        if n_pred is not None:
            with dimension("n", n.shape[-1], coords=n):
                # gp.predict("nu", n)
                numpyro.sample("nu", gp.condition(nu, n).gp.numpyro_dist())
            with dimension("n_pred", n_pred.shape[-1], coords=n_pred):
                # gp.predict("nu_pred", n_pred)
                numpyro.sample(
                    "nu_pred",
                    gp.condition(nu, n_pred).gp.numpyro_dist()
                )

                nu_bkg = numpyro.deterministic(
                    "nu_bkg_pred", bkg_func(n_pred)[1]
                )
                numpyro.deterministic("dnu_he_pred", he_glitch_func(nu_bkg))
                numpyro.deterministic("dnu_cz_pred", cz_glitch_func(nu_bkg))

        # Other deterministics and priors
        self._amplitude_prior(*self._glitch_amplitudes(nu))

        # LIKELIHOOD
        # Model comparison - if nu is not None, then nu0 == nu
        logL0 = dist0.log_prob(nu0)
        logL = dist.log_prob(nu)

        numpyro.factor("obs", (logL0 + logL).sum())

        # Log10 Bayes factor
        numpyro.deterministic("log_k", (logL - logL0).sum() / np.log(10.0))
