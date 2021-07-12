import math

import jax
import jax.numpy as jnp

from jax import random

import numpyro
import numpyro.distributions as dist

from numpyro.infer.reparam import CircularReparam
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

from collections import namedtuple

def estimate_n_max(epsilon, delta_nu, nu_max):
    return nu_max / delta_nu - epsilon

def asy_background(n, epsilon, alpha, delta_nu, nu_max):
    n_max = estimate_n_max(epsilon, delta_nu, nu_max)
    return delta_nu * (n + epsilon + 0.5 * alpha * (n - n_max)**2)

def glitch(nu, tau, phi):
    return jnp.sin(4 * math.pi * tau * nu + phi)
    
def he_amplitude(nu, b0, b1):
    return b0 * nu * jnp.exp(- b1 * nu**2)

def he_glitch(nu, b0, b1, tau_he, phi_he):
    return he_amplitude(nu, b0, b1) * glitch(nu, tau_he, phi_he)

def cz_amplitude(nu, c0):
    return c0 / nu**2

def cz_glitch(nu, c0, tau_cz, phi_cz):
    return cz_amplitude(nu, c0) * glitch(nu, tau_cz, phi_cz)

def average_he_amplitude(b0, b1, low, high):
    ''' Derived average amplitude over the fitting range.'''
    return b0 * (jnp.exp(-b1*low**2) - jnp.exp(-b1*high**2)) / (2*b1*(high - low))


class _Model:
    def __init__(self):
        self.mcmc = None

    @property
    def model(self):
        raise NotImplementedError()

    def sample(self, num_warmup=1000, num_samples=1000, num_chains=5, seed=0, 
               model_args=(), model_kwargs={}, kernel_kwargs={}, mcmc_kwargs={}):
        """
        Parameters
        ----------
        num_warmup : int
        num_samples : int
        num_chains : int
        seed : int
        model_args : tuple
        model_kwargs : dict
        kernel_kwargs : dict
        mcmc_kwargs : dict

        Returns
        -------
        samples : dict

        """
        
        target_accept_prob = kernel_kwargs.pop('target_accept_prob', 0.99)
        init_strategy = kernel_kwargs.pop('init_strategy', init.init_to_median)
        step_size = kernel_kwargs.pop('step_size', 0.1)

        kernel = NUTS(self.model, target_accept_prob=target_accept_prob, init_strategy=init_strategy, 
                      step_size=step_size, **kernel_kwargs)

        self.mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, **mcmc_kwargs)

        rng_key = random.PRNGKey(seed)
        self.mcmc.run(rng_key, *model_args, **model_kwargs)

        # TODO: warn diverging (and rhat if num_chains > 1)

        samples = self.mcmc.get_samples()
        return samples


class Prior(_Model):
    """
    Parameters
    ----------
    num_orders : int
        Number of radial orders per star.

    """

    def __init__(self, delta_nu, nu_max, epsilon, log_alpha, n=None, num_orders=None):
        self.delta_nu = delta_nu
        self.nu_max = nu_max
        self.epsilon = epsilon
        self.log_alpha = log_alpha
        self.n = self._validate_n(n, num_orders)

        self.phi_he = [0.0, 0.1]
        self.phi_cz = [0.0, 0.1]
    
    def _validate_n(self, n, num_orders):
        if num_orders is None and n is None:
            raise ValueError('One of either n or num_orders must be given.')
    
        if n is None:
            n_max = estimate_n_max(self.epsilon[..., 0], self.delta_nu[..., 0], self.nu_max[..., 0])
            start = jnp.floor(n_max - jnp.floor(num_orders/2))
            stop = jnp.floor(n_max + jnp.ceil(num_orders/2)) - 1
            n = jnp.linspace(start, stop, num_orders, dtype=int)
        else:
            n = jnp.array(n, dtype=int)
            # TODO: check that the shape of n makes sense
            # TODO: warn if num_orders is also passed

        return n

    @property
    def model(self):
        m_tau = 1.05
        s_tau = 3.0
        log_tau = - m_tau * jnp.log(self.nu_max[0])  # Approx form of log(tau_he)
        
        @handlers.reparam(config={'phi_he': CircularReparam(), 'phi_cz': CircularReparam()})
        def _model():
            epsilon = numpyro.sample('epsilon', dist.Normal(*self.epsilon))
            alpha = numpyro.sample('alpha', dist.LogNormal(*self.log_alpha))

            delta_nu = numpyro.sample('delta_nu', dist.Normal(*self.delta_nu))
            nu_max = numpyro.sample('nu_max', dist.Normal(*self.nu_max))
            
            b0 = numpyro.sample('b0', dist.LogNormal(jnp.log(50/self.nu_max[0]), 1.0))
            b1 = numpyro.sample('b1', dist.LogNormal(jnp.log(5/self.nu_max[0]**2), 1.0))

            tau_he = numpyro.sample('tau_he', dist.LogNormal(log_tau, 0.8))
            phi_he = numpyro.sample('phi_he', dist.VonMises(*self.phi_he))

            c0 = numpyro.sample('c0', dist.LogNormal(jnp.log(0.5*self.nu_max[0]**2), 1.0))
            tau_cz = numpyro.sample('tau_cz', dist.LogNormal(log_tau + jnp.log(s_tau), 0.8))

            # Ensure that tau_cz > tau_he (more stable than tau_cz = tau_he + delta_tau)
            delta_tau_logp = dist.LogNormal(log_tau, 0.8).log_prob(tau_cz - tau_he)
            delta_tau = numpyro.factor('delta_tau', delta_tau_logp)

            phi_cz = numpyro.sample('phi_cz', dist.VonMises(*self.phi_cz))
            
            nu_asy = numpyro.deterministic('nu_asy', asy_background(self.n, epsilon, alpha, delta_nu, nu_max))
            dnu_he = he_glitch(nu_asy, b0, b1, tau_he, phi_he)
            dnu_cz = cz_glitch(nu_asy, c0, tau_cz, phi_cz)

            # average_he = numpyro.deterministic('<he>', average_he_amplitude())
            he_nu_max = numpyro.deterministic('he_nu_max', he_amplitude(nu_max, b0, b1))

            nu = numpyro.deterministic('nu', nu_asy + dnu_he + dnu_cz)
            nu_err = numpyro.sample('nu_err', dist.HalfNormal(0.1))

            return nu, nu_err

        return _model

    def sample(self, num_warmup=1000, num_samples=1000, num_chains=5, seed=0, 
               kernel_kwargs={}, mcmc_kwargs={}):
        """
        Parameters
        ----------
        num_warmup : int
        num_samples : int
        num_chains : int
        seed : int
        kernel_kwargs : dict
        mcmc_kwargs : dict

        Returns
        -------
        samples : dict

        """
        return super().sample(num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, 
                              seed=seed, kernel_kwargs=kernel_kwargs, mcmc_kwargs=mcmc_kwargs)


class Observed(_Model):
    def __init__(self, nu, nu_err=None):
        self.nu = nu
        self.nu_err = nu_err

    @property
    def model(self):
        def _model(nu, nu_err):
            if self.nu_err is not None:
                nu_err = jnp.sqrt(nu_err**2 + self.nu_err**2)

            nu_obs = numpyro.sample('nu_obs', dist.Normal(nu, nu_err), obs=self.nu)
        return _model

    def sample(self, nu, nu_err=None, num_warmup=1000, num_samples=1000, num_chains=5, 
               seed=0, kernel_kwargs={}, mcmc_kwargs={}):
        """
        Parameters
        ----------
        nu : array-like
            Model frequencies.
        nu_err : float or array-like
            Model error.
        num_warmup : int
        num_samples : int
        num_chains : int
        seed : int
        kernel_kwargs : dict
        mcmc_kwargs : dict

        Returns
        -------
        samples : dict

        """
        model_args = (nu,)
        model_kwargs = {'nu_err': nu_err}
        return super().sample(num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                              seed=seed, model_args=model_args, model_kwargs=model_kwargs,
                              kernel_kwargs=kernel_kwargs, mcmc_kwargs=mcmc_kwargs)


class Posterior(_Model):
    def __init__(self, prior, observed):
        self.prior = prior
        self.observed = observed

    @property
    def model(self):
        def _model():
            nu, nu_err = self.prior.model()
            self.observed.model(nu, nu_err)

        return _model

    def sample(self, num_warmup=1000, num_samples=1000, num_chains=5, seed=0, 
               kernel_kwargs={}, mcmc_kwargs={}):
        """
        Parameters
        ----------
        num_warmup : int
        num_samples : int
        num_chains : int
        seed : int
        kernel_kwargs : dict
        mcmc_kwargs : dict

        Returns
        -------
        samples : dict

        """
        return super().sample(num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, 
                              seed=seed, kernel_kwargs=kernel_kwargs, mcmc_kwargs=mcmc_kwargs)
