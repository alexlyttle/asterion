import math

import numpy as np

import jax
import jax.numpy as jnp

from jax import random

import numpyro
import numpyro.distributions as dist

from numpyro.infer.reparam import CircularReparam, LocScaleReparam
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.distributions import constraints

from collections import namedtuple

import warnings

from numpyro.primitives import CondIndepStackFrame, plate


class dimension(plate):

    def __init__(self, name, size, subsample_size=None, dim=None, coords=None):
        super().__init__(name, size, subsample_size=subsample_size, dim=dim)
        self.coords = coords

    def process_message(self, msg):
        """ Modify process message """
        super().process_message(msg)
        if msg["type"] == "deterministic":
            if "cond_indep_stack" not in msg.keys():
                msg["cond_indep_stack"] = []
            frame = CondIndepStackFrame(self.name, self.dim, self.subsample_size)
            msg["cond_indep_stack"].append(frame)


def estimate_n(num_orders, delta_nu, nu_max, epsilon):
    """ Estimates n from delta_nu, nu_max and epsilon, given num_orders about
    nu_max.

    E.g. to get n for num_orders about nu_max in a sample of model stars
    as a flag for 'observed' modes.
    """
    n_max = get_n_max(epsilon, delta_nu, nu_max)
    start = np.floor(n_max - np.floor(num_orders/2))
    stop = np.floor(n_max + np.ceil(num_orders/2)) - 1
    n = np.linspace(start, stop, num_orders, dtype=int, axis=-1)
    return n

def get_n_max(epsilon, delta_nu, nu_max):
    return nu_max / delta_nu - epsilon

def asy_background(n, epsilon, alpha, delta_nu, nu_max):
    n_max = get_n_max(epsilon, delta_nu, nu_max)
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
    prior_reparam = {}
    posterior_reparam = {}
    coords = {}
    dims = {}

    def __init__(self, n, nu, nu_err=None, l=0):
        self.nu = self._validate_nu(nu)
        self.obs_mask = ~np.isnan(self.nu)
        self.nu_err = self._validate_nu_err(nu_err)
        self.n = self._validate_n(n)
        self.l = l

    def _validate_nu(self, nu):
        return np.array(nu)
    
    def _validate_nu_err(self, nu_err):
        if nu_err is None:
            nu_err = np.zeros(self.nu)
        else:
            nu_err[~self.obs_mask] = 0.0
        return nu_err
    
    def _validate_n(self, n):
        # TODO: n must be 1-D
        return n

    def _prior(self):
        raise NotImplementedError

    def _predictions(self):
        raise NotImplementedError
    
    def _likelihood(self):
        raise NotImplementedError
    
    def _posterior(self):
        raise NotImplementedError

    @property
    def prior(self):
        return self._prior

    @property
    def predictions(self):
        return self._predictions

    @property
    def likelihood(self):
        return self._likelihood

    @property
    def posterior(self):
        return self._posterior

    def _get_trace(self, rng_key, model):
        model = handlers.trace(
            handlers.seed(
                model, rng_key
            )
        )
        trace = model.get_trace()
        return trace

    def get_prior_trace(self, rng_key):
        return self._get_trace(rng_key, self.prior)
    
    def get_posterior_trace(self, rng_key):
        return self._get_trace(rng_key, self.posterior)


class GlitchModel(_Model):
    """
    Parameters
    ----------
    delta_nu : array-like of float
        Two elements containing the respective prior mean and standard 
        deviation for the large frequency separation.
    nu_max : array-like of float
        Two elements containing the respective prior mean and standard
        deviation for the frequency at maximum power.
    epsilon : array-like of float, optional
        Two elements containing the respective prior mean and standard
        deviation for the asymptotic phase term. Default is [1.3, 0.2] which is
        suitable for main sequence solar-like oscillators.
    alpha : array-like of float, optional
        Two elements containing the respective prior mean and standard
        deviation for the asymptotic curvature term. Default is [0.0015, 0.002]
        which is suitable for most solar-like oscillators.
    nu : array-like of float
        Locations of the l=0 (radial) stellar pulsation modes. Null data may be
        passed as NaN. The shape of nu should be (num_orders,) or 
        (N, num_orders) for N stars in the model.
    nu_err : array-like of float, optional
        The observational uncertainty for each element of nu. Null data may be
        passed as NaN. Default is None, for which only a star-by-star model
        error is inferred and observational uncertainty is assumed to be zero.
    n : array-like of int, optional
        The observers radial order of each mode in nu. Default is None and n is
        inferred from num_orders.
    num_orders : int, optional
        The number of radial orders to model. The observers n is inferred from
        the priors for delta_nu, nu_max and epsilon. Default is None, and
        num_orders is inferred from the length of the final dimension of nu.
    num_pred : int, optional
        The number of predictions to make for nu as a function of n. Default is
        501.
    """
    posterior_reparam = {
        'phi_he': CircularReparam(),
        'phi_cz': CircularReparam(),
    }

    def __init__(self, name, delta_nu, nu_max, epsilon=None, alpha=None, *,
                 nu, nu_err=None, n=None):
        self.name = np.array(name)
        self.delta_nu = np.array(delta_nu)
        self.nu_max = np.array(nu_max)
        self.epsilon = np.array([1.3, 0.2]) if epsilon is None else np.array(epsilon)
        
        # self.num_stars = 1 if len(np.shape(name)) == 0 else np.shape(name)[0]
        # self.name = np.broadcast_to(name, (self.num_stars,))

        if alpha is None:
            self.log_alpha = np.array([-7.0, 1.0])  # natural logarithm
        else:
            self.log_alpha = np.array([
                np.log(alpha[0]**2 / np.sqrt(alpha[0]**2 + alpha[1]**2)),  # loc
                np.sqrt(np.log(1 + alpha[1]**2 / alpha[0]**2))             # scale
            ])
        # self.n = n
        # if n is None and num_orders is None:
        #     # Infer num_orders from final dimension of nu
        #     num_orders = np.shape(nu)[-1]
        #     warnings.warn("Neither argument 'n' nor 'num_orders' passed, " +
        #                   f"inferring num_orders = {num_orders} from final " +
        #                   "dimension of 'nu'.", UserWarning)

        # n = self._estimate_n(n, num_orders)
        # self.n_pred = np.linspace(self.n[..., 0], self.n[..., 0], num_pred)
        
        super().__init__(
            n, nu, nu_err=nu_err
        )



        # self.observed = Observed()
        self.num_stars = 1 if len(self.name.shape) == 0 else self.name.shape[0]
        self.num_orders = self.n.shape[0]
        self.name = np.broadcast_to(self.name, (self.num_stars,))
        self.n = np.broadcast_to(self.n, (self.num_orders,))

        shape = (self.num_stars, self.num_orders)
        self.nu = np.broadcast_to(self.nu, shape)
        self.nu_err = np.broadcast_to(self.nu_err, shape)
        self.obs_mask = np.broadcast_to(self.obs_mask, shape)
    
        self.dimensions = {
            'name': dimension('name', self.num_stars, coords=self.name),
            'n': dimension('n', self.num_orders, coords=self.n)
        }

    # def _estimate_n(self, n, num_orders):
    #     if num_orders is None and n is None:
    #         raise ValueError("One of either 'n' or 'num_orders' must be given.")
    
    #     if n is None:
    #         n_max = get_n_max(self.epsilon[..., 0], self.delta_nu[..., 0], self.nu_max[..., 0])
    #         start = np.floor(n_max - np.floor(num_orders/2))
    #         stop = np.floor(n_max + np.ceil(num_orders/2)) - 1
    #         n = np.linspace(start, stop, num_orders, dtype=int, axis=1)
    #     else:
    #         n = np.array(n, dtype=int)
    #         # TODO: check that the shape of n makes sense
    #         # TODO: warn if num_orders is also passed

    #     return n

    def _prior(self):
        m_tau = 1.05
        # s_tau = 3.0
        log_tau = - m_tau * np.log(self.nu_max[0])  # Approx form of log(tau_he)

        with self.dimensions['name']:
            epsilon = numpyro.sample('epsilon', dist.Normal(*self.epsilon))
            alpha = numpyro.sample('alpha', dist.LogNormal(*self.log_alpha))

            delta_nu = numpyro.sample('delta_nu', dist.Normal(*self.delta_nu))
            nu_max = numpyro.sample('nu_max', dist.Normal(*self.nu_max))
        
            b0 = numpyro.sample('b0', dist.LogNormal(np.log(50/self.nu_max[0]), 1.0))
            b1 = numpyro.sample('b1', dist.LogNormal(np.log(5/self.nu_max[0]**2), 1.0))

            tau_he = numpyro.sample('tau_he', dist.LogNormal(log_tau, 0.8))
            phi_he = numpyro.sample('phi_he', dist.VonMises(0.0, 0.1))

            c0 = numpyro.sample('c0', dist.LogNormal(np.log(0.5*self.nu_max[0]**2), 1.0))

            # Ensure that tau_cz > tau_he
            delta_tau = numpyro.sample('delta_tau', dist.LogNormal(log_tau, 0.8))
            tau_cz = numpyro.deterministic('tau_cz', tau_he + delta_tau)

            phi_cz = numpyro.sample('phi_cz', dist.VonMises(0.0, 0.1))
            
            err = numpyro.sample('err', dist.HalfNormal(0.1))

            with self.dimensions['n']:
                # n = self.n_pred if pred else self.n
                nu_asy = numpyro.deterministic('nu_asy', asy_background(self.n[np.newaxis, ...], epsilon, alpha, delta_nu, nu_max))

                dnu_he = he_glitch(nu_asy, b0, b1, tau_he, phi_he)
                dnu_cz = cz_glitch(nu_asy, c0, tau_cz, phi_cz)

                nu = numpyro.deterministic('nu', nu_asy + dnu_he + dnu_cz)

            # average_he = numpyro.deterministic('<he>', average_he_amplitude(...))
            # he_nu_max = numpyro.deterministic('he_nu_max', he_amplitude(nu_max, b0, b1))
        return nu, err

    def _likelihood(self, nu, err):
        # if self.nu_err is not None:
        err = jnp.sqrt(err**2 + self.nu_err**2)

        with self.dimensions['name']:
            with self.dimensions['n']:
                nu_obs = numpyro.sample('nu_obs', dist.Normal(nu, err),
                                        obs=self.nu, obs_mask=self.obs_mask)

    # def _predictive(self, n=None):
    #     self._likelihood(*self._prior(n=n))

    # @property
    # def predictive(self):
    #     # def _predictive():
    #     #     # nu, nu_err = self._prior()  # prior without reparam
    #     #     self._likelihood(*self._prior(n=n))

    #     return self._predictive

    # def _predict(self):
    #     self._prior(pred=True)
    
    def _posterior(self):
        self._likelihood(*self._prior())
    
    # @property
    # def prior(self):
    #     return self._prior

    # @property
    # def predictions(self):
    #     return self._predictions

    # @property
    # def likelihood(self):
    #     return self._likelihood

    # @property
    # def posterior(self):
    #     return self._posterior
