"""Probabilistic models for asteroseismic oscillation mode frequencies.
"""
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
from numpyro.primitives import Messenger, plate, apply_stack

from typing import Union, Optional, Callable, Dict, ClassVar, Any
from .annotations import Array1D, Array2D, Array3D

from .gp import GP, SquaredExponential

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import warnings

from .priors import Prior


__all__ = [
    "estimate_n",
    "dimension",
    "Model",
    "GlitchModel",
]


def estimate_n(
    num_orders: int,
    delta_nu: Union[float, Array1D[float]], 
    nu_max: Union[float, Array1D[float]], 
    epsilon: Union[float, Array1D[float]]=0.0,
) -> np.ndarray:
    """Estimates n from delta_nu, nu_max and epsilon, given num_orders about
    nu_max.

    Args:
        num_orders: Number of radial orders to estimate n for.
        delta_nu: Large frequecy spacing (microHz).
        nu_max: Frequency at maximum power (microHz).
        epsilon: Phase or offset of the asymptotic approximation.

    Returns:
        An array of radial order, n.
    """
    n_max = get_n_max(epsilon, delta_nu, nu_max)
    start = np.floor(n_max - np.floor(num_orders/2))
    stop = np.floor(n_max + np.ceil(num_orders/2)) - 1
    n = np.linspace(start, stop, num_orders, dtype=int, axis=-1)
    return n


def get_n_max(epsilon, delta_nu, nu_max):
    return nu_max / delta_nu - epsilon


def asy_background(n, epsilon, alpha, delta_nu, nu_max, beta=0.0, gamma=0.0):
    n_max = get_n_max(epsilon, delta_nu, nu_max)
    return delta_nu * (n + epsilon + 0.5 * alpha * (n - n_max)**2 + beta*n**3 - gamma*(n - n_max)**4) 


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
    """ Derived average amplitude over the fitting range."""
    return b0 * (jnp.exp(-b1*low**2) - jnp.exp(-b1*high**2)) / \
        (2*b1*(high - low))


class dimension(Messenger):
    """
    """

    def __init__(self, name, size, coords=None, dim=None):
        self.name = name
        self.size = size
        self.dim = -1 if dim is None else dim  # Defaults to rightmost dim
        assert self.dim < 0
        if coords is None:
            coords = np.arange(self.size)
        self.coords = np.array(coords)
        msg = self._get_message()
        apply_stack(msg)
        super().__init__()
    
    def _get_message(self):
        msg = {
            "name": self.name,
            "type": "dimension",
            "dim": self.dim,
            "value": self.coords,
        }
        return msg
        
    def __enter__(self):
        super().__enter__()
        return self._get_message()

    def process_message(self, msg):
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

# class dimension(plate):
#     """Context manager for variables along a given dimension with coordinates.

#     Based on `numpyro.plate` but includes deterministics defined within the
#     context.
    
#     Args:
#         name: Name of the dimension.
#         size: Size of the dimension.
#         subsample_size: Optional argument denoting the size of the mini-batch.
#             This can be used to apply a scaling factor by inference algorithms.
#             e.g. when computing ELBO using a mini-batch.
#         dim: Optional argument to specify which dimension in the tensor
#             is used as the plate dim. If `None` (default), the rightmost
#             available dim is allocated.
#         coords: Optional coordinates for each point in the dimension. If `None`
#             (default), the coords are `np.arange(size)`.
    
#     Attributes:
#         name (str): Name of the dimension
#         size (int): Size of the dimension
#         subsample_size (int or None): Size of the mini-batch.
#         dim (int or None): The dimension in the tensor which is used as the
#             plate dim.
#     """
#     def __init__(self, name: str, size: int,
#                  subsample_size: Optional[int]=None,
#                  dim: Optional[int]=None, coords: Optional[Array1D[Any]]=None):
#         super().__init__(name, size, subsample_size=subsample_size, dim=dim)
#         self.coords: np.ndarray  #: Coordinates for each point in dimension.
#         if coords is None:
#             self.coords = np.arange(self.size)
#         else:
#             self.coords = np.array(coords)

#     def process_message(self, msg: Dict[str, Any]):
#         """Modified process message to also include deterministics. 
        
#         Args:
#             msg: Process message
#         """
#         super().process_message(msg)
#         if msg["type"] == "deterministic":
#             if "cond_indep_stack" not in msg.keys():
#                 msg["cond_indep_stack"] = []
#             frame = CondIndepStackFrame(self.name, self.dim, self.subsample_size)
#             msg["cond_indep_stack"].append(frame)


# class Model:
#     """Base Model class for modelling asteroseismic oscillation modes. 
    
#     All models must have a :attr:`name` (usually the name of the star),
#     array of radial order :attr:`n` and array of angular degree :attr:`l`
#     (optional). The observed mode frequencies :attr:`nu` must then be
#     broadcastable to an array with shape :attr:`num_stars` x :attr:`num_orders`
#     (x :attr:`num_degrees`). Optionally, the observed uncertainties may be
#     passed with the same shape via :attr:`nu_err`. Any unobserved or null data
#     should be passed as :obj:`numpy.nan`, and an :attr:`obs_mask` will be made
#     which can be used in the likelihood.
    
#     Args:
#         name: Name of the star(s) in the model.
#         n: Radial orders of the modes to be modelled.
#         l: Angular degree(s) of the modes to be modelled. Defaults to
#             :obj:`None`.
#         nu: Observed central mode frequencies. Must be broadcastable to a 2D or
#             3D array:

#             +--------------------------------+--------------------------------+
#             | Shape                          | Description                    |
#             +================================+================================+
#             | (:attr:`num_stars`,            | If :attr:`l` is :class:`float` |
#             | :attr:`num_orders`)            | or :obj:`None`.                |
#             +--------------------------------+--------------------------------+
#             | (:attr:`num_stars`,            | If :attr:`l` is                |
#             | :attr:`num_orders`,            | :class:`Array1D`.              |
#             | :attr:`num_degrees`)           |                                |
#             +--------------------------------+--------------------------------+
            
#             Unobserved or null data may be passed as
#             :obj:`numpy.nan`.
#         nu_err: The observational uncertainty for each element of
#             :attr:`nu`. Defaults to :obj:`None` which is equivalent to an
#             uncertainty of 0.0.
    
#     Example:
#         .. code-block:: python

#             import numpyro
#             import numpyro.distributions as dist
#             import numpy as np
#             import jax.numpy as jnp
#             from asterion import Model

#             class MyModel(Model):
#                 def _prior(self):
                    
#                     # Use dimension context manager to track dimensionality \
# of output
#                     with self.dimensions['name'] as stars:
#                         delta_nu = numpyro.sample('delta_nu', \
# dist.Normal(20., 1.))
#                         epsilon = numpyro.sample('epsilon', \
# dist.Normal(1.0, 0.1))
#                         err = numpyro.sample('err', dist.HalfNormal(0.1))
                        
#                         with self.dimensions['n'] as orders:
#                             # Make sure that n has the correct shape
#                             n = np.broadcast_to(self.n, \
# (stars.size, orders.size))
#                             nu = numpyro.deterministic('nu', \
# delta_nu * (n + epsilon))
                    
#                     # Return model nu and model error
#                     return nu, err
                
#                 def _likelihood(self, nu, err):
#                     # Combine model and obs error
#                     # Operations applied to RVs must be jax-compatible
#                     nu_err = jnp.sqrt(err**2 + self.nu_err**2)

#                     with self.dimensions['name']:
#                         with self.dimensions['n']:
#                             numpyro.sample(
#                                 'nu_obs',
#                                 dist.Normal(nu, nu_err),
#                                 obs=self.nu,
#                                 obs_mask=self.obs_mask
#                             )
                
#                 def _posterior(self):
#                     self._likelihood(*self._prior())

#                 def _predictions(self):
#                     # Here, you can return the prior, or a version of the prior
#                     # with, e.g. continuous n. Here, we just return the prior.
#                     return self._prior()
#     """
#     reparam: ClassVar[Dict[str, Reparam]] = {}  #: [description]
#     circ_var_names: ClassVar[List[str]] = []  #: [description]

#     def __init__(
#         self,
#         name: Union[str, Array1D[str]],
#         n: Array1D[int],
#         l: Optional[Union[int, Array1D[int]]]=None, 
#         *,
#         nu: Union[Array1D[float], Array2D[float], Array3D[float]],
#         nu_err: Optional[
#             Union[Array1D[float], Array2D[float], Array3D[float]]
#         ]=None,
#     ):        
#         self.name: np.ndarray = self._validate_name(name)  #: [description]
#         self.n: np.ndarray = self._validate_n(n)  #: [description]
#         self.l: np.ndarray = self._validate_l(l)  #: [description]

#         self.num_stars: int = self.name.shape[0]  #: [description]
#         self.num_orders: int = self.n.shape[0]  #: [description]
#         self.num_degrees: Optional[int] = None  #: [description]

#         self.dimensions: Dict[str, dimension] = {
#             "name": dimension("name", self.num_stars, coords=self.name),
#             "n": dimension("n", self.num_orders, coords=self.n)
#         }  #: [description]
#         self._shape = (self.num_stars, self.num_orders)

#         if self.l is not None:
#             self.num_degrees = self.l.shape[0]
#             self._shape += (self.num_degrees,)
#             self.dimension["l"] = dimension("l", self.num_degrees, coords=self.l)
        
#         self.nu: np.ndarray = self._validate_nu(nu)  #: [description]
#         self.obs_mask: np.ndarray = ~np.isnan(self.nu)  #: [description]
#         self.nu_err: np.ndarray = self._validate_nu_err(nu_err)  #: [description]

#     def _validate_name(self, name: Union[str, Array1D[str]]) -> np.ndarray:
#         name = np.squeeze(name)
#         if name.ndim == 0:
#             name = np.broadcast_to(name, (1,))
#         elif name.ndim > 1:
#             raise ValueError("Variable 'name' is greater than 1-D.")
#         assert name.astype(str)
#         assert name.ndim == 1
#         return name

#     def _validate_n(self, n: Array1D[int]) -> np.ndarray:
#         n = np.squeeze(n)
#         if n.ndim == 0 or n.ndim > 1:
#             raise ValueError("Variable 'n' must be 1-D.")
#         assert n.ndim == 1
#         return np.array(n)
    
#     def _validate_l(
#         self, 
#         l: Optional[Union[int, Array1D[int]]]
#     ) -> Optional[np.ndarray]:
#         if l is None:
#             return
#         l = np.squeeze(l)
#         if l.ndim == 0:
#             l = np.broadcast_to(l, (1,))
#         elif l.ndim > 1:
#             raise ValueError("Variable 'l' is greater than 1-D.")
#         assert l.ndim == 1
#         return np.array(l)
    
#     def _broadcast(self, x) -> np.ndarray:
#         return np.broadcast_to(x, self._shape)

#     def _validate_nu(self, nu: Union[Array1D[float], Array2D[float], Array3D[float]]) -> np.ndarray:
#         return self._broadcast(nu)
    
#     def _validate_nu_err(self, nu_err: Optional[Union[Array1D[float], Array2D[float], Array3D[float]]]) -> np.ndarray:
#         if nu_err is None:
#             nu_err = np.zeros(self._shape)
#         else:
#             nu_err = self._broadcast(nu_err).copy()
#         nu_err[~self.obs_mask] = 0.0
#         return nu_err

#     def _prior(self):
#         raise NotImplementedError

#     def _predictions(self):
#         raise NotImplementedError
    
#     def _likelihood(self):
#         raise NotImplementedError
    
#     def _posterior(self):
#         raise NotImplementedError

#     @property
#     def prior(self) -> Callable:
#         """Function which samples from the model prior and returns arguments 
#         for :attr:`likelihood`.
#         """        
#         return self._prior

#     @property
#     def predictions(self) -> Callable:
#         """Function which makes predictions which may have different dimensions
#         to observed values.
#         """        
#         return self._predictions

#     @property
#     def likelihood(self) -> Callable:
#         """Function which takes the output of :attr:`prior` and samples
#         from the model likelihood.
#         """ 
#         return self._likelihood

#     @property
#     def posterior(self) -> Callable:
#         """Function which combines the :attr:`prior` and
#         :attr:`likelihood` to sample from the model posterior.
#         """        
#         return self._posterior

#     def _get_trace(self, rng_key, model):
#         model = handlers.trace(
#             handlers.seed(
#                 model, rng_key
#             )
#         )
#         trace = model.get_trace()
#         return trace

#     def get_prior_trace(self, rng_key: Union[int, jnp.ndarray]) -> dict:
#         """Sample from :attr:`prior` given a random seed.

#         Args:
#             rng_key: Random seed or key for generating the sample.

#         Returns:
#             Trace from the :attr:`prior`.
#         """        
#         return self._get_trace(rng_key, self.prior)
    
#     def get_posterior_trace(self, rng_key: Union[int, jnp.ndarray]) -> dict:
#         """Sample from :attr:`posterior` given a random seed.

#         Args:
#             rng_key: Random seed or key for generating the sample.

#         Returns:
#             Trace from the :attr:`posterior`.
#         """        
#         return self._get_trace(rng_key, self.posterior)

#     def get_predictions_trace(self, rng_key: Union[int, jnp.ndarray]) -> dict:
#         """Sample from :attr:`predictions` given a random seed.

#         Args:
#             rng_key: Random seed or key for generating the sample.

#         Returns:
#             Trace from the :attr:`predictions`.
#         """        
#         return self._get_trace(rng_key, self.predictions)


# class GlitchModel(Model):
#     """Model the glitch in the asteroseismic radial mode frequencies.

#     Args:
#         delta_nu: Two elements containing the respective prior mean and standard 
#             deviation for the large frequency separation.
#         nu_max: Two elements containing the respective prior mean and standard
#             deviation for the frequency at maximum power.
#         epsilon: Two elements containing the respective prior mean and standard
#             deviation for the asymptotic phase term. Default is [1.3, 0.2] which is
#             suitable for main sequence solar-like oscillators.
#         alpha: Two elements containing the respective prior mean and standard
#             deviation for the asymptotic curvature term. Default is [0.0015, 0.002]
#             which is suitable for most solar-like oscillators.
#         nu: Locations of the l=0 (radial) stellar pulsation modes. Null data may be
#             passed as NaN. The shape of nu should be (num_orders,) or 
#             (N, num_orders) for N stars in the model.
#         nu_err: The observational uncertainty for each element of nu. Null data may be
#             passed as NaN. Default is None, for which only a star-by-star model
#             error is inferred and observational uncertainty is assumed to be zero.
#         n: The observers radial order of each mode in nu. Default is None and n is
#             inferred from num_orders.
#         num_orders: The number of radial orders to model. The observers n is inferred from
#             the priors for delta_nu, nu_max and epsilon. Default is None, and
#             num_orders is inferred from the length of the final dimension of nu.
#         num_pred: The number of predictions to make in the range of n. Default
#             is 200.
#     """
#     reparam = {
#         "phi_he": CircularReparam(),
#         "phi_cz": CircularReparam(),
#     }
#     circ_var_names = [
#         "phi_he",
#         "phi_cz"
#     ]

#     def __init__(
#         self,
#         name: Union[str, Array1D[str]],
#         delta_nu: Array1D[float],
#         nu_max: Array1D[float],
#         epsilon: Array1D[float]=None,
#         alpha: Array1D[float]=None,
#         *,
#         nu: Union[Array1D[float], Array2D[float]],
#         nu_err: Optional[Union[Array1D[float], Array2D[float]]]=None, 
#         n: Optional[Array1D[int]]=None,
#         num_orders: Optional[int]=None,
#         num_pred: int=200,
#         regularization: float=1e-6,
#         beta: Array1D[float]=[-14.0, 1.0],
#         gamma: Array1D[float]=[-12.0, 1.0],
#     ):

#         self._delta_nu = np.array(delta_nu)
#         self._nu_max = np.array(nu_max)
#         self._epsilon = np.array([1.3, 0.2]) if epsilon is None else np.array(epsilon)

#         if alpha is None:
#             self._log_alpha = np.array([-7.0, 1.0])  # natural logarithm
#         else:
#             self._log_alpha = np.array([
#                 np.log(alpha[0]**2 / np.sqrt(alpha[0]**2 + alpha[1]**2)),  # loc
#                 np.sqrt(np.log(1 + alpha[1]**2 / alpha[0]**2))             # scale
#             ])
        
#         super().__init__(
#             name, n, nu=nu, nu_err=nu_err,
#         )

#         self.num_pred = num_pred
#         self.n_pred = np.linspace(self.n[0], self.n[-1], num_pred)
#         self.dimensions["n_pred"] = dimension(
#             "n_pred", self.num_pred, coords=self.n_pred
#         )

#         self.regularization = regularization
#         self._beta = beta
#         self._gamma = gamma
#         # self.num_stars = 1 if len(self.name.shape) == 0 else self.name.shape[0]
#         # self.num_orders = self.n.shape[0]
#         # self.name = np.broadcast_to(self.name, (self.num_stars,))
#         # self.n = np.broadcast_to(self.n, (self.num_orders,))

#         # shape = (self.num_stars, self.num_orders)
#         # self.nu = np.broadcast_to(self.nu, shape)
#         # self.nu_err = np.broadcast_to(self.nu_err, shape)
#         # self.obs_mask = np.broadcast_to(self.obs_mask, shape)
    
#         # self.dimensions = {
#         #     'name': dimension('name', self.num_stars, coords=self.name),
#         #     'n': dimension('n', self.num_orders, coords=self.n)
#         # }

#     def _prior(self, pred: bool=False):
#         # m_tau = 1.05
#         m_tau = 0.91
#         # s_tau = 3.0
#         log_tau = - m_tau * np.log(self._nu_max[0])  # Approx form of log(tau_he)

#         with self.dimensions["name"]:
#             epsilon = numpyro.sample("epsilon", dist.Normal(*self._epsilon))
#             alpha = numpyro.sample("alpha", dist.LogNormal(*self._log_alpha))

#             delta_nu = numpyro.sample("delta_nu", dist.Normal(*self._delta_nu))
#             nu_max = numpyro.sample("nu_max", dist.Normal(*self._nu_max))

#             # TODO: define these
#             beta = numpyro.sample("beta", dist.LogNormal(*self._beta))
#             gamma = numpyro.sample("gamma", dist.LogNormal(*self._gamma))

#             b0 = numpyro.sample("b0", dist.LogNormal(np.log(1/self._nu_max[0]), 1.0))

#             # b0 = numpyro.sample("b0", dist.HalfNormal(100/self._nu_max[0]))
#             b1 = numpyro.sample("b1", dist.LogNormal(np.log(1/self._nu_max[0]**2), 1.0))

#             tau_he = numpyro.sample("tau_he", dist.LogNormal(log_tau, 0.8))
#             phi_he = numpyro.sample("phi_he", dist.VonMises(0.0, 0.1))

#             c0 = numpyro.sample("c0", dist.LogNormal(np.log(0.1*self._nu_max[0]**2), 1.0))

#             # Ensure that tau_cz > tau_he
#             delta_tau = numpyro.sample("delta_tau", dist.LogNormal(log_tau, 0.8))
#             tau_cz = numpyro.deterministic("tau_cz", tau_he + delta_tau)

#             phi_cz = numpyro.sample("phi_cz", dist.VonMises(0.0, 0.1))
            
#             err = numpyro.sample("err", dist.HalfNormal(0.1))

#             dim_name = "n_pred" if pred else "n"
#             with self.dimensions[dim_name]:
#                 # Broadcast to the shape of output freq
#                 if pred:
#                     n = np.broadcast_to(self.n_pred, (self.num_stars, self.num_pred))
#                 else:
#                     n = np.broadcast_to(self.n, self._shape)

#                 nu_asy = numpyro.deterministic("nu_asy", asy_background(n, epsilon, alpha, delta_nu, nu_max, beta, gamma))

#                 # So factor is not done for predictions
#                 if numpyro.get_mask() is not False:
#                     # L2 regularisation on d3 nu_asy / d n3 same as Gaussian prior
#                     # with sd = 1/lambda where lambda = self.regularization
#                     log_prob = dist.Normal(0.0, 1/self.regularization).log_prob(6*beta - 24*gamma*n)
#                     numpyro.factor('reg', jnp.sum(log_prob))  # TODO: uncomment

#                 dnu_he = numpyro.deterministic("dnu_he", he_glitch(nu_asy, b0, b1, tau_he, phi_he))
#                 dnu_cz = numpyro.deterministic("dnu_cz", cz_glitch(nu_asy, c0, tau_cz, phi_cz))
#                 # dnu_cz = 0.0

#                 nu = numpyro.deterministic("nu", nu_asy + dnu_he + dnu_cz)

#             # average_he = numpyro.deterministic('<he>', average_he_amplitude(...))
#             # he_nu_max = numpyro.deterministic('he_nu_max', he_amplitude(nu_max, b0, b1))
#         return nu, err

#     def _likelihood(self, nu, err):
#         # if self.nu_err is not None:
#         err = jnp.sqrt(err**2 + self.nu_err**2)

#         with self.dimensions["name"]:
#             with self.dimensions["n"]:
#                 nu_obs = numpyro.sample("nu_obs", dist.Normal(nu, err),
#                                         obs=self.nu, obs_mask=self.obs_mask)

#     # def _predictive(self, n=None):
#     #     self._likelihood(*self._prior(n=n))

#     # @property
#     # def predictive(self):
#     #     # def _predictive():
#     #     #     # nu, nu_err = self._prior()  # prior without reparam
#     #     #     self._likelihood(*self._prior(n=n))

#     #     return self._predictive

#     # def _predict(self):
#     #     self._prior(pred=True)
    
#     def _posterior(self):
#         self._likelihood(*self._prior())

#     def _predictions(self):
#         return self._prior(pred=True)


# class AsyModel(Model):
#     """Model the glitch in the asteroseismic radial mode frequencies.

#     Args:
#         delta_nu: Two elements containing the respective prior mean and standard 
#             deviation for the large frequency separation.
#         nu_max: Two elements containing the respective prior mean and standard
#             deviation for the frequency at maximum power.
#         epsilon: Two elements containing the respective prior mean and standard
#             deviation for the asymptotic phase term. Default is [1.3, 0.2] which is
#             suitable for main sequence solar-like oscillators.
#         alpha: Two elements containing the respective prior mean and standard
#             deviation for the asymptotic curvature term. Default is [0.0015, 0.002]
#             which is suitable for most solar-like oscillators.
#         nu: Locations of the l=0 (radial) stellar pulsation modes. Null data may be
#             passed as NaN. The shape of nu should be (num_orders,) or 
#             (N, num_orders) for N stars in the model.
#         nu_err: The observational uncertainty for each element of nu. Null data may be
#             passed as NaN. Default is None, for which only a star-by-star model
#             error is inferred and observational uncertainty is assumed to be zero.
#         n: The observers radial order of each mode in nu. Default is None and n is
#             inferred from num_orders.
#         num_orders: The number of radial orders to model. The observers n is inferred from
#             the priors for delta_nu, nu_max and epsilon. Default is None, and
#             num_orders is inferred from the length of the final dimension of nu.
#         num_pred: The number of predictions to make in the range of n. Default
#             is 200.
#     """
#     def __init__(
#         self,
#         name: Union[str, Array1D[str]],
#         delta_nu: Array1D[float],
#         nu_max: Array1D[float],
#         epsilon: Array1D[float]=None,
#         alpha: Array1D[float]=None,
#         *,
#         nu: Union[Array1D[float], Array2D[float]],
#         nu_err: Optional[Union[Array1D[float], Array2D[float]]]=None, 
#         n: Optional[Array1D[int]]=None,
#         num_orders: Optional[int]=None,
#         num_pred: int=200,
#         regularization: float=1e-6,
#         beta: Array1D[float]=[-14.0, 1.0],
#         gamma: Array1D[float]=[-12.0, 1.0],
#     ):

#         self._delta_nu = np.array(delta_nu)
#         self._nu_max = np.array(nu_max)
#         self._epsilon = np.array([1.3, 0.2]) if epsilon is None else np.array(epsilon)

#         if alpha is None:
#             self._log_alpha = np.array([-7.0, 1.0])  # natural logarithm
#         else:
#             self._log_alpha = np.array([
#                 np.log(alpha[0]**2 / np.sqrt(alpha[0]**2 + alpha[1]**2)),  # loc
#                 np.sqrt(np.log(1 + alpha[1]**2 / alpha[0]**2))             # scale
#             ])
        
#         super().__init__(
#             name, n, nu=nu, nu_err=nu_err,
#         )

#         self.num_pred = num_pred
#         self.n_pred = np.linspace(self.n[0], self.n[-1], num_pred)
#         self.dimensions["n_pred"] = dimension(
#             "n_pred", self.num_pred, coords=self.n_pred
#         )

#         self.regularization = regularization
#         self._beta = beta
#         self._gamma = gamma
#         # self.num_stars = 1 if len(self.name.shape) == 0 else self.name.shape[0]
#         # self.num_orders = self.n.shape[0]
#         # self.name = np.broadcast_to(self.name, (self.num_stars,))
#         # self.n = np.broadcast_to(self.n, (self.num_orders,))

#         # shape = (self.num_stars, self.num_orders)
#         # self.nu = np.broadcast_to(self.nu, shape)
#         # self.nu_err = np.broadcast_to(self.nu_err, shape)
#         # self.obs_mask = np.broadcast_to(self.obs_mask, shape)
    
#         # self.dimensions = {
#         #     'name': dimension('name', self.num_stars, coords=self.name),
#         #     'n': dimension('n', self.num_orders, coords=self.n)
#         # }

#     def _prior(self, pred: bool=False):

#         with self.dimensions["name"]:
#             epsilon = numpyro.sample("epsilon", dist.Normal(*self._epsilon))
#             alpha = numpyro.sample("alpha", dist.LogNormal(*self._log_alpha))

#             delta_nu = numpyro.sample("delta_nu", dist.Normal(*self._delta_nu))
#             nu_max = numpyro.sample("nu_max", dist.Normal(*self._nu_max))

#             # TODO: define these
#             beta = numpyro.sample("beta", dist.LogNormal(*self._beta))
#             gamma = numpyro.sample("gamma", dist.LogNormal(*self._gamma))
            
#             err = numpyro.sample("err", dist.HalfNormal(0.1))

#             dim_name = "n_pred" if pred else "n"
#             with self.dimensions[dim_name]:
#                 # Broadcast to the shape of output freq
#                 if pred:
#                     n = np.broadcast_to(self.n_pred, (self.num_stars, self.num_pred))
#                 else:
#                     n = np.broadcast_to(self.n, self._shape)

#                 nu = numpyro.deterministic("nu", asy_background(n, epsilon, alpha, delta_nu, nu_max, beta, gamma))

#                 # So factor is not done for predictions
#                 if numpyro.get_mask() is not False:
#                     # L2 regularisation on d3 nu_asy / d n3 same as Gaussian prior
#                     # with sd = 1/lambda where lambda = self.regularization
#                     log_prob = dist.Normal(0.0, 1/self.regularization).log_prob(6*beta - 24*gamma*n)
#                     numpyro.factor('reg', jnp.sum(log_prob))  # TODO: uncomment

#             # average_he = numpyro.deterministic('<he>', average_he_amplitude(...))
#             # he_nu_max = numpyro.deterministic('he_nu_max', he_amplitude(nu_max, b0, b1))
#         return nu, err

#     def _likelihood(self, nu, err):
#         # if self.nu_err is not None:
#         err = jnp.sqrt(err**2 + self.nu_err**2)

#         with self.dimensions["name"]:
#             with self.dimensions["n"]:
#                 nu_obs = numpyro.sample("nu_obs", dist.Normal(nu, err),
#                                         obs=self.nu, obs_mask=self.obs_mask)

#     # def _predictive(self, n=None):
#     #     self._likelihood(*self._prior(n=n))

#     # @property
#     # def predictive(self):
#     #     # def _predictive():
#     #     #     # nu, nu_err = self._prior()  # prior without reparam
#     #     #     self._likelihood(*self._prior(n=n))

#     #     return self._predictive

#     # def _predict(self):
#     #     self._prior(pred=True)
    
#     def _posterior(self):
#         self._likelihood(*self._prior())

#     def _predictions(self):
#         return self._prior(pred=True)


class Model:
    """Base Model class"""
    def __call__(self, n, nu=None, nu_err=None):
        raise NotImplementedError


class GlitchModel(Model):

    def __init__(self, background, he_glitch=None, cz_glitch=None):
        self.background = background
        if he_glitch is None:
            he_glitch = ZerosPrior()
        if cz_glitch is None:
            cz_glitch = ZerosPrior()
        self.he_glitch = he_glitch
        self.cz_glitch = cz_glitch

    def plot_glitch(self, data, kind='He', group='posterior', quantiles=None, observed=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        dim = ('chain', 'draw')
        pred_group = group + "_predictive"
        if pred_group not in data.keys():
            pred_group = group
        n = data[group]['n']
        dnu = data[group]['dnu_'+kind.lower()]

        if observed:
            if 'observed_data' in data.keys():
                res = data.observed_data['nu_obs'] - data[pred_group]['nu']
                dnu_obs = dnu + res
                ax.errorbar(n, dnu_obs.mean(dim=dim), yerr=dnu_obs.std(dim=dim),
                            color='C0', marker='o', linestyle='none', label='observed')
            else:
                warnings.warn('No \'observed_data\' found in data. Set observed=False to surpress this message', UserWarning)

        n_pred = data[pred_group].get('n_pred', n)
        dnu_pred = data[pred_group].get('dnu_'+kind.lower()+'_pred', dnu) 
        dnu_med = dnu_pred.median(dim=dim)
        ax.plot(n_pred, dnu_med, label='median', color='C1')

        if quantiles is not None:
            dnu_quant = dnu_pred.quantile(quantiles, dim=dim)
            num_quant = len(quantiles)//2
            alphas = np.linspace(0.1, 0.5, num_quant*2+1)
            for i in range(num_quant):
                delta = quantiles[-i-1] - quantiles[i]
                ax.fill_between(n_pred, dnu_quant[i], dnu_quant[-i-1],
                                color='C1', alpha=alphas[2*i+1], label=f'{delta:.1%} CI')
        
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(r'$n$')
        ax.set_ylabel(r'$\delta\nu_\mathrm{\,'+kind+'}\,(\mathrm{\mu Hz})$')
        ax.legend()
        return ax

    def __call__(self, n, nu=None, nu_err=None, pred=False, num_pred=250):
        bkg_func = self.background()
        he_glitch_func = self.he_glitch()
        cz_glitch_func = self.cz_glitch()

        def mean(n):
            nu_bkg = bkg_func(n)
            return nu_bkg + he_glitch_func(nu_bkg) + cz_glitch_func(nu_bkg)

        var = numpyro.param('kernel_var', 10.0)
        length = numpyro.sample('kernel_length', dist.Gamma(5.0))
        noise = numpyro.sample('noise', dist.HalfNormal(0.1))
        if nu_err is not None:
            noise = jnp.sqrt(noise**2 + nu_err**2)
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
            n_pred = np.linspace(n.min(), n.max(), num_pred)
            with dimension('n_pred', n_pred.shape[-1], coords=n_pred):
                gp.predict('nu_pred', n_pred)
                nu_bkg = numpyro.deterministic('nu_bkg_pred', bkg_func(n_pred))
                numpyro.deterministic('dnu_he_pred', he_glitch_func(nu_bkg))
                numpyro.deterministic('dnu_cz_pred', cz_glitch_func(nu_bkg))
