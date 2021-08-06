"""[summary]
"""
import math

import numpy as np

import jax
import jax.numpy as jnp

from jax import random

import numpyro
import numpyro.distributions as dist

from numpyro.infer.reparam import CircularReparam, Reparam
from numpyro import handlers
from numpyro.primitives import CondIndepStackFrame, plate

from typing import Union, Optional, Callable, Dict, ClassVar, Any
from .annotations import Array1D, Array2D, Array3D


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
    epsilon: Union[float, Array1D[float]],
) -> np.ndarray:
    """Estimates n from delta_nu, nu_max and epsilon, given num_orders about
    nu_max.

    Args:
        num_orders: [description]
        delta_nu: [description]
        nu_max: [description]
        epsilon: [description]

    Returns:
        [description]
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
    """ Derived average amplitude over the fitting range."""
    return b0 * (jnp.exp(-b1*low**2) - jnp.exp(-b1*high**2)) / (2*b1*(high - low))


class dimension(plate):
    """Context manager for variables along a given dimension with coordinates.

    Based on `numpyro.plate` but includes deterministics defined within the
    context.
    
    Args:
        name: Name of the dimension.
        size: Size of the dimension.
        subsample_size: Optional argument denoting the size of the mini-batch.
            This can be used to apply a scaling factor by inference algorithms.
            e.g. when computing ELBO using a mini-batch.
        dim: Optional argument to specify which dimension in the tensor
            is used as the plate dim. If `None` (default), the rightmost
            available dim is allocated.
        coords: Optional coordinates for each point in the dimension. If `None`
            (default), the coords are `np.arange(size)`.
    
    Attributes:
        name (str): Name of the dimension
        size (int): Size of the dimension
        subsample_size (int or None): Size of the mini-batch.
        dim (int or None): The dimension in the tensor which is used as the
            plate dim.
    """
    def __init__(self, name: str, size: int,
                 subsample_size: Optional[int]=None,
                 dim: Optional[int]=None, coords: Optional[Array1D[Any]]=None):
        super().__init__(name, size, subsample_size=subsample_size, dim=dim)
        self.coords: np.ndarray  #: [description]
        if coords is None:
            self.coords = np.arange(self.size)
        else:
            self.coords = np.array(coords)

    def process_message(self, msg: Dict[str, Any]):
        """Modified process message to also include deterministics. 
        
        Args:
            msg: Process message
        """
        super().process_message(msg)
        if msg["type"] == "deterministic":
            if "cond_indep_stack" not in msg.keys():
                msg["cond_indep_stack"] = []
            frame = CondIndepStackFrame(self.name, self.dim, self.subsample_size)
            msg["cond_indep_stack"].append(frame)


class Model:
    """Base Model class for modelling asteroseismic oscillation modes. 
    
    All models must have a `name` (usually the name of the star), array of radial order
    `n` and array of angular degree `l` (optional). The observed mode
    frequencies `nu` must then be broadcastable to an array with shape
    `num_stars` x `num_orders` (x `num_degrees`). Optionally, the
    observed uncertainties may be passed with the same shape via `nu_err`.
    Any unobserved or null data should be passed as NaN, and an `obs_mask`
    will be made which can be used in the likelihood.
    
    Args:
        name: [description]
        n: [description]
        nu: [description]
        l: [description]. Defaults to None.
        nu_err: [description]. Defaults to None.
    """
    reparam: ClassVar[Dict[str, Reparam]] = {}  #: [description]

    def __init__(
        self,
        name: Union[str, Array1D[str]],
        n: Array1D[int],
        l: Optional[Union[int, Array1D[int]]]=None, 
        *,
        nu: Union[Array1D[float], Array2D[float], Array3D[float]],
        nu_err: Optional[
            Union[Array1D[float], Array2D[float], Array3D[float]]
        ]=None,
    ):        
        self.name: np.ndarray = self._validate_name(name)  #: [description]
        self.n: np.ndarray = self._validate_n(n)  #: [description]
        self.l: np.ndarray = self._validate_l(l)  #: [description]

        self.num_stars: int = self.name.shape[0]  #: [description]
        self.num_orders: int = self.n.shape[0]  #: [description]
        self.num_degrees: Optional[int] = None

        self.dimensions: Dict[str, dimension] = {
            "name": dimension("name", self.num_stars, coords=self.name),
            "n": dimension("n", self.num_orders, coords=self.n)
        }  #: [description]
        self._shape = (self.num_stars, self.num_orders)

        if self.l is not None:
            self.num_degrees = self.l.shape[0]
            self._shape += (self.num_degrees,)
            self.dimension["l"] = dimension("l", self.num_degrees, coords=self.l)
        
        self.nu: np.ndarray = self._validate_nu(nu)  #: [description]
        self.obs_mask: np.ndarray = ~np.isnan(self.nu)  #: [description]
        self.nu_err: np.ndarray = self._validate_nu_err(nu_err)  #: [description]

    def _validate_name(self, name: Union[str, Array1D[str]]) -> np.ndarray:
        name = np.sqeeze(name)
        if name.ndim == 0:
            name = np.broadcast_to(name, (1,))
        elif name.ndim > 1:
            raise ValueError("Variable 'name' is greater than 1-D.")
        assert name.astype(str)
        assert name.ndim == 1
        return name

    def _validate_n(self, n: Array1D[int]) -> np.ndarray:
        n = np.squeeze(n)
        if n.ndim == 0 or n.ndim > 1:
            raise ValueError("Variable 'n' must be 1-D.")
        assert n.ndim == 1
        return np.array(n)
    
    def _validate_l(
        self, 
        l: Optional[Union[int, Array1D[int]]]
    ) -> Optional[np.ndarray]:
        if l is None:
            return
        l = np.squeeze(l)
        if l.ndim == 0:
            l = np.broadcast_to(l, (1,))
        elif l.ndim > 1:
            raise ValueError("Variable 'l' is greater than 1-D.")
        assert l.ndim == 1
        return np.array(l)
    
    def _broadcast(self, x) -> np.ndarray:
        return np.broadcast_to(x, self._shape)

    def _validate_nu(self, nu: Union[Array1D[float], Array2D[float], Array3D[float]]) -> np.ndarray:
        return self._broadcast(nu)
    
    def _validate_nu_err(self, nu_err: Optional[Union[Array1D[float], Array2D[float], Array3D[float]]]) -> np.ndarray:
        if nu_err is None:
            nu_err = np.zeros(self._shape)
        else:
            nu_err = self._broadcast(nu_err)
        nu_err[~self.obs_mask] = 0.0
        return nu_err

    def _prior(self):
        raise NotImplementedError

    def _predictions(self):
        raise NotImplementedError
    
    def _likelihood(self):
        raise NotImplementedError
    
    def _posterior(self):
        raise NotImplementedError

    @property
    def prior(self) -> Callable:
        """[summary]
        """        
        return self._prior

    @property
    def predictions(self) -> Callable:
        """[summary]
        """        
        return self._predictions

    @property
    def likelihood(self) -> Callable:
        """[summary]
        """        
        return self._likelihood

    @property
    def posterior(self) -> Callable:
        """[summary]
        """        
        return self._posterior

    def _get_trace(self, rng_key, model):
        model = handlers.trace(
            handlers.seed(
                model, rng_key
            )
        )
        trace = model.get_trace()
        return trace

    def get_prior_trace(self, rng_key: Union[int, jax.random.PRNGKey]) -> dict:
        """[summary]

        Args:
            rng_key: [description]

        Returns:
            [description]
        """        
        return self._get_trace(rng_key, self.prior)
    
    def get_posterior_trace(self, rng_key: Union[int, jax.random.PRNGKey]) -> dict:
        """[summary]

        Args:
            rng_key: [description]

        Returns:
            [description]
        """        
        return self._get_trace(rng_key, self.posterior)


class GlitchModel(Model):
    """
    Args:
        delta_nu: Two elements containing the respective prior mean and standard 
            deviation for the large frequency separation.
        nu_max: Two elements containing the respective prior mean and standard
            deviation for the frequency at maximum power.
        epsilon: Two elements containing the respective prior mean and standard
            deviation for the asymptotic phase term. Default is [1.3, 0.2] which is
            suitable for main sequence solar-like oscillators.
        alpha: Two elements containing the respective prior mean and standard
            deviation for the asymptotic curvature term. Default is [0.0015, 0.002]
            which is suitable for most solar-like oscillators.
        nu: Locations of the l=0 (radial) stellar pulsation modes. Null data may be
            passed as NaN. The shape of nu should be (num_orders,) or 
            (N, num_orders) for N stars in the model.
        nu_err: The observational uncertainty for each element of nu. Null data may be
            passed as NaN. Default is None, for which only a star-by-star model
            error is inferred and observational uncertainty is assumed to be zero.
        n: The observers radial order of each mode in nu. Default is None and n is
            inferred from num_orders.
        num_orders: The number of radial orders to model. The observers n is inferred from
            the priors for delta_nu, nu_max and epsilon. Default is None, and
            num_orders is inferred from the length of the final dimension of nu.
    """
    reparam = {
        "phi_he": CircularReparam(),
        "phi_cz": CircularReparam(),
    }

    def __init__(
        self,
        name: Union[str, Array1D[str]],
        delta_nu: Array1D[float],
        nu_max: Array1D[float],
        epsilon: Array1D[float]=None,
        alpha: Array1D[float]=None,
        *,
        nu: Union[Array1D[float], Array2D[float], Array3D[float]],
        nu_err: Optional[Union[Array1D[float], Array2D[float], Array3D[float]]]=None, 
        n: Optional[Array1D[int]]=None,
        num_orders: Optional[int]=None
    ):

        self._delta_nu = np.array(delta_nu)
        self._nu_max = np.array(nu_max)
        self._epsilon = np.array([1.3, 0.2]) if epsilon is None else np.array(epsilon)

        if alpha is None:
            self._log_alpha = np.array([-7.0, 1.0])  # natural logarithm
        else:
            self._log_alpha = np.array([
                np.log(alpha[0]**2 / np.sqrt(alpha[0]**2 + alpha[1]**2)),  # loc
                np.sqrt(np.log(1 + alpha[1]**2 / alpha[0]**2))             # scale
            ])
        
        super().__init__(
            name, n, nu=nu, nu_err=nu_err,
        )

        # self.num_stars = 1 if len(self.name.shape) == 0 else self.name.shape[0]
        # self.num_orders = self.n.shape[0]
        # self.name = np.broadcast_to(self.name, (self.num_stars,))
        # self.n = np.broadcast_to(self.n, (self.num_orders,))

        # shape = (self.num_stars, self.num_orders)
        # self.nu = np.broadcast_to(self.nu, shape)
        # self.nu_err = np.broadcast_to(self.nu_err, shape)
        # self.obs_mask = np.broadcast_to(self.obs_mask, shape)
    
        # self.dimensions = {
        #     'name': dimension('name', self.num_stars, coords=self.name),
        #     'n': dimension('n', self.num_orders, coords=self.n)
        # }

    def _prior(self):
        m_tau = 1.05
        # s_tau = 3.0
        log_tau = - m_tau * np.log(self._nu_max[0])  # Approx form of log(tau_he)

        with self.dimensions["name"]:
            epsilon = numpyro.sample("epsilon", dist.Normal(*self._epsilon))
            alpha = numpyro.sample("alpha", dist.LogNormal(*self._log_alpha))

            delta_nu = numpyro.sample("delta_nu", dist.Normal(*self._delta_nu))
            nu_max = numpyro.sample("nu_max", dist.Normal(*self._nu_max))
        
            b0 = numpyro.sample("b0", dist.LogNormal(np.log(100/self._nu_max[0]), 1.0))
            # b0 = numpyro.sample("b0", dist.HalfNormal(100/self._nu_max[0]))
            b1 = numpyro.sample("b1", dist.LogNormal(np.log(10/self._nu_max[0]**2), 1.0))

            tau_he = numpyro.sample("tau_he", dist.LogNormal(log_tau, 0.8))
            phi_he = numpyro.sample("phi_he", dist.VonMises(0.0, 0.1))

            c0 = numpyro.sample("c0", dist.LogNormal(np.log(0.5*self._nu_max[0]**2), 1.0))

            # Ensure that tau_cz > tau_he
            delta_tau = numpyro.sample("delta_tau", dist.LogNormal(log_tau, 0.8))
            tau_cz = numpyro.deterministic("tau_cz", tau_he + delta_tau)

            phi_cz = numpyro.sample("phi_cz", dist.VonMises(0.0, 0.1))
            
            err = numpyro.sample("err", dist.HalfNormal(0.1))

            with self.dimensions["n"]:
                # n = self.n_pred if pred else self.n
                n = np.broadcast_to(self.n, self._shape)  # Broadcast to the shape of output freq
                nu_asy = numpyro.deterministic("nu_asy", asy_background(n, epsilon, alpha, delta_nu, nu_max))

                dnu_he = he_glitch(nu_asy, b0, b1, tau_he, phi_he)
                dnu_cz = cz_glitch(nu_asy, c0, tau_cz, phi_cz)

                nu = numpyro.deterministic("nu", nu_asy + dnu_he + dnu_cz)

            # average_he = numpyro.deterministic('<he>', average_he_amplitude(...))
            # he_nu_max = numpyro.deterministic('he_nu_max', he_amplitude(nu_max, b0, b1))
        return nu, err

    def _likelihood(self, nu, err):
        # if self.nu_err is not None:
        err = jnp.sqrt(err**2 + self.nu_err**2)

        with self.dimensions["name"]:
            with self.dimensions["n"]:
                nu_obs = numpyro.sample("nu_obs", dist.Normal(nu, err),
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
