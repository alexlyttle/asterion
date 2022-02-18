"""Gaussian process kernels."""
from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    'Kernel',
    'WhiteNoise',
    'SquaredExponential',
]


class Kernel:
    """Abstract base class for a GP kernel.
    """
    def __call__(self, x, xp):
        """Returns a covariance matrix.
        
        Args:
            x (:term:`array_like`): First input vector.
            xp (:term:`array_like`): Second input vector. Can be optional.

        Raises:
            NotImplementedError
        """
        NotImplementedError
    
    def __add__(self, obj):
        if not callable(obj):
            raise TypeError("Added object must be callable")
        kernel = Kernel()
        def call(x, xp):
            return self(x, xp) + obj(x, xp)
        kernel.__call__ = call
        return kernel


class WhiteNoise(Kernel):
    r"""White noise kernel.
    
    .. math::
        :nowrap:

        \begin{equation}
            k(x_i, x_j) = \begin{cases}
                \sigma^2 &\quad \mathrm{if}\,i=j,\\
                0 &\quad \mathrm{else}.
            \end{cases}
        \end{equation}

    Args:
        scale (float, or :term:`array_like`): The scale of the white noise 
            (:math:`\sigma`).
    """
    def __init__(self, scale):
        self.scale = jnp.array(scale)
        # self.heteroscedastic = (self.scale.shape != ())
        self.heteroscedastic = False  # TODO no need for this.
        # assert len(self.scale.shape) < 2

    def __call__(self, x, xp=None):
        """Returns the white noise covariance matrix.
        
        Args:
            x (:term:`array_like`): First input vector.
            xp (:term:`array_like`, optional): Second input vector. If x is not xp, 
                returns zeros((x.shape[0], xp.shape[0])).
        
        Raises:
            ValueError: Inputs x and xp must have the same shape as the
                scale if white noise is heteroscedastic.
        
        Returns:
            jax.numpy.ndarray: Covariance matrix.
        """
#         cov = jnp.zeros((x.shape[0], xp.shape[0]))
        if x is xp or xp is None:
#             jnp.fill_diagonal(cov, self.scale**2)
            if self.heteroscedastic and x.shape[0] != self.scale.shape[0]:
                raise ValueError(f"Inputs must have shape {self.scale.shape}")
            return self.scale**2 * jnp.eye(x.shape[0])
        
        return jnp.zeros((x.shape[0], xp.shape[0]))


class SquaredExponential(Kernel):
    r"""Squared exponential kernel.

    .. math::

        k(x_i, x_j) = \sigma^2 \exp\left[ \frac{(x_j - x_i)^2}{2 \lambda^2} \right]
    
    Args:
        var (float): Variance (or amplitude, :math:`\sigma^2`) of the kernel.
        length (float): Length-scale (:math:`\lambda`) of the kernel.
    """
    def __init__(self, var, length):
        self.var = var
        self.length = length

    def __call__(self, x, xp):
        """Returns the squared exponential covariance matrix.
        
        Returns:
            jax.numpy.ndarray: Covariance matrix.
        """
        exponant = jnp.power((xp[:, None] - x) / self.length, 2.0)
        # exponant = jnp.power((xp[..., None] - x[..., None, :]) / self.length, 2.0)
        cov = self.var * jnp.exp(-0.5 * exponant)
        return cov
