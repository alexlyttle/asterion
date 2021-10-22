import jax.numpy as jnp


class Kernel:
    """Base class for a GP kernel."""
    def __call__(self, x, xp):
        """Returns a covariance matrix.
        
        Args:
            x (array-like): First input vector.
            xp (array-like): Second input vector. Can be optional.

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
    """White noise kernel.
    
    Args:
        scale (float or array-like): The scale of the white noise ($\sigma$).
    
    Attributes:
        scale (jnp.ndarray): The scale of the white noise ($\sigma$).
        heteroscedastic (bool): True if the scale is heteroscedastic, i.e.
            the white noise varies across the elements of x.
    """
    def __init__(self, scale):
        self.scale = jnp.array(scale)
        # self.heteroscedastic = (self.scale.shape != ())
        self.heteroscedastic = False  # TODO no need for this.
        # assert len(self.scale.shape) < 2

    def __call__(self, x, xp=None):
        """Returns the white noise covariance matrix.
        
        Args:
            x (array-like): First input vector.
            xp (array-like, optional): Second input vector. If x is not xp, 
                returns zeros((x.shape[0], xp.shape[0])).
        
        Raises:
            ValueError: Inputs x and xp must have the same shape as the
                scale if white noise is heteroscedastic.
        """
#         cov = jnp.zeros((x.shape[0], xp.shape[0]))
        if x is xp or xp is None:
#             jnp.fill_diagonal(cov, self.scale**2)
            if self.heteroscedastic and x.shape[0] != self.scale.shape[0]:
                raise ValueError(f"Inputs must have shape {self.scale.shape}")
            return self.scale**2 * jnp.eye(x.shape[0])
        
        return jnp.zeros((x.shape[0], xp.shape[0]))


class SquaredExponential(Kernel):
    """Squared exponential kernel.
    
    Args:
        var (float): Variance (or amplitude, $\sigma^2$) of the kernel.
        length (float): Length-scale ($\lambda$) of the kernel.
    """
    def __init__(self, var, length):
        self.var = var
        self.length = length

    def __call__(self, x, xp):
        """Returns the squared exponential covariance matrix."""
        exponant = jnp.power((xp[:, None] - x) / self.length, 2.0)
        # exponant = jnp.power((xp[..., None] - x[..., None, :]) / self.length, 2.0)
        cov = self.var * jnp.exp(-0.5 * exponant)
        return cov
