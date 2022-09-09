import jax.numpy as jnp

class Asy:
    """Base class for asymptotic expressions."""
    pass


class FirstOrderAsy(Asy):
    """First-order asymptotic expression for radial mode frequencies only.
    
    Args:
        delta_nu (float): Large frequency separation.
        epsilon (float): Phase (offset).
    """
    def __init__(self, delta_nu, epsilon):
        self.delta_nu = delta_nu
        self.epsilon = epsilon
    
    def __call__(self, n):
        """The asymptotic mode frequency at a given radial order.
        
        Args:
            n (:term:`array_like`): Radial order.

        Returns:
            jax.numpy.ndarray: Asymptotic mode frequency.
        """
        return self.delta_nu * (jnp.array(n) + self.epsilon)
