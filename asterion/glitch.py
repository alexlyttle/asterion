import jax.numpy as jnp

class Glitch:
    """Base class for Glitch models.
    
    Args:
        tau (float): The acoustic depth of the glitch.
        phi (float): The phase of the glitch.
    """
    def __init__(self, tau, phi):
        self.tau = tau
        self.phi = phi

    def amplitude(self, nu):
        """Amplitude of the glitch at a given frequency.

        Args:
            nu (:term:`array_like`): Frequency.

        Raises:
            NotImplementedError: Amplitude not implemented in base class.
        """
        raise NotImplementedError()

    def oscillation(self, nu):
        """Oscillatory component of the glitch at a given frequency.

        Args:
            nu (:term:`array_like`): Frequency.

        Returns:
            jax.numpy.ndarray: Oscillatory component of the glitch.
        """
        return jnp.sin(4.0 * jnp.pi * self.tau * nu + self.phi)

    def __call__(self, nu):
        """Returns the value of the glitch at the given frequency.

        Args:
            nu (:term:`array_like`): Frequency.

        Returns:
            jax.numpy.ndarray: Value of the glitch.
        """
        return self.amplitude(nu) * self.oscillation(nu)


class HeGlitch(Glitch):
    """Helium glitch class.
    
    Args:
        a (float): Amplitude parameter of the glitch.
        b (float): Decay parameter of the glitch.
        tau (float): The acoustic depth of the glitch.
        phi (float): The phase of the glitch.
    """
    def __init__(self, a, b, *args, **kwargs):
        self.a = a
        self.b = b
        super().__init__(*args, **kwargs)

    def amplitude(self, nu):
        """Amplitude of the helium glitch at a given frequency.

        Args:
            nu (:term:`array_like`): Frequency.

        Returns:
            jax.numpy.ndarray: Amplitude of the helium glitch.
        """
        return self.a * nu * jnp.exp(- self.b * nu**2)

    def average_amplitude(self, low, high):
        """Average amplitude of the helium glitch over a given range.

        Args:
            low (float): Lower frequency bound.
            high (float): Upper frequency bound.

        Returns:
            jax.numpy.ndarray: Average amplitude of the helium glitch.
        """
        return (
            self.a
            * (jnp.exp(- self.b * low**2) - jnp.exp(- self.b * high**2))
            / (2 * self.b * (high - low))
        )


class BCZGlitch(Glitch):
    """Base of the Convective Zone (BCZ) glitch class.

    Args:
        a (float): Amplitude parameter of the glitch.
        tau (float): The acoustic depth of the glitch.
        phi (float): The phase of the glitch.
    """
    def __init__(self, a, *args, **kwargs):
        self.a = a
        super().__init__(*args, **kwargs)

    def amplitude(self, nu):
        """Amplitude of the BCZ glitch at a given frequency.

        Args:
            nu (:term:`array_like`): Frequency.

        Returns:
            jax.numpy.ndarray: Amplitude of the BCZ glitch.
        """
        return self.a / nu**2
    
    def average_amplitude(self, low, high):
        """Average amplitude of the BCZ glitch over a given range.

        Args:
            low (float): Lower frequency bound.
            high (float): Upper frequency bound.

        Returns:
            jax.numpy.ndarray: Average amplitude of the BCZ glitch.
        """
        return self.a / low / high
