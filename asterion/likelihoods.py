import jax.numpy as jnp
from .distributions import Distribution
from tinygp import kernels, GaussianProcess


class Model:
    """Base class for model likelihoods. Call this to get the distribution
    representing the probability of the data given the model parameters."""
    def __call__(self) -> Distribution:
        raise NotImplementedError


class Glitch(Model):
    def __init__(self, *, delta_nu, epsilon, a_he, b_he, tau_he, phi_he,
                 a_cz, tau_cz, phi_cz, kernel_amp, kernel_scale):
        self.delta_nu = delta_nu
        self.epsilon = epsilon
        self.a_he = a_he
        self.b_he = b_he
        self.tau_he = tau_he
        self.phi_he = phi_he
        self.a_cz = a_cz
        self.tau_cz = tau_cz
        self.phi_cz = phi_cz
        self.kernel_amp = kernel_amp
        self.kernel_scale = kernel_scale
        
    def asymptotic(self, n):
        return self.delta_nu * (n + self.epsilon)
    
    @staticmethod
    def _oscillation(nu, *, tau, phi):
        return jnp.sin(4 * jnp.pi * tau * nu + phi)
    
    def heII_amplitude(self, nu):
        return self.a_he * nu * jnp.exp(- self.b_he * nu**2)

    def heII_glitch(self, nu):
        osc = self._oscillation(nu, tau=self.tau_he, phi=self.phi_he)
        return self.heII_amplitude(nu) * osc

    def bcz_amplitude(self, nu):
        return self.a_cz / nu**2

    def bcz_glitch(self, nu):
        osc = self._oscillation(nu, tau=self.tau_cz, phi=self.phi_cz)
        return self.bcz_amplitude(nu) * osc

    def mean(self, n):
        nu_asy = self.asymptotic(n)
        dnu_he = self.heII_glitch(nu_asy)
        dnu_cz = self.bcz_glitch(nu_asy)
        return nu_asy + dnu_he + dnu_cz

    @property
    def kernel(self):
        return self.kernel_amp * kernels.ExpSquared(self.kernel_scale)

    def build_gp(self, n, **kwargs):
        return GaussianProcess(self.kernel, n, mean=self.mean, **kwargs)
