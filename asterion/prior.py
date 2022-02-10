from __future__ import annotations

from scipy import rand
from .nn import BayesianNN
from .annotations import DistLike
from .models import distribution
from jax import random
import jax.numpy as jnp
from numpyro import handlers
from . import PACKAGE_DIR
import os


class TauPrior:
    def __init__(self, nu_max: DistLike, teff: DistLike) -> None:
        self.nu_max = distribution(nu_max)
        self.teff = distribution(teff)
        self.log_tau_he = None
        self.log_tau_cz = None

    def condition(self, rng_key, num_obs=1000, kind='trained', num_samples=None):
        rng_key, key1, key2 = random.split(rng_key, 3)
        sample_shape = (num_obs,)
        log_numax = jnp.log10(
            self.nu_max.sample(key1, sample_shape=sample_shape)
        )
        teff = self.teff.sample(key2, sample_shape=sample_shape)
        x = jnp.stack([log_numax, teff], axis=-1)
            
        prior = BayesianNN.from_file(os.path.join(PACKAGE_DIR, 'prior.hdf5'))
        
        rng_key, key = random.split(rng_key)
        samples = prior.predict(key, x, kind=kind, num_samples=num_samples)
        tau_he = samples['y'][..., 0] - 6  # log mega-seconds
        tau_cz = samples['y'][..., 1] - 6  # log kega-seconds
        log_tau_he = distribution((tau_he.mean(), tau_he.std()))
        log_tau_cz = distribution((tau_cz.mean(), tau_cz.std()))
        return log_tau_he, log_tau_cz
