import jax.numpy as jnp

from jax import random
from jax.scipy.stats import norm, multivariate_normal
from jaxns.prior_transforms import PriorChain, MVNPrior, NormalPrior, LogNormalPrior, UniformPrior

from tinygp import GaussianProcess as TinyGaussianProcess


class Distribution:
    def __init__(self, name):
        self.name = name

    def jaxns_dist(self):
        raise NotImplementedError()

    def numpyro_dist(self):
        raise NotImplementedError()

    def log_probability(self, x):
        raise NotImplementedError()
    
    def sample(self, key, shape=()):
        raise NotImplementedError()
    
    def prior_chain(self):
        with PriorChain() as _prior_chain:
            _ = self.jaxns_dist()
        return _prior_chain


class Normal(Distribution):
    def __init__(self, name, loc=0.0, scale=1.0):
        super().__init__(name)
        self.loc = loc
        self.scale = scale
    
    def jaxns_dist(self):
        return NormalPrior(self.name, self.loc, self.scale)
    
    def log_probability(self, x):
        return norm.logpdf(x, loc=self.loc, scale=self.scale)
    
    def sample(self, key, shape=()):
        return self.loc + self.scale * random.normal(key, shape=shape)


class LogNormal(Distribution):
    def __init__(self, name, loc=0.0, scale=1.0):
        super().__init__(name)
        self.loc = loc
        self.scale = scale
    
    def jaxns_dist(self):
        return LogNormalPrior(self.name, self.loc, self.scale)

    def log_probability(self, x):
        return norm.logpdf(jnp.log(x), loc=self.loc, scale=self.scale)

    def sample(self, key, shape=()):
        return jnp.exp(self.loc + self.scale * random.normal(key, shape=shape))


class Uniform(Distribution):
    def __init__(self, name, low=0.0, high=1.0):
        super().__init__(name)
        self.low = low
        self.high = high
    
    def jaxns_dist(self):
        return UniformPrior(self.name, self.low, self.high)
    
    def log_probability(self, x):
        return jnp.log(1.0) - jnp.log(self.high - self.low)

    def sample(self, key, shape=()):
        return self.low + (self.high - self.low) * random.uniform(key, shape=shape)


class JointDistribution(Distribution):
    """
    Takes distributions, an iterable of Distributions or callables which
    return distributions.
    
    Example:

        distributions = [
            Normal("a"),
            Normal("b"),
            lambda a, b: Normal("c", a)
        ]
        joint_dist = JointDistribution("joint_dist", distributions)

    """
    def __init__(self, name, distributions):
        super().__init__(name)
        self.distributions = distributions

    def jaxns_dist(self):
        params = []
        for fn in self.distributions:
            if isinstance(fn, Distribution):
                params.append(fn.jaxns_dist())
            elif callable(fn):
                params.append(fn(*params).jaxns_dist())
        return params

    def log_probability(self, **x):
        params = x.copy()
        logp = 0.0
        for fn in self.distributions[::-1]:
            # Loop through distributions in reverse
            if isinstance(fn, Distribution):
                logp += fn.log_probability(params.pop(fn.name))
            elif callable(fn):
                logp += fn(*params.values()).log_probability(params.pop(fn.name))
        return logp
    
    def sample(self, key, shape=()):
        samples = {}
        keys = random.split(key, len(self.distributions))
        for k, fn in zip(keys, self.distributions):
            if isinstance(fn, Distribution):
                samples[fn.name] = fn.sample(k, shape=shape)
            elif callable(fn):
                d = fn(*samples.values())
                samples[d.name] = d.sample(k, shape=shape)
        return samples


class MVNormal(Distribution):
    """Wrapper for tinygp Gaussian process which inherits Distribution."""
    def __init__(self, name, loc, cov):
        # first argument is reserved for name
        super().__init__(name)
        self.loc = loc
        self.covariance = cov

    def jaxns_dist(self):
        return MVNPrior(self.name, self.loc, self.cov)

    def sample(self, key, shape=()):
        return random.multivariate_normal(key, self.loc, self.cov, shape=shape)

    def log_probability(self, x):
        return multivariate_normal.logpdf(x, self.loc, self.cov)
