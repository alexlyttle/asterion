import numpyro 
import numpyro.distributions as dist
import jax 
import jax.numpy as jnp

from .kernels import WhiteNoise

class GP:
    """Gaussian process class.

    The function f(x) is described by a Gaussian process: a collection of
    random variables for which any finite number are a part of a multivariate
    normal distribution.
    
    f(x) ~ GP(m(x), k(x, x')),

    where m(x) and m(x, x') are our expectation for the mean and covariance
    of f(x) and f(x'). I.e. k(x, x') = cov(f(x), f(x')).

    Models y = f(x) or, y = f(x) + n_x if there is additional independent
    noise.

    The kernel implies a distribution over all possible functional forms of f.
    Thus, f evaluated at some set of points x is drawn from a multivariate
    normal distribution,

        f ~ N(m(x), k(x, x))
    
    or with additional independent noise for each observation of f(x),
        y ~ N(m(x), k(x, x) + n_x)
    
    Likelihood (just add observation),

        y_obs | theta ~ N(y_obs | m(x), k(x, x) + n_x)
    
    Predicted truth,

        f(x_pred) | y ~ N(
            m(x_pred) + k(x, x_pred)·(k(x, x) + n_x)^{-1}·(y - m(x)),
            k(x_pred, x_pred) - k(x, x_pred)·(k(x, x) + n_x)^{-1}·k(x_pred, x)
        )

    Predicted observations (just add noise),
    
        y_pred | y ~ N(
            m(x_pred) + k(x, x_pred)·(k(x, x) + n_x)^{-1}·(y - m(x)),
            k(x_pred, x_pred) + n_pred - k(x, x_pred)·(k(x, x) + n_x)^{-1}·k(x_pred, x)
        )

    Args:
        kernel (Kernel or callable): Kernel function,
            default is the squared exponential kernel.
        mean (float or callable): Mean model function. If float, mean
            function is constant at this value.
        jitter (float or callable): Small amount to add to the covariance.
            If float, this is multiplied by the identity matrix.

    Attributes:
        kernel (callable): Kernel function.
        mean (callable): Mean function.
        jitter (callable): Jitter function.
        x (jnp.ndarray or None): Input array passed to distribution or sample.
        y (jnp.ndarray or None): Output array of sample.
        loc (jnp.ndarray or None): Output of mean(x).
        cov (jnp.ndarray or None): Output of kernel(x, x).
        noise (callable or None): Noise function passed to distribution or 
            sample.

    Methods:
        distribution: Returns the distribution of y | theta.
        sample: Returns a sample from the distribution of y | theta.
        conditional: Returns the conditional distribution of y* | y, theta.
        predict: Samples the conditional distribution of y* given y and theta.
    
    Example:

        .. code-block:: python

            import numpyro
            import numpyro.distributions as dist
            from asterion.gp import GP, SquaredExponential

            def model(x, x_pred=None, y=None):
                var = numpyro.sample('var', dist.HalfNormal(1.0))
                length = numpyro.sample('length', dist.Normal(100.0, 1.0))
                noise = numpyro.sample('noise', dist.HalfNormal(0.1))

                kernel = SquaredExponential(var, length)
                gp = GP(kernel)
                gp.sample('y', x, noise=noise, obs=y)

                if x_pred is not None:
                    gp.predict('f_pred', x_pred, noise=noise)
                    gp.predict('y_pred', x_pred, noise=noise)

    """
    def __init__(self, kernel, mean=0.0, jitter=1e-6):
        
        if not callable(kernel):
            raise TypeError("Argument 'kernel' is not callable")
        self.kernel = kernel

        if not callable(mean):
            self.mean = lambda x: jnp.full(x.shape, mean)
        else:
            self.mean = mean
        
        if not callable(jitter):
            self.jitter = lambda x: jitter * jnp.eye(x.shape[-1])
        else:
            self.jitter = jitter

        self.x = None
        self.y = None
        self.noise = None
        self.loc = None
        self.cov = None
    
    def __add__(self, obj):
        if not isinstance(obj, self.__class__):
            raise TypeError(f"Object added must be instance of {self.__class__}")
        kernel = self.kernel + obj.kernel
        mean = self.mean + obj.mean
        jitter = max(self.jitter, obj.jitter)  # Take the max jitter.
        gp = GP(kernel, mean=mean, noise=noise, jitter=jitter)
        return gp

    def _validate_noise(self, noise):
        if noise is None or noise is False:
            noise = WhiteNoise(0.0)
        elif not callable(noise):
            noise = WhiteNoise(noise)
        return noise

    def distribution(self, x, noise=None, **kwargs):
        """Distribution for the GP.

        Args:
            params (dict): Kernel and mean parameters.
            x: The x values for which to sample.
            kwargs (dict): Keyword arguments to pass to dist.MultivariateNormal 
        """
        self.x = x
        self.noise = self._validate_noise(noise)
        self.loc = self.mean(x)
        self.cov = self.kernel(x, x) + self.noise(x) + self.jitter(x)
        return dist.MultivariateNormal(self.loc, self.cov, **kwargs)
    
    def sample(self, name, x, noise=None, obs=None, rng_key=None,
               sample_shape=(), infer=None, obs_mask=None, **kwargs):
        fn = self.distribution(x, noise=noise, **kwargs)
        self.y = numpyro.sample(name, fn, obs=obs, rng_key=rng_key,
                                sample_shape=sample_shape, infer=infer,
                                obs_mask=obs_mask)
        return self.y

    def _build_conditional(self, x, noise=None, gp=None, diag=False):
        """Make a prediction for the loc and cov of f(x_pred) given y,
        
        loc = mp + kxp·(kxx + nx)^{-1}·(y - mx),
        cov = kpp + np - kxp·(kxx + nx)^{-1}·kxp^T,
        var = sqrt(diag(cov)),
        
        where mx = mean(x), mp = mean(x_pred), kxx = kernel(x, x),
        kxp = kernel(x, x_pred), kpp = kernel(x_pred, x_pred),
        nx = noise(x), and np = noise(x_pred).

        Args:
            x: The x values for which to make predictions.
            noise (bool, callable or array-like): If True, add self.noise to the
                prediction. If callable, must be a function of (x_pred, x_pred).
                Otherwise, pass the scale parameter for WhiteNoise. Default is False 
                (no noise).
            gp (GP, optional): The GP from which to make predictions. E.g. the total
                GP in which self is a term. Default is self. 
            diag (bool): If True, returns the variance. Default is False.
        
        Returns:
            loc: The mean of the prediction.
            cov or var: The covarience or variance of the prediction.
        """
        if gp is None:
            # Predict given a different GP (e.g. additive)
            gp = self
        
        if gp.x is None:
            raise ValueError("GP must be sampled to make predictions, consider the `gp` keyword argument")

        kxx = gp.cov    
        L = jax.scipy.linalg.cho_factor(kxx, lower=True)
        A = jax.scipy.linalg.cho_solve(L, gp.y - gp.loc)
        
        # Cross terms and prediction terms are always self.
        kxp = self.kernel(gp.x, x)
        v = jax.scipy.linalg.cho_solve(L, kxp.T)

        kpp = self.kernel(x, x) + self.jitter(x)
        
        noise = self._validate_noise(noise)
        if noise is True:
            noise = gp.noise

        kpp += noise(x)

        loc = self.mean(x) + jnp.dot(kxp, A)
        cov = kpp - jnp.dot(kxp, v)
          
        if diag:
            var = jnp.diag(cov)
            return loc, var

        return loc, cov

    def conditional(self, x, noise=None, diag=False, gp=None, **kwargs):
        """Make a conditional distribution for y_pred = f(x_pred) + noise_pred,
        
        y_pred | y ~ N(
            mp + kxp·(kxx + nx)^{-1}·(y - mx),
            kpp + np - kxp·(kxx + nx)^{-1}·kxp^T
        ),
        
        where mx = mean(x), mp = mean(x_pred), kxx = kernel(x, x),
        kxp = kernel(x, x_pred), kpp = kernel(x_pred, x_pred),
        nx = noise(x), and np = noise(x_pred).
        
        Args:
            x: The x values for which to make predictions.
            noise (bool, callable or array-like): If True, add self.noise to the
                prediction. If callable, must be a function of (x_pred, x_pred). 
                Otherwise, pass the scale parameter for WhiteNoise. Default is False
                (no noise).
            gp (GP, optional): The GP from which to make predictions. Default is
                self. E.g. the total GP in which self is a term.
            diag (bool): If True, diagonalises the varience. Default is False.
        """
        args = self._build_conditional(x, noise=noise, gp=gp, diag=diag)
        if diag:
            return dist.Normal(*args, **kwargs)

        return dist.MultivariateNormal(*args, **kwargs)
    
    def predict(self, name, x, noise=None, gp=None, diag=False,
                rng_key=None, sample_shape=(), infer=None, **kwargs):
        fn = self.conditional(x, noise=noise, gp=gp, diag=diag, **kwargs)
        return numpyro.sample(name, fn, rng_key=rng_key, sample_shape=sample_shape,
                              infer=infer)
