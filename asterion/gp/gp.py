from __future__ import annotations

from typing import Callable, Optional, Union
import numpyro 
import numpyro.distributions as dist
import jax 
import jax.numpy as jnp

from numpy.typing import ArrayLike

from .kernels import Kernel, WhiteNoise


__all__ = [
    'GP',
]


class GP:
    r"""Gaussian process class.

    The function f(x) is described by a Gaussian process: a collection of
    random variables for which any finite number are a part of a multivariate
    normal distribution.

    .. math::
    
        f(x) \sim \mathcal{GP}(m(x), k(x, x')),

    where :math:`m(x)` and :math:`k(x, x')` are the mean and covariance
    of :math:`f(x)` and :math:`f(x')`. I.e. :math:`k(x, x') = \mathrm{Cov}(f(x), f(x'))`.

    The kernel implies a distribution over all possible functional forms of :math:`f`.
    For example, :math:`f` given :math:`x` is drawn from a multivariate
    normal distribution,

    .. math::

        f | x \sim \mathcal{N}(m(x), k(x, x)).
    
    The marginal likelihood of some observation :math:`y` of :math:`f(x)`,

    .. math::

        p(y | x) = \int p(y | f, x)\,p(f | \theta)\,\mathrm{d}f
    
    can be shown to be,

    .. math::

        y | x \sim \mathcal{N}(m(x), k(x, x) + n(x))

    where :math:`n(x)` is some uncorrelated Gaussian noise term such that
    :math:`y | f \sim \mathcal{N}(f, n(x))`.

    Making predictions from the GP given :math:`x` and :math:`y` for some
    new points :math:`x_\star`,

    .. math::
        :nowrap:

        \begin{equation}
            \begin{bmatrix}
                f_\star\\y
            \end{bmatrix}
            \sim \mathcal{N} \left(
            \begin{bmatrix}
                m(x_\star)\\m(x)
            \end{bmatrix}
            ,\,
            \begin{bmatrix}
                k(x_\star, x_\star) & k(x_\star, x) \\
                k(x, x_\star) & k(x, x) + n(x)
            \end{bmatrix}
            \right)
        \end{equation}
    
    or making predictions with noise :math:`n(x_star)`,

    .. math::
        :nowrap:

        \begin{equation}
            \begin{bmatrix}
                y_\star\\y
            \end{bmatrix}
            \sim \mathcal{N} \left(
            \begin{bmatrix}
                m(x_\star)\\m(x)
            \end{bmatrix}
            ,\,
            \begin{bmatrix}
                k(x_\star, x_\star) + n(x_\star) & k(x_\star, x) \\
                k(x, x_\star) & k(x, x) + n(x)
            \end{bmatrix}
            \right)
        \end{equation}
    
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
        kernel (Kernel, or function): Kernel function,
            default is the squared exponential kernel.
        mean (function): Mean model function. If float, mean
            function is constant at this value.
        jitter (float, or function): Small amount to add to the covariance.
            If float, this is multiplied by the identity matrix.
    
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
                    gp.predict('f_pred', x_pred, noise=None)
                    gp.predict('y_pred', x_pred, noise=noise)

    """
    def __init__(self, kernel: Union[Kernel, Callable], mean: Union[float, Callable]=0.0,
                 jitter: Union[float, Callable]=1e-6):
        
        if not callable(kernel):
            raise TypeError("Argument 'kernel' is not callable")
        self.kernel: Callable = kernel  #: Kernel function.

        if not callable(mean):
            _mean = lambda x: jnp.full(x.shape, mean)
        else:
            _mean = mean
        self.mean: Callable = _mean  #: Mean function.

        if not callable(jitter):
            _jitter = lambda x: jitter * jnp.eye(x.shape[-1])
        else:
            _jitter = jitter

        self.jitter: Callable = _jitter  #: Jitter function.
        
        self.noise: Optional[Callable] = None
        """: Independent noise function passed to
        :meth:`distribution` or :meth:`sample`."""
        
        self.x: Optional[ArrayLike] = None
        """: Input array passed to :meth:`distribution` or :meth:`sample`."""
        
        self.y: Optional[ArrayLike] = None
        """: Output of :meth:`sample`."""
        
        self.loc: Optional[ArrayLike] = None
        """: Output of :attr:`mean` :code:`(x)`."""
        
        self.cov: Optional[ArrayLike] = None
        """: Output of :attr:`kernel` :code:`(x, x)`."""

    def __add__(self, obj):
        if not isinstance(obj, self.__class__):
            raise TypeError(
                f"Object added must be instance of {self.__class__}"
            )
        kernel = self.kernel + obj.kernel
        mean = self.mean + obj.mean
        jitter = max(self.jitter, obj.jitter)  # Take the max jitter.
        gp = GP(kernel, mean=mean, jitter=jitter)
        return gp

    def _validate_noise(self,
                        noise: Optional[Union[Callable, float]]) -> Callable:
        if noise is None or noise is False:
            noise = WhiteNoise(0.0)
        elif not callable(noise):
            noise = WhiteNoise(noise)
        return noise

    def distribution(self, x: ArrayLike,
                    noise: Optional[Union[Callable, float]]=None,
                    **kwargs) -> dist.MultivariateNormal:
        """Distribution for the GP. Calling this method updates :attr:`x`,
        :attr:`noise`, :attr:`loc` and :attr:`cov`.

        Args:
            x: The x values for which to construct the distribution
            noise: The independent noise function.
            **kwargs: Keyword arguments to pass to dist.MultivariateNormal.
        """
        self.x = x
        self.noise = self._validate_noise(noise)
        self.loc = self.mean(x)
        self.cov = self.kernel(x, x) + self.noise(x) + self.jitter(x)
        return dist.MultivariateNormal(self.loc, self.cov, **kwargs)
    
    def sample(self, name: str, x: ArrayLike,
               noise: Optional[Union[Callable, float]]=None,
               obs: Optional[ArrayLike]=None,
               rng_key: Optional[jnp.ndarray]=None,
               sample_shape: tuple=(), infer: Optional[dict]=None,
               obs_mask: Optional[ArrayLike]=None,
               **kwargs) -> jnp.ndarray:
        """Sample from the GP likelihood. Calling this method updates
        :attr:`x`, :attr:`noise`, :attr:`loc` and :attr:`cov` and assigns the
        result to :attr:`y`.

        Args:
            name: [description]
            x: [description]
            noise: [description]. Defaults to None.
            obs: [description]. Defaults to None.
            rng_key: [description]. Defaults to None.
            sample_shape: [description]. Defaults to ().
            infer: [description]. Defaults to None.
            obs_mask: [description]. Defaults to None.

        Returns:
            A sample from the GP likelihood.
        """
        fn = self.distribution(x, noise=noise, **kwargs)
        self.y = numpyro.sample(name, fn, obs=obs, rng_key=rng_key,
                                sample_shape=sample_shape, infer=infer,
                                obs_mask=obs_mask)
        return self.y

    def _build_conditional(self, x: ArrayLike,
                           noise: Optional[Union[float, Callable]]=None,
                           gp=None, diag: bool=False):
        """Make a prediction for the loc and cov of f(x) given y,
        
        loc = mp + kxp·(kxx + nx)^{-1}·(y - mx),
        cov = kpp + np - kxp·(kxx + nx)^{-1}·kxp^T,
        var = sqrt(diag(cov)),
        
        where mx = mean(x), mp = mean(x_pred), kxx = kernel(x, x),
        kxp = kernel(x, x_pred), kpp = kernel(x_pred, x_pred),
        nx = noise(x), and np = noise(x_pred).

        Args:
            x: The x values for which to make predictions.
            noise: If True, add self.noise to the
                prediction. If callable, must be a function of (x_pred, x_pred).
                Otherwise, pass the scale parameter for WhiteNoise.
                Default is None (no noise).
            gp (GP, optional): The GP from which to make predictions. 
                For example, used in GP addition. Default is self. 
            diag: If True, returns the variance. Default is False.
        
        Returns:
            loc: The mean of the prediction.
            cov or var: The covariance or variance of the prediction.
        """
        if gp is None:
            # Predict given a different GP (e.g. additive)
            gp = self
        
        if gp.x is None:
            raise ValueError("GP must be sampled to make predictions," + 
                             " consider the `gp` keyword argument")

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

    def conditional(self, x: ArrayLike,
                    noise: Optional[Union[float, Callable]]=None,
                    diag: bool=False, gp=None, **kwargs) -> dist.Distribution:
        """Make a conditional distribution for y' = f(x') + noise',
        
        y' | y ~ N(
        mp + kxp·(kxx + nx)^{-1}·(y - mx),
        kpp + np - kxp·(kxx + nx)^{-1}·kxp^T
        ),
        
        where mx = mean(x), mp = mean(x_pred), kxx = kernel(x, x),
        kxp = kernel(x, x_pred), kpp = kernel(x_pred, x_pred),
        nx = noise(x), and np = noise(x_pred).
        
        Args:
            x: The x values for which to make predictions.
            noise: If True, add self.noise to 
                the prediction. If callable, must be a function of 
                (x_pred, x_pred). Otherwise, pass the scale parameter for
                WhiteNoise. Default is None (no noise).
            gp (GP, optional): The GP from which to make predictions. Default is
                self. E.g. the total GP in which self is a term.
            diag: If True, diagonalises the variance. Default is False.
            **kwargs: Keyword arguments to pass to :class:`dist.Normal` or
                :class:`dist.MultivariateNormal`.
        """
        args = self._build_conditional(x, noise=noise, gp=gp, diag=diag)
        if diag:
            return dist.Normal(*args, **kwargs)

        return dist.MultivariateNormal(*args, **kwargs)
    
    def predict(self, name: str, x: ArrayLike,
                noise: Optional[Union[float, Callable]]=None, gp=None,
                diag: bool=False, rng_key: Optional[jnp.ndarray]=None, 
                sample_shape: tuple=(), infer: Optional[dict]=None,
                **kwargs) -> jnp.ndarray:
        """Sample from the GP conditional distribution.

        Args:
            name: [description]
            x: [description]
            noise: [description]. Defaults to None.
            gp: [description]. Defaults to None.
            diag: [description]. Defaults to False.
            rng_key: [description]. Defaults to None.
            sample_shape: [description]. Defaults to ().
            infer: [description]. Defaults to None.
            **kwargs: Keyword arguments to pass to :meth:`conditional`.

        Returns:
            [description]
        """
        fn = self.conditional(x, noise=noise, gp=gp, diag=diag, **kwargs)
        return numpyro.sample(name, fn, rng_key=rng_key, sample_shape=sample_shape,
                              infer=infer)
