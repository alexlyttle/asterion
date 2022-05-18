"""The inference module contains a handy :class:`Inference` class which wraps
:mod:`numpyro` inference methods to facilitate the Bayesian workflow.
"""
from __future__ import annotations

import arviz as az
import astropy.units as u
import numpy as np
import numpyro, time, xarray, warnings

from collections import OrderedDict
from jax import random
from numpyro import handlers as hdl
from numpyro.infer import MCMC, NUTS, init_to_median, Predictive, SVI
from numpyro.distributions import constraints
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.infer.reparam import Reparam, CircularReparam
from typing import Optional, Union, List

from .models import Model

__all__ = [
    "Inference",
]


class Inference:
    """Perform inference on a given model.

    Args:
        model (Model): Model which predicts the asteroseismic mode frequencies.
        nu (:term:`array_like`): Asteroseismic mode frequencies.
        nu_err (:term:`array_like`, optional): Observational uncertainty on
            the asteroseismic mode frequencies.
        seed (int): Seed for pseudo-random number generation.

    Example:
        .. code-block:: python

            from asterion import Model, Inference

            n = [9, 10, 11, 12, 13]
            nu = [111.1, 122.2, 133.3, 144.4, 155.5]
            nu_err = 0.01
            model = Model(...)  # Construct the model here
            infer = Inference(model, n=n, nu=nu, nu_err=nu_err, seed=42)
            # Sample from prior predictive
            infer.prior_predictive(num_samples=2000)
            # Sample from posterior
            infer.sample()
            # Sample from posterior predictive
            infer.posterior_predictive()
            # Get data from inference
            data = infer.get_data()
            # Save data
            data.to_netcdf('inference_data.nc')

    Attributes:
        model (Model): Model with which to perform inference.
        nu (numpy.ndarray): Observed mode frequencies.
        nu_err (numpy.ndarray, optional): Uncertainty on observed mode
            frequencies.
        samples (dict, optional): Posterior samples.
        weighted_samples (dict, optional): Posterior weighted samples.
        sample_stats (dict, optional): Posterior sample statistics.
        prior_predictive_samples (dict, optional): Prior predictive samples.
        predictive_samples (dict, optional): Posterior predictive samples.
        sample_method (str, optional): Posterior sampling method.
    """

    def __init__(self, model: Model, *, n, nu, nu_err=None, seed: int = 0):
        self._rng_key = random.PRNGKey(seed)
        self.model: Model = model

        self.n = np.asarray(n)
        self.n_pred = np.linspace(self.n[0], self.n[-1], 250)
        self.nu = np.asarray(nu)
        self.nu_err = None if nu_err is None else np.asarray(nu_err)

        self.samples: Optional[dict] = None
        self.weighted_samples: Optional[dict] = None
        self.sample_stats: Optional[dict] = None
        self.prior_predictive_samples: Optional[dict] = None
        self.predictive_samples: Optional[dict] = None
        # self.mcmc: Optional[MCMC] = None

        self._map_loss: Optional[dict] = None
        self._map_guide: Optional[dict] = None
        self._map_params: Optional[dict] = None
        # self.ns: Optional[NestedSampler] = None

        self.sample_metadata = {}

    def _mcmc_support_warnings(self):
        # Catch circular-like parameters
        trace = self.get_trace()
        for _, value in trace.items():
            if value["type"] == "sample":
                if (
                    value["fn"].support == constraints.circular
                    and value["fn"].support is not constraints.circular
                ):
                    # Catches parameters with circular-like support
                    warnings.warn(
                        f"Parameter '{value['name']}' has "
                        + "circular-like support but the distribution is "
                        + "not circular. Consider changing its distribution "
                        + "to numpyro.distributions.VonMises for better "
                        + "performance during MCMC."
                    )

    def _get_dims(self):
        coords = {}
        dims = {}
        for _, v in self.get_trace(pred=True).items():
            name = v["name"]
            if v["type"] == "dimension":
                coords[name] = v["value"]
            elif "dims" in v.keys():
                dims[name] = v["dims"]
        return dims, coords

    def get_trace(self, pred=False) -> OrderedDict:
        """[summary]

        Args:
            pred (bool): Whether to trace the predictive model or
                not. Default is False.

        Returns:
            OrderedDict: Model trace.
        """
        rng_key, self._rng_key = random.split(self._rng_key)
        n_pred = self.n_pred if pred else None
        model = hdl.trace(hdl.seed(self.model, rng_key))
        trace = model.get_trace(
            self.n, nu=self.nu, nu_err=self.nu_err, n_pred=n_pred
        )
        return trace

    def get_circ_var_names(self) -> List[str]:
        """[summary]

        Returns:
            list: Circular variable names in the model.
        """
        var_names = []
        trace = self.get_trace()
        for _, value in trace.items():
            if value["type"] == "sample":
                if value["fn"].support is constraints.circular:
                    var_names.append(value["name"])
        return var_names

    def _auto_reparam(self) -> numpyro.handlers.reparam:
        """Automatically reparameterise circular parameters."""
        return hdl.reparam(
            config={k: CircularReparam() for k in self.get_circ_var_names()}
        )

    def _init_handlers(
        self, handlers: list, reparam: Union[str, hdl.reparam] = "auto"
    ) -> list:
        """Initialise handlers with/without reparameterisation."""
        # handlers = handlers.copy()
        if handlers is None:
            handlers = []
        if reparam == "auto":
            handlers.append(self._auto_reparam())
        elif reparam == "none" or reparam is None:
            pass
        else:
            handlers.append(reparam)
        return handlers

    def _expand_batch_dims(self, samples):
        """Expand the batch dimensions of samples (add leading dimension)."""
        new_samples = {}
        for k, v in samples.items():
            # Hack because InferenceData assumes a chains dimension.
            new_samples[k] = v[None, ...]  # Add a leading dimension to samples
        return new_samples

    def init_mcmc(
        self,
        model,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 1,
        sampler="NUTS",
        sampler_kwargs={},
        **kwargs,
    ) -> MCMC:
        """Initialises the MCMC sampler.

        Args:
            model (callable): [desc]
            num_warmup (int): [description]. Defaults to 1000.
            num_samples (int): [description]. Defaults to 1000.
            num_chains (int): [description]. Defaults to 1.
            sampler (str, or numpyro.infer.mcmc.MCMCKernel): Choose one of
                ['NUTS'], or pass a numpyro mcmc kernel.
            sampler_kwargs (dict): Keyword arguments to pass to the chosen
                sampler.
            **kwargs: Keyword arguments to pass to mcmc instance.

        """
        self._mcmc_support_warnings()

        if isinstance(sampler, str):
            sampler = sampler.lower()
            if sampler != "nuts":
                raise ValueError(f"Sampler '{sampler}' not supported.")
            target_accept_prob = sampler_kwargs.pop("target_accept_prob", 0.98)
            init_strategy = sampler_kwargs.pop(
                "init_strategy",
                lambda site=None: init_to_median(site=site, num_samples=100),
            )
            step_size = sampler_kwargs.pop("step_size", 0.1)
            sampler = NUTS(
                model,
                target_accept_prob=target_accept_prob,
                init_strategy=init_strategy,
                step_size=step_size,
                **sampler_kwargs,
            )

        # if num_chains > 1:
        # self.batch_ndims = 2  # I.e. two dims for chains then samples

        return MCMC(
            sampler,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            **kwargs,
        )

    # def _update_args_kwargs(self, model_args: tuple, model_kwargs: dict):
    #     self._model_args = model_args
    #     self._model_kwargs.update(model_kwargs)

    def run_mcmc(
        self,
        model,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 1,
        sampler="NUTS",
        sampler_kwargs={},
        extra_fields: tuple = (),
        init_params: dict = None,
        **kwargs,
    ) -> tuple:
        """Runs MCMC for a given set of model arguments.

        Args:
            model (callable): [desc]
            num_warmup (int): [description]. Defaults to 1000.
            num_samples (int): [description]. Defaults to 1000.
            num_chains (int): [description]. Defaults to 1.
            sampler (str): Choose one of ['NUTS']
            sampler_kwargs (dict): Keyword arguments to pass to the chosen
                sampler.
            extra_fields (tuple): Extra fields to report in sample_stats.
                Defaults to ().
            init_params (dict): Initial parameter values prior to sampling.
                Defaults to None.
            **kwargs: Keyword arguments to pass to mcmc instance.

        Returns:
            tuple: [description]
        """
        mcmc = self.init_mcmc(
            model,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            sampler=sampler,
            sampler_kwargs=sampler_kwargs,
            **kwargs,
        )

        rng_key, self._rng_key = random.split(self._rng_key)
        mcmc.run(
            rng_key,
            self.n,
            nu=self.nu,
            nu_err=self.nu_err,
            extra_fields=extra_fields,
            init_params=init_params,
        )

        self.samples = mcmc.get_samples(group_by_chain=True)
        self.sample_stats = mcmc.get_extra_fields(group_by_chain=True)
        self.sample_metadata = {
            "method": "MCMC",
            "sampler": sampler
            if isinstance(sampler, str)
            else sampler.__name__,
        }

    def init_nested(
        self,
        model: Model,
        num_live_points: int = 50,
        max_samples: int = 50000,
        sampler: str = "multi_ellipsoid",
        **kwargs,
    ) -> NestedSampler:
        """[summary]

        Args:
            model (Model): [description]
            num_live_points (int): [description]. Defaults to 50.
            max_samples (int): [description]. Defaults to 50000.
            sampler (str): [description]. Defaults to "multi_ellipsoid".
            **kwargs: Keyword arguments to pass to nested sampler instance.

        Returns:
            numpyro.contrib.nested_sampling.NestedSampler: [description]
        """
        depth = kwargs.pop("depth", 5)
        return NestedSampler(
            model,
            num_live_points=num_live_points,
            max_samples=max_samples,
            sampler_name=sampler,
            depth=depth,
            **kwargs,
        )

    def run_nested(
        self,
        model: Model,
        num_live_points: int = 50,
        num_samples: int = 1000,
        max_samples: int = 50000,
        sampler: str = "multi_ellipsoid",
        **kwargs,
    ):
        """[summary]

        Args:
            model (Model): [description]
            num_live_points (int): [description]. Defaults to 100.
            num_samples (int): [description]. Defaults to 1000.
            max_samples (int): [description]. Defaults to 100000.
            sampler (str): [description]. Defaults to "multi_ellipsoid".
            **kwargs: Keyword arguments to pass to nested sampler instance.
        """
        nested_sampler = self.init_nested(
            model,
            num_live_points=num_live_points,
            max_samples=max_samples,
            sampler=sampler,
            **kwargs,
        )

        key1, key2, self._rng_key = random.split(self._rng_key, 3)
        print(
            f"Running nested sampling using the '{sampler}' sampler "
            + f"with {num_live_points} live points "
            + f"and {max_samples} maximum samples..."
        )
        start_time = time.time()
        nested_sampler.run(key1, self.n, nu=self.nu, nu_err=self.nu_err)
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.1f} seconds.")

        samples = nested_sampler.get_samples(key2, num_samples)
        weighted_samples = nested_sampler.get_weighted_samples()[0]

        num_weighted_samples = nested_sampler._results.num_samples
        logX = nested_sampler._results.log_X[:num_weighted_samples]
        logL = nested_sampler._results.log_L_samples[:num_weighted_samples]
        logp = nested_sampler._results.log_p[:num_weighted_samples]
        eff = nested_sampler._results.sampler_efficiency[:num_weighted_samples]

        self.samples = self._expand_batch_dims(samples)
        self.weighted_samples = self._expand_batch_dims(weighted_samples)
        self.sample_metadata = {
            "method": "nested",
            "sampler": sampler,
            "num_likelihood_evals": int(
                nested_sampler._results.num_likelihood_evaluations
            ),
            "num_weighted_samples": int(num_weighted_samples),
            "logZ": float(nested_sampler._results.logZ),  # evidence
            "logZ_err": float(nested_sampler._results.logZerr),
            "ESS": float(nested_sampler._results.ESS),
        }
        self.sample_stats = {
            "logX": logX,
            "logL": logL,
            "lp": logp,  # log joint posterior (i.e. sample weights)
            "sampler_efficiency": eff,
        }

    def _add_handlers_to_model(
        self,
        handlers: Optional[list] = None,
        reparam: Union[str, Reparam] = "auto",
    ) -> Model:
        """[summary]

        Args:
            handlers (list, optional): [description]. Defaults to
                None.
            reparam (str, or numpyro.infer.reparam.Reparam):
                [description]. Defaults to 'auto'.

        Returns:
            Model: [description]
        """
        handlers = self._init_handlers(handlers, reparam=reparam)
        model = self.model
        for h in handlers:
            model = h(model)
        return model

    def sample(
        self,
        num_samples: int = 1000,
        method: str = "nested",
        handlers: Optional[list] = None,
        reparam="auto",
        **kwargs,
    ):
        """[summary]

        Args:
            num_samples (int): Number of samples after warmup.
            method (str): Sampling method, choose from
                ['mcmc', 'nested'].
            handlers (list, optional): Handlers to apply to the model during
                inference.
            reparam (str, or numpyro.infer.reparam.Reparam): Default is 'auto'
                will automatically reparameterise the model to improve sampling
                during MCMC.
            **kwargs: Keyword arguments to pass to the sampling method.
        """
        # Add handlers to model
        model = self._add_handlers_to_model(handlers=handlers, reparam=reparam)

        if method == "mcmc":
            self.run_mcmc(
                model,
                num_samples=num_samples,
                **kwargs,
            )
        elif method == "nested":
            self.run_nested(model, num_samples=num_samples, **kwargs)
        else:
            raise ValueError(
                f"Method '{method}' not one of ['mcmc', 'nested']."
            )

        # TODO: diagnostics and warnings

    def find_map(
        self,
        num_steps: int = 10000,
        handlers: Optional[list] = None,
        reparam: Union[str, hdl.reparam] = "auto",
        svi_kwargs: dict = {},
    ):
        """EXPERIMENTAL: find MAP.

        Args:
            num_steps (int): [description]. Defaults to 10000.
            handlers (list, optional): [description]. Defaults to None.
            reparam (str, or numpyro.handlers.reparam): [description]. Defaults to 'auto'.
            svi_kwargs (dict): [description]. Defaults to {}.
        """
        model = self._add_handlers_to_model(handlers=handlers, reparam=reparam)

        guide = numpyro.infer.autoguide.AutoDelta(model)

        optim = svi_kwargs.pop("optim", numpyro.optim.Minimize())
        loss = svi_kwargs.pop("loss", numpyro.infer.Trace_ELBO())
        map_svi = SVI(model, guide, optim, loss=loss, **svi_kwargs)

        rng_key, self._rng_key = random.split(self._rng_key)
        map_result = map_svi.run(
            rng_key, num_steps, self.n, nu=self.nu, nu_err=self.nu_err
        )

        self._map_loss = map_result.losses
        self._map_guide = map_svi.guide
        self._map_params = map_result.params

    def predictive(
        self, n, nu=None, nu_err=None, n_pred=None, **kwargs
    ) -> dict:
        """[summary]

        Args:
            model_args (tuple): Positional arguments to pass to the model
                callable.
            model_kwargs (dict): Keyword arguments to pass to the model
                callable.
            **kwargs: Kwargs to pass to Predictive.

        Returns:
            dict: [description]
        """
        posterior_samples = kwargs.pop("posterior_samples", None)
        num_samples = kwargs.pop("num_samples", None)
        batch_ndims = kwargs.pop("batch_ndims", 2)
        return_sites = kwargs.pop("return_sites", None)

        posterior = {} if posterior_samples is None else posterior_samples
        if return_sites is None:
            trace = self.get_trace(pred=True)
            return_sites = []
            for k, site in trace.items():
                # Only return non-observed sample sites not in samples and
                # all deterministic sites.
                if site["type"] == "sample":
                    if not site["is_observed"] and k not in posterior:
                        return_sites.append(k)
                elif site["type"] == "deterministic":
                    return_sites.append(k)

        predictive = Predictive(
            self.model,
            posterior_samples=posterior_samples,
            num_samples=num_samples,
            return_sites=return_sites,
            batch_ndims=batch_ndims,
            **kwargs,
        )

        if predictive.batch_ndims == 0:
            # Fix bug in Predictive for computing batch shape
            predictive._batch_shape = ()

        rng_key, self._rng_key = random.split(self._rng_key)
        samples = predictive(rng_key, n, nu=nu, nu_err=nu_err, n_pred=n_pred)
        # self._update_args_kwargs(model_args, model_kwargs)
        return samples

    def prior_predictive(self, num_samples: int = 1000, **kwargs):
        """[summary]

        Args:
            num_samples (int): Number of samples to take from the prior.
            **kwargs: Keyword arguments to pass to Predictive instance.
        """
        self.prior_predictive_samples = self.predictive(
            self.n, n_pred=self.n_pred, num_samples=num_samples, **kwargs
        )

    def posterior_predictive(self, **kwargs):
        """[summary]

        Args:
            **kwargs: Keyword arguments to pass to Predictive instance.
        """
        self.predictive_samples = self.predictive(
            self.n,
            nu=self.nu,
            nu_err=self.nu_err,
            n_pred=self.n_pred,
            posterior_samples=self.samples,
            **kwargs,
        )

    def map_predictive(self, **kwargs) -> xarray.Dataset:
        """EXPERIMENTAL: Get predictive from MAP.

        Args:
            model_args (tuple): [description]. Defaults to ().
            model_kwargs (dict): [description]. Defaults to {}.

        Returns:
            xarray.Dataset: [description]
        """
        batch_ndims = 0
        guide = self._map_guide
        params = self._map_params
        map_pred = self.predictive(
            self.n,
            n_pred=self.n_pred,
            guide=guide,
            params=params,
            num_samples=1,
            batch_ndims=batch_ndims,
            **kwargs,
        )
        dims, coords = self._get_dims()
        ds = {}
        for k, v in map_pred.items():
            ds[k] = xarray.DataArray(v, dims=dims.get(k))
        return xarray.Dataset(ds, coords=coords)

    def get_data(self) -> az.InferenceData:
        """Get inference data.

        Returns:
            arviz.InferenceData: Inference data.
        """
        dims, coords = self._get_dims()
        observed_data = {"nu": self.nu}
        constant_data = {
            "nu_err": np.zeros_like(self.nu)
            if self.nu_err is None
            else self.nu_err
        }
        data = az.from_dict(
            posterior=self.samples,
            prior_predictive=self.prior_predictive_samples,
            posterior_predictive=self.predictive_samples,
            sample_stats=self.sample_stats,
            observed_data=observed_data,
            constant_data=constant_data,
            dims=dims,
            coords=coords,
        )

        # Add sample metadata info
        if self.sample_stats is not None:
            data.sample_stats.attrs.update(self.sample_metadata)

        if self.sample_metadata.get("method", None) == "nested":
            # The weights are just the logP in sampler stats
            with warnings.catch_warnings():
                # Catch user warnings
                warnings.filterwarnings("ignore", category=UserWarning)
                data.add_groups(
                    {"weighted_posterior": self.weighted_samples},
                    coords=coords,
                    dims=dims,
                )

        circ_var_names = self.get_circ_var_names()
        # Add unit, symbol and circular attributes to groups
        for group in data.groups():
            for key in data[group].keys():
                sub_key = key
                if key.endswith("_pred"):
                    sub_key = key[:-5]
                else:
                    sub_key = key

                unit = self.model.units.get(sub_key, u.Unit())
                sym = self.model.symbols.get(sub_key, "")
                circ = 1 if sub_key in circ_var_names else 0

                data[group][key].attrs["unit"] = unit.to_string()
                data[group][key].attrs["symbol"] = sym
                data[group][key].attrs["is_circular"] = circ

        return data
