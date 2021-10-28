"""[summary]
"""
import copy
import arviz as az
import numpy as np
from typing import List, Dict, Optional
from arviz.data.io_numpyro import NumPyroConverter
from arviz.data.base import dict_to_dataset
from arviz import utils

__all__ = [
    "ModifiedNumPyroConverter",
]

class ModifiedNumPyroConverter(NumPyroConverter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove observed values with size 0
        if self.observations is not None:
            obs = {}
            for key, value in self.observations.items():
                if value.size > 0:
                    obs[key] = value
            self.observations = obs

    def priors_to_xarray(self):
        """Convert prior samples (and if possible prior predictive too) to xarray."""
        if self.prior is None:
            return {"prior": None, "prior_predictive": None}
        if self.posterior is not None:
            # Modified to make sure samples keys are in prior keys
            prior_vars = [key for key in self._samples.keys() if key in self.prior.keys()]
            prior_predictive_vars = [key for key in self.prior.keys() if key not in prior_vars]
        else:
            prior_vars = self.prior.keys()
            prior_predictive_vars = None
        priors_dict = {}
        for group, var_names in zip(
            ("prior", "prior_predictive"), (prior_vars, prior_predictive_vars)
        ):
            priors_dict[group] = (
                None
                if var_names is None
                else dict_to_dataset(
                    {k: utils.expand_dims(self.prior[k]) for k in var_names},
                    library=self.numpyro,
                    coords=self.coords,
                    dims=self.dims,
                    index_origin=self.index_origin,
                )
            )
        return priors_dict
