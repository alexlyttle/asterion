"""[summary]
"""
import copy
import arviz as az
import numpy as np
from typing import List, Dict, Optional


__all__ = [
    "Data",
]


def summary(data, group="posterior"):
    stat_funcs = {
        "16th": lambda x: np.quantile(x, .16),
        "50th": np.median,
        "84th": lambda x: np.quantile(x, .84),
    }
    circ_var_names = [v for v in data.constant_data["circ_var_names"].values if v in data[group].keys()]
    return az.summary(data, group=group, fmt="xarray", round_to="none", 
        stat_funcs=stat_funcs, circ_var_names=circ_var_names)


def save(data, filename):
    return data.to_netcdf(filename)


def load(filename):
    return az.from_netcdf(filename)


class Data:
    """[summary]

    Args:
        observed: [description]. Defaults to None.
        constant: [description]. Defaults to None.
        coords: [description]. Defaults to None.
        dims: [description]. Defaults to None.
    """    
    def __init__(self, observed: Optional[Dict[str, np.ndarray]]=None, 
                 constant: Optional[Dict[str, np.ndarray]]=None,
                 coords: Optional[Dict[str, np.ndarray]]=None,
                 dims: Optional[Dict[str, List[str]]]=None):
                       
        self._observed_data = observed
        self._constant_data = constant
        self._coords = coords
        self._dims = dims

        self._prior_predictive = None
        self._posterior = None
        self._sample_stats = None
        self._posterior_predictive = None
        self._predictions = None
        self._pred_coords = None
        self._pred_dims = None
    
    @property
    def observed_data(self) -> Optional[Dict[str, np.ndarray]]:
        """[description]
        """
        return self._observed_data
    
    @property
    def constant_data(self) -> Optional[Dict[str, np.ndarray]]:
        """[description]
        """
        return self._constant_data
    
    @property
    def coords(self) -> Optional[Dict[str, np.ndarray]]:
        """[description]
        """
        return self._coords
    
    @property
    def dims(self) -> Optional[Dict[str, List[str]]]:
        """[description]
        """
        return self._dims
    
    def _to_dict_of_array(self, value):
        new_value = {}
        for k, v in value.items():
            new_value[k] = np.array(v)
        return new_value

    def _none_or_dict_of_array(self, value):
        if value is None:
            return
        else:
            return self._to_dict_of_array(value)

    @property
    def prior_predictive(self) -> Optional[Dict[str, np.ndarray]]:
        """[description]
        """
        return self._prior_predictive
    
    @prior_predictive.setter
    def prior_predictive(self, value):
        self._prior_predictive = self._none_or_dict_of_array(value)

    @property
    def posterior(self) -> Optional[Dict[str, np.ndarray]]:
        """[description]
        """
        return self._posterior
    
    @posterior.setter
    def posterior(self, value):
        self._posterior = self._none_or_dict_of_array(value)

    @property
    def sample_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """[description]
        """
        return self._sample_stats
    
    @sample_stats.setter
    def sample_stats(self, value):
        self._sample_stats = self._none_or_dict_of_array(value)

    @property
    def posterior_predictive(self) -> Optional[Dict[str, np.ndarray]]:
        """[description]
        """
        return self._posterior_predictive
    
    @posterior_predictive.setter
    def posterior_predictive(self, value):
        self._posterior_predictive = self._none_or_dict_of_array(value)

    @property
    def predictions(self) -> dict:
        return self._predictions
        
    @predictions.setter
    def predictions(self, value):
        self._predictions = self._none_or_dict_of_array(value)

    @property
    def pred_coords(self) -> dict:
        return self._pred_coords
    
    @pred_coords.setter
    def pred_coords(self, value):
        self._pred_coords = value

    @property
    def pred_dims(self) -> dict:
        return self._pred_dims

    @pred_dims.setter
    def pred_dims(self, value):
        self._pred_dims = value

    def to_arviz(self) -> az.InferenceData:
        """[summary]

        Returns:
            [description]
        """               
        data = az.from_dict(
            observed_data=self.observed_data,
            constant_data=self.constant_data,
            prior_predictive=self.prior_predictive,
            posterior=self.posterior,
            sample_stats=self.sample_stats,
            posterior_predictive=self.posterior_predictive,
            predictions=self.predictions,
            coords=self.coords,
            dims=self.dims,
            pred_coords=self.pred_coords,
            pred_dims=self.pred_dims,
        )
        return data
