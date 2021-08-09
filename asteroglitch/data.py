"""[summary]
"""
import arviz as az
import numpy as np
from typing import List, Dict, Optional


__all__ = [
    "Data",
]


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
                       
        self._observed = observed
        self._constant = constant
        self._coords = coords
        self._dims = dims

        self.prior_predictive = None
        self.posterior = None
        self.sample_stats = None
        self.posterior_predictive = None
    
    @property
    def observed(self) -> Optional[Dict[str, np.ndarray]]:
        """[description]
        """
        return self._observed
    
    @property
    def constant(self) -> Optional[Dict[str, np.ndarray]]:
        """[description]
        """
        return self._constant
    
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

    def to_arviz(self) -> az.InferenceData:
        """[summary]

        Returns:
            [description]
        """               
        data = az.from_dict(
            observed_data=self.observed,
            constant_data=self.constant,
            prior_predictive=self.prior_predictive,
            posterior=self.posterior,
            sample_stats=self.sample_stats,
            posterior_predictive=self.posterior_predictive,
            coords=self.coords,
            dims=self.dims,
        )
        return data
