import arviz as az

class Data:
    """[summary]

    Note
    ----
    [notes]

    Parameters
    ----------
    observed : dict, optional
        [description], by default None
    constant : dict, optional
        [description], by default None
    coords : dict, optional
        [description], by default None
    dims : dict, optional
        [description], by default None
    
    Attributes
    ----------
    constant : dict
    coords : dict
    dims : dict
    observed : dict
    posterior
    posterior_predictive
    prior_predictive
    sample_stats
    """   
    def __init__(self, observed: dict=None, constant: dict=None,
                 coords: dict=None, dims: dict=None): 
                       
        self.observed = observed
        self.coords = coords
        self.dims = dims
        self.constant = constant
        # self.prior = None
        # self.prior_sample_stats = None
        self.prior_predictive = None
        # self.posterior = None
        # self.posterior_sample_stats = None
        self.posterior = None
        self.sample_stats = None
        self.posterior_predictive = None
    
    def to_arviz(self) -> az.InferenceData:
        """[summary]

        Returns
        -------
        az.InferenceData
            [description]
        """               
        data = az.from_dict(
            observed_data=self.observed,
            constant_data=self.constant,
            # prior=self.prior,
            # sample_stats_prior=self.prior_sample_stats,
            prior_predictive=self.prior_predictive,
            posterior=self.posterior,
            # sample_stats=self.posterior_sample_stats,
            sample_stats=self.sample_stats,
            posterior_predictive=self.posterior_predictive,
            coords=self.coords,
            dims=self.dims,
        )
        return data
