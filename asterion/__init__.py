"""[summary]

Warning:
    This module is a work-in-progress, use with caution.
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

import numpyro

numpyro.enable_x64()
numpyro.set_host_device_count(10)

from ._version import __version__
from .priors import TauPrior, AsyFunction, HeGlitchFunction, CZGlitchFunction
from .models import Model, GlitchModel
from .inference import Inference
from .plotting import plot_corner, plot_glitch, get_labeller
from .results import get_dims, get_summary, get_table, get_var_names
