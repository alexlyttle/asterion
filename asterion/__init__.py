"""Fitting acoustic glitches to 

Warning:
    This module is a work-in-progress, use with caution.
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

import numpyro
numpyro.enable_x64()

from ._version import __version__
from .inference import Inference
from .models import GlitchModel
from .plotting import plot_corner, plot_glitch, get_labeller
from .results import get_dims, get_summary, get_table, get_var_names
