"""Fitting acoustic glitches to the mode frequencies of solar-like oscillators.

Warning:
    This module is a work-in-progress, use with caution.
"""
import numpyro

numpyro.enable_x64()

from .version import __version__
from .models import GlitchModel
from .inference import Inference
from .plotting import plot_corner, plot_echelle, plot_glitch, get_labeller
from .results import get_dims, get_summary, get_table, get_var_names
