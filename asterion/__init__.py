"""Fitting acoustic glitches to the mode frequencies of solar-like oscillators.

Warning:
    This module is a work-in-progress, use with caution.
"""
from .version import __version__
from .models import GlitchModel, GlitchModelComparison
from .inference import Inference
from .plotting import (
    style,
    plot_corner,
    plot_echelle,
    plot_glitch,
    get_labeller,
)
from .results import get_dims, get_summary, get_table, get_var_names
