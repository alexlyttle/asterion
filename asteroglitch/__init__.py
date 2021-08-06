"""[summary]

Notes:
    [notes]
"""
import numpyro

from ._version import __version__
from .data import Data
from .inference import Inference
from .model import Model, GlitchModel

numpyro.enable_x64()
numpyro.set_host_device_count(10)
