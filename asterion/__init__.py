"""[summary]

Warning:
    This module is a work-in-progress, use with caution.
"""
import numpyro

from ._version import __version__

numpyro.enable_x64()
numpyro.set_host_device_count(10)
