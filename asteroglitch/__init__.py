import numpyro

from .version import __version__

numpyro.enable_x64()
numpyro.set_host_device_count(10)
