import os
import numpyro.distributions as dist

from typing import Optional, Iterable
from .typing import DistLike

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

def distribution(
    value: DistLike,
    default_dist: Optional[dist.Distribution]=None,
) -> dist.Distribution:
    """Return a numpyro distribition.

    If value is not a distribution, this returns the default distribution,
    unpacking value as its arguments.

    Args:
        value (:term:`dist_like`): Iterable of args to pass to default_dist, 
            or a Distribution.
        default_dist (numpyro.distributions.distribution.Distribution, \
optional): Default distribution. Defaults to dist.Normal if None.

    Returns:
        numpyro.distributions.distribution.Distribution: [description]
    """
    if default_dist is None:
        default_dist = dist.Normal
    if not isinstance(value, dist.Distribution):
        if not isinstance(value, Iterable):
            value = (value,)
        value = default_dist(*value)
    return value

