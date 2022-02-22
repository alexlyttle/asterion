"""Annotations for use in type hinting.
"""
from __future__ import annotations

from typing import Union
from collections.abc import Iterable
from numpyro.distributions import Distribution

__all__ = [
    "DistLike",
]

DistLike = Union[Iterable, Distribution]  
"""A generic type for a distribution object which is accepted as the argument
of :func:`asterion.models.distribution`.
"""
