"""Annotations for use in type hinting.
"""
from __future__ import annotations

from typing import Union
from collections.abc import Iterable
from numpyro.distributions import Distribution

__all__ = [
    "DistLike",
]

# T = TypeVar('T')  # type
# """A generic type variable
# """

# S = TypeVar('S')  # sequence
# Array = Union[S, np.ndarray]


# class Array1D(Generic[T]):
#     """A generic type for a 1-dimensional array-like object
    
#     alias of Union\[Sequence\[:obj:`T`\], :obj:`numpy.ndarray`\]
#     """
#     def __getitem__(self, T):
#         return Array[Sequence[T]]


# class Array2D(Generic[T]):
#     """A generic type for a 2-dimensional array-like object
    
#     alias of Union\[Sequence\[Sequence\[:obj:`T`\]\], :obj:`numpy.ndarray`\]
#     """
#     def __getitem__(self, T):
#         return Array[Sequence[Sequence[T]]]


# class Array3D(Generic[T]):
#     """A generic type for a 3-dimensional array-like object
    
#     alias of Union\[Sequence\[Sequence\[Sequence\[:obj:`T`\]\]\],
#     :obj:`numpy.ndarray`\]
#     """
#     def __getitem__(self, T):
#         return Array[Sequence[Sequence[Sequence[T]]]]


DistLike = Union[Iterable, Distribution]  
"""A generic type for a distribution object which is accepted as the argument
of :func:`asterion.models.distribution`.
"""

# alias of Union\[:obj:`Iterable`, :obj:`Distribution`\]
