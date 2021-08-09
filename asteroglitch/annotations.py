"""Annotations for use in type hinting.
"""
from __future__ import annotations

import numpy as np
from typing import Union, TypeVar, Sequence, Generic


__all__ = [
    "T",
    "Array1D",
    "Array2D",
    "Array3D",
]


T = TypeVar('T')  # type
""".. _type-variable:

A generic type variable
"""

S = TypeVar('S')  # sequence
Array = Union[S, np.ndarray]


class Array1D(Generic[T]):
    """A generic type for a 1-dimensional array-like object
    
    alias of Union\[Sequence\[`T <#asteroglitch.annotations.T>`_\],
    :ref:`numpy.ndarray<numpy:arrays.ndarray>`\]
    """
    def __getitem__(self, T):
        return Array[Sequence[T]]


class Array2D(Generic[T]):
    """A generic type for a 2-dimensional array-like object
    
    alias of Union\[Sequence\[Sequence\[`T <#asteroglitch.annotations.T>`_\]\],
    :ref:`numpy.ndarray<numpy:arrays.ndarray>`\]
    """
    def __getitem__(self, T):
        return Array[Sequence[Sequence[T]]]


class Array3D(Generic[T]):
    """A generic type for a 3-dimensional array-like object
    
    alias of Union\[Sequence\[Sequence\[
    Sequence\[`T <#asteroglitch.annotations.T>`_\]\]\],
    :ref:`numpy.ndarray<numpy:arrays.ndarray>`\]
    """
    def __getitem__(self, T):
        return Array[Sequence[Sequence[Sequence[T]]]]
