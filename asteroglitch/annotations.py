"""[summary]

Notes:
    [notes]
"""
import numpy as np
from typing import Union, TypeVar, Sequence


__all__ = [
    "Array1D",
    "Array2D",
    "Array3D",
]


_T = TypeVar('_T')  # type
_S = TypeVar('_S')  # sequence
_Array = Union[_S, np.ndarray]

Array1D = _Array[Sequence[_T]]
"""[description]"""

Array2D = _Array[Sequence[Sequence[_T]]]
"""[description]"""

Array3D = _Array[Sequence[Sequence[Sequence[_T]]]]
"""[description]"""
