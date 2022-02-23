"""Messengers module.
"""
import numpy as np

from numpyro.primitives import Messenger, apply_stack
from typing import Callable, Optional, Dict
from numpy.typing import ArrayLike


class dimension(Messenger):
    """Context manager for a model dimension.

    Args:
        name (str): Name of the dimension.
        size (int): Size of the dimension.
        coords (:term:`array_like`, optional): Coordinates for points in the
            dimension. Defaults to :code:`np.arange(size)`.
        dim (int, optional): Where to place the dimension. Defaults to
            :code:`-1` which corresponds to the rightmost dimension. Must be
            negative.
    """

    def __init__(
        self,
        name: str,
        size: int,
        coords: Optional[ArrayLike] = None,
        dim: Optional[ArrayLike] = None,
    ):
        self.name: str = name
        self.size: int = size
        self.dim: int = -1 if dim is None else dim
        """int: Location in which to insert the dimension."""

        assert self.dim < 0
        if coords is None:
            coords = np.arange(self.size)
        self.coords: np.ndarray = np.array(coords)
        """numpy.ndarray: Coordinates for the dimension."""

        msg = self._get_message()
        apply_stack(msg)
        super().__init__()

    def _get_message(self) -> dict:
        msg = {
            "name": self.name,
            "type": "dimension",
            "dim": self.dim,
            "value": self.coords,
        }
        return msg

    def __enter__(self) -> dict:
        super().__enter__()
        return self._get_message()

    def process_message(self, msg: dict):
        """Process the message.

        Args:
            msg (dict): Message.

        Raises:
            ValueError: If the corresponding dimension of the site is of
                incorrect size.
        """
        if msg["type"] not in ("param", "sample", "deterministic"):
            # We don't add dimensions to dimensions
            return

        if msg["value"] is None:
            shape = ()
            if "fn" in msg.keys():
                sample_shape = msg["kwargs"].get("sample_shape", ())
                shape = msg["fn"].shape(sample_shape)
        else:
            shape = msg["value"].shape

        if "dims" not in msg.keys():
            dims = [f"{msg['name']}_dim_{i}" for i in range(len(shape))]
            msg["dims"] = dims

        if "dim_stack" not in msg.keys():
            msg["dim_stack"] = []

        dim = self.dim
        while dim in msg["dim_stack"]:
            dim -= 1

        msg["dim_stack"].append(dim)
        msg["dims"][dim] = self.name

        if shape[dim] != self.size:
            raise ValueError(
                f"Dimension {dim} of site '{msg['name']}' should have length {self.size}"
            )
