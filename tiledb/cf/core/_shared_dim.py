from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from tiledb.datatypes import DataType
from typing_extensions import Self

from .._utils import DType
from .registry import RegisteredByNameMixin, Registry


class SharedDim(RegisteredByNameMixin):
    """Definition for the name, domain and data type of a collection of dimensions.

    Parameters
    ----------
    name
        The name of the shared dimension.
    domain
        The domain for the shared dimension.
    dtype
        The datatype of the shared dimension.
    registry
        If provided, a registry for the shared dimension.
    """

    def __init__(
        self,
        name: str,
        domain: Optional[Tuple[Optional[DType], Optional[DType]]],
        dtype: np.dtype,
        *,
        registry: Optional[Registry[Self]] = None,
    ):
        self._name = name
        self.domain = domain
        self.dtype = DataType.from_numpy(dtype).np_dtype
        super().__init__(name, registry)

    def __eq__(self, other):
        if not isinstance(other, self.__class__) or not isinstance(
            self, other.__class__
        ):
            return False
        return (
            self.name == other.name
            and self.domain == other.domain
            and self.dtype == other.dtype
        )

    def __repr__(self) -> str:
        return (
            f"SharedDim(name={self.name}, domain={self.domain}, dtype='{self.dtype!s}')"
        )

    def html_input_summary(self) -> str:
        """Returns a HTML string summarizing the input for the dimension."""
        return ""

    def html_output_summary(self) -> str:
        """Returns a string HTML summary of the :class:`SharedDim`."""
        return f"name={self.name}, domain={self.domain}, dtype='{self.dtype!s}'"

    @property
    def is_index_dim(self) -> bool:
        """Returns ``True`` if this is an `index dimension` and ``False`` otherwise.

        An index dimension is a dimension with an integer data type and whose domain
        starts at 0.
        """
        if self.domain:
            return np.issubdtype(self.dtype, np.integer) and self.domain[0] == 0
        return False
