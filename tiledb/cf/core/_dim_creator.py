from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from typing_extensions import Protocol

import tiledb

from .._utils import DType
from ._shared_dim import SharedDim
from .source import FieldData, NumpyData


class DimRegistry(Protocol):
    def set_fragment_data(self, fragment_index: int, attr_name: str, data: FieldData):
        ...


class DimCreator:
    """Creator for a TileDB dimension using a SharedDim.

    Attributes:
        tile: The tile size for the dimension.
        filters: Specifies compression filters for the dimension.
    """

    def __init__(
        self,
        base: SharedDim,
        *,
        tile: Optional[Union[int, float]] = None,
        filters: Optional[Union[tiledb.FilterList]] = None,
        registry: Optional[DimRegistry] = None,
    ):
        self._base = base
        self.tile = tile
        self.filters = filters
        self._registry = registry

    def __repr__(self):
        filters_str = ""
        if self.filters:
            filters_str = ", filters=FilterList(["
            for dim_filter in self.filters:
                filters_str += repr(dim_filter) + ", "
            filters_str += "])"
        return f"DimCreator({repr(self._base)}, tile={self.tile}{filters_str})"

    @property
    def base(self) -> SharedDim:
        """Shared definition for the dimensions name, domain, and dtype."""
        return self._base

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the values and domain of the dimension."""
        return self._base.dtype

    @property
    def domain(self) -> Optional[Tuple[Optional[DType], Optional[DType]]]:
        """The (inclusive) interval on which the dimension is valid."""
        return self._base.domain

    def html_summary(self) -> str:
        """Returns a string HTML summary of the :class:`DimCreator`."""
        filters_str = ""
        if self.filters:
            filters_str = ", filters=FilterList(["
            for dim_filter in self.filters:
                filters_str += repr(dim_filter) + ", "
            filters_str += "])"
        return (
            f"{self._base.html_input_summary()} &rarr; tiledb.Dim("
            f"{self._base.html_output_summary()}, tile={self.tile}{filters_str})"
        )

    @property
    def name(self) -> str:
        """Name of the dimension."""
        return self._base.name

    def set_fragment_data(
        self, fragment_index: int, attr_data: Union[np.ndarray, FieldData]
    ):
        if self._registry is None:
            raise ValueError("Dimension creator is not registered to an array.")
        if isinstance(attr_data, np.ndarray):
            data = NumpyData(attr_data.astype(self.dtype))
        elif isinstance(attr_data, int):
            data = NumpyData(np.ndarray(attr_data, dtype=self.dtype))
        else:
            data = attr_data
        if data.dtype != self.dtype:
            # Relax?
            raise ValueError(
                f"Cannot set data with dtype='{attr_data.dtype}' to an attribute witha"
                f"dtype='{self.dtype}'."
            )
        # TODO: Check variable length?
        self._registry.set_fragment_data(fragment_index, self.name, data)

    def to_tiledb(self, ctx: Optional[tiledb.Ctx] = None) -> tiledb.Domain:
        """Returns a :class:`tiledb.Dim` using the current properties.

        Parameters:
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.

        Returns:
            A tiledb dimension with the set properties.
        """
        if self.domain is None:
            raise ValueError(
                f"Cannot create a TileDB dimension for dimension '{self.name}'. No "
                f"domain is set."
            )
        return tiledb.Dim(
            name=self.name,
            domain=self.domain,
            tile=self.tile,
            filters=self.filters,
            dtype=self.dtype,
            ctx=ctx,
        )
