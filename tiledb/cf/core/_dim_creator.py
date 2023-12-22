from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from typing_extensions import Protocol

import tiledb

from .._utils import DType
from ._shared_dim import SharedDim
from .source import FieldData, create_field_data


class DimRegistry(Protocol):
    def set_writer_data(
        self, writer_index: Optional[int], dim_name: str, data: FieldData
    ):
        """Set the data to the requested frgament writer."""


class DimCreator:
    """Creator for a TileDB dimension using a SharedDim.

    Parameters
    ----------
    base
        The core shared dimension describing the dimension.
    tile
        The tile size for the dimension.
    filters
        Specifies compression filters for the dimension.
    registry
        An optional registry for the dimension registry.

    Attributes
    ----------
    tile
        The tile size for the dimension.
    filters
        Specifies compression filters for the dimension.
    """

    def __init__(
        self,
        base: SharedDim,
        *,
        tile: Optional[Union[int, float]] = None,
        filters: Optional[tiledb.FilterList] = None,
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

    def set_writer_data(
        self,
        dim_data: Union[np.ndarray, FieldData],
        *,
        writer_index: Optional[int] = None,
    ):
        """Set dimension data on a fragment writer

        Parameters
        ----------
        dim_data
            The dimension data to set.
        writer_index
            The index of the fragment writer to set the data on.
        """
        if self._registry is None:
            raise ValueError("Dimension creator is not registered to an array.")
        data = create_field_data(dim_data, self.dtype)
        self._registry.set_writer_data(writer_index, self.name, data)

    def to_tiledb(self, ctx: Optional[tiledb.Ctx] = None) -> tiledb.Dim:
        """Returns a `tiledb.Dim` using the creator properties.

        Parameters
        ----------
        ctx
            If not ``None``, TileDB context wrapper for a TileDB storage manager.

        Returns
        -------
        tiledb.Dim
            A tiledb dimension with the set properties.
        """
        return tiledb.Dim(
            name=self.name,
            domain=self.domain,
            tile=self.tile,
            filters=self.filters,
            dtype=self.dtype,
            ctx=ctx,
        )
