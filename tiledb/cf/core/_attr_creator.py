from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from typing_extensions import Self

import tiledb

from .._utils import DType
from ._fragment_writer import FragmentWriter
from .registry import RegisteredByNameMixin, Registry
from .source import BufferData


class AttrCreator(RegisteredByNameMixin):
    """Creator for a TileDB attribute.

    Attributes:
        dtype: Numpy dtype of the attribute.
        fill: Fill value for unset cells.
        var: Specifies if the attribute is variable length (automatic for
            byte/strings).
        nullable: Specifies if the attribute is nullable using validity tiles.
        filters: Specifies compression filters for the attribute.
    """

    def __init__(
        self,
        name: str,
        dtype: np.dtype,
        *,
        fill: Optional[DType] = None,
        var: bool = False,
        nullable: bool = False,
        filters: Optional[tiledb.FilterList] = None,
        registry: Optional[Registry[Self]] = None,
        fragment_writers: Optional[Sequence[FragmentWriter]] = None,
    ):
        self.dtype = np.dtype(dtype)
        self.fill = fill
        self.var = var
        self.nullable = nullable
        self.filters = filters
        self._fragment_writers = fragment_writers
        super().__init__(name, registry)

    def __repr__(self):
        filters_str = f", filters=FilterList({self.filters})" if self.filters else ""
        return (
            f"AttrCreator(name={self.name}, dtype='{self.dtype!s}', var={self.var}, "
            f"nullable={self.nullable}{filters_str})"
        )

    def html_summary(self) -> str:
        """Returns a string HTML summary of the :class:`AttrCreator`."""
        filters_str = f", filters=FilterList({self.filters})" if self.filters else ""
        return (
            f" &rarr; tiledb.Attr(name={self.name}, dtype='{self.dtype!s}', "
            f"var={self.var}, nullable={self.nullable}{filters_str})"
        )

    def set_fragment_data(self, fragment_index: int, buffer_data: BufferData):
        if self._fragment_writers is None:
            raise ValueError("Attribute creator has not fragment writers")
        if self.buffer_data.dtype != self.buffer_data:
            raise ValueError(
                "Cannot set data with dtype='{buffer_data.dtype}' to an attribute witha"
                f"dtype='{self._dtype}'."
            )
        # TODO: Check variable length?
        self._fragment_writers[fragment_index].set_attr_data(self.name, buffer_data)

    def to_tiledb(self, ctx: Optional[tiledb.Ctx] = None) -> tiledb.Attr:
        """Returns a :class:`tiledb.Attr` using the current properties.

        Parameters:
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.

        Returns:
            Returns an attribute with the set properties.
        """
        return tiledb.Attr(
            name=self.name,
            dtype=self.dtype,
            fill=self.fill,
            var=self.var,
            nullable=self.nullable,
            filters=self.filters,
            ctx=ctx,
        )
