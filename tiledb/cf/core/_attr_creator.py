from __future__ import annotations

from typing import Optional

import numpy as np
from typing_extensions import Self

import tiledb

from .._utils import DType
from .registry import RegisteredByNameMixin, Registry


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
    ):
        self.dtype = np.dtype(dtype)
        self.fill = fill
        self.var = var
        self.nullable = nullable
        self.filters = filters
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
