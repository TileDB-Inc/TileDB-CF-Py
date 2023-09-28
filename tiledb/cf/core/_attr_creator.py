from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from typing_extensions import Protocol

import tiledb

from .._utils import DType
from ._fragment_writer import FragmentWriter
from .registry import RegisteredByNameMixin
from .source import FieldData, NumpyData


class AttrRegistry(Protocol):
    def __delitem__(self, name: str):
        ...

    def __getitem__(self, name: str) -> AttrCreator:
        ...

    def __setitem__(self, name: str, value: AttrCreator):
        ...

    def set_fragment_data(self, fragment_index: int, attr_name: str, data: FieldData):
        ...

    def rename(self, old_name: str, new_name: str):
        ...


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
        registry: Optional[AttrRegistry] = None,
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

    def set_fragment_data(
        self, fragment_index: int, attr_data: Union[np.ndarray, FieldData]
    ):
        if self._registry is None:
            raise ValueError("Attribute creator is not registered to an array.")
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
