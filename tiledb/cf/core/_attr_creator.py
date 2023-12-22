from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from tiledb.datatypes import DataType
from typing_extensions import Protocol

import tiledb

from .._utils import DType
from ._fragment_writer import FragmentWriter
from .registry import RegisteredByNameMixin
from .source import FieldData, create_field_data


class AttrRegistry(Protocol):
    def __delitem__(self, name: str):
        """Delete the element with the provided name."""

    def __getitem__(self, name: str) -> AttrCreator:
        """Get the element with the provided name."""

    def __setitem__(self, name: str, value: AttrCreator):
        """Set the elemetn with the provided name to the provided value."""

    def set_writer_data(
        self, writer_index: Optional[int], attr_name: str, data: FieldData
    ):
        """Set the data to the requested frgament writer."""

    def rename(self, old_name: str, new_name: str):
        """Rename an element of the registry.

        If the rename fails, the registry should be left unchanged.
        """


class AttrCreator(RegisteredByNameMixin):
    """Creator for a TileDB attribute.

    Parameters
    ----------
    name
        Name of the attribute that will be created.
    dtype
        The datatype of the attribute that will be created.
    fill
        Optional fill value for the attribute that will be created.
    var
        Specifies if the attribute that will be created will be variable length
        (automatic for byte/strings).
    nullable
        Specifies if the attribute that will be created will be nullable using
        validity tiles.
    filters
        Filter pipeline to apply to the attribute.
    registry
        Registry for this attribute creator.
    fragment_writers
        Fragment writers for this attribute creator.

    Attributes
    ----------
    dtype: np.dtype
        Numpy dtype of the attribute.
    fill: int or float or str, optional
        Fill value for unset cells.
    var: bool
        Specifies if the attribute is variable length (automatic for
        byte/strings).
    nullable: bool
        Specifies if the attribute is nullable using validity tiles.
    filters: tiledb.FilterList, optional
        Specifies compression filters for the attribute.
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
        self.dtype = DataType.from_numpy(dtype).np_dtype
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
        """Returns a string HTML summary of the ``AttrCreator``."""
        filters_str = f", filters=FilterList({self.filters})" if self.filters else ""
        return (
            f" &rarr; tiledb.Attr(name={self.name}, dtype='{self.dtype!s}', "
            f"var={self.var}, nullable={self.nullable}{filters_str})"
        )

    def set_writer_data(
        self,
        attr_data: Union[np.ndarray, FieldData],
        *,
        writer_index: Optional[int] = None,
    ):
        """Set attribute data to the specified fragment writer.

        Parameters
        ----------
        attr_data
            Attribute data to add to the writer.
        writer_index
            The index of the fragment writer to add to.
        """
        if self._registry is None:
            raise ValueError("Attribute creator is not registered to an array.")
        data = create_field_data(attr_data, self.dtype)
        self._registry.set_writer_data(writer_index, self.name, data)

    def to_tiledb(self, ctx: Optional[tiledb.Ctx] = None) -> tiledb.Attr:
        """Returns a :class:`tiledb.Attr` using the current properties.

        Parameters
        ----------
        ctx
            If not ``None``, TileDB context wrapper for a TileDB storage manager.

        Returns
        -------
        tiledb.Attr
            An attribute with the properties defined in this attribute creator.
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
