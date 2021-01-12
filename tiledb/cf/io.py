# Copyright 2020 TileDB Inc.
# Licensed under the MIT License.
"""Input and output routines for the TileDB-CF data model."""

from __future__ import annotations

import warnings
from enum import Enum, unique
from io import StringIO
from typing import Collection, Dict, Generic, Iterable, Optional, Tuple, TypeVar, Union

import numpy as np

import tiledb

_DEFAULT_TILEDB_CF_VERSION = (0, 1, 0)
DType = TypeVar("DType", covariant=True)


class DataspaceSchema:
    """Schema for the TileDB-CF Dataspace representation

    Parameters:
        domain: Domain definition dimension space of
    """

    __slots__ = [
        "_allow_private_dimensions",
        "_array_schema_table",
        "_narray",
        "_dimensions",
        "_tiledb_cf_version",
    ]

    @staticmethod
    def load(uri, ctx: tiledb.Ctx, key: Optional[str] = None):
        pass

    def __init__(
        self,
        array_schemas: Collection[Tuple[str, tiledb.ArraySchema]] = None,
        tiledb_cf_version: Tuple[int, int, int] = _DEFAULT_TILEDB_CF_VERSION,
        allow_private_dimensions: bool = False,
    ):
        self._tiledb_cf_version = tiledb_cf_version
        self._allow_private_dimensions = allow_private_dimensions
        if array_schemas is None:
            self._array_schema_table = {}
            self._narray = 0
        else:
            self._array_schema_table = {pair[0]: pair[1] for pair in array_schemas}
            self._narray = len(array_schemas)
        if len(self._array_schema_table) != self._narray:
            raise ValueError(
                "Invalid array_schema input. Cannot have multiple array schemas with "
                "the same names. Names provided were: "
                f"{self._array_schema_table.keys()}"
            )
        self._dimensions: Dict[str, SharedDimension] = {}
        for (schema_name, schema) in self._array_schema_table.items():
            for dim in schema.domain:
                new_dim = SharedDimension.create(dim)
                if new_dim is not None:
                    dim_name = dim.name
                    if dim_name in self._dimensions:
                        if new_dim != self._dimensions[dim_name]:
                            raise ValueError(
                                f"Dimension {dim} in ArraySchema {schema_name} does not"
                                f" match SharedDimension {self._dimensions[dim_name]}."
                            )
                    else:
                        self._dimensions[dim.name] = new_dim
                elif not allow_private_dimensions:
                    raise ValueError(
                        f"ArraySchema {schema_name} contains a private dimension "
                        f"{dim_name}. To allow private dimensions set "
                        ":param:`allow_private_dimensions` to :code:`True`."
                    )

    def __eq__(self, other):
        """Instance is equal to another ArraySchema"""
        if not isinstance(other, DataspaceSchema):
            return False
        if self._narray != len(other):
            return False
        for name, schema in self._array_schema_table:
            if schema != other.array_schema.get(name):
                return False
        return True

    def __getitem__(self, schema_name: str) -> tiledb.ArraySchema:
        """Returns schema with name given by :param:`schema_name`

        Parameters:
            schema_name: Name of the ArraySchema to be returned.

        Returns:
            ArraySchema with name :param:`schema_name`
        """
        return self._array_schema_table[schema_name]

    def __iter__(self) -> Iterable[Tuple[str, tiledb.ArraySchema]]:
        """Returns a generator that iterates over (name, ArraySchema) pairs."""
        return self._array_schema_table.items()

    def __len__(self):
        """Returns the number of ArraySchemas in the DataspaceSchema"""
        return self._narray

    def check(self):
        """Checks the correctness of the DataspaceSchema

        Raises:
            tiledb.TileDBError: if an ArraySchema in the DataspaceSchema is invalid
            RuntimeError: if a shared :class:`tiledb.Dim` fails to match the defintion
            from the DataspaceSchema
            RuntimeError: if :param:`allow_private_dimensions` is false and there is a
            private :class:`tiledb.Dim`
        """
        for (schema_name, schema) in self._array_schema_table:
            schema.check()
            for dim in schema.domain:
                new_dim = SharedDimension.create(dim)
                if new_dim is not None:
                    if new_dim != self._dimensions[dim.name]:
                        raise RuntimeError(
                            "Incompatible dimension definition for Dimension "
                            f"{dim.name}."
                        )
                elif not self._allow_private_dimensions:
                    raise RuntimeError(
                        f"ArraySchema {schema_name} contains a private Dimension "
                        f"{dim}. To allow private dimensions set "
                        ":param:`allow_private_dimensions` to :code:`True`."
                    )

    def __repr__(self):
        output = StringIO()
        output.write("DataspaceSchema(\n")
        output.write("  SharedDomain (\n")
        for shared_dim in self._dimensions.values():
            output.write(f"    {repr(shared_dim)},")
        output.write("  )\n")
        for name, schema in self:
            output.write("  ArraySchema (\n")
            output.write(f"    name={name}\n")
            output.write("    domain=Domain(*[\n")
            for dim in schema.domain:
                if dim.name.startswith("__."):
                    dim_repr = repr(dim).replace("__.", "")
                else:
                    dim_repr = "(private) " + repr(dim)
                output.write(f"      {dim_repr},\n")
                output.write("    ]),\n")
            output.write("    attrs=[\n")
            for i in range(schema.nattr):
                output.write(f"      {repr(schema.attr(i))},\n")
                output.write("    ],\n")
                output.write(
                    f"    cell_order='{schema.cell_order}',\n"
                    f"    tile_order='{schema.tile_order}',\n"
                )
                output.write(f"    capacity={schema.capacity},\n")
                output.write(f"    sparse={schema.sparse},\n")
                if schema.sparse:
                    output.write(f"    allows_duplicates={schema.allows_duplicates},\n")
            if schema.coords_filters is not None:
                output.write("  coords_filters=FilterList([")
                for index, coord_filter in enumerate(schema.coords_filters):
                    output.write(f"{repr(coord_filter)}")
                    if index < len(schema.coords_filters):
                        output.write(", ")
                output.write("])\n")
            output.write(")\n")
        output.write(")\n")
        return output.getvalue()


class SharedDimension(Generic[DType]):
    """A class for a shared one-dimensional dimension."""

    @classmethod
    def create(cls, dimension: tiledb.Dim):
        """Create a SharedDimension from a tiledb.Dim"""
        dim_name = dimension.name
        if dim_name.startswith("__."):
            if dim_name == "__.":
                raise ValueError(
                    f"Dimension {dimension} does not follow TileDB CF Convention. The "
                    "prefix '__.' denoting a shared dimension must be followed by a "
                    "valid dimension name."
                )
            return cls(dimension.name, dimension.domain, dimension.dtype)
        return None

    __slots__ = [
        "_name",
        "_domain",
        "_data_type",
    ]

    def __init__(
        self,
        name: str,
        domain: Tuple[Optional[DType], Optional[DType]],
        data_type: Union[DataType, str, np.dtype, np.datetime64, np.generic],
    ):
        """Constructor for Axis.

        Parameters:
            name: name of the axis
            domain: domain the axis is defined on
            data_type: TileDBType of Axis data or key to generate the Type
        """
        if "." in name:
            raise ValueError(f"Invalid name {name}. Cannot have '.' in dimension name.")
        self._name = name
        self._domain = domain
        if isinstance(data_type, np.dtype):
            self._dtype = data_type
        elif isinstance(data_type, DataType):
            self._dtype = DataType.dtype
        else:
            self._dtype = DataType.create(data_type).dtype

    def __eq__(self, other: object) -> bool:
        """Returns True if :class:`SharedDimension` is equal to self."""
        if not isinstance(other, SharedDimension):
            return False
        return (
            self._name == other.name
            and self._domain == other._domain
            and self._dtype == other._dtype
        )

    def __repr__(self) -> str:
        return "SharedDimension(name={0!r}, domain={1!s}, dtype='{2!s})".format(
            self._name, self._domain, self._dtype
        )

    @property
    def domain(self) -> Tuple[Optional[DType], Optional[DType]]:
        """A tuple providing the (inclusive) interval the Axis is defined on."""
        return self._domain

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the values this axis stores."""
        return self._dtype

    @property
    def name(self) -> str:
        """The name of the axis."""
        return self._name


@unique
class DataType(Enum):
    """Enumerator for allowable TileDB data types."""

    TILEDB_INT32 = np.dtype("int32")
    TILEDB_UINT32 = np.dtype("uint32")
    TILEDB_INT64 = np.dtype("int64")
    TILEDB_UINT64 = np.dtype("uint64")
    TILEDB_FLOAT32 = np.dtype("float32")
    TILEDB_FLOAT64 = np.dtype("float64")
    TILEDB_INT8 = np.dtype("int8")
    TILEDB_UINT8 = np.dtype("uint8")
    TILEDB_INT16 = np.dtype("int16")
    TILEDB_UINT16 = np.dtype("uint16")
    TILEDB_STRING_UTF8 = np.dtype("U")
    TILEDB_CHAR = np.dtype("S")
    TILEDB_DATETIME_YEAR = np.dtype("M8[Y]")
    TILEDB_DATETIME_MONTH = np.dtype("M8[M]")
    TILEDB_DATETIME_WEEK = np.dtype("M8[W]")
    TILEDB_DATETIME_DAY = np.dtype("M8[D]")
    TILEDB_DATETIME_HR = np.dtype("M8[h]")
    TILEDB_DATETIME_MIN = np.dtype("M8[m]")
    TILEDB_DATETIME_SEC = np.dtype("M8[s]")
    TILEDB_DATETIME_MS = np.dtype("M8[ms]")
    TILEDB_DATETIME_US = np.dtype("M8[us]")
    TILEDB_DATETIME_NS = np.dtype("M8[ns]")
    TILEDB_DATETIME_PS = np.dtype("M8[ps]")
    TILEDB_DATETIME_FS = np.dtype("M8[fs]")
    TILEDB_DATETIME_AS = np.dtype("M8[as]")

    @classmethod
    def create(cls, key: Union[str, np.dtype, np.datetime64, np.generic]) -> DataType:
        """Creates a TileDBType enum from a string or a Numpy dtype or datetime64
        object.

        If the input key is a string, then TileDBType will be created using the key as a
        name.

        If the key is a Numpy datetime64 object, the objects dtype will be used to
        create the TileDBType from value.

        If the key is a Numpy dtype it will be used to create the TileDBType from value,
        unless it is a complex data type. Complex data types will be converted to float
        data types.

        Parameters:
            key: Input key that defines what type to use.

        Returns:
            Corresponding TileDBType enum

        Raises:
            ValueError: if key is a datetime64 object without a specified unit
        """
        if isinstance(key, str):
            return cls[key]
        if isinstance(key, np.datetime64):
            date_unit = np.datetime_data(key)[0]
            if date_unit == "generic":
                raise ValueError(f"datetime {key} does not speficy a date unit")
            return cls(np.dtype("M8[" + date_unit + "]"))
        dtype = np.dtype(key)
        if dtype == np.dtype("complex64"):
            warnings.warn("converting complex64 dtype to multi-value float32 dtype")
            return cls(np.dtype("float32"))
        if dtype == np.dtype("complex128"):
            warnings.warn("converting complex128 dtype to multi-value float64 dtype")
            return cls(np.dtype("float64"))
        return cls(dtype)

    @property
    def dtype(self) -> np.dtype:
        """A Numpy data type that corresponds to this TileDB Type."""
        return self.value
