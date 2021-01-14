# Copyright 2020 TileDB Inc.
# Licensed under the MIT License.
"""Input and output routines for the TileDB-CF data model."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from enum import Enum, unique
from io import StringIO
from typing import Collection, Dict, Generic, Iterator, Optional, Tuple, TypeVar, Union

import numpy as np

import tiledb

DType = TypeVar("DType", covariant=True)
_METADATA_ARRAY = "__tiledb_group"


class DataspaceArray:
    """Array wrapper to access arrays inside the TileDB-CF Dataspace"""

    __slots__ = [
        "_array",
        "_ctx",
        "_key",
        "_is_array",
        "_metadata_array",
        "_mode",
        "_schema",
        "_timestamp",
        "_uri",
    ]

    def __init__(
        self,
        uri,
        mode="r",
        key: Optional[Union[Dict[str, Union[str, bytes]], str, bytes]] = None,
        timestamp=None,
        array=None,
        attr=None,
        ctx: Optional[tiledb.Ctx] = None,
        directory_separator: str = "/",
    ):
        if tiledb.object_type(uri, ctx) == "group":
            if attr is not None and array is None:
                group_schema = GroupSchema.load(uri, ctx, key, directory_separator)
                array = group_schema.get_attribute_array(attr)
            if array is None:
                raise ValueError(
                    "Failed to open dataspace array. No array or attribute specified "
                    "for URI is for a TileDB group."
                )
            array_uri = (
                uri + array
                if uri.endswith(directory_separator)
                else uri + directory_separator + array
            )
        elif tiledb.object_type(uri, ctx) == "array":
            array_uri = uri
            array_name = (
                uri.split(directory_separator)[-2]
                if uri.endswith(directory_separator)
                else uri.split(directory_separator)[-1]
            )
            if array is not None and array != array_name:
                raise ValueError(
                    f"Failed to open dataspace array. URI is for a TileDB array with "
                    f"name {array_name} that does not match input array={array}."
                )
        else:
            raise ValueError(
                "Failed to open dataspace group. URI is not a valid TileDB object."
            )
        self._array = tiledb.Array(array_uri, mode, key, timestamp, attr, ctx)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def array(self):
        """TileDB array opened through dataspace interface."""
        return self._array

    def close(self):
        """Closes this DataspaceGroup, flushing all buffered data."""
        if self._array is not None:
            self._array.close()

    def reopen(self):
        """Reopen this DataspaceGroup

        This is useful when the DataspaceGroup is updated after it was opened.
        To sync-up with the updates, the user must either close the array and open
        again, or just use ``reopen()`` without closing. ``reopen`` will be generally
        faster than a close-then-open.
        """
        if self._array is not None:
            self._array.reopen()


class DataspaceGroup:
    """Array wrapper to access arrays inside the TileDB-CF Dataspace"""

    @classmethod
    def create(
        cls,
        uri: str,
        dataspace_schema: GroupSchema,
        key: Optional[Union[Dict[str, Union[str, bytes]], str, bytes]] = None,
        ctx: Optional[tiledb.Ctx] = None,
        directory_separator: str = "/",
    ):
        tiledb.group_create(uri, ctx)
        separator = "" if uri.endswith(directory_separator) else directory_separator
        if dataspace_schema.metadata_schema is not None:
            tiledb.DenseArray.create(
                uri + separator + _METADATA_ARRAY,
                dataspace_schema.metadata_schema,
                key.get(_METADATA_ARRAY) if isinstance(key, dict) else key,
                ctx,
            )
        for array_name, array_schema in dataspace_schema.items():
            tiledb.Array.create(
                uri + separator + array_name,
                array_schema,
                key.get(array_name) if isinstance(key, dict) else key,
                ctx,
            )

    __slots__ = [
        "_ctx",
        "_directory_separator",
        "_key",
        "_metadata_array",
        "_metadata_key",
        "_metadata_uri",
        "_mode",
        "_schema",
        "_timestamp",
        "_uri",
    ]

    def __init__(
        self,
        uri,
        mode="r",
        key: Optional[Union[Dict[str, Union[str, bytes]], str, bytes]] = None,
        timestamp=None,
        ctx: Optional[tiledb.Ctx] = None,
        directory_separator: str = "/",
    ):
        self._uri = uri
        self._mode = mode
        self._key = key
        self._timestamp = timestamp
        self._ctx = ctx
        self._directory_separator = directory_separator
        if tiledb.object_type(uri, ctx) != "group":
            raise ValueError(
                "Cannot load Dataspace group. URI does not point to a valid TileDB "
                "group."
            )
        self._schema = GroupSchema.load(uri, ctx, key, directory_separator)
        self._metadata_uri = (
            uri + _METADATA_ARRAY
            if uri.endswith(directory_separator)
            else uri + directory_separator + _METADATA_ARRAY
        )
        self._metadata_key = key.get(_METADATA_ARRAY) if isinstance(key, dict) else key
        self._metadata_array = (
            None
            if self._schema.metadata_scheman is None
            else tiledb.Array(
                uri=self._metadata_uri,
                mode=self._mode,
                key=self._metadata_key,
                timestamp=self._timestamp,
                ctx=self._ctx,
            )
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Closes this DataspaceGroup, flushing all buffered data."""
        self._metadata_array.close()

    def create_metadata_array(self):
        """Create a metadata array for this group.

        This routine will create a metadata array for the group. An error will be raised
        if the metadata array already exists (either directly if the array is open or
        in directly during the Array creation step.

        The user must either close the group and open it again, or just use
        :meth:`reopen` without closing to read and write to the metadata array.

        Raises:
           RuntimeError: Metadata array exists and is open.
        """
        if self._metadata_array is not None:
            raise RuntimeError(
                "Failed to create metadata array; array exists and is open."
            )
        if self._schema.metadata_schema is None:
            self._schema.set_default_metadata_schema()
        tiledb.Array.create(
            self._metadata_uri,
            self._schema.metadata_schema,
            self._metadata_key,
            self._ctx,
        )

    def reopen(self, timestamp=None):
        """Reopen this DataspaceGroup

        This is useful when the DataspaceGroup is updated after it was opened.
        To sync-up with the updates, the user must either close the array and open
        again, or just use ``reopen()`` without closing. ``reopen`` will be generally
        faster than a close-then-open.
        """
        self._schema = GroupSchema.load(
            self._uri,
            self._ctx,
            self._key,
            self._directory_separator,
        )
        self._timestamp = timestamp
        if self._metadata_array is None:
            self._metadata_array = (
                None
                if self._schema.metadata_schema is None
                else tiledb.Array(
                    uri=self._metadata_uri,
                    mode=self._mode,
                    key=self._key,
                    timestamp=self._timestamp,
                    ctx=self._ctx,
                )
            )
        else:
            self._metadata_array.reopen()


class GroupSchema(Mapping):
    """Schema for the TileDB-CF Dataspace representation

    Parameters:
        domain: Domain definition dimension space of
    """

    @classmethod
    def load(
        cls,
        uri: str,
        ctx: tiledb.Ctx,
        key: Optional[Union[Dict[str, Union[str, bytes]], str, bytes]] = None,
        directory_separator: str = "/",
    ):
        """Load a dataspace schema for a TileDB group

        Parameters:
            uri: uniform resource identifier for the TileDB group
            ctx: a TileDB context
            key: encryption key or dictionary of encryption keys (by array name)
        """
        metadata_schema = None
        if tiledb.object_type(uri, ctx) != "group":
            raise ValueError(
                f"Failed to load the dataspace schema. Provided uri {uri} is not a "
                f"valid TileDB group."
            )
        vfs = tiledb.VFS(ctx=ctx)
        array_schemas = []
        for item in vfs.ls(uri):
            if not tiledb.object_type(item) == "array":
                continue
            array_name = (
                item.split(directory_separator)[-2]
                if item.endswith(directory_separator)
                else item.split(directory_separator)[-1]
            )
            local_key = key.get(array_name) if isinstance(key, dict) else key
            if array_name == _METADATA_ARRAY:
                metadata_schema = tiledb.ArraySchema.load(uri, ctx, local_key)
            else:
                array_schemas.append(
                    (array_name, tiledb.ArraySchema.load(uri, ctx, local_key))
                )
        return cls(array_schemas, metadata_schema)

    __slots__ = [
        "_allow_private_dimensions",
        "_array_schema_table",
        "_attribute_to_arrays",
        "_dimensions",
        "_group_schema",
        "_metadata_schema",
        "_narray",
    ]

    def __init__(
        self,
        array_schemas: Optional[Collection[Tuple[str, tiledb.ArraySchema]]] = None,
        metadata_schema: Optional[tiledb.ArraySchema] = None,
    ):
        self._metadata_schema = metadata_schema
        if array_schemas is None:
            self._array_schema_table = {}
            self._narray = 0
        else:
            self._array_schema_table = {pair[0]: pair[1] for pair in array_schemas}
            self._narray = len(array_schemas)
        if len(self._array_schema_table) != self._narray:
            raise ValueError(
                "Initializing dataspace schema failed; ArraySchemas must have unique "
                "names."
            )
        self._dimensions: Dict[str, SharedDimension] = {}
        self._attribute_to_arrays: Dict[str, Tuple[str, ...]] = {}
        for (schema_name, schema) in self._array_schema_table.items():
            for attr in schema:
                attr_name = attr.name
                self._attribute_to_arrays[attr_name] = (
                    (schema_name,)
                    if attr_name not in self._attribute_to_arrays
                    else self._attribute_to_arrays[attr_name] + (schema_name,)
                )
            for array_dim in schema.domain:
                dim_name = array_dim.name
                if dim_name in self._dimensions:
                    if SharedDimension.create(array_dim) != self._dimensions[dim_name]:
                        raise ValueError(
                            f"Initializing dataspace schema failed; dimension "
                            f" {dim_name} in array schema {schema_name} does not match "
                            f"existing shared dimension {self._dimensions[dim_name]}."
                        )
                else:
                    self._dimensions[dim_name] = SharedDimension.create(array_dim)

    def __eq__(self, other):
        """Instance is equal to another ArraySchema"""
        if not isinstance(other, GroupSchema):
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

    def __iter__(self) -> Iterator[str]:
        """Returns a generator that iterates over (name, ArraySchema) pairs."""
        return self._array_schema_table.__iter__()

    def __len__(self) -> int:
        """Returns the number of ArraySchemas in the GroupSchema"""
        return self._narray

    def __repr__(self) -> str:
        """Returns the object representation of this GroupSchema in string orm."""
        output = StringIO()
        output.write("GroupSchema(\n")
        output.write("  SharedDomain (\n")
        for shared_dim in self._dimensions.values():
            output.write(f"    {repr(shared_dim)},")
        output.write("  )\n")
        for name, schema in self.keys():
            output.write(f"{name} {repr(schema)}")
        output.write(")\n")
        return output.getvalue()

    def check(self):
        """Checks the correctness of the GroupSchema

        Raises:
            tiledb.TileDBError: if an ArraySchema in the GroupSchema is invalid
            RuntimeError: if a shared :class:`tiledb.Dim` fails to match the defintion
            from the GroupSchema
        """
        for (schema_name, schema) in self._array_schema_table:
            schema.check()
            for dim in schema.domain:
                if SharedDimension.create(dim) != self._dimensions[dim.name]:
                    raise RuntimeError(
                        f"Database schema check failed; dimension definition for "
                        f"dimension {dim.name} in array schema {schema_name}."
                    )

    def get_all_attribute_arrays(
        self, attribute_name: str
    ) -> Optional[Tuple[str, ...]]:
        """Return a tuple of the names of all arrays with a matching attribute

        Parameter:
            attribute_name: Name of the attribute to query on.

        Returns:
            A tuple of the name of all arrays with a matching attribute.
        """
        return self._attribute_to_arrays.get(attribute_name)

    def get_attribute_array(self, attribute_name: str) -> str:
        """Return the name of the array :param:`attribute_name` is contained

        Parameters:
            attribute_name: Name of the attribute to query on.

        Returns:
            Name of the array that contains the attribute with a matching name.

        Raises:
            KeyError: No attribute with name :param:`attribute_name` found.
            ValueError: More than one array with :param:`attribute_name` found.
        """
        arrays = self._attribute_to_arrays.get(attribute_name)
        if arrays is None:
            raise KeyError(f"No attribute with name {attribute_name} found.")
        assert len(arrays) > 0
        if len(arrays) > 1:
            raise ValueError(
                f"More than one array with attribute name {attribute_name} found."
                f"Arrays with that attribute are: {arrays}."
            )
        return arrays[0]

    @property
    def metadata_schema(self):
        """ArraySchema for the dataspace-level metadata."""
        return self._metadata_schema

    def set_default_metadata_schema(self, ctx):
        """Set the metadata schema to a default placeholder DenseArray."""
        self._metadata_schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="dim", domain=(0, 0), tile=1, dtype=np.int32, ctx=ctx)
            ),
            attrs=[tiledb.Attr(name="attr", dtype=np.int32, ctx=ctx)],
            sparse=False,
        )


class SharedDimension(Generic[DType]):
    """A class for a shared one-dimensional dimension."""

    @classmethod
    def create(cls, dimension: tiledb.Dim):
        """Create a SharedDimension from a tiledb.Dim.

        Create a SharedDimension using the name, domain, and type of a tiledb.dim.

        Parameters:
            dimension:
        """
        SharedDimension(dimension.name, dimension.domain, dimension.dtype)

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
