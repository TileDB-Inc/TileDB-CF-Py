# Copyright 2020 TileDB Inc.
# Licensed under the MIT License.
"""Input and output routines for the TileDB-CF data model."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, MutableMapping
from enum import Enum, unique
from io import StringIO
from typing import (
    Any,
    Collection,
    Dict,
    Generic,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

import tiledb

DType = TypeVar("DType", covariant=True)
_METADATA_ARRAY = "__tiledb_group"
_ATTRIBUTE_METADATA_FLAG = "__tiledb_attr."


class DataspaceArray:
    """Array wrapper to access arrays inside the TileDB-CF Dataspace.

    Parameters:
        uri: Uniform resource identifier for TileDB group or array.
        mode: Open the array object in read 'r' or write 'w' mode.
        key: If not None, encryption key or dictionary of encryption keys to decrypt
            arrays.
        timestamp: If not None, open the TileDB array at the given timestamp.
        array: If not None, specifies which array in a TileDB group to open. If the
            URI is for a TileDB array this must either be None or match the URI
            basename.
        attr: If not None, open one attribute of the TileDB array. If the array is
            dense, indexing the array will return a Numpy nd.array directly.
        ctx: TileDB context
    """

    __slots__ = ["_array", "_attr"]

    def __init__(
        self,
        uri,
        mode="r",
        key: Optional[Union[Dict[str, Union[str, bytes]], str, bytes]] = None,
        timestamp=None,
        array=None,
        attr=None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Constructs a new :class:`DataspaceArray`.

        If the URI is for a TileDB Group either :param:`array` or :param:`attr` must be
        specified. If the URI is for a TileDB Group :param:`array` must either be
        ``None`` or match the basename of the URI.

        Raises:
            ValueError: No array or attribute specified for TileDB group.
            ValueError: URI basename for TileDB array does not match array input
                parameter.
            ValueError: URI is not a valid TileDB object.
        """
        if tiledb.object_type(uri, ctx) == "group":
            if attr is not None and array is None:
                group_schema = GroupSchema.load(uri, ctx, key)
                array = group_schema.get_attribute_array(attr)
                print(f"Array name: {array}")
            if array is None:
                raise ValueError(
                    "Failed to open dataspace array. No array or attribute specified "
                    "for TileDB group."
                )
            array_uri = uri + array if uri.endswith("/") else uri + "/" + array

        elif tiledb.object_type(uri, ctx) == "array":
            array_uri = uri
            array_name = uri.split("/")[-2] if uri.endswith("/") else uri.split("/")[-1]
            if array is not None and array != array_name:
                raise ValueError(
                    f"Failed to open dataspace array. URI for TileDB array with "
                    f"basename={array_name} that does not match parameter "
                    f"array={array}."
                )
        else:
            raise ValueError(
                "Failed to open dataspace group. URI is not a valid TileDB object."
            )
        self._array = tiledb.open(
            array_uri,
            mode=mode,
            key=key,
            attr=attr,
            config=None,
            timestamp=timestamp,
            ctx=ctx,
        )
        self._attr = attr

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    @property
    def array(self):
        """TileDB array opened through dataspace interface."""
        return self._array

    @property
    def array_metadata(self) -> ArrayMetadata:
        return ArrayMetadata(self._array.meta)

    @property
    def attribute_metadata(self) -> AttributeMetadata:
        """Attribute metadata object for array metadata."""
        if self._attr is not None:
            return AttributeMetadata(self._array.meta, self._attr)
        if len(self._array.nattr) == 1:
            return AttributeMetadata(self._array.meta, 0)
        raise ValueError(
            "Failed to open attribute metadata. Dataspace array has multiple "
            "attributes; use get_attribute_metadata to specify which attribute to open."
        )

    def close(self):
        """Closes this :class:`DataspaceGroup`, flushing all buffered data."""
        self._array.close()

    def get_attribute_metadata(self, key: Union[str, int]) -> AttributeMetadata:
        """Returns attribute metadata object corresponding to requested attribute.

        Parameters:
            key: Name or index of the requested array attribute.
        """
        return AttributeMetadata(self._array.meta, key)

    def reopen(self):
        """Reopens this :class:`DataspaceGroup`.

        This is useful when the DataspaceGroup is updated after it was opened.
        To sync-up with the updates, the user must either close the array and open
        again, or just use ``reopen()`` without closing. ``reopen`` will be generally
        faster than a close-then-open.
        """
        self._array.reopen()


class ArrayMetadata(MutableMapping):
    """Metadata wrapper for accesssing array metadata.

    Parameters:
        metadata: TileDB array metadata
    """

    def __init__(self, metadata: tiledb.Metadata):
        """Constructs a new :class:`ArrayMetadata` class."""
        self._metadata = metadata

    def __setitem__(self, key, value):
        """Implementation of [key] <- val (dict item assignment)

        Paremeters:
            key: key to set
            value: corresponding value

        Raises:
            ValueError: Key is reserved for attribute metadata.
        """
        if key.startswith(_ATTRIBUTE_METADATA_FLAG):
            raise ValueError(
                f"Key is reserved for attribute metadata. Cannot set value with key "
                f"`{key}` in array metadata."
            )
        self._metadata[key] = value

    def __getitem__(self, key) -> Any:
        """Implementation of [key] -> val (dict item retrieval)

        Parameters:
            key: Target key to find value from.

        Returns:
            Value stored with provided key.

        Raises:
            ValueError: Key is reserved for attribute metadata.
        """
        if key.startswith(_ATTRIBUTE_METADATA_FLAG):
            raise ValueError(
                f"Key is reserved for attribute metadata. Cannot get value with key "
                f"`{key}` in array metadata."
            )
        return self._metadata.get(key)

    def __contains__(self, key: Any) -> bool:
        """Returns True if 'key' is found in metadata store.

        Provides support for python 'in' syntax ('k in A.meta')

        Parameters:
            key: Target key to check against self.

        Returns:
            True is 'key' is found in the attribute metadata store.
        """
        if key.startswith(_ATTRIBUTE_METADATA_FLAG):
            return False
        return key in self._metadata

    def __delitem__(self, key):
        """Remove key from metadata.

        Parameters:
            key: Target key for item to remove from attribute metadata store.

        Raises:
            ValueError: Key is reserved for attribute metadata.
        """
        if key.startswith(_ATTRIBUTE_METADATA_FLAG):
            raise ValueError(
                f"Key is reserved for attribute metadata. Cannot delete value with key "
                f"`{key}` in array metadata."
            )
        del self._metadata[key]

    def __iter__(self) -> Iterator[str]:
        """Iterates over all attribute metadata keys."""
        for key in self._metadata.keys():
            if not key.startswith(_ATTRIBUTE_METADATA_FLAG):
                yield key

    def __len__(self) -> int:
        """Returns the number of attribute metadata items."""
        return sum(1 for _item in self.__iter__())


class AttributeMetadata(MutableMapping):
    """Metadata wrapper for accesssing attribute metadata.

    Parameters:
        metadata: Full metadata class for the TileDB array.
        attr: Name or index of the array attribute being requested.
    """

    def __init__(self, metadata: tiledb.Metadata, attr: Union[str, int]):
        """Constructs a new :class:`AttributeMetadata`."""
        self._metadata = metadata
        if isinstance(attr, int):
            self._attribute_name = metadata.array.attr(attr).name
        else:
            self._attribute_name = attr
            try:
                metadata.array.attr(attr)
            except tiledb.TileDBError as err:
                raise ValueError(f"Attribute `{attr}` not found in array.") from err
        self._key_prefix = _ATTRIBUTE_METADATA_FLAG + self._attribute_name + "."

    def __setitem__(self, key, value):
        """Implementation of [key] <- val (dict item assignment)

        Paremeters:
            key: key to set
            value: corresponding value

        Raise:
            TypeError: Key is not type str.
        """
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str type")
        self._metadata.__setitem__(self._key_prefix + key, value)

    def __getitem__(self, key) -> Any:
        """Implementation of [key] -> val (dict item retrieval)

        Parameters:
            key: Target key to find value from.

        Returns:
            Value stored with provided key.

        Raise:
            TypeError: Key is not type str.
        """
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str type")
        return self._metadata.__getitem__(self._key_prefix + key)

    def __contains__(self, key: Any) -> bool:
        """Returns True if 'key' is found in metadata store.

        Provides support for python 'in' syntax ('k in A.meta')

        Parameters:
            key: Target key to check against self.

        Returns:
            True is 'key' is found in the attribute metadata store.
        """
        try:
            self[self._key_prefix + key]
        except KeyError:
            return False
        return True

    def __delitem__(self, key):
        """Remove key from metadata.

        Parameters:
            key: Target key for item to remove from attribute metadata store.

        Raise:
            TypeError: Key is not type str.
        """
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str type")
        self._metadata.__delitem__(self._key_prefix + key)

    def __iter__(self) -> Iterator[str]:
        """Iterates over all attribute metadata keys."""
        for key in self._metadata.keys():
            if key.startswith(self._key_prefix):
                yield key[len(self._key_prefix) :]

    def __len__(self) -> int:
        """Returns the number of attribute metadata items."""
        return sum(1 for _item in self.__iter__())


class DataspaceGroup:
    """Array wrapper to access arrays through the TileDB-CF Dataspace API.

    Parameters:
        uri: Uniform resource identifier for TileDB group or array.
        mode: Open the array object in read 'r' or write 'w' mode.
        key: If not None, encryption key or dictionary of encryption keys to decrypt
            arrays.
        timestamp: If not None, open the TileDB metadata array at the given
            timestamp.
        ctx: TileDB context
    """

    @staticmethod
    def metadata_uri(uri: str) -> str:
        """Returns URI for the metadata array given a group URI."""
        return (
            uri + _METADATA_ARRAY if uri.endswith("/") else uri + "/" + _METADATA_ARRAY
        )

    @classmethod
    def create(
        cls,
        uri: str,
        dataspace_schema: GroupSchema,
        key: Optional[Union[Dict[str, Union[str, bytes]], str, bytes]] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Create the group and arrays for a dataspace from a :class:`GroupSchema`.

        Parameters:
            uri: Uniform resource identifier for TileDB group or array.
            dataspace_schema: Schema that defines the group to be created.
            key: If not None, encryption key or dictionary of encryption keys to decrypt
                arrays.
            ctx: TileDB context
        """
        tiledb.group_create(uri, ctx)
        separator = "" if uri.endswith("/") else "/"
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
    ):
        """Constructs a new :class:`GroupSchema`.

        Raises:
            ValueError: URI does not point to a valid TileDB group.
        """
        self._uri = uri
        self._mode = mode
        self._key = key
        self._timestamp = timestamp
        self._ctx = ctx
        if tiledb.object_type(uri, ctx) != "group":
            raise ValueError(
                "Cannot load Dataspace group. URI does not point to a valid TileDB "
                "group."
            )
        self._schema = GroupSchema.load(uri, ctx, key)
        self._metadata_uri = (
            uri + _METADATA_ARRAY if uri.endswith("/") else uri + "/" + _METADATA_ARRAY
        )
        self._metadata_key = key.get(_METADATA_ARRAY) if isinstance(key, dict) else key
        self._metadata_array = (
            None
            if self._schema.metadata_schema is None
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

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    def close(self):
        """Closes this DataspaceGroup, flushing all buffered data."""
        if self._metadata_array is not None:
            self._metadata_array.close()

    def create_metadata_array(self):
        """Creates a metadata array for this group.

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

    @property
    def has_metadata_array(self) -> bool:
        """Returns true if there a metadata array for storing group metadata."""
        return self._metadata_array is not None

    @property
    def meta(self) -> Optional[tiledb.Metadata]:
        if self._metadata_array is None:
            return None
        return self._metadata_array.meta

    def reopen(self, timestamp=None):
        """Reopens this DataspaceGroup.

        This is useful when the DataspaceGroup is updated after it was opened.
        To sync-up with the updates, the user must either close the array and open
        again, or just use ``reopen()`` without closing. ``reopen`` will be generally
        faster than a close-then-open.
        """
        self._schema = GroupSchema.load(
            self._uri,
            self._ctx,
            self._key,
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
    """Schema for the TileDB-CF Dataspace representation.

    Parameters:
        array_schemas: A collection of (name, ArraySchema) tuples for Arrays that belong
            to this group.
        metadata_schema: If not None, a schema for the group metadata array.
    """

    @classmethod
    def load(
        cls,
        uri: str,
        ctx: tiledb.Ctx,
        key: Optional[Union[Dict[str, Union[str, bytes]], str, bytes]] = None,
    ):
        """Load a dataspace schema for a TileDB group

        Parameters:
            uri: uniform resource identifier for the TileDB group
            ctx: a TileDB context
            key: encryption key or dictionary of encryption keys (by array name)
        """
        metadata_schema = None
        print("Check group")
        if tiledb.object_type(uri, ctx) != "group":
            raise ValueError(
                f"Failed to load the dataspace schema. Provided uri '{uri}' is not a "
                f"valid TileDB group."
            )
        vfs = tiledb.VFS(ctx=ctx)
        array_schemas = []
        for item_uri in vfs.ls(uri):
            if not tiledb.object_type(item_uri) == "array":
                continue
            array_name = (
                item_uri.split("/")[-2]
                if item_uri.endswith("/")
                else item_uri.split("/")[-1]
            )
            local_key = key.get(array_name) if isinstance(key, dict) else key
            if array_name == _METADATA_ARRAY:
                metadata_schema = tiledb.ArraySchema.load(item_uri, ctx, local_key)
            else:
                array_schemas.append(
                    (array_name, tiledb.ArraySchema.load(item_uri, ctx, local_key))
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
        """Constructs a :class:`GroupSchema`.

        Raises:
            ValueError: ArraySchema has duplicate names.
        """
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
        if not isinstance(other, GroupSchema):
            return False
        if self._narray != len(other):
            return False
        for name, schema in self._array_schema_table.items():
            if schema != other.get(name):
                return False
        if self._metadata_schema != other.metadata_schema:
            return False
        return True

    def __getitem__(self, schema_name: str) -> tiledb.ArraySchema:
        """Returns the :class:`Arrayschema` with the name given by :param:`schema_name`.

        Parameters:
            schema_name: Name of the ArraySchema to be returned.

        Returns:
            ArraySchema with name :param:`schema_name`.
        """
        return self._array_schema_table[schema_name]

    def __iter__(self) -> Iterator[str]:
        """Returns a generator that iterates over (name, ArraySchema) pairs."""
        return self._array_schema_table.__iter__()

    def __len__(self) -> int:
        """Returns the number of ArraySchemas in the GroupSchema"""
        return self._narray

    def __repr__(self) -> str:
        """Returns the object representation of this GroupSchema in string form."""
        output = StringIO()

        output.write("  SharedDomain (\n")
        for shared_dim in self._dimensions.values():
            output.write(f"    {repr(shared_dim)},")
        output.write("  )\n")
        for name, schema in self.items():
            output.write(f"{name} {repr(schema)}")
        output.write(")\n")
        return output.getvalue()

    def check(self):
        """Checks the correctness of the GroupSchema.

        Raises:
            tiledb.TileDBError: An ArraySchema in the GroupSchema is invalid.
            RuntimeError: A shared :class:`tiledb.Dim` fails to match the definition
                from the GroupSchema.
        """
        for (schema_name, schema) in self._array_schema_table.items():
            schema.check()
            for dim in schema.domain:
                if SharedDimension.create(dim) != self._dimensions[dim.name]:
                    raise RuntimeError(
                        f"Database schema check failed; dimension definition for "
                        f"dimension {dim.name} in array schema {schema_name}."
                    )
        if self._metadata_schema is not None:
            self._metadata_schema.check()

    def get_all_attribute_arrays(
        self, attribute_name: str
    ) -> Optional[Tuple[str, ...]]:
        """Returns a tuple of the names of all arrays with a matching attribute.

        Parameter:
            attribute_name: Name of the attribute to look up arrays for.

        Returns:
            A tuple of the name of all arrays with a matching attribute, or `None` if no
                such array.
        """
        return self._attribute_to_arrays.get(attribute_name)

    def get_attribute_array(self, attribute_name: str) -> str:
        """Returns the name of the array in which the :param:`attribute_name` is
            contained.

        Parameters:
            attribute_name: Name of the attribute to look up array for.

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

    def set_default_metadata_schema(self, ctx=None):
        """Set the metadata schema to a default placeholder DenseArray."""
        self._metadata_schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="dim", domain=(0, 0), tile=1, dtype=np.int32, ctx=ctx)
            ),
            attrs=[tiledb.Attr(name="attr", dtype=np.int32, ctx=ctx)],
            sparse=False,
        )


class SharedDimension(Generic[DType]):
    """A class for a shared one-dimensional dimension.

    Parameters:
        name: Name of the :class:`SharedDimension`.
        domain: The (inclusive) interval on which the :class:`SharedDimension` is
            valid.
        data_type: Numpy dtype of the dimension values and domain.
    """

    @classmethod
    def create(cls, dimension: tiledb.Dim):
        """Create a SharedDimension from a tiledb.Dim.

        Create a SharedDimension using the name, domain, and type of a tiledb.dim.

        Parameters:
            dimension: TileDB dimension that will be used to create the shared
                dimension.
        """
        return SharedDimension(dimension.name, dimension.domain, dimension.dtype)

    __slots__ = [
        "_name",
        "_domain",
        "_dtype",
    ]

    def __init__(
        self,
        name: str,
        domain: Tuple[Optional[DType], Optional[DType]],
        data_type: Union[DataType, str, np.dtype, np.datetime64, np.generic],
    ):
        """Constructs a new :class:`SharedDimension`.

        Raises:
            ValueError: Name contains reserved character '.'.
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
        return (
            f"SharedDimension(name={self._name}, domain={self._domain}, "
            f"dtype={self._dtype})"
        )

    @property
    def domain(self) -> Tuple[Optional[DType], Optional[DType]]:
        """A tuple providing the (inclusive) interval which the :class:`SharedDimension`
        is defined on."""
        return self._domain

    @property
    def dtype(self) -> np.dtype:
        """The numpy.dtype of the values and domain."""
        return self._dtype

    @property
    def name(self) -> str:
        """The name of the dimension."""
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
            Corresponding TileDBType enum.

        Raises:
            ValueError: Key is a datetime64 object without a specified unit.
        """
        if isinstance(key, str):
            return cls[key]
        if isinstance(key, np.datetime64):
            date_unit = np.datetime_data(key)[0]
            if date_unit == "generic":
                raise ValueError(f"Datetime {key} does not speficy a date unit.")
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
