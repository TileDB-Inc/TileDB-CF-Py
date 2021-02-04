# Copyright 2020 TileDB Inc.
# Licensed under the MIT License.
"""Input and output routines for the TileDB-CF data model."""

from __future__ import annotations

import os.path
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

import numpy as np

import tiledb

DType = TypeVar("DType", covariant=True)
_METADATA_ARRAY = "__tiledb_group"
_ATTRIBUTE_METADATA_FLAG = "__tiledb_attr."
_CF_COORDINATE_SUFFIX = ".axis_data"


def _get_array_uri(group_uri: str, array_name: str) -> str:
    """Returns a URI for an array with name :param:`array_name` inside a group at URI
        :param:`group_uri`.

    Parameters:
        group_uri: URI of the group containing the array
        array_name: name of the array

    Returns:
        Array URI of an array with name :param:`array_name` inside a group at URI
            :param:`group_uri`.
    """
    return os.path.join(group_uri, array_name)


class Metadata(MutableMapping):
    """Class for accessing Metadata using the standard MutableMapping API.

    Parameters:
        metadata (tiledb.Metadata): TileDB array metadata object.
    """

    def __init__(self, metadata: tiledb.Metadata):
        self._metadata = metadata

    def __iter__(self) -> Iterator[str]:
        """Iterates over all metadata keys."""
        for tiledb_key in self._metadata.keys():
            key = self._from_tiledb_key(tiledb_key)
            if key is not None:
                yield key

    def __len__(self) -> int:
        """Returns the number of metadata items."""
        return sum(1 for _ in self)

    def __getitem__(self, key: str) -> Any:
        """Implementation of [key] -> val (dict item retrieval).

        Parameters:
            key: Key to find value from.

        Returns:
            Value stored with provided key.
        """
        return self._metadata[self._to_tiledb_key(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        """Implementation of [key] <- val (dict item assignment).

        Paremeters:
            key: key to set
            value: corresponding value
        """
        self._metadata[self._to_tiledb_key(key)] = value

    def __delitem__(self, key):
        """Implementation of del [key] (dict item deletion).

        Parameters:
            key: Key to remove.
        """
        del self._metadata[self._to_tiledb_key(key)]

    def _to_tiledb_key(self, key: str) -> str:
        """Map an external user metadata key to an internal tiledb key.

        Raises
            KeyError: If `key` cannot be mapped.
        """
        return key

    def _from_tiledb_key(self, tiledb_key: str) -> Optional[str]:
        """Map an internal tiledb key to an external user metadata key.

        Returns:
            The external user metadata key corresponding to `tiledb_key`,
            or None if there is no such corresponding key.
        """
        return tiledb_key


class ArrayMetadata(Metadata):
    """Class for accessing array-related metadata from a TileDB metadata object.

    This class provides a way for accessing the TileDB array metadata that excludes
    attribute-specific metadata.

    Parameters:
        metadata (tiledb.Metadata): TileDB array metadata object for the desired array.
    """

    def _to_tiledb_key(self, key: str) -> str:
        if key.startswith(_ATTRIBUTE_METADATA_FLAG):
            raise KeyError("Key is reserved for attribute metadata.")
        return key

    def _from_tiledb_key(self, tiledb_key: str) -> Optional[str]:
        if not tiledb_key.startswith(_ATTRIBUTE_METADATA_FLAG):
            return tiledb_key
        return None


class AttributeMetadata(Metadata):
    """Metadata wrapper for accessing attribute metadata.

    This class allows access to the metadata for an attribute stored in the metadata
    for a TileDB array.

    Parameters:
        metadata (tiledb.Metadata): TileDB array metadata for the array containing the
            desired attribute.
        attr (str): Name or index of the arrary attribute being requested.
    """

    def __init__(self, metadata: tiledb.Metadata, attr: Union[str, int]):
        super().__init__(metadata)
        try:
            attr_name = metadata.array.attr(attr).name
        except tiledb.TileDBError as err:
            raise ValueError(f"Attribute `{attr}` not found in array.") from err
        self._key_prefix = _ATTRIBUTE_METADATA_FLAG + attr_name + "."

    def _to_tiledb_key(self, key: str) -> str:
        return self._key_prefix + key

    def _from_tiledb_key(self, tiledb_key: str) -> Optional[str]:
        if tiledb_key.startswith(self._key_prefix):
            return tiledb_key[len(self._key_prefix) :]
        return None


class AttributeDataspaceMap:

    __slots__ = ["_attribute_map"]

    def __init__(self):
        self._attribute_map: Dict[str, Tuple[str, Optional[str]]] = dict()

    def __getitem__(self, key: str) -> Union[str, Tuple[str, str]]:
        return self._attribute_map[key][0]

    def __len__(self) -> int:
        return len(self._attribute_map)

    def add_array(self, array_name: Optional[str], array_schema: tiledb.ArraySchema):
        for attr in array_schema:
            attr_name = attr.name
            attr_key = (
                attr_name[: -len(_CF_COORDINATE_SUFFIX)]
                if attr_name.endswith(_CF_COORDINATE_SUFFIX)
                else attr_name
            )
            if attr_key in self._attribute_map:
                raise RuntimeError(
                    f"Failed to add attribute '{attr_key}'. Attribute already exitsts."
                )
            if array_name is None:
                self._attribute_map[attr_key] = attr_name
            else:
                self._attribute_map[attr_key] = (attr_name, array_name)

    def has_attr(self, key: str) -> bool:
        return key in self._attribute_map


class Dataspace:
    """Class for opening a TileDB Dataspace."""

    def __init__(
        self,
        uri: str,
        key: Optional[Union[Dict[str, str], str]] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        self._uri = uri
        self._key = key
        self._ctx = ctx
        self._attribute_map = AttributeDataspaceMap()
        self._dimension_map: Dict[str, SharedDimension] = {}
        if tiledb.object_type(uri, ctx) == "array":
            self._schema = tiledb.ArraySchema.load(uri, ctx, key)
            self._add_array(self._schema)
            self._attribute_map.add_array(None, self._schema)
        elif tiledb.object_type(uri, ctx) == "group":
            self._schema = GroupSchema.load_group(uri, ctx, key)
            for array_name, array_schema in self._schema.items():
                self._add_array(array_schema)
        else:
            raise ValueError(
                f"Failed to load the dataspace schema. Provided uri '{uri}' is not a "
                f"valid TileDB object."
            )

    def _add_array(self, array_schema):
        for tiledb_dim in array_schema.domain:
            dim = SharedDimension.from_tiledb_dim(tiledb_dim)
            if dim.name in self._dimension_map:
                if dim != self._dimension_map[dim.name]:
                    raise RuntimeError(
                        f"Failed to initialize Dataspace; all dimensions in the "
                        f"group with the same name must have the same domain and "
                        f"type. Dimension {dim.name} has inconsisent definitions."
                    )
            else:
                self._dimension_map[dim.name] = dim

    @property
    def array_names(self) -> Iterator[Optional[str]]:
        if self.is_array:
            yield None
        else:
            return self._schema.keys()

    @property
    def array_schemas(self) -> Iterator[tiledb.ArraySchema]:
        if self.is_array:
            yield self._schema
        else:
            return self._schema.values()

    def array(self, array: Optional[str], attr: Optional[str]) -> DataspaceArrayWrapper:
        return DataspaceArrayWrapper(self, array, attr)

    @property
    def dim_names(self) -> List[str]:
        """A list of names of dimensions in this :class:`DataspaceMap`."""
        return list(self._dimension_map.keys())

    def array_wrapper(
        self,
        array: Optional[str] = None,
        attr: Optional[str] = None,
        mode: str = "r",
        timestamp: Optional[int] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ) -> DataspaceArrayWrapper:
        return DataspaceArrayWrapper(
            self,
            array=array,
            attr=attr,
            mode=mode,
            timestamp=timestamp,
            ctx=ctx,
        )

    def attr_name(self, key: Union[int, str]) -> str:
        if isinstance(key, int):
            raise NotImplementedError
        return self._attribute_map[key][1]

    def metadata_array_wrapper(
        self,
        mode: str = "r",
        timestamp: Optional[int] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ) -> DataspaceArrayWrapper:
        if self.is_array:
            return DataspaceArrayWrapper(self)
        return DataspaceArrayWrapper(
            self,
            array=_METADATA_ARRAY,
            mode=mode,
            timestamp=timestamp,
            ctx=ctx,
        )

    def has_attr(self, name: str) -> bool:
        return self._attribute_map.has_attr(name)

    @property
    def key(self) -> Optional[Union[Dict[str, str], str]]:
        return self._key

    @property
    def is_array(self) -> bool:
        return isinstance(self._schema, tiledb.ArraySchema)

    @property
    def nattr(self) -> int:
        return len(self._attribute_map)

    @property
    def ndim(self) -> int:
        return len(self._dimension_map)

    @property
    def uri(self) -> str:
        return self._uri


class DataspaceArrayWrapper:
    def __init__(
        self,
        dataspace: Dataspace,
        array: Optional[str] = None,
        attr: Optional[str] = None,
        mode: str = "r",
        timestamp: Optional[int] = None,
        ctx: Optional[tiledb.ctx] = None,
    ):
        self._dataspace = dataspace
        if dataspace.is_array:
            if array is not None:
                raise ValueError(
                    "Failed to open DataspaceArrayWrapper. Cannot specify array for "
                    "dataspace of type array."
                )
            self._group = None
            self._array = tiledb.open(
                uri=dataspace.uri,
                mode=mode,
                attr=attr,
                key=dataspace.key,
                timestamp=timestamp,
                ctx=ctx,
            )
        else:
            self._group = Group(
                uri=dataspace.uri,
                mode=mode,
                array=array,
                attr=attr,
                key=dataspace.key,
                timestamp=timestamp,
                ctx=ctx,
            )
            self._array = self._group.array
            assert self._array is not None
        self._attr_map = (
            {dataspace.attr_name(attr): attr}
            if attr is not None
            else {
                dataspace.attr_name(attr.name): attr.name for attr in self._array.schema
            }
        )

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    @property
    def array(self) -> tiledb.Array:
        """The opened array, or ``None`` if no array was opened."""
        return self._array

    @property
    def attribute_metadata(self) -> Dict[str, AttributeMetadata]:
        return {
            attr_key: AttributeMetadata(self._array.meta, attr_key)
            for attr_key in self._attr_map.keys()
        }

    def close(self):
        self._array.close()

    @property
    def meta(self) -> ArrayMetadata:
        return ArrayMetadata(self._array.meta)

    def reopen(self):
        self._array.reopen()

    def size(self):
        return self._array.size


class Group:
    """Class for opening TileDB group metadata, arrays, and attributes.

    Parameters:
        uri: Uniform resource identifier for TileDB group or array.
        mode: Mode the array and metadata objects are opened in. Either read 'r' or
            write 'w' mode.
        key: If not ``None``, encryption key, or dictionary of encryption keys, to
            decrypt arrays.
        timestamp: If not ``None``, timestamp to open the group metadata and array at.
        ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
    """

    @classmethod
    def create(
        cls,
        uri: str,
        group_schema: GroupSchema,
        key: Optional[Union[Dict[str, str], str]] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Create the TileDB group and arrays from a :class:`GroupSchema`.

        Parameters:
            uri: Uniform resource identifier for TileDB group or array.
            group_schema: Schema that defines the group to be created.
            key: If not ``None``, encryption key, or dictionary of encryption keys to
                decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        tiledb.group_create(uri, ctx)
        if group_schema.metadata_schema is not None:
            tiledb.Array.create(
                _get_array_uri(uri, _METADATA_ARRAY),
                group_schema.metadata_schema,
                key.get(_METADATA_ARRAY) if isinstance(key, dict) else key,
                ctx,
            )
        for array_name, array_schema in group_schema.items():
            tiledb.Array.create(
                _get_array_uri(uri, array_name),
                array_schema,
                key.get(array_name) if isinstance(key, dict) else key,
                ctx,
            )

    __slots__ = [
        "_array",
        "_attr",
        "_ctx",
        "_key",
        "_metadata_array",
        "_mode",
        "_schema",
        "_uri",
    ]

    def __init__(
        self,
        uri: str,
        array: Optional[str] = None,
        attr: Optional[str] = None,
        mode: str = "r",
        key: Optional[Union[Dict[str, str], str]] = None,
        timestamp: Optional[int] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Constructs a new :class:`Group`."""
        self._uri = uri
        self._mode = mode
        self._key = key
        self._ctx = ctx
        self._schema = GroupSchema.load_group(uri, ctx, key)
        self._metadata_array = self._get_metadata_array(timestamp)
        self._attr = attr
        if array is None and attr is not None:
            group_schema = GroupSchema.load_group(uri, ctx, key)
            arrays = group_schema.get_attribute_arrays(attr)
            if len(arrays) != 1:
                raise ValueError(
                    f"Failed to open a single array with attribute {attr}. There is "
                    f"{len(arrays)} with attribute named {attr}."
                )
            array = arrays[0]
        self._array = (
            None
            if array is None
            else tiledb.open(
                _get_array_uri(self._uri, array),
                mode=self._mode,
                key=key.get(array) if isinstance(key, dict) else key,
                attr=attr,
                config=None,
                timestamp=timestamp,
                ctx=ctx,
            )
        )

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    def _get_metadata_array(self, timestamp: Optional[int]) -> tiledb.Array:
        return (
            None
            if self._schema.metadata_schema is None
            else tiledb.open(
                uri=_get_array_uri(self._uri, _METADATA_ARRAY),
                mode=self._mode,
                key=self.metadata_key,
                timestamp=timestamp,
                ctx=self._ctx,
            )
        )

    def close(self):
        """Closes this Group, flushing all buffered data."""
        if self._metadata_array is not None:
            self._metadata_array.close()
        if self._array is not None:
            self._array.close()

    @property
    def array(self) -> tiledb.Array:
        """The opened array, or ``None`` if no array was opened."""
        return self._array

    @property
    def array_metadata(self) -> ArrayMetadata:
        """Metadata object for the array.

        Raises:
            RuntimeError: No array was opened.
        """
        if self._array is None:
            raise RuntimeError(
                "Cannot access group array metadata: no array was opened."
            )
        return ArrayMetadata(self._array.meta)

    @property
    def attribute_metadata(self) -> Dict[str, AttributeMetadata]:
        """Metadata object for the attribute metadata.

        Raises:
            RuntimeError: No array was opened.
            RuntimeError: Array has multiple open attributes.
        """
        if self._array is None:
            raise RuntimeError("Cannot access attribute metadata: no array was opened.")
        if self._attr is not None:
            return {self._attr: AttributeMetadata(self._array.meta, self._attr)}
        return {
            attr.name: AttributeMetadata(self._array.meta, attr.name)
            for attr in self._array.schema
        }

    def create_metadata_array(self):
        """Creates a metadata array for this group.

        This routine will create a metadata array for the group. An error will be raised
        if the metadata array already exists.

        The user must either close the group and open it again, or just use
        :meth:`reopen` without closing to read and write to the metadata array after it
        is created.

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
            _get_array_uri(self._uri, _METADATA_ARRAY),
            self._schema.metadata_schema,
            self.metadata_key,
            self._ctx,
        )

    @property
    def has_metadata_array(self) -> bool:
        """Flag that is true if there a metadata array for storing group metadata."""
        return self._metadata_array is not None

    @property
    def meta(self) -> Optional[tiledb.Metadata]:
        """Metadata object for the group, or None if no array to store group
        metadata in."""
        if self._metadata_array is None:
            return None
        return self._metadata_array.meta

    @property
    def metadata_key(self) -> Optional[str]:
        """Key for the metadata array."""
        return (
            self._key.get(_METADATA_ARRAY) if isinstance(self._key, dict) else self._key
        )

    def reopen(self, timestamp: Optional[int] = None):
        """Reopens this group.

        This is useful when the Group is updated after it was opened.
        To sync-up with the updates, the user must either close the array and open
        again, or just use ``reopen()`` without closing. ``reopen`` will be generally
        faster than a close-then-open.
        """
        self._schema = GroupSchema.load_group(
            self._uri,
            self._ctx,
            self._key,
        )
        if self._metadata_array is None:
            self._metadata_array = self._get_metadata_array(timestamp)
        else:
            self._metadata_array.reopen()
        if self._array is not None:
            self._array.reopen()


class GroupSchema(Mapping):
    """Schema for a TileDB group.

    Parameters:
        array_schemas: A collection of (name, ArraySchema) tuples for Arrays that belong
            to this group.
        metadata_schema: If not None, a schema for the group metadata array.
    """

    @classmethod
    def load_group(
        cls,
        uri: str,
        ctx: Optional[tiledb.Ctx] = None,
        key: Optional[Union[Dict[str, str], str]] = None,
    ):
        """Load a schema for a TileDB group from a TileDB URI.

        Parameters:
            uri: uniform resource identifier for the TileDB group.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            key: If not ``None``, encryption key, or dictionary of encryption keys, to
                decrypt arrays.
        """
        metadata_schema = None
        if tiledb.object_type(uri, ctx) != "group":
            raise ValueError(
                f"Failed to load the group schema. Provided uri '{uri}' is not a "
                f"valid TileDB group."
            )
        vfs = tiledb.VFS(ctx=ctx)
        array_schemas = {}
        for item_uri in vfs.ls(uri):
            if not tiledb.object_type(item_uri) == "array":
                continue
            array_name = item_uri.split("/")[-1]
            local_key = key.get(array_name) if isinstance(key, dict) else key
            if array_name == _METADATA_ARRAY:
                metadata_schema = tiledb.ArraySchema.load(item_uri, ctx, local_key)
            else:
                array_schemas[array_name] = tiledb.ArraySchema.load(
                    item_uri,
                    ctx,
                    local_key,
                )
        return cls(array_schemas, metadata_schema)

    __slots__ = [
        "_array_schema_table",
        "_metadata_schema",
    ]

    def __init__(
        self,
        array_schemas: Optional[Dict[str, tiledb.ArraySchema]] = None,
        metadata_schema: Optional[tiledb.ArraySchema] = None,
    ):
        """Constructs a :class:`GroupSchema`.

        Raises:
            ValueError: ArraySchema has duplicate names.
        """
        self._metadata_schema = metadata_schema
        if array_schemas is None:
            self._array_schema_table = {}
        else:
            self._array_schema_table = dict(array_schemas)

    def __eq__(self, other: Any):
        if not isinstance(other, GroupSchema):
            return False
        if len(self) != len(other):
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
        return len(self._array_schema_table)

    def __repr__(self) -> str:
        """Returns the object representation of this GroupSchema in string form."""
        output = StringIO()
        output.write("  GroupSchema (\n")
        for name, schema in self.items():
            output.write(f"{name} {repr(schema)}")
        output.write(")\n")
        return output.getvalue()

    def check(self):
        """Checks the correctness of each array in the :class:`GroupSchema`.

        Raises:
            tiledb.TileDBError: An ArraySchema in the GroupSchema is invalid.
        """
        for schema in self._array_schema_table.values():
            schema.check()
        if self._metadata_schema is not None:
            self._metadata_schema.check()

    def get_attribute_arrays(self, attribute_name: str) -> List[str]:
        """Returns a list of the names of all arrays with a matching attribute.

        Parameter:
            attribute_name: Name of the attribute to look up arrays for.

        Returns:
            A tuple of the name of all arrays with a matching attribute, or `None` if no
                such array.
        """
        arrays = []
        for array_name, array_schema in self._array_schema_table.items():
            for attr in array_schema:
                if attribute_name == attr.name:
                    arrays.append(array_name)
                    break
        return arrays

    @property
    def metadata_schema(self) -> Optional[tiledb.ArraySchema]:
        """ArraySchema for the group-level metadata."""
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


@dataclass(frozen=True)
class SharedDimension(Generic[DType]):
    """A class for a shared one-dimensional dimension.

    Parameters:
        name: Name of the :class:`SharedDimension`.
        domain: The (inclusive) interval on which the :class:`SharedDimension` is
            valid.
        data_type: Numpy dtype of the dimension values and domain.
    """

    name: str
    domain: Tuple[Optional[DType], Optional[DType]]
    dtype: np.dtype

    @classmethod
    def from_tiledb_dim(cls, dimension: tiledb.Dim):
        """Converts a tiledb.Dim to a SharedDimension

        Parameters:
            dimension: TileDB dimension that will be used to create the shared
                dimension.

        Returns:
            A :class:`SharedDimension` that has the name, domain, and dtype of the
                tiledb.Dim.
        """
        return cls(dimension.name, dimension.domain, dimension.dtype)

    def __post_init__(self):
        """Verify validity of SharedDimension after initializeation

        Raises:
            ValueError: Name contains reserved character '.'.
        """
        if "." in self.name:
            raise ValueError(
                f"Invalid name {self.name}. Cannot have '.' in dimension name."
            )
