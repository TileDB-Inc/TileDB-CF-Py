# Copyright 2020 TileDB Inc.
# Licensed under the MIT License.
"""Input and output routines for the TileDB-CF data model."""

from __future__ import annotations

import os.path
import typing
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


class DataspaceArray:
    def __init__(
        self,
        uri,
        mode: str = "r",
        attr: Optional[str] = None,
        key: Optional[str] = None,
        timestamp: Optional[int] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        self._attr_map = DataspaceAttributeMap()
        self._attr_map.add_array(tiledb.ArraySchema.load(uri, ctx, key))
        self._attr = None if attr is None else self._attr_map[attr]
        self._array = tiledb.open(
            uri=uri,
            mode=mode,
            attr=self._attr,
            key=key,
            timestamp=timestamp,
            ctx=ctx,
        )

    def __array__(self, dtype=None):
        if self._array.sparse:
            raise NotImplementedError(
                "Opening a sparse TileDB array directly as a Numpy array is not yet"
                " implemented."
            )
        return np.asarray(self._array, dtype)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    @property
    def base(self) -> tiledb.Array:
        """Direct access to the base TileDB array api."""
        return self._array

    @property
    def attribute_metadata(self) -> Dict[str, AttributeMetadata]:
        return (
            {self._attr: AttributeMetadata(self._array.meta, self._attr)}
            if self._attr is not None
            else {
                attr_key: AttributeMetadata(self._array.meta, attr_key)
                for attr_key in self._attr_map.keys()
            }
        )

    def close(self):
        self._array.close()

    @property
    def meta(self) -> ArrayMetadata:
        return ArrayMetadata(self._array.meta)

    def reopen(self):
        self._array.reopen()

    def size(self):
        return self._array.size


class DataspaceAttributeMap(Mapping):

    __slots__ = ["_attribute_map"]

    def __init__(self):
        self._attribute_map: Dict[str, Tuple[str, Optional[str]]] = dict()

    def __getitem__(self, key: str) -> str:
        return self._attribute_map[key][0]

    def __iter__(self) -> Iterator[str]:
        return self._attribute_map.__iter__()

    def __len__(self) -> int:
        return len(self._attribute_map)

    def add_array(
        self,
        array_schema: tiledb.ArraySchema,
        array_name: Optional[str] = None,
    ):
        for attr in array_schema:
            self.add_attr(attr.name, array_name)

    def add_attr(self, attr_name: str, array_name: Optional[str] = None):
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
            self._attribute_map[attr_key] = (attr_name, None)
        else:
            self._attribute_map[attr_key] = (attr_name, array_name)

    def get_array(self, attr_key) -> str:
        (_, array) = self._attribute_map[attr_key]
        if array is None:
            raise KeyError(
                f"Failed to get array name. No array name stored for key {attr_key}."
            )
        return array


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
        "_ctx",
        "_key",
        "_schema",
        "_uri",
    ]

    def __init__(
        self,
        uri: str,
        key: Optional[Union[Dict[str, str], str]] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Constructs a new :class:`Group`."""
        self._uri = uri
        self._key = key
        self._ctx = ctx
        self._schema = GroupSchema.load_group(self._uri, self._ctx, self._key)

    def array(
        self,
        array: Optional[str] = None,
        attr: Optional[str] = None,
        mode: str = "r",
        timestamp: Optional[int] = None,
    ) -> tiledb.Array:
        return tiledb.open(
            self.array_uri(array, attr),
            mode=mode,
            key=self.array_key(array, attr),
            attr=attr,
            config=None,
            timestamp=timestamp,
            ctx=self._ctx,
        )

    def array_key(
        self, array: Optional[str] = None, attr: Optional[str] = None
    ) -> Optional[str]:
        if array is None:
            if attr is None:
                raise ValueError(
                    "Failed to find array key. No array or attribute name provided."
                )
            array = self._schema.get_array_from_attr(attr)
        return self._key.get(array) if isinstance(self._key, dict) else self._key

    @property
    def array_names(self) -> Iterator[Optional[str]]:
        return self._schema.keys()

    @property
    def array_schemas(self) -> Iterator[tiledb.ArraySchema]:
        return self._schema.values()

    def array_uri(self, array: Optional[str] = None, attr: Optional[str] = None):
        if array is None:
            if attr is None:
                raise ValueError(
                    "Failed to find array URI. No array or attribute name provided."
                )
            array = self._schema.get_array_from_attr(attr)
        return _get_array_uri(self._uri, array)

    def create_metadata_array(self):
        """Creates a metadata array for this group.

        Raises:
            tiledb.TileDBError: Array already exists.
        """
        if self._schema.metadata_schema is None:
            self._schema.set_default_metadata_schema()
        tiledb.Array.create(
            self.array_uri(_METADATA_ARRAY),
            self._schema.metadata_schema,
            self.metadata_key,
            self._ctx,
        )

    def has_metadata_array(self) -> bool:
        return self._schema.metadata_schema is not None

    def metadata_array(
        self,
        mode: str = "r",
        timestamp: Optional[int] = None,
    ) -> tiledb.Array:
        return self.array(_METADATA_ARRAY, None, mode, timestamp)

    @property
    def metadata_key(self) -> Optional[str]:
        """Key for the metadata array."""
        return self.array_key(_METADATA_ARRAY)

    @property
    def schema(self) -> GroupSchema:
        return self._schema

    def reload(self):
        self._schema = GroupSchema.load_group(self._uri, self._ctx, self._key)


class DataspaceGroup(Group):
    """Class for opening a TileDB Dataspace."""

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
        if not isinstance(group_schema, DataspaceGroupSchema):
            group_schema = DataspaceGroupSchema.from_group_schema(group_schema)
        super().create(uri, group_schema, key, ctx)

    def __init__(
        self,
        uri: str,
        key: Optional[Union[Dict[str, str], str]] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        self._uri = uri
        self._key = key
        self._ctx = ctx
        self._schema = DataspaceGroupSchema.load_group(uri, ctx, key)

    def dataspace_array(
        self,
        array: Optional[str] = None,
        attr: Optional[str] = None,
        mode: str = "r",
        timestamp: Optional[int] = None,
    ) -> DataspaceArray:
        return DataspaceArray(
            self.array_uri(array, attr),
            mode=mode,
            attr=attr,
            key=self.array_key(array, attr),
            timestamp=timestamp,
            ctx=self._ctx,
        )

    def dataspace_metadata_array(
        self,
        mode: str = "r",
        timestamp: Optional[int] = None,
    ) -> DataspaceArray:
        return self.dataspace_array(_METADATA_ARRAY, None, mode, timestamp)

    def has_attr(self, name: str) -> bool:
        return self._schema.has_attr

    @property
    def nattr(self) -> int:
        return self._schema.nattr

    @property
    def ndim(self) -> int:
        return self._schema.ndim


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
        array_schemas: Optional[typing.Mapping[str, tiledb.ArraySchema]] = None,
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
            self._array_schema_table = {
                key: value for (key, value) in array_schemas.items()
            }

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

    def get_array_from_attr(self, attr: str) -> str:
        arrays = self.get_attr_arrays(attr)
        if len(arrays) != 1:
            raise ValueError(
                f"Failed to a single array with attribute {attr}. "
                f"There is {len(arrays)} with attribute named {attr}."
            )
        return arrays[0]

    def get_attr_arrays(self, attr: str) -> List[str]:
        """Returns a list of the names of all arrays with a matching attribute.

        Parameter:
            attribute_name: Name of the attribute to look up arrays for.

        Returns:
            A list of the name of all arrays with a matching attribute.
        """
        arrays = []
        for array_name, array_schema in self._array_schema_table.items():
            for tiledb_attr in array_schema:
                if attr == tiledb_attr.name:
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


class DataspaceGroupSchema(GroupSchema):
    """Schema for a TileDB dataspce group.

    Parameters:
        array_schemas: A collection of (name, ArraySchema) tuples for Arrays that belong
             to this group.
        metadata_schema: If not None, a schema for the group metadata array.
    """

    @classmethod
    def from_group_schema(cls, group_schema: GroupSchema):
        if isinstance(group_schema, cls):
            return group_schema
        return cls(group_schema, group_schema.metadata_schema)

    __slots__ = ["_attr_map", "_dimension_map"]

    def __init__(
        self,
        array_schemas: Optional[typing.Mapping[str, tiledb.ArraySchema]] = None,
        metadata_schema: Optional[tiledb.ArraySchema] = None,
    ):
        super().__init__(array_schemas, metadata_schema)
        self._dimension_map: Dict[str, SharedDimension] = {}
        self._attr_map = DataspaceAttributeMap()
        if array_schemas is not None:
            for array_name, array_schema in array_schemas.items():
                self._add_array(array_schema)
                self._attr_map.add_array(array_schema, array_name)

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

    def get_array_from_attr(self, attr: str) -> str:
        return self._attr_map.get_array(attr)

    def get_attr_arrays(self, attr: str) -> List[str]:
        """Returns a list of the names of all arrays with a matching attribute.

        Parameter:
            attr: Name of the attribute to look up arrays for.

        Returns:
            A tuple of the name of all arrays with a matching attribute, or `None` if no
            such array.
        """
        if attr not in self._attr_map:
            return []
        return [
            self._attr_map.get_array(attr),
        ]

    @property
    def dim_names(self) -> List[str]:
        """A list of names of dimensions in this :class:`DataspaceMap`."""
        return list(self._dimension_map.keys())

    def has_attr(self, name: str) -> bool:
        return name in self._attr_map

    @property
    def nattr(self) -> int:
        return len(self._attr_map)

    @property
    def ndim(self) -> int:
        return len(self._dimension_map)


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
