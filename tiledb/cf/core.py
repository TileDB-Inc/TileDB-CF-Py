# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Input and output routines for the TileDB-CF data model."""

from __future__ import annotations

import os.path
from collections import defaultdict
from collections.abc import Mapping, MutableMapping
from io import StringIO
from typing import Any, Dict, Iterator, List, Optional, TypeVar, Union

import numpy as np

import tiledb

DType = TypeVar("DType", covariant=True)
METADATA_ARRAY_NAME = "__tiledb_group"
ATTR_METADATA_FLAG = "__tiledb_attr."


def _array_schema_html(schema: tiledb.ArraySchema) -> str:
    """Returns a HTML representation of a TileDB array."""
    output = StringIO()
    output.write("<ul>\n")
    output.write("<li>\n")
    output.write("Domain\n")
    output.write("<table>\n")
    for i in range(schema.domain.ndim):
        output.write(
            f'<tr><td style="text-align: left;">{repr(schema.domain.dim(i))}</td>'
            f"</tr>\n"
        )
    output.write("</table>\n")
    output.write("</li>\n")
    output.write("<li>\n")
    output.write("Attributes\n")
    output.write("<table>\n")
    for i in range(schema.nattr):
        output.write(
            f'<tr><td style="text-align: left;">{repr(schema.attr(i))}</td></tr>\n'
        )
    output.write("</table>\n")
    output.write("</li>\n")
    output.write("<li>\n")
    output.write("Array properties")
    output.write(
        f"<table>\n"
        f'<tr><td style="text-align: left;">cell_order={schema.cell_order}</td></tr>\n'
        f'<tr><td style="text-align: left;">tile_order={schema.tile_order}</td></tr>\n'
        f'<tr><td style="text-align: left;">capacity={schema.capacity}</td></tr>\n'
        f'<tr><td style="text-align: left;">sparse={schema.sparse}</td></tr>\n'
    )
    if schema.sparse:
        output.write(
            f'<tr><td style="text-align: left;">allows_duplicates'
            f"={schema.allows_duplicates}</td></tr>\n"
        )
    output.write(
        f'<tr><td style="text-align: left">coords_filters={schema.coords_filters}'
        f"</td>\n"
    )
    output.write("</table>\n")
    output.write("</li>\n")
    output.write("</ul>\n")
    return output.getvalue()


def _get_array_uri(group_uri: str, array_name: str, is_virtual: bool) -> str:
    """Returns a URI for an array with name ``array_name`` inside a group at URI
        ``group_uri``.

    Parameters:
        group_uri: URI of the group containing the array
        array_name: name of the array
        is_virtual: If ``True``, return the URI for an array in a virtual group.
             Otherwise, return the URI for an array inside a "standard" group.

    Returns:
        Array URI of an array with name ``array_name`` inside a group at URI
            ``group_uri``.
    """
    if is_virtual:
        return f"{group_uri}_{array_name}"
    return os.path.join(group_uri, array_name)


def _get_metadata_array_uri(group_uri: str, is_virtual: bool) -> str:
    """Returns a URI for an array with name ``array_name`` inside a group at URI
        ``group_uri``.

    Parameters:
        group_uri: URI of the group containing the array
        array_name: name of the array
        is_virtual: If ``True``, return the URI for the metadata array in a virtual
             group. Otherwise, return the URI for an array inside a "standard" group.

    Returns:
        Array URI of an array with name ``array_name`` inside a group at URI
            ``group_uri``.
    """
    if is_virtual:
        return group_uri
    return os.path.join(group_uri, METADATA_ARRAY_NAME)


def _get_array_key(
    key: Optional[Union[Dict[str, str], str]], array_name
) -> Optional[str]:
    """Returns a key for the array with name ``array_name``.

    Parameters:
        key: If not ``None``, encryption key, or dictionary of encryption keys, to
            decrypt arrays.
        array_name: Name of the array to decrypt.

    Returns:
       Key for the array with name ``array_name``.
    """
    return key.get(array_name) if isinstance(key, dict) else key


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
        return key  # pragma: no cover

    def _from_tiledb_key(self, tiledb_key: str) -> Optional[str]:
        """Map an internal tiledb key to an external user metadata key.

        Returns:
            The external user metadata key corresponding to `tiledb_key`,
            or None if there is no such corresponding key.
        """
        return tiledb_key  # pragma: no cover


class ArrayMetadata(Metadata):
    """Class for accessing array-related metadata from a TileDB metadata object.

    This class provides a way for accessing the TileDB array metadata that excludes
    attribute-specific metadata.

    Parameters:
        metadata (tiledb.Metadata): TileDB array metadata object for the desired array.
    """

    def _to_tiledb_key(self, key: str) -> str:
        if key.startswith(ATTR_METADATA_FLAG):
            raise KeyError("Key is reserved for attribute metadata.")
        return key

    def _from_tiledb_key(self, tiledb_key: str) -> Optional[str]:
        if not tiledb_key.startswith(ATTR_METADATA_FLAG):
            return tiledb_key
        return None


class AttrMetadata(Metadata):
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
        self._key_prefix = ATTR_METADATA_FLAG + attr_name + "."

    def _to_tiledb_key(self, key: str) -> str:
        return self._key_prefix + key

    def _from_tiledb_key(self, tiledb_key: str) -> Optional[str]:
        if tiledb_key.startswith(self._key_prefix):
            return tiledb_key[len(self._key_prefix) :]
        return None


class Group:
    """Class for opening TileDB group metadata, arrays, and attributes.

    Parameters:
        uri: Uniform resource identifier for TileDB group or array.
        mode: Mode the array and metadata objects are opened in. Either read 'r' or
            write 'w' mode.
        key: If not ``None``, encryption key, or dictionary of encryption keys, to
            decrypt arrays.
        timestamp: If not ``None``, timestamp to open the group metadata and array at.
        array: If not ``None``, open the array in the group with this name.
        attr: If not ``None``, open one attribute of the group; indexing a dense array
            will return a Numpy ndarray directly rather than a dictionary. If ``array``
            is specified, the attribute must be inside the specified array. If ``array``
            is not specified, there must be only one attribute in the group with this
            name.
        ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
    """

    @classmethod
    def create(
        cls,
        uri: str,
        group_schema: GroupSchema,
        key: Optional[Union[Dict[str, str], str]] = None,
        ctx: Optional[tiledb.Ctx] = None,
        is_virtual: bool = False,
    ):
        """Create the TileDB group and arrays from a :class:`GroupSchema`.

        Parameters:
            uri: Uniform resource identifier for TileDB group or array.
            group_schema: Schema that defines the group to be created.
            key: If not ``None``, encryption key, or dictionary of encryption keys to
                decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            is_virtual: If ``True``, create a virtual group using ``uri`` as the name
                for the group metadata array. All other arrays will be named using the
                convention ``{uri}_{array_name}`` where ``array_name`` is the name of
                the array.
        """
        if not is_virtual:
            tiledb.group_create(uri, ctx)
        if group_schema.metadata_schema is not None:
            tiledb.Array.create(
                _get_metadata_array_uri(uri, is_virtual),
                group_schema.metadata_schema,
                _get_array_key(key, METADATA_ARRAY_NAME),
                ctx,
            )
        for array_name, array_schema in group_schema.items():
            tiledb.Array.create(
                _get_array_uri(uri, array_name, is_virtual),
                array_schema,
                _get_array_key(key, array_name),
                ctx,
            )

    @classmethod
    def create_virtual(
        cls,
        uri: str,
        group_schema: GroupSchema,
        key: Optional[Union[Dict[str, str], str]] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Create the arrays from a :class:`GroupSchema`.

        Parameters:
            uri: Uniform resource identifier for TileDB group or array.
            group_schema: Schema that defines the group to be created.
            key: If not ``None``, encryption key, or dictionary of encryption keys to
                decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        cls.create(uri, group_schema, key, ctx, is_virtual=True)

    def __init__(
        self,
        uri: str,
        mode: str = "r",
        key: Optional[Union[Dict[str, str], str]] = None,
        timestamp: Optional[int] = None,
        array: Optional[str] = None,
        attr: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Constructs a new :class:`Group`."""
        group_schema = GroupSchema.load(uri, ctx, key)
        self._metadata_array = (
            None
            if group_schema.metadata_schema is None
            else tiledb.open(
                uri=_get_metadata_array_uri(uri, False),
                mode=mode,
                key=_get_array_key(key, METADATA_ARRAY_NAME),
                timestamp=timestamp,
                ctx=ctx,
            )
        )
        self._attr = attr
        if array is None and attr is not None:
            array_names = group_schema.arrays_with_attr(attr)
            if array_names is None:
                raise KeyError(f"No attribute with name '{attr}' found.")
            if len(array_names) > 1:
                raise ValueError(
                    f"The array must be specified when opening an attribute that "
                    f"exists in multiple arrays in a group. Arrays with attribute "
                    f"'{attr}' include: {array_names}."
                )
            array = array_names[0]
        self._array = (
            None
            if array is None
            else tiledb.open(
                _get_array_uri(uri, array, False),
                mode=mode,
                key=_get_array_key(key, array),
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

    def close(self):
        """Closes this Group, flushing all buffered data."""
        if self._metadata_array is not None:
            self._metadata_array.close()
        if self._array is not None:
            self._array.close()

    @property
    def array(self) -> tiledb.Array:
        """The opened array, or ``None`` if no array was opened.

        Raises:
            RuntimeError: No array was opened.
        """
        if self._array is None:
            raise RuntimeError("Cannot access group array: no array was opened.")
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
    def attr_metadata(self) -> AttrMetadata:
        """Metadata object for the attribute metadata.

        Raises:
            RuntimeError: No array was opened.
            RuntimeError: Array has multiple open attributes.
        """
        if self._array is None:
            raise RuntimeError("Cannot access attribute metadata: no array was opened.")
        if self._attr is not None:
            return AttrMetadata(self._array.meta, self._attr)
        if self._array.nattr == 1:
            return AttrMetadata(self._array.meta, 0)
        raise RuntimeError(
            "Cannot access attribute metadata. Array has multiple open attributes; use "
            "get_attr_metadata to specify which attribute to open."
        )

    def get_attr_metadata(self, attr: Union[str, int]) -> AttrMetadata:
        """Returns attribute metadata object corresponding to requested attribute.

        Parameters:
            attr: Name or index of the requested array attribute.

        Raises:
            RuntimeError: No array was opened.
        """
        if self._array is None:
            raise RuntimeError("Cannot access attribute metadata: no array was opened.")
        return AttrMetadata(self._array.meta, attr)

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


class VirtualGroup(Group):
    """Class for opening TileDB metadata, arrays, and attributes of a virtual group.

    Parameters:
        array_uris: Mapping from array names to array uniform resource identifiers.
        mode: Mode the array and metadata objects are opened in. Either read 'r' or
            write 'w' mode.
        key: If not ``None``, encryption key, or dictionary of encryption keys, to
            decrypt arrays.
        timestamp: If not ``None``, timestamp to open the group metadata and array at.
        array: If not ``None``, open the array with this name.
        attr: If not ``None``, open one attribute of the group; indexing a dense array
            will return a Numpy ndarray directly rather than a dictionary. If ``array``
            is specified, the attribute must be inside the specified array. If ``array``
            is not specified, there must be only one attribute in the group with this
            name.
        ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
    """

    def __init__(
        self,
        array_uris: Dict[str, str],
        mode: str = "r",
        key: Optional[Union[Dict[str, str], str]] = None,
        timestamp: Optional[int] = None,
        array: Optional[str] = None,
        attr: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Constructs a new :class:`Group`."""
        self._metadata_array = (
            tiledb.open(
                uri=array_uris[METADATA_ARRAY_NAME],
                mode=mode,
                key=_get_array_key(key, METADATA_ARRAY_NAME),
                timestamp=timestamp,
                ctx=ctx,
            )
            if METADATA_ARRAY_NAME in array_uris
            else None
        )
        self._attr = attr
        if array is None and attr is not None:
            array_names = []
            for array_name, array_uri in array_uris.items():
                if array_name == METADATA_ARRAY_NAME:
                    continue
                array_schema = tiledb.ArraySchema.load(
                    array_uri, ctx, _get_array_key(key, array_name)
                )
                if array_schema.has_attr(attr):
                    array_names.append(array_name)
            if len(array_names) == 0:
                raise KeyError(f"No attribute with name '{attr}' found.")
            if len(array_names) > 1:
                raise ValueError(
                    f"The array must be specified when opening an attribute that "
                    f"exists in multiple arrays in a group. Arrays with attribute "
                    f"'{attr}' include: {array_names}."
                )
            array = array_names[0]
        self._array = (
            None
            if array is None
            else tiledb.open(
                uri=array_uris[array],
                mode=mode,
                key=_get_array_key(key, array),
                attr=attr,
                config=None,
                timestamp=timestamp,
                ctx=ctx,
            )
        )


class GroupSchema(Mapping):
    """Schema for a TileDB group.

    Parameters:
        array_schemas: A collection of (name, ArraySchema) tuples for Arrays that belong
            to this group.
        metadata_schema: If not ``None``, a schema for the group metadata array.
        use_default_metadata_schema: If ``True`` and ``metadata_schema=None`` a default
            schema will be created for the metadata array.
        ctx: TileDB Context used for generatign default metadata schema.
    """

    @classmethod
    def load(
        cls,
        uri: str,
        ctx: Optional[tiledb.Ctx] = None,
        key: Optional[Union[Dict[str, str], str]] = None,
    ):
        """Load a schema for a TileDB group from a TileDB URI.

        Parameters:
            uri: uniform resource identifier for the TileDB group
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
            local_key = _get_array_key(key, array_name)
            if array_name == METADATA_ARRAY_NAME:
                metadata_schema = tiledb.ArraySchema.load(item_uri, ctx, local_key)
            else:
                array_schemas[array_name] = tiledb.ArraySchema.load(
                    item_uri,
                    ctx,
                    local_key,
                )
        return cls(array_schemas, metadata_schema, False)

    @classmethod
    def load_virtual(
        cls,
        array_uris: Dict[str, str],
        ctx: Optional[tiledb.Ctx] = None,
        key: Optional[Union[Dict[str, str], str]] = None,
    ):
        """Load a schema for a TileDB group from a TileDB URI.

        Parameters:
            array_uris: Mapping from array names to array uniform resource identifiers.
            metadata_uri: Array uniform resource identifier for array where metadata is
                stored.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            key: If not ``None``, encryption key, or dictionary of encryption keys, to
                decrypt arrays.
        """
        array_schemas = {
            array_name: tiledb.ArraySchema.load(
                array_uri, ctx, _get_array_key(key, array_name)
            )
            for array_name, array_uri in array_uris.items()
        }
        metadata_schema = (
            array_schemas.pop(METADATA_ARRAY_NAME)
            if METADATA_ARRAY_NAME in array_schemas
            else None
        )
        return cls(array_schemas, metadata_schema, False)

    def __init__(
        self,
        array_schemas: Optional[Dict[str, tiledb.ArraySchema]] = None,
        metadata_schema: Optional[tiledb.ArraySchema] = None,
        use_default_metadata_schema: bool = True,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Constructs a :class:`GroupSchema`.

        Raises:
            ValueError: ArraySchema has duplicate names.
        """
        if metadata_schema is None and use_default_metadata_schema:
            self._metadata_schema = tiledb.ArraySchema(
                domain=tiledb.Domain(
                    tiledb.Dim(name="dim", domain=(0, 0), dtype=np.int32, ctx=ctx)
                ),
                attrs=[tiledb.Attr(name="attr", dtype=np.int32, ctx=ctx)],
                sparse=False,
            )
        else:
            self._metadata_schema = metadata_schema
        if array_schemas is None:
            self._array_schema_table = {}
        else:
            self._array_schema_table = dict(array_schemas)
        self._attr_to_arrays: Dict[str, List[str]] = defaultdict(list)
        for (schema_name, schema) in self._array_schema_table.items():
            for attr in schema:
                attr_name = attr.name
                self._attr_to_arrays[attr_name].append(schema_name)

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
        """Returns the :class:`Arrayschema` with the name given by `schema_name`.

        Parameters:
            schema_name: Name of the ArraySchema to be returned.

        Returns:
            ArraySchema with name `schema_name`.
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
        output.write("GroupSchema:\n")
        if self._metadata_schema is not None:
            output.write(f"Group metadata schema: {repr(self._metadata_schema)}")
        for name, schema in self.items():
            output.write(f"'{name}': {repr(schema)}")
        return output.getvalue()

    def _repr_html_(self) -> str:
        """Returns the object representation of this GroupSchame as HTML."""
        output = StringIO()
        output.write("<section>\n")
        output.write(f"<h3>{self.__class__.__name__}</h3>\n")
        if self._metadata_schema is not None:
            output.write("<details>\n")
            output.write("<summary>ArraySchema for Group Metadata Array</summary>\n")
            output.write(_array_schema_html(self._metadata_schema))
            output.write("</details>\n")
        for name, schema in self.items():
            output.write("<details>\n")
            output.write(f"<summary>ArraySchema <em>{name}</em></summary>\n")
            output.write(_array_schema_html(schema))
            output.write("</details>\n")
        output.write("</section>\n")
        return output.getvalue()

    def check(self):
        """Checks the correctness of each array in the GroupSchema.

        Raises:
            tiledb.TileDBError: An ArraySchema in the GroupSchema is invalid.
            RuntimeError: A shared :class:`tiledb.Dim` fails to match the definition
                from the GroupSchema.
        """
        for schema in self._array_schema_table.values():
            schema.check()
        if self._metadata_schema is not None:
            self._metadata_schema.check()

    def arrays_with_attr(self, attr_name: str) -> Optional[List[str]]:
        """Returns a tuple of the names of all arrays with a matching attribute.

        Parameter:
            attr_name: Name of the attribute to look up arrays for.

        Returns:
            A tuple of the name of all arrays with a matching attribute, or `None` if no
                such array.
        """
        return self._attr_to_arrays.get(attr_name)

    def has_attr(self, attr_name: str) -> bool:
        return attr_name in self._attr_to_arrays

    @property
    def metadata_schema(self) -> Optional[tiledb.ArraySchema]:
        """ArraySchema for the group-level metadata."""
        return self._metadata_schema
