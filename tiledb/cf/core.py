# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Classes for additional group and metadata support useful for the TileDB-CF data
model."""

from __future__ import annotations

import os.path
import warnings
from collections import defaultdict
from collections.abc import Mapping, MutableMapping
from io import StringIO
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

import tiledb

from ._utils import check_valid_group

DType = TypeVar("DType", covariant=True)
ATTR_METADATA_FLAG = "__tiledb_attr."
DIM_METADATA_FLAG = "__tiledb_dim."


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
    output.write("</table>\n")
    output.write("</li>\n")
    output.write("</ul>\n")
    return output.getvalue()


def _get_array_uri(group_uri: str, array_name: str) -> str:
    """Returns a URI for an array with name ``array_name`` inside a group at URI
        ``group_uri``.

    Parameters:
        group_uri: URI of the group containing the array
        array_name: name of the array

    Returns:
        Array URI of an array with name ``array_name`` inside a group at URI
            ``group_uri``.
    """
    return os.path.join(group_uri, array_name)


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
        """Map an external user metadata key to an internal tiledb key."""
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
    attribute and dimension specific metadata.

    Parameters:
        metadata (tiledb.Metadata): TileDB array metadata object for the desired array.
    """

    def _to_tiledb_key(self, key: str) -> str:
        if key.startswith(ATTR_METADATA_FLAG):
            raise KeyError("Key is reserved for attribute metadata.")
        if key.startswith(DIM_METADATA_FLAG):
            raise KeyError("Key is reserved for dimension metadata.")
        return key

    def _from_tiledb_key(self, tiledb_key: str) -> Optional[str]:
        if not (
            tiledb_key.startswith(ATTR_METADATA_FLAG)
            or tiledb_key.startswith(DIM_METADATA_FLAG)
        ):
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
            raise KeyError(f"Attribute `{attr}` not found in array.") from err
        self._key_prefix = ATTR_METADATA_FLAG + attr_name + "."

    def _to_tiledb_key(self, key: str) -> str:
        return self._key_prefix + key

    def _from_tiledb_key(self, tiledb_key: str) -> Optional[str]:
        if tiledb_key.startswith(self._key_prefix):
            return tiledb_key[len(self._key_prefix) :]
        return None


class DimMetadata(Metadata):
    """Metadata wrapper for accessing dimension metadata.

    This class allows access to the metadata for a dimension stored in the metadata
    for a TileDB array.

    Parameters:
        metadata (tiledb.Metadata): TileDB array metadata for the array containing the
            desired attribute.
        dim (str): Name or index of the arrary attribute being requested.
    """

    def __init__(self, metadata: tiledb.Metadata, dim: Union[str, int]):
        super().__init__(metadata)
        try:
            dim_name = metadata.array.dim(dim).name
        except tiledb.TileDBError as err:
            raise KeyError(f"Dimension `{dim}` not found in array.") from err
        self._key_prefix = DIM_METADATA_FLAG + dim_name + "."

    def _to_tiledb_key(self, key: str) -> str:
        return self._key_prefix + key

    def _from_tiledb_key(self, tiledb_key: str) -> Optional[str]:
        if tiledb_key.startswith(self._key_prefix):
            return tiledb_key[len(self._key_prefix) :]
        return None


def create_group(
    uri: str,
    group_schema: Mapping[str, tiledb.ArraySchema],
    *,
    key: Optional[Union[Dict[str, str], str]] = None,
    ctx: Optional[tiledb.Ctx] = None,
    config: Optional[tiledb.Config] = None,
    append: bool = False,
):
    """Creates a TileDB group with arrays at relative locations inside the group.

    All arrays in the group will be added at a relative URI that matches the array name.

    Parameters:
        uri: Uniform resource identifier for TileDB group or array.
        group_schema: A mapping from array names to array schemas to add to the group.
        ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        append: If ``True``, add arrays from the provided group schema to an
            already existing group. The names for the arrays in the group schema
            cannot already exist in the group being append to.
    """
    if append:
        check_valid_group(uri, ctx=ctx)
        with tiledb.Group(uri, ctx=ctx) as group:
            for array_name in group_schema:
                if array_name in group:
                    raise ValueError(
                        f"Cannot append to group. Array `{array_name}` already exists."
                    )
    else:
        tiledb.group_create(uri, ctx)
    with tiledb.Group(uri, mode="w", ctx=ctx) as group:
        for array_name, array_schema in group_schema.items():
            tiledb.Array.create(
                uri=_get_array_uri(uri, array_name),
                schema=array_schema,
                key=_get_array_key(key, array_name),
                ctx=ctx,
            )
            group.add(uri=array_name, name=array_name, relative=True)


def open_group_array(
    group: tiledb.Group,
    *,
    array: Optional[str] = None,
    attr: Optional[str] = None,
    **kwargs,
) -> tiledb.Array:
    """Opens an array in a group either by specifying the name of the array or the name
    of an attribute in the array.

    If only providing the attribute, there must be exactly one array in the group with
    an attribute with the requested name.

    Parameters:
        array: If not ``None``, the name of the array to open. Overrides attr if
            both are provided.
        attr: If not ``None``, open the array that contains this attr. Attr must be in
            only one of the group arrays.
        **kwargs: Keyword arguments to pass to the ``tiledb.open`` method.

    Returns:
        tiledb.Array opened in the specified mode
    """
    # Get the item in the group that either has the requested array name or
    # requested attribute.
    if array is not None:
        item = group[array]
    elif attr is not None:
        arrays = tuple(
            item
            for item in group
            if item.type == tiledb.libtiledb.Array
            and tiledb.ArraySchema.load(item.uri).has_attr(attr)
        )
        if not arrays:
            raise KeyError(f"No attribute with name '{attr}' found.")
        if len(arrays) > 1:
            raise ValueError(
                f"The array must be specified when opening an attribute that "
                f"exists in multiple arrays in a group. Arrays with attribute "
                f"'{attr}' include: {item.name for item in group}."
            )
        item = arrays[0]
    else:
        raise ValueError(
            "Cannot open array. Either an array or attribute must be specified."
        )
    return tiledb.open(item.uri, attr=attr, **kwargs)


class GroupSchema(Mapping):
    """Schema for a TileDB group.

    A TileDB group is completely defined by the arrays in the group. This class is
    a mapping from array names to array schemas. It also contains an optional array
    schema for an array to store group-level metadata.

    Parameters:
        array_schemas: A dict of array names to array schemas in the group.
        ctx: TileDB Context used for generating default metadata schema.
    """

    @classmethod
    def load(
        cls,
        uri: str,
        ctx: Optional[tiledb.Ctx] = None,
        key: Optional[Union[Dict[str, str], str]] = None,
    ):
        """Loads a schema for a TileDB group from a TileDB URI.

        Parameters:
            uri: uniform resource identifier for the TileDB group
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            key: If not ``None``, encryption key, or dictionary of encryption keys, to
                decrypt arrays.
        """
        if tiledb.object_type(uri, ctx) != "group":
            raise ValueError(
                f"Failed to load the group schema. Provided uri '{uri}' is not a "
                f"valid TileDB group."
            )
        with tiledb.Group(uri, ctx=ctx) as group:
            array_schemas = {}
            for item in group:
                if item.type != tiledb.libtiledb.Array:
                    continue
                if item.name is None:
                    with warnings.catch_warnings:
                        warnings.warn(
                            f"Skipping unnamed array at URI: {item.uri}", stacklevel=3
                        )
                    continue
                array_name = item.name
                local_key = _get_array_key(key, array_name)
                array_schemas[array_name] = tiledb.ArraySchema.load(
                    item.uri, ctx, local_key
                )
        return cls(array_schemas)

    def __init__(
        self,
        array_schemas: Optional[Dict[str, tiledb.ArraySchema]] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        if array_schemas is None:
            self._array_schema_table = {}
        else:
            self._array_schema_table = dict(array_schemas)
        self._attr_to_arrays: Dict[str, List[str]] = defaultdict(list)
        for schema_name, schema in self._array_schema_table.items():
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
        return True

    def __getitem__(self, schema_name: str) -> tiledb.ArraySchema:
        """Returns the requested array schema.

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
        for name, schema in self.items():
            output.write(f"'{name}': {repr(schema)}")
        return output.getvalue()

    def _repr_html_(self) -> str:
        """Returns the object representation of this GroupSchame as HTML."""
        output = StringIO()
        output.write("<section>\n")
        output.write(f"<h3>{self.__class__.__name__}</h3>\n")
        for name, schema in self.items():
            output.write("<details>\n")
            output.write(f"<summary>ArraySchema <em>{name}</em></summary>\n")
            output.write(_array_schema_html(schema))
            output.write("</details>\n")
        output.write("</section>\n")
        return output.getvalue()

    def check(self):
        """Checks the correctness of each array in the GroupSchema."""
        for schema in self._array_schema_table.values():
            schema.check()

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
