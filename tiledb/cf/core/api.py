from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Optional, Union

import tiledb

from .._utils import check_valid_group, get_array_key, get_array_uri


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

    Parameters
    ----------
    uri
        Uniform resource identifier for TileDB group or array.
    group_schema
        A mapping from array names to array schemas to add to the group.
    key
        A encryption key or dict from array names to encryption keys.
    ctx
        If not ``None``, TileDB context wrapper for a TileDB storage manager.
    append
        If ``True``, add arrays from the provided group schema to an already existing
        group. The names for the arrays in the group schema cannot already exist in the
        group being append to.
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
                uri=get_array_uri(uri, array_name),
                schema=array_schema,
                key=get_array_key(key, array_name),
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

    Parameters
    ----------
    group
        The tiledb group to open the array in.
    array
        If not ``None``, the name of the array to open. Overrides attr if both are
        provided.
    attr
        If not ``None``, open the array that contains this attr. Attr must be in only
        one of the group arrays.
    **kwargs: dict, optional
        Keyword arguments to pass to the ``tiledb.open`` method.

    Returns
    -------
    tiledb.Array:
        An array opened in the specified mode
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
                f"'{attr}' include: {list(item.name for item in group)}."
            )
        item = arrays[0]
    else:
        raise ValueError(
            "Cannot open array. Either an array or attribute must be specified."
        )
    return tiledb.open(item.uri, attr=attr, **kwargs)
