"""Helper functions for internal use only."""
from __future__ import annotations

import os.path
from typing import Dict, Optional, Union

import numpy as np

import tiledb

DType = Union[int, float, str, None]


def check_valid_group(group_uri, ctx):
    """Raise a ValueError if the provided URI is not for a TileDB group."""
    object_type = tiledb.object_type(group_uri, ctx=ctx)
    if object_type != "group":
        raise ValueError(
            f"Cannot open group at URI '{group_uri}'. TileDB object with "
            f"type '{object_type}' is no a valid TileDB group."
        )


def get_array_key(
    key: Optional[Union[Dict[str, str], str]], array_name
) -> Optional[str]:
    """Returns a key for the array with name ``array_name``.

    Parameters
    ----------
    key
        If not ``None``, encryption key, or dictionary of encryption keys, to decrypt
        arrays.
    array_name
        Name of the array to decrypt.

    Returns
    -------
    Optional[str]
       Key for the array with name ``array_name``.
    """
    return key.get(array_name) if isinstance(key, dict) else key


def get_array_uri(group_uri: str, array_name: str) -> str:
    """Returns a URI for an array with name ``array_name`` inside a group at URI
     ``group_uri``.

     This method is only needed for creating relative arrays before adding them
     to a group.

    Parameters
    ----------
    group_uri
        URI of the group containing the array
    array_name
        name of the array

    Returns
    -------
    str:
        Array URI of an array with name ``array_name`` inside a group at URI
        ``group_uri``.
    """
    return os.path.join(group_uri, array_name)


def safe_set_metadata(meta, key, value):
    """Copy a metadata item to a TileDB array catching any errors as warnings."""
    if isinstance(value, np.ndarray):
        value = tuple(value.tolist())
    elif isinstance(value, np.generic):
        value = (value.tolist(),)
    meta[key] = value
