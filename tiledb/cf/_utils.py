"""Helper functions for internal use only."""
import numpy as np

import tiledb


def check_valid_group(group_uri, ctx):
    """Raise a ValueError if the provided URI is not for a TileDB group."""
    object_type = tiledb.object_type(group_uri, ctx=ctx)
    if object_type != "group":
        raise ValueError(
            f"Cannot open group at URI '{group_uri}'. TileDB object with "
            f"type '{object_type}' is no a valid TileDB group."
        )


def safe_set_metadata(meta, key, value):
    """Copy a metadata item to a TileDB array catching any errors as warnings."""
    if isinstance(value, np.ndarray):
        value = tuple(value.tolist())
    elif isinstance(value, np.generic):
        value = (value.tolist(),)
    meta[key] = value
