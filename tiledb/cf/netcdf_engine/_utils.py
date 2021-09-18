# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Class for helper functions for NetCDF to TileDB conversion."""

import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import netCDF4
import numpy as np

import tiledb

_DEFAULT_INDEX_DTYPE = np.dtype("uint64")
COORDINATE_SUFFIX = ".data"


def copy_group_metadata(netcdf_group: netCDF4.Group, meta: tiledb.libtiledb.Metadata):
    """Copy all NetCDF group attributs to a the metadata in a TileDB array."""
    for key in netcdf_group.ncattrs():
        value = netcdf_group.getncattr(key)
        if key == "history":
            value = f"{value} - TileDB array created on {time.ctime(time.time())}"
        safe_set_metadata(meta, key, value)


def get_ncattr(netcdf_item, key: str) -> Any:
    """Returns a NetCDF value from a key if it exists and ``None`` otherwise."""
    if key in netcdf_item.ncattrs():
        return netcdf_item.getncattr(key)
    return None


def get_variable_chunks(variable: netCDF4.Variable) -> Optional[Tuple[int, ...]]:
    """Returns the chunks from a NetCDF variable if chunked and ``None`` otherwise."""
    chunks = variable.chunking()
    return None if chunks is None or chunks == "contiguous" else tuple(chunks)


@contextmanager
def open_netcdf_group(
    group: Optional[Union[netCDF4.Dataset, netCDF4.Group]] = None,
    input_file: Optional[Union[str, Path]] = None,
    group_path: Optional[str] = None,
):
    """Context manager for opening a NetCDF group.

    If both an input file and group are provided, this function will prioritize
    opening from the group.

    Parameters:
        group: A NetCDF group to read from.
        input_file: A NetCDF file to read from.
        group_path: The path to the NetCDF group to read from in a NetCDF file. Use
            ``'/'`` to specify the root group.
    """
    if group is not None:
        if not isinstance(group, (netCDF4.Dataset, netCDF4.Group)):
            raise TypeError(
                f"Invalid input: group={group} of type {type(group)} is not a netCDF "
                f"Group or or Dataset."
            )
        yield group
    else:
        if input_file is None:
            raise ValueError(
                "An input file must be provided; no default input file was set."
            )
        if group_path is None:
            raise ValueError(
                "A group path must be provided; no default group path was set. Use "
                "``'/'`` for the root group."
            )
        root_group = netCDF4.Dataset(input_file)
        root_group.set_auto_maskandscale(False)
        try:
            netcdf_group = root_group
            if group_path != "/":
                for child_group_name in group_path.strip("/").split("/"):
                    netcdf_group = netcdf_group.groups[child_group_name]
            yield netcdf_group
        finally:
            root_group.close()


def safe_set_metadata(meta, key, value):
    """Copy a metadata item to a TileDB array catching any errors as warnings."""
    if isinstance(value, np.ndarray):
        value = tuple(value.tolist())
    elif isinstance(value, np.generic):
        value = (value.tolist(),)
    try:
        meta[key] = value
    except ValueError as err:  # pragma: no cover
        with warnings.catch_warnings():
            warnings.warn(f"Failed to set metadata `{key}={value}` with error: {err}")
