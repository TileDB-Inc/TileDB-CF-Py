# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Class for helper functions for NetCDF to TileDB conversion."""

import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

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


def get_netcdf_metadata(
    netcdf_item, key: str, default: Any = None, is_number: bool = False
) -> Any:
    """Returns a NetCDF attribute value from a key if it exists and the default value
    otherwise.

    If ``is_number=True``, the result is only returned if it is a numpy number. If the
    key exists but is not a numpy number, then a warning is raised. If the key exists
    and is an array of length 1, the scalar value is returned.

    Parameters:
        key: NetCDF attribute name to return.
        default: Default value to return if the attribute is not found.

    Returns:
        The NetCDF attribute value, if found. Otherwise, return the default value.
    """
    if key in netcdf_item.ncattrs():
        value = netcdf_item.getncattr(key)
        if is_number:
            if (
                isinstance(value, str)
                or not np.issubdtype(value.dtype, np.number)
                or np.size(value) != 1
            ):
                with warnings.catch_warnings():
                    warnings.warn(
                        f"Attribute '{key}' has value='{value}' that not a number. "
                        f"Using default {key}={default} instead."
                    )
                return default
            if not np.isscalar(value):
                value = value.item()
        return value
    return default


def get_unpacked_dtype(variable: netCDF4.Variable) -> np.dtype:
    """Returns the Numpy data type of a variable after it has been unpacked by applying
    any scale_factor or add_offset.

    Parameters:
        variable: The NetCDF variable to get the unpacked data type of.
    """
    input_dtype = np.dtype(variable.dtype)
    if not np.issubdtype(input_dtype, np.number):
        raise ValueError(
            f"Unpacking only support NetCDF variables with integer or floating-point "
            f"data. Input variable has datatype {input_dtype}."
        )
    test = np.array(0, dtype=input_dtype)
    scale_factor = get_netcdf_metadata(variable, "scale_factor", is_number=True)
    add_offset = get_netcdf_metadata(variable, "add_offset", is_number=True)
    if scale_factor is not None:
        test = scale_factor * test
    if add_offset is not None:
        test = test + add_offset
    return test.dtype


def get_variable_values(
    variable: netCDF4.Variable,
    indexer: Union[slice, Sequence[slice]],
    fill: Optional[Union[int, float, str]],
    unpack: bool,
) -> np.ndarray:
    """Returns the values for a NetCDF variable at the requested indices.

    Parameters:
        variable: NetCDF variable to get values from.
        indexer: Sequence of slices used to index the NetCDF variable.
        fill: If not ``None``, the fill value to use for the output data.
        unpack: If ``True``, unpack the variable if it contains a ``scale_factor``
            or ``add_offset``.
    """
    values = variable.getValue() if variable.ndim == 0 else variable[indexer]
    netcdf_fill = get_netcdf_metadata(variable, "_FillValue")
    if fill is not None and netcdf_fill is not None and fill != netcdf_fill:
        np.putmask(values, values == netcdf_fill, fill)
    if unpack:
        scale_factor = get_netcdf_metadata(variable, "scale_factor", is_number=True)
        if scale_factor is not None:
            values = scale_factor * values
        add_offset = get_netcdf_metadata(variable, "add_offset", is_number=True)
        if add_offset is not None:
            values = values + add_offset
    return values


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
