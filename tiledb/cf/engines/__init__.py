# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
from typing import Dict, Optional, Tuple, Union

import numpy as np

import tiledb

_DEFAULT_INDEX_DTYPE = np.dtype("uint64")


def from_netcdf_file(
    input_file: str,
    output_uri: str,
    input_group_path: str = "/",
    output_key: Optional[Union[Dict[str, str], str]] = None,
    output_ctx: Optional[tiledb.Ctx] = None,
    unlimited_dim_size: int = 10000,
    dim_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
    tiles: Dict[Tuple[str, ...], Tuple[int, ...]] = None,
):
    """Converts a NetCDF group from a provided input file to a TileDB CF dataspace.

    See :class:`~tiledb.cf.engines.netcdf4_engine.NetCDF4ConverterEngine` for more
    information on the backend converter engine used for the conversion.

    Parameters:
        input_file: The input NetCDF file to generate the converter engine from.
        output_uri: Uniform resource identifier for the TileDB group to be created.
        input_group_path: The path to the NetCDF group to copy data from. Use ``'/'``
            for the root group.
        output_key: If not ``None``, encryption key, or dictionary of encryption keys,
            to decrypt arrays.
        output_ctx: If not ``None``, TileDB context wrapper for a TileDB storage
            manager.
               dim_dtype: The numpy dtype for TileDB dimensions.
        unlimited_dim_size: The size of the domain for TileDB dimensions created
            from unlimited NetCDF dimensions.
        dim_dtype: The numpy dtype for TileDB dimensions.
        tiles: A map from the name of NetCDF dimensions defining a variable to the
            tiles of those dimensions in the generated NetCDF array.
    """
    from .netcdf4_engine import NetCDF4ConverterEngine

    converter = NetCDF4ConverterEngine.from_file(
        input_file,
        input_group_path,
        unlimited_dim_size,
        dim_dtype,
        tiles,
    )
    converter.convert(output_uri, output_key, output_ctx)


def from_netcdf_group(
    input_group,
    output_uri: str,
    output_key: Optional[Union[Dict[str, str], str]] = None,
    output_ctx: Optional[tiledb.Ctx] = None,
    unlimited_dim_size: int = 10000,
    dim_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
    tiles: Dict[Tuple[str, ...], Tuple[int, ...]] = None,
):
    """Converts a :class:`netCDF4.Group` to a TileDB CF dataspace.

    See :class:`~tiledb.cf.engines.netcdf4_engine.NetCDF4ConverterEngine` for more
    information on the backend converter engine used for the conversion.

    Parameters:
        input_group (netCDF4.Group): The NetCDF group to generate the converter engine
            from.
        output_uri: Uniform resource identifier for the TileDB group to be created.
        output_key: If not ``None``, encryption key, or dictionary of encryption keys,
            to decrypt arrays.
        output_ctx: If not ``None``, TileDB context wrapper for a TileDB storage
            manager.
        unlimited_dim_size: The size of the domain for TileDB dimensions created
            from unlimited NetCDF dimensions.
        dim_dtype: The numpy dtype for TileDB dimensions.
        tiles: A map from the name of NetCDF dimensions defining a variable to the
            tiles of those dimensions in the generated NetCDF array.
    """
    from .netcdf4_engine import NetCDF4ConverterEngine

    converter = NetCDF4ConverterEngine.from_group(
        input_group,
        unlimited_dim_size,
        dim_dtype,
        tiles,
    )
    converter.convert(output_uri, output_key, output_ctx, netcdf_group=input_group)
