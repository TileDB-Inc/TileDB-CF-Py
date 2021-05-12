# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

import tiledb

_DEFAULT_INDEX_DTYPE = np.dtype("uint64")


def from_netcdf(
    input_file: Union[str, Path],
    output_uri: str,
    input_group_path: str = "/",
    recursive: bool = True,
    output_key: Optional[str] = None,
    output_ctx: Optional[tiledb.Ctx] = None,
    unlimited_dim_size: int = 10000,
    dim_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
    tiles: Dict[str, Dict[Tuple[str, ...], Tuple[int, ...]]] = None,
):
    """Converts a NetCDF input file to nested TileDB CF dataspaces.

    See :class:`~tiledb.cf.engines.netcdf4_engine.NetCDF4ConverterEngine` for more
    information on the backend converter engine used for the conversion.

    Parameters:
        input_file: The input NetCDF file to generate the converter engine from.
        output_uri: The uniform resource identifier for the TileDB group to be created.
        input_group_path: The path to the NetCDF group to copy data from. Use ``'/'``
            for the root group.
        recursive: If ``True``, recursively convert groups in a NetCDF file. Otherwise,
            only convert group provided.
        output_key: If not ``None``, encryption key to decrypt arrays.
        output_ctx: If not ``None``, TileDB context wrapper for a TileDB storage
            manager.
        dim_dtype: The numpy dtype for the TileDB dimensions created from NetCDF
            dimensions.
        unlimited_dim_size: The size of the domain for TileDB dimensions created
            from unlimited NetCDF dimensions.
        dim_dtype: The numpy dtype for TileDB dimensions.
        tiles: A map from the NetCDF group name to a map from the name of NetCDF

            dimensions defining a variable to the tiles of those dimensions in the
            generated NetCDF array.
    """
    from .netcdf4_engine import NetCDF4ConverterEngine, open_netcdf_group

    output_uri = output_uri if not output_uri.endswith("/") else output_uri[:-1]

    def recursive_convert_group(group):
        group_uri = output_uri + group.path
        converter = NetCDF4ConverterEngine.from_group(
            group,
            unlimited_dim_size,
            dim_dtype,
            tiles.get(group.path) if tiles is not None else None,
        )
        converter.convert(group_uri, output_key, output_ctx, netcdf_group=group)
        if recursive:
            for subgroup in group.groups.values():
                recursive_convert_group(subgroup)

    with open_netcdf_group(
        input_file=input_file,
        group_path=input_group_path,
    ) as dataset:
        recursive_convert_group(dataset)


def from_netcdf_group(
    netcdf_input,
    output_uri: str,
    input_group_path: str = "/",
    output_key: Optional[str] = None,
    output_ctx: Optional[tiledb.Ctx] = None,
    unlimited_dim_size: int = 10000,
    dim_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
    tiles: Optional[Dict[Tuple[str, ...], Optional[Tuple[int, ...]]]] = None,
):
    """Converts a group in a NetCDF file or :class:`netCDF4.Group` to a TileDB CF
    dataspace.

    See :class:`~tiledb.cf.engines.netcdf4_engine.NetCDF4ConverterEngine` for more
    information on the backend converter engine used for the conversion.

    Parameters:
        netcdf_input (Union[str, Path, netCDF4.Dataset, netCDF4.Group]): Either the
            NetCDF group to convert or the filepath (as a string) to the NetCDF group.
        output_uri: Uniform resource identifier for the TileDB group to be created.
        input_group_path: The path to the NetCDF group to copy data from. Use ``'/'``
            for the root group. This is only used if ``netcdf_input`` is a filepath.
        output_key: If not ``None``, encryption key to decrypt arrays.
        output_ctx: If not ``None``, TileDB context wrapper for a TileDB storage
            manager.
        unlimited_dim_size: The size of the domain for TileDB dimensions created
            from unlimited NetCDF dimensions.
        dim_dtype: The numpy dtype for TileDB dimensions.
        tiles: A map from the name of NetCDF dimensions defining a variable to the
            tiles of those dimensions in the generated NetCDF array.
    """
    from .netcdf4_engine import NetCDF4ConverterEngine

    if isinstance(netcdf_input, (str, Path)):
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_input,
            input_group_path,
            unlimited_dim_size,
            dim_dtype,
            tiles,
        )
        converter.convert(output_uri, output_key, output_ctx)
    else:
        converter = NetCDF4ConverterEngine.from_group(
            netcdf_input,
            unlimited_dim_size,
            dim_dtype,
            tiles,
        )
        converter.convert(output_uri, output_key, output_ctx, netcdf_group=netcdf_input)
