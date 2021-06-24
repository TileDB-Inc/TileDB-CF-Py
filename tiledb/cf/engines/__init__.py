# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

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
    tiles_by_var: Optional[Dict[str, Dict[str, Optional[Sequence[int]]]]] = None,
    tiles_by_dims: Optional[
        Dict[str, Dict[Sequence[str], Optional[Sequence[int]]]]
    ] = None,
    use_virtual_groups: bool = False,
    collect_attrs: bool = True,
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
        tiles_by_var: A map from the name of a NetCDF variable to the tiles of the
            dimensions of the variable in the generated NetCDF array.
        tiles_by_dims: A map from the name of NetCDF dimensions defining a variable
            to the tiles of those dimensions in the generated NetCDF array.
        use_virtual_groups: If ``True``, create a virtual group using ``output_uri``
            as the name for the group metadata array. All other arrays will be named
            using the convention ``{uri}_{array_name}`` where ``array_name`` is the
            name of the array.
        collect_attrs: If True, store all attributes with the same dimensions
            in the same array. Otherwise, store each attribute in a scalar array.
    """
    from .netcdf4_engine import NetCDF4ConverterEngine, open_netcdf_group

    output_uri = output_uri if not output_uri.endswith("/") else output_uri[:-1]

    if tiles_by_var is None:
        tiles_by_var = {}
    if tiles_by_dims is None:
        tiles_by_dims = {}

    def recursive_convert_to_virtual_group(netcdf_group):
        converter = NetCDF4ConverterEngine.from_group(
            netcdf_group,
            unlimited_dim_size,
            dim_dtype,
            tiles_by_var.get(netcdf_group.path),
            tiles_by_dims.get(netcdf_group.path),
            collect_attrs=collect_attrs,
        )
        group_uri = (
            output_uri
            if netcdf_group.path == "/"
            else output_uri + netcdf_group.path.replace("/", "_")
        )
        converter.convert_to_virtual_group(
            group_uri, output_key, output_ctx, input_netcdf_group=netcdf_group
        )
        if recursive:
            for subgroup in netcdf_group.groups.values():
                recursive_convert_to_virtual_group(subgroup)

    def recursive_convert_to_group(netcdf_group):
        converter = NetCDF4ConverterEngine.from_group(
            netcdf_group,
            unlimited_dim_size,
            dim_dtype,
            tiles_by_var.get(netcdf_group.path),
            tiles_by_dims.get(netcdf_group.path),
            collect_attrs=collect_attrs,
        )
        group_uri = output_uri + netcdf_group.path
        converter.convert_to_group(
            group_uri, output_key, output_ctx, input_netcdf_group=netcdf_group
        )
        if recursive:
            for subgroup in netcdf_group.groups.values():
                recursive_convert_to_group(subgroup)

    if use_virtual_groups:
        with open_netcdf_group(
            input_file=input_file,
            group_path=input_group_path,
        ) as dataset:
            recursive_convert_to_virtual_group(dataset)
    else:
        with open_netcdf_group(
            input_file=input_file,
            group_path=input_group_path,
        ) as dataset:
            recursive_convert_to_group(dataset)


def from_netcdf_group(
    netcdf_input,
    output_uri: str,
    input_group_path: str = "/",
    output_key: Optional[str] = None,
    output_ctx: Optional[tiledb.Ctx] = None,
    unlimited_dim_size: int = 10000,
    dim_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
    tiles_by_var: Optional[Dict[str, Optional[Sequence[int]]]] = None,
    tiles_by_dims: Optional[Dict[Sequence[str], Optional[Sequence[int]]]] = None,
    use_virtual_groups: bool = False,
    collect_attrs: bool = True,
):
    """Converts a group in a NetCDF file or :class:`netCDF4.Group` to a TileDB CF
    dataspace.

    See :class:`~tiledb.cf.engines.netcdf4_engine.NetCDF4ConverterEngine` for more
    information on the backend converter engine used for the conversion.

    Parameters:
        netcdf_input (Union[str, Path, netCDF4.Dataset]): Either the
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
        tiles_by_var: A map from the name of a NetCDF variable to the tiles of the
            dimensions of the variable in the generated NetCDF array.
        tiles_by_dims: A map from the name of NetCDF dimensions defining a variable
            to the tiles of those dimensions in the generated NetCDF array.
        use_virtual_groups: If ``True``, create a virtual group using ``output_uri``
            as the name for the group metadata array. All other arrays will be named
            using the convention ``{uri}_{array_name}`` where ``array_name`` is the
            name of the array.
    """
    import netCDF4

    from .netcdf4_engine import NetCDF4ConverterEngine

    if isinstance(netcdf_input, netCDF4.Dataset):
        converter = NetCDF4ConverterEngine.from_group(
            netcdf_input,
            unlimited_dim_size,
            dim_dtype,
            tiles_by_var,
            tiles_by_dims,
            collect_attrs=collect_attrs,
        )
        if use_virtual_groups:
            converter.convert_to_virtual_group(
                output_uri, output_key, output_ctx, input_netcdf_group=netcdf_input
            )
        else:
            converter.convert_to_group(
                output_uri, output_key, output_ctx, input_netcdf_group=netcdf_input
            )
    else:
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_input,
            input_group_path,
            unlimited_dim_size,
            dim_dtype,
            tiles_by_var,
            tiles_by_dims,
            collect_attrs=collect_attrs,
        )
        if use_virtual_groups:
            converter.convert_to_virtual_group(output_uri, output_key, output_ctx)
        else:
            converter.convert_to_group(output_uri, output_key, output_ctx)
