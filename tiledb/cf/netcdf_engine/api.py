# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Functions for converting NetCDF files to TileDB."""

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

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
    coords_to_dims: bool = False,
    collect_attrs: bool = True,
    unpack_vars: bool = False,
    coords_filters: Optional[tiledb.FilterList] = None,
    offsets_filters: Optional[tiledb.FilterList] = None,
    attrs_filters: Optional[tiledb.FilterList] = None,
    copy_metadata: bool = True,
    use_virtual_groups: bool = False,
):
    """Converts a NetCDF input file to nested TileDB CF dataspaces.

    See :class:`~tiledb.cf.NetCDF4ConverterEngine` for more
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
            dimensions of the variable in the generated TileDB array.
        tiles_by_dims: A map from the name of NetCDF dimensions defining a variable
            to the tiles of those dimensions in the generated TileDB array.
        coords_to_dims: If ``True``, convert the NetCDF coordinate variable into a
            TileDB dimension for sparse arrays. Otherwise, convert the coordinate
            dimension into a TileDB dimension and the coordinate variable into a
            TileDB attribute.
        collect_attrs: If ``True``, store all attributes with the same dimensions in
            the same array. Otherwise, store each attribute in a scalar array.
        unpack_vars: Unpack NetCDF variables with NetCDF attributes ``scale_factor``
            or ``add_offset`` using the transformation ``scale_factor * value +
            unpack``.
        coords_filters: Default filters for all dimensions.
        offsets_filters: Default filters for all offsets for variable attributes
            and dimensions.
        attrs_filters: Default filters for all attributes.
        copy_metadata: If  ``True`` copy NetCDF group and variable attributes to
            TileDB metadata. If ``False`` do not copy metadata.
        use_virtual_groups: If ``True``, create a virtual group using ``output_uri``
            as the name for the group metadata array. All other arrays will be named
            using the convention ``{uri}_{array_name}`` where ``array_name`` is the
            name of the array.
    """
    from .converter import NetCDF4ConverterEngine, open_netcdf_group

    output_uri = output_uri if not output_uri.endswith("/") else output_uri[:-1]

    if tiles_by_var is None:
        tiles_by_var = {}
    if tiles_by_dims is None:
        tiles_by_dims = {}

    def recursive_convert(netcdf_group):
        converter = NetCDF4ConverterEngine.from_group(
            netcdf_group,
            unlimited_dim_size,
            dim_dtype,
            tiles_by_var.get(netcdf_group.path),
            tiles_by_dims.get(netcdf_group.path),
            coords_to_dims=coords_to_dims,
            collect_attrs=collect_attrs,
            unpack_vars=unpack_vars,
            coords_filters=coords_filters,
            offsets_filters=offsets_filters,
            attrs_filters=attrs_filters,
        )
        if use_virtual_groups:
            group_uri = (
                output_uri
                if netcdf_group.path == "/"
                else output_uri + netcdf_group.path.replace("/", "_")
            )
            converter.convert_to_virtual_group(
                group_uri,
                output_key,
                output_ctx,
                input_netcdf_group=netcdf_group,
                copy_metadata=copy_metadata,
            )
        else:
            group_uri = output_uri + netcdf_group.path
            converter.convert_to_group(
                group_uri,
                output_key,
                output_ctx,
                input_netcdf_group=netcdf_group,
                copy_metadata=copy_metadata,
            )
        if recursive:
            for subgroup in netcdf_group.groups.values():
                recursive_convert(subgroup)

    with open_netcdf_group(
        input_file=input_file,
        group_path=input_group_path,
    ) as dataset:
        recursive_convert(dataset)
