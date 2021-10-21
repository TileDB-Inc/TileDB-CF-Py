# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
from typing import Optional

import click
import numpy as np

from .netcdf_engine import from_netcdf


@click.group()
def cli():
    pass


@cli.command("netcdf-convert")
@click.option(
    "-i",
    "--input-file",
    required=True,
    type=str,
    help="The path or URI to the NetCDF file that will be converted.",
)
@click.option(
    "-o",
    "--output-uri",
    required=True,
    type=str,
    help="The URI for the output TileDB group.",
)
@click.option(
    "--input-group-path",
    type=str,
    default="/",
    show_default=True,
    help="The path in the input NetCDF for the root group that will be converted.",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    show_default=True,
    help="Recursively convert all groups contained in the input group path.",
)
@click.option(
    "--collect-attrs/--array-per-attr",
    default=True,
    show_default=True,
    help="Collect variables with the same dimensions into a single array.",
)
@click.option(
    "-k",
    "--output-key",
    type=str,
    default=None,
    show_default=True,
    help="Key for the generated TileDB arrays.",
)
@click.option(
    "--unlimited-dim-size",
    type=int,
    default=10000,
    show_default=True,
    help="Size to convert unlimited dimensions to.",
)
@click.option(
    "--dim-dtype",
    type=click.Choice(
        [
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ]
    ),
    default="uint64",
    show_default=True,
    help="The data type for TileDB dimensions created from converted NetCDF.",
)
def netcdf_convert(
    input_file: str,
    output_uri: str,
    input_group_path: str,
    recursive: bool,
    output_key: Optional[str],
    unlimited_dim_size: int,
    dim_dtype: str,
    collect_attrs: bool,
):
    """Converts a NetCDF input file to nested TileDB groups."""
    from_netcdf(
        input_file=input_file,
        output_uri=output_uri,
        input_group_path=input_group_path,
        recursive=recursive,
        output_key=output_key,
        output_ctx=None,
        unlimited_dim_size=unlimited_dim_size,
        dim_dtype=np.dtype(dim_dtype),
        tiles_by_var=None,
        tiles_by_dims=None,
        coords_to_dims=False,
        collect_attrs=collect_attrs,
    )
