from typing import Iterable, Mapping, Optional

import tiledb


def create_group_from_xarray(
    dataset,
    group_uri,
    *,
    config: Optional[tiledb.Config] = None,
    ctx: Optional[tiledb.Ctx] = None,
    encoding: Optional[Mapping] = None,
    append: bool = False,
    copy_group_metadata: bool = True,
    copy_variable_metadata: bool = True,
    unlimited_dims: Optional[Iterable[str]] = None,
):
    """TODO: add create_group_from_xarray docstring"""
    from .writer import from_xarray as from_xarray_impl

    return from_xarray_impl(
        dataset,
        group_uri,
        config=config,
        ctx=ctx,
        encoding=encoding,
        append=append,
        create_arrays=True,
        copy_group_metadata=copy_group_metadata,
        copy_variable_data=False,
        copy_variable_metadata=copy_variable_metadata,
        region=None,
        unlimited_dims=unlimited_dims,
    )


def copy_data_from_xarray(
    dataset,
    group_uri,
    *,
    config: Optional[tiledb.Config] = None,
    ctx: Optional[tiledb.Ctx] = None,
    region: Optional[Mapping[str, slice]] = None,
    copy_group_metadata=False,
    copy_variable_metadata=False,
):
    """TODO: add copy_data_from_xarray docstring"""
    from .writer import from_xarray as from_xarray_impl

    return from_xarray_impl(
        dataset,
        group_uri,
        config=config,
        ctx=ctx,
        encoding=None,
        append=True,
        create_arrays=False,
        copy_group_metadata=copy_group_metadata,
        copy_variable_data=True,
        copy_variable_metadata=copy_variable_metadata,
        region=region,
        unlimited_dims=None,
    )


def copy_metadata_from_xarray(
    dataset,
    group_uri,
    *,
    config: Optional[tiledb.Config] = None,
    ctx: Optional[tiledb.Ctx] = None,
    encoding: Optional[Mapping] = None,
    copy_group_metadata: bool = True,
    copy_variable_metadata: bool = True,
):
    """TODO: add copy_data_from_xarray docstring"""
    from .writer import from_xarray as from_xarray_impl

    return from_xarray_impl(
        dataset,
        group_uri,
        config=config,
        ctx=ctx,
        encoding=encoding,
        append=True,
        create_arrays=False,
        copy_group_metadata=True,
        copy_variable_data=False,
        copy_variable_metadata=True,
        region=None,
        unlimited_dims=None,
    )


def from_xarray(
    dataset,
    group_uri,
    *,
    config: Optional[tiledb.Config] = None,
    ctx: Optional[tiledb.Ctx] = None,
    encoding: Optional[Mapping] = None,
    region: Optional[Mapping[str, slice]] = None,
    unlimited_dims: Optional[Iterable[str]] = None,
):
    """Creates a TileDB group from an xarray dataset and copies all
    xarray data and metadata over to the TileDB group.

    dataset: The xarray Dataset to write.
    group_uri: The URI to the TileDB group to create or append to.
    config: A TileDB config object to use for TileDB objects.
    ctx: A TileDB context object to use for TileDB operations.
    encoding: A nested dictionary with variable names as keys and dictionaries
        of TileDB specific encoding.
    region: A mapping from dimension names to integer slices along the
        dataset dimensions to indicate the region to write this dataset's data in.
    unlimited_dims: Set of dimensions to use the maximum dimension size for. Only used
        for variables in the dataset that do not have `max_size` encoding provided.
    """
    from .writer import from_xarray as from_xarray_impl

    return from_xarray_impl(
        dataset,
        group_uri,
        config=config,
        ctx=ctx,
        encoding=encoding,
        append=False,
        create_arrays=True,
        copy_group_metadata=True,
        copy_variable_data=True,
        copy_variable_metadata=True,
        region=region,
        unlimited_dims=unlimited_dims,
    )
