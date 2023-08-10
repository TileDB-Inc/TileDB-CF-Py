"""This module contains API for working with TileDB and xarray."""
from typing import Any, Iterable, Mapping, Optional

import tiledb


def create_group_from_xarray(
    dataset,
    group_uri,
    *,
    config: Optional[tiledb.Config] = None,
    ctx: Optional[tiledb.Ctx] = None,
    append: bool = False,
    encoding: Optional[Mapping[str, Any]] = None,
    unlimited_dims: Optional[Iterable[str]] = None,
    copy_group_metadata: bool = True,
    copy_variable_metadata: bool = True,
):
    """Creates a TileDB group and arrays from a xarray dataset and optionally copies
    metadata over.

    Parameters:
    ----------
    dataset: The xarray Dataset to write.
    group_uri: The URI to the TileDB group to create or append to.
    config: A TileDB config object to use for TileDB objects.
    ctx: A TileDB context object to use for TileDB operations.
    encoding: A nested dictionary with variable names as keys and dictionaries
        of TileDB specific encoding.
    unlimited_dims: Set of dimensions to use the maximum dimension size for. Only used
        for variables in the dataset that do not have `max_size` encoding provided.
    config: TileDB configuration to use for writing metadata to groups and arrays.
    ctx: Context object to use for TileDB operations.
    copy_group_metadata: If true, copy xarray dataset metadata to the TileDB group.
    copy_variable_metadata: If true, copy xarray variable metadata to the TileDB
        arrays as TileDB attribute metadata.

    """

    from ._writer import copy_from_xarray, create_from_xarray, extract_encoded_data

    # Splits dataset into variables and group-level metadata using the CF Convention
    # where possible.
    variables, group_metadata = extract_encoded_data(dataset)

    # Create the group and arrays in the group.
    create_from_xarray(
        group_uri=group_uri,
        dataset=dataset,
        variables=variables,
        append=append,
        encoding=encoding,
        unlimited_dims=unlimited_dims,
        config=config,
        ctx=ctx,
    )

    # Copy metadata to TileDB.
    if copy_group_metadata or copy_variable_metadata:
        copy_from_xarray(
            group_uri=group_uri,
            dataset=dataset,
            variables=variables,
            group_metadata=group_metadata,
            region=None,
            config=config,
            ctx=ctx,
            copy_group_metadata=copy_group_metadata,
            copy_variable_metadata=copy_variable_metadata,
            copy_variable_data=False,
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
    """Copies data from an xarray dataset to a TileDB group corresponding to the
    dataset.

    Optionally copies metadata as well as variable data.

    dataset: The xarray Dataset to write.
    group_uri: The URI to the TileDB group to create or append to.
    config: A TileDB config object to use for TileDB objects.
    ctx: A TileDB context object to use for TileDB operations.
    config: TileDB configuration to use for writing metadata to groups and arrays.
    region: A mapping from dimension names to integer slices along the
        dataset dimensions to indicate the region to write this dataset's data in.
    copy_group_metadata: If true, copy xarray dataset metadata to the TileDB group.
    copy_variable_metadata: If true, copy xarray variable metadata to the TileDB
        arrays as TileDB attribute metadata.

    """

    from ._writer import copy_from_xarray, extract_encoded_data

    # Splits dataset into variables and group-level metadata using the CF Convention
    # where possible.
    variables, group_metadata = extract_encoded_data(dataset)

    # Copy data and metadata to TileDB.
    copy_from_xarray(
        group_uri=group_uri,
        dataset=dataset,
        variables=variables,
        group_metadata=group_metadata,
        region=region,
        config=config,
        ctx=ctx,
        copy_variable_data=True,
        copy_group_metadata=copy_group_metadata,
        copy_variable_metadata=copy_variable_metadata,
    )


def copy_metadata_from_xarray(
    dataset,
    group_uri,
    *,
    config: Optional[tiledb.Config] = None,
    ctx: Optional[tiledb.Ctx] = None,
    copy_group_metadata: bool = True,
    copy_variable_metadata: bool = True,
):
    """Copies metadata from an xarray dataset to a TileDB group corresponding
    to the dataset.

    dataset: The xarray Dataset to write.
    group_uri: The URI to the TileDB group to create or append to.
    config: A TileDB config object to use for TileDB objects.
    ctx: A TileDB context object to use for TileDB operations.
    copy_group_metadata: If true, copy xarray dataset metadata to the TileDB group.
    copy_variable_metadata: If true, copy xarray variable metadata to the TileDB
        arrays as TileDB attribute metadata.
    """

    from ._writer import copy_from_xarray, extract_encoded_data

    # Splits dataset into variables and group-level metadata using the CF Convention
    # where possible.
    variables, group_metadata = extract_encoded_data(dataset)

    # Copy data to TileDB.
    copy_from_xarray(
        group_uri=group_uri,
        dataset=dataset,
        variables=variables,
        group_metadata=group_metadata,
        region=None,
        config=config,
        ctx=ctx,
        copy_group_metadata=copy_group_metadata,
        copy_variable_metadata=copy_variable_metadata,
        copy_variable_data=False,
    )


def from_xarray(
    dataset,
    group_uri,
    *,
    config: Optional[tiledb.Config] = None,
    ctx: Optional[tiledb.Ctx] = None,
    encoding: Optional[Mapping[str, Any]] = None,
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
    from ._writer import copy_from_xarray, create_from_xarray, extract_encoded_data

    # Splits dataset into variables and group-level metadata using the CF Convention
    # where possible.
    variables, group_metadata = extract_encoded_data(dataset)

    # Create the group and group arrays.
    create_from_xarray(
        group_uri=group_uri,
        dataset=dataset,
        variables=variables,
        encoding=encoding,
        unlimited_dims=unlimited_dims,
        config=config,
        ctx=ctx,
        append=False,
    )

    # Copy data and metadata to TileDB.
    copy_from_xarray(
        group_uri=group_uri,
        dataset=dataset,
        variables=variables,
        group_metadata=group_metadata,
        region=region,
        config=config,
        ctx=ctx,
        copy_group_metadata=True,
        copy_variable_metadata=True,
        copy_variable_data=True,
    )
