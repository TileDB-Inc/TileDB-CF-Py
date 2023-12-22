"""This module contains API for working with TileDB and xarray."""
from typing import Any, Iterable, Mapping, Optional, Set

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
    skip_vars: Optional[Set[str]] = None,
    copy_group_metadata: bool = True,
    copy_variable_metadata: bool = True,
):
    """Creates a TileDB group and arrays from an xarray dataset.

    Optionally copies metadata as well.

    Parameters
    ----------
    dataset
        The xarray Dataset to write.
    group_uri
        The URI to the TileDB group to create or append to.
    config
        A TileDB config object to use for TileDB objects.
    ctx
        A TileDB context object to use for TileDB operations.
    append
        If true, add arrays to an existing TileDB Group. Otherwise, create a new TileDB
        group to add arrays to.
    encoding
        A nested dictionary with variable names as keys and dictionaries of TileDB
        specific encodings as values.
    unlimited_dims
        Set of dimensions to use the maximum dimension size for. Only used for
        variables in the dataset that do not have ``max_size`` encoding provided.
    skip_vars
        A set of names of variables not to add to the TileDB group.
    copy_group_metadata
        If true, copy xarray dataset metadata to the TileDB group.
    copy_variable_metadata
        If true, copy xarray variable metadata to the TileDB arrays as TileDB attribute
        metadata.
    """

    from ._writer import copy_from_xarray, create_from_xarray, extract_encoded_data

    # Splits dataset into variables and group-level metadata using the CF Convention
    # where possible.
    variables, group_metadata = extract_encoded_data(dataset, skip_vars=skip_vars)

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
    skip_vars: Optional[Set[str]] = None,
    copy_group_metadata=False,
    copy_variable_metadata=False,
):
    """Copies data from an xarray dataset to a TileDB group.

    Optionally copies metadata as well as variable data.

    Parameters
    ----------
    dataset
        The xarray Dataset to write.
    group_uri
        The URI to the TileDB group to create or append to.
    config
        A TileDB config object to use for TileDB objects.
    ctx
        A TileDB context object to use for TileDB operations.
    region
        A mapping from dimension names to integer slices that specify what regions in
        the TileDB arrays to write the data. Regions include the first value of the
        slice and exclude the final value.
    skip_vars
        A set of names of variables not to copy to the TileDB group.
    copy_group_metadata
        If true, copy xarray dataset metadata to the TileDB group.
    copy_variable_metadata
        If true, copy xarray variable metadata to the TileDB arrays as TileDB
        attribute metadata.
    """

    from ._writer import copy_from_xarray, extract_encoded_data

    # Splits dataset into variables and group-level metadata using the CF Convention
    # where possible.
    variables, group_metadata = extract_encoded_data(dataset, skip_vars)

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
    skip_vars: Optional[Set[str]] = None,
    copy_group_metadata: bool = True,
    copy_variable_metadata: bool = True,
):
    """Copies metadata from an xarray dataset to a TileDB group.

    Parameters
    ----------
    dataset
        The xarray Dataset to write.
    group_uri
        The URI to the TileDB group to create or append to.
    config
        A TileDB config object to use for TileDB objects.
    ctx
        A TileDB context object to use for TileDB operations.
    skip_vars
        A set of names of variables not to copy to the group.
    copy_group_metadata
        If true, copy xarray dataset metadata to the TileDB group.
    copy_variable_metadata
        If true, copy xarray variable metadata to the TileDB arrays as TileDB
        attribute metadata.
    """

    from ._writer import copy_from_xarray, extract_encoded_data

    # Splits dataset into variables and group-level metadata using the CF Convention
    # where possible.
    variables, group_metadata = extract_encoded_data(dataset, skip_vars)

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
    skip_vars: Optional[Set[str]] = None,
):
    """Creates a TileDB group and copies all data from an xarray dataset.

    Parameters
    ----------
    dataset
        The xarray Dataset to write.
    group_uri
        The URI to the TileDB group to create or append to.
    config
        A TileDB config object to use for TileDB objects.
    ctx
        A TileDB context object to use for TileDB operations.
    encoding
        A nested dictionary with variable names as keys and dictionaries of TileDB
        specific encoding.
    encoding
        A nested dictionary with variable names as keys and dictionaries of TileDB
        specific encodings as values.
    region
        A mapping from dimension names to integer slices that specify what regions in
        the TileDB arrays to write the data. Regions include the first value of the
        slice and exclude the final value.
    unlimited_dims
        Set of dimensions to use the maximum dimension size for. Only used for variables
        in the dataset that do not have `max_size` encoding provided.
    skip_vars
        A set of names of variables not to add to the TileDB group.
    """
    from ._writer import copy_from_xarray, create_from_xarray, extract_encoded_data

    # Splits dataset into variables and group-level metadata using the CF Convention
    # where possible.
    variables, group_metadata = extract_encoded_data(dataset, skip_vars)

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
