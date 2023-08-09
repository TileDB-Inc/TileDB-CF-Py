from itertools import product
from typing import Iterable, Mapping, Optional

from xarray.coding import times
from xarray.conventions import encode_dataset_coordinates
from xarray.core.dataset import Dataset

import tiledb

from .._utils import check_valid_group
from ._encoding import TileDBVariableEncoder
from .array_wrapper import TileDBArrayWrapper


def copy_from_xarray(  # noqa: C901
    group_uri,
    dataset,
    variables,
    attributes,
    *,
    region,
    config,
    ctx,
    copy_group_metadata,
    copy_variable_metadata,
    copy_variable_data,
):
    # Check that there is a group at the location.
    check_valid_group(group_uri, ctx)

    # Check the region input is valid.
    region = dict() if region is None else region
    for dim_name, dim_slice in region.items():
        if dim_name not in dataset.dims:
            raise ValueError(
                f"``region`` contains key '{dim_name}' that is not a valid "
                f"dimension in the dataset."
            )
        if not isinstance(dim_slice, slice):
            raise TypeError(
                f"``region`` contains a value {dim_slice} for dimension '{dim_name}'"
                f"with type {type(dim_slice)}. All values must be slices."
            )
        if dim_slice.step not in {1, None}:
            raise ValueError(
                f"``region`` contains a slice with step={dim_slice.step} on dimension "
                f"'{dim_name}. All slices must have step size 1 or None."
            )

    # Copy group metadata
    if copy_group_metadata:
        with tiledb.Group(group_uri, mode="w", config=config, ctx=ctx) as group:
            for key, val in attributes:
                group.meta[key] = val

    # Skip iterating over full variable list if only writing group metadata.
    if not copy_variable_data and not copy_variable_metadata:
        return

    # Copy variable data and metadata.
    with tiledb.Group(group_uri, config=config, ctx=ctx) as group:
        for var_name, var in variables.items():
            try:
                array_item = group[var_name]
            except KeyError as err:
                raise KeyError(
                    f"Cannot write variable '{var_name}'. No item in the TileDB "
                    f"group with that name."
                ) from err
            schema = tiledb.ArraySchema.load(array_item.uri, ctx)
            if schema.nattr != 1:
                raise ValueError(
                    f"Cannot write variable '{var.name}' to a TileDB array "
                    f"with more than one TileDB attribute."
                )
            array_wrapper = TileDBArrayWrapper(
                variable_name=var_name,
                uri=array_item.uri,
                schema=schema,
                attr_key=0,
                config=config,
                ctx=ctx,
                timestamp=None,
                fixed_dimensions=set(),
                dimension_sizes=dict(),
            )
            if copy_variable_metadata:
                array_wrapper.set_metadata(var.attrs)
            if copy_variable_data:
                for target_region, source_region in get_chunk_regions(
                    target_shape=array_wrapper.shape,
                    source_shape=var.shape,
                    dimensions=var.dims,
                    region=region,
                    chunks=var.chunks,
                ):
                    array_wrapper[target_region] = var[source_region].compute().data


def create_from_xarray(
    group_uri,
    dataset,
    variables,
    attributes,
    *,
    append,
    encoding,
    unlimited_dims,
    config,
    ctx,
):
    # Check the TileDB encoding is for valid variables.
    encoding = dict() if encoding is None else encoding
    for var_name in encoding:
        if var_name not in variables:
            raise ValueError(
                f"``encoding`` contains variable `{var_name}` not in the dataset."
            )

    # Either create a new TileDB group or check that a group already exists.
    if append:
        check_valid_group(group_uri, ctx)
    else:
        tiledb.Group.create(group_uri, ctx=ctx)

    # Get encoding for unlimited dimensions.
    if unlimited_dims is None:
        # If unlimited dimeensions is not explicitly set, than get it from the
        # dataset. Drop any dimension names not actually in the dataset.
        unlimited_dims = dataset.encoding.get("unlimited_dims", set())
        unlimited_dims = set(
            dim_name for dim_name in unlimited_dims if dim_name in dataset.dims
        )
    else:
        # If unlimited dimensions is explicitly set by the user, check they
        # provided valid dimension names.
        unlimited_dims = set(dim_name for dim_name in unlimited_dims)
        for dim_name in unlimited_dims:
            if dim_name not in dataset.dims:
                raise ValueError(
                    f"``unlimited_dims`` contains dimension '{dim_name}' not in"
                    f"the dataset."
                )

    # Create new TileDB arrays to store variable data in.
    with tiledb.Group(group_uri, mode="w", config=config, ctx=ctx) as group:
        for var_name, var in variables.items():
            # Get the array URI.
            array_uri = f"{group_uri}/{var_name}"

            # Create the array and add it to the group
            encoder = TileDBVariableEncoder(
                var_name, var, encoding.get(var_name, dict()), unlimited_dims, ctx
            )
            schema = encoder.create_array_schema()
            tiledb.Array.create(array_uri, schema)
            group.add(uri=var_name, name=var_name, relative=True)

            # Add any group-level encoding as metadata.
            for key, value in encoder.get_encoding_metadata().items():
                group.meta[key] = value


def extract_encoded_data(dataset):
    """Returns encoded xarray variables and attribtues (metadata) from an
    input xarray dataset.
    """

    # Get variables and apply xarray encoders.
    variables, attributes = encode_dataset_coordinates(dataset)
    variables = {
        var_name: times.CFDatetimeCoder().encode(times.CFTimedeltaCoder().encode(var))
        for var_name, var in variables.items()
    }

    # Check the input dataset is supported in TileDB.
    for var_name, var in variables.items():
        if var.dims == tuple():
            raise NotImplementedError(
                f"Failed to write variable '{var_name}'. Support for writing scalar "
                f"functions to TileDB is not implemented. Consider converting any "
                f"scalar variables to metadata or 1D variables."
            )

    return variables, attributes


def get_chunk_regions(region, dimensions, target_shape, source_shape, chunks):
    """Returns a generator of (target_region, source_region) for writing the
    input source by chunks.
    """
    # Use the dictionary of region slices to compute the indices of the
    # full region that is being set.
    region_indices = tuple(
        region.get(dim_name, slice(None)).indices(target_shape[index])
        for index, dim_name in enumerate(dimensions)
    )

    # Check the shape of the region is valid.
    region_shape = tuple(_range[1] - _range[0] for _range in region_indices)
    if region_shape != source_shape:
        raise RuntimeError(
            f"Cannot add variable with shape {source_shape} to region "
            f"{region_indices} with mismatched shape {region_shape}."
        )

    # Return a single (target region, source region ) pair for a non-chunked array.
    if chunks is None:
        target_region = tuple(slice(_range[0], _range[1]) for _range in region_indices)
        source_region = tuple(len(source_shape) * [slice(None)])
        return ((target_region, source_region) for _ in range(1))

    # Convert tuple of dimension sizes to slices per chunk.
    chunks = list(list(dim_chunks) for dim_chunks in chunks)
    for dim_chunks in chunks:
        for index in range(len(dim_chunks)):
            start = 0 if index == 0 else dim_chunks[index - 1].stop
            dim_chunks[index] = slice(start, start + dim_chunks[index])

    # Return (target region, source region) pairs for each chunk region.
    return (
        tuple(
            (
                tuple(
                    slice(
                        global_index[0] + local_slice.start,
                        global_index[0] + local_slice.stop,
                    )
                    for global_index, local_slice in zip(region_indices, chunk_region)
                ),
                tuple(chunk_region),
            )
        )
        for chunk_region in product(*chunks)
    )
