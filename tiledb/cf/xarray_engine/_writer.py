from itertools import product
from typing import Any, Generator, Iterable, Mapping, Optional, Tuple

from xarray.coding import times
from xarray.conventions import encode_dataset_coordinates
from xarray.core.dataset import Dataset, Variable

import tiledb

from .._utils import check_valid_group, safe_set_metadata
from ._array_wrapper import TileDBArrayWrapper
from ._encoding import TileDBVariableEncoder


def copy_from_xarray(  # noqa: C901
    group_uri: str,
    dataset: Dataset,
    variables: Mapping[str, Variable],
    group_metadata: Mapping[str, Any],
    *,
    config: Optional[tiledb.Config],
    ctx: Optional[tiledb.Ctx],
    region: Optional[Mapping[str, slice]],
    copy_group_metadata: bool,
    copy_variable_metadata: bool,
    copy_variable_data: bool,
):
    """Copies data and metadata from an xarray dataset to a TileDB group corresponding
    to the dataset.


    Parameters
    ----------
    group_uri
        The URI to the TileDB group to create or append to.
    dataset
        The xarray Dataset to write.
    variables
        A mapping of encoded xarray variables.
    group_metadata
        A mapping of key-value pairs correspoding to dataset metadata.
    config
        A TileDB config object to use for TileDB objects.
    ctx
        A TileDB context object to use for TileDB operations.
    region
        A mapping from dimension names to integer slices along thevdataset dimensions
        to indicate the region to write this dataset's data in.
    copy_group_metadata
        If true, copy xarray dataset metadata to the TileDB group.
    copy_variable_metadata
        If true, copy xarray variable metadata to the TileDBvarrays as TileDB attribute
        metadata.
    copy_variable_data
        If true, copy variable data to the TileDB arrays.
    """

    # Check that there is a group at the location.
    check_valid_group(group_uri, ctx)

    # Check the region input is valid.
    region = dict() if region is None else region
    for dim_name, dim_slice in region.items():
        if dim_name not in dataset.dims:
            raise ValueError(
                f"``region`` contains key '{dim_name}' that is not a valid "
                f"dimension in the provided dataset."
            )
        if not isinstance(dim_slice, slice):
            raise TypeError(
                f"``region`` contains a value {dim_slice} for dimension '{dim_name}' "
                f"with type {type(dim_slice)}. All values must be slices."
            )
        if dim_slice.step not in {1, None}:
            raise ValueError(
                f"``region`` contains a slice with step={dim_slice.step} on dimension "
                f"'{dim_name}'. All slices must have step size 1 or None."
            )
        if (dim_slice.start is not None and dim_slice.start < 0) or (
            dim_slice.stop is not None and dim_slice.stop < 0
        ):
            raise ValueError(
                f"``region`` contains a value {dim_slice} with a negative value on "
                f"dimension '{dim_name}'. All slice values must be non-negative."
            )
        if (
            dim_slice.start is not None
            and dim_slice.stop is not None
            and dim_slice.stop <= dim_slice.start
        ):
            raise ValueError(
                f"``region`` contains a value {dim_slice} with end value greater "
                f"than or equal to the starting value. All slice values must be for a "
                f"non-zero range."
            )

    # Copy group metadata
    if copy_group_metadata:
        with tiledb.Group(group_uri, mode="w", config=config, ctx=ctx) as group:
            for key, val in group_metadata.items():
                safe_set_metadata(group.meta, key, val)

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
                copy_variable(var_name, var, array_wrapper, region)


def copy_variable(name, variable, array_wrapper, region):
    # Check the number of dimensions match.
    if len(array_wrapper.shape) != variable.ndim:
        raise ValueError(
            f"Cannot write variable '{name}' with {variable.ndim} dimensions "
            f"to an array with {len(array_wrapper.shape)} dimensions."
        )

    # Get the target region to write the data into.
    def get_dimension_slice(dim_name, var_dim_size, array_dim_size):
        """Returns target dimension slice for xarray write.

        Raises a value error if the size of the target slice does not match the
        size of the source dimension.

        Parameters
        ----------
        dim_name
            Name of the dimension to return the slice for.
        var_dim_size
            Size of the dimension in the source xarray variable.
        array_dim_size
            Size of the domain of the dimension in the target TileDB array.
        """
        if var_dim_size > array_dim_size:
            raise ValueError(
                f"Cannot write variable '{name}' with shape {variable.shape} on a "
                f"TileDB array with maximum shape {array_wrapper.shape}."
            )
        if dim_name not in region:
            if var_dim_size != array_dim_size:
                raise ValueError(
                    f"Failed to  write variable '{name}' with shape {variable.shape} "
                    f"on a TileDB array with shape {array_wrapper.shape}. Missing "
                    f"``region`` value for dimension '{dim_name}' with mismatched "
                    f"sizes."
                )
            return slice(0, array_dim_size, None)
        else:
            dim_slice = region[dim_name]
            start = 0 if dim_slice.start is None else dim_slice.start
            stop = array_dim_size if dim_slice.stop is None else dim_slice.stop
            if stop > array_dim_size:
                raise ValueError(
                    f"Provided region {dim_slice} for dimension '{dim_name}' is out of "
                    f"bounds on variable '{name}'. The maximum size of dimension "
                    f"'{dim_name}' on variable '{name}' is {array_dim_size}."
                )
            return slice(start, stop, None)

    target_region = tuple(
        get_dimension_slice(dim_name, var_dim_size, array_dim_size)
        for dim_name, var_dim_size, array_dim_size in zip(
            variable.dims, variable.shape, array_wrapper.shape
        )
    )

    # Check the shape of the region is valid.
    region_shape = tuple(
        dim_slice.stop - dim_slice.start for dim_slice in target_region
    )
    if region_shape != variable.shape:
        local_region = {
            dim_name: region.get(dim_name, slice(None)) for dim_name in variable.dims
        }
        raise ValueError(
            f"Cannot add variable '{name}' with shape {variable.shape} to region "
            f"{local_region} with mismatched shape {region_shape}."
        )

    # Iterate over all chunks.
    for target_chunk, source_chunk in get_chunk_regions(
        target_region=target_region,
        source_shape=variable.shape,
        chunks=variable.chunks,
    ):
        array_wrapper[target_chunk] = variable[source_chunk].compute().data


def create_from_xarray(
    group_uri: str,
    dataset: Dataset,
    variables: Mapping[str, Variable],
    *,
    config: Optional[tiledb.Config],
    ctx: Optional[tiledb.Ctx],
    append: bool,
    encoding: Optional[Mapping[str, Any]],
    unlimited_dims: Optional[Iterable[str]],
):
    """Creates a TileDB group and arrays from a xarray dataset and optionally copies
    metadata over.

    Parameters
    ----------
    dataset
        The xarray Dataset to write.
    group_uri
        The URI to the TileDB group to create or append to.
    variables
        A mapping of encoded xarray variables.
    config
        A TileDB config object to use for TileDB objects.
    ctx
        A TileDB context object to use for TileDB operations.
    encoding
        A nested dictionary with variable names as keys and dictionaries of TileDB
        specific encoding.
    unlimited_dims
        Set of dimensions to use the maximum dimension size for. Only used for variables
        in the dataset that do not have `max_size` encoding provided.
    config
        TileDB configuration to use for writing metadata to groups and arrays.
    ctx
        Context object to use for TileDB operations.
    """

    # Check the TileDB encoding is for valid variables.
    encoding = dict() if encoding is None else encoding
    for var_name in encoding:
        if var_name not in variables:
            raise KeyError(
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
                raise KeyError(
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


def extract_encoded_data(dataset, skip_vars=None):
    """Returns encoded xarray variables and attribtues (metadata) from an
    input xarray dataset.

    Parameters
    ----------
    dataset
        Xarray dataset to encode.
    skip_vars
        Variables to exclude if they exist in the dataset
    """

    if skip_vars is None:
        skip_vars = set()

    # Get variables and apply xarray encoders.
    variables, group_metadata = encode_dataset_coordinates(dataset)
    variables = {
        var_name: times.CFDatetimeCoder().encode(times.CFTimedeltaCoder().encode(var))
        for var_name, var in variables.items()
        if var_name not in skip_vars
    }

    # Check the input dataset is supported in TileDB.
    for var_name, var in variables.items():
        if var.dims == tuple():
            raise NotImplementedError(
                f"Failed to write variable '{var_name}'. Support for writing scalar "
                f"functions to TileDB is not implemented. Consider converting any "
                f"scalar variables to metadata or 1D variables."
            )

    return variables, group_metadata


def get_chunk_regions(
    target_region: Tuple[slice, ...],
    source_shape: Tuple[int, ...],
    chunks: Optional[Tuple[Tuple[int, ...], ...]],
) -> Generator[Tuple[Tuple[slice, ...], Tuple[slice, ...]], None, None]:
    """Returns a generator of (target_region, source_region) for writing the
    input source by chunks.

    Parameters
    ----------
    target_region
        Slices that correspond to the target region to query using
        numpy-style indexing.
    source_shape
        Shape of the input data.
    chunks
        A tuple or tuples containing the fully explicit size of chunks along each
        dimension.

    Yields
    ------
    Tuple[Tuple[slice, ...], Tuple[slice, ....]] or None
        Yields tuples of (target_chunk, source_chunk) pairs that cover the entire target
        and source regions.
    """

    # Return a single (target region, source region ) pair for a non-chunked array.
    if chunks is None:
        source_region = tuple(len(source_shape) * [slice(None)])
        yield (target_region, source_region)

    else:
        # The input chunks is a tuple of a tuple of chunk sizes for each dimension.
        # This step converts that to a list of a list of slices for the region each
        # chunk is located at.
        chunk_regions = list(list(dim_chunks) for dim_chunks in chunks)
        for dim_chunks in chunk_regions:
            for index in range(len(dim_chunks)):
                start = 0 if index == 0 else dim_chunks[index - 1].stop
                dim_chunks[index] = slice(start, start + dim_chunks[index])

        # Return (target region, source region) pairs for each chunk region by
        # taking the outer product of chunk regions and shifting them to the target
        # location.
        for source_chunk in product(*chunk_regions):
            target_chunk = tuple(
                slice(
                    global_slice.start + local_slice.start,
                    global_slice.start + local_slice.stop,
                )
                for global_slice, local_slice in zip(target_region, source_chunk)
            )
            yield tuple((target_chunk, tuple(source_chunk)))
