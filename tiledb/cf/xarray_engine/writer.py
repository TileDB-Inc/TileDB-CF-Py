import warnings
from itertools import product
from typing import Iterable, Mapping, Optional

import numpy as np
from xarray.coding import times
from xarray.conventions import encode_dataset_coordinates
from xarray.core.dataset import Dataset

import tiledb

from .._utils import check_valid_group
from ._common import (
    _ATTR_FILTERS_ENCODING,
    _ATTR_NAME_ENCODING,
    _DIM_DTYPE_ENCODING,
    _MAX_SHAPE_ENCODING,
    _TILE_SIZES_ENCODING,
)
from .array_wrapper import TileDBArrayWrapper


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


class TileDBVariableEncoder:
    def __init__(self, name, variable, encoding, unlimited_dims, ctx):
        self._ctx = ctx
        self._name = name
        self._variable = variable
        self._encoding = {}

        # Get variable specific unlimited dimensions.
        self._unlimited_dims = {
            dim_name for dim_name in unlimited_dims if dim_name in variable.dims
        }

        # Set default encodings.
        self.dim_dtype = encoding.pop(_DIM_DTYPE_ENCODING, np.dtype(np.uint32))
        self.attr_filters = encoding.pop(
            _ATTR_FILTERS_ENCODING,
            tiledb.FilterList(
                (tiledb.ZstdFilter(level=5, ctx=self._ctx),), ctx=self._ctx
            ),
        )
        self.attr_name = encoding.pop(
            _ATTR_NAME_ENCODING,
            f"{self._name}.data" if self._name in variable.dims else self._name,
        )

        # Set the maximum shape.
        if _MAX_SHAPE_ENCODING in encoding:
            self.max_shape = encoding.pop(_MAX_SHAPE_ENCODING)
        else:
            if unlimited_dims == set():
                self.max_shape = variable.shape
            else:
                unlim = np.iinfo(self.dim_dtype).max
                self.max_shape = tuple(
                    max(unlim, dim_size) if dim_name in unlimited_dims else dim_size
                    for dim_name, dim_size in zip(variable.dims, variable.shape)
                )

        self.tiles = encoding.pop(_TILE_SIZES_ENCODING, None)

        if encoding:
            warnings.warn(
                f"Unused encoding provided for variable '{self._name}'. Unused "
                f"keys: {encoding.keys()}",
                stacklevel=1,
            )

    @property
    def attr_dtype(self):
        return self._variable.dtype

    @property
    def attr_name(self):
        return self._encoding.get(_ATTR_NAME_ENCODING, self._name)

    @attr_name.setter
    def attr_name(self, name):
        if name in self._variable.dims:
            raise ValueError(
                f"Cannot create variable '{self._name}' with TileBD attribute named "
                f"'{name}'. This is already a dimension name. Set a different name "
                f"using the '{_ATTR_NAME_ENCODING}' encoding key."
            )
        self._encoding[_ATTR_NAME_ENCODING] = name

    @property
    def attr_filters(self):
        return self._encoding[_ATTR_FILTERS_ENCODING]

    @attr_filters.setter
    def attr_filters(self, filters):
        self._encoding[_ATTR_FILTERS_ENCODING] = filters

    @property
    def dim_dtype(self):
        return self._encoding[_DIM_DTYPE_ENCODING]

    @dim_dtype.setter
    def dim_dtype(self, dim_dtype):
        if dim_dtype.kind not in ("i", "u"):
            raise ValueError(
                f"Cannot set the dimension dtype to {dim_dtype} on variable."
                f"Provide a signed of unsigned integer dtype instead."
            )
        self._encoding[_DIM_DTYPE_ENCODING] = dim_dtype

    def create_array_schema(self):
        """Returns a TileDB attribute from the provided variable and encodings."""
        attr = tiledb.Attr(
            name=self.attr_name,
            dtype=self.attr_dtype,
            fill=self.fill,
            filters=self.attr_filters,
            ctx=self._ctx,
        )
        tiles = self.tiles
        max_shape = self.max_shape
        dims = tuple(
            tiledb.Dim(
                name=dim_name,
                dtype=self.dim_dtype,
                domain=(0, max_shape[index] - 1),
                tile=None if tiles is None else tiles[index],
                ctx=self._ctx,
            )
            for index, dim_name in enumerate(self._variable.dims)
        )
        return tiledb.ArraySchema(
            domain=tiledb.Domain(*dims, ctx=self._ctx),
            attrs=(attr,),
            ctx=self._ctx,
        )

    def get_encoding_metadata(self):
        meta = dict()
        return meta

    @property
    def fill(self):
        return self._encoding.get("_FillValue", None)

    @fill.setter
    def fill(self, fill):
        if fill is np.nan:
            self._encoding["_FILL_VALUE_ENCODING"] = None
        self._encoding["_FILL_VALUE_ENCODING"] = fill

    @property
    def max_shape(self):
        return self._encoding[_MAX_SHAPE_ENCODING]

    @max_shape.setter
    def max_shape(self, max_shape):
        if len(max_shape) != self._variable.ndim:
            raise ValueError(
                f"Encoding error for variable '{self._name}'. Incompatible "
                f"shape {max_shape} for variable with {self._variable.ndim} "
                f"dimensions. "
            )
        for dim_size, var_size in zip(max_shape, self._variable.shape):
            if dim_size < var_size:
                raise ValueError(
                    f"Encoding error for variable '{self._name}'. The max shape "
                    f"from the encoding is {(max_shape)} which has a dimension "
                    f"less than the variable shape {self._variable.shape}."
                )
        self._encoding[_MAX_SHAPE_ENCODING] = max_shape

    @property
    def encoding(self):
        return self._encoding

    @property
    def tiles(self):
        return self._encoding[_TILE_SIZES_ENCODING]

    @tiles.setter
    def tiles(self, tiles):
        if tiles is not None and len(tiles) != self._variable.ndim:
            raise ValueError(
                f"Encoding error for variable '{self._name}'. {len(tiles)} provided "
                f"for a variable with {self._variable.ndim} dimensions. There must be "
                f"exactly one tile per dimension."
            )
        self._encoding[_TILE_SIZES_ENCODING] = tiles

    @property
    def unlimited_dims(self):
        return self._unlimited_dims

    @property
    def variable_name(self):
        return self._variable_name


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

    return variables, attributes, dataset.encoding


def create_group_from_xarray(
    group_uri,
    variables,
    attributes,
    encoding,
    unlimited_dims,
    config,
    ctx,
    append,
    create_arrays,
):
    # Either create a new TileDB group or check that a group already exists.
    if append:
        check_valid_group(group_uri, ctx)
    else:
        tiledb.Group.create(group_uri, ctx=ctx)

    # Create new TileDB arrays to store variable data in.
    if create_arrays:
        with tiledb.Group(group_uri, mode="w", config=config, ctx=ctx) as group:
            for var_name, var in variables.items():
                # Get the array URI.
                array_uri = f"{group_uri}/{var_name}"

                # Create the array and add it to the group
                encoder = TileDBVariableEncoder(
                    var_name, var, encoding.pop(var_name, dict()), unlimited_dims, ctx
                )
                schema = encoder.create_array_schema()
                tiledb.Array.create(array_uri, schema)
                group.add(uri=var_name, name=var_name, relative=True)

                # Add any group-level encoding as metadata.
                for key, value in encoder.get_encoding_metadata().items():
                    group.meta[key] = value


def copy_from_xarray(  # noqa: C901
    group_uri,
    variables,
    attributes,
    region,
    config,
    ctx,
    copy_group_metadata,
    copy_variable_metadata,
    copy_variable_data,
):
    # Check that there is a group at the location.
    check_valid_group(group_uri, ctx)

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


def from_xarray(
    dataset: Dataset,
    group_uri,
    *,
    config: Optional[tiledb.Config] = None,
    ctx: Optional[tiledb.Ctx] = None,
    encoding: Optional[Mapping] = None,
    append: bool = False,
    create_arrays: bool = True,
    copy_group_metadata: bool = True,
    copy_variable_data: bool = True,
    copy_variable_metadata: bool = True,
    region: Optional[Mapping[str, slice]] = None,
    unlimited_dims: Optional[Iterable[str]] = None,
):
    # Check the region is valid for this datasets.
    region = dict() if region is None else region
    encoding = dict() if encoding is None else encoding

    # Splits dataset into variables and attributes (metadata) using the CF Convention
    # where possible.
    variables, attributes, dataset_encoding = extract_encoded_data(dataset)

    if create_arrays:
        # Get encoding for unlimited dimensions.
        if unlimited_dims is None:
            unlimited_dims = dataset.encoding.get("unlimited_dims", set())
        unlimited_dims = set(
            dim_name for dim_name in unlimited_dims if dim_name in dataset.dims
        )

        # Create the group and group arrays.
        create_group_from_xarray(
            group_uri=group_uri,
            variables=variables,
            attributes=attributes,
            encoding=encoding,
            unlimited_dims=unlimited_dims,
            config=config,
            ctx=ctx,
            append=append,
            create_arrays=create_arrays,
        )

    if copy_group_metadata or copy_variable_data or copy_variable_metadata:
        # Copy data and metadata to TileDB.
        copy_from_xarray(
            group_uri,
            variables,
            attributes,
            region,
            config,
            ctx,
            copy_group_metadata,
            copy_variable_metadata,
            copy_variable_data,
        )
