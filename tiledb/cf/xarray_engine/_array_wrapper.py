import numpy as np
from xarray.backends.common import BackendArray

import tiledb

from .._utils import safe_set_metadata
from ._common import _ATTR_PREFIX


def _to_zero_based_tiledb_index(dim_name, dim_size, index):
    """Converts an xarray integer, array, or slice to an index object usable by the
    TileDB multi_index function. Only for dimensions with integer domains that start
    at zero.

    The following is assumed about xarray indices:
       * An index may be an integer, a slice, or a Numpy array of integer indices.
       * An integer index or component of an array is such that -size <= value < size.
         * Non-negative values are a standard zero-based index.
         * Negative values count backwards from the end of the array with the last value
           of the array starting at -1.

    Parameters
    ----------
    dim_name
        Name of the dimension. Used for errors.
    dim_size
        Size of the dimension as interpreted by xarray. May be smaller than the
        full domain of the TileDB dimension.
    index
        An integer index, array of integer indices, or a slice for indexing an
        xarray dimension.

    Returns
    -------
    Union[int, List[int], slice]
        An integer, a list of integer values, or a slice for indexing a
        TileDB dimension using mulit_index.
    """
    if np.isscalar(index):
        # Convert xarray index to TileDB dimension coordinate
        if not -dim_size <= index < dim_size:
            raise IndexError(
                f"Index {index} out of bounds for dimension '{dim_name}' with size "
                f"{dim_size}."
            )
        return index if index >= 0 else index + dim_size - 1

    if isinstance(index, slice):
        # Using range handles negative numbers and `None` values.
        index = range(dim_size)[index]
        if index.step in (1, None):
            # Convert from index slice to coordinate slice (note that xarray
            # includes the starting point and excludes the ending point vs. TileDB
            # multi_index which includes both the staring point and ending point).
            return slice(index.start, index.stop - 1)
        # This can be replaced with a proper slice when TileDB supports steps.
        return list(np.arange(index.start, index.stop, index.step))

    if isinstance(index, np.ndarray):
        # Check numpy array has valid data.
        if index.ndim != 1:
            raise TypeError(
                f"Invalid indexer array for dimension '{dim_name}'. Input array index "
                f"must have exactly 1 dimension."
            )
        if not ((-dim_size <= index).all() and (index < dim_size).all()):
            raise IndexError(
                f"Index {index} out of bounds for dimension '{dim_name}' with size "
                f"{dim_size}."
            )
        # Convert negative indices to positive indices and return as a list of
        # values.
        return list(index + np.where(index >= 0, 0, dim_size - 1))
    raise TypeError(
        f"Unexpected indexer type {type(index)} for dimension '{dim_name}'."
    )


class TileDBArrayWrapper(BackendArray):
    """Wrapper that allows xarray to access a TileDB array."""

    __slots__ = (
        "dtype",
        "shape",
        "variable_name",
        "_array_kwargs",
        "_attr_name",
        "_dim_names",
        "_fill",
        "_index_converters",
    )

    def __init__(
        self,
        *,
        variable_name,
        uri,
        schema,
        attr_key,
        config,
        ctx,
        timestamp,
        fixed_dimensions,
        dimension_sizes,
    ):
        # Set basic properties.
        self.variable_name = variable_name
        self._array_kwargs = {
            "uri": uri,
            "config": config,
            "ctx": ctx,
            "timestamp": timestamp,
        }
        self._dim_names = tuple(dim.name for dim in schema.domain)

        # Check the array.
        if schema.sparse:
            raise ValueError(
                f"Error for variable '{self.variable_name}'; sparse arrays are not "
                f"supported."
            )

        # Check dimensions and get the array shape.
        for dim in schema.domain:
            if dim.domain[0] != 0:
                raise ValueError(
                    f"Error for variable '{self.variable_name}'; dimension "
                    f"'{dim.name}' does not have a domain with lower bound of 0."
                )
            if dim.dtype.kind not in ("i", "u"):
                raise ValueError(
                    f"Error for variable '{self.variable_name}'. Dimension "
                    f"'{dim.name}' has unsupported dtype={dim.dtype}."
                )

        # Set TileDB attribute properties.
        _attr = schema.attr(attr_key)
        self._attr_name = _attr.name
        self.dtype = _attr.dtype
        self._fill = _attr.fill

        # Get the shape.
        if dimension_sizes is None:
            self.shape = schema.shape
        else:
            self.shape = tuple(
                int(dim.domain[1]) + 1
                if dim.name in fixed_dimensions
                else dimension_sizes.get(dim.name, int(dim.domain[1]) + 1)
                for dim in schema.domain
            )

    def __getitem__(self, key):
        # Check the length of the input.
        indices = key.tuple
        if len(indices) != len(self.shape):
            ndim = len(self.shape)
            raise ValueError(
                f"key of length {len(indices)} cannot be used for a TileDB array"
                f" with {ndim} {'dimension' if ndim == 1 else 'dimensions'}"
            )

        # Compute the shape of the output, collapsing any dimensions with scalar input.
        # If a dimension is of length zero, return an appropriately shaped enpty array.
        shape = tuple(
            len(range(dim_size)[index] if isinstance(index, slice) else index)
            for dim_size, index in zip(self.shape, indices)
            if not np.isscalar(index)
        )
        if 0 in shape:
            return np.zeros(shape)

        # Get data from the TileDB array.
        tiledb_indices = tuple(
            _to_zero_based_tiledb_index(self._dim_names[idim], dim_size, index)
            for idim, (dim_size, index) in enumerate(zip(self.shape, indices))
        )
        with tiledb.open(**self._array_kwargs, attr=self._attr_name) as array:
            result = array.multi_index[tiledb_indices][self._attr_name]

        # TileDB multi_index returns the same number of dimensions as the initial array.
        # To match the expected xarray output, we need to reshape the result to remove
        # any dimensions corresponding to scalar-valued input.
        return result.reshape(shape)

    def __setitem__(self, key, value):
        with tiledb.open(**self._array_kwargs, mode="w") as array:
            array[key] = value.astype(dtype=self.dtype)

    @property
    def dim_names(self):
        """A tuple of the dimension names."""
        return self._dim_names

    def get_metadata(self):
        """Returns a dictionary of the variable metadata including xarray specific
        encodings.
        """
        full_key_prefix = f"{_ATTR_PREFIX}{self._attr_name}."
        with tiledb.open(**self._array_kwargs) as array:
            variable_metadata = {"_FillValue": self._fill}
            for key in array.meta:
                if key.startswith(full_key_prefix) and not len(key) == len(
                    full_key_prefix
                ):
                    variable_metadata[key[len(full_key_prefix) :]] = array.meta[key]
            return variable_metadata

    def set_metadata(self, input_meta):
        key_prefix = f"{_ATTR_PREFIX}{self._attr_name}"
        with tiledb.open(**self._array_kwargs, mode="w") as array:
            for key, value in input_meta.items():
                safe_set_metadata(array.meta, f"{key_prefix}.{key}", value)
