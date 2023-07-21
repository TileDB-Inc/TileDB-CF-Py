"""Module for xarray backend plugin using the TileDB-Xarray Convention.

This plugin will only open groups using the TileDB-Xarray Convention. It has
stricter requirements for the TileDB group and array structures than standard
TileDB. See spec `tiledb-xr-spec.md` in project root.

Example:
  Open a TileDB group with the xarray engine::

    import xarray as xr
    dataset = xr.open_dataset(
        "dataset.tiledb-xr",
        backend_kwargs={"key": key, "timestamp": timestamp},
        engine="tiledb-xr"
    )


"""

import os
from typing import Iterable, ClassVar

import tiledb
import numpy as np
from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    AbstractWritableDataStore,
    BackendArray,
    BackendEntrypoint,
    ArrayWriter,
)
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import (
    FrozenDict,
    close_on_error,
)
from xarray.core.variable import Variable
from xarray.core.dataset import Dataset
from xarray.core.pycompat import integer_types


UNLIMITED_DIMENSION_KEY = "__xr_unlimited"
DIMENSION_KEY_PREFIX = "__xr_dim."
RESERVED_GROUP_KEYS = {UNLIMITED_DIMENSION_KEY}
RESERVED_PREFIXES = {DIMENSION_KEY_PREFIX}


class TileDBIndexConverter:
    """Converter from xarray-style indices to TileDB-style coordinates.

    This class converts the values contained in an xarray ExplicitIndexer tuple to
    values usable for indexing a TileDB Dimension. The xarray ExplicitIndexer uses
    standard 0-based integer indices to look-up values in DataArrays and variables;
    whereas, Tiledb accesses values directly from the dimension coordinates (analogous
    to looking-up values by "location" in xarray).

    The max value of the index is defined by the maximum of the non-empty domain at
    creation time.

    The following is assumed about xarray indices:
       * An index may be an integer, a slice, or a Numpy array of integer indices.
       * An integer index or component of an array is such that -size <= value < size.
         * Non-negative values are a standard zero-based index.
         * Negative values count backwards from the end of the array with the last value
           of the array starting at -1.
    """

    def __init__(self, dim, domain):
        if dim.dtype.kind not in ("i", "u"):
            raise NotImplementedError(
                f"support for reading TileDB arrays with a dimension of type "
                f"{dim.dtype} is not implemented"
            )

        if dim.domain[0] != 0:
            raise ValueError(
                f"Not a valid TileDB-Xarray group. All dimensions must have "
                f"a domain with lower bound=0, but dimension '{dim.name}' in "
                f"array '{self.variable_name}' has lower bound={dim.domain[0]}."
            )

        self.dtype = dim.dtype
        self.size = int(dim.domain[1]) + 1

    def __getitem__(self, index):
        """Converts an xarray integer, array, or slice to an index object usable by the
            TileDB multi_index function.

        Parameters
        ----------
        index : Union[int, np.array, slice]
            An integer index, array of integer indices, or a slice for indexing an
            xarray dimension.

        Returns
        -------
        new_index : Union[self.dtype, List[self.dtype], slice]
            A value of type `self.dtype`, a list of values, or a slice for indexing a
            TileDB dimension using mulit_index.
        """
        if isinstance(index, integer_types):
            # Convert xarray index to TileDB dimension coordinate
            if not -self.size <= index < self.size:
                raise IndexError(f"index {index} out of bounds for {type(self)}")
            return self.to_coordinate(index)

        if isinstance(index, slice) and index.step in (1, None):
            # Convert from index slice to coordinate slice (note that xarray
            # includes the starting point and excludes the ending point vs. TileDB
            # multi_index which includes both the staring point and ending point).
            index = range(self.size)[index]
            if index.step in (1, None):
                return slice(index.start, index.stop - 1)

        # Convert slice or array of xarray indices to list of TileDB dimension
        # coordinates
        return list(self.to_coordinates(index))

    def to_coordinate(self, index):
        """Converts an xarray index to a coordinate for the TileDB dimension.

        Parameters
        ----------
        index : int
            An integer index for indexing an xarray dimension.

        Returns
        -------
        new_index : self.dtype
            A `self.dtype` coordinate for indexing a TileDB dimension.
        """
        return index if index >= 0 else index + self.size - 1

    def to_coordinates(self, index):
        """
        Converts an xarray-style slice or Numpy array of indices to an array of
        coordinates for the TileDB dimension.

        Parameters
        ----------
        index : Union[slice, np.ndarray]
            A slice or an array of integer indices for indexing an xarray dimension.

        Returns
        -------
        new_index : Union[np.ndarray]
            An array of `self.dtype` coordinates for indexing a TileDB dimension.
        """
        if isinstance(index, slice):
            # Using range handles negative start/stop, out-of-bounds, and None values.
            index = range(self.size)[index]
            return np.arange(index.start, index.stop, index.step)

        if isinstance(index, np.ndarray):
            if np.isscalar(index):
                return index if index >= 0 else index + self.size - 1
            if index.ndim != 1:
                raise TypeError(
                    f"invalid indexer array for {type(self)}; input array index must "
                    f"have exactly 1 dimension"
                )
            # vectorized version of self.to_coordinate
            if not ((-self.size <= index).all() and (index < self.size).all()):
                raise IndexError(f"index {index} out of bounds for {type(self)}")
            return index + np.where(index >= 0, 0, self.size - 1)

        raise TypeError(f"unexpected indexer type for {type(self)}")


class TileDBArrayWrapper(BackendArray):
    # TODO: Make sure slots are okay.
    __slots__ = (
        "dtype",
        "shape",
        "variable_name",
        "_array_kwargs",
        "_index_converters",
        "_schema",
    )

    def __init__(self, variable_name, uri, config, ctx, unlimited_dimensions):
        self.variable_name = variable_name
        self._array_kwargs = {"uri": uri, "config": config, "ctx": ctx}
        self._schema = tiledb.ArraySchema.load(uri, ctx=ctx)

        if self._schema.sparse:
            raise ValueError(
                f"Not a valid TileDB-Xarray group. Arrays must be dense, but array "
                f"'{self.variable_name}' is sparse."
            )

        # Check attributes are valid and get attribute properties.
        # Note: A TileDB attribute is roughly equivalent to a xarray/NetCDF variable.
        if self._schema.nattr != 1:
            raise ValueError(
                f"Not a valid TileDB-Xarray group. Each array must have exactly one "
                f"TileDB attribute, but array '{self.variable_name}' has "
                f"{self._schema.nattr} attributes."
            )
        attr = self._schema.attr(0)
        self._array_kwargs["attr"] = attr.name
        self.dtype = attr.dtype

        # Check dimensions and get the array shape.
        for dim in self._schema.domain:
            if dim.domain[0] != 0:
                raise ValueError(
                    f"Not a valid TileDB-Xarray group. All dimensions must have "
                    f"a domain with lower bound=0, but dimension '{dim.name}' in "
                    f"array '{self.variable_name}' has lower bound={dim.domain[0]}."
                )
        self._index_converters = tuple(
            TileDBIndexConverter(
                dim, [0, unlimited_dimensions.get(dim.name, dim.domain[1])]
            )
            for dim in self._schema.domain
        )
        self.shape = tuple(converter.size for converter in self._index_converters)

    def __getitem__(self, indexer):
        xarray_indices = indexer.tuple
        if len(xarray_indices) != len(self._index_converters):
            raise ValueError(
                f"key of length {len(xarray_indices)} cannot be used for a TileDB array"
                f" of length {len(self._index_converters)}"
            )
        # Compute the shape, collapsing any dimensions with integer input.
        shape = tuple(
            len(range(converter.size)[index] if isinstance(index, slice) else index)
            for index, converter in zip(xarray_indices, self._index_converters)
            if not isinstance(index, integer_types)
        )
        # TileDB multi_index does not except empty arrays/slices. If a dimension is
        # length zero, return an empty numpy array of the correct length.
        if 0 in shape:
            return np.zeros(shape)
        tiledb_indices = tuple(
            converter[index]
            for index, converter in zip(xarray_indices, self._index_converters)
        )
        with tiledb.open(**self._array_kwargs) as array:
            result = array.multi_index[tiledb_indices][self._array_kwargs["attr"]]
        # Note: TileDB multi_index returns the same number of dimensions as the initial
        # array. To match the expected xarray output, we need to reshape the result to
        # remove any dimensions corresponding to scalar-valued input.
        return result.reshape(shape)


class TileDBXarrayStore(AbstractWritableDataStore):
    """Store for reading and writing data via TileDB using the TileDB-xarray
    specification.

    TODO: document parameters
    """

    # TODO: Set slots
    # __slots__ = ()

    def __init__(
        self,
        group_uri,
        config=None,
        ctx=None,
    ):
        # Set input properties
        self._group_uri = group_uri
        self._config = config
        self._ctx = ctx
        if tiledb.object_type(self._group_uri, ctx=self._ctx) != "group":
            raise ValueError(
                f"Failed to open dataset using `tiledb-xr` engine. There is not a "
                f"valid TileDB Group at provided location '{self._group_uri}'."
            )

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        pass

    def encode(self, variables, attributes):  # pragma: no cover
        """Disabled function from the parent class"""
        raise NotImplementedError()

    def get_attrs(self):  # pragma: no cover
        """Disabled function from the parent class"""
        raise NotImplementedError()

    def get_dimensions(self):  # pragma: no cover
        """Disabled function from the parent class"""
        raise NotImplementedError()

    def get_encoding(self):
        """Return special encoding information for xarray backend."""
        return {}

    def get_variables(self):
        """Disabled function from the parent class"""
        raise NotImplementedError()

    def load(self):
        """This is the method used to load the dataset."""
        with tiledb.Group(
            self._group_uri,
            mode="r",
            config=self._config,
            ctx=self._ctx,
        ) as group:
            # Get group level metadata
            group_metadata = {key: val for key, val in group.meta}

            # Get unlimited dimensions.
            # -- This also removes unlimited dimension encoding data.
            unlimited_dimensions = {}
            if UNLIMITED_DIMENSION_KEY in group_metadata:
                unlim_dim_names = group_metadata.pop(UNLIMITED_DIMENSION_KEY)
                for dim_name in unlim_dim_names:
                    key = f"{DIMENSION_KEY_PREFIX}{dim_name}"
                    if key not in group.meta:
                        raise KeyError(
                            f"Invalid TileDB-Xarray group. Missing size for unlimited "
                            f"dimension '{dim_name}'."
                        )
                    unlimited_dimensions[dim_name] = group_metadata.pop(key)

            # Get one variable from each TileDB array.
            variables = {}
            for item in group:
                if item.name is None:
                    continue
                if item.type is not tiledb.libtiledb.Array:
                    continue
                with tiledb.open(
                    item.uri, mode="r", config=self._config, ctx=self._ctx
                ) as array:
                    var_meta = {key: val for key, val in array.meta}
                    var_dims = (dim.name for dim in array.schema.domain)
                var_data = indexing.LazilyIndexedArray(
                    TileDBArrayWrapper(
                        variable_name=item.name,
                        uri=item.uri,
                        config=self._config,
                        ctx=self._ctx,
                        unlimited_dimensions=unlimited_dimensions,
                    )
                )
                variables[item.name] = Variable(
                    dims=var_dims, data=var_data, attrs=var_meta
                )
        return FrozenDict(variables), FrozenDict(group_metadata)

    def encode_variable(self, v):
        """encode one variable"""
        return v

    def encode_attribute(self, a):
        """encode one attribute"""
        return a

    def store(
        self,
        variables,
        attributes,
        check_encoding_set=frozenset(),
        writer=None,
        unlimited_dims=None,
    ):
        """
        Top level method for putting data on this store, this method:
          - encodes variables/attributes
          - sets dimensions
          - sets variables

        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        attributes : dict-like
            Dictionary of key/value (attribute name / attribute) pairs
        check_encoding_set : list-like
            List of variables that should be checked for invalid encoding
            values
        writer : ArrayWriter
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
        """
        if writer is None:
            writer = ArrayWriter()

        # TODO: Write to TileDB Group
        raise NotImplementedError()

    def set_dimension(self, dim, length):  # pragma: no cover
        """Disabled function from the parent class"""
        raise NotImplementedError()

    def set_attribute(self, k, v):  # pragma: no cover
        """Disabled function from the parent class"""
        raise NotImplementedError()

    def set_variable(self, k, v):  # pragma: no cover
        """Disabled function from the parent class"""
        raise NotImplementedError()

    def set_attributes(self, attributes):  # pragma: no cover
        """Disabled function from the parent class"""
        raise NotImplementedError()

    def set_variables(
        self, variables, check_encoding_set, writer, unlimited_dims=None
    ):  # pragma: no cover
        """Disabled function from the parent class"""
        raise NotImplementedError()

    def set_dimensions(self, variables, unlimited_dims=None):  # pragma: no cover
        """Disabled function from the parent class"""
        raise NotImplementedError()


class TileDBXarrayBackendEntrypoint(BackendEntrypoint):
    """
    TODO: Add docs for TileDBXarrayBackendEntrypoint
    """

    open_dataset_parameters: ClassVar[tuple | None] = [
        "filename_or_obj",
        "drop_variables",
        "config",
        "ctx",
    ]
    description: ClassVar[
        str
    ] = "TileDB backend for xarray using the TileDB-Xarray specification"
    url: ClassVar[str] = "https://github.com/TileDB-Inc/TileDB-CF-Py"

    def open_dataset(
        self,
        filename_or_obj,
        *,
        config=None,
        ctx=None,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables: str | Iterable[str] | None = None,
        use_cftime=None,
        decode_timedelta=None,
    ) -> Dataset:
        """
        TODO: Document open_dataset method in TileDBXarrayBackendEntrypoint

        """

        # TODO: Add in xarray encodings as is appropriate.
        datastore = TileDBXarrayStore(filename_or_obj, config=config, ctx=ctx)

        # Xarray indirection to open dataset defined in a plugin.
        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(datastore):
            dataset = store_entrypoint.open_dataset(
                datastore,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
        return dataset

    def guess_can_open(self, filename_or_obj) -> bool:
        """ """
        if isinstance(filename_or_obj, (str, os.PathLike)):
            _, ext = os.path.splitext(filename_or_obj)
            if ext in {".tiledb-xr"}:
                return True
        try:
            return tiledb.object_type(filename_or_obj) == "group"
        except tiledb.TileDBError:
            return False


BACKEND_ENTRYPOINTS["tiledb-xr"] = ("tiledb-xr", TileDBXarrayBackendEntrypoint)
