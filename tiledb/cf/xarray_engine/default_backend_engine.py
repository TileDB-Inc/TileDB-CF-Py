"""Module for xarray backend plugin using the TileDB-Xarray Convention.

This plugin will only open groups using the TileDB-Xarray Convention. It has
stricter requirements for the TileDB group and array structures than standard
TileDB. See spec `tiledb-xr-spec.md` in project root.

Example:
  Open a TileDB group with the xarray engine::

    import xarray as xr
    dataset = xr.open_dataset(
        "dataset.tiledb-xr",
        backend_kwargs={"Ctx": ctx},
        engine="tiledb"
    )


"""

import os
import warnings
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

from .special_backend_engine import TileDBDataStore

UNLIMITED_DIMENSION_KEY = "__xr_unlimited"
DIMENSION_KEY_PREFIX = "__xr_dim."  # TODO: Make this more specific
VARIABLE_KEY_PREFIX = "__xr_variable_attribute_name"
RESERVED_GROUP_KEYS = {UNLIMITED_DIMENSION_KEY}
RESERVED_PREFIXES = {DIMENSION_KEY_PREFIX}


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
    dim_name: int
        Name of the dimension. Used for errors.
    dim_size: int
        Size of the dimension as interpreted by xarray. May be smaller than the
        full domain of the TileDB dimension.
    index : Union[int, np.array, slice]
        An integer index, array of integer indices, or a slice for indexing an
        xarray dimension.

    Returns
    -------
    new_index : Union[int, List[int], slice]
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
    # TODO: Add documentation for TileDBArrayWrapper
    __slots__ = (
        "dtype",
        "shape",
        "variable_name",
        "_array_kwargs",
        "_dim_names",
        "_index_converters",
        "_schema",
    )

    def __init__(
        self,
        variable_name,
        attr_key,
        uri,
        *,
        config=None,
        ctx=None,
        timestamp=None,
        dimension_sizes=None,
    ):
        if dimension_sizes is None:
            dimension_sizes = {}
        self.variable_name = variable_name
        self._array_kwargs = {
            "uri": uri,
            "config": config,
            "ctx": ctx,
            "timestamp": timestamp,
        }
        self._schema = tiledb.ArraySchema.load(uri, ctx=ctx)

        if self._schema.sparse:
            raise ValueError(
                f"Cannot load variable '{self.variable_name}'; sparse arrays are not "
                f"supported."
            )

        # Set TileDB attribute properties.
        attr = self._schema.attr(attr_key)
        self._array_kwargs["attr"] = attr.name
        self.dtype = attr.dtype

        # Check dimensions and get the array shape.
        for dim in self._schema.domain:
            if dim.domain[0] != 0:
                raise ValueError(
                    f"Cannot load variable '{self.variable_name}'; dimension "
                    f"'{dim.name}' does not have a domain with lower bound of 0."
                )
            if dim.dtype.kind not in ("i", "u"):
                raise ValueError(
                    f"Cannot load variable '{self.variable_name}'. Dimension "
                    f"'{dim.name}' has unsupported dtype={dim.dtype}."
                )
        self.shape = tuple(
            dimension_sizes.get(dim.name, int(dim.domain[1]) + 1)
            for dim in self._schema.domain
        )
        self._dim_names = tuple(dim.name for dim in self._schema.domain)

    def __getitem__(self, indexer):
        # Check the length of the input.
        indices = indexer.tuple
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
        with tiledb.open(**self._array_kwargs) as array:
            result = array.multi_index[tiledb_indices][self._array_kwargs["attr"]]

        # TileDB multi_index returns the same number of dimensions as the initial array.
        # To match the expected xarray output, we need to reshape the result to remove
        # any dimensions corresponding to scalar-valued input.
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
        uri,
        config=None,
        ctx=None,
        timestamp=None,
    ):
        # Set input properties
        self._uri = uri
        self._config = config
        self._ctx = ctx
        object_type = tiledb.object_type(self._uri, ctx=self._ctx)
        if object_type == "group":
            self._is_group = True
            self._timestamp = None
            if timestamp is not None:
                warnings.warn(
                    "Ignoring keyword `timestamp`. Time traveling is not supported "
                    "for the xarray backend on groups.",
                    stacklevel=1,
                )
        elif object_type == "array":
            self._is_group = False
            self._timestamp = timestamp
        else:
            raise ValueError(
                f"Failed to open dataset using `tiledb-xr` engine. There is not a "
                f"valid TileDB Group at provided location '{self._uri}'."
            )

    def _load_array(self):
        """This is the method used to load the dataset."""
        with tiledb.open(
            self._uri,
            mode="r",
            config=self._config,
            ctx=self._ctx,
            timestamp=self._timestamp,
        ) as array:
            # Get group level metadata
            group_metadata = {key: val for key, val in array.meta.items()}
            unlimited_dimensions = self._pop_unlimited_dimensions(group_metadata)

            # Get unlimited dimensions.
            # -- This also removes unlimited dimension encoding data.
            unlimited_dimensions = {}
            if UNLIMITED_DIMENSION_KEY in group_metadata:
                unlim_dim_names = group_metadata.pop(UNLIMITED_DIMENSION_KEY)
                for dim_name in unlim_dim_names:
                    key = f"{DIMENSION_KEY_PREFIX}{dim_name}"
                    if key not in group_metadata:
                        raise KeyError(
                            f"Invalid TileDB-Xarray group. Missing size for unlimited "
                            f"dimension '{dim_name}'."
                        )
                    unlimited_dimensions[dim_name] = group_metadata.pop(key)

            # Get one variable from each TileDB array.
            variables = {}
            var_dims = tuple(dim.name for dim in array.schema.domain)
            print(f"DIMS: {var_dims}")
            for attr in array.schema:
                # TODO: Pop variable metadata
                var_meta = {}
                var_data = indexing.LazilyIndexedArray(
                    TileDBArrayWrapper(
                        variable_name=attr.name,
                        uri=self._uri,
                        config=self._config,
                        ctx=self._ctx,
                        timestamp=self._timestamp,
                        dimension_sizes=unlimited_dimensions,
                        attr_key=attr.name,
                    )
                )
                variables[attr.name] = Variable(
                    dims=var_dims, data=var_data, attrs=var_meta
                )
        return FrozenDict(variables), FrozenDict(group_metadata)

    def _load_group(self):
        """This is the method used to load the dataset."""
        with tiledb.Group(
            self._uri,
            mode="r",
            config=self._config,
            ctx=self._ctx,
        ) as group:
            # Get group level metadata
            group_metadata = {key: val for key, val in group.meta}
            unlimited_dimensions = self._pop_unlimited_dimensions(group_metadata)
            # TODO: Add decoder for attribute names

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
                    key = f"__xr_variable_attribute_name.{item.name}"
                    if key in group_metadata:
                        attr_key = group_metadata.pop(key)
                    else:
                        if array.schema.nattr != 1:
                            raise ValueError(
                                f"Cannot load variable '{item.name}'. Missing group "
                                f"metadata '{key}' for the attribute key."
                            )
                        attr_key = 0
                    var_data = indexing.LazilyIndexedArray(
                        TileDBArrayWrapper(
                            variable_name=item.name,
                            uri=item.uri,
                            attr_key=attr_key,
                            timestamp=self._timestamp,
                            config=self._config,
                            ctx=self._ctx,
                            dimension_sizes=unlimited_dimensions,
                        )
                    )
                    variables[item.name] = Variable(
                        dims=var_dims, data=var_data, attrs=var_meta
                    )
        return FrozenDict(variables), FrozenDict(group_metadata)

    def _pop_unlimited_dimensions(self, meta):
        """Separate unlimited dimension encodings from general metadata.."""
        unlimited_dimensions = {}
        if UNLIMITED_DIMENSION_KEY in meta:
            unlim_dim_names = meta.pop(UNLIMITED_DIMENSION_KEY)
            for dim_name in unlim_dim_names:
                key = f"{DIMENSION_KEY_PREFIX}{dim_name}"
                if key not in meta:
                    raise KeyError(
                        f"Invalid TileDB-Xarray group. Missing size for unlimited "
                        f"dimension '{dim_name}'."
                    )
                unlimited_dimensions[dim_name] = meta.pop(key)
        return unlimited_dimensions

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
        if self._is_group:
            return self._load_group()
        return self._load_array()

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
        timestamp=None,
        open_full_domain=False,
        use_deprecated_engine=False,
        key=None,
        encode_fill=False,
        coord_dims=None,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables: str | Iterable[str] | None = None,
        use_cftime=None,
        decode_timedelta=None,
    ) -> Dataset:
        """
        Open a TileDB group or array as an xarray dataset.


        TODO: Full description of usage.

        Parameters
        ----------
        filename_or_obj: TileDB URI for the group or array to open in xarray.
        config: TileDB config object to pass to TileDB objects.
        ctx: TileDB context to use for TileDB operations.
        timestamp: Timestamp to use opening a TileDB array. Not supported on groups.
        open_full_domain: If ``True``, use the full dimension to define the size of
            dimensions that aren't otherwise specified. If ``False``, use the non-empty
            domain of the array (computed when the dataset is first loaded).
        key: [Deprecated] Encryption key to use for the backend array.
        encode_fill: [Deprecated] Encode the TileDB fill using `_FillValue` if ``True``.
        coord_dims: [Deprecated] Interpret the following dimension as coordinates.
        """

        if use_deprecated_engine:
            warnings.warn(
                "Using deprecated TileDB-Xarray plugin",
                DeprecationWarning,
                stacklevel=1,
            )
            datastore = TileDBDataStore(
                uri=filename_or_obj,
                key=key,
                timestamp=timestamp,
                ctx=ctx,
                encode_fill=encode_fill,
                open_full_domain=open_full_domain,
                coord_dims=coord_dims,
            )
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

        # Warn if a deprecated keyword was used.
        if key is not None:
            warnings.warn(
                "Deprecated keyword 'key' provided. To use the deprecated backend "
                "engine set `use_deprecated_engine=True`",
                DeprecationWarning,
                stacklevel=1,
            )
        if encode_fill:
            warnings.warn(
                "Deprecated keyword 'encode_fill' provided. To use the deprecated "
                "backend engine set `use_deprecated_engine=True`",
                DeprecationWarning,
                stacklevel=1,
            )
        if coord_dims is not None:
            warnings.warn(
                "Deprecated keyword 'coord_dims' provided. To use the deprecated  "
                "backend engine set `use_deprecated_engine=True`",
                DeprecationWarning,
                stacklevel=1,
            )

        try:
            datastore = TileDBXarrayStore(
                filename_or_obj, config=config, ctx=ctx, timestamp=timestamp
            )

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
        except ValueError as err:
            raise ValueError(
                "Failed to open with current TileDB-xarray backend. To use the "
                "old TileDB-xarray backend set `use_deprecated_engine=True`"
            ) from err

    def guess_can_open(self, filename_or_obj) -> bool:
        """Check for datasets that can be opened with this backend."""
        if isinstance(filename_or_obj, (str, os.PathLike)):
            _, ext = os.path.splitext(filename_or_obj)
            if ext in {".tiledb", ".tdb", ".tdb-xr"}:
                return True
        try:
            return tiledb.object_type(filename_or_obj) in {"array", "group"}
        except tiledb.TileDBError:
            return False


BACKEND_ENTRYPOINTS["tiledb"] = ("tiledb", TileDBXarrayBackendEntrypoint)
