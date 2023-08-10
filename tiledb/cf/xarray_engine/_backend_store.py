"""Module for xarray backend store.

Example:
  Open a TileDB group with the xarray engine::

    import xarray as xr
    dataset = xr.open_dataset(
        "dataset.tiledb",
        backend_kwargs={"Ctx": ctx},
        engine="tiledb"
    )


"""

import warnings

from xarray.backends.common import AbstractDataStore
from xarray.core import indexing
from xarray.core.utils import FrozenDict
from xarray.core.variable import Variable

import tiledb

from ._array_wrapper import TileDBArrayWrapper
from ._common import _ARRAY_FIXED_DIMS_PREFIX, _ATTR_PREFIX


class TileDBXarrayStore(AbstractDataStore):
    """Store for reading and writing data via TileDB using the TileDB-xarray
    specification.

    Parameters:
        uri: URI of the TileDB group or array to read.
        config: TileDB configuration to use for the group and all arrays.
        ctx: TileDB context to use for TileDB operations.
        timestamp: Timestamp to read the array at. Not supported on groups.
    """

    __slots__ = ("_config", "_ctx", "_timestamp", "_uri")

    def __init__(
        self,
        uri,
        *,
        config=None,
        ctx=None,
        timestamp=None,
    ):
        # Set input properties
        self._uri = uri
        self._config = config
        self._ctx = ctx
        self._timestamp = timestamp

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

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
            group_metadata = {
                key: val
                for key, val in array.meta.items()
                if not key.startswith(_ATTR_PREFIX)
            }

            dim_sizes = {}
            self._update_dimension_sizes(array, set(), dim_sizes)

            # Get the variables.
            variables = {}
            for attr in array.schema:
                array_wrapper = TileDBArrayWrapper(
                    variable_name=attr.name,
                    uri=self._uri,
                    attr_key=attr.name,
                    config=self._config,
                    ctx=self._ctx,
                    dimension_sizes=dim_sizes,
                    fixed_dimensions=set(),
                    schema=array.schema,
                    timestamp=self._timestamp,
                )
                variables[attr.name] = Variable(
                    dims=array_wrapper.dim_names,
                    data=indexing.LazilyIndexedArray(array_wrapper),
                    attrs=array_wrapper.get_metadata(),
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
            group_metadata = {key: val for key, val in group.meta.items()}

            # Pre-process information for creating variales.
            dim_sizes = {}
            wrapper_kwargs = {}
            for item in group:
                # Skip group items that are unnamed or not arrays.
                if item.type is not tiledb.libtiledb.Array:
                    continue

                # Get the schema and dimension sizes.
                with tiledb.open(
                    item.uri,
                    config=self._config,
                    ctx=self._ctx,
                    timestamp=self._timestamp,
                ) as array:
                    if item.name is not None:
                        fixed_dims = self._pop_variable_encodings(
                            group_metadata, item.name
                        )
                    else:
                        fixed_dims = set()
                    self._update_dimension_sizes(array, fixed_dims, dim_sizes)
                    schema = array.schema

                # Get name/index of the TileDB attribute to load.
                # Add the xarray variable.
                if item.name is not None and schema.nattr == 1:
                    array_variables = {item.name: 0}
                else:
                    array_variables = {attr.name: attr.name for attr in schema}

                for var_name, attr_key in array_variables.items():
                    if var_name in wrapper_kwargs:
                        raise ValueError(
                            f"Cannot load group. It contains multiple variables with "
                            f"the name {var_name}."
                        )
                    wrapper_kwargs[var_name] = {
                        "uri": item.uri,
                        "schema": schema,
                        "attr_key": attr_key,
                        "config": self._config,
                        "ctx": self._ctx,
                        "timestamp": self._timestamp,
                        "fixed_dimensions": fixed_dims,
                    }

            # Create the xarray variables.
            variables = {}
            for name, kwargs in wrapper_kwargs.items():
                array_wrapper = TileDBArrayWrapper(
                    variable_name=name, dimension_sizes=dim_sizes, **kwargs
                )
                variables[name] = Variable(
                    dims=array_wrapper.dim_names,
                    data=indexing.LazilyIndexedArray(array_wrapper),
                    attrs=array_wrapper.get_metadata(),
                )
        return FrozenDict(variables), FrozenDict(group_metadata)

    def _pop_variable_encodings(self, group_metadata, array_name):
        # Get fixed dimensions for this array.
        key = f"{_ARRAY_FIXED_DIMS_PREFIX}{array_name}"
        if key in group_metadata:
            fixed_dims = set(group_metadata.pop(key).split(";"))
        else:
            fixed_dims = set()
        return fixed_dims

    def _update_dimension_sizes(self, array, fixed_dims, dim_sizes):
        """Updates the map of dimension sizes to be the size of the smallest non-empty
        domain for all dimensions in the array that are not fixed dimesions.
        """
        nonempty_domain = array.nonempty_domain()
        for index, dim in enumerate(array.schema.domain):
            if dim.domain[0] != 0 or dim.dtype.kind not in ("i", "u"):
                # Skip update. Error messages will be thrown later with more
                # specific information for the user when attempting to load
                # attributes from this array.
                return

            if dim.name not in fixed_dims:
                # Set the dimension size to be the smallest non-empty domain.
                nonempty_size = (
                    0 if nonempty_domain is None else int(nonempty_domain[index][1]) + 1
                )
                dim_sizes[dim.name] = min(
                    nonempty_size, dim_sizes.get(dim.name, nonempty_size)
                )

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
        object_type = tiledb.object_type(self._uri, ctx=self._ctx)
        if object_type == "group":
            if self._timestamp is not None:
                warnings.warn(
                    "Setting `timestamp=None`. Time traveling is not supported "
                    "on groups for the TileDB-xarray backend engine.",
                    stacklevel=1,
                )
            self._timestamp = None
            return self._load_group()
        elif object_type == "array":
            return self._load_array()
        else:
            raise ValueError(
                f"Failed to open dataset using `tiledb-xr` engine. There is not a "
                f"valid TileDB group or array at provided location '{self._uri}'."
            )
