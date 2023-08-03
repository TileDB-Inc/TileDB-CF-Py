"""Module for xarray backend store.

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

import warnings

from xarray.backends.common import AbstractDataStore
from xarray.core import indexing
from xarray.core.utils import FrozenDict
from xarray.core.variable import Variable

import tiledb

from ._common import (
    _ATTR_PREFIX,
    _VARIABLE_ATTR_NAME_PREFIX,
    _VARIABLE_UNLIMITED_DIMS_PREFIX,
)
from .array_wrapper import TileDBArrayWrapper


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

            # Get dimension sizes for the unlimited dimensions.
            dimension_sizes = {}
            self._update_dimensions(array, set(), dimension_sizes)

            # Get one variable from each TileDB attribute.
            variables = {}
            for attr in array.schema:
                array_wrapper = TileDBArrayWrapper(
                    variable_name=attr.name,
                    uri=self._uri,
                    attr_key=attr.name,
                    config=self._config,
                    ctx=self._ctx,
                    fixed_dimension_sizes=dimension_sizes,
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
            dimension_sizes = {}
            wrapper_kwargs = {}
            for item in group:
                # Skip group items that are unnamed or not arrays.
                if item.name is None or item.type is not tiledb.libtiledb.Array:
                    continue

                # Get the schema and dimension sizes.
                with tiledb.open(
                    item.uri,
                    config=self._config,
                    ctx=self._ctx,
                    timestamp=self._timestamp,
                ) as array:
                    attr_key, unlimited_dims = self._pop_variable_encodings(
                        group_metadata, array, item.name
                    )
                    self._update_dimensions(array, unlimited_dims, dimension_sizes)
                    schema = array.schema

                # Get name/index of the TileDB attribute to load.
                # Add the xarray variable.
                wrapper_kwargs[item.name] = {
                    "variable_name": item.name,
                    "uri": item.uri,
                    "attr_key": attr_key,
                    "config": self._config,
                    "ctx": self._ctx,
                    "timestamp": self._timestamp,
                    "schema": schema,
                }

            # Create the xarray variables.
            variables = {}
            for name, kwargs in wrapper_kwargs.items():
                array_wrapper = TileDBArrayWrapper(
                    **kwargs, fixed_dimension_sizes=dimension_sizes
                )
                variables[name] = Variable(
                    dims=array_wrapper.dim_names,
                    data=indexing.LazilyIndexedArray(array_wrapper),
                    attrs=array_wrapper.get_metadata(),
                )
        return FrozenDict(variables), FrozenDict(group_metadata)

    def _pop_variable_encodings(self, group_metadata, array, variable_name):
        # Get attribute name or index.
        key = f"{_VARIABLE_ATTR_NAME_PREFIX}{variable_name}"
        if key in group_metadata:
            _attr_key = group_metadata.pop(key)
            try:
                attr_key = array.schema.attr(_attr_key).name
            except KeyError as err:
                raise KeyError(
                    f"Unable to load variable '{variable_name}'. No attribute "
                    f"matching the key '{_attr_key}' provided in the group "
                    f"metadata."
                ) from err
        else:
            if array.schema.nattr != 1:
                raise ValueError(
                    f"Cannot load variable '{variable_name}'. Missing group "
                    f"metadata '{key}' for the attribute key."
                )
            attr_key = 0
        # Get unlimited dimensions for this variable.
        key = f"{_VARIABLE_UNLIMITED_DIMS_PREFIX}{variable_name}"
        if key in group_metadata:
            unlimited_dims = set(group_metadata.pop(key).split(";"))
        else:
            unlimited_dims = set()
        return attr_key, unlimited_dims

    def _update_dimensions(self, array, unlimited_dimensions, dimension_sizes):
        if any(
            array.schema.domain.has_dim(dim_name) for dim_name in unlimited_dimensions
        ):
            nonempty_domain = array.nonempty_domain()
            for index, dim in enumerate(array.schema.domain):
                if dim.domain[0] != 0:
                    raise ValueError(
                        f"Cannot load  variable '{self.variable_name}'; dimension "
                        f"'{dim.name}' does not have a domain with lower bound of 0."
                    )
                if dim.dtype.kind not in ("i", "u"):
                    raise ValueError(
                        f"Cannot load variable '{self.variable_name}'. Dimension "
                        f"'{dim.name}' has unsupported dtype={dim.dtype}."
                    )
                if dim.name in unlimited_dimensions:
                    dim_size = (
                        0
                        if nonempty_domain is None
                        else int(nonempty_domain[index][1]) + 1
                    )
                    dimension_sizes[dim.name] = max(
                        dim_size, dimension_sizes.get(dim.name, dim_size)
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
