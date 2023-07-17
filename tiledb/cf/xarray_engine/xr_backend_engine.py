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


UNLIMITED_DIMENSION_KEY = "__xr_unlimited"
DIMENSION_KEY_PREFIX = "__xr_dim."
RESERVED_GROUP_KEYS = {UNLIMITED_DIMENSION_KEY}
RESERVED_PREFIXES = {DIMENSION_KEY_PREFIX}


class TileDBArrayWrapper(BackendArray):
    # TODO: Make sure slots are okay.
    __slots__ = (
        "dtype",
        "shape",
        "variable_name",
        "_attr_name",
        "_config",
        "_ctx",
        "_schema",
        "_uri",
    )

    def __init__(self, variable_name, uri, config, ctx, unlimited_dimensions):
        self.variable_name = variable_name
        self._uri = uri
        self._config = config
        self._ctx = ctx
        self._schema = tiledb.ArraySchema.load(self._uri, ctx=ctx)

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
        self._attr_name = attr.name
        self.dtype = attr.dtype

        # Check dimensions and get the array shape.
        for dim in self._schema.domain:
            if dim.domain[0] != 0:
                raise ValueError(
                    f"Not a valid TileDB-Xarray group. All dimensions must have "
                    f"a domain with lower bound=0, but dimension '{dim.name}' in "
                    f"array '{self.variable_name}' has lower bound={dim.domain[0]}."
                )
        self.shape = tuple(
            unlimited_dimensions.get(dim.name, dim.domain[1])
            for dim in self._schema.domain
        )

    def __getitem__(self, key):
        # TODO: Test the following indexing types.
        with tiledb.open(
            self.datastore.tiledb_group[self.variable_name],
            mode="r",
            confix=self.datastore.config,
            ctx=self.datastore.ctx,
        ) as array:
            if isinstance(key, indexing.BasicIndexer):
                return array[key.tuple]
            elif isinstance(key, indexing.VectorizedIndexer):
                # TODO: Fix this to return the correct type
                return array.multi_index[
                    indexing._arrayize_vectorized_indexer(key, self.shape).tuple
                ][self._attr_name]
            elif isinstance(key, indexing.OuterIndexer):
                # TODO: Fix this to return the correct type
                return array.multi_index[
                    indexing._arrayize_vectorized_indexer(key, self.shape).tuple
                ][self._attr_name]
            else:
                raise NotImplementedError(
                    f"TileDBArrayWrapper received an unexpected indexer of type "
                    f"{type(key)}"
                )


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
            return ext in {".tiledb-xr"}
        return False


BACKEND_ENTRYPOINTS["tiledb-xr"] = ("tiledb", TileDBXarrayBackendEntrypoint)
