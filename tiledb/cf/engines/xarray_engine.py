# Copyright 2020 TileDB Inc.
# Licensed under the MIT License.

"""Module for using TileDB as an xarray backend plugin.

Example:
  Open a dense TileDB array with the xarray engine::

    import xarray as xr
    dataset = xr.open_dataset(
        "tiledb_array_uri",
        backend_kwargs={"key": key, "timestamp": timestamp},
        engine="tiledb"
    )
"""

from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from affine import Affine

from xarray.core import indexing
from xarray.core.pycompat import integer_types
from xarray.core.utils import FrozenDict, HiddenKeyDict, close_on_error
from xarray.core.variable import Variable
from xarray.backends.file_manager import DummyFileManager, CachingFileManager

from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    _normalize_path,
)

from xarray.backends.store import StoreBackendEntrypoint

try:
    import tiledb
    has_tiledb = True
except:
    has_tiledb = False

from ..creator import DATA_SUFFIX, INDEX_SUFFIX

_ATTR_PREFIX = "__tiledb_attr."
_DIM_PREFIX = "__tiledb_dim."
_COORD_SUFFIX = ".data"


class TileDBDenseArrayWrapper(BackendArray):
    """A backend array wrapper for a TileDB attribute.

    This class is not intended to accessed directly. Instead it should be used
    through a :class:`LazilyIndexedArray` object.
    """
    def __init__(self, variable_name, datastore):

        self.datastore = datastore
        self.variable_name = variable_name
        tdbarr = self.datastore.ds
        tdb_attr = tdbarr.attr(self.variable_name)
        if tdb_attr.isanon:
            raise NotImplementedError(
                "Support for anonymous TileDB attributes has not been implemented yet"
            )

        array = self.get_array()

        dtype_kind = array.dtype.kind
        if dtype_kind not in "iuM":
            raise NotImplementedError(
                f"support for reading TileDB arrays with a dimension "
                f"of type {array.dtype}"
            )

        self.dtype = array.dtype
        self.shape = array.shape
        # might need to add test for datetime dtype and adjust

    def get_array(self):
        return self.datastore.ds[:][self.variable_name]

    def __getitem__(self, key):
        array = self.get_array()
        if isinstance(key, indexing.BasicIndexer):
            return array[key.tuple]
        # elif isinstance(key, indexing.VectorizedIndexer):
        #     return array.vindex[
        #         indexing._arrayize_vectorized_indexer(key, self.shape).tuple
        #     ]
        # else:
        #     assert isinstance(key, indexing.OuterIndexer)
        #     return array.oindex[key.tuple]
        else:
            raise NotImplementedError("fancy indexing not implemented yet")


def parse_var(name):
    if name.endswith(DATA_SUFFIX):
        name = name[: -len(DATA_SUFFIX)]
    elif name.endswith(INDEX_SUFFIX):
        name = name[: -len(INDEX_SUFFIX)]
    return name

def verify_and_open_group(path):
    pass

class TileDBDataStore(AbstractDataStore):
    #     __slots__()

    def __init__(self, manager):
        self._manager = manager

    @classmethod
    def open(
        cls,
        filename,
        mode="r",
        tdb_key=None,
        timestamp=None,
        ctx=None,
    ):
        manager = CachingFileManager(tiledb.open,
                                     filename,
                                     # using same mode for manager and tdb
                                     # could raise error if allowed modes are not same
                                     # might need to split to manager_mode and tdb_mode
                                     mode=mode,
                                     kwargs={
                                         "mode": mode,
                                         "key": tdb_key,
                                         "timestamp": timestamp,
                                         "ctx": ctx,
                                     })
        return cls(manager)

    def _acquire(self, needs_lock=False):
        with self._manager.acquire_context(needs_lock) as tdbarr:
            ds = tdbarr
        return ds

    @property
    def ds(self):
        return self._acquire()

    def get_data_var(self, variable_name, metadata):
        variable_name = parse_var(variable_name)
        dims = self.get_dimensions()
        data = indexing.LazilyIndexedArray(
            TileDBDenseArrayWrapper(variable_name, self)
        )
        variable = Variable(dims, data, metadata)
        return variable

    def get_coord_var(self, dim, metadata):
        coord_name = parse_var(dim.name)
        coord_size = dim.size
        dims = {coord_name: coord_size}

        x_names = ('x', 'w', 'width')
        y_names = ('y', 'h', 'height')
        band_names = ('band', 'bands', 'count')

        coord_data = None
        if coord_name.lower() in (*x_names, *y_names, *band_names):
            try:
                GeoTransform = self.ds.meta.get('GeoTransform')
            except:
                GeoTransform = None
            if GeoTransform is not None:
                # assumes GeoTransform is stored as List[float,...]
                transform = Affine.from_gdal(*GeoTransform)

                if coord_name.lower() in x_names:
                    coord_data, _ = transform * \
                                    np.meshgrid(np.arange(coord_size) + 0.5, np.zeros(coord_size) + 0.5)
                    coord_data = coord_data[0, :]

                elif coord_name.lower() in y_names:
                    _, coord_data = transform * \
                                    np.meshgrid(np.zeros(coord_size) + 0.5, np.arange(coord_size) + 0.5)
                    coord_data = coord_data[:, 0]

                elif coord_name.lower() in band_names:
                    coord_data = np.arange(1, coord_size + 1, dtype=np.int64)

        # coord name is not in x, y, or band names or wasn't parsed correctly, then a standard dim coord
        # is set up with range(length_dim) and dtype np.int64

        if coord_data is None:
            min_value = dim.domain[0]
            max_value = dim.domain[1]
            dtype = dim.dtype
            if metadata and "time reference" in metadata:
                start_date = np.datetime64(metadata["time reference"])
                freq = metadata["freq"]
                coord_data = pd.date_range(start_date, periods=dim.size, freq=freq)
                dtype = f"datetime64[{freq}]"
            else:
                if min_value == 1:
                    min_value = 0
                    max_value = max_value - 1
                coord_data = np.arange(min_value, max_value + 1, dtype=dtype)

        variable = Variable(dims, coord_data, metadata)
        return variable

    def get_variables(self):
        variable_metadata = self.get_variable_metadata()
        data_vars = {attr.name: self.get_data_var(attr.name,
                                                  variable_metadata.get(attr.name))
                     for attr in self.ds.schema}
        coords_vars = {dim.name: self.get_coord_var(dim,
                                                    variable_metadata.get(dim.name))
                       for dim in self.ds.schema.domain}
        return FrozenDict({**data_vars, **coords_vars})

    def get_attrs(self):
        meta = self.ds.meta
        attrs = {
            key: meta[key]
            for key in meta.keys()
            if not key.startswith((_ATTR_PREFIX, _DIM_PREFIX))
        }
        return attrs

    def get_dimensions(self):
        dims = {dim.name: dim.size for dim in self.ds.schema.domain}
        return FrozenDict(dims)

    def get_variable_metadata(self):
        variable_metadata = defaultdict(dict)
        meta = self.ds.meta
        for key in meta.keys():
            if key.startswith((_ATTR_PREFIX, _DIM_PREFIX)):
                last_dot_ix = key.rindex(".")
                attr_name = key[key.index(".") + 1: last_dot_ix]
                if not attr_name:
                    raise RuntimeError(
                        f"cant parse attribute metadata '{key}' with "
                        "missing name or key value"
                    )
                attr_key = key[last_dot_ix + 1:]
                variable_metadata[attr_name][attr_key] = meta[key]
        return variable_metadata


class TileDBBackendEntrypoint(BackendEntrypoint):
    available = has_tiledb

    def guess_can_open(self, filename_or_obj):
        try:
            return tiledb.object_type(filename_or_obj) == "array"
        except tiledb.TileDBError:
            return False

    def open_dataset(
        self,
        filename_or_obj,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=None,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        mode="r",
        tdb_key=None,
        timestamp=None,
        ctx=None,
    ):
        filename_or_obj = _normalize_path(filename_or_obj)
        store = TileDBDataStore.open(filename_or_obj, mode=mode, tdb_key=tdb_key, timestamp=timestamp, ctx=ctx)
        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
        return ds
