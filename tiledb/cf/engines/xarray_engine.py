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
from typing import Dict, Any, Hashable, Optional
import numpy as np
import xarray as xr
import pandas as pd
from affine import Affine

from xarray.core import indexing
from xarray.core.utils import FrozenDict, close_on_error
from xarray.core.variable import Variable
from xarray.backends.file_manager import CachingFileManager

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
        else:
            raise NotImplementedError("fancy indexing not implemented yet")


def parse_var(name):
    if name.endswith(DATA_SUFFIX):
        name = name[: -len(DATA_SUFFIX)]
    elif name.endswith(INDEX_SUFFIX):
        name = name[: -len(INDEX_SUFFIX)]
    return name


class TileDBDataStore(AbstractDataStore):

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

        w_dims = ("x", "w", "width", "lon", "lng", "long", "longitude")
        h_dims = ("y", "h", "height", "lat", "latitude")
        band_dims = ("band", "bands", "count")

        coord_data = None
        if coord_name.lower() in (*w_dims, *h_dims, *band_dims):
            try:
                transform = self.ds.meta.get('transform')
            except:
                transform = None
            if transform is not None:
                # assumes transform is stored as List[float,...]
                transform = [float(num) for num in transform.split(" ")]
                transform = Affine(*transform)

                if coord_name.lower() in w_dims:
                    coord_data, _ = transform * \
                                    np.meshgrid(np.arange(coord_size) + 0.5, np.zeros(coord_size) + 0.5)
                    coord_data = coord_data[0, :]

                elif coord_name.lower() in h_dims:
                    _, coord_data = transform * \
                                    np.meshgrid(np.zeros(coord_size) + 0.5, np.arange(coord_size) + 0.5)
                    coord_data = coord_data[:, 0]

                elif coord_name.lower() in band_dims:
                    coord_data = np.arange(1, coord_size + 1, dtype=np.int64)

        if coord_data is None:
            min_value = dim.domain[0]
            max_value = dim.domain[1]
            dtype = dim.dtype
            if metadata and "time" in metadata:
                start_date = np.datetime64(metadata["time"]["reference"])
                freq = metadata["time"]["freq"]
                coord_data = pd.date_range(start_date, periods=dim.size, freq=freq)
                dtype = f"datetime64[{freq}]"
            else:
                # parsing NetCDF dim type that is set up domain(1, len(dim))
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
        for key, value in meta.items():
            # need to add re-format value if it was reformatted
            # from tuple, list, or np.dtype in setting into tdb meta
            if key.startswith((_ATTR_PREFIX, _DIM_PREFIX)):
                last_dot_ix = key.rindex(".")
                attr_name = key[key.index(".") + 1: last_dot_ix]
                attr_key = key[last_dot_ix + 1:]
                if not attr_name:
                    raise RuntimeError(
                        f"cant parse attribute metadata '{key}' with "
                        "missing name or key value"
                    )
                # splitting attr_name for x.time metadata for datetime coord
                key_split = attr_name.split(".")
                if len(key_split) > 1:
                    if key_split[0] not in variable_metadata:
                        variable_metadata[key_split[0]] = dict()
                    if key_split[1] not in variable_metadata[key_split[0]]:
                        variable_metadata[key_split[0]][key_split[1]] = dict()
                    variable_metadata[key_split[0]][key_split[1]][attr_key] = value
                else:
                    variable_metadata[attr_name][attr_key] = value
            else:
                variable_metadata[key] = value
        return variable_metadata


class TileDBBackendEntrypoint(BackendEntrypoint):
    available = True

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


BACKEND_ENTRYPOINTS["tiledb"] = TileDBBackendEntrypoint


def td_to_str(td):
    return str(td.astype("timedelta64[ns]").astype(float))


TIMEDELTAS = {
    td_to_str(np.timedelta64(1, "D")): "D",
    td_to_str(np.timedelta64(1, "W")): "W",
    td_to_str(np.timedelta64(1, "M")): "M",
    td_to_str(np.timedelta64(1, "Y")): "Y",
    td_to_str(np.timedelta64(1, "h")): "h",
    td_to_str(np.timedelta64(1, "m")): "m",
    td_to_str(np.timedelta64(1, "s")): "s",
    td_to_str(np.timedelta64(1, "ms")): "ms",
    td_to_str(np.timedelta64(1, "us")): "us",
    td_to_str(np.timedelta64(1, "ns")): "ns",
    td_to_str(np.timedelta64(1, "ps")): "ps",
    td_to_str(np.timedelta64(1, "fs")): "fs",
    td_to_str(np.timedelta64(1, "as")): "as",
}


class XarrayToTileDBWriter:
    def __init__(self, dataset: xr.Dataset, path: str):
        self.default_dtype = np.int32
        self.dataset = dataset
        self.path = path

    def write_tdb_array(
        self, data: Dict[Optional[Hashable], Any], metadata: Dict[Hashable, Any] = None
    ):
        try:
            with tiledb.open(self.path, "w") as array:
                array[:] = data
                if metadata is not None:
                    for key, value in metadata.items():
                        array.meta[key] = value
        except:
            return False

    def to_tiledb(self):  # noqa: C901
        #  set default int/uint dtype if in dims,
        #  overwrites itself with last 'iu' dim dtype
        dataset = self.dataset
        path = self.path

        dtypes = []
        for dim in dataset.dims.keys():
            dt = dataset[dim].dtype
            dtypes.append(dt)
        for dt in dtypes:
            if dt.kind in "iu":
                self.default_dtype = dt

        coords = dataset.coords
        w_dims = ("x", "w", "width", "lon", "lng", "long", "longitude")
        h_dims = ("y", "h", "height", "lat", "latitude")
        band_dims = ("band", "bands", "count")

        tdb_dims = []
        for name in coords:
            if name in dataset.dims:
                coord = coords[name]

                min_value = coord.data[0]
                max_value = coord.data[-1]
                dtype = coord.dtype

                is_spatial_ds = False
                if dataset.attrs and "transform" in dataset.attrs:
                    is_spatial_ds = True
                if is_spatial_ds and str(name).lower() in (*w_dims, *h_dims, *band_dims):
                    min_value = 1
                    max_value = len(coord.data)
                    dtype = self.default_dtype

                if dtype.kind not in "iuM":
                    raise NotImplementedError(
                        f"TDB Arrays don't work yet with this dtype coord {dtype}"
                    )

                if dtype.kind == "M":
                    second_value = coord.data[1]
                    step = second_value - min_value
                    freq_key = str(step.astype("timedelta64[ns]").astype(float))
                    assert freq_key in TIMEDELTAS
                    freq = TIMEDELTAS[freq_key]
                    date_range = pd.date_range(min_value, max_value, freq=freq)
                    assert (coord.data == date_range).all()

                    if not dataset.attrs:
                        dataset.attrs = dict()
                    if name not in dataset.attrs:
                        dataset.attrs[name] = dict()
                    dataset.attrs[name]["time"] = dict(
                        reference=str(min_value), freq=freq
                    )

                    min_value = 1
                    max_value = len(coord.data)
                    dtype = self.default_dtype

                # test for NetCDF dimension type coord starting at 0
                if min_value == 0:
                    min_value, max_value = min_value + 1, max_value + 1

                domain = (min_value, max_value)
                tdb_dim = tiledb.Dim(name=name, domain=domain, dtype=dtype)
                tdb_dims.append(tdb_dim)

        dom = tiledb.Domain(*tdb_dims)

        data_vars = []
        data = dict()
        for var in dataset.data_vars:
            var = dataset[var]
            data_var = tiledb.Attr(name=var.name, dtype=var.dtype)
            data_vars.append(data_var)
            data[var.name] = var.data

        schema = tiledb.ArraySchema(domain=dom, attrs=data_vars, sparse=False)

        if tiledb.array_exists(path):
            tiledb.remove(path)
        tiledb.DenseArray.create(path, schema)

        metadata = dict()

        data_var_attrs = dict()
        dim_attrs = dict()
        for key, value in dataset.attrs.items():
            if type(value) in (tuple, list):
                value = " ".join([str(val) for val in value])
            elif "dtype" in value.__dir__():
                if value.dtype.kind in "iu":
                    value = int(value)
                elif value.dtype.kind == "f":
                    value = float(value)
                elif value.dtype.kind == "M":
                    value = str(value)
            if key not in dataset.data_vars and key not in dataset.dims:
                metadata[key] = value
            elif key in dataset.data_vars:
                data_var_attrs[key] = value
            elif key in dataset.dims:
                dim_attrs[key] = value

        for var_name, attrs in data_var_attrs.items():
            key_prefix = f"{_ATTR_PREFIX}{var_name}"
            if isinstance(attrs, dict):
                for attr_name, value in attrs.items():
                    key = f"{key_prefix}.{attr_name}"
                    if isinstance(value, np.datetime64):
                        value = str(value)
                    metadata[key] = value
            else:
                metadata[key_prefix] = attrs

        for dim_name, attrs in dim_attrs.items():
            key_prefix = f"{_DIM_PREFIX}{dim_name}"
            if isinstance(attrs, dict):
                for attr_name, value in attrs.items():
                    key = f"{key_prefix}.{attr_name}"
                    if isinstance(value, dict):
                        for sub_attr, sub_value in value.items():
                            sub_key = f"{key}.{sub_attr}"
                            metadata[sub_key] = sub_value
                    else:
                        if isinstance(value, np.datetime64):
                            value = str(value)
                        metadata[key] = value
            else:
                metadata[key_prefix] = attrs

        return self.write_tdb_array(data, metadata)
