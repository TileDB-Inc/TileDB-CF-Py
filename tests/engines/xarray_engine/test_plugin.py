# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.

import numpy as np
import pytest
import xarray as xr
from pytest import importorskip
from xarray import Variable, open_dataset
from xarray.testing import assert_equal
from xarray.backends.common import BACKEND_ENTRYPOINTS

tiledb = importorskip("tiledb")

from tiledb_cf.engines.xarray_engine import TileDBBackendEntrypoint
BACKEND_ENTRYPOINTS["tiledb"] = TileDBBackendEntrypoint

INT_1D_DATA = np.arange(0, 16, dtype=np.int32)
INT_1D_ATTRS = {"description": "int 1D data var"}
INT_1D_VAR = Variable(["rows"], INT_1D_DATA, INT_1D_ATTRS)
INT_1D_COORDS = {"rows": np.arange(0, 16, dtype=np.int32)}

INT_2D_DATA = np.arange(0, 32, dtype=np.int32).reshape(8, 4)
INT_2D_ATTRS = {"description": "int 2D data var"}
INT_2D_VAR = Variable(["rows", "cols"], INT_2D_DATA, INT_2D_ATTRS)
INT_2D_COORDS = {
    "rows": np.arange(0, 8, dtype=np.int32),
    "cols": np.arange(0, 4, dtype=np.int32),
}

INT_1D_SHIFTED_ATTRS = {"description": "int 1d shifted data var"}
INT_1D_SHIFTED_VAR = Variable(["rows"], INT_1D_DATA, INT_1D_SHIFTED_ATTRS)
INT_1D_SHIFTED_COORDS = {"rows": np.arange(5, 21, dtype=np.int32)}

INT_2D_SHIFTED_ATTRS = {"description": "int 2d shifted data var"}
INT_2D_SHIFTED_VAR = Variable(["rows", "cols"], INT_2D_DATA, INT_2D_SHIFTED_ATTRS)
INT_2D_SHIFTED_COORDS = {
    "rows": np.arange(4, 12, dtype=np.int32),
    "cols": np.arange(1, 5, dtype=np.int32),
}

DATETIME_1D_VAR_ATTRS = {"description": "datetime 1d data var"}
DATETIME_1D_VAR = Variable(["rows"], INT_1D_DATA, DATETIME_1D_VAR_ATTRS)
DATETIME_1D_COORDS = {
    "rows": np.arange(
        np.datetime64("2000-01-01"), np.datetime64("2000-01-17"), np.timedelta64(1, "D")
    )
}

DS_ATTRS = {"description": "test dataset"}
DATETIME_1D_DS_ATTRS = {
    **DS_ATTRS,
    "rows": {"time reference": np.datetime64("2000-01-01"), "freq": "D"},
}

SAMPLE_DATASETS = {
    "simple_1d": xr.Dataset(
        data_vars={"data": INT_1D_VAR}, coords=INT_1D_COORDS, attrs=DS_ATTRS
    ),
    "simple_2d": xr.Dataset(
        data_vars={"data": INT_2D_VAR}, coords=INT_2D_COORDS, attrs=DS_ATTRS
    ),
    "1d_shifted": xr.Dataset(
        data_vars={"data": INT_1D_SHIFTED_VAR},
        coords=INT_1D_SHIFTED_COORDS,
        attrs=DS_ATTRS,
    ),
    "2d_shifted": xr.Dataset(
        data_vars={"data": INT_2D_SHIFTED_VAR},
        coords=INT_2D_SHIFTED_COORDS,
        attrs=DS_ATTRS,
    ),
    "1d_datetime": xr.Dataset(
        data_vars={"data": DATETIME_1D_VAR},
        coords=DATETIME_1D_COORDS,
        attrs=DATETIME_1D_DS_ATTRS,
    ),
}


class TestTDBBackend:
    engine = "tiledb"

    def open(self, path):
        with open_dataset(path, engine=self.engine) as ds:
            ds_obj = ds
        return ds_obj

    def write_tdb_array(self, path, data, metadata):
        with tiledb.open(path, "w") as array:
            array[:] = data
            for key, value in metadata.items():
                array.meta[key] = value

    def to_tiledb(self, dataset, path):
        coords = dataset.coords

        tdb_dims = []
        for name in coords:
            if name in dataset.dims:
                coord = coords[name]
                dtype = coord.dtype
                if dtype.kind not in "iuM":
                    raise NotImplementedError(
                        f"TDB Arrays don't work yet with this dtype coord {dtype}"
                    )
                min_value = coord.data[0]
                max_value = coord.data[-1]
                if dtype.kind == "M":
                    domain = (1, len(coord.data))
                    dtype = np.int32
                else:
                    # test for NetCDF dimension type coord starting at 0
                    if min_value == 0:
                        min_value, max_value = min_value + 1, max_value + 1
                    domain = (min_value, max_value)
                tdb_dim = tiledb.Dim(name=name, domain=domain, dtype=dtype)
                tdb_dims.append(tdb_dim)

        dom = tiledb.Domain(*tdb_dims)

        data_vars = []
        data = dict()
        vars_attrs = dict()
        for var in dataset.data_vars:
            var = dataset[var]
            data_var = tiledb.Attr(name=var.name, dtype=var.dtype)
            data_vars.append(data_var)
            data[var.name] = var.data
            vars_attrs[var.name] = var.attrs

        schema = tiledb.ArraySchema(domain=dom, attrs=data_vars, sparse=False)

        data = list(data.values())[0]

        if tiledb.array_exists(path):
            tiledb.remove(path)
        tiledb.DenseArray.create(path, schema)

        metadata = dict()

        for key, value in dataset.attrs.items():
            if key not in dataset.dims and key not in dataset.data_vars:
                metadata[key] = value
            elif key in dataset.dims:
                if isinstance(value, dict):
                    for k, v in value.items():
                        array_key = f"__tiledb_dim.{key}.{k}"
                        if isinstance(v, np.datetime64):
                            v = str(v)
                        metadata[array_key] = v
                else:
                    array_key = f"__tiledv_dim.{key}"
                    metadata[array_key] = value
            elif key in dataset.data_vars:
                if isinstance(value, dict):
                    for k, v in value.items():
                        array_key = f"__tiledb_attr.{key}.{k}"
                        metadata[array_key] = v
                else:
                    array_key = f"__tiledv_attr.{key}"
                    metadata[array_key] = value

        self.write_tdb_array(path, data, metadata)

        with tiledb.DenseArray(path) as array:
            schema = array.schema
            meta = array.meta
        return schema, meta

    def roundtrip(self, dataset, path):
        self.to_tiledb(dataset, path)
        return self.open(path)

    @pytest.mark.parametrize("ds_name, expected", SAMPLE_DATASETS.items())
    def test_totiledb(self, ds_name, expected, tmpdir):
        path = f"{tmpdir}/{ds_name}"
        print(self.to_tiledb(expected, path))

    @pytest.mark.parametrize("ds_name, expected", SAMPLE_DATASETS.items())
    def test_datasets(self, ds_name, expected, tmpdir):
        path = f"{tmpdir}/{ds_name}"
        result = self.roundtrip(expected, path)
        assert_equal(result, expected)

    @pytest.mark.parametrize("ds_name, expected", SAMPLE_DATASETS.items())
    def test_tdb_xarray_indexing_match(self, ds_name, expected, tmpdir):
        path = f"{tmpdir}/{ds_name}"
        self.to_tiledb(expected, path)
        with tiledb.DenseArray(path) as array:
            dims = array.domain
            min_vals = [dim.domain[0] for dim in dims]
            for ii in np.ndindex(array.shape):
                jj = list(ii)
                for idx in range(len(jj)):
                    jj[idx] += min_vals[idx]
                jj = tuple(jj)
                assert array[jj]["data"] == expected.data[ii].data
