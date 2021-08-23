# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.

import numpy as np

import xarray as xr
from xarray import Variable, open_dataset
from xarray.testing import assert_equal

import pytest
from pytest import importorskip

tiledb = importorskip("tiledb")

INT_1D_DATA = np.arange(0, 16, dtype=np.int32)
INT_1D_ATTRS = {"description": "int 1D data var"}
INT_1D_VAR = Variable(["rows"], INT_1D_DATA, INT_1D_ATTRS)
INT_1D_COORDS = {"rows": np.arange(0, 16, dtype=np.int32)}

INT_2D_DATA = np.arange(0, 32, dtype=np.int32).reshape(8, 4)
INT_2D_ATTRS = {"description": "int 2D data var"}
INT_2D_VAR = Variable(["rows", "cols"], INT_2D_DATA, INT_2D_ATTRS)
INT_2D_COORDS = {"rows": np.arange(0, 8, dtype=np.int32), "cols": np.arange(0, 4, dtype=np.int32)}

INT_1D_SHIFTED_ATTRS = {"description": "int 1d shifted data var"}
INT_1D_SHIFTED_VAR = Variable(["rows"], INT_1D_DATA, INT_1D_SHIFTED_ATTRS)
INT_1D_SHIFTED_COORDS = {"rows": np.arange(5, 21, dtype=np.int32)}

INT_2D_SHIFTED_ATTRS = {"description": "int 2d shifted data var"}
INT_2D_SHIFTED_VAR = Variable(["rows", "cols"], INT_2D_DATA, INT_2D_SHIFTED_ATTRS)
INT_2D_SHIFTED_COORDS = {"rows": np.arange(4, 12, dtype=np.int32), "cols": np.arange(1, 5, dtype=np.int32)}

DATETIME_1D_VAR_ATTRS = {"description": "datetime 1d data var"}
DATETIME_1D_VAR = Variable(["rows"], INT_1D_DATA, DATETIME_1D_VAR_ATTRS)
DATETIME_1D_COORDS = {"rows": np.arange(np.datetime64("2000-01-01"), np.datetime64("2000-01-17"), np.timedelta64(1, 'D'))}

DS_ATTRS = {"description": "test dataset"}
DATETIME_1D_DS_ATTRS = {**DS_ATTRS, "rows": {"time reference": np.datetime64("2000-01-01"), "freq": "D"}}

SAMPLE_DATASETS = {
    "simple_1d": xr.Dataset(data_vars={"data": INT_1D_VAR}, coords=INT_1D_COORDS, attrs=DS_ATTRS),
    "simple_2d": xr.Dataset(data_vars={"data": INT_2D_VAR}, coords=INT_2D_COORDS, attrs=DS_ATTRS),
    "1d_shifted": xr.Dataset(data_vars={"data": INT_1D_SHIFTED_VAR}, coords=INT_1D_SHIFTED_COORDS, attrs=DS_ATTRS),
    "2d_shifted": xr.Dataset(data_vars={"data": INT_2D_SHIFTED_VAR}, coords=INT_2D_SHIFTED_COORDS, attrs=DS_ATTRS),
    "1d_datetime": xr.Dataset(data_vars={"data": DATETIME_1D_VAR}, coords=DATETIME_1D_COORDS, attrs=DATETIME_1D_DS_ATTRS)
}

class TestTDBBackend:
    engine = "tiledb"

    def open(self, path):
        with open_dataset(path, engine=self.engine) as ds:
            ds_obj = ds
        return ds_obj

    def to_tiledb(self, dataset, path):
        tdb_dims = []
        coords = dataset.coords
        coords_attrs = dict()
        for name in coords:
            if name in dataset.dims:
                if dataset.attrs.get(name) is not None:
                    coords_attrs[name] = dataset.attrs[name]
                coord = coords[name]
                dtype = coord.dtype
                if dtype.kind not in 'iuM':
                    raise NotImplementedError(f"TDB Arrays don't work yet with this dtype coord {dtype}")
                min_value = coord.data[0]
                max_value = coord.data[-1]
                if dtype.kind == "M":
                    domain = (1, len(coord.data))
                    dtype = np.int32
                else:
                    domain = (min_value + 1, max_value + 1)
                tdb_dim = tiledb.Dim(name=name, domain=domain, dtype=dtype)
                tdb_dims.append(tdb_dim)
        tdb_attrs = []
        data = dict()
        vars_attrs = dict()
        for var in dataset.data_vars:
            var = dataset[var]
            tdb_attr = tiledb.Attr(name=var.name, dtype=var.dtype)
            tdb_attrs.append(tdb_attr)
            data[var.name] = var.data
            vars_attrs[var.name] = var.attrs

        dom = tiledb.Domain(*tdb_dims)
        schema = tiledb.ArraySchema(domain=dom, attrs=tdb_attrs, sparse=False)
        if tiledb.array_exists(path):
            tiledb.remove(path)
        tiledb.DenseArray.create(path, schema)
        with tiledb.DenseArray(path, "w") as array:
            # only one data attr works for now
            array[:] = list(data.values())[0]
            for key, value in dataset.attrs.items():
                if key not in dataset.dims and key not in dataset.data_vars:
                    array.meta[key] = value
            for var_name, var_attrs in vars_attrs.items():
                for key, value in var_attrs.items():
                    array_key = f"__tiledb_attr.{var_name}.{key}"
                    array.meta[array_key] = value
            for dim_name, dim_attrs in coords_attrs.items():
                for key, value in dim_attrs.items():
                    array_key = f"__tiledb_dim.{dim_name}.{key}"
                    if isinstance(value, tuple) or isinstance(value, list):
                        value = "".join(value)
                    if isinstance(value, np.datetime64):
                        value = str(value)
                    array.meta[array_key] = value
        with tiledb.DenseArray(path) as array:
            print(array.schema, dict(array.meta))

    def roundtrip(self, dataset, path):
        self.to_tiledb(dataset, path)
        return self.open(path)

    @pytest.mark.parametrize(
        "ds_name, expected",
        SAMPLE_DATASETS.items()
    )
    def test_datasets(self, ds_name, expected, tmpdir):
        path = f"{tmpdir}/{ds_name}"
        result = self.roundtrip(expected, path)
        assert_equal(result, expected)

    @pytest.mark.parametrize(
        "ds_name, expected",
        SAMPLE_DATASETS.items()
    )
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
                assert (array[jj]["data"] == expected.data[ii].data)

