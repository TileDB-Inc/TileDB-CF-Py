# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import contextlib

import numpy as np
import pytest
import xarray as xr
from pytest import importorskip
from xarray import Variable, open_dataset
from xarray.backends.common import BACKEND_ENTRYPOINTS
from xarray.testing import assert_equal

tiledb = importorskip("tiledb")

from tiledb.cf.engines.xarray_engine import TileDBBackendEntrypoint

BACKEND_ENTRYPOINTS["tiledb"] = TileDBBackendEntrypoint

_ATTR_PREFIX = "__tiledb_attr."
_DIM_PREFIX = "__tiledb_dim."

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
    "cols": np.arange(2, 6, dtype=np.int32),
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

    @contextlib.contextmanager
    def open(self, path):
        with open_dataset(path, engine=self.engine) as ds:
            yield ds

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

        data_var_attrs = dict()
        dim_attrs = dict()
        for key, value in dataset.attrs.items():
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
                    if isinstance(value, np.datetime64):
                        value = str(value)
                    metadata[key] = value
            else:
                metadata[key_prefix] = attrs

        self.write_tdb_array(path, data, metadata)

    @contextlib.contextmanager
    def roundtrip(self, dataset, path):
        self.to_tiledb(dataset, path)
        with self.open(path) as ds:
            yield ds

    @pytest.mark.parametrize("ds_name, expected", SAMPLE_DATASETS.items())
    def test_totiledb(self, ds_name, expected, tmpdir):
        path = f"{tmpdir}/{ds_name}"
        self.to_tiledb(expected, path)
        with tiledb.DenseArray(path) as array:
            for data_attr in array.schema:
                assert data_attr.name in expected.data_vars
            for dim in array.domain:
                assert dim.name in expected.dims
                assert dim.size == expected[dim.name].size
                dt_kind = expected[dim.name].dtype.kind
                if dt_kind == "M":
                    expect_min = 1
                    expect_max = expected[dim.name].size
                else:
                    expect_min = expected[dim.name][0]
                    expect_min = expect_min if expect_min > 0 else 1
                    expect_max = expect_min + expected[dim.name].size - 1
                assert dim.domain[0] == expect_min
                assert dim.domain[1] == expect_max

    @pytest.mark.parametrize("ds_name, expected", SAMPLE_DATASETS.items())
    def test_datasets(self, ds_name, expected, tmpdir):
        path = f"{tmpdir}/{ds_name}"
        with self.roundtrip(expected, path) as result:
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
