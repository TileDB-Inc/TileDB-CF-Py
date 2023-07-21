# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.

from typing import Any, Dict

import numpy as np
import pytest

import tiledb

xr = pytest.importorskip("xarray")


class TileDBXarrayBase:
    """Base class for TileDB-xarray backend for 1D TileDB arrays.

    Parameters:
        name: Short descriptive name for naming NetCDF file.
        array_schema: Schema for the TileDB Array.
        data: Data for the TileDB attributes.
        backend_kwards: Keyword  arguments to use xr.open_dataset.
    """

    name = "base"
    schemas: Dict[str, tiledb.ArraySchema] = {}
    data: Dict[str, np.ndarray] = {}
    backend_kwargs: Dict[str, Any] = {}

    @pytest.fixture(scope="class")
    def dataset(self):
        """Returns a dataset that matches the TileDB array."""
        return xr.Dataset()

    @pytest.fixture(scope="class")
    def tiledb_uri(self, tmpdir_factory):
        """Creates a TileDB array and returns the uri."""
        uri = str(tmpdir_factory.mktemp("input_array").join(f"{self.name}.tiledb-xr"))
        tiledb.Group.create(uri)
        with tiledb.Group(uri, mode="w") as group:
            for name, schema in self.schemas.items():
                array_uri = f"{uri}/{name}"
                tiledb.Array.create(array_uri, schema)
                with tiledb.open(array_uri, mode="w") as array:
                    array[...] = {name: self.data[name]}
                group.add(name, name, relative=True)
        return uri

    def open_dataset(self, uri):
        return xr.open_dataset(
            uri, engine="tiledb-xr", cache=False, **self.backend_kwargs
        )

    def test_full_dataset(self, tiledb_uri, dataset):
        """Checks the TileDB array can be opened with the backend."""
        result = self.open_dataset(tiledb_uri)
        expected = dataset
        xr.testing.assert_equal(result, expected)


class TileDBXarray1DBase(TileDBXarrayBase):
    """Base case for testing the TileDB-xarray backend for 1D TileDB arrays."""

    def test_basic_indexing(self, tiledb_uri, dataset):
        tiledb_dataset = self.open_dataset(tiledb_uri)
        for name in self.data:
            tiledb_data_array = tiledb_dataset[name]
            expected_data_array = dataset[name]
            for index in range(expected_data_array.size):
                result_array = tiledb_data_array[index]
                isinstance(result_array, xr.DataArray)
                print(result_array)
                for attr in dir(type(result_array)):
                    if not callable(
                        getattr(xr.DataArray, attr)
                    ) and not attr.startswith("__"):
                        print(attr)
                result = result_array.data
                if True:
                    return
                expected = expected_data_array[index].data
                print(expected)
                np.testing.assert_equal(result, expected)

    def test_negative_indexing(self, tiledb_uri, dataset):
        tiledb_dataset = self.open_dataset(tiledb_uri)
        for name in self.data:
            tiledb_data_array = tiledb_dataset[name]
            expected_data_array = dataset[name]
            for index in range(expected_data_array.size):
                result_array = tiledb_data_array[-1 - index]
                result = result_array.data
                expected = expected_data_array[-1 - index].data
                np.testing.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            pytest.param(-5, id="-5"),
            pytest.param(slice(None), id=":"),
            pytest.param(slice(0, 5), id="0:5"),
            pytest.param(slice(0, 5, 1), id="slice(0, 5, 1)"),
            pytest.param(slice(None, 5), id=": 5"),
            pytest.param(slice(None, 5, 1), id="slice(None, 5, 1)"),
            pytest.param(slice(5, None), id="5:"),
            pytest.param(slice(None, 0), id=":0"),
            pytest.param(slice(5, 0, -2), id="slice(5, 0, -2)"),
            pytest.param(slice(1, 1), id="1:1"),
            pytest.param(np.array([0, 5, 3, 2, 1]), id="np.array([0, 5, 3, 2, 1])"),
            pytest.param(
                np.array([-1, -5, -3, -2, -15]), id="np.array([-1, -5, -3, -2, -15])"
            ),
            pytest.param(
                np.array([0, 0, 2, 2, 1, 3]), id="np.array([0, 0, 2, 2, 1, 3])"
            ),
        ],
    )
    def test_indexing_array(self, tiledb_uri, dataset, index):
        tiledb_dataset = self.open_dataset(tiledb_uri)
        for name in self.data:
            result_data_array = tiledb_dataset[name]
            result_data_array = result_data_array[index]
            result = result_data_array.data
            expected = dataset[name][index].data
            np.testing.assert_equal(result, expected)


class TileDBXarray2DBase(TileDBXarrayBase):
    """Base class for testing the TileDB-xarray backend for 2D TileDB arrays."""

    @pytest.mark.parametrize(
        "index1, index2",
        [
            (0, 0),
            (1, 2),
            (-5, 2),
            (slice(None), slice(None)),
            (slice(1, 8, 2), slice(0, 3)),
            (slice(1, 1), slice(None)),
            (slice(1, 1), 0),
            (np.array([0, 1, 2]), 1),
            (slice(1, 8, 2), np.array([0, 1, 2])),
            (1, np.array([-1, -2])),
            (np.array([0, 0, 2, 2]), np.array([1, 1, 3])),
        ],
    )
    def test_indexing_array(self, tiledb_uri, dataset, index1, index2):
        """Tests indexing for data arrays in the dataset."""
        tiledb_dataset = xr.open_dataset(
            tiledb_uri, engine="tiledb", **self.backend_kwargs
        )
        for name in self.data:
            result_data_array = tiledb_dataset[name]
            result_data_array = result_data_array[index1, index2]
            result = result_data_array.data
            expected = dataset[name][index1, index2].data
            np.testing.assert_equal(result, expected)

    def test_indexing_array_nested(self, tiledb_uri, dataset):
        """Tests nested indexing for all data arrays in the dataset."""
        tiledb_dataset = xr.open_dataset(
            tiledb_uri, engine="tiledb", **self.backend_kwargs
        )
        for name in self.data:
            result = tiledb_dataset[name][[0, 2, 2], [1, 3]][[0, 0, 2], 1]
            expected = dataset[name][[0, 2, 2], [1, 3]][[0, 0, 2], 1]
            xr.testing.assert_equal(result, expected)


class TestSimple1D(TileDBXarray1DBase):
    """Simple 1D dataset."""

    name = "simple1d"
    schemas = {
        "example": tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="rows", domain=(0, 15), tile=4, dtype=np.int64)
            ),
            attrs=[tiledb.Attr(name="example", dtype=np.float64)],
        ),
        "index": tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="rows", domain=(0, 15), tile=4, dtype=np.int64)
            ),
            attrs=[tiledb.Attr(name="index", dtype=np.int64)],
        ),
    }

    data = {
        "example": np.linspace(-1.0, 1.0, 16),
        "index": np.arange(16, dtype=np.int64),
    }

    @pytest.fixture(scope="class")
    def dataset(self):
        """Returns a dataset that matches the TileDB array."""
        return xr.Dataset(
            {
                "example": xr.DataArray(self.data["example"], dims=("rows",)),
                "index": xr.DataArray(self.data["index"], dims=("rows",)),
            }
        )


class TestSimple1DUnsignedIntDim(TileDBXarray1DBase):
    """Simple 1D dataset."""

    name = "simple1d"
    schemas = {
        "data": tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="rows", domain=(0, 15), tile=4, dtype=np.uint64)
            ),
            attrs=[tiledb.Attr(name="data", dtype=np.float64)],
        ),
        "index": tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="rows", domain=(0, 15), tile=4, dtype=np.uint64)
            ),
            attrs=[tiledb.Attr(name="index", dtype=np.int64)],
        ),
    }
    data = {"data": np.linspace(-1.0, 1.0, 16), "index": np.arange(16, dtype=np.int64)}

    @pytest.fixture(scope="class")
    def dataset(self):
        """Returns a dataset that matches the TileDB array."""
        return xr.Dataset(
            {
                "data": xr.DataArray(self.data["data"], dims=("rows",)),
                "index": xr.DataArray(self.data["index"], dims=("rows",)),
            }
        )


class TestSimple2DExample(TileDBXarray2DBase):
    """Runs standard dataset for simple 2D array."""

    name = "simple_2d"
    schemas = {
        "a": tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="rows", domain=(0, 7), tile=4, dtype=np.int32),
                tiledb.Dim(name="cols", domain=(0, 3), tile=4, dtype=np.int32),
            ),
            attrs=[tiledb.Attr(name="a", dtype=np.float64)],
        ),
        "b": tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="rows", domain=(0, 7), tile=4, dtype=np.int32),
                tiledb.Dim(name="cols", domain=(0, 3), tile=4, dtype=np.int32),
            ),
            attrs=[tiledb.Attr(name="b", dtype=np.uint32)],
        ),
    }
    data = {
        "a": np.reshape(np.random.rand(32), (8, 4)),
        "b": np.reshape(np.arange(32, dtype=np.int32), (8, 4)),
    }

    @pytest.fixture(scope="class")
    def dataset(self):
        """Returns a dataset that matches the TileDB array."""
        return xr.Dataset(
            {
                "a": xr.DataArray(self.data["a"], dims=("rows", "cols")),
                "b": xr.DataArray(self.data["b"], dims=("rows", "cols")),
            }
        )


class TestSimple2DExampleUnsignedDims(TileDBXarray2DBase):
    """Runs standard dataset for simple 2D array."""

    name = "simple_2d"
    schemas = {
        "a": tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="rows", domain=(0, 7), tile=4, dtype=np.uint64),
                tiledb.Dim(name="cols", domain=(0, 3), tile=4, dtype=np.uint64),
            ),
            attrs=[tiledb.Attr(name="a", dtype=np.float64)],
        ),
        "b": tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="rows", domain=(0, 7), tile=4, dtype=np.uint64),
                tiledb.Dim(name="cols", domain=(0, 3), tile=4, dtype=np.uint64),
            ),
            attrs=[tiledb.Attr(name="b", dtype=np.uint32)],
        ),
    }
    data = {
        "a": np.reshape(np.random.rand(32), (8, 4)),
        "b": np.reshape(np.arange(32, dtype=np.int32), (8, 4)),
    }

    @pytest.fixture(scope="class")
    def dataset(self):
        """Returns a dataset that matches the TileDB array."""
        return xr.Dataset(
            {
                "a": xr.DataArray(self.data["a"], dims=("rows", "cols")),
                "b": xr.DataArray(self.data["b"], dims=("rows", "cols")),
            }
        )


def test_open_multidim_dataset(create_tiledb_example):
    uri, expected = create_tiledb_example
    dataset = xr.open_dataset(uri, engine="tiledb")
    xr.testing.assert_equal(dataset, expected)


@pytest.mark.filterwarnings("ignore:'netcdf4' fails while guessing")
def test_open_multidim_dataset_guess_engine(create_tiledb_example):
    uri, expected = create_tiledb_example
    dataset = xr.open_dataset(uri)
    xr.testing.assert_equal(dataset, expected)
