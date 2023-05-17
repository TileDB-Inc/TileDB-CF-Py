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
    schema = None
    data: Dict[str, np.ndarray] = {}
    backend_kwargs: Dict[str, Any] = {}

    @pytest.fixture(scope="class")
    def dataset(self):
        """Returns a dataset that matches the TileDB array."""
        return xr.Dataset()

    @pytest.fixture(scope="class")
    def tiledb_uri(self, tmpdir_factory):
        """Creates a TileDB array and returns the uri."""
        uri = str(tmpdir_factory.mktemp("input_array").join(f"{self.name}.nc"))
        assert self.schema is not None, "Array schema not set for test."
        tiledb.Array.create(uri, self.schema)
        with tiledb.open(uri, mode="w") as array:
            array[...] = self.data
        return uri

    def open_dataset(self, uri):
        return xr.open_dataset(uri, engine="tiledb", cache=False, **self.backend_kwargs)

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
                result = result_array.data
                expected = expected_data_array[index].data
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
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="rows", domain=(0, 15), tile=4, dtype=np.int64),
        ),
        attrs=[
            tiledb.Attr(name="data", dtype=np.float64),
            tiledb.Attr(name="index", dtype=np.int64),
        ],
    )
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


class TestSimple1DUnsignedIntDim(TileDBXarray1DBase):
    """Simple 1D dataset."""

    name = "simple1d"
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="rows", domain=(0, 15), tile=4, dtype=np.uint64),
        ),
        attrs=[
            tiledb.Attr(name="data", dtype=np.float64),
            tiledb.Attr(name="index", dtype=np.int64),
        ],
    )
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


class TestShiftedDim1D(TileDBXarray1DBase):
    """Simple 1D dataset with lower bound not at 0."""

    name = "simple1d"
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="rows", domain=(-5, 10), tile=4, dtype=np.int32),
        ),
        attrs=[tiledb.Attr(name="data", dtype=np.int64)],
    )
    data = {"data": np.arange(16, dtype=np.int64)}

    @pytest.fixture(scope="class")
    def dataset(self):
        """Returns a dataset that matches the TileDB array."""
        return xr.Dataset(
            {
                "data": xr.DataArray(
                    self.data["data"],
                    dims=("rows",),
                ),
            }
        )


@pytest.mark.skip(reason="skip this test until xarray datetime handling stabalizes")
class TestDatetimeDim1D(TileDBXarray1DBase):
    """Simple 1D dataset with datetime  dimension."""

    name = "simple1d"
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(
                name="rows",
                domain=(
                    np.datetime64("2000-01-01", "D"),
                    np.datetime64("2000-01-16", "D"),
                ),
                tile=np.timedelta64(4, "D"),
                dtype=np.datetime64("", "D"),
            ),
        ),
        attrs=[tiledb.Attr(name="data", dtype=np.int64)],
    )
    data = {"data": np.arange(16, dtype=np.int64)}

    @pytest.fixture(scope="class")
    def dataset(self):
        """Returns a dataset that matches the TileDB array."""
        return xr.Dataset(
            {
                "data": xr.DataArray(
                    self.data["data"],
                    dims=("rows",),
                    coords={
                        "rows": np.arange(
                            np.datetime64("2000-01-01", "D"),
                            np.datetime64("2000-01-17", "D"),
                        )
                    },
                )
            }
        )


class TestSimple2DExample(TileDBXarray2DBase):
    """Runs standard dataset for simple 2D array."""

    name = "simple_2d"
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="rows", domain=(0, 7), tile=4, dtype=np.int32),
            tiledb.Dim(name="cols", domain=(0, 3), tile=4, dtype=np.int32),
        ),
        attrs=[
            tiledb.Attr(name="a", dtype=np.float64),
            tiledb.Attr(name="b", dtype=np.uint32),
        ],
    )
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
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="rows", domain=(0, 7), tile=4, dtype=np.uint64),
            tiledb.Dim(name="cols", domain=(0, 3), tile=4, dtype=np.uint64),
        ),
        attrs=[
            tiledb.Attr(name="a", dtype=np.float64),
            tiledb.Attr(name="b", dtype=np.uint32),
        ],
    )
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


class TestShifted2DExample(TileDBXarray2DBase):
    """Runs standard dataset for simple 2D array."""

    name = "simple_2d"
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="rows", domain=(-3, 4), tile=4, dtype=np.int32),
            tiledb.Dim(name="cols", domain=(2, 5), tile=4, dtype=np.int32),
        ),
        attrs=[tiledb.Attr(name="data", dtype=np.int32)],
    )
    data = {"data": np.reshape(np.arange(32, dtype=np.int32), (8, 4))}

    @pytest.fixture(scope="class")
    def dataset(self):
        """Returns a dataset that matches the TileDB array."""
        return xr.Dataset(
            {
                "data": xr.DataArray(
                    self.data["data"],
                    dims=("rows", "cols"),
                ),
            }
        )


class TestShiftedAddCoords(TileDBXarray2DBase):
    """Runs standard dataset for simple 2D array."""

    name = "simple_2d"
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="rows", domain=(-3, 4), tile=4, dtype=np.int32),
            tiledb.Dim(name="cols", domain=(2, 5), tile=4, dtype=np.int32),
        ),
        attrs=[tiledb.Attr(name="data", dtype=np.int32)],
    )
    data = {"data": np.reshape(np.arange(32, dtype=np.int32), (8, 4))}
    backend_kwargs = {"coord_dims": {"rows", "cols"}}

    @pytest.fixture(scope="class")
    def dataset(self):
        """Returns a dataset that matches the TileDB array."""
        return xr.Dataset(
            {
                "data": xr.DataArray(
                    self.data["data"],
                    dims=("rows", "cols"),
                    coords={"rows": np.arange(-3, 5), "cols": np.arange(2, 6)},
                ),
            }
        )


def test_open_multidim_dataset(create_tiledb_example):
    uri, expected = create_tiledb_example
    dataset = xr.open_dataset(uri, engine="tiledb")
    xr.testing.assert_equal(dataset, expected)


def test_open_multidim_dataset_guess_engine(create_tiledb_example):
    uri, expected = create_tiledb_example
    dataset = xr.open_dataset(uri)
    xr.testing.assert_equal(dataset, expected)
