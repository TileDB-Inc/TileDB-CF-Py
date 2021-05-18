# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.

import numpy as np
import pytest

import tiledb

xr = pytest.importorskip("xarray")


class TestTileDB:

    FLOAT_DATA_2D = np.linspace(
        -1.0, 1.0, num=32, endpoint=True, dtype=np.float64
    ).reshape(8, 4)
    INT_DATA_1D = np.arange(0, 16, dtype=np.int64)
    INT_DATA_2D = np.arange(0, 32, dtype=np.int32).reshape(8, 4)

    SIMPLE_DATA_ARRAYS = {
        "simple_example_1d": xr.DataArray(
            data=INT_DATA_1D,
            dims="rows",
        ),
        "shifted_example_1d": xr.DataArray(
            data=INT_DATA_1D,
            dims="rows",
            coords={"rows": np.arange(-8, 8, dtype=np.int32)},
        ),
        "datetime_example_1d": xr.DataArray(
            data=INT_DATA_1D,
            dims="rows",
            coords={
                "rows": np.arange(
                    np.datetime64("2000-01-01"), np.datetime64("2000-01-17")
                )
            },
        ),
        "simple_example_2d": xr.DataArray(
            data=INT_DATA_2D,
            dims=["rows", "cols"],
        ),
        "shifted_example_2d": xr.DataArray(
            data=FLOAT_DATA_2D,
            dims=["rows", "cols"],
            coords={
                "rows": np.arange(-3, 5, dtype=np.int32),
                "cols": np.arange(2, 6, dtype=np.int32),
            },
        ),
    }

    simple_examples_1d = [
        "simple_example_1d",
        "shifted_example_1d",
        "datetime_example_1d",
    ]

    simple_examples_2d = ["simple_example_2d", "shifted_example_2d"]

    @pytest.fixture(scope="class")
    def test_directory(self, tmpdir_factory):
        return str(tmpdir_factory.mktemp("test_data"))

    @pytest.fixture(scope="class")
    def create_simple_example_data(self, test_directory):
        # zero-based dimension
        simple_array_uri = test_directory + "/simple_example_1d"
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="rows", domain=(0, 15), tile=4, dtype=np.int32),
            ),
            sparse=False,
            attrs=[tiledb.Attr(name="data", dtype=np.int64)],
        )
        tiledb.DenseArray.create(simple_array_uri, schema)
        with tiledb.DenseArray(simple_array_uri, mode="w") as array:
            array[:] = {"data": self.INT_DATA_1D}
        # shifted dimension (with negative values)
        simple_array_uri = test_directory + "/shifted_example_1d"
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="rows", domain=(-8, 7), tile=4, dtype=np.int32),
            ),
            sparse=False,
            attrs=[tiledb.Attr(name="data", dtype=np.int64)],
        )
        tiledb.DenseArray.create(simple_array_uri, schema)
        with tiledb.DenseArray(simple_array_uri, mode="w") as array:
            array[:] = {"data": self.INT_DATA_1D}
        # datetime dimension
        datetime_uri = test_directory + "/datetime_example_1d"
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
        tiledb.DenseArray.create(datetime_uri, schema)
        with tiledb.DenseArray(datetime_uri, mode="w") as array:
            array[:] = {"data": self.INT_DATA_1D}
        # simple example 2d
        array_uri = test_directory + "/simple_example_2d"
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="rows", domain=(0, 7), tile=4, dtype=np.int32),
                tiledb.Dim(name="cols", domain=(0, 3), tile=4, dtype=np.int32),
            ),
            sparse=False,
            attrs=[tiledb.Attr(name="data", dtype=np.int32)],
        )
        tiledb.DenseArray.create(array_uri, schema)
        with tiledb.DenseArray(array_uri, mode="w") as array:
            array[:, :] = {"data": self.INT_DATA_2D}
        # shifted example 2d
        array_uri = test_directory + "/shifted_example_2d"
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="rows", domain=(-3, 4), tile=4, dtype=np.int32),
                tiledb.Dim(name="cols", domain=(2, 5), tile=4, dtype=np.int32),
            ),
            sparse=False,
            attrs=[tiledb.Attr(name="data", dtype=np.float64)],
        )
        tiledb.DenseArray.create(array_uri, schema)
        with tiledb.DenseArray(array_uri, mode="w") as array:
            array[:, :] = {"data": self.FLOAT_DATA_2D}

    @pytest.fixture(scope="function")
    def simple_data_arrays(self, test_directory, create_simple_example_data, request):
        name = request.param
        array_uri = test_directory + "/" + name
        dataset = xr.open_dataset(array_uri, engine="tiledb-cf")
        return dataset["data"], self.SIMPLE_DATA_ARRAYS[name]

    def test_open_multidim_dataset(self, create_tiledb_example):
        uri, expected = create_tiledb_example
        dataset = xr.open_dataset(uri, engine="tiledb-cf")
        xr.testing.assert_allclose(dataset, expected)

    def test_open_multidim_dataset_guess_engine(self, create_tiledb_example):
        uri, expected = create_tiledb_example
        dataset = xr.open_dataset(uri)
        xr.testing.assert_allclose(dataset, expected)

    @pytest.mark.parametrize("simple_data_arrays", simple_examples_1d, indirect=True)
    def test_basic_indexing_1D(self, simple_data_arrays):
        (result, expected) = simple_data_arrays
        for index in range(expected.size):
            xr.testing.assert_allclose(result[index], expected[index])
            xr.testing.assert_allclose(result[-1 - index], expected[-1 - index])

    @pytest.mark.parametrize("simple_data_arrays", simple_examples_1d, indirect=True)
    @pytest.mark.parametrize(
        "index",
        [
            -5,
            slice(None),
            slice(0, 5),
            slice(0, 5, 1),
            slice(None, 5, 1),
            slice(5, None, 1),
            slice(5, 0, -2),
            slice(1, 1),
            np.array([0, 5, 3, 2, 1]),
            np.array([-1, -5, -3, -2, -15]),
            np.array([0, 0, 2, 2, 1, 3]),
        ],
    )
    def test_indexing_array_1D(self, simple_data_arrays, index):
        (tiledb_data_array, xarray_data_array) = simple_data_arrays
        xr.testing.assert_allclose(tiledb_data_array, xarray_data_array)

    @pytest.mark.parametrize("simple_data_arrays", simple_examples_2d, indirect=True)
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
            (1, np.array([])),
        ],
    )
    def test_indexing_array_2D(self, simple_data_arrays, index1, index2):
        (tiledb_data_array, xarray_data_array) = simple_data_arrays
        xr.testing.assert_allclose(tiledb_data_array, xarray_data_array)

    @pytest.mark.parametrize("simple_data_arrays", simple_examples_2d, indirect=True)
    def test_indexing_array_nested_2D(self, simple_data_arrays):
        (tiledb_data_array, xarray_data_array) = simple_data_arrays
        result = tiledb_data_array[[0, 2, 2], [1, 3]][[0, 0, 2], 1]
        expected = xarray_data_array[[0, 2, 2], [1, 3]][[0, 0, 2], 1]
        xr.testing.assert_allclose(result, expected)
