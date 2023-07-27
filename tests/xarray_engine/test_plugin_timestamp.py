# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.

import numpy as np
import pytest

import tiledb

xr = pytest.importorskip("xarray")


class TestOpenDatasetTimestep:
    """Test reading an empty TileDB array into xarray."""

    @pytest.fixture(scope="class")
    def tiledb_uri(self, tmpdir_factory):
        """Creates a TileDB array and returns the URI."""
        uri = str(tmpdir_factory.mktemp("output").join("empty_array"))
        tiledb.Array.create(
            uri,
            tiledb.ArraySchema(
                domain=tiledb.Domain(
                    tiledb.Dim("x", domain=(0, 3), dtype=np.uint64),
                ),
                attrs=[tiledb.Attr("z", dtype=np.float64)],
            ),
        )
        with tiledb.open(uri, mode="w", timestamp=1) as array:
            array[:] = np.zeros((4))
            array.meta["global"] = 0
            array.meta["__tiledb_attr.z.variable"] = 0
        with tiledb.open(uri, mode="w", timestamp=2) as array:
            array[:] = np.ones((4))
            array.meta["global"] = 1
            array.meta["__tiledb_attr.z.variable"] = 1
        with tiledb.open(uri, mode="w", timestamp=3) as array:
            array[1] = 2
            array.meta["global"] = 2
            array.meta["__tiledb_attr.z.variable"] = 2
        with tiledb.open(uri, mode="w", timestamp=4) as array:
            array[2] = 3
            array.meta["global"] = 3
            array.meta["__tiledb_attr.z.variable"] = 3
        return uri

    def test_variable_data_timestamp_int(self, tiledb_uri):
        result = xr.open_dataset(tiledb_uri, timestamp=2, engine="tiledb")
        expected = xr.Dataset({"z": xr.DataArray(np.ones((4)), dims=("x",))})
        xr.testing.assert_equal(result, expected)

    def test_variable_metadata_timestamp_int(self, tiledb_uri):
        result = xr.open_dataset(tiledb_uri, timestamp=2, engine="tiledb")
        assert result["z"].attrs["variable"] == 1

    def test_global_metadata_timestamp_int(self, tiledb_uri):
        result = xr.open_dataset(tiledb_uri, timestamp=2, engine="tiledb")
        assert result.attrs["global"] == 1

    def test_variable_data_timestamp_tuple(self, tiledb_uri):
        result = xr.open_dataset(tiledb_uri, timestamp=(2, 3), engine="tiledb")
        expected = xr.Dataset({"z": xr.DataArray(np.array((1, 2, 1, 1)), dims=("x",))})
        xr.testing.assert_equal(result, expected)

    def test_variable_metadata_timestamp_tuple(self, tiledb_uri):
        result = xr.open_dataset(tiledb_uri, timestamp=(2, 3), engine="tiledb")
        assert result["z"].attrs["variable"] == 2

    def test_global_metadata_timestamp_tuple(self, tiledb_uri):
        result = xr.open_dataset(tiledb_uri, timestamp=(2, 3), engine="tiledb")
        assert result.attrs["global"] == 2
