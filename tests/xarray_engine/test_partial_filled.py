# Copyright 2022 TileDB Inc.
# Licensed under the MIT License.

import sys

import numpy as np
import pytest

import tiledb

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 9), reason="xarray requires python3.9 or higher"
)


class TestEmptyArray:
    """Test reading an empty TileDB array into xarray."""

    @pytest.fixture(scope="class")
    def tiledb_uri(self, tmpdir_factory):
        """Creates a TileDB array and returns the URI."""
        uri = str(tmpdir_factory.mktemp("output").join("empty_array.tiledb"))
        tiledb.Group.create(uri)
        array_uri = f"{uri}/z"
        tiledb.Array.create(
            array_uri,
            tiledb.ArraySchema(
                domain=tiledb.Domain(
                    tiledb.Dim("x", domain=(0, 7), dtype=np.uint64),
                    tiledb.Dim("y", domain=(0, 7), dtype=np.uint64),
                ),
                attrs=[tiledb.Attr("z", dtype=np.float64)],
            ),
        )
        with tiledb.Group(uri, mode="w") as group:
            group.add(name="z", uri="z", relative=True)
            group.meta["__tiledb_array_flexible_dimensions.z"] = "x;y"
        return uri

    def test_open_dataset(self, tiledb_uri):
        result = xr.open_dataset(tiledb_uri, engine="tiledb")
        expected = xr.Dataset(
            {"z": xr.DataArray(np.array(()).reshape(0, 0), dims=("x", "y"))}
        )
        xr.testing.assert_equal(result, expected)


class TestFrontFilledArray:
    """Test reading a TileDB array into xarray where the nonempty domain
    is smaller than the entire domain."""

    z_data = np.random.rand(16).reshape(4, 4)

    @pytest.fixture(scope="class")
    def tiledb_uri(self, tmpdir_factory):
        """Creates a TileDB array and returns the URI."""
        uri = str(tmpdir_factory.mktemp("output").join("front_half_filled"))
        tiledb.Group.create(uri)
        array_uri = f"{uri}/z"
        tiledb.Array.create(
            array_uri,
            tiledb.ArraySchema(
                domain=tiledb.Domain(
                    tiledb.Dim("x", domain=(0, 7), dtype=np.uint64),
                    tiledb.Dim("y", domain=(0, 7), dtype=np.uint64),
                ),
                attrs=[tiledb.Attr("z", np.float64)],
            ),
        )
        with tiledb.open(array_uri, mode="w") as array:
            array[0:4, 0:4] = self.z_data
        with tiledb.Group(uri, mode="w") as group:
            group.add(name="z", uri="z", relative=True)
            group.meta["__tiledb_array_flexible_dimensions.z"] = "x;y"
        return uri

    def test_open_dataset(self, tiledb_uri):
        result = xr.open_dataset(tiledb_uri, engine="tiledb")
        expected = xr.Dataset({"z": xr.DataArray(self.z_data, dims=("x", "y"))})
        xr.testing.assert_equal(result, expected)


class TestBackFilledArray:
    """Test reading a TileDB array into xarray where the nonempty domain
    is smaller than the entire domain."""

    z_data = np.random.rand(16).reshape(4, 4)

    @pytest.fixture(scope="class")
    def tiledb_uri(self, tmpdir_factory):
        """Creates a TileDB array and returns the URI."""
        uri = str(tmpdir_factory.mktemp("output").join("back_half_filled"))
        tiledb.Group.create(uri)
        array_uri = f"{uri}/z"
        tiledb.Array.create(
            array_uri,
            tiledb.ArraySchema(
                domain=tiledb.Domain(
                    tiledb.Dim("x", domain=(0, 7), dtype=np.uint64),
                    tiledb.Dim("y", domain=(0, 7), dtype=np.uint64),
                ),
                attrs=[tiledb.Attr("z", np.float64)],
            ),
        )
        with tiledb.open(array_uri, mode="w") as array:
            array[4:, 4:] = self.z_data
        with tiledb.Group(uri, mode="w") as group:
            group.add(name="z", uri="z", relative=True)
            group.meta["__tiledb_array_flexible_dimensions.z"] = "x;y"
        return uri

    def test_open_dataset(self, tiledb_uri):
        expected_data = np.full((8, 8), np.nan)
        expected_data[4:, 4:] = self.z_data
        expected = xr.Dataset({"z": xr.DataArray(expected_data, dims=("x", "y"))})
        result = xr.open_dataset(tiledb_uri, engine="tiledb")
        xr.testing.assert_equal(result, expected)


class TestMiddleFilledArray:
    """Test reading a TileDB array into xarray where the nonempty domain
    is smaller than the entire domain."""

    z_data = np.random.rand(16).reshape(4, 4)

    @pytest.fixture(scope="class")
    def tiledb_uri(self, tmpdir_factory):
        """Creates a TileDB array and returns the URI."""
        uri = str(tmpdir_factory.mktemp("output").join("middle_filled"))
        tiledb.Group.create(uri)
        array_uri = f"{uri}/z"
        tiledb.Array.create(
            array_uri,
            tiledb.ArraySchema(
                domain=tiledb.Domain(
                    tiledb.Dim("x", domain=(0, 7), dtype=np.uint64),
                    tiledb.Dim("y", domain=(0, 7), dtype=np.uint64),
                ),
                attrs=[tiledb.Attr("z", np.float64)],
            ),
        )
        with tiledb.open(array_uri, mode="w") as array:
            array[2:6, 2:6] = self.z_data
        with tiledb.Group(uri, mode="w") as group:
            group.add(name="z", uri="z", relative=True)
            group.meta["__tiledb_array_flexible_dimensions.z"] = "x;y"
        return uri

    def test_open_dataset(self, tiledb_uri):
        expected_data = np.full((6, 6), np.nan)
        expected_data[2:6, 2:6] = self.z_data
        expected = xr.Dataset({"z": xr.DataArray(expected_data, dims=("x", "y"))})
        result = xr.open_dataset(tiledb_uri, engine="tiledb")
        xr.testing.assert_equal(result, expected)
