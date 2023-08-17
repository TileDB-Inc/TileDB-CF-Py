# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf import NetCDF4ConverterEngine, open_group_array

netCDF4 = pytest.importorskip("netCDF4")


class TestSimplyCopyChunks:
    """Test converting a simple NetCDF in chunks.

    NetCDF File:

    dimensions:
        x (8)
        y (8)
        z (8)

    variables:
        f (x, y, z) = reshape([0, ..., 511], (8, 8, 8))
    """

    attr_data = np.reshape(np.arange(512), (8, 8, 8))

    @pytest.fixture(scope="class")
    def netcdf_file(self, tmpdir_factory):
        """Returns the NetCDF file that will be used to test the conversion."""
        filepath = tmpdir_factory.mktemp("input_file").join("simple_copy_chunks.nc")
        with netCDF4.Dataset(filepath, mode="w") as dataset:
            dataset.createDimension("x", 8)
            dataset.createDimension("y", 8)
            dataset.createDimension("z", 8)
            var = dataset.createVariable(
                varname="f", datatype=np.int64, dimensions=("x", "y", "z")
            )
            var[:, :, :] = self.attr_data
        return filepath

    @pytest.mark.parametrize(
        "sparse,expected_result", ((False, attr_data), (True, np.arange(512)))
    )
    def test_convert_chunks(self, netcdf_file, tmpdir, sparse, expected_result):
        """Test copying NetCDF file in chunks for a simple NetCDF file."""
        uri = str(tmpdir.mkdir("output").join("simple_copy_chunks"))
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        array_creator = converter.get_array_creator_by_attr("f")
        array_creator.sparse = sparse
        assert array_creator.domain_creator.max_fragment_shape == (None, None, None)
        array_creator.domain_creator.max_fragment_shape = (4, 8, 2)
        assert array_creator.domain_creator.max_fragment_shape == (4, 8, 2)
        converter.convert_to_group(uri)
        with tiledb.Group(uri) as group:
            with open_group_array(group, attr="f") as array:
                array_uri = array.uri
                result = array[...]
        result = result["f"] if isinstance(result, dict) else result
        np.testing.assert_equal(result, expected_result)
        fragment_info = tiledb.FragmentInfoList(array_uri)
        assert len(fragment_info) == 8

    @pytest.mark.parametrize(
        "sparse,expected_result",
        ((False, np.reshape(np.arange(512), (8, 8, 8))), (True, np.arange(512))),
    )
    def test_convert_chunks_with_injected(
        self, netcdf_file, tmpdir, sparse, expected_result
    ):
        """Test copying NetCDF file in chunks for a simple NetCDF file with externally
        provided dimension and attribute values."""
        uri = str(tmpdir.mkdir("output").join("simple_copy_chunks"))
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        converter.add_shared_dim("t", domain=(0, 3), dtype=np.uint64)
        array_creator = converter.get_array_creator_by_attr("f")
        array_creator.sparse = sparse
        array_creator.add_attr_creator(name="g", dtype=np.float64)
        array_creator.domain_creator.inject_dim_creator("t", 0)
        array_creator.domain_creator.max_fragment_shape = (1, 4, 8, 2)
        # Define data for extra variable
        g_data = np.reshape(np.random.random_sample((512)), (1, 8, 8, 8))
        converter.convert_to_group(
            uri,
            assigned_dim_values={"t": 0},
            assigned_attr_values={"g": g_data},
        )
        with tiledb.Group(uri) as group:
            with open_group_array(group, array="array0") as array:
                array_uri = array.uri
                result = array[0, :, :, :]
        f_result = result["f"]
        np.testing.assert_equal(f_result, expected_result)
        g_result = np.reshape(result["g"], (1, 8, 8, 8))
        np.testing.assert_equal(g_data, g_result)
        fragment_info = tiledb.FragmentInfoList(array_uri)
        assert len(fragment_info) == 8


class TestCoordinateCopyChunks:
    """Test converting a simple NetCDF in chunks.

    NetCDF File:

    dimensions:
        x (8)
        y (8)

    variables:
        x (x) = linspace(-1, 1, 8)
        y (y) = linspace(0, 2, 8)
        f (x, y) = [[0, 1, ...],...,[...,62,63]]
    """

    x_data = np.arange(-4, 4)
    y_data = np.arange(10, 81, 10)
    attr_data = np.reshape(np.arange(64), (8, 8))

    @pytest.fixture(scope="class")
    def netcdf_file(self, tmpdir_factory):
        """Returns the NetCDF file that will be used to test the conversion."""
        filepath = tmpdir_factory.mktemp("input_file").join("simple_copy_chunks.nc")
        with netCDF4.Dataset(filepath, mode="w") as dataset:
            dataset.createDimension("x", 8)
            dataset.createDimension("y", 8)
            var = dataset.createVariable(
                varname="f", datatype=np.int64, dimensions=("x", "y")
            )
            var[:, :] = self.attr_data
            var = dataset.createVariable(
                varname="x", datatype=np.int64, dimensions=("x")
            )
            var[:] = self.x_data
            var = dataset.createVariable(
                varname="y", datatype=np.int64, dimensions=("y")
            )
            var[:] = self.y_data
        return filepath

    def test_convert_chunks(self, netcdf_file, tmpdir):
        """Test copying NetCDF file in chunks for a NetCDF to TileDB conversion that
        maps NetCDF coordinates to dimensions."""
        uri = str(tmpdir.mkdir("output").join("simple_copy_chunks"))
        converter = NetCDF4ConverterEngine.from_file(netcdf_file, coords_to_dims=True)
        converter.get_shared_dim("x").domain = (-4, 3)
        converter.get_shared_dim("y").domain = (10, 80)
        array_creator = converter.get_array_creator_by_attr("f")
        array_creator.domain_creator.max_fragment_shape = (4, 4)
        converter.convert_to_group(uri)
        with tiledb.Group(uri) as group:
            with open_group_array(group, attr="f") as array:
                array_uri = array.uri
                result = array[...]
        for x_value, y_value, f_value in zip(result["x"], result["y"], result["f"]):
            ix = np.argwhere(self.x_data == x_value)
            assert len(ix) == 1
            assert 0 <= ix[0] <= 7
            iy = np.argwhere(self.y_data == y_value)
            assert len(iy) == 1
            assert 0 <= iy[0] <= 7
            f_expected = self.attr_data[ix[0], iy[0]]
            assert f_value == f_expected
        fragment_info = tiledb.FragmentInfoList(array_uri)
        assert len(fragment_info) == 4
