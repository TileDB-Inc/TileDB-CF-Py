# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf import Group, NetCDF4ConverterEngine

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

    def test_convert_chunks(self, netcdf_file, tmpdir):
        """Test copying NetCDF file in chunks."""
        uri = str(tmpdir.mkdir("output").join("simple_copy_chunks"))
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        array_creator = converter.get_array_creator_by_attr("f")
        assert array_creator.domain_creator.max_fragment_shape == (None, None, None)
        array_creator.domain_creator.max_fragment_shape = (4, 8, 2)
        assert array_creator.domain_creator.max_fragment_shape == (4, 8, 2)
        converter.convert_to_group(uri)
        with Group(uri) as group:
            with group.open_array(attr="f") as array:
                array_uri = array.uri
                result = array[...]
        assert np.array_equal(result, self.attr_data)
        fragment_info = tiledb.FragmentInfoList(array_uri)
        assert len(fragment_info) == 8
