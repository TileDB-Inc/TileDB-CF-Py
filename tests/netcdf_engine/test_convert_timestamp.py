# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf import NetCDF4ConverterEngine

netCDF4 = pytest.importorskip("netCDF4")


class TestCopyAtTimestamp:
    """Test copying a simple NetCDF file at a specified timestamp.

    NetCDF File:

    dimensions:
        x (8)

    variables:
        f (x) = np.linspace(0, 1, 8)
    """

    attr_data = np.linspace(0, 1, 8)

    def test_copy_to_timestamp(self, tmpdir):
        uri = str(tmpdir.mkdir("output").join("timestamp_array"))
        timestamp = 1
        with netCDF4.Dataset("tmp", mode="w", diskless=True) as dataset:
            dataset.setncatts({"title": "test timestamp"})
            dataset.createDimension("x", 8)
            var = dataset.createVariable("f", np.float64, ("x",))
            var[:] = self.attr_data
            converter = NetCDF4ConverterEngine.from_group(dataset)
            converter.convert_to_array(
                uri, input_netcdf_group=dataset, timestamp=timestamp
            )
        with tiledb.open(uri, timestamp=(1, 1)) as array:
            assert array.meta["title"] == "test timestamp"
            result_data = array[:]["f"]
            np.testing.assert_equal(self.attr_data, result_data)
