# Copyright 2021 TileDB Inc.
# Licensed under the MIT License
import numpy as np
import pytest

from tiledb.cf.engines.netcdf4_engine import NetCDFCoordToDimConverter


def test_coord_converter_simple(simple_coord_netcdf_example):
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(simple_coord_netcdf_example.filepath, mode="r") as dataset:
        x = dataset.variables["x"]
        converter = NetCDFCoordToDimConverter.from_netcdf(x)
        assert converter.name == "x"
        assert converter.dtype == np.dtype("float64")
        assert converter.domain == (None, None)


def test_bad_size_error(tmpdir):
    netCDF4 = pytest.importorskip("netCDF4")
    filepath = tmpdir.mkdir("examples").join("bad_size.nc")
    with netCDF4.Dataset(filepath, mode="w") as group:
        group.createDimension("x", 16)
        group.createDimension("y", 16)
        x = group.createVariable("x", np.dtype("float64"), ("x", "y"))
        with pytest.raises(ValueError):
            NetCDFCoordToDimConverter.from_netcdf(x)
