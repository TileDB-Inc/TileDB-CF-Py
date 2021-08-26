# Copyright 2021 TileDB Inc.
# Licensed under the MIT License
import numpy as np
import pytest

from tiledb.cf.creator import DataspaceRegistry
from tiledb.cf.engines.netcdf4_engine import NetCDF4CoordToDimConverter

netCDF4 = pytest.importorskip("netCDF4")


def test_coord_converter_simple():
    with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
        dataset.createDimension("x", 4)
        x = dataset.createVariable("x", datatype=np.float64, dimensions=("x",))
        registry = DataspaceRegistry()
        converter = NetCDF4CoordToDimConverter.from_netcdf(registry, x)
        assert converter.name == "x"
        assert converter.dtype == np.dtype("float64")
        assert converter.domain is None


def test_bad_size_error():
    with netCDF4.Dataset("example.nc", mode="w", diskless=True) as group:
        group.createDimension("x", 16)
        group.createDimension("y", 16)
        x = group.createVariable("x", np.dtype("float64"), ("x", "y"))
        with pytest.raises(ValueError):
            registry = DataspaceRegistry()
            NetCDF4CoordToDimConverter.from_netcdf(registry, x)
