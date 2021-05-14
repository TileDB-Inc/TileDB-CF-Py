# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

from tiledb.cf.engines.netcdf4_engine import NetCDFDimensionConverter


def test_dim_converter_simple(tmpdir_factory):
    netCDF4 = pytest.importorskip("netCDF4")
    filepath = str(tmpdir_factory.mktemp("dim_converter").join("simple_dim.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dim = dataset.createDimension("row", 8)
        converter = NetCDFDimensionConverter.from_netcdf(dim, 10000, np.uint64)
        assert isinstance(repr(converter), str)
        assert converter.input_name == dim.name
        assert converter.input_size == dim.size
        assert not converter.is_unlimited
        assert converter.name == dim.name
        assert converter.domain == (0, dim.size - 1)
        assert converter.dtype == np.uint64


def test_dim_converter_unlim(tmpdir_factory):
    netCDF4 = pytest.importorskip("netCDF4")
    filepath = str(tmpdir_factory.mktemp("dim_converter").join("unlim_dim.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dim = dataset.createDimension("row", None)
        max_size = 1000000
        converter = NetCDFDimensionConverter.from_netcdf(dim, max_size, np.uint64)
        assert isinstance(repr(converter), str)
        assert converter.input_name == dim.name
        assert converter.input_size == dim.size
        assert converter.is_unlimited
        assert converter.name == dim.name
        assert converter.domain == (0, max_size - 1)
        assert converter.dtype == np.uint64
