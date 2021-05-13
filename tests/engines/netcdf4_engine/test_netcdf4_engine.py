# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

from tiledb.cf.engines.netcdf4_engine import NetCDFDimensionConverter, open_netcdf_group


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


def test_open_netcdf_group_with_group(tmpdir):
    netCDF4 = pytest.importorskip("netCDF4")
    filepath = str(tmpdir.mkdir("open_group").join("simple_dataset.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        with open_netcdf_group(dataset) as group:
            assert isinstance(group, netCDF4.Dataset)
            assert group == dataset


def test_open_netcdf_group_with_file(tmpdir):
    netCDF4 = pytest.importorskip("netCDF4")
    filepath = str(tmpdir.mkdir("open_group").join("simple_dataset.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        group1 = dataset.createGroup("group1")
        group1.createGroup("group2")
    with open_netcdf_group(input_file=filepath, group_path="/group1/group2") as group:
        assert isinstance(group, netCDF4.Group)
        assert group.path == "/group1/group2"


def test_open_netcdf_group_bad_type_error():
    with pytest.raises(TypeError):
        with open_netcdf_group("input_file"):
            pass


def test_open_netcdf_group_no_file_error():
    with pytest.raises(ValueError):
        with open_netcdf_group():
            pass


def test_open_netcdf_group_no_group_error():
    with pytest.raises(ValueError):
        with open_netcdf_group(input_file="test.nc"):
            pass
