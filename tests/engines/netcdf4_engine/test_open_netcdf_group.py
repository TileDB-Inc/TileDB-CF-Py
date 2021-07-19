# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import pytest

from tiledb.cf.engines.netcdf4_engine import open_netcdf_group

netCDF4 = pytest.importorskip("netCDF4")


def test_open_netcdf_group_with_group(tmpdir):
    with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
        with open_netcdf_group(dataset) as group:
            assert isinstance(group, netCDF4.Dataset)
            assert group == dataset


def test_open_netcdf_group_with_file(tmpdir):
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
