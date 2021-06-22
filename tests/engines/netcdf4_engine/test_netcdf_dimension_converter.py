# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

from tiledb.cf.engines.netcdf4_engine import (
    NetCDFDimToDimConverter,
    NetCDFScalarDimConverter,
)


class TestNetCDFDimToDimConverterSimpleDim:
    """This class tests the NetCDFDimToDimConverter class for a simple NetCDF
    dimension.

    This test uses an example NetCDF file with a dimension row(8) in the root
    group.
    """

    def test_class_properties(self):
        netCDF4 = pytest.importorskip("netCDF4")
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", 8)
            converter = NetCDFDimToDimConverter.from_netcdf(dim, 1000, np.uint64)
            assert isinstance(repr(converter), str)
            assert converter.input_name == dim.name
            assert converter.input_size == dim.size
            assert not converter.is_unlimited
            assert converter.name == dim.name
            assert converter.domain == (0, dim.size - 1)
            assert converter.dtype == np.uint64

    def test_sparse_values(self):
        netCDF4 = pytest.importorskip("netCDF4")
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", 8)
            converter = NetCDFDimToDimConverter.from_netcdf(dim, 1000, np.uint64)
            values = converter.get_values(dataset, sparse=True)
            assert np.array_equal(np.arange(0, 8), values)

    def test_sparse_values_from_subgroup(self):
        netCDF4 = pytest.importorskip("netCDF4")
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", 8)
            group = dataset.createGroup("group1")
            converter = NetCDFDimToDimConverter.from_netcdf(dim, 1000, np.uint64)
            values = converter.get_values(group, sparse=True)
            assert np.array_equal(np.arange(0, 8), values)

    def test_dense_values(self):
        netCDF4 = pytest.importorskip("netCDF4")
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", 8)
            converter = NetCDFDimToDimConverter.from_netcdf(dim, 1000, np.uint64)
            values = converter.get_values(dataset, sparse=False)
            assert slice(8) == values

    def test_dense_values_from_subgroup(self):
        netCDF4 = pytest.importorskip("netCDF4")
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", 8)
            group = dataset.createGroup("group1")
            converter = NetCDFDimToDimConverter.from_netcdf(dim, 1000, np.uint64)
            values = converter.get_values(group, sparse=False)
            assert slice(8) == values

    def test_no_dim_error(self):
        netCDF4 = pytest.importorskip("netCDF4")
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", 8)
            converter = NetCDFDimToDimConverter.from_netcdf(dim, 1000, np.uint64)
        with netCDF4.Dataset("no_dims.nc", mode="w", diskless=True) as dataset:
            group = dataset.createGroup("group")
            with pytest.raises(KeyError):
                converter.get_values(group, sparse=False)


class TestNetCDFDimToDimConverterUnlimitedDim:
    """This class teests the NetCDFDimToDimConverter class for an unlimited
    NetCDF dimension."""

    def test_class_properties(self):
        netCDF4 = pytest.importorskip("netCDF4")
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", None)
            max_size = 100
            converter = NetCDFDimToDimConverter.from_netcdf(dim, max_size, np.uint64)
            assert isinstance(repr(converter), str)
            assert converter.input_name == dim.name
            assert converter.input_size == dim.size
            assert converter.is_unlimited
            assert converter.name == dim.name
            assert converter.domain == (0, max_size - 1)
            assert converter.dtype == np.uint64

    def test_sparse_values_no_data(self):
        netCDF4 = pytest.importorskip("netCDF4")
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", None)
            converter = NetCDFDimToDimConverter.from_netcdf(dim, 100, np.uint64)
            with pytest.raises(ValueError):
                converter.get_values(dataset, sparse=True)

    def test_dense_values_no_data(self):
        netCDF4 = pytest.importorskip("netCDF4")
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", None)
            converter = NetCDFDimToDimConverter.from_netcdf(dim, 100, np.uint64)
            with pytest.raises(ValueError):
                converter.get_values(dataset, sparse=False)

    def test_get_sparse_values(self):
        netCDF4 = pytest.importorskip("netCDF4")
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", None)
            var = dataset.createVariable("data", np.int32, ("row",))
            size = 10
            var[:] = np.arange(size)
            converter = NetCDFDimToDimConverter.from_netcdf(dim, 100, np.uint64)
            values = converter.get_values(dataset, sparse=True)
            assert np.array_equal(np.arange(0, size), values)

    def test_get_dense_values(self):
        netCDF4 = pytest.importorskip("netCDF4")
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", None)
            var = dataset.createVariable("data", np.int32, ("row",))
            size = 10
            var[:] = np.arange(size)
            converter = NetCDFDimToDimConverter.from_netcdf(dim, 100, np.uint64)
            values = converter.get_values(dataset, sparse=False)
            assert slice(size) == values

    def test_data_too_large_error(self):
        netCDF4 = pytest.importorskip("netCDF4")
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", None)
            var = dataset.createVariable("data", np.int32, ("row",))
            size = 11
            var[:] = np.arange(size)
            converter = NetCDFDimToDimConverter.from_netcdf(dim, 10, np.uint64)
            with pytest.raises(IndexError):
                converter.get_values(dataset, sparse=True)


class TestNetCDFScalarDimConverter:
    def test_class_properties(self):
        converter = NetCDFScalarDimConverter.create("__scalars", np.uint32)
        assert converter.name == "__scalars"
        assert converter.domain == (0, 0)
        assert converter.dtype == np.dtype(np.uint32)

    def test_repr(self):
        converter = NetCDFScalarDimConverter.create("__scalars", np.uint32)
        isinstance(repr(converter), str)

    def test_sparse_values(self):
        netCDF4 = pytest.importorskip("netCDF4")
        converter = NetCDFScalarDimConverter.create("__scalars", np.uint32)
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            values = converter.get_values(dataset, sparse=True)
            assert np.array_equal(values, np.array([0]))

    def test_dense_values(self):
        netCDF4 = pytest.importorskip("netCDF4")
        converter = NetCDFScalarDimConverter.create("__scalars", np.uint32)
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            values = converter.get_values(dataset, sparse=False)
            assert np.array_equal(values, slice(1))
