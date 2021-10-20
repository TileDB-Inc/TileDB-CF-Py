# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf.creator import DataspaceRegistry, SharedDim

netCDF4 = pytest.importorskip("netCDF4")
netcdf_engine = pytest.importorskip("tiledb.cf.netcdf_engine")


class TestAttrsFilters:
    """Collection of tests for setting default attribute filters."""

    def test_default_filter(self):
        """Tests new attribute filter is set to the attrs_filters value if the
        ``filters`` parameter is not specified."""
        attrs_filters = tiledb.FilterList([tiledb.ZstdFilter()])
        registry = DataspaceRegistry()
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", 64)
            var = dataset.createVariable("x", np.float64, ("row",))
            netcdf_engine.NetCDF4DimToDimConverter.from_netcdf(
                registry, dim, None, np.uint64
            )
            converter = netcdf_engine.NetCDF4ArrayConverter(
                registry, "array", ("row",), attrs_filters=attrs_filters
            )
            converter.add_var_to_attr_converter(var)
        assert converter.attr_creator("x").filters == attrs_filters

    def test_overwrite_default_filters(self):
        """Tests new attribute filter is set to the provided ``filters`` parameter when
        ``filters is not ``None``."""
        attrs_filters = tiledb.FilterList([tiledb.ZstdFilter()])
        new_filters = tiledb.FilterList([tiledb.GzipFilter(level=5)])
        registry = DataspaceRegistry()
        with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("row", 64)
            var = dataset.createVariable("x", np.float64, ("row",))
            netcdf_engine.NetCDF4DimToDimConverter.from_netcdf(
                registry, dim, None, np.uint64
            )
            converter = netcdf_engine.NetCDF4ArrayConverter(
                registry, "array", ("row",), attrs_filters=attrs_filters
            )
            converter.add_var_to_attr_converter(var, filters=new_filters)
        assert converter.attr_creator("x").filters == new_filters


def test_remove_dim_creator_front():
    """Tests removing a dimension in the front of the domain."""
    registry = DataspaceRegistry()
    SharedDim(registry, "x0", (0, 7), np.uint32)
    SharedDim(registry, "x1", (0, 7), np.uint32)
    SharedDim(registry, "x2", (0, 4), np.uint32)
    creator = netcdf_engine.NetCDF4ArrayConverter(registry, "array", ("x0", "x1", "x2"))
    creator.domain_creator.remove_dim_creator("x0")
    dim_names = tuple(dim_creator.name for dim_creator in creator.domain_creator)
    assert dim_names == ("x1", "x2")


def test_remove_dim_creator_back():
    """Tests removing a dimension in the back of the domain."""
    registry = DataspaceRegistry()
    SharedDim(registry, "x1", (0, 7), np.uint32)
    SharedDim(registry, "x2", (0, 7), np.uint32)
    SharedDim(registry, "x3", (0, 4), np.uint32)
    creator = netcdf_engine.NetCDF4ArrayConverter(registry, "array", ("x1", "x2", "x3"))
    creator.domain_creator.remove_dim_creator("x3")
    dim_names = tuple(dim_creator.name for dim_creator in creator.domain_creator)
    assert dim_names == ("x1", "x2")


def test_remove_dim_creator_middle():
    """Tests removing a dimension in the middle of the domain."""
    registry = DataspaceRegistry()
    SharedDim(registry, "x0", (0, 7), np.uint32)
    SharedDim(registry, "x1", (0, 7), np.uint32)
    SharedDim(registry, "x2", (0, 4), np.uint32)
    creator = netcdf_engine.NetCDF4ArrayConverter(registry, "array", ("x0", "x1", "x2"))
    creator.domain_creator.remove_dim_creator("x1")
    dim_names = tuple(dim_creator.name for dim_creator in creator.domain_creator)
    assert dim_names == ("x0", "x2")


def test_remove_dim_creator_key_error():
    """Tests key error when removing a dimension by name."""
    registry = DataspaceRegistry()
    SharedDim(registry, "x0", (0, 7), np.uint32)
    SharedDim(registry, "x1", (0, 7), np.uint32)
    SharedDim(registry, "x2", (0, 4), np.uint32)
    creator = netcdf_engine.NetCDF4ArrayConverter(registry, "array", ("x0", "x1", "x2"))
    with pytest.raises(KeyError):
        creator.domain_creator.remove_dim_creator("x4")


def test_set_max_fragment_shape_error():
    """Tests raising an error when attempting to set max_fragment_shape with a value
    that is a bad length."""
    registry = DataspaceRegistry()
    SharedDim(registry, "x", (0, 7), np.uint32)
    creator = netcdf_engine.NetCDF4ArrayConverter(registry, "array", ("x"))
    creator.add_attr_creator("y0", dtype=np.dtype("int32"))
    with pytest.raises(ValueError):
        creator.domain_creator.max_fragment_shape = (None, None)


def test_array_converter_indexer_error():
    """Tests value error when copying with an indexer of bad length."""
    registry = DataspaceRegistry()
    SharedDim(registry, "x", (0, 7), np.uint32)
    creator = netcdf_engine.NetCDF4ArrayConverter(registry, "array", ("x"))
    creator.add_attr_creator("y0", dtype=np.dtype("int32"))
    with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
        with pytest.raises(ValueError):
            creator.domain_creator.get_query_coordinates(
                netcdf_group=dataset,
                sparse=False,
                indexer=[slice(None), slice(None)],
                assigned_dim_values={"x": 0},
            )
