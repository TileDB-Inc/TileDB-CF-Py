# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

from tiledb.cf.creator import DataspaceRegistry, SharedDim

netcdf_engine = pytest.importorskip("tiledb.cf.netcdf_engine")


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
