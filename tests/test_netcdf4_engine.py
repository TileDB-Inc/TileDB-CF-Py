# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

try:
    import netCDF4

    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

from tiledb.cf.engines.netcdf4_engine import (
    NetCDF4ConverterEngine,
    NetCDFArrayConverter,
    NetCDFDimensionConverter,
    NetCDFVariableConverter,
    open_netcdf_group,
)

from . import NetCDF4TestCase

simple_dim_1 = NetCDF4TestCase("simple_dim_1", (("row", 8),), tuple(), {})


@pytest.mark.skipif(not HAS_NETCDF4, reason="netCDF4 not found")
def test_dim_converter_simple(tmpdir_factory):
    filepath = str(tmpdir_factory.mktemp("dim_converter").join("simple_dim.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dim = dataset.createDimension("row", 8)
        converter = NetCDFDimensionConverter.from_netcdf(dim, 10000, np.uint64)
        assert isinstance(repr(converter), str)
        assert converter.input_name == dim.name
        assert converter.input_size == dim.size
        assert not converter.is_unlimited
        assert converter.output_name == dim.name
        assert converter.output_domain == (0, dim.size - 1)
        assert converter.output_dtype == np.uint64


@pytest.mark.skipif(not HAS_NETCDF4, reason="netCDF4 not found")
def test_dim_converter_unlim(tmpdir_factory):
    filepath = str(tmpdir_factory.mktemp("dim_converter").join("unlim_dim.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dim = dataset.createDimension("row", None)
        max_size = 1000000
        converter = NetCDFDimensionConverter.from_netcdf(dim, max_size, np.uint64)
        assert isinstance(repr(converter), str)
        assert converter.input_name == dim.name
        assert converter.input_size == dim.size
        assert converter.is_unlimited
        assert converter.output_name == dim.name
        assert converter.output_domain == (0, max_size - 1)
        assert converter.output_dtype == np.uint64


def test_array_converter_tiles_from_chunks():
    dims = (
        NetCDFDimensionConverter("row", 8, False, "row", (0, 7), np.dtype("uint64")),
        NetCDFDimensionConverter("col", 8, False, "col", (0, 7), np.dtype("uint64")),
    )
    var = [
        NetCDFVariableConverter(
            "variable",
            np.dtype("float64"),
            -1.0,
            (4, 4),
            "a1",
            np.dtype("float64"),
            -1.0,
        ),
        NetCDFVariableConverter(
            "variable",
            np.dtype("int32"),
            None,
            (4, 4),
            "a2",
            np.dtype("int32"),
            None,
        ),
    ]
    array_converter = NetCDFArrayConverter(dims, var)
    assert array_converter.tiles == (4, 4)


def test_array_converter_mixed_chunks():
    dims = (
        NetCDFDimensionConverter("row", 8, False, "row", (0, 7), np.dtype("uint64")),
        NetCDFDimensionConverter("col", 8, False, "col", (0, 7), np.dtype("uint64")),
    )
    var = [
        NetCDFVariableConverter(
            "variable",
            np.dtype("float64"),
            -1.0,
            (4, 4),
            "a1",
            np.dtype("float64"),
            -1.0,
        ),
        NetCDFVariableConverter(
            "variable",
            np.dtype("int32"),
            None,
            (2, 4),
            "a2",
            np.dtype("int32"),
            None,
        ),
    ]
    array_converter = NetCDFArrayConverter(dims, var)
    assert array_converter.tiles is None


def test_array_converter_mixed_chunks_with_none():
    dims = (
        NetCDFDimensionConverter("row", 8, False, "row", (0, 7), np.dtype("uint64")),
        NetCDFDimensionConverter("col", 8, False, "col", (0, 7), np.dtype("uint64")),
    )
    var = [
        NetCDFVariableConverter(
            "variable",
            np.dtype("float64"),
            -1.0,
            (4, 4),
            "a1",
            np.dtype("float64"),
            -1.0,
        ),
        NetCDFVariableConverter(
            "variable",
            np.dtype("int32"),
            None,
            None,
            "a2",
            np.dtype("int32"),
            None,
        ),
    ]
    array_converter = NetCDFArrayConverter(dims, var)
    assert array_converter.tiles == (4, 4)


def test_array_converter_from_chunks_1D():
    dims = (
        NetCDFDimensionConverter("row", 8, False, "row", (0, 7), np.dtype("uint64")),
    )
    var = [
        NetCDFVariableConverter(
            "variable",
            np.dtype("float64"),
            -1.0,
            (4,),
            "a1",
            np.dtype("float64"),
            -1.0,
        ),
        NetCDFVariableConverter(
            "variable",
            np.dtype("int32"),
            None,
            (4,),
            "a2",
            np.dtype("int32"),
            None,
        ),
    ]
    array_converter = NetCDFArrayConverter(dims, var)
    assert array_converter.tiles == (4,)


def test_array_converter_hardset_no_tiles():
    dims = (
        NetCDFDimensionConverter("row", 8, False, "row", (0, 7), np.dtype("uint64")),
        NetCDFDimensionConverter("col", 8, False, "col", (0, 7), np.dtype("uint64")),
    )
    var = [
        NetCDFVariableConverter(
            "variable",
            np.dtype("float64"),
            -1.0,
            (4, 4),
            "attr",
            np.dtype("float64"),
            -1.0,
        ),
    ]
    tiles = tuple()
    array_converter = NetCDFArrayConverter(dims, var, tiles)
    assert array_converter.tiles is None


def test_rename_array():
    dim_converters = (
        NetCDFDimensionConverter("row", 8, False, "row", (0, 7), np.dtype("uint64")),
        NetCDFDimensionConverter("col", 8, False, "col", (0, 7), np.dtype("uint64")),
    )
    var_converters = [
        NetCDFVariableConverter(
            "variable",
            np.dtype("float64"),
            -1.0,
            (4, 4),
            "a1",
            np.dtype("float64"),
            -1.0,
        ),
        NetCDFVariableConverter(
            "variable",
            np.dtype("int32"),
            None,
            (4, 4),
            "a2",
            np.dtype("int32"),
            None,
        ),
    ]
    array_converters = {"A1": NetCDFArrayConverter(dim_converters, var_converters)}
    converter = NetCDF4ConverterEngine(
        {dim.output_name: dim for dim in dim_converters},
        {var.output_name: var for var in var_converters},
        array_converters,
    )
    converter.rename_array("A1", "B1")
    assert set(converter.array_names) == {"B1"}


def test_rename_array_name_exists_error():
    dim_converters = (
        NetCDFDimensionConverter("row", 8, False, "row", (0, 7), np.dtype("uint64")),
        NetCDFDimensionConverter("col", 8, False, "col", (0, 7), np.dtype("uint64")),
    )
    var1 = NetCDFVariableConverter(
        "variable",
        np.dtype("float64"),
        -1.0,
        (4, 4),
        "a1",
        np.dtype("float64"),
        -1.0,
    )
    var2 = NetCDFVariableConverter(
        "variable",
        np.dtype("int32"),
        None,
        (4, 4),
        "a2",
        np.dtype("int32"),
        None,
    )
    array_converters = {
        "A1": NetCDFArrayConverter(dim_converters, [var1]),
        "B1": NetCDFArrayConverter(dim_converters, [var2]),
    }
    converter = NetCDF4ConverterEngine(
        {dim.output_name: dim for dim in dim_converters},
        {"a1": var1, "a2": var2},
        array_converters,
    )
    with pytest.raises(ValueError):
        converter.rename_array("A1", "B1")


def test_rename_attr():
    dim_converters = (
        NetCDFDimensionConverter("row", 8, False, "row", (0, 7), np.dtype("uint64")),
        NetCDFDimensionConverter("col", 8, False, "col", (0, 7), np.dtype("uint64")),
    )
    var_converters = [
        NetCDFVariableConverter(
            "variable",
            np.dtype("float64"),
            -1.0,
            (4, 4),
            "a1",
            np.dtype("float64"),
            -1.0,
        ),
        NetCDFVariableConverter(
            "variable",
            np.dtype("int32"),
            None,
            (4, 4),
            "a2",
            np.dtype("int32"),
            None,
        ),
    ]
    array_converters = {"A1": NetCDFArrayConverter(dim_converters, var_converters)}
    converter = NetCDF4ConverterEngine(
        {dim.output_name: dim for dim in dim_converters},
        {var.output_name: var for var in var_converters},
        array_converters,
    )
    converter.rename_attr("a1", "b1")
    assert set(converter.attr_names) == {"a2", "b1"}


def test_rename_attr_name_exists_error():
    dim_converters = (
        NetCDFDimensionConverter("row", 8, False, "row", (0, 7), np.dtype("uint64")),
        NetCDFDimensionConverter("col", 8, False, "col", (0, 7), np.dtype("uint64")),
    )
    var_converters = [
        NetCDFVariableConverter(
            "variable",
            np.dtype("float64"),
            -1.0,
            (4, 4),
            "a1",
            np.dtype("float64"),
            -1.0,
        ),
        NetCDFVariableConverter(
            "variable",
            np.dtype("int32"),
            None,
            (4, 4),
            "b1",
            np.dtype("int32"),
            None,
        ),
    ]
    array_converters = {"A1": NetCDFArrayConverter(dim_converters, var_converters)}
    converter = NetCDF4ConverterEngine(
        {dim.output_name: dim for dim in dim_converters},
        {var.output_name: var for var in var_converters},
        array_converters,
    )
    with pytest.raises(ValueError):
        converter.rename_attr("a1", "b1")


def test_rename_dim():
    dim_converters = (
        NetCDFDimensionConverter("row", 8, False, "row", (0, 7), np.dtype("uint64")),
    )
    var_converters = [
        NetCDFVariableConverter(
            "variable",
            np.dtype("float64"),
            -1.0,
            (4,),
            "a1",
            np.dtype("float64"),
            -1.0,
        ),
    ]
    array_converters = {"A1": NetCDFArrayConverter(dim_converters, var_converters)}
    converter = NetCDF4ConverterEngine(
        {dim.output_name: dim for dim in dim_converters},
        {var.output_name: var for var in var_converters},
        array_converters,
    )
    converter.rename_dim("row", "dim1")
    assert set(converter.dim_names) == {"dim1"}


def test_rename_dim_name_exists_error():
    dim_converters = (
        NetCDFDimensionConverter("row", 8, False, "row", (0, 7), np.dtype("uint64")),
        NetCDFDimensionConverter("col", 4, False, "col", (0, 4), np.dtype("uint64")),
    )
    var_converters = [
        NetCDFVariableConverter(
            "variable",
            np.dtype("float64"),
            -1.0,
            (4, 4),
            "a1",
            np.dtype("float64"),
            -1.0,
        ),
    ]
    array_converters = {"A1": NetCDFArrayConverter(dim_converters, var_converters)}
    converter = NetCDF4ConverterEngine(
        {dim.output_name: dim for dim in dim_converters},
        {var.output_name: var for var in var_converters},
        array_converters,
    )
    with pytest.raises(ValueError):
        converter.rename_dim("row", "col")


def test_open_netcdf_group_with_group(tmpdir):
    filepath = str(tmpdir.mkdir("open_group").join("simple_dataset.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        with open_netcdf_group(dataset) as group:
            assert isinstance(group, netCDF4.Dataset)
            assert group == dataset


def test_open_netcdf_group_bad_type_error():
    with pytest.raises(TypeError):
        with open_netcdf_group("input_file"):
            pass


def test_open_netcdf_group_with_file(tmpdir):
    filepath = str(tmpdir.mkdir("open_group").join("simple_dataset.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        group1 = dataset.createGroup("group1")
        group1.createGroup("group2")
    with open_netcdf_group(input_file=filepath, group_path="/group1/group2") as group:
        assert isinstance(group, netCDF4.Group)
        assert group.path == "/group1/group2"


def test_open_netcdf_group_no_file_error():
    with pytest.raises(ValueError):
        with open_netcdf_group():
            pass


def test_open_netcdf_group_no_group_error():
    with pytest.raises(ValueError):
        with open_netcdf_group(input_file="test.nc"):
            pass
