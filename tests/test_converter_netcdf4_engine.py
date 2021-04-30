# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf import Group, from_netcdf, from_netcdf_group
from tiledb.cf.engines.netcdf4_engine import NetCDF4ConverterEngine

from . import NetCDF4TestCase

netCDF4 = pytest.importorskip("netCDF4")

simple_coord_1 = NetCDF4TestCase(
    "simple_coord_1",
    (("row", 4), ("col", 4)),
    (
        ("data", np.dtype("uint16"), ("row", "col")),
        ("x", np.dtype("uint16"), ("row",)),
        ("y", np.dtype("uint16"), ("col",)),
        ("row", np.dtype("float64"), ("row",)),
    ),
    {
        "data": np.array(
            ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16])
        ),
        "x": np.array([1, 2, 3, 4]),
        "y": np.array([5, 6, 7, 8]),
        "row": np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float64),
    },
)

simple_unlim_dim = NetCDF4TestCase(
    "simple_unlim_dim",
    (("row", None), ("col", 4)),
    (
        ("data", np.dtype("uint16"), ("row", "col")),
        ("x", np.dtype("uint16"), ("row",)),
        ("y", np.dtype("uint16"), ("col",)),
    ),
    {
        "data": np.array(
            ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16])
        ),
        "x": np.array([1, 2, 3, 4]),
        "y": np.array([5, 6, 7, 8]),
    },
)

scalar_variables = NetCDF4TestCase(
    "scalar_variables",
    tuple(),
    (
        ("x", np.dtype("int32"), tuple()),
        ("y", np.dtype("int32"), tuple()),
    ),
    {
        "x": np.array([1]),
        "y": np.array([5]),
    },
)


attr_to_var_map = {
    "simple_coord_1": {"data": "data", "x": "x", "y": "y", "row.axis_data": "row"},
    "simple_unlim_dim": {"data": "data", "x": "x", "y": "y"},
    "scalar_variables": {"x": "x", "y": "y"},
}


@pytest.mark.parametrize(
    "netcdf4_test_case",
    [simple_coord_1, simple_unlim_dim, scalar_variables],
    indirect=True,
)
def test_from_netcdf(netcdf4_test_case, tmpdir):
    """Integration test for `from_netcdf_file` function call."""
    name, filename, test_case = netcdf4_test_case
    uri = str(tmpdir.mkdir("output").join(name))
    from_netcdf(filename, uri)
    for attr_name, var_name in attr_to_var_map[name].items():
        with Group(uri, attr=attr_name) as group:
            nonempty_domain = group.array.nonempty_domain()
            result = group.array.multi_index[nonempty_domain]
        assert np.array_equal(
            result[attr_name], test_case.variable_data[var_name]
        ), f"unexpected values for attribute '{attr_name}'"


@pytest.mark.parametrize(
    "netcdf4_test_case",
    [simple_coord_1, simple_unlim_dim, scalar_variables],
    indirect=True,
)
def test_from_netcdf_group(netcdf4_test_case, tmpdir):
    """Integration test for `from_netcdf_group` function call."""
    name, filename, test_case = netcdf4_test_case
    uri = str(tmpdir.mkdir("output").join(name))
    with netCDF4.Dataset(filename) as dataset:
        from_netcdf_group(dataset, uri)
    for attr_name, var_name in attr_to_var_map[name].items():
        with Group(uri, attr=attr_name) as group:
            nonempty_domain = group.array.nonempty_domain()
            result = group.array.multi_index[nonempty_domain]
        assert np.array_equal(
            result[attr_name], test_case.variable_data[var_name]
        ), f"unexpected values for attribute '{attr_name}'"


@pytest.mark.parametrize(
    "netcdf4_test_case",
    [simple_coord_1, simple_unlim_dim, scalar_variables],
    indirect=True,
)
def test_from_netcdf_group2(netcdf4_test_case, tmpdir):
    """Integration test for `from_netcdf_group` function call."""
    name, filename, test_case = netcdf4_test_case
    uri = str(tmpdir.mkdir("output").join(name))
    from_netcdf_group(filename, uri)
    for attr_name, var_name in attr_to_var_map[name].items():
        with Group(uri, attr=attr_name) as group:
            nonempty_domain = group.array.nonempty_domain()
            result = group.array.multi_index[nonempty_domain]
        assert np.array_equal(
            result[attr_name], test_case.variable_data[var_name]
        ), f"unexpected values for attribute '{attr_name}'"


@pytest.mark.parametrize(
    "netcdf4_test_case", [simple_coord_1, simple_unlim_dim], indirect=True
)
def test_converter_from_netcdf(netcdf4_test_case, tmpdir):
    name, filename, test_case = netcdf4_test_case
    converter = NetCDF4ConverterEngine.from_file(filename)
    uri = str(tmpdir.mkdir("output").join(name))
    assert isinstance(repr(converter), str)
    converter.convert(uri)
    for attr_name, var_name in attr_to_var_map[name].items():
        with Group(uri, attr=attr_name) as group:
            nonempty_domain = group.array.nonempty_domain()
            result = group.array.multi_index[nonempty_domain]
        assert np.array_equal(result[attr_name], test_case.variable_data[var_name])


@pytest.mark.parametrize("netcdf4_test_case", [simple_coord_1], indirect=True)
def test_converter_from_netcdf_2(netcdf4_test_case, tmpdir):
    name, filename, test_case = netcdf4_test_case
    converter = NetCDF4ConverterEngine.from_file(filename)
    uri = str(tmpdir.mkdir("output").join(name))
    assert isinstance(repr(converter), str)
    converter.create(uri)
    converter.copy(uri)
    for attr_name, var_name in attr_to_var_map[name].items():
        with Group(uri, attr=attr_name) as group:
            nonempty_domain = group.array.nonempty_domain()
            result = group.array.multi_index[nonempty_domain]
        assert np.array_equal(result[attr_name], test_case.variable_data[var_name])


def test_group_metadata(tmpdir):
    filepath = str(tmpdir.mkdir("data").join("test_group_metadata.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dataset.setncattr("name", "Group metadata example")
        dataset.setncattr("array", [0.0, 1.0, 2.0])
    uri = str(tmpdir.mkdir("output").join("test_group_metadata"))
    from_netcdf(filepath, uri)
    with Group(uri) as group:
        assert group.meta["name"] == "Group metadata example"
        assert group.meta["array"] == (0.0, 1.0, 2.0)


def test_variable_metadata(tmpdir):
    filepath = str(tmpdir.mkdir("data").join("test_variable_metadata.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dataset.createDimension("row", 4)
        variable = dataset.createVariable("x1", np.float64, ("row",))
        variable[:] = np.array([1.0, 2.0, 3.0, 4.0])
        variable.setncattr("fullname", "Example variable")
        variable.setncattr("array", [1, 2])
        variable.setncattr("singleton", [1.0])
    uri = str(tmpdir.mkdir("output").join("test_variable_metadata"))
    from_netcdf(filepath, uri)
    with Group(uri, attr="x1") as group:
        attr_meta = group.attr_metadata
        assert attr_meta is not None
        assert attr_meta["fullname"] == "Example variable"
        assert attr_meta["array"] == (1, 2)
        assert attr_meta["singleton"] == 1.0


def test_nested_groups(tmpdir, group1_netcdf_file):
    root_uri = str(tmpdir.mkdir("output").join("test_example_group1"))
    from_netcdf(group1_netcdf_file, root_uri)
    x = np.linspace(-1.0, 1.0, 8)
    y = np.linspace(-1.0, 1.0, 4)
    # Test root
    with Group(root_uri, attr="x1") as group:
        x1 = group.array[:]
    assert np.array_equal(x1, x)
    # Test group 1
    with Group(root_uri + "/group1", attr="x2") as group:
        x2 = group.array[:]
    assert np.array_equal(x2, 2.0 * x)
    # Test group 2
    with Group(root_uri + "/group1/group2", attr="y1") as group:
        y1 = group.array[:]
    assert np.array_equal(y1, y)
    # Test group 3
    with tiledb.open(root_uri + "/group3/array0") as array:
        array0 = array[:, :]
        A1 = array0["A1"]
        A2 = array0["A2"]
        A3 = array0["A3"]
    assert np.array_equal(A1, np.outer(y, y))
    assert np.array_equal(A2, np.zeros((4, 4), dtype=np.float64))
    assert np.array_equal(A3, np.identity(4, dtype=np.int32))


def test_not_implemented(empty_netcdf_file):
    converter = NetCDF4ConverterEngine.from_file(empty_netcdf_file)
    with pytest.raises(NotImplementedError):
        converter.add_array("A1", [])
    with pytest.raises(NotImplementedError):
        converter.add_attr("a1", "A1", np.float64)


def test_rename_array(simple1_netcdf_file):
    converter = NetCDF4ConverterEngine.from_file(simple1_netcdf_file)
    converter.rename_array("array0", "A1")
    assert set(converter.array_names) == set(["A1"])


def test_rename_attr(simple1_netcdf_file):
    converter = NetCDF4ConverterEngine.from_file(simple1_netcdf_file)
    print(converter.attr_names)
    converter.rename_attr("x1", "y1")
    assert set(converter.attr_names) == set(["y1"])


def test_rename_dim(simple1_netcdf_file):
    converter = NetCDF4ConverterEngine.from_file(simple1_netcdf_file)
    converter.rename_dim("row", "col")
    assert set(converter.dim_names) == set(["col"])


def test_copy_no_var_error(tmpdir, simple1_netcdf_file, simple2_netcdf_file):
    converter = NetCDF4ConverterEngine.from_file(simple2_netcdf_file)
    uri = str(tmpdir.mkdir("output").join("test_copy_error"))
    converter.create(uri)
    with pytest.raises(KeyError):
        converter.copy(uri, input_file=simple1_netcdf_file)
