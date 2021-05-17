# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf import Group, from_netcdf, from_netcdf_group
from tiledb.cf.engines.netcdf4_engine import NetCDF4ConverterEngine

simple_coord_1 = {
    "name": "simple_coord_1",
    "dimension_args": [("row", 4), ("col", 4)],
    "variable_kwargs": [
        {
            "varname": "data",
            "datatype": np.dtype("uint16"),
            "dimensions": ("row", "col"),
        },
        {"varname": "x", "datatype": np.dtype("uint16"), "dimensions": ("row",)},
        {"varname": "y", "datatype": np.dtype("uint16"), "dimensions": ("col",)},
        {"varname": "row", "datatype": np.dtype("float64"), "dimensions": ("row",)},
    ],
    "variable_data": {
        "data": np.array(
            ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16])
        ),
        "x": np.array([1, 2, 3, 4]),
        "y": np.array([5, 6, 7, 8]),
        "row": np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float64),
    },
}

simple_unlim_dim = {
    "name": "simple_unlim_dim",
    "dimension_args": (("row", None), ("col", 4)),
    "variable_kwargs": [
        {
            "varname": "data",
            "datatype": np.dtype("uint16"),
            "dimensions": ("row", "col"),
        },
        {"varname": "x", "datatype": np.dtype("uint16"), "dimensions": ("row",)},
        {"varname": "y", "datatype": np.dtype("uint16"), "dimensions": ("col",)},
    ],
    "variable_data": {
        "data": np.array(
            ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16])
        ),
        "x": np.array([1, 2, 3, 4]),
        "y": np.array([5, 6, 7, 8]),
    },
}

scalar_variables = {
    "name": "scalar_variables",
    "dimension_args": tuple(),
    "variable_kwargs": [
        {"varname": "x", "datatype": np.dtype("int32")},
        {"varname": "y", "datatype": np.dtype("int32")},
    ],
    "variable_data": {
        "x": np.array([1]),
        "y": np.array([5]),
    },
}

matching_chunks = {
    "name": "matching_chunks",
    "dimension_args": [("row", 8), ("col", 8)],
    "variable_kwargs": [
        {
            "varname": "x1",
            "datatype": np.int32,
            "dimensions": ("row", "col"),
            "chunksizes": (4, 4),
        },
        {
            "varname": "x2",
            "datatype": np.int32,
            "dimensions": ("row", "col"),
            "chunksizes": (4, 4),
        },
    ],
    "variable_data": {
        "x1": np.arange(64).reshape(8, 8),
        "x2": np.arange(64, 128).reshape(8, 8),
    },
}

mismatching_chunks = {
    "name": "mismatching_chunks",
    "dimension_args": [("row", 8), ("col", 8)],
    "variable_kwargs": [
        {
            "varname": "x1",
            "datatype": np.int32,
            "dimensions": ("row", "col"),
            "chunksizes": (4, 4),
        },
        {
            "varname": "x2",
            "datatype": np.int32,
            "dimensions": ("row", "col"),
            "chunksizes": (2, 2),
        },
    ],
    "variable_data": {
        "x1": np.arange(64).reshape(8, 8),
        "x2": np.arange(64, 128).reshape(8, 8),
    },
}

single_chunk_variable = {
    "name": "single_chunk_variable",
    "dimension_args": [("row", 8), ("col", 8)],
    "variable_kwargs": [
        {
            "varname": "x1",
            "datatype": np.int32,
            "dimensions": ("row", "col"),
            "chunksizes": (4, 4),
        },
        {
            "varname": "x2",
            "datatype": np.int32,
            "dimensions": ("row", "col"),
        },
    ],
    "variable_data": {
        "x1": np.arange(64).reshape(8, 8),
        "x2": np.arange(64, 128).reshape(8, 8),
    },
}


examples = [simple_coord_1, simple_unlim_dim, scalar_variables, matching_chunks]

attr_to_var_map = {
    "simple_coord_1": {"data": "data", "x": "x", "y": "y", "row.data": "row"},
    "simple_unlim_dim": {"data": "data", "x": "x", "y": "y"},
    "scalar_variables": {"x": "x", "y": "y"},
    "matching_chunks": {"x1": "x1", "x2": "x2"},
    "mismatching_chunks": {"x1": "x1", "x2": "x2"},
}


@pytest.mark.parametrize("netcdf_test_case", examples, indirect=True)
def test_from_netcdf(netcdf_test_case, tmpdir):
    """Integration test for `from_netcdf_file` function call."""
    name = netcdf_test_case.name
    uri = str(tmpdir.mkdir("output").join(name))
    from_netcdf(netcdf_test_case.filepath, uri)
    for attr_name, var_name in attr_to_var_map[name].items():
        with Group(uri, attr=attr_name) as group:
            nonempty_domain = group.array.nonempty_domain()
            result = group.array.multi_index[nonempty_domain]
        assert np.array_equal(
            result[attr_name], netcdf_test_case.variable_data[var_name]
        ), f"unexpected values for attribute '{attr_name}'"


@pytest.mark.parametrize("netcdf_test_case", examples, indirect=True)
def test_from_netcdf_group(netcdf_test_case, tmpdir):
    """Integration test for `from_netcdf_group` function call."""
    netCDF4 = pytest.importorskip("netCDF4")
    name = netcdf_test_case.name
    uri = str(tmpdir.mkdir("output").join(name))
    with netCDF4.Dataset(netcdf_test_case.filepath) as dataset:
        from_netcdf_group(dataset, uri)
    for attr_name, var_name in attr_to_var_map[name].items():
        with Group(uri, attr=attr_name) as group:
            nonempty_domain = group.array.nonempty_domain()
            result = group.array.multi_index[nonempty_domain]
        assert np.array_equal(
            result[attr_name], netcdf_test_case.variable_data[var_name]
        ), f"unexpected values for attribute '{attr_name}'"


@pytest.mark.parametrize("netcdf_test_case", examples, indirect=True)
def test_from_netcdf_group2(netcdf_test_case, tmpdir):
    """Integration test for `from_netcdf_group` function call."""
    name = netcdf_test_case.name
    uri = str(tmpdir.mkdir("output").join(name))
    from_netcdf_group(str(netcdf_test_case.filepath), uri)
    for attr_name, var_name in attr_to_var_map[name].items():
        with Group(uri, attr=attr_name) as group:
            nonempty_domain = group.array.nonempty_domain()
            result = group.array.multi_index[nonempty_domain]
        assert np.array_equal(
            result[attr_name], netcdf_test_case.variable_data[var_name]
        ), f"unexpected values for attribute '{attr_name}'"


@pytest.mark.parametrize("netcdf_test_case", examples, indirect=True)
def test_converter_from_netcdf(netcdf_test_case, tmpdir):
    name = netcdf_test_case.name
    converter = NetCDF4ConverterEngine.from_file(netcdf_test_case.filepath)
    uri = str(tmpdir.mkdir("output").join(name))
    assert isinstance(repr(converter), str)
    converter.convert(uri)
    for attr_name, var_name in attr_to_var_map[name].items():
        with Group(uri, attr=attr_name) as group:
            nonempty_domain = group.array.nonempty_domain()
            result = group.array.multi_index[nonempty_domain]
        assert np.array_equal(
            result[attr_name], netcdf_test_case.variable_data[var_name]
        )


@pytest.mark.parametrize("netcdf_test_case", examples, indirect=True)
def test_converter_from_netcdf_2(netcdf_test_case, tmpdir):
    name = netcdf_test_case.name
    converter = NetCDF4ConverterEngine.from_file(netcdf_test_case.filepath)
    uri = str(tmpdir.mkdir("output").join(name))
    assert isinstance(repr(converter), str)
    converter.create(uri)
    converter.copy(uri)
    for attr_name, var_name in attr_to_var_map[name].items():
        with Group(uri, attr=attr_name) as group:
            nonempty_domain = group.array.nonempty_domain()
            result = group.array.multi_index[nonempty_domain]
        assert np.array_equal(
            result[attr_name], netcdf_test_case.variable_data[var_name]
        )


def test_group_metadata(tmpdir):
    netCDF4 = pytest.importorskip("netCDF4")
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
    netCDF4 = pytest.importorskip("netCDF4")
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


def test_variable_fill(tmpdir):
    """Test converting a NetCDF variable will the _FillValue NetCDF attribute set."""
    netCDF4 = pytest.importorskip("netCDF4")
    filepath = str(tmpdir.mkdir("sample_netcdf").join("test_fill.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dataset.createDimension("row", 4)
        dataset.createVariable("x1", np.dtype("int64"), ("row",), fill_value=-1)
        converter = NetCDF4ConverterEngine.from_group(dataset)
        group_schema = converter.to_schema()
        array_schema = group_schema[group_schema.get_attr_array("x1")]
        x1_attr = array_schema.attr("x1")
        assert x1_attr.fill == -1


@pytest.mark.parametrize("netcdf_test_case", [matching_chunks], indirect=True)
def test_tile_from_matching_chunks(netcdf_test_case):
    converter = NetCDF4ConverterEngine.from_file(netcdf_test_case.filepath)
    group_schema = converter.to_schema()
    print(f"GROUP: {group_schema}")
    tiles = tuple(dim.tile for dim in group_schema["array0"].domain)
    assert tiles == (4, 4)


@pytest.mark.parametrize("netcdf_test_case", [mismatching_chunks], indirect=True)
def test_tile_from_mismatching_chunks(netcdf_test_case):
    converter = NetCDF4ConverterEngine.from_file(netcdf_test_case.filepath)
    group_schema = converter.to_schema()
    tiles = tuple(dim.tile for dim in group_schema["array0"].domain)
    assert tiles == (8, 8)


@pytest.mark.parametrize("netcdf_test_case", [single_chunk_variable], indirect=True)
def test_tile_from_single_variable_chunks(netcdf_test_case):
    converter = NetCDF4ConverterEngine.from_file(netcdf_test_case.filepath)
    group_schema = converter.to_schema()
    print(f"GROUP: {group_schema}")
    tiles = tuple(dim.tile for dim in group_schema["array0"].domain)
    assert tiles == (4, 4)


@pytest.mark.parametrize("netcdf_test_case", [matching_chunks], indirect=True)
def test_overwrite_autotile(netcdf_test_case):
    converter = NetCDF4ConverterEngine.from_file(
        netcdf_test_case.filepath, tiles={("row", "col"): (2, 4)}
    )
    group_schema = converter.to_schema()
    print(f"GROUP: {group_schema}")
    tiles = tuple(dim.tile for dim in group_schema["array0"].domain)
    assert tiles == (2, 4)


def test_rename_array(simple1_netcdf_file):
    converter = NetCDF4ConverterEngine.from_file(simple1_netcdf_file.filepath)
    converter.rename_array("array0", "A1")
    assert set(converter.array_names) == set(["A1"])


def test_rename_attr(simple1_netcdf_file):
    converter = NetCDF4ConverterEngine.from_file(simple1_netcdf_file.filepath)
    print(converter.attr_names)
    converter.rename_attr("x1", "y1")
    assert set(converter.attr_names) == set(["y1"])


def test_rename_dim(simple1_netcdf_file):
    converter = NetCDF4ConverterEngine.from_file(simple1_netcdf_file.filepath)
    converter.rename_dim("row", "col")
    assert set(converter.dim_names) == set(["col"])


def test_not_implemented_error(empty_netcdf_file):
    converter = NetCDF4ConverterEngine.from_file(empty_netcdf_file.filepath)
    converter.add_array("A1", [])
    with pytest.raises(NotImplementedError):
        converter.add_attr("a1", "A1", np.float64)


def test_copy_no_var_error(tmpdir, simple1_netcdf_file, simple2_netcdf_file):
    converter = NetCDF4ConverterEngine.from_file(simple2_netcdf_file.filepath)
    uri = str(tmpdir.mkdir("output").join("test_copy_error"))
    converter.create(uri)
    with pytest.raises(KeyError):
        converter.copy(uri, input_file=simple1_netcdf_file.filepath)
