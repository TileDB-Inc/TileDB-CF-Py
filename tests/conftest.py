# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

from . import NetCDF4TestCase

netCDF4 = pytest.importorskip("netCDF4")


@pytest.fixture(scope="session")
def empty_netcdf_file(tmpdir_factory):
    filepath = str(tmpdir_factory.mktemp("sample_netcdf").join("empty.nc"))
    with netCDF4.Dataset(filepath, mode="w"):
        pass
    return filepath


@pytest.fixture(scope="session")
def simple1_netcdf_file(tmpdir_factory):
    filepath = str(tmpdir_factory.mktemp("sample_netcdf").join("simple1.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dataset.createDimension("row", 8)
        x1 = dataset.createVariable("x1", np.float64, ("row",))
        x1[:] = np.linspace(1.0, 4.0, 8)
    return filepath


@pytest.fixture(scope="session")
def simple2_netcdf_file(tmpdir_factory):
    filepath = str(tmpdir_factory.mktemp("sample_netcdf").join("simple1.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dataset.createDimension("row", 8)
        x1 = dataset.createVariable("x1", np.float64, ("row",))
        x1[:] = np.linspace(0.0, 1.0, 8)
        x2 = dataset.createVariable("x2", np.float64, ("row",))
        x2[:] = np.linspace(0.0, 1.0, 8) ** 2
    return filepath


@pytest.fixture(scope="session")
def group1_netcdf_file(tmpdir_factory):
    """Sample NetCDF file with groups

    root:
      dimensions:  row(8)
      variables: x1(row) = np.linspace(-1.0, 1.0, 8)
      group1:
        variables: x2(row) = 2 * np.linspace(-1.0, 1.0, 8)
        group2:
          dimensions: col(4)
          variables: y1(col) = np.linspace(-1.0, 1.0, 4)
      group3:
          dimensions: row(4), col(4)
          variables:
            A1[:, :] = np.outer(y1, y1)
            A2[:, :] = np.zeros((4,4), dtype=np.float64)
            A3[:, :] = np.identity(4)
    """
    filepath = str(tmpdir_factory.mktemp("sample_netcdf").join("simple1.nc"))
    x = np.linspace(-1.0, 1.0, 8)
    y = np.linspace(-1.0, 1.0, 4)
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dataset.createDimension("row", 8)
        x1 = dataset.createVariable("x1", np.float64, ("row",))
        x1[:] = x
        group1 = dataset.createGroup("group1")
        x2 = group1.createVariable("x2", np.float64, ("row",))
        x2[:] = 2.0 * x
        group2 = group1.createGroup("group2")
        group2.createDimension("col", 4)
        y1 = group2.createVariable("y1", np.float64, ("col",))
        y1[:] = y
        group3 = dataset.createGroup("group3")
        group3.createDimension("row", 4)
        group3.createDimension("col", 4)
        A1 = group3.createVariable("A1", np.float64, ("row", "col"))
        A2 = group3.createVariable("A2", np.float64, ("row", "col"))
        A3 = group3.createVariable("A3", np.int32, ("row", "col"))
        A1[:, :] = np.outer(y, y)
        A2[:, :] = np.zeros((4, 4), dtype=np.float64)
        A3[:, :] = np.identity(4)
    return filepath


@pytest.fixture(scope="function")
def netcdf4_test_case(tmpdir_factory, request):
    """Creates a NetCDF file and returns the filepath stem, filepath, and dict of
    expected attribtues.
    """
    test_case: NetCDF4TestCase = request.param
    filepath = str(tmpdir_factory.mktemp("data").join(f"{test_case.name}.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        if test_case.group_metadata:
            dataset.setncatts(test_case.group_metadata)
        for dim_args in test_case.dimension_args:
            dataset.createDimension(*dim_args)
        for var in test_case.variable_args:
            variable = dataset.createVariable(*var)
            variable[...] = test_case.variable_data[variable.name]
            if variable.name in test_case.variable_metadata:
                variable.setncatts(test_case.variable_metadata[variable.name])
    return test_case.name, filepath, test_case
