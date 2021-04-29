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
        dataset.createVariable("x1", np.float64, ("row",))
    return filepath


@pytest.fixture(scope="session")
def simple2_netcdf_file(tmpdir_factory):
    filepath = str(tmpdir_factory.mktemp("sample_netcdf").join("simple1.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dataset.createDimension("row", 8)
        dataset.createVariable("x1", np.float64, ("row",))
        dataset.createVariable("x2", np.float64, ("row",))
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
