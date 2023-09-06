from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytest


@dataclass(frozen=True)
class NetCDFSingleGroupExample:
    """Dataclass the holds values required to generate NetCDF test cases

    name: name of the test case
    dimension_args: sequence of arguments required to create NetCDF4 dimensions
    variable_args: sequence of arguments required to create NetCDF4 variables
    variable_data: dict of variable data by variable name
    variable_matadata: dict of variable metadata key-value pairs by variable name
    group_metadata: group metadata key-value pairs
    """

    name: str
    directory_path: str
    dimension_args: Sequence[Tuple[str, Optional[int]]]
    variable_kwargs: Sequence[Dict[str, Any]]
    variable_data: Dict[str, np.ndarray]
    variable_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    group_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        netCDF4 = pytest.importorskip("netCDF4")
        with netCDF4.Dataset(self.filepath, mode="w") as dataset:
            if self.group_metadata:
                dataset.setncatts(self.group_metadata)
            for dim_args in self.dimension_args:
                dataset.createDimension(*dim_args)
            for var_kwargs in self.variable_kwargs:
                variable = dataset.createVariable(**var_kwargs)
                variable[...] = self.variable_data[variable.name]
                if variable.name in self.variable_metadata:
                    variable.setncatts(self.variable_metadata[variable.name])

    @property
    def filepath(self):
        return self.directory_path.join(f"{self.name}.nc")


@pytest.fixture(scope="session")
def simple1_netcdf_file(tmpdir_factory):
    directory_path = tmpdir_factory.mktemp("sample_netcdf")
    example = NetCDFSingleGroupExample(
        "simple1",
        directory_path,
        dimension_args=[
            ("row", 8),
        ],
        variable_kwargs=[
            {"varname": "x1", "datatype": np.float64, "dimensions": ("row",)},
        ],
        variable_data={"x1": np.linspace(1.0, 4.0, 8)},
    )
    return example


@pytest.fixture(scope="session")
def simple2_netcdf_file(tmpdir_factory):
    directory_path = tmpdir_factory.mktemp("sample_netcdf")
    xdata = np.linspace(0.0, 1.0, 8)
    example = NetCDFSingleGroupExample(
        "simple2",
        directory_path,
        dimension_args=[("row", 8)],
        variable_kwargs=[
            {"varname": "x1", "datatype": np.float64, "dimensions": ("row",)},
            {"varname": "x2", "datatype": np.float64, "dimensions": ("row",)},
        ],
        variable_data={"x1": xdata, "x2": xdata**2},
        group_metadata={"name": "simple2"},
    )
    return example


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
    netCDF4 = pytest.importorskip("netCDF4")
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


@pytest.fixture
def netcdf_test_case(tmpdir_factory, request):
    """Creates a NetCDF file and returns the filepath stem, filepath, and dict of
    expected attribtues.
    """
    return NetCDFSingleGroupExample(
        **request.param,
        directory_path=tmpdir_factory.mktemp("sample_netcdf"),
    )
