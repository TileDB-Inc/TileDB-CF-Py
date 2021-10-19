# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

from tiledb.cf.netcdf_engine._utils import get_netcdf_metadata, get_unpacked_dtype

netCDF4 = pytest.importorskip("netCDF4")


@pytest.mark.parametrize(
    "input_dtype,scale_factor,add_offset,output_dtype",
    (
        (np.int16, None, None, np.int16),
        (np.int16, np.float32(1), None, np.float32),
        (np.int16, None, np.float32(1), np.float32),
        (np.int16, np.float64(1), np.float32(1), np.float64),
    ),
)
def test_unpacked_dtype(input_dtype, scale_factor, add_offset, output_dtype):
    """Tests computing the unpacked data type for a NetCDF variable."""
    with netCDF4.Dataset("tmp.nc", diskless=True, mode="w") as dataset:
        dataset.createDimension("t", None)
        variable = dataset.createVariable("x", dimensions=("t",), datatype=input_dtype)
        if scale_factor is not None:
            variable.setncattr("scale_factor", scale_factor)
        if add_offset is not None:
            variable.setncattr("add_offset", add_offset)
        dtype = get_unpacked_dtype(variable)
    assert dtype == output_dtype


def test_unpacked_dtype_dtype_error():
    """Tests attempting to unpack a NetCDF variable with a data type that does not
    support packing/unpacking."""
    with netCDF4.Dataset("tmp.nc", diskless=True, mode="w") as dataset:
        variable = dataset.createVariable("x", dimensions=tuple(), datatype="S1")
        with pytest.raises(ValueError):
            get_unpacked_dtype(variable)


@pytest.mark.parametrize(
    "value, expected_result",
    (
        (np.float64(1), np.float64(1)),
        (np.array((1), dtype=np.float64), np.float64(1)),
        (np.array([1], dtype=np.int32), np.int32(1)),
    ),
)
def test_get_netcdf_metadata_number(value, expected_result):
    """Tests computing the unpacked data type for a NetCDF variable."""
    key = "name"
    with netCDF4.Dataset("tmp.nc", diskless=True, mode="w") as dataset:
        dataset.setncattr(key, value)
        result = get_netcdf_metadata(dataset, key, is_number=True)
        assert result == expected_result


@pytest.mark.parametrize(
    "value",
    (
        ("",),
        (
            np.array(
                (1, 2),
            )
        ),
    ),
)
def test_get_netcdf_metadata_number_with_warning(value):
    """Tests computing the unpacked data type for a NetCDF variable."""
    key = "name"
    with netCDF4.Dataset("tmp.nc", diskless=True, mode="w") as dataset:
        dataset.setncattr(key, value)
        with pytest.warns(Warning):
            result = get_netcdf_metadata(dataset, key, is_number=True)
        assert result is None
