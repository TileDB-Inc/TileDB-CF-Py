# Copyright 2020 TileDB Inc.
# Licensed under the MIT License.
"""Test for tiledb_type module"""

import numpy as np
import pytest

import tiledb
from tiledb.cf.io import DataType, SharedDimension


class TestSharedDimension:

    _tiledb_dim = [
        tiledb.Dim(name="dim", domain=(1, 4), tile=4, dtype=np.int32),
    ]

    _bad_name_data = [
        ("bad.name", (1, 4), np.int32),
        (".badname", (1, 4), np.int32),
    ]

    _compare_data = [
        (
            tiledb.Dim(name="dim", domain=(1, 4), tile=4, dtype=np.int32),
            tiledb.Dim(name="dim", domain=(1, 4), tile=2, dtype=np.int32),
            True,
        ),
        (
            tiledb.Dim(name="dim", domain=(1, 4), tile=4, dtype=np.int32),
            tiledb.Dim(name="dim0", domain=(1, 4), tile=2, dtype=np.int32),
            False,
        ),
        (
            tiledb.Dim(name="dim", domain=(1, 4), tile=4, dtype=np.int32),
            tiledb.Dim(name="dim", domain=(1, 2), tile=2, dtype=np.int32),
            False,
        ),
        (
            tiledb.Dim(name="dim", domain=(1, 4), tile=4, dtype=np.int32),
            tiledb.Dim(name="dim", domain=(1, 4), tile=2, dtype=np.int64),
            False,
        ),
    ]

    @pytest.mark.parametrize("dimension", _tiledb_dim)
    def test_create(self, dimension):
        shared_dimension = SharedDimension.create(dimension)
        assert shared_dimension.name == dimension.name
        assert shared_dimension.domain == dimension.domain
        assert shared_dimension.dtype == dimension.dtype
        assert repr(shared_dimension) is not None

    @pytest.mark.parametrize("name, domain, data_type", _bad_name_data)
    def test_bad_name(self, name, domain, data_type):
        with pytest.raises(ValueError):
            SharedDimension(name, domain, data_type)

    @pytest.mark.parametrize("dimension1, dimension2, is_equal", _compare_data)
    def test_compare_dimensions(self, dimension1, dimension2, is_equal):
        dim1 = SharedDimension.create(dimension1)
        dim2 = SharedDimension.create(dimension2)
        if is_equal:
            assert dim1 == dim2
        else:
            assert dim1 != dim2


class TestDataType:

    _data = [
        ("TILEDB_INT32", "TILEDB_INT32", np.dtype("int32")),
        ("TILEDB_UINT32", "TILEDB_UINT32", np.dtype("uint32")),
        ("TILEDB_INT64", "TILEDB_INT64", np.dtype("int64")),
        ("TILEDB_UINT64", "TILEDB_UINT64", np.dtype("uint64")),
        ("TILEDB_FLOAT32", "TILEDB_FLOAT32", np.dtype("float32")),
        ("TILEDB_FLOAT64", "TILEDB_FLOAT64", np.dtype("float64")),
        ("TILEDB_INT8", "TILEDB_INT8", np.dtype("int8")),
        ("TILEDB_UINT8", "TILEDB_UINT8", np.dtype("uint8")),
        ("TILEDB_INT16", "TILEDB_INT16", np.dtype("int16")),
        ("TILEDB_UINT16", "TILEDB_UINT16", np.dtype("uint16")),
        ("TILEDB_STRING_UTF8", "TILEDB_STRING_UTF8", np.dtype("U")),
        ("TILEDB_CHAR", "TILEDB_CHAR", np.dtype("S")),
        ("TILEDB_DATETIME_YEAR", "TILEDB_DATETIME_YEAR", np.dtype("M8[Y]")),
        ("TILEDB_DATETIME_MONTH", "TILEDB_DATETIME_MONTH", np.dtype("M8[M]")),
        ("TILEDB_DATETIME_WEEK", "TILEDB_DATETIME_WEEK", np.dtype("M8[W]")),
        ("TILEDB_DATETIME_DAY", "TILEDB_DATETIME_DAY", np.dtype("M8[D]")),
        ("TILEDB_DATETIME_HR", "TILEDB_DATETIME_HR", np.dtype("M8[h]")),
        ("TILEDB_DATETIME_MIN", "TILEDB_DATETIME_MIN", np.dtype("M8[m]")),
        ("TILEDB_DATETIME_SEC", "TILEDB_DATETIME_SEC", np.dtype("M8[s]")),
        ("TILEDB_DATETIME_MS", "TILEDB_DATETIME_MS", np.dtype("M8[ms]")),
        ("TILEDB_DATETIME_US", "TILEDB_DATETIME_US", np.dtype("M8[us]")),
        ("TILEDB_DATETIME_NS", "TILEDB_DATETIME_NS", np.dtype("M8[ns]")),
        ("TILEDB_DATETIME_PS", "TILEDB_DATETIME_PS", np.dtype("M8[ps]")),
        ("TILEDB_DATETIME_FS", "TILEDB_DATETIME_FS", np.dtype("M8[fs]")),
        ("TILEDB_DATETIME_AS", "TILEDB_DATETIME_AS", np.dtype("M8[as]")),
        (np.int32, "TILEDB_INT32", np.dtype("int32")),
        (np.uint32, "TILEDB_UINT32", np.dtype("uint32")),
        (np.int64, "TILEDB_INT64", np.dtype("int64")),
        (np.uint64, "TILEDB_UINT64", np.dtype("uint64")),
        (np.float32, "TILEDB_FLOAT32", np.dtype("float32")),
        (np.float64, "TILEDB_FLOAT64", np.dtype("float64")),
        (np.complex64, "TILEDB_FLOAT32", np.dtype("float32")),
        (np.complex128, "TILEDB_FLOAT64", np.dtype("float64")),
        (np.int8, "TILEDB_INT8", np.dtype("int8")),
        (np.uint8, "TILEDB_UINT8", np.dtype("uint8")),
        (np.int16, "TILEDB_INT16", np.dtype("int16")),
        (np.uint16, "TILEDB_UINT16", np.dtype("uint16")),
        (np.unicode_, "TILEDB_STRING_UTF8", np.dtype("U")),
        (np.bytes_, "TILEDB_CHAR", np.dtype("S")),
        (np.datetime64("", "Y"), "TILEDB_DATETIME_YEAR", np.dtype("M8[Y]")),
        (np.datetime64("", "M"), "TILEDB_DATETIME_MONTH", np.dtype("M8[M]")),
        (np.datetime64("", "W"), "TILEDB_DATETIME_WEEK", np.dtype("M8[W]")),
        (np.datetime64("", "D"), "TILEDB_DATETIME_DAY", np.dtype("M8[D]")),
        (np.datetime64("", "h"), "TILEDB_DATETIME_HR", np.dtype("M8[h]")),
        (np.datetime64("", "m"), "TILEDB_DATETIME_MIN", np.dtype("M8[m]")),
        (np.datetime64("", "s"), "TILEDB_DATETIME_SEC", np.dtype("M8[s]")),
        (np.datetime64("", "ms"), "TILEDB_DATETIME_MS", np.dtype("M8[ms]")),
        (np.datetime64("", "us"), "TILEDB_DATETIME_US", np.dtype("M8[us]")),
        (np.datetime64("", "ns"), "TILEDB_DATETIME_NS", np.dtype("M8[ns]")),
        (np.datetime64("", "ps"), "TILEDB_DATETIME_PS", np.dtype("M8[ps]")),
        (np.datetime64("", "fs"), "TILEDB_DATETIME_FS", np.dtype("M8[fs]")),
        (np.datetime64("", "as"), "TILEDB_DATETIME_AS", np.dtype("M8[as]")),
    ]

    @pytest.mark.parametrize("key, expected_name, expected_value", _data)
    @pytest.mark.filterwarnings("ignore:converting complex")
    def test_data_type(self, key, expected_name, expected_value):
        """Test create class method for TileDBType"""
        data_type = DataType.create(key)
        assert data_type.name == expected_name
        assert data_type.dtype == expected_value
