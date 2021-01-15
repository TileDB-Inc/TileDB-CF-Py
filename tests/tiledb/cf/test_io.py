# Copyright 2020 TileDB Inc.
# Licensed under the MIT License.
"""Test for tiledb_type module"""

import numpy as np
import pytest

import tiledb
from tiledb.cf.io import DataType, GroupSchema, SharedDimension


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


class TestGroupSchema:

    _dim0 = tiledb.Dim(name="dim", domain=(0, 0), tile=1, dtype=np.int32)
    _row = tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.uint64)
    _col = tiledb.Dim(name="cols", domain=(1, 8), tile=2, dtype=np.uint64)
    _attr0 = tiledb.Attr(name="attr", dtype=np.int32)
    _attr_a = tiledb.Attr(name="a", dtype=np.uint64)
    _attr_b = tiledb.Attr(name="b", dtype=np.float64)
    _attr_c = tiledb.Attr(name="c", dtype=np.dtype("U"))
    _attr_d = tiledb.Attr(name="d", dtype=np.uint64)
    _empty_array_schema = tiledb.ArraySchema(
        domain=tiledb.Domain(_dim0),
        attrs=[_attr0],
        sparse=False,
    )
    _array_schema_1 = tiledb.ArraySchema(
        domain=tiledb.Domain(_row, _col),
        attrs=[_attr_a, _attr_b, _attr_c],
    )
    _array_schema_2 = tiledb.ArraySchema(
        domain=tiledb.Domain(_row, _col), sparse=True, attrs=[_attr_a]
    )
    _array_schema_3 = tiledb.ArraySchema(domain=tiledb.Domain(_row), attrs=[_attr_b])
    _array_schema_4 = tiledb.ArraySchema(
        domain=tiledb.Domain(_col), attrs=[_attr_b, _attr_d]
    )

    _scenarios = [
        {
            "array_schemas": None,
            "metadata_schema": None,
            "attribute_map": {},
            "num_schemas": 0,
        },
        {
            "array_schemas": [("A1", _array_schema_1)],
            "metadata_schema": _empty_array_schema,
            "attribute_map": {"a": ("A1",), "b": ("A1",), "c": ("A1",)},
            "num_schemas": 1,
        },
        {
            "array_schemas": [
                ("A1", _array_schema_1),
                ("A2", _array_schema_2),
                ("A3", _array_schema_3),
                ("A4", _array_schema_4),
            ],
            "metadata_schema": None,
            "attribute_map": {
                "a": ("A1", "A2"),
                "b": ("A1", "A3", "A4"),
                "c": ("A1",),
                "d": ("A4",),
            },
            "num_schemas": 4,
        },
    ]

    @pytest.mark.parametrize("scenario", _scenarios)
    def test_schema(self, scenario):
        array_schemas = scenario["array_schemas"]
        metadata_schema = scenario["metadata_schema"]
        attribute_map = scenario["attribute_map"]
        group_schema = GroupSchema(array_schemas, metadata_schema)
        group_schema.check()
        assert group_schema.metadata_schema == metadata_schema
        assert repr(group_schema) is not None
        assert len(group_schema) == scenario["num_schemas"]
        for attr_name, arrays in attribute_map.items():
            result = group_schema.get_all_attribute_arrays(attr_name)
            assert (
                result == arrays
            ), f"Get all arrays for attribute '{attr_name}' failed."
            if len(result) == 1:
                assert result[0] == group_schema.get_attribute_array(attr_name)

    def test_set_metadata_array(self):
        """Test setting default metadata schema."""
        group_schema = GroupSchema()
        assert group_schema.metadata_schema is None
        group_schema.set_default_metadata_schema()
        group_schema.metadata_schema.check()
        group_schema.set_default_metadata_schema()

    def test_repeat_name_error(self):
        """Test ValueError is raised when multiple array schemas have the same name."""
        array_schemas = [
            ("dense", self._array_schema_1),
            ("dense", self._array_schema_2),
        ]
        with pytest.raises(ValueError):
            GroupSchema(array_schemas)

    def test_dim_match_error(self):
        """Test ValueError is raised when two schemas have a dimension that doesn't
        match."""
        array_schemas = [
            ("dense1", self._array_schema_1),
            (
                "dense2",
                tiledb.ArraySchema(
                    domain=tiledb.Domain(
                        tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.int64),
                    ),
                    sparse=False,
                    attrs=[self._attr_a],
                ),
            ),
        ]
        with pytest.raises(ValueError):
            GroupSchema(array_schemas)

    def test_no_attr_error(self):
        """Test a KeyError is raised when querying for an attribute that isn't in
        schema"""
        group_schema = GroupSchema(
            [
                ("dense", self._array_schema_1),
            ]
        )
        with pytest.raises(KeyError):
            group_schema.get_attribute_array("missing")

    def test_multi_attr_array_error(self):
        """Test a ValueError is raised when calling `get_attribute_array` for an
        attribute that exists in multiple array schemas."""
        group_schema = GroupSchema(
            [
                ("dense", self._array_schema_1),
                ("sparse", self._array_schema_2),
            ]
        )
        with pytest.raises(ValueError):
            group_schema.get_attribute_array("a")
