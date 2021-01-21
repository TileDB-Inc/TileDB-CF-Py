from typing import Any, Dict

import numpy as np
import pytest

import tiledb
from tiledb.cf import GroupSchema

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

_empty_group: Dict[str, Any] = {
    "array_schemas": None,
    "metadata_schema": None,
    "attribute_map": {},
    "num_schemas": 0,
}
_single_array_group: Dict[str, Any] = {
    "array_schemas": [("A1", _array_schema_1)],
    "metadata_schema": _empty_array_schema,
    "attribute_map": {"a": ("A1",), "b": ("A1",), "c": ("A1",)},
    "num_schemas": 1,
}
_multi_array_group: Dict[str, Any] = {
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
}


class TestGroupSchema:

    _scenarios = [_empty_group, _single_array_group, _multi_array_group]

    @pytest.mark.parametrize("scenario", _scenarios)
    def test_initialize_group_schema(self, scenario):
        array_schemas = scenario["array_schemas"]
        metadata_schema = scenario["metadata_schema"]
        attribute_map = scenario["attribute_map"]
        group_schema = GroupSchema(array_schemas, metadata_schema)
        group_schema.check()
        assert group_schema == group_schema
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

    def test_not_equal(self):
        schema1 = GroupSchema([("A1", _array_schema_1)])
        schema2 = GroupSchema([("A1", _array_schema_1)], _empty_array_schema)
        schema3 = GroupSchema([("A2", _array_schema_1)])
        schema4 = GroupSchema([("A1", _array_schema_1), ("A2", _array_schema_2)])
        assert schema1 != schema2
        assert schema1 != schema3
        assert schema1 != schema4
        assert schema1 != "not a group schema"

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
            ("dense", _array_schema_1),
            ("dense", _array_schema_2),
        ]
        with pytest.raises(ValueError):
            GroupSchema(array_schemas)

    def test_dim_match_error(self):
        """Test ValueError is raised when two schemas have a dimension that doesn't
        match."""
        array_schemas = [
            ("dense1", _array_schema_1),
            (
                "dense2",
                tiledb.ArraySchema(
                    domain=tiledb.Domain(
                        tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.int64),
                    ),
                    sparse=False,
                    attrs=[_attr_a],
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
                ("dense", _array_schema_1),
            ]
        )
        with pytest.raises(KeyError):
            group_schema.get_attribute_array("missing")

    def test_multi_attr_array_error(self):
        """Test a ValueError is raised when calling `get_attribute_array` for an
        attribute that exists in multiple array schemas."""
        group_schema = GroupSchema(
            [
                ("dense", _array_schema_1),
                ("sparse", _array_schema_2),
            ]
        )
        with pytest.raises(ValueError):
            group_schema.get_attribute_array("a")


class TestLoadEmptyGroup:
    @pytest.fixture(scope="class")
    def create_group(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("empty_group"))
        tiledb.group_create(uri)
        return uri

    def test_group_schema(self, create_group):
        uri = create_group
        schema = GroupSchema.load(uri, key=None, ctx=None)
        assert schema.metadata_schema is None
        assert len(schema) == 0


class TestLoadGroup:

    _array_schemas = [
        ("A1", _array_schema_1),
        ("A2", _array_schema_2),
        ("A3", _array_schema_3),
        ("A4", _array_schema_4),
    ]
    _metadata_array = _empty_array_schema

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("simple_group"))
        tiledb.group_create(uri)
        tiledb.Array.create(uri + "/A1", _array_schema_1)
        tiledb.Array.create(uri + "/A2", _array_schema_2)
        tiledb.Array.create(uri + "/A3", _array_schema_3)
        tiledb.Array.create(uri + "/A4", _array_schema_4)
        tiledb.Array.create(uri + "/__tiledb_group", _empty_array_schema)
        return uri

    def test_group_schema(self, group_uri):
        schema = GroupSchema.load(group_uri, key=None, ctx=None)
        assert schema == GroupSchema(self._array_schemas, self._metadata_array)

    def test_not_group_exception(self, group_uri):
        with pytest.raises(ValueError):
            GroupSchema.load(group_uri + "/A1")
