from typing import Any, Dict

import numpy as np
import pytest

import tiledb
from tiledb.cf import DataspaceGroupSchema

_dim0 = tiledb.Dim(name="dim", domain=(0, 0), tile=1, dtype=np.int32)
_row = tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.uint64)
_col = tiledb.Dim(name="cols", domain=(1, 8), tile=2, dtype=np.uint64)
_attr0 = tiledb.Attr(name="attr", dtype=np.int32)
_attr_a = tiledb.Attr(name="a", dtype=np.uint64)
_attr_b = tiledb.Attr(name="b", dtype=np.float64)
_attr_c = tiledb.Attr(name="c", dtype=np.dtype("U"))
_attr_d = tiledb.Attr(name="d", dtype=np.uint64)
_attr_e = tiledb.Attr(name="e", dtype=np.float64)
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
    domain=tiledb.Domain(_row, _col), sparse=True, attrs=[_attr_d]
)
_array_schema_3 = tiledb.ArraySchema(domain=tiledb.Domain(_row), attrs=[_attr_e])


_empty_group: Dict[str, Any] = {
    "array_schemas": None,
    "metadata_schema": None,
    "attribute_map": {},
    "num_schemas": 0,
}
_single_array_group: Dict[str, Any] = {
    "array_schemas": {"A1": _array_schema_1},
    "metadata_schema": _empty_array_schema,
    "attribute_map": {"a": ("A1",), "b": ("A1",), "c": ("A1",)},
    "num_schemas": 1,
}
_multi_array_group: Dict[str, Any] = {
    "array_schemas": {
        "A1": _array_schema_1,
        "A2": _array_schema_2,
        "A3": _array_schema_3,
    },
    "metadata_schema": None,
    "attribute_map": {
        "a": ("A1",),
        "b": ("A1",),
        "c": ("A1",),
        "d": ("A2",),
        "e": ("A3",),
    },
    "num_schemas": 3,
}


class TestDataspaceGroupSchema:

    _scenarios = [_empty_group, _single_array_group, _multi_array_group]

    @pytest.mark.parametrize("scenario", _scenarios)
    def test_initialize_group_schema(self, scenario):
        array_schemas = scenario["array_schemas"]
        metadata_schema = scenario["metadata_schema"]
        attribute_map = scenario["attribute_map"]
        group_schema = DataspaceGroupSchema(array_schemas, metadata_schema)
        group_schema.check()
        assert group_schema == group_schema
        assert group_schema.metadata_schema == metadata_schema
        assert repr(group_schema) is not None
        assert len(group_schema) == scenario["num_schemas"]
        for attr_name, arrays in attribute_map.items():
            result = group_schema.get_attr_arrays(attr_name)
            assert result == list(
                arrays
            ), f"Get all arrays for attribute '{attr_name}' failed."

    def test_set_metadata_array(self):
        """Test setting default metadata schema."""
        group_schema = DataspaceGroupSchema()
        assert group_schema.metadata_schema is None
        group_schema.set_default_metadata_schema()
        group_schema.metadata_schema.check()
        group_schema.set_default_metadata_schema()

    def test_no_attr(self):
        """Test a KeyError is raised when querying for an attribute that isn't in
        schema"""
        group_schema = DataspaceGroupSchema({"dense": _array_schema_1})
        assert len(group_schema.get_attr_arrays("missing")) == 0


class TestLoadEmptyGroup:
    @pytest.fixture(scope="class")
    def create_group(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("empty_group"))
        tiledb.group_create(uri)
        return uri

    def test_group_schema(self, create_group):
        uri = create_group
        schema = DataspaceGroupSchema.load_group(uri, key=None, ctx=None)
        assert schema.metadata_schema is None
        assert len(schema) == 0


class TestLoadGroup:

    _array_schemas = {
        "A1": _array_schema_1,
        "A2": _array_schema_2,
        "A3": _array_schema_3,
    }
    _metadata_array = _empty_array_schema

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("simple_group"))
        tiledb.group_create(uri)
        tiledb.Array.create(uri + "/A1", _array_schema_1)
        tiledb.Array.create(uri + "/A2", _array_schema_2)
        tiledb.Array.create(uri + "/A3", _array_schema_3)
        tiledb.Array.create(uri + "/__tiledb_group", _empty_array_schema)
        return uri

    def test_group_schema(self, group_uri):
        schema = DataspaceGroupSchema.load_group(group_uri, key=None, ctx=None)
        assert schema == DataspaceGroupSchema(self._array_schemas, self._metadata_array)


class TestNotDataspace:

    _array_schemas = {
        "A1": _array_schema_1,
        "A2": _array_schema_1,
    }

    def test_not_dataspace(self):
        with pytest.raises(RuntimeError):
            _ = DataspaceGroupSchema(self._array_schemas)
