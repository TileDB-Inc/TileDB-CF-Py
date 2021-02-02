from typing import Any, Dict

import numpy as np
import pytest

import tiledb
from tiledb.cf import DataspaceSchema

_dim0 = tiledb.Dim(name="dim", domain=(0, 0), tile=1, dtype=np.int32)
_row = tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.uint64)
_col = tiledb.Dim(name="cols", domain=(1, 8), tile=2, dtype=np.uint64)
_attr0 = tiledb.Attr(name="attr", dtype=np.int32)
_attr_a = tiledb.Attr(name="a", dtype=np.uint64)
_attr_b = tiledb.Attr(name="b", dtype=np.float64)
_attr_c = tiledb.Attr(name="c", dtype=np.dtype("U"))
_attr_d = tiledb.Attr(name="d", dtype=np.uint64)
_attr_e = tiledb.Attr(name="e", dtype=np.uint64)
_empty_array_schema = tiledb.ArraySchema(
    domain=tiledb.Domain(_dim0),
    attrs=[_attr0],
    sparse=False,
)
_array_schema_1 = tiledb.ArraySchema(
    domain=tiledb.Domain(_row, _col),
    attrs=[_attr_b, _attr_c],
)
_array_schema_2 = tiledb.ArraySchema(
    domain=tiledb.Domain(_row, _col), sparse=True, attrs=[_attr_a]
)
_array_schema_3 = tiledb.ArraySchema(domain=tiledb.Domain(_row), attrs=[_attr_d])
_array_schema_4 = tiledb.ArraySchema(domain=tiledb.Domain(_col), attrs=[_attr_e])

_empty_group: Dict[str, Any] = {
    "array_schemas": None,
    "metadata_schema": None,
    "attribute_map": {},
    "num_schemas": 0,
}
_single_array_group: Dict[str, Any] = {
    "array_schemas": {"A1": _array_schema_1},
    "metadata_schema": _empty_array_schema,
    "attribute_map": {"b": ("A1",), "c": ("A1",)},
    "num_schemas": 1,
}
_multi_array_group: Dict[str, Any] = {
    "array_schemas": {
        "A1": _array_schema_1,
        "A2": _array_schema_2,
        "A3": _array_schema_3,
        "A4": _array_schema_4,
    },
    "metadata_schema": None,
    "attribute_map": {
        "a": ("A2",),
        "b": ("A1",),
        "c": ("A1",),
        "d": ("A3",),
        "e": ("A4",),
    },
    "num_schemas": 4,
}


class TestDataspaceSchema:

    _scenarios = [_empty_group, _single_array_group, _multi_array_group]

    @pytest.mark.parametrize("scenario", _scenarios)
    def test_initialize_dataspace_schema(self, scenario):
        array_schemas = scenario["array_schemas"]
        metadata_schema = scenario["metadata_schema"]
        attribute_map = scenario["attribute_map"]
        dataspace_schema = DataspaceSchema(array_schemas, metadata_schema)
        dataspace_schema.check()
        assert dataspace_schema == dataspace_schema
        assert dataspace_schema.metadata_schema == metadata_schema
        assert repr(dataspace_schema) is not None
        assert len(dataspace_schema) == scenario["num_schemas"]
        for attr_name, arrays in attribute_map.items():
            result = dataspace_schema.get_attribute_arrays(attr_name)
            assert result == list(
                arrays
            ), f"Get all arrays for attribute '{attr_name}' failed."

    def test_not_equal(self):
        schema1 = DataspaceSchema({"A1": _array_schema_1})
        schema2 = DataspaceSchema({"A1": _array_schema_1}, _empty_array_schema)
        schema3 = DataspaceSchema({"A2": _array_schema_1})
        schema4 = DataspaceSchema({"A1": _array_schema_1, "A2": _array_schema_2})
        assert schema1 != schema2
        assert schema2 != schema1
        assert schema1 != schema3
        assert schema3 != schema1
        assert schema1 != schema4
        assert schema4 != schema1
        assert schema1 != "not a group schema"

    def test_set_metadata_array(self):
        """Test setting default metadata schema."""
        dataspace_schema = DataspaceSchema()
        assert dataspace_schema.metadata_schema is None
        dataspace_schema.set_default_metadata_schema()
        dataspace_schema.metadata_schema.check()
        dataspace_schema.set_default_metadata_schema()

    def test_no_attr(self):
        """Test a KeyError is raised when querying for an attribute that isn't in
        schema"""
        dataspace_schema = DataspaceSchema({"dense": _array_schema_1})
        assert len(dataspace_schema.get_attribute_arrays("missing")) == 0
