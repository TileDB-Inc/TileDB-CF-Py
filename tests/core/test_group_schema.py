# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
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
_attr_c = tiledb.Attr(name="c", dtype=np.bytes_)
_attr_d = tiledb.Attr(name="d", dtype=np.uint64)
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

_single_array_group: Dict[str, Any] = {
    "array_schemas": {"A1": _array_schema_1},
    "attr_map": {"a": ("A1",), "b": ("A1",), "c": ("A1",)},
    "num_schemas": 1,
}
_multi_array_group: Dict[str, Any] = {
    "array_schemas": {
        "A1": _array_schema_1,
        "A2": _array_schema_2,
        "A3": _array_schema_3,
        "A4": _array_schema_4,
    },
    "attr_map": {
        "a": ("A1", "A2"),
        "b": ("A1", "A3", "A4"),
        "c": ("A1",),
        "d": ("A4",),
    },
    "num_schemas": 4,
}


class TestGroupSchema:
    _scenarios = [_single_array_group, _multi_array_group]

    @pytest.mark.parametrize("scenario", _scenarios)
    def test_initialize_group_schema(self, scenario):
        array_schemas = scenario["array_schemas"]
        attr_map = scenario["attr_map"]
        group_schema = GroupSchema(array_schemas)
        group_schema.check()
        assert group_schema == group_schema
        assert repr(group_schema) is not None
        assert len(group_schema) == scenario["num_schemas"]
        for attr_name, arrays in attr_map.items():
            result = group_schema.arrays_with_attr(attr_name)
            assert result == list(
                arrays
            ), f"Get all arrays for attribute '{attr_name}' failed."
            assert group_schema.has_attr(attr_name)

    @pytest.mark.parametrize("scenario", _scenarios)
    def test_repr_html(self, scenario):
        try:
            tidylib = pytest.importorskip("tidylib")
            group_schema = GroupSchema(
                array_schemas=scenario["array_schemas"],
            )
            html_summary = group_schema._repr_html_()
            _, errors = tidylib.tidy_fragment(html_summary)
        except OSError:
            pytest.skip("unable to import libtidy backend")
        assert not bool(errors), str(errors)

    def test_not_equal(self):
        schema1 = GroupSchema({"A1": _array_schema_1})
        schema2 = GroupSchema({"A2": _array_schema_1})
        schema3 = GroupSchema({"A1": _array_schema_1, "A2": _array_schema_2})
        assert schema1 != schema2
        assert schema2 != schema1
        assert schema1 != schema3
        assert schema3 != schema1
        assert schema1 != "not a group schema"


class TestLoadEmptyGroup:
    @pytest.fixture(scope="class")
    def create_group(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("empty_group"))
        tiledb.group_create(uri)
        return uri

    def test_group_schema(self, create_group):
        uri = create_group
        schema = GroupSchema.load(uri, key=None, ctx=None)
        assert len(schema) == 0


class TestLoadGroup:
    _array_schemas = {
        "A1": _array_schema_1,
        "A2": _array_schema_2,
        "A3": _array_schema_3,
        "A4": _array_schema_4,
    }

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("simple_group"))
        tiledb.group_create(uri)
        tiledb.Array.create(uri + "/A1", _array_schema_1)
        tiledb.Array.create(uri + "/A2", _array_schema_2)
        tiledb.Array.create(uri + "/A3", _array_schema_3)
        tiledb.Array.create(uri + "/A4", _array_schema_4)
        with tiledb.Group(uri, mode="w") as group:
            group.add(uri="A1", name="A1", relative=True)
            group.add(uri="A2", name="A2", relative=True)
            group.add(uri="A3", name="A3", relative=True)
            group.add(uri="A4", name="A4", relative=True)
        return uri

    def test_group_schema(self, group_uri):
        schema = GroupSchema.load(group_uri, key=None, ctx=None)
        assert schema == GroupSchema(self._array_schemas)

    def test_not_group_exception(self, group_uri):
        with pytest.raises(ValueError):
            GroupSchema.load(group_uri + "/A1")
