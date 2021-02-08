from typing import Any, Dict

import numpy as np
import pytest

import tiledb
from tiledb.cf import DataspaceGroupSchema, GroupSchema

_CF_COORDINATE_SUFFIX = ".axis_data"
_pressure_dim = tiledb.Dim(name="pressure", domain=(0, 3), tile=2, dtype=np.uint64)
_temperature_dim = tiledb.Dim(name="cols", domain=(0, 7), tile=2, dtype=np.uint64)
_metadata_schema = tiledb.ArraySchema(
    domain=tiledb.Domain(tiledb.Dim(name="dim", domain=(0, 0), tile=1, dtype=np.int32)),
    attrs=[tiledb.Attr(name="attr", dtype=np.int32)],
    sparse=False,
)
_array_schema_1 = tiledb.ArraySchema(
    domain=tiledb.Domain(_pressure_dim),
    attrs=[
        tiledb.Attr(name="pressure" + _CF_COORDINATE_SUFFIX, dtype=np.float64),
        tiledb.Attr(name="b", dtype=np.float64),
        tiledb.Attr(name="c", dtype=np.uint64),
    ],
)
_array_schema_2 = tiledb.ArraySchema(
    domain=tiledb.Domain(_pressure_dim, _temperature_dim),
    sparse=True,
    attrs=[tiledb.Attr(name="d", dtype=np.uint64)],
)
_array_schema_3 = tiledb.ArraySchema(
    domain=tiledb.Domain(_temperature_dim),
    attrs=[tiledb.Attr(name="e", dtype=np.float64)],
)


_empty_group: Dict[str, Any] = {
    "group_schema": GroupSchema(),
    "attribute_map": {},
}
_single_array_group: Dict[str, Any] = {
    "group_schema": GroupSchema(
        {"A1": _array_schema_1},
        _metadata_schema,
    ),
    "attribute_map": {"pressure": "A1", "b": "A1", "c": "A1"},
}
_multi_array_group: Dict[str, Any] = {
    "group_schema": GroupSchema(
        {
            "A1": _array_schema_1,
            "A2": _array_schema_2,
            "A3": _array_schema_3,
        },
        None,
    ),
    "attribute_map": {
        "pressure": "A1",
        "b": "A1",
        "c": "A1",
        "d": "A2",
        "e": "A3",
    },
}


class TestDataspaceGroupSchema:

    _scenarios = [_empty_group, _single_array_group]  # , _multi_array_group]

    @pytest.mark.parametrize("scenario", _scenarios)
    def test_initialize_group_schema(self, scenario):
        group_schema = scenario["group_schema"]
        attribute_map = scenario["attribute_map"]
        dataspace_schema = DataspaceGroupSchema(group_schema)
        assert dataspace_schema == dataspace_schema, "group does not equal itselt"
        assert set(dataspace_schema.attr_names) == set(
            attribute_map.keys()
        ), "attributes do not match"
        print(f"Attributes: {dataspace_schema.attr_names}")
        for attr_name, array in attribute_map.items():
            assert (
                dataspace_schema.get_array_from_attr(attr_name) == array
            ), f"attribute {attr_name} not found in array {array}"


class TestLoadGroup:

    _array_schemas = {
        "A1": _array_schema_1,
        "A2": _array_schema_2,
        "A3": _array_schema_3,
    }
    _metadata_array = _metadata_schema

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("simple_group"))
        tiledb.group_create(uri)
        tiledb.Array.create(uri + "/A1", _array_schema_1)
        tiledb.Array.create(uri + "/A2", _array_schema_2)
        tiledb.Array.create(uri + "/A3", _array_schema_3)
        tiledb.Array.create(uri + "/__tiledb_group", _metadata_schema)
        return uri

    def test_group_schema(self, group_uri):
        schema = DataspaceGroupSchema.load_group(group_uri, key=None, ctx=None)
        assert schema == DataspaceGroupSchema(
            GroupSchema(self._array_schemas, self._metadata_array)
        )


class TestNotDataspace:

    _array_schemas = {
        "A1": _array_schema_1,
        "A2": _array_schema_1,
    }

    def test_not_dataspace(self):
        group_schema = GroupSchema(self._array_schemas)
        with pytest.raises(RuntimeError):
            _ = DataspaceGroupSchema(group_schema)
