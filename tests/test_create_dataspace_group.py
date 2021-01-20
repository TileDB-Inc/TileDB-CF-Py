import numpy as np
import pytest

import tiledb
from tiledb.cf import DataspaceGroup, GroupSchema

_row = tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.uint64)
_col = tiledb.Dim(name="cols", domain=(1, 8), tile=2, dtype=np.uint64)

_attr_a = tiledb.Attr(name="a", dtype=np.uint64)
_attr_b = tiledb.Attr(name="b", dtype=np.float64)
_attr_c = tiledb.Attr(name="c", dtype=np.dtype("U"))
_array_schema_1 = tiledb.ArraySchema(
    domain=tiledb.Domain(_row, _col),
    attrs=[_attr_a, _attr_b, _attr_c],
)
_array_schema_2 = tiledb.ArraySchema(
    domain=tiledb.Domain(_row, _col), sparse=True, attrs=[_attr_a]
)


@pytest.fixture(scope="module")
def create_dataspace_group(tmpdir_factory):
    uri = str(tmpdir_factory.mktemp("create_group1"))
    array_schemas = [
        (
            "A1",
            _array_schema_1,
        ),
        ("A2", _array_schema_2),
    ]
    group_schema = GroupSchema(array_schemas)
    group_schema.set_default_metadata_schema()
    metadata_schema = group_schema.metadata_schema
    key = None
    ctx = None
    scenario = {
        "uri": uri,
        "array_schemas": array_schemas,
        "metadata_schema": metadata_schema,
        "key": key,
        "ctx": ctx,
    }
    DataspaceGroup.create(uri, group_schema, key, ctx)
    return scenario


def test_array_schemas(create_dataspace_group):
    group_uri = create_dataspace_group["uri"]
    assert tiledb.object_type(group_uri) == "group"
    for name, schema in create_dataspace_group["array_schemas"]:
        array_uri = group_uri + "/" + name
        assert tiledb.ArraySchema.load(array_uri, key=create_dataspace_group["key"])
