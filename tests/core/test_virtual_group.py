# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf import GroupSchema, VirtualGroup

_row = tiledb.Dim(name="rows", domain=(1, 4), tile=4, dtype=np.uint64)
_col = tiledb.Dim(name="cols", domain=(1, 4), tile=4, dtype=np.uint64)


_attr_a = tiledb.Attr(name="a", dtype=np.uint64)
_attr_b = tiledb.Attr(name="b", dtype=np.float64)
_attr_c = tiledb.Attr(name="c", dtype=np.dtype("U"))
_array_schema_1 = tiledb.ArraySchema(
    domain=tiledb.Domain(_row, _col),
    attrs=[_attr_a],
)
_array_schema_2 = tiledb.ArraySchema(
    domain=tiledb.Domain(_row),
    sparse=True,
    attrs=[_attr_b, _attr_c],
)
_array_schema_3 = tiledb.ArraySchema(
    domain=tiledb.Domain(_row, _col),
    attrs=[_attr_c],
)


class TestCreateVirtualGroup:

    _metadata_schema = _array_schema_1
    _array_schemas = [
        ("A1", _array_schema_1),
        ("A2", _array_schema_2),
    ]
    _group_schema = GroupSchema(_array_schemas, _metadata_schema)

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        """Creates a TileDB Group from GroupSchema and returns scenario dict."""
        uri = str(tmpdir_factory.mktemp("group1").join("virtual"))
        ctx = None
        VirtualGroup.create(uri, self._group_schema, ctx=ctx)
        return {"__tiledb_group": uri, "A1": f"{uri}_A1", "A2": f"{uri}_A2"}

    def test_array_schemas(self, group_uri):
        assert (
            tiledb.ArraySchema.load(group_uri["__tiledb_group"])
            == self._metadata_schema
        )
        assert tiledb.ArraySchema.load(group_uri["A1"]) == _array_schema_1
        assert tiledb.ArraySchema.load(group_uri["A2"]) == _array_schema_2


class TestMetadataOnlyGroup:

    _metadata_schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.uint64)
        ),
        attrs=[tiledb.Attr(name="a", dtype=np.uint64)],
        sparse=True,
    )

    @pytest.fixture(scope="class")
    def group_uris(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("group1"))
        tiledb.Array.create(uri, self._metadata_schema)
        return {"__tiledb_group": uri}

    def test_has_metadata(self, group_uris):
        with VirtualGroup(group_uris) as group:
            assert isinstance(group, VirtualGroup)
            assert group.has_metadata_array
            assert group.meta is not None

    def test_no_such_attr_error(self, group_uris):
        with VirtualGroup(group_uris) as group:
            with pytest.raises(KeyError):
                group.open_array(attr="a")


class TestVirtualGroupWithArrays:

    _metadata_schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.uint64)
        ),
        attrs=[tiledb.Attr(name="a", dtype=np.uint64)],
        sparse=True,
    )

    _A1_data = np.array(
        ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]), dtype=np.uint64
    )

    @pytest.fixture(scope="class")
    def group_uris(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("simple_group"))
        tiledb.Array.create(uri + "/metadata", self._metadata_schema)
        tiledb.Array.create(uri + "/array1", _array_schema_1)
        with tiledb.DenseArray(uri + "/array1", mode="w") as array:
            array[:] = self._A1_data
        tiledb.Array.create(uri + "/array2", _array_schema_2)
        tiledb.Array.create(uri + "/array3", _array_schema_3)
        return {
            "__tiledb_group": f"{uri}/metadata",
            "A1": f"{uri}/array1",
            "A2": f"{uri}/array2",
            "A3": f"{uri}/array3",
        }

    def test_open_array_from_group(self, group_uris):
        with VirtualGroup(group_uris) as group:
            with group.open_array(array="A1") as array:
                assert isinstance(array, tiledb.Array)
                assert array.mode == "r"
                assert np.array_equal(array[:, :]["a"], self._A1_data)

    def test_open_attr(self, group_uris):
        with VirtualGroup(group_uris) as group:
            with group.open_array(attr="a") as array:
                assert isinstance(array, tiledb.Array)
                assert array.mode == "r"
                assert np.array_equal(array[:, :], self._A1_data)

    def test_attr_ambiguous_error(self, group_uris):
        with VirtualGroup(group_uris) as group:
            with pytest.raises(ValueError):
                group.open_array(attr="c")
