# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf import create_group, open_group_array

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


class TestCreateGroup:
    _array_schemas = {"A1": _array_schema_1, "A2": _array_schema_2}
    _key = None

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        """Creates a TileDB Group from a mapping of arrays and returns scenario dict."""
        uri = str(tmpdir_factory.mktemp("group1"))
        ctx = None
        create_group(uri, self._array_schemas, key=self._key, ctx=ctx)
        return uri

    def test_array_schemas(self, group_uri):
        uri = group_uri
        assert tiledb.object_type(uri) == "group"
        for name, schema in self._array_schemas.items():
            with tiledb.Group(uri) as group:
                assert tiledb.ArraySchema.load(group[name].uri) == schema


class TestGroupWithArrays:
    _A1_data = np.array(
        ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]), dtype=np.uint64
    )

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("simple_group"))
        tiledb.group_create(uri)
        tiledb.Array.create(uri + "/A1", _array_schema_1)
        with tiledb.DenseArray(uri + "/A1", mode="w") as array:
            array[:] = self._A1_data
        tiledb.Array.create(uri + "/A2", _array_schema_2)
        tiledb.Array.create(uri + "/A3", _array_schema_3)
        with tiledb.Group(uri, mode="w") as group:
            group.add(uri="A1", name="A1", relative=True)
            group.add(uri="A2", name="A2", relative=True)
            group.add(uri="A3", name="A3", relative=True)
        filesystem = tiledb.VFS()
        filesystem.create_dir(uri + "/empty_dir")
        return uri

    def test_open_array_from_group(self, group_uri):
        with tiledb.Group(group_uri) as group:
            with open_group_array(group, array="A1") as array:
                assert isinstance(array, tiledb.Array)
                assert array.mode == "r"
                np.testing.assert_equal(array[:, :]["a"], self._A1_data)

    def test_open_attr(self, group_uri):
        with tiledb.Group(group_uri) as group:
            with open_group_array(group, attr="a") as array:
                assert isinstance(array, tiledb.Array)
                assert array.mode == "r"
                np.testing.assert_equal(array[:, :], self._A1_data)

    def test_no_array_with_attr_exception(self, group_uri):
        with tiledb.Group(group_uri) as group:
            with pytest.raises(KeyError):
                open_group_array(group, attr="bad_name")

    def test_ambiguous_array_exception(self, group_uri):
        with tiledb.Group(group_uri) as group:
            with pytest.raises(ValueError):
                open_group_array(group, attr="c")

    def test_no_values_error(self, group_uri):
        with tiledb.Group(group_uri) as group:
            with pytest.raises(ValueError):
                open_group_array(group)


def test_append_group(tmpdir):
    uri = str(tmpdir.mkdir("append_group_test"))
    create_group(uri, {"A1": _array_schema_1})
    create_group(uri, {"A2": _array_schema_2}, append=True)
    with tiledb.Group(uri) as group:
        assert group["A1"].type == tiledb.libtiledb.Array
        assert group["A2"].type == tiledb.libtiledb.Array
        a1_schema = tiledb.ArraySchema.load(group["A1"].uri)
        a2_schema = tiledb.ArraySchema.load(group["A2"].uri)
        assert a1_schema == _array_schema_1
        assert a2_schema == _array_schema_2


def test_append_group_array_exists_error(tmpdir):
    uri = str(tmpdir.mkdir("append_group_test"))
    create_group(uri, {"A1": _array_schema_1})
    with pytest.raises(ValueError):
        create_group(uri, {"A1": _array_schema_1}, append=True)
