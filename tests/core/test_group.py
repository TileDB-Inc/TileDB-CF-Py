# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf import Group, GroupSchema

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
    _array_schemas = [
        ("A1", _array_schema_1),
        ("A2", _array_schema_2),
    ]
    _group_schema = GroupSchema(_array_schemas)
    _key = None

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        """Creates a TileDB Group from GroupSchema and returns scenario dict."""
        uri = str(tmpdir_factory.mktemp("group1"))
        ctx = None
        Group.create(uri, self._group_schema, self._key, ctx)
        return uri

    def test_array_schemas(self, group_uri):
        uri = group_uri
        assert tiledb.object_type(uri) == "group"
        for name, schema in self._array_schemas:
            array_uri = group_uri + "/" + name
            assert tiledb.ArraySchema.load(array_uri, key=self._key) == schema


class TestNotTileDBURI:
    @pytest.fixture(scope="class")
    def empty_uri(self, tmpdir_factory):
        """Create an empty directory and return URI."""
        return str(tmpdir_factory.mktemp("empty"))

    def test_not_group_exception(self, empty_uri):
        with pytest.raises(ValueError):
            Group(empty_uri)


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
        with Group(group_uri) as group:
            with group.open_array(array="A1") as array:
                assert isinstance(array, tiledb.Array)
                assert array.mode == "r"
                np.testing.assert_equal(array[:, :]["a"], self._A1_data)

    def test_open_attr(self, group_uri):
        with Group(group_uri) as group:
            with group.open_array(attr="a") as array:
                assert isinstance(array, tiledb.Array)
                assert array.mode == "r"
                np.testing.assert_equal(array[:, :], self._A1_data)

    def test_no_array_with_attr_exception(self, group_uri):
        with Group(group_uri) as group:
            with pytest.raises(KeyError):
                group.open_array(attr="bad_name")

    def test_ambiguous_array_exception(self, group_uri):
        with Group(group_uri) as group:
            with pytest.raises(ValueError):
                group.open_array(attr="c")

    def test_no_values_error(self, group_uri):
        with Group(group_uri) as group:
            with pytest.raises(ValueError):
                group.open_array()

    def test_close_array_with_array_name(self, group_uri):
        with Group(group_uri) as group:
            with group.open_array(array="A1") as array:
                group.close_array(array="A1")
                assert not array.isopen

    def test_close_array_with_attr_name(self, group_uri):
        with Group(group_uri) as group:
            with group.open_array(attr="a") as array:
                group.close_array(attr="a")
                assert not array.isopen

    def test_close_array_no_values_error(self, group_uri):
        with Group(group_uri) as group:
            with pytest.raises(ValueError):
                group.close_array()

    def test_close_no_array_with_attr_exception(self, group_uri):
        with Group(group_uri) as group:
            with pytest.raises(KeyError):
                group.close_array(attr="bad_name")

    def test_close_ambiguous_array_exception(self, group_uri):
        with Group(group_uri) as group:
            with pytest.raises(ValueError):
                group.close_array(attr="c")


def test_append_group(tmpdir):
    uri = str(tmpdir.mkdir("append_group_test"))
    group_schema_1 = GroupSchema({"A1": _array_schema_1})
    Group.create(uri, group_schema_1)
    group_schema_2 = GroupSchema({"A2": _array_schema_2})
    Group.create(uri, group_schema_2, append=True)
    result = GroupSchema.load(uri)
    expected = GroupSchema({"A1": _array_schema_1, "A2": _array_schema_2})
    assert result == expected


def test_append_group_add_metadata(tmpdir):
    uri = str(tmpdir.mkdir("append_group_test"))
    group_schema_1 = GroupSchema({"A1": _array_schema_1})
    Group.create(uri, group_schema_1)
    group_schema_2 = GroupSchema({"A2": _array_schema_2})
    Group.create(uri, group_schema_2, append=True)
    result = GroupSchema.load(uri)
    expected = GroupSchema({"A1": _array_schema_1, "A2": _array_schema_2})
    assert result == expected


def test_append_group_array_exists_error(tmpdir):
    uri = str(tmpdir.mkdir("append_group_test"))
    group_schema_1 = GroupSchema({"A1": _array_schema_1})
    Group.create(uri, group_schema_1)
    with pytest.raises(ValueError):
        Group.create(uri, group_schema_1, append=True)
