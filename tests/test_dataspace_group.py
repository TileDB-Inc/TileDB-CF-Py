import numpy as np
import pytest

import tiledb
from tiledb.cf import DataspaceArray, DataspaceGroup, GroupSchema

_row = tiledb.Dim(name="rows", domain=(1, 4), tile=4, dtype=np.uint64)
_col = tiledb.Dim(name="cols", domain=(1, 4), tile=4, dtype=np.uint64)

_array_schema_1 = tiledb.ArraySchema(
    domain=tiledb.Domain(_row, _col),
    attrs=[tiledb.Attr(name="a", dtype=np.uint64)],
)
_array_schema_2 = tiledb.ArraySchema(
    domain=tiledb.Domain(_row),
    sparse=True,
    attrs=[
        tiledb.Attr(name="b", dtype=np.float64),
        tiledb.Attr(name="c", dtype=np.dtype("U")),
    ],
)
_array_schema_3 = tiledb.ArraySchema(
    domain=tiledb.Domain(_row, _col),
    attrs=[tiledb.Attr(name="d", dtype=np.float64)],
)
_metadata_schema = tiledb.ArraySchema(
    domain=tiledb.Domain(
        tiledb.Dim(name="dim", domain=(1, 4), tile=2, dtype=np.uint64)
    ),
    attrs=[tiledb.Attr(name="attr", dtype=np.uint64)],
    sparse=True,
)


class TestCreateDataspaceGroup:

    _array_schemas = {
        "A1": _array_schema_1,
        "A2": _array_schema_2,
    }
    _group_schema = GroupSchema(_array_schemas, _metadata_schema)
    _key = None

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        """Creates a TileDB DataspaceGroup from DataspaceGroupSchema and returns
        scenario dict."""
        uri = str(tmpdir_factory.mktemp("group1"))
        ctx = None
        DataspaceGroup.create(uri, self._group_schema, self._key, ctx)
        return uri

    def test_array_schemas(self, group_uri):
        uri = group_uri
        assert tiledb.object_type(uri) == "group"
        for name, schema in self._array_schemas.items():
            array_uri = group_uri + "/" + name
            assert tiledb.ArraySchema.load(array_uri, key=self._key) == schema


class TestNotTileDBURI:
    @pytest.fixture(scope="class")
    def empty_uri(self, tmpdir_factory):
        """Create an empty directory and return URI."""
        return str(tmpdir_factory.mktemp("empty"))

    def test_not_group_exception(self, empty_uri):
        with pytest.raises(ValueError):
            DataspaceGroup(empty_uri)


class TestSimpleDataspaceGroup:
    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("group1"))
        DataspaceGroup.create(
            uri,
            GroupSchema(None, _metadata_schema),
        )
        return uri

    def test_dataspace(self, group_uri):
        dataspace_group = DataspaceGroup(group_uri)
        with dataspace_group.dataspace_metadata_array(mode="r") as array:
            assert isinstance(array, DataspaceArray)


class TestDataspaceGroupWithArrays:

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
        return uri

    def test_open_array_from_group(self, group_uri):
        dataspace_group = DataspaceGroup(group_uri)
        with dataspace_group.dataspace_array("A1") as array:
            assert isinstance(array, DataspaceArray)
            assert array.base.mode == "r"
            assert np.array_equal(array.base[:, :]["a"], self._A1_data)

    # def test_open_attr(self, group_uri):
    #     group = DataspaceGroup(group_uri)
    #     with group.dataspace_array(attr="a") as array:
    #         assert isinstance(array, DataspaceArray)
    #         assert array.base.mode == "r"
    #         assert np.array_equal(array.base[:, :], self._A1_data)
