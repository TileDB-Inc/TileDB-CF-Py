import numpy as np
import pytest

import tiledb
from tiledb.cf import DataspaceGroup, DataspaceGroupSchema

_row = tiledb.Dim(name="rows", domain=(1, 4), tile=4, dtype=np.uint64)
_col = tiledb.Dim(name="cols", domain=(1, 4), tile=4, dtype=np.uint64)


_attr_a = tiledb.Attr(name="a", dtype=np.uint64)
_attr_b = tiledb.Attr(name="b", dtype=np.float64)
_attr_c = tiledb.Attr(name="c", dtype=np.dtype("U"))
_attr_d = tiledb.Attr(name="d", dtype=np.float64)
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
    attrs=[_attr_d],
)


class TestCreateDataspaceGroup:

    _metadata_schema = _array_schema_1
    _array_schemas = {
        "A1": _array_schema_1,
        "A2": _array_schema_2,
    }
    _group_schema = DataspaceGroupSchema(_array_schemas, _metadata_schema)
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


class TestCreateMetadata:
    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        """Creates a TileDB group and return URI."""
        uri = str(tmpdir_factory.mktemp("empty_group"))
        tiledb.group_create(uri)
        return uri

    def test_create_metadata(self, group_uri):
        group = DataspaceGroup(group_uri)
        assert group.schema.metadata_schema is None
        group.create_metadata_array()
        assert isinstance(group.schema.metadata_schema, tiledb.ArraySchema)


class TestSimpleDataspaceGroup:

    _metadata_schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.uint64)
        ),
        attrs=[tiledb.Attr(name="a", dtype=np.uint64)],
        sparse=True,
    )

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("group1"))
        DataspaceGroup.create(uri, DataspaceGroupSchema(None, self._metadata_schema))
        return uri

    def test_has_metadata(self, group_uri):
        group = DataspaceGroup(group_uri)
        with group.metadata_array() as metadata_array:
            assert isinstance(metadata_array, tiledb.Array)


class TestDataspaceGroupWithArrays:

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
        group = DataspaceGroup(group_uri)
        with group.array("A1") as array:
            assert isinstance(array, tiledb.Array)
            assert array.mode == "r"
            assert np.array_equal(array[:, :]["a"], self._A1_data)

    def test_open_attr(self, group_uri):
        group = DataspaceGroup(group_uri)
        with group.array(attr="a") as array:
            assert isinstance(array, tiledb.Array)
            assert array.mode == "r"
            assert np.array_equal(array[:, :], self._A1_data)
