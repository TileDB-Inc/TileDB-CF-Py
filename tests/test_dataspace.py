import numpy as np
import pytest

import tiledb
from tiledb.cf import ArrayMetadata, AttributeMetadata, Dataspace, Group, GroupSchema

_row = tiledb.Dim(name="rows", domain=(1, 4), tile=4, dtype=np.uint64)
_col = tiledb.Dim(name="cols", domain=(1, 4), tile=4, dtype=np.uint64)


_attr_a = tiledb.Attr(name="a", dtype=np.uint64)
_attr_b = tiledb.Attr(name="b", dtype=np.float64)
_attr_c = tiledb.Attr(name="c", dtype=np.dtype("U"))
_attr_d = tiledb.Attr(name="d", dtype=np.int32)
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


class TestCreateArray:
    @pytest.fixture(scope="class")
    def array_uri(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("tmp")) + "/array"
        Dataspace.create(uri, _array_schema_1)
        return uri

    def test_create_group_dataspace(self, array_uri):
        assert tiledb.object_type(array_uri) == "array"
        assert tiledb.ArraySchema.load(array_uri, key=None) == _array_schema_1


class TestCreateGroup:

    _metadata_schema = _array_schema_1
    _array_schemas = [
        ("A1", _array_schema_1),
        ("A2", _array_schema_2),
    ]
    _group_schema = GroupSchema(_array_schemas, _metadata_schema)

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        """Creates a TileDB Group from GroupSchema and returns scenario dict."""
        uri = str(tmpdir_factory.mktemp("group1"))
        ctx = tiledb.Ctx()
        Dataspace.create(uri, self._group_schema, ctx=ctx)
        return uri

    def test_create_group_dataspace(self, group_uri):
        assert tiledb.object_type(group_uri) == "group"
        for name, schema in self._array_schemas:
            array_uri = group_uri + "/" + name
            assert tiledb.ArraySchema.load(array_uri) == schema


class TestCreateGroupExceptions:

    _array_schemas = [
        ("A1", _array_schema_1),
        ("A2", _array_schema_1),
    ]
    _group_schema = GroupSchema(_array_schemas)

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("group2"))
        return uri

    def test_not_dataspace_exception(self, group_uri):
        with pytest.raises(RuntimeError):
            Dataspace.create(group_uri, self._group_schema)

    def test_not_group_exception(self, group_uri):
        with pytest.raises(TypeError):
            Dataspace.create(group_uri, None)


class TestSimpleArray:
    @pytest.fixture(scope="class")
    def array_uri(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("simple")) + "/array"
        tiledb.Array.create(uri, _array_schema_2)
        return uri

    def test_reopen(self, array_uri):
        with Dataspace(array_uri) as dataspace:
            dataspace.reopen()


class TestSimpleGroup:

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
        Group.create(uri, GroupSchema(None, self._metadata_schema))
        return uri

    def test_has_metadata(self, group_uri):
        with Dataspace(group_uri) as dataspace:
            assert isinstance(dataspace, Dataspace)
            assert dataspace.meta is not None

    def test_reopen(self, group_uri):
        with Dataspace(group_uri) as dataspace:
            dataspace.reopen()


class TestGroupWithArrays:

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
        filesystem = tiledb.VFS()
        filesystem.create_dir(uri + "/empty_dir")
        return uri

    def test_open_array_from_dataspace(self, group_uri):
        with Dataspace(group_uri, array="A1") as dataspace:
            array = dataspace.array
            assert isinstance(array, tiledb.Array)
            assert array.mode == "r"
            assert np.array_equal(array[:, :]["a"], self._A1_data)
            dataspace.reopen()
            assert array.mode == "r"

    def test_array_metadata(self, group_uri):
        with Dataspace(group_uri, array="A1") as dataspace:
            isinstance(dataspace.array_metadata, ArrayMetadata)

    def test_attribute_metadata_with_attr(self, group_uri):
        with Dataspace(group_uri, attr="a") as dataspace:
            isinstance(dataspace.attribute_metadata, AttributeMetadata)

    def test_attribute_metadata_with_single_attribute_array(self, group_uri):
        with Dataspace(group_uri, array="A3") as dataspace:
            isinstance(dataspace.attribute_metadata, AttributeMetadata)

    def test_get_attribute_metadata(self, group_uri):
        with Dataspace(group_uri, array="A2") as dataspace:
            isinstance(dataspace.get_attribute_metadata("b"), AttributeMetadata)

    def test_open_attr(self, group_uri):
        with Dataspace(group_uri, attr="a") as dataspace:
            array = dataspace.array
            assert isinstance(array, tiledb.Array)
            assert array.mode == "r"
            assert np.array_equal(array[:, :], self._A1_data)
