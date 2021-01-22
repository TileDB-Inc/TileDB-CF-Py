import numpy as np
import pytest

import tiledb
from tiledb.cf import Array, ArrayMetadata, AttributeMetadata, GroupSchema

_row = tiledb.Dim(name="rows", domain=(1, 4), tile=4, dtype=np.int32)
_col = tiledb.Dim(name="cols", domain=(1, 4), tile=4, dtype=np.int32)


_attr_a = tiledb.Attr(name="a", dtype=np.int32)
_attr_b = tiledb.Attr(name="b", dtype=np.int32)
_attr_c = tiledb.Attr(name="c", dtype=np.int32)
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


class TestSimpleArray:

    _metadata_schema = None
    _array_schemas = [
        ("A1", _array_schema_1),
        ("A2", _array_schema_2),
        ("A3", _array_schema_3),
    ]
    _group_schema = GroupSchema(_array_schemas, _metadata_schema)
    _key = None

    _A1_data = np.array(([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]))

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

    def test_open_array_from_group(self, group_uri):
        with Array(group_uri, mode="r", key=self._key, array="A1") as cf_array:
            array = cf_array.core
            assert isinstance(array, tiledb.Array)
            assert array.mode == "r"
            assert np.array_equal(array[:, :]["a"], self._A1_data)
            cf_array.reopen()
            assert array.mode == "r"

    def test_open_array(self, group_uri):
        uri = group_uri + "/A1"
        with Array(uri, mode="r", key=self._key, array="A1") as cf_array:
            array = cf_array.core
            assert isinstance(array, tiledb.Array)
            assert array.mode == "r"
            assert np.array_equal(array[:, :]["a"], self._A1_data)
            cf_array.reopen()
            assert cf_array.core.mode == "r"

    def test_array_metadata(self, group_uri):
        uri = group_uri + "/A1"
        with Array(uri, mode="r", key=self._key) as dataspace:
            isinstance(dataspace.array_metadata, ArrayMetadata)

    def test_attribute_metadata_with_attr(self, group_uri):
        with Array(group_uri, key=self._key, attr="a") as dataspace:
            isinstance(dataspace.attribute_metadata, AttributeMetadata)

    def test_attribute_metadata_with_single_attribute_array(self, group_uri):
        with Array(group_uri, key=self._key, array="A3") as dataspace:
            isinstance(dataspace.attribute_metadata, AttributeMetadata)

    def test_get_attribute_metadata(self, group_uri):
        with Array(group_uri, key=self._key, array="A2") as dataspace:
            isinstance(dataspace.get_attribute_metadata("b"), AttributeMetadata)

    def test_open_attr(self, group_uri):
        with Array(group_uri, mode="r", key=self._key, attr="a") as cf_array:
            array = cf_array.core
            assert isinstance(array, tiledb.Array)
            assert array.mode == "r"
            assert np.array_equal(array[:, :], self._A1_data)

    def test_no_array_exception(self, group_uri):
        with pytest.raises(ValueError):
            Array(group_uri, key=self._key)

    def test_conflicting_array_exception(self, group_uri):
        with pytest.raises(ValueError):
            Array(group_uri + "/A1", key=self._key, array="A2")

    def test_invalid_uri_exception(self, group_uri):
        with pytest.raises(ValueError):
            Array(group_uri + "/empty_dir", key=self._key)

    def test_ambiguous_metadata_attribute_exception(self, group_uri):
        with pytest.raises(ValueError):
            with Array(group_uri, key=self._key, array="A2") as dataspace:
                isinstance(dataspace.attribute_metadata, AttributeMetadata)