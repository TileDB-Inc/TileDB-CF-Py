import numpy as np
import pytest

import tiledb
from tiledb.cf import AttributeMetadata


class TestAttributeMetadata:
    @pytest.fixture(scope="class")
    def array_uri(self, tmpdir_factory):
        array_uri = str(tmpdir_factory.mktemp("test_array"))
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="dim", domain=(0, 0), tile=1, dtype=np.int32)
            ),
            attrs=[
                tiledb.Attr(name="attr", dtype=np.int32),
            ],
        )
        tiledb.Array.create(array_uri, schema)
        with tiledb.DenseArray(array_uri, mode="w") as array:
            array.meta["array_key"] = "array_value"
        return array_uri

    def test_modify_metadata(self, array_uri):
        with tiledb.DenseArray(array_uri, mode="r") as array:
            meta = AttributeMetadata(array.meta, "attr")
            assert len(meta) == 0
        with tiledb.DenseArray(array_uri, mode="w") as array:
            meta = AttributeMetadata(array.meta, "attr")
            meta["key0"] = "attribute_value"
            meta["key1"] = 10
            meta["key2"] = 0.1
        with tiledb.DenseArray(array_uri, mode="w") as array:
            meta = AttributeMetadata(array.meta, "attr")
            del meta["key2"]
        with tiledb.DenseArray(array_uri, mode="r") as array:
            meta = AttributeMetadata(array.meta, "attr")
            assert set(meta.keys()) == set(["key0", "key1"])
            assert "key0" in meta
            assert meta["key0"] == "attribute_value"

    def test_open_from_index(self, array_uri):
        with tiledb.DenseArray(array_uri, mode="r") as array:
            AttributeMetadata(array.meta, 0)

    def test_attr_not_in_array_exception(self, array_uri):
        with pytest.raises(ValueError):
            with tiledb.DenseArray(array_uri, mode="w") as array:
                _ = AttributeMetadata(array.meta, "x")

    def test_contains_not_string_exception(self, array_uri):
        with pytest.raises(TypeError):
            with tiledb.DenseArray(array_uri, mode="r") as array:
                meta = AttributeMetadata(array.meta, "attr")
                _ = 1 in meta

    def test_delitem_not_string_exception(self, array_uri):
        with pytest.raises(TypeError):
            with tiledb.DenseArray(array_uri, mode="w") as array:
                meta = AttributeMetadata(array.meta, "attr")
                del meta[1]

    def test_getitem_not_string_exception(self, array_uri):
        with pytest.raises(TypeError):
            with tiledb.DenseArray(array_uri, mode="r") as array:
                meta = AttributeMetadata(array.meta, "attr")
                _ = meta[1]

    def test_setitem_not_string_exception(self, array_uri):
        with pytest.raises(TypeError):
            with tiledb.DenseArray(array_uri, mode="w") as array:
                meta = AttributeMetadata(array.meta, "attr")
                meta[1] = "value"
