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
            attr_metadata = AttributeMetadata(array.meta, "attr")
            assert len(attr_metadata) == 0
        with tiledb.DenseArray(array_uri, mode="w") as array:
            attr_metadata = AttributeMetadata(array.meta, "attr")
            attr_metadata["key0"] = "attribute_value"
            attr_metadata["key1"] = 10
            attr_metadata["key2"] = 0.1
        with tiledb.DenseArray(array_uri, mode="w") as array:
            attr_metadata = AttributeMetadata(array.meta, "attr")
            del attr_metadata["key2"]
        with tiledb.DenseArray(array_uri, mode="r") as array:
            attr_metadata = AttributeMetadata(array.meta, "attr")
            assert set(attr_metadata.keys()) == set(["key0", "key1"])
            assert attr_metadata["key0"] == "attribute_value"
