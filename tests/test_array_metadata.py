import numpy as np
import pytest

import tiledb
from tiledb.cf import ArrayMetadata


class TestArrayMetadata:
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
            array.meta["__tiledb_attr.attr"] = "attribute value"
        return array_uri

    def test_modify_metadata(self, array_uri):
        with tiledb.DenseArray(array_uri, mode="r") as array:
            array_metadata = ArrayMetadata(array.meta)
            assert len(array_metadata) == 0
        with tiledb.DenseArray(array_uri, mode="w") as array:
            array_metadata = ArrayMetadata(array.meta)
            array_metadata["key0"] = "array value"
            array_metadata["key1"] = 10
            array_metadata["key2"] = 0.1
        with tiledb.DenseArray(array_uri, mode="w") as array:
            array_metadata = ArrayMetadata(array.meta)
            del array_metadata["key2"]
        with tiledb.DenseArray(array_uri, mode="r") as array:
            array_metadata = ArrayMetadata(array.meta)
            assert set(array_metadata.keys()) == set(["key0", "key1"])
            assert array_metadata["key0"] == "array value"
