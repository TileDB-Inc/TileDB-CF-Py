# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
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
            array.meta["__tiledb_dim.dim"] = "dimension value"
        return array_uri

    def test_modify_metadata(self, array_uri):
        with tiledb.DenseArray(array_uri, mode="r") as array:
            meta = ArrayMetadata(array.meta)
            assert len(meta) == 0
            assert "__tiledb_attr.attr" not in meta
            assert "__tiledb_dim.dim" not in meta
        with tiledb.DenseArray(array_uri, mode="w", timestamp=1) as array:
            meta = ArrayMetadata(array.meta)
            meta["key0"] = "array value"
            meta["key1"] = 10
            meta["key2"] = 0.1
        with tiledb.DenseArray(array_uri, mode="w", timestamp=2) as array:
            meta = ArrayMetadata(array.meta)
            del meta["key2"]
        with tiledb.DenseArray(array_uri, mode="r") as array:
            meta = ArrayMetadata(array.meta)
            assert set(meta.keys()) == set(["key0", "key1"])
            assert "key0" in meta
            assert meta["key0"] == "array value"

    def test_delitem_attr_key_exception(self, array_uri):
        with pytest.raises(KeyError):
            with tiledb.DenseArray(array_uri, mode="w") as array:
                meta = ArrayMetadata(array.meta)
                del meta["__tiledb_attr.attr"]

    def test_delitem_dim_key_exeception(self, array_uri):
        with pytest.raises(KeyError):
            with tiledb.DenseArray(array_uri, mode="w") as array:
                meta = ArrayMetadata(array.meta)
                del meta["__tiledb_dim.dim"]

    def test_getitem_attr_key_exception(self, array_uri):
        with pytest.raises(KeyError):
            with tiledb.DenseArray(array_uri, mode="r") as array:
                meta = ArrayMetadata(array.meta)
                _ = meta["__tiledb_attr.attr"]

    def test_getitem_dim_key_exception(self, array_uri):
        with pytest.raises(KeyError):
            with tiledb.DenseArray(array_uri, mode="r") as array:
                meta = ArrayMetadata(array.meta)
                _ = meta["__tiledb_dim.dim"]

    def test_setitem_attr_key_exception(self, array_uri):
        with pytest.raises(KeyError):
            with tiledb.DenseArray(array_uri, mode="w") as array:
                meta = ArrayMetadata(array.meta)
                meta["__tiledb_attr.a"] = "value"

    def test_setitem_dim_key_exception(self, array_uri):
        with pytest.raises(KeyError):
            with tiledb.DenseArray(array_uri, mode="w") as array:
                meta = ArrayMetadata(array.meta)
                meta["__tiledb_dim.a"] = "value"
