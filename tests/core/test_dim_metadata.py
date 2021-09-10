# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf import DimMetadata


class TestDimMetadata:
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
            meta = DimMetadata(array.meta, "dim")
            assert len(meta) == 0
        with tiledb.DenseArray(array_uri, mode="w", timestamp=1) as array:
            meta = DimMetadata(array.meta, "dim")
            meta["key0"] = "dim_value"
            meta["key1"] = 10
            meta["key2"] = 0.1
        with tiledb.DenseArray(array_uri, mode="w", timestamp=2) as array:
            meta = DimMetadata(array.meta, "dim")
            del meta["key2"]
        with tiledb.DenseArray(array_uri, mode="r") as array:
            meta = DimMetadata(array.meta, "dim")
            assert set(meta.keys()) == set(["key0", "key1"])
            assert "key0" in meta
            assert meta["key0"] == "dim_value"

    def test_open_from_index(self, array_uri):
        with tiledb.DenseArray(array_uri, mode="r") as array:
            DimMetadata(array.meta, 0)

    def test_attr_not_in_array_exception(self, array_uri):
        with pytest.raises(KeyError):
            with tiledb.DenseArray(array_uri, mode="w") as array:
                _ = DimMetadata(array.meta, "x")

    def test_contains_not_string_exception(self, array_uri):
        with pytest.raises(TypeError):
            with tiledb.DenseArray(array_uri, mode="r") as array:
                meta = DimMetadata(array.meta, "dim")
                _ = 1 in meta

    def test_delitem_not_string_exception(self, array_uri):
        with pytest.raises(TypeError):
            with tiledb.DenseArray(array_uri, mode="w") as array:
                meta = DimMetadata(array.meta, "dim")
                del meta[1]

    def test_getitem_not_string_exception(self, array_uri):
        with pytest.raises(TypeError):
            with tiledb.DenseArray(array_uri, mode="r") as array:
                meta = DimMetadata(array.meta, "dim")
                _ = meta[1]

    def test_setitem_not_string_exception(self, array_uri):
        with pytest.raises(TypeError):
            with tiledb.DenseArray(array_uri, mode="w") as array:
                meta = DimMetadata(array.meta, "dim")
                meta[1] = "value"
