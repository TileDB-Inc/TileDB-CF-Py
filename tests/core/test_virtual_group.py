# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf import VirtualGroup

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


class TestVirtualGroupWithArrays:

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
    def array_uris(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("simple_group"))
        array_uris = {
            "A1": uri + "/example1",
            "A2": uri + "/example2",
            "A3": uri + "/example3",
            "__tiledb_group": uri + "/metadta",
        }
        tiledb.group_create(uri)
        tiledb.Array.create(array_uris["__tiledb_group"], self._metadata_schema)
        tiledb.Array.create(array_uris["A1"], _array_schema_1)
        with tiledb.DenseArray(array_uris["A1"], mode="w") as array:
            array[:] = self._A1_data
        tiledb.Array.create(array_uris["A2"], _array_schema_2)
        tiledb.Array.create(array_uris["A3"], _array_schema_3)
        return array_uris

    def test_open_array_from_group(self, array_uris):
        with VirtualGroup(array_uris, array="A1") as group:
            array = group.array
            assert isinstance(array, tiledb.Array)
            assert array.mode == "r"
            assert np.array_equal(array[:, :]["a"], self._A1_data)

    def test_open_attr(self, array_uris):
        with VirtualGroup(array_uris, attr="a") as group:
            array = group.array
            assert isinstance(array, tiledb.Array)
            assert array.mode == "r"
            assert np.array_equal(array[:, :], self._A1_data)

    def test_array_metadata(self, array_uris):
        with VirtualGroup(array_uris, mode="w") as group:
            group.meta["test_key"] = "test_value"
        with VirtualGroup(array_uris, mode="r") as group:
            assert group.meta["test_key"] == "test_value"
