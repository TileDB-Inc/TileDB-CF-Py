# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf.creator import SharedDim

_tiledb_dim = [
    tiledb.Dim(name="dim", domain=(1, 4), tile=4, dtype=np.int32),
]

_compare_convert_data = [
    (
        tiledb.Dim(name="dim", domain=(1, 4), tile=4, dtype=np.int32),
        tiledb.Dim(name="dim", domain=(1, 4), tile=2, dtype=np.int32),
        True,
    ),
    (
        tiledb.Dim(name="dim", domain=(1, 4), tile=4, dtype=np.int32),
        tiledb.Dim(name="dim0", domain=(1, 4), tile=2, dtype=np.int32),
        False,
    ),
    (
        tiledb.Dim(name="dim", domain=(1, 4), tile=4, dtype=np.int32),
        tiledb.Dim(name="dim", domain=(1, 2), tile=2, dtype=np.int32),
        False,
    ),
    (
        tiledb.Dim(name="dim", domain=(1, 4), tile=4, dtype=np.int32),
        tiledb.Dim(name="dim", domain=(1, 4), tile=2, dtype=np.int64),
        False,
    ),
]


@pytest.mark.parametrize("dim", _tiledb_dim)
def test_convert_dim(dim):
    """Tests converting a SharedDim."""
    shared_dim = SharedDim.from_tiledb_dim(dim)
    assert shared_dim.name == dim.name
    assert shared_dim.domain == dim.domain
    assert shared_dim.dtype == dim.dtype
    assert repr(shared_dim) is not None


@pytest.mark.parametrize(
    "domain, dtype, result",
    [
        ((0, 4), np.int32, True),
        ((0, 4), np.uint32, True),
        ((1, 4), np.int32, False),
        ((0, 4), np.float64, False),
        (None, np.int32, False),
    ],
)
def test_is_index_dim(domain, dtype, result):
    shared_dim = SharedDim("name", domain, dtype)
    assert shared_dim.is_index_dim == result


@pytest.mark.parametrize("dim1, dim2, is_equal", _compare_convert_data)
def test_compare_convert_dims(dim1, dim2, is_equal):
    """Tests __eq__ function for comparing 2 SharedDims."""
    dim1 = SharedDim.from_tiledb_dim(dim1)
    dim2 = SharedDim.from_tiledb_dim(dim2)
    if is_equal:
        assert dim1 == dim2
    else:
        assert dim1 != dim2


def test_compare_other_object():
    assert SharedDim("dim", (1, 4), np.int32) != "not a dimension"
