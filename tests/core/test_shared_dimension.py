import numpy as np
import pytest

import tiledb
from tiledb.cf.core._creator import DataspaceRegistry, SharedDim

_tiledb_dim = [
    tiledb.Dim(name="dim", domain=(1, 4), tile=4, dtype=np.int32),
]


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
    shared_dim = SharedDim(DataspaceRegistry(), "name", domain, dtype)
    assert shared_dim.is_index_dim == result


def test_compare_other_object():
    assert SharedDim(DataspaceRegistry(), "dim", (1, 4), np.int32) != "not a dimension"
