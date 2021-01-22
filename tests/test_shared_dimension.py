import numpy as np
import pytest

import tiledb
from tiledb.cf import DataType, SharedDimension

_tiledb_dim = [
    tiledb.Dim(name="dim", domain=(1, 4), tile=4, dtype=np.int32),
]

_bad_name_data = [
    ("bad.name", (1, 4), np.int32),
    (".badname", (1, 4), np.int32),
]

_compare_create_data = [
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

_compare_init_data = [
    (("dim", (1, 4), np.int32), ("dim", (1, 4), DataType["TILEDB_INT32"]), True),
]


@pytest.mark.parametrize("dimension", _tiledb_dim)
def test_create_dimension(dimension):
    """Tests creating a SharedDimension."""
    shared_dimension = SharedDimension.create(dimension)
    assert shared_dimension.name == dimension.name
    assert shared_dimension.domain == dimension.domain
    assert shared_dimension.dtype == dimension.dtype
    assert repr(shared_dimension) is not None
    assert hash(shared_dimension) == hash(
        (dimension.name, dimension.domain, dimension.dtype)
    )


@pytest.mark.parametrize("name, domain, data_type", _bad_name_data)
def test_bad_dimension_name(name, domain, data_type):
    """Tests that a dimension with a bad name raises a ValueError."""
    with pytest.raises(ValueError):
        SharedDimension(name, domain, data_type)


@pytest.mark.parametrize("dimension1, dimension2, is_equal", _compare_create_data)
def test_compare_create_dimensions(dimension1, dimension2, is_equal):
    """Tests __eq__ function for comparing 2 SharedDimensions."""
    dim1 = SharedDimension.create(dimension1)
    dim2 = SharedDimension.create(dimension2)
    if is_equal:
        assert dim1 == dim2
    else:
        assert dim1 != dim2


@pytest.mark.parametrize("dimension1, dimension2, is_equal", _compare_init_data)
def test_compare_init_dimensions(dimension1, dimension2, is_equal):
    """Tests __eq__ function for comparing 2 SharedDimensions."""
    dim1 = SharedDimension(dimension1[0], dimension1[1], dimension1[2])
    dim2 = SharedDimension(dimension2[0], dimension2[1], dimension2[2])
    if is_equal:
        assert dim1 == dim2
    else:
        assert dim1 != dim2


def test_compare_other_object():
    assert SharedDimension("dim", (1, 4), np.int32) != "not a dimension"
