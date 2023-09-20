from collections import OrderedDict

import numpy as np

import tiledb
from tiledb.cf.core._array_creator import ArrayCreator
from tiledb.cf.core._shared_dim import SharedDim


def assert_dict_arrays_equal(d1, d2, ordered=True):
    assert d1.keys() == d2.keys(), "Keys not equal"

    if ordered:
        for k in d1.keys():
            np.testing.assert_array_equal(d1[k], d2[k])
    else:
        d1_dtypes = [tuple((name, value.dtype)) for name, value in d1.items()]
        d1_records = [tuple(values) for values in zip(*d1.values())]
        array1 = np.sort(np.array(d1_records, dtype=d1_dtypes))

        d2_dtypes = [tuple((name, value.dtype)) for name, value in d2.items()]
        d2_records = [tuple(values) for values in zip(*d2.values())]
        array2 = np.sort(np.array(d2_records, dtype=d2_dtypes))

        np.testing.assert_array_equal(array1, array2)


def test_write_array_dense_1D_full(tmpdir):
    uri = str(tmpdir.mkdir("output").join("dense_1D_full"))
    attr_data = np.arange(-3, 5)

    creator = ArrayCreator(
        dim_order=("dim1",),
        shared_dims=[SharedDim("dim1", (0, 7), np.uint32)],
    )
    creator.add_attr_creator("attr1", dtype=np.int64)
    creator.add_fragment_writer()
    creator["attr1"].set_fragment_data(0, attr_data)

    creator.write(uri)

    with tiledb.open(uri) as array:
        result = array[...]

    assert_dict_arrays_equal(result, {"attr1": attr_data})


def test_write_array_sparse_1D_full(tmpdir):
    uri = str(tmpdir.mkdir("output").join("sparse_1D_full"))
    attr_data = np.arange(-3, 5)

    creator = ArrayCreator(
        dim_order=("dim1",),
        shared_dims=[SharedDim("dim1", (0, 7), np.uint32)],
        sparse=True,
    )
    creator.add_attr_creator("attr1", dtype=np.int64)
    creator.add_fragment_writer()
    creator["attr1"].set_fragment_data(0, attr_data)

    creator.write(uri)

    with tiledb.open(uri) as array:
        result = array[...]

    expected = OrderedDict()
    expected["dim1"] = np.arange(8, dtype=np.uint32)
    expected["attr1"] = attr_data
    assert_dict_arrays_equal(result, expected)
