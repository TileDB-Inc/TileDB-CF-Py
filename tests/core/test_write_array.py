from collections import OrderedDict

import numpy as np

import tiledb
from tiledb.cf.core._array_creator import ArrayCreator
from tiledb.cf.core._shared_dim import SharedDim
from tiledb.cf.testing import assert_dict_arrays_equal


def test_write_array_dense_1D_full(tmpdir):
    uri = str(tmpdir.mkdir("output").join("dense_1D_full"))
    attr_data = np.arange(-3, 5)

    creator = ArrayCreator(
        dim_order=("dim1",),
        shared_dims=[SharedDim("dim1", (0, 7), np.uint32)],
    )
    creator.add_dense_fragment_writer()
    creator.add_attr_creator("attr1", dtype=np.int64)
    creator["attr1"].set_writer_data(attr_data)

    creator.write(uri)

    with tiledb.open(uri) as array:
        result = array[...]

    assert_dict_arrays_equal(result, {"attr1": attr_data})


def test_write_array_sparse_1D_dense_region_full(tmpdir):
    uri = str(tmpdir.mkdir("output").join("sparse_1D_dense_full"))
    attr_data = np.arange(-3, 5)

    creator = ArrayCreator(
        dim_order=("dim1",),
        shared_dims=[SharedDim("dim1", (0, 7), np.uint32)],
        sparse=True,
    )
    creator.add_attr_creator("attr1", dtype=np.int64)
    creator.add_dense_fragment_writer()
    creator["attr1"].set_writer_data(attr_data)

    creator.write(uri)

    with tiledb.open(uri) as array:
        result = array[...]

    expected = OrderedDict()
    expected["dim1"] = np.arange(8, dtype=np.uint32)
    expected["attr1"] = attr_data
    assert_dict_arrays_equal(result, expected)


def test_write_array_sparse_1D_sparse_coo_region(tmpdir):
    uri = str(tmpdir.mkdir("output").join("sparse_1D_sparse_region"))
    dim_data = np.array([7, 1, 5, 3], dtype=np.uint32)
    attr_data = np.array([-3, 0, 100, -100], dtype=np.int64)

    creator = ArrayCreator(
        dim_order=("dim1",),
        shared_dims=[SharedDim("dim1", (0, 7), np.uint32)],
        sparse=True,
    )
    creator.add_attr_creator("attr1", dtype=np.int64)
    creator.add_sparse_fragment_writer(size=4)
    creator["attr1"].set_writer_data(attr_data)
    creator.domain_creator["dim1"].set_writer_data(dim_data)

    creator.write(uri)

    with tiledb.open(uri) as array:
        result = array.multi_index[:]

    expected = OrderedDict()
    expected["dim1"] = dim_data
    expected["attr1"] = attr_data
    assert_dict_arrays_equal(result, expected, False)


def test_write_array_sparse_1D_sparse_row_major_region(tmpdir):
    uri = str(tmpdir.mkdir("output").join("sparse_1D_sparse_row_major_region"))
    dim_data = np.array([7, 1, 5, 3], dtype=np.uint32)
    attr_data = np.array([-3, 0, 100, -100], dtype=np.int64)

    creator = ArrayCreator(
        dim_order=("dim1",),
        shared_dims=[SharedDim("dim1", (0, 7), np.uint32)],
        sparse=True,
    )
    creator.add_attr_creator("attr1", dtype=np.int64)
    creator.add_sparse_fragment_writer(shape=(4,), form="row-major")
    creator["attr1"].set_writer_data(attr_data)
    creator.domain_creator["dim1"].set_writer_data(dim_data)

    creator.write(uri)

    with tiledb.open(uri) as array:
        result = array.multi_index[:]

    expected = OrderedDict()
    expected["dim1"] = dim_data
    expected["attr1"] = attr_data
    assert_dict_arrays_equal(result, expected, False)


def test_write_array_dense_2D_full(tmpdir):
    uri = str(tmpdir.mkdir("output").join("dense_2D_full"))
    attr_data = np.resize(np.arange(-3, 28), (8, 4))

    creator = ArrayCreator(
        dim_order=("dim1", "dim2"),
        shared_dims=[
            SharedDim("dim1", (0, 7), np.uint32),
            SharedDim("dim2", (0, 3), np.uint32),
        ],
    )
    creator.add_attr_creator("attr1", dtype=np.int64)
    creator.add_dense_fragment_writer()
    creator["attr1"].set_writer_data(attr_data)

    creator.write(uri)

    with tiledb.open(uri) as array:
        result = array[...]

    assert_dict_arrays_equal(result, {"attr1": attr_data})


def test_write_array_sparse_2D_dense_region_full(tmpdir):
    uri = str(tmpdir.mkdir("output").join("sparse_2D_dense_full"))
    attr_data = np.resize(np.arange(-3, 28), (8, 4))

    creator = ArrayCreator(
        dim_order=("dim1", "dim2"),
        shared_dims=[
            SharedDim("dim1", (0, 7), np.uint32),
            SharedDim("dim2", (0, 3), np.uint32),
        ],
        sparse=True,
    )
    creator.add_attr_creator("attr1", dtype=np.int64)
    creator.add_dense_fragment_writer()
    creator["attr1"].set_writer_data(attr_data)

    creator.write(uri)

    with tiledb.open(uri) as array:
        result = array[...]

    expected = OrderedDict()
    dim1_coords, dim2_coords = np.meshgrid(
        np.arange(8, dtype=np.uint32), np.arange(4, dtype=np.uint32), indexing="ij"
    )
    expected["dim1"] = dim1_coords.reshape(-1)
    expected["dim2"] = dim2_coords.reshape(-1)
    expected["attr1"] = attr_data.reshape(-1)
    assert_dict_arrays_equal(result, expected)


def test_write_array_sparse_2D_sparse_coo_region(tmpdir):
    uri = str(tmpdir.mkdir("output").join("sparse_1D_sparse_region"))
    dim1_data = np.array([7, 1, 5, 3], dtype=np.uint32)
    dim2_data = np.array([0, 1, 1, 0], dtype=np.uint32)
    attr_data = np.array([-3, 0, 100, -100], dtype=np.int64)

    creator = ArrayCreator(
        dim_order=("dim1", "dim2"),
        shared_dims=[
            SharedDim("dim1", (0, 7), np.uint32),
            SharedDim("dim2", (0, 3), np.uint32),
        ],
        sparse=True,
    )
    creator.add_attr_creator("attr1", dtype=np.int64)
    creator.add_sparse_fragment_writer(size=4)
    creator["attr1"].set_writer_data(attr_data)
    creator.domain_creator["dim1"].set_writer_data(dim1_data)
    creator.domain_creator["dim2"].set_writer_data(dim2_data)

    creator.write(uri)

    with tiledb.open(uri) as array:
        result = array.multi_index[:]

    expected = OrderedDict()
    expected["dim1"] = dim1_data
    expected["dim2"] = dim2_data
    expected["attr1"] = attr_data
    assert_dict_arrays_equal(result, expected, False)


def test_write_array_sparse_2D_sparse_row_major_region(tmpdir):
    uri = str(tmpdir.mkdir("output").join("sparse_1D_sparse_region"))
    dim1_data = np.array([7, 1, 5], dtype=np.uint32)
    dim2_data = np.array([0, 3, 1, 2], dtype=np.uint32)
    attr_data = np.arange(-6, 6, dtype=np.int64)

    creator = ArrayCreator(
        dim_order=("dim1", "dim2"),
        shared_dims=[
            SharedDim("dim1", (0, 7), np.uint32),
            SharedDim("dim2", (0, 3), np.uint32),
        ],
        sparse=True,
    )
    creator.add_attr_creator("attr1", dtype=np.int64)
    creator.add_sparse_fragment_writer(shape=(3, 4), form="row-major")
    creator["attr1"].set_writer_data(attr_data)
    creator.domain_creator["dim1"].set_writer_data(dim1_data)
    creator.domain_creator["dim2"].set_writer_data(dim2_data)

    creator.write(uri)

    with tiledb.open(uri) as array:
        result = array.multi_index[:]

    expected = OrderedDict()
    expected["dim1"] = np.repeat(dim1_data, 4)
    print(expected["dim1"])
    expected["dim2"] = np.tile(dim2_data, 3)
    expected["attr1"] = attr_data
    assert_dict_arrays_equal(result, expected, False)
