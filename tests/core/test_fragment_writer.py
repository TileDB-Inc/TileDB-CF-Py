from collections import OrderedDict

import numpy as np
import pytest

import tiledb
from tiledb.cf.core._fragment_writer import FragmentWriter
from tiledb.cf.core._shared_dim import SharedDim
from tiledb.cf.core.source import NumpyData
from tiledb.cf.testing import assert_dict_arrays_equal


def test_fragment_writer_create_dense():
    dims = (
        SharedDim("dim1", (0, 100), np.uint32),
        SharedDim("dim2", (0, 100), np.uint32),
    )
    attr_names = ["attr1", "attr2", "attr3", "attr4"]
    writer = FragmentWriter.create_dense(dims, attr_names, ((0, 10), (0, 100)))
    assert writer.is_dense_region
    assert writer.ndim == 2
    assert writer.nattr == 4


def test_fragment_writer_create_sparse_coo():
    dims = (
        SharedDim("dim1", (0, 100), np.uint32),
        SharedDim("dim2", (0, 100), np.uint32),
    )
    attr_names = ["attr1"]
    writer = FragmentWriter.create_sparse_coo(dims, attr_names, 8)
    assert not writer.is_dense_region
    assert writer.ndim == 2
    assert writer.nattr == 1


def test_fragment_writer_create_sparse_row_major():
    dims = (
        SharedDim("dim1", (0, 100), np.uint32),
        SharedDim("dim2", (0, 100), np.uint32),
    )
    attr_names = ["attr1", "attr2", "attr3"]
    writer = FragmentWriter.create_sparse_row_major(
        dims, attr_names, ((0, 10), (0, 100))
    )
    assert not writer.is_dense_region
    assert writer.ndim == 2
    assert writer.nattr == 3


def test_fragment_writer_remove_attr():
    dims = (
        SharedDim("dim1", (0, 100), np.uint32),
        SharedDim("dim2", (0, 100), np.uint32),
    )
    attr_names = ["attr1", "attr2", "attr3", "attr4"]
    writer = FragmentWriter.create_dense(dims, attr_names, ((0, 10), (0, 100)))
    assert writer.is_dense_region
    assert writer.nattr == 4
    writer.remove_attr("attr3")
    assert writer.nattr == 3


def test_fragment_writer_dense_1D_full(tmpdir):
    # Define data.
    attr_data = np.arange(-3, 5)

    # Create fragment writer.
    writer = FragmentWriter.create_dense(
        (SharedDim("dim1", (0, 7), np.uint32),), [], None
    )

    # Check fragment writer.
    assert writer.ndim == 1
    assert writer.nattr == 0

    # Add attribute and check update.
    writer.add_attr("attr1")
    assert writer.nattr == 1

    # Add attribute data.
    writer.set_attr_data("attr1", NumpyData(attr_data, metadata={"key": "value"}))

    # Create base array.
    uri = str(tmpdir.join("test_fragment_writer_dense,_1D_full"))
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(tiledb.Dim("dim1", domain=(0, 7), dtype=np.uint32)),
        attrs=[tiledb.Attr("attr1", dtype=np.int64)],
    )
    tiledb.Array.create(uri, schema)

    with tiledb.open(uri, "w") as array:
        writer.write(array)

    with tiledb.open(uri) as array:
        result = array[...]
        meta = dict(array.meta.items())

    assert_dict_arrays_equal(result, {"attr1": attr_data})
    assert len(meta) == 1
    assert meta["__tiledb_attr.attr1.key"] == "value"


def test_fragment_writer_sparse_row_major_1D_full(tmpdir):
    # Define data.
    attr_data = np.arange(-3, 5, dtype=np.int64)
    dim_data = np.arange(8, dtype=np.uint32)

    # Create fragment writer.
    writer = FragmentWriter.create_sparse_row_major(
        (SharedDim("dim1", (0, 7), np.uint32),), [], (8,)
    )

    # Check fragment writer.
    assert writer.ndim == 1
    assert writer.nattr == 0

    # Add attribute and check update.
    writer.add_attr("attr1")
    assert writer.nattr == 1

    # Add attribute data and dimension data.
    writer.set_attr_data("attr1", NumpyData(attr_data, metadata={"key1": "attr_value"}))
    writer.set_dim_data("dim1", NumpyData(dim_data, metadata={"key2": "dim_value"}))

    # Create base array.
    uri = str(tmpdir.join("test_fragment_writer_dense,_1D_full"))
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(tiledb.Dim("dim1", domain=(0, 7), dtype=np.uint32)),
        attrs=[tiledb.Attr("attr1", dtype=np.int64)],
        sparse=True,
    )
    tiledb.Array.create(uri, schema)

    with tiledb.open(uri, "w") as array:
        writer.write(array)

    with tiledb.open(uri) as array:
        result = array[...]
        meta = dict(array.meta.items())

    assert_dict_arrays_equal(
        result, OrderedDict([("attr1", attr_data), ("dim1", dim_data)]), False
    )
    assert len(meta) == 2
    assert meta["__tiledb_attr.attr1.key1"] == "attr_value"
    assert meta["__tiledb_dim.dim1.key2"] == "dim_value"


def test_fragment_writer_set_attr_data_key_error():
    dims = (
        SharedDim("dim1", (0, 100), np.uint32),
        SharedDim("dim2", (0, 100), np.uint32),
    )
    attr_names = ["attr1", "attr2"]
    writer = FragmentWriter.create_dense(dims, attr_names, ((0, 10), (0, 0)))
    with pytest.raises(KeyError):
        writer.set_attr_data("attr3", NumpyData(np.arange(11)))


def test_fragment_writer_set_attr_data_size_value_error():
    dims = (
        SharedDim("dim1", (0, 100), np.uint32),
        SharedDim("dim2", (0, 100), np.uint32),
    )
    attr_names = ["attr1", "attr2", "attr3"]
    writer = FragmentWriter.create_dense(dims, attr_names, ((0, 10), (0, 10)))
    with pytest.raises(ValueError):
        writer.set_attr_data("attr3", NumpyData(np.arange(11)))
