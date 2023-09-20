import numpy as np

import tiledb
from tiledb.cf.core._array_creator import ArrayCreator
from tiledb.cf.core._shared_dim import SharedDim


def test_write_array_dense_1D_full(tmpdir):
    attr_data = np.arange(-3, 5)

    creator = ArrayCreator(
        dim_order=("dim1",),
        shared_dims=[SharedDim("dim1", (0, 7), np.uint32)],
    )
    creator.add_attr_creator("attr1", dtype=np.int64)
    creator.add_fragment_writer()
    creator["attr1"].set_fragment_data(0, attr_data)

    uri = str(tmpdir.mkdir("output").join("array_1D_full"))
    creator.write(uri)

    with tiledb.open(uri) as array:
        result = array[...]

    np.testing.assert_equal(result["attr1"], attr_data)
