# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf.creator import ArrayCreator, AttrCreator, SharedDim


class TestArrayCreatorSparseExample1:
    @pytest.fixture
    def array_creator(self):
        dims = [
            SharedDim("row", (0, 63), np.uint32),
            SharedDim("col", (0, 31), np.uint32),
        ]
        creator = ArrayCreator(
            dims,
            coords_filters=tiledb.FilterList(
                [
                    tiledb.ZstdFilter(level=6),
                ]
            ),
            offsets_filters=tiledb.FilterList(
                [
                    tiledb.Bzip2Filter(),
                ]
            ),
            tiles=(32, 16),
        )
        creator.dim_filters = {
            "row": tiledb.FilterList(
                [
                    tiledb.ZstdFilter(level=1),
                ]
            ),
            "col": tiledb.FilterList(
                [
                    tiledb.GzipFilter(level=5),
                ]
            ),
        }
        attr_filters = tiledb.FilterList(
            [
                tiledb.ZstdFilter(level=7),
            ]
        )
        creator.add_attr(
            AttrCreator("enthalpy", np.dtype("float64"), filters=attr_filters)
        )
        return creator

    def test_repr(self, array_creator):
        assert isinstance(repr(array_creator), str)

    def test_create(self, tmpdir, array_creator):
        uri = str(tmpdir.mkdir("output").join("sparse_example_1"))
        array_creator.create(uri)
        assert tiledb.object_type(uri) == "array"

    def test_dim_filters(self, array_creator):
        filters = array_creator.dim_filters
        assert filters == {
            "row": tiledb.FilterList(
                [
                    tiledb.ZstdFilter(level=1),
                ]
            ),
            "col": tiledb.FilterList(
                [
                    tiledb.GzipFilter(level=5),
                ]
            ),
        }

    def test_tiles(self, array_creator):
        tiles = array_creator.tiles
        assert tiles == (32, 16)


def test_rename_attr_set_attr_properties():
    dims = [
        SharedDim("pressure", (0.0, 1000.0), np.float64),
        SharedDim("temperature", (-200.0, 200.0), np.float64),
    ]
    creator = ArrayCreator(dims, sparse=True)
    creator.add_attr(AttrCreator("enthalp", np.dtype("float64")))
    assert set(creator.attr_names) == {"enthalp"}
    creator.set_attr_properties("enthalp", name="enthalpy")
    assert set(creator.attr_names) == {"enthalpy"}


def test_array_no_dim_error():
    with pytest.raises(ValueError):
        ArrayCreator([])


def test_name_exists_error():
    dims = [
        SharedDim("pressure", (0.0, 1000.0), np.float64),
        SharedDim("temperature", (-200.0, 200.0), np.float64),
    ]
    creator = ArrayCreator(dims, sparse=True)
    creator.add_attr(AttrCreator("enthalpy", np.float64))
    with pytest.raises(ValueError):
        creator.add_attr(AttrCreator("enthalpy", np.float64))


def test_dim_name_exists_error():
    dims = [
        SharedDim("pressure", (0.0, 1000.0), np.float64),
        SharedDim("temperature", (-200.0, 200.0), np.float64),
    ]
    creator = ArrayCreator(dims, sparse=True)
    creator.add_attr(AttrCreator("enthalpy", np.float64))
    with pytest.raises(ValueError):
        creator.add_attr(AttrCreator("pressure", np.float64))


def test_bad_tiles_error():
    dims = [
        SharedDim("row", (0, 63), np.uint32),
        SharedDim("col", (0, 31), np.uint32),
    ]
    creator = ArrayCreator(dims)
    with pytest.raises(ValueError):
        creator.tiles = (4,)


def test_to_schema_no_attrs_error():
    dims = [
        SharedDim("row", (0, 63), np.uint32),
        SharedDim("col", (0, 31), np.uint32),
    ]
    creator = ArrayCreator(dims)
    with pytest.raises(ValueError):
        creator.to_schema()
