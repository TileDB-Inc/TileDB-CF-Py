import numpy as np
import pytest

import tiledb
from tiledb.cf.core._creator import ArrayCreator, SharedDim


class TestArrayCreatorSparseExample1:
    @pytest.fixture
    def array_creator(self):
        creator = ArrayCreator(
            dim_order=("row", "col"),
            shared_dims=(
                SharedDim(None, "row", (0, 63), np.uint32),
                SharedDim(None, "col", (0, 31), np.uint32),
            ),
            sparse=True,
            offsets_filters=tiledb.FilterList([tiledb.Bzip2Filter()]),
            tiles=(32, 16),
            dim_filters={
                "row": tiledb.FilterList([tiledb.ZstdFilter(level=1)]),
                "col": tiledb.FilterList([tiledb.GzipFilter(level=5)]),
            },
        )
        attr_filters = tiledb.FilterList([tiledb.ZstdFilter(level=7)])
        creator.add_attr_creator("enthalpy", np.dtype("float64"), filters=attr_filters)
        return creator

    def test_repr(self, array_creator):
        assert isinstance(repr(array_creator), str)

    def test_create(self, tmpdir, array_creator):
        uri = str(tmpdir.mkdir("output").join("sparse_example_1"))
        print(array_creator)
        array_creator.create(uri)
        assert tiledb.object_type(uri) == "array"

    def test_dim_filters(self, array_creator):
        filters = {
            dim_creator.name: dim_creator.filters
            for dim_creator in array_creator.domain_creator
        }
        assert filters == {
            "row": tiledb.FilterList([tiledb.ZstdFilter(level=1)]),
            "col": tiledb.FilterList([tiledb.GzipFilter(level=5)]),
        }

    def test_tiles(self, array_creator):
        tiles = tuple(dim_creator.tile for dim_creator in array_creator.domain_creator)
        assert tiles == (32, 16)

    def test_nattr(self, array_creator):
        nattr = array_creator.nattr
        assert nattr == 1

    def test_ndim(self, array_creator):
        ndim = array_creator.ndim
        assert ndim == 2


class TestArrayCreatorDense1:
    @pytest.fixture
    def array_creator(self):
        creator = ArrayCreator(
            shared_dims=[SharedDim(None, "row", (0, 63), np.uint32)], dim_order=("row",)
        )
        attr_filters = tiledb.FilterList([tiledb.ZstdFilter(level=7)])
        creator.add_attr_creator("enthalpy", np.dtype("float64"), filters=attr_filters)
        return creator

    def test_repr(self, array_creator):
        assert isinstance(repr(array_creator), str)

    def test_create(self, tmpdir, array_creator):
        uri = str(tmpdir.mkdir("output").join("dense_example_1"))
        array_creator.create(uri)
        assert tiledb.object_type(uri) == "array"

    def test_dim_filters(self, array_creator):
        filters = {
            dim_creator.name: dim_creator.filters
            for dim_creator in array_creator.domain_creator
        }
        assert filters == {"row": None}

    def test_tiles(self, array_creator):
        tiles = tuple(dim_creator.tile for dim_creator in array_creator.domain_creator)
        assert tiles == (None,)

    def test_nattr(self, array_creator):
        nattr = array_creator.nattr
        assert nattr == 1


class TestAttrsFitlers:
    """Collection of tests for setting default attribute filters."""

    def test_default_filter(self):
        """Tests new attribute filter is set to the attrs_filters value if the
        ``filters`` parameter is not specified."""
        attrs_filters = tiledb.FilterList([tiledb.ZstdFilter()])
        creator = ArrayCreator(
            dim_order=("row",),
            shared_dims=[SharedDim(None, "row", (0, 63), np.uint32)],
            attrs_filters=attrs_filters,
        )
        creator.add_attr_creator("x", np.dtype("float64"))
        assert creator.attr_creator("x").filters == attrs_filters

    def test_overwrite_default_filters(self):
        """Tests new attribute filter is set to the provided ``filters`` parameter when
        ``filters is not ``None``."""
        attrs_filters = tiledb.FilterList([tiledb.ZstdFilter()])
        new_filters = tiledb.FilterList([tiledb.GzipFilter(level=5)])
        creator = ArrayCreator(
            dim_order=("row",),
            shared_dims=[SharedDim(None, "row", (0, 63), np.uint32)],
            attrs_filters=attrs_filters,
        )
        creator.add_attr_creator("x", np.dtype("float64"), filters=new_filters)
        assert creator.attr_creator("x").filters == new_filters


def test_rename_attr():
    shared_dims = [
        SharedDim(None, "pressure", (0.0, 1000.0), np.float64),
        SharedDim(None, "temperature", (-200.0, 200.0), np.float64),
    ]
    array_creator = ArrayCreator(
        dim_order=("pressure", "temperature"), shared_dims=shared_dims, sparse=True
    )
    array_creator.add_attr_creator("enthalp", np.dtype("float64"))
    attr_names = tuple(attr_creator.name for attr_creator in array_creator)
    assert attr_names == ("enthalp",)
    array_creator.attr_creator("enthalp").name = "enthalpy"
    attr_names = tuple(attr_creator.name for attr_creator in array_creator)
    assert attr_names == ("enthalpy",)
    assert not array_creator.has_attr_creator("enthalp")
    assert array_creator.has_attr_creator("enthalpy")


def test_array_no_dim():
    creator = ArrayCreator()
    assert creator.domain_creator.ndim == 0


def test_repeating_name_error():
    with pytest.raises(ValueError):
        ArrayCreator(
            dim_order=("x", "x"),
            shared_dims=[SharedDim(None, "x", (1, 4), np.int32)],
        )


def test_name_exists_error():
    shared_dims = [
        SharedDim(None, "pressure", (0.0, 1000.0), np.float64),
        SharedDim(None, "temperature", (-200.0, 200.0), np.float64),
    ]
    creator = ArrayCreator(
        dim_order=("pressure", "temperature"), shared_dims=shared_dims, sparse=True
    )
    creator.add_attr_creator("enthalpy", np.float64)
    with pytest.raises(ValueError):
        creator.add_attr_creator("enthalpy", np.float64)


def test_dim_name_exists_error():
    shared_dims = [
        SharedDim(None, "pressure", (0.0, 1000.0), np.float64),
        SharedDim(None, "temperature", (-200.0, 200.0), np.float64),
    ]
    creator = ArrayCreator(
        dim_order=("pressure", "temperature"), shared_dims=shared_dims, sparse=True
    )
    creator.add_attr_creator("enthalpy", np.float64)
    with pytest.raises(ValueError):
        creator.add_attr_creator("pressure", np.float64)


def test_bad_tiles_error():
    shared_dims = [
        SharedDim(None, "row", (0, 63), np.uint32),
        SharedDim(None, "col", (0, 31), np.uint32),
    ]
    with pytest.raises(ValueError):
        ArrayCreator(shared_dims=shared_dims, dim_order=("row", "col"), tiles=(4,))


def test_to_schema_no_attrs_error():
    shared_dims = [
        SharedDim(None, "row", (0, 63), np.uint32),
        SharedDim(None, "col", (0, 31), np.uint32),
    ]
    creator = ArrayCreator(shared_dims=shared_dims, dim_order=("row", "col"))
    with pytest.raises(ValueError):
        creator.to_schema()


def test_inject_dim_creator_front():
    """Tests injecting a dimension into the front of the domain."""
    shared_dims = [
        SharedDim(None, "x1", (0, 7), np.uint32),
        SharedDim(None, "x2", (0, 7), np.uint32),
        SharedDim(None, "x0", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(shared_dims=shared_dims, dim_order=("x1", "x2"))
    creator.domain_creator.inject_dim_creator("x0", 0)
    dim_names = tuple(dim_creator.name for dim_creator in creator.domain_creator)
    assert dim_names == ("x0", "x1", "x2")


def test_inject_dim_creator_back():
    """Tests injecting a dimension into the back of the domain."""
    shared_dims = [
        SharedDim(None, "x1", (0, 7), np.uint32),
        SharedDim(None, "x2", (0, 7), np.uint32),
        SharedDim(None, "x3", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(dim_order=("x1", "x2"), shared_dims=shared_dims)
    creator.domain_creator.inject_dim_creator("x3", -1)
    dim_names = tuple(dim_creator.name for dim_creator in creator.domain_creator)
    assert dim_names == ("x1", "x2", "x3")


def test_inject_dim_creator_middle():
    """Tests injecting a dimension into the middle of the domain."""
    shared_dims = [
        SharedDim(None, "x0", (0, 7), np.uint32),
        SharedDim(None, "x2", (0, 7), np.uint32),
        SharedDim(None, "x1", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(dim_order=("x0", "x2"), shared_dims=shared_dims)
    creator.domain_creator.inject_dim_creator("x1", 1)
    dim_names = tuple(dim_creator.name for dim_creator in creator.domain_creator)
    assert dim_names == ("x0", "x1", "x2")


def test_inject_dim_attr_name_conflict_error():
    """Tests error when injecting a dimension with name matching a current attribute
    name."""
    shared_dims = [
        SharedDim(None, "x0", (0, 7), np.uint32),
        SharedDim(None, "x2", (0, 7), np.uint32),
        SharedDim(None, "x1", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(dim_order=("x0", "x1"), shared_dims=shared_dims)
    creator.add_attr_creator("x2", dtype=np.int32)
    with pytest.raises(ValueError):
        creator.domain_creator.inject_dim_creator("x2", 0)


def test_inject_dim_name_conflict_error():
    """Tests error when injecting a dimension with name matching a current dimension
    name."""
    shared_dims = [
        SharedDim(None, "x0", (0, 7), np.uint32),
        SharedDim(None, "x1", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(dim_order=("x0", "x1"), shared_dims=shared_dims)
    with pytest.raises(ValueError):
        creator.domain_creator.inject_dim_creator("x1", 0)


def test_inject_dim_neg_out_of_bound_error():
    """Tests error when injecting a dimension when poviding a negative position that is
    one element out-of-bounds."""
    shared_dims = [
        SharedDim(None, "x1", (0, 7), np.uint32),
        SharedDim(None, "x2", (0, 7), np.uint32),
        SharedDim(None, "x0", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(dim_order=("x0", "x2"), shared_dims=shared_dims)
    with pytest.raises(IndexError):
        creator.domain_creator.inject_dim_creator("x1", -4)


def test_inject_dim_pos_out_of_bound_error():
    """Tests error when injecting a dimension when providing a positive position that is
    one more than the size of the domain after creation."""
    shared_dims = [
        SharedDim(None, "x1", (0, 7), np.uint32),
        SharedDim(None, "x2", (0, 7), np.uint32),
        SharedDim(None, "x0", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(dim_order=("x0", "x2"), shared_dims=shared_dims)
    with pytest.raises(IndexError):
        creator.domain_creator.inject_dim_creator("x1", 3)


def test_remove_dim_creator_by_positive_int():
    """Tests removing a dimension using a positive dimension index."""
    shared_dims = [
        SharedDim(None, "x1", (0, 7), np.uint32),
        SharedDim(None, "x2", (0, 7), np.uint32),
        SharedDim(None, "x0", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(dim_order=("x0", "x1", "x2"), shared_dims=shared_dims)
    creator.domain_creator.remove_dim_creator(0)
    dim_names = tuple(dim_creator.name for dim_creator in creator.domain_creator)
    assert dim_names == ("x1", "x2")


def test_remove_dim_creator_by_negative_int():
    """Tests removing a dimension using a negative dimension index."""
    shared_dims = [
        SharedDim(None, "x1", (0, 7), np.uint32),
        SharedDim(None, "x2", (0, 7), np.uint32),
        SharedDim(None, "x0", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(dim_order=("x0", "x1", "x2"), shared_dims=shared_dims)
    creator.domain_creator.remove_dim_creator(-3)
    dim_names = tuple(dim_creator.name for dim_creator in creator.domain_creator)
    assert dim_names == ("x1", "x2")


def test_remove_dim_creator_front():
    """Tests removing the first dimension in the domain."""
    shared_dims = [
        SharedDim(None, "x1", (0, 7), np.uint32),
        SharedDim(None, "x2", (0, 7), np.uint32),
        SharedDim(None, "x0", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(dim_order=("x0", "x1", "x2"), shared_dims=shared_dims)
    creator.domain_creator.remove_dim_creator("x0")
    dim_names = tuple(dim_creator.name for dim_creator in creator.domain_creator)
    assert dim_names == ("x1", "x2")


def test_remove_dim_creator_back():
    """Tests removing the last dimension in the domain."""
    shared_dims = [
        SharedDim(None, "x1", (0, 7), np.uint32),
        SharedDim(None, "x2", (0, 7), np.uint32),
        SharedDim(None, "x3", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(dim_order=("x1", "x2", "x3"), shared_dims=shared_dims)
    creator.domain_creator.remove_dim_creator("x3")
    dim_names = tuple(dim_creator.name for dim_creator in creator.domain_creator)
    assert dim_names == ("x1", "x2")


def test_remove_dim_creator_middle():
    """Tests removing a dimension in the middle of the domain."""
    shared_dims = [
        SharedDim(None, "x1", (0, 7), np.uint32),
        SharedDim(None, "x2", (0, 7), np.uint32),
        SharedDim(None, "x0", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(dim_order=("x0", "x1", "x2"), shared_dims=shared_dims)
    creator.domain_creator.remove_dim_creator("x1")
    dim_names = tuple(dim_creator.name for dim_creator in creator.domain_creator)
    assert dim_names == ("x0", "x2")


def test_remove_dim_creator_position_index_error():
    """Tests attempting to remove a dimension that does not exist with a dimension
    index."""
    shared_dims = [
        SharedDim(None, "x1", (0, 7), np.uint32),
        SharedDim(None, "x2", (0, 7), np.uint32),
        SharedDim(None, "x0", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(dim_order=("x0", "x1", "x2"), shared_dims=shared_dims)
    with pytest.raises(IndexError):
        creator.domain_creator.remove_dim_creator(4)


def test_remove_dim_creator_name_key_error():
    """Tests attempting to remove a dimension that does not exist by name."""
    shared_dims = [
        SharedDim(None, "x1", (0, 7), np.uint32),
        SharedDim(None, "x2", (0, 7), np.uint32),
        SharedDim(None, "x0", (0, 4), np.uint32),
    ]
    creator = ArrayCreator(dim_order=("x0", "x1", "x2"), shared_dims=shared_dims)
    with pytest.raises(KeyError):
        creator.domain_creator.remove_dim_creator("x4")
