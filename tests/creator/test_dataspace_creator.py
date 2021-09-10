# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
import numpy as np
import pytest

import tiledb
from tiledb.cf import METADATA_ARRAY_NAME, DataspaceCreator, GroupSchema
from tiledb.cf.creator import ArrayCreator, SharedDim


class TestDataspaceCreatorExample1:
    @pytest.fixture
    def dataspace_creator(self):
        creator = DataspaceCreator()
        creator.add_dim("pressure.index", (0, 3), np.uint64)
        creator.add_dim("temperature", (1, 8), np.uint64)
        creator.add_array("A1", ("pressure.index",))
        filters = tiledb.FilterList(
            [
                tiledb.ZstdFilter(level=1),
            ]
        )
        creator.add_array(
            "A2",
            ("pressure.index", "temperature"),
            sparse=True,
            dim_filters={"temperature": filters},
        )
        creator.add_array("A3", ("temperature",))
        creator.set_array_properties("A1", tiles=(2,))
        creator.set_array_properties("A2", tiles=(2, 4))
        creator.set_array_properties("A3", tiles=[8])
        creator.add_attr("pressure.data", "A1", np.float64)
        creator.add_attr("b", "A1", np.float64)
        creator.add_attr("c", "A1", np.uint64)
        creator.add_attr("d", "A2", np.uint64)
        creator.add_attr("e", "A3", np.float64)
        return creator

    def test_repr(self, dataspace_creator):
        assert isinstance(repr(dataspace_creator), str)

    def test_repr_html(self, dataspace_creator):
        try:
            tidylib = pytest.importorskip("tidylib")
            html_summary = dataspace_creator._repr_html_()
            _, errors = tidylib.tidy_fragment(html_summary)
        except OSError:
            pytest.skip("unable to import libtidy backend")
        assert not bool(errors)

    def test_array_creators(self, dataspace_creator):
        array_names = set()
        for array_creator in dataspace_creator.array_creators():
            assert isinstance(array_creator, ArrayCreator)
            array_names.add(array_creator.name)
        assert array_names == {"A1", "A2", "A3"}

    def test_get_array_creator(self, dataspace_creator):
        array_creator = dataspace_creator.get_array_creator("A2")
        assert isinstance(array_creator, ArrayCreator)
        assert array_creator.name == "A2"

    def test_shared_dims(self, dataspace_creator):
        dim_names = set()
        for shared_dim in dataspace_creator.shared_dims():
            assert isinstance(shared_dim, SharedDim)
            dim_names.add(shared_dim.name)
        assert dim_names == {"pressure.index", "temperature"}

    def test_get_shared_dims(self, dataspace_creator):
        shared_dim = dataspace_creator.get_shared_dim("temperature")
        assert isinstance(shared_dim, SharedDim)
        assert shared_dim.name == "temperature"

    def test_array_names(self, dataspace_creator):
        assert set(dataspace_creator.array_names) == {"A1", "A2", "A3"}

    def test_attr_names(self, dataspace_creator):
        assert set(dataspace_creator.attr_names) == {
            "pressure.data",
            "b",
            "c",
            "d",
            "e",
        }

    def test_dim_names(self, dataspace_creator):
        assert set(dataspace_creator.dim_names) == {
            "pressure.index",
            "temperature",
        }

    def test_get_array_property(self, dataspace_creator):
        tiles = dataspace_creator.get_array_property("A1", "tiles")
        assert tiles == (2,)

    def test_get_attr_property(self, dataspace_creator):
        dtype = dataspace_creator.get_attr_property("b", "dtype")
        assert dtype == np.dtype(np.float64)

    def test_get_dim_property(self, dataspace_creator):
        dtype = dataspace_creator.get_dim_property("temperature", "dtype")
        assert dtype == np.dtype(np.uint64)

    def test_to_schema(self, dataspace_creator):
        group_schema = dataspace_creator.to_schema()
        assert isinstance(group_schema, GroupSchema)
        assert len(group_schema) == 3
        assert group_schema["A1"] == tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(
                    name="pressure.index", domain=(0, 3), tile=2, dtype=np.uint64
                )
            ),
            attrs=[
                tiledb.Attr(name="pressure.data", dtype=np.float64),
                tiledb.Attr(name="b", dtype=np.float64),
                tiledb.Attr(name="c", dtype=np.uint64),
            ],
        )
        assert group_schema["A2"] == tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(
                    name="pressure.index", domain=(0, 3), tile=2, dtype=np.uint64
                ),
                tiledb.Dim(
                    name="temperature",
                    domain=(1, 8),
                    tile=4,
                    dtype=np.uint64,
                    filters=tiledb.FilterList(
                        [
                            tiledb.ZstdFilter(level=1),
                        ]
                    ),
                ),
            ),
            sparse=True,
            attrs=[tiledb.Attr(name="d", dtype=np.uint64)],
        )
        assert group_schema["A3"] == tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="temperature", domain=(1, 8), tile=8, dtype=np.uint64),
            ),
            attrs=[tiledb.Attr(name="e", dtype=np.float64)],
        )


def test_repr_empty_dataspace():
    assert isinstance(repr(DataspaceCreator()), str)


def test_create_array_no_array_error():
    creator = DataspaceCreator()
    with pytest.raises(ValueError):
        creator.create_array("tmp")


def test_array_name_exists_error():
    creator = DataspaceCreator()
    creator.add_dim("row", (0, 3), np.int64)
    creator.add_array("array1", ("row",))
    with pytest.raises(ValueError):
        creator.add_array("array1", ("row",))


def test_array_name_reserved_error():
    creator = DataspaceCreator()
    creator.add_dim("row", (0, 3), np.int64)
    with pytest.raises(ValueError):
        creator.add_array(METADATA_ARRAY_NAME, ("row",))


def test_add_attr_no_array_error():
    creator = DataspaceCreator()
    with pytest.raises(KeyError):
        creator.add_attr("attr", "array1", np.float64)


def test_add_attr_name_exists_error():
    creator = DataspaceCreator()
    creator.add_dim("row", (0, 3), np.int64)
    creator.add_array("array1", ("row",))
    creator.add_array("array2", ("row",))
    creator.add_attr("attr1", "array1", np.float64)
    with pytest.raises(ValueError):
        creator.add_attr("attr1", "array2", np.float64)


def test_add_attr_dim_name_in_array_exists_error():
    creator = DataspaceCreator()
    creator.add_dim("row", (0, 3), np.uint64)
    creator.add_array("array1", ("row",))
    with pytest.raises(ValueError):
        creator.add_attr("row", "array1", np.float64)


def test_add_attr_axis_data_coord_exists_error():
    creator = DataspaceCreator()
    creator.add_dim("row", (0, 3), np.int64)
    creator.add_array("array1", ("row",))
    creator.add_array("array2", ("row",))
    creator.add_attr("attr1", "array1", np.float64)
    with pytest.raises(ValueError):
        creator.add_attr("attr1.data", "array2", np.float64)


def test_add_dim_name_exists_error():
    creator = DataspaceCreator()
    creator.add_dim("row", (1, 4), np.uint64)
    with pytest.raises(ValueError):
        creator.add_dim("row", (0, 3), np.int64)


def test_get_property_attr_key_error():
    creator = DataspaceCreator()
    with pytest.raises(KeyError):
        creator.get_attr_property("a1", "nullable")


def test_remove_empty_array():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 10], np.int64)
    creator.add_array("A1", ["row"])
    creator.remove_array("A1")
    assert set(creator.array_names) == set()


def test_remove_array_with_attrs():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 10], np.int64)
    creator.add_array("A1", ["row"])
    creator.add_array("A2", ["row"])
    creator.add_attr("x1", "A1", np.float64)
    creator.add_attr("x2", "A1", np.float64)
    creator.add_attr("y1", "A2", np.float64)
    creator.remove_array("A2")
    assert set(creator.array_names) == {"A1"}
    assert set(creator.attr_names) == {"x1", "x2"}


def test_remove_renamed_array():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 10], np.int64)
    creator.add_array("A1", ["row"])
    creator.add_array("A2", ["row"])
    creator.add_attr("x1", "A1", np.float64)
    creator.add_attr("x2", "A1", np.float64)
    creator.add_attr("y1", "A2", np.float64)
    creator.rename_array("A2", "B1")
    creator.remove_array("B1")
    assert set(creator.array_names) == {"A1"}
    assert set(creator.attr_names) == {"x1", "x2"}


def test_remove_attr():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int64)
    creator.add_array("A1", ["row"])
    creator.add_attr("x1", "A1", np.float64)
    assert set(creator.attr_names) == {"x1"}
    creator.remove_attr("x1")
    assert set(creator.attr_names) == set()
    creator.add_array("A2", ["row"])
    creator.add_attr("x1", "A2", np.float64)
    assert set(creator.attr_names) == {"x1"}


def test_remove_attr_no_attr_error():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int64)
    creator.add_array("A1", ["row"])
    with pytest.raises(KeyError):
        creator.remove_attr("x1")


def test_remove_dim():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int32)
    assert set(creator.dim_names) == set(["row"])
    creator.remove_dim("row")
    assert set(creator.dim_names) == set()


def test_remove_dim_axis_data():
    creator = DataspaceCreator()
    creator.add_dim("row.data", [0.0, 100.0], np.float64)
    assert set(creator.dim_names) == {"row.data"}
    creator.remove_dim("row.data")
    assert set(creator.dim_names) == set()


def test_remove_dim_in_use_error():
    creator = DataspaceCreator()
    creator.add_dim("row.data", [0.0, 100.0], np.float64)
    creator.add_array("A1", ["row.data"], sparse=True)
    creator.add_array("A2", ["row.data"], sparse=True)
    with pytest.raises(ValueError):
        creator.remove_dim("row.data")


def test_rename_array():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int64)
    creator.add_array("A1", ["row"])
    assert set(creator.array_names) == {"A1"}
    creator.rename_array("A1", "B1")
    assert set(creator.array_names) == {"B1"}


def test_rename_array_with_attrs():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 3], np.int32)
    creator.add_dim("col", [0, 3], np.int32)
    creator.add_array("A1", ["row", "col"], sparse=True, tiles=[2, 2])
    creator.add_attr("x1", "A1", np.float64)
    creator.add_attr("x2", "A1", np.float64)
    assert set(creator.array_names) == {"A1"}
    creator.rename_array("A1", "B1")
    assert set(creator.array_names) == {"B1"}


def test_rename_array_name_exists_error():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int64)
    creator.add_array("A1", ["row"])
    creator.add_array("B1", ["row"])
    with pytest.raises(ValueError):
        creator.rename_array("A1", "B1")


def test_rename_attr():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int64)
    creator.add_array("A1", ["row"])
    creator.add_attr("x1", "A1", np.float64)
    assert set(creator.attr_names) == {"x1"}
    creator.rename_attr("x1", "y1")
    assert set(creator.attr_names) == {"y1"}
    group_schema = creator.to_schema()
    assert group_schema["A1"].has_attr("y1")


def test_rename_attr_with_set_attr_properties():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int64)
    creator.add_array("A1", ["row"])
    creator.add_attr("x1", "A1", np.float64)
    assert set(creator.attr_names) == {"x1"}
    creator.set_attr_properties("x1", name="y1")
    assert set(creator.attr_names) == {"y1"}
    group_schema = creator.to_schema()
    assert group_schema["A1"].has_attr("y1")


def test_rename_attr_name_exists_error():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int64)
    creator.add_array("A1", ["row"])
    creator.add_attr("x1", "A1", np.float64)
    creator.add_attr("y1", "A1", np.float64)
    with pytest.raises(ValueError):
        creator.rename_attr("x1", "y1")


def test_rename_attr_dim_name_in_array_exists_error():
    creator = DataspaceCreator()
    creator.add_dim("y1", (0, 4), np.int32)
    creator.add_array("A1", ["y1"])
    creator.add_attr("x1", "A1", np.float64)
    with pytest.raises(ValueError):
        creator.rename_attr("x1", "y1")


def test_rename_dim_not_used():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 3], np.int32)
    assert set(creator.dim_names) == {"row"}
    creator.rename_dim("row", "col")
    assert set(creator.dim_names) == {"col"}


def test_rename_dim_dataspace_axis():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 3], np.float64)
    assert set(creator.dim_names) == {"row"}
    creator.rename_dim("row", "col")
    assert set(creator.dim_names) == {"col"}


def test_rename_dim_used():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int64)
    creator.add_array("A1", ["row"])
    assert set(creator.dim_names) == {"row"}
    creator.rename_dim("row", "unit")
    assert set(creator.dim_names) == {"unit"}
    creator.add_attr("x1", "A1", np.int32)
    group_schema = creator.to_schema()
    assert group_schema["A1"].domain.has_dim("unit")


def test_array_no_dims_error():
    creator = DataspaceCreator()
    with pytest.raises(ValueError):
        creator.add_array("A1", [])


def test_rename_dim_name_exists_in_dataspace_error():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 3], np.int32)
    creator.add_dim("col", [0, 7], np.int32)
    with pytest.raises(NotImplementedError):
        creator.rename_dim("col", "row")


def test_rename_dim_no_merge_error():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 3], np.int32)
    creator.add_dim("col", [0, 3], np.int32)
    with pytest.raises(NotImplementedError):
        creator.rename_dim("col", "row")


def test_rename_dim_attr_name__in_array_exists_error():
    creator = DataspaceCreator()
    creator.add_dim("y1", (0, 4), np.int32)
    creator.add_array("A1", ["y1"])
    creator.add_attr("x1", "A1", np.float64)
    with pytest.raises(ValueError):
        creator.rename_dim("y1", "x1")


def test_set_attr_property():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int64)
    creator.add_array("A1", ["row"])
    creator.add_attr("x1", "A1", np.float64)
    creator.set_attr_properties("x1", fill=-1)
    group_schema = creator.to_schema()
    attr_schema = group_schema["A1"].attr(0)
    assert attr_schema.fill == -1


def test_set_dim_dtype():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int64)
    creator.set_dim_properties("row", dtype=np.float64)
    dtype = creator.get_dim_property("row", "dtype")
    assert dtype == np.dtype(np.float64)


def test_set_dim_domain():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int64)
    creator.set_dim_properties("row", domain=(1, 8))
    domain = creator.get_dim_property("row", "domain")
    assert domain == (1, 8)


def test_set_dim_name():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int64)
    creator.set_dim_properties("row", name="column")
    name = tuple(creator.dim_names)[0]
    assert name == "column"


def test_set_dim_dtype_with_array_check():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, 7], np.int64)
    creator.add_array("array1", ("row",))
    creator.set_dim_properties("row", dtype=np.uint32)
    dtype = creator.get_dim_property("row", "dtype")
    assert dtype == np.dtype(np.uint32)


def test_set_attr_property_no_attr_err():
    creator = DataspaceCreator()
    with pytest.raises(KeyError):
        creator.set_attr_properties("x1", fill=-1)


def test_dataspace_creator_name():
    from tiledb.cf.creator import dataspace_name

    assert dataspace_name("name.index") == "name"
    assert dataspace_name("name.data") == "name"
    assert dataspace_name("name.other") == "name.other"


def test_to_schema_bad_array():
    creator = DataspaceCreator()
    creator.add_dim("row", [0, -1], np.int64)
    creator.add_array("A1", ("row",))
    creator.add_attr("x1", "A1", dtype=np.float64)
    with pytest.raises(RuntimeError):
        creator.to_schema()
