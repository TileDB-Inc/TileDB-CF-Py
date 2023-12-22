import numpy as np
import pytest

import tiledb
from tiledb.cf import DataspaceCreator
from tiledb.cf.core._array_creator import ArrayCreator
from tiledb.cf.core._shared_dim import SharedDim


class TestDataspaceCreatorExample1:
    @pytest.fixture
    def dataspace_creator(self):
        creator = DataspaceCreator()
        creator.add_shared_dim("pressure.index", (0, 3), np.uint64)
        creator.add_shared_dim("temperature", (1, 8), np.uint64)
        creator.add_array_creator("A1", ("pressure.index",))
        filters = tiledb.FilterList(
            [
                tiledb.ZstdFilter(level=1),
            ]
        )
        creator.add_array_creator(
            "A2",
            ("pressure.index", "temperature"),
            sparse=True,
            dim_filters={"temperature": filters},
        )
        creator.add_array_creator("A3", ("temperature",))
        A1_domain = creator.get_array_creator("A1").domain_creator
        A1_domain.dim_creator(0).tile = 2
        A2_domain = creator.get_array_creator("A2").domain_creator
        A2_domain.dim_creator(0).tile = 2
        A2_domain.dim_creator(1).tile = 4
        A3_domain = creator.get_array_creator("A3").domain_creator
        A3_domain.dim_creator(0).tile = 8
        creator.add_attr_creator("pressure.data", "A1", np.float64)
        creator.add_attr_creator("b", "A1", np.float64)
        creator.add_attr_creator("c", "A1", np.uint64)
        creator.add_attr_creator("d", "A2", np.uint64)
        creator.add_attr_creator("e", "A3", np.float64)
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

    def test_get_array_creator_by_attr(self, dataspace_creator):
        array_creator = dataspace_creator.get_array_creator_by_attr("d")
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

    def test_to_schema(self, dataspace_creator):
        expected_schemas = {
            "A1": tiledb.ArraySchema(
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
            ),
            "A2": tiledb.ArraySchema(
                domain=tiledb.Domain(
                    tiledb.Dim(
                        name="pressure.index", domain=(0, 3), tile=2, dtype=np.uint64
                    ),
                    tiledb.Dim(
                        name="temperature",
                        domain=(1, 8),
                        tile=4,
                        dtype=np.uint64,
                        filters=tiledb.FilterList([tiledb.ZstdFilter(level=1)]),
                    ),
                ),
                sparse=True,
                attrs=[tiledb.Attr(name="d", dtype=np.uint64)],
            ),
            "A3": tiledb.ArraySchema(
                domain=tiledb.Domain(
                    tiledb.Dim(
                        name="temperature",
                        domain=(1, 8),
                        tile=8,
                        dtype=np.uint64,
                    ),
                ),
                attrs=[tiledb.Attr(name="e", dtype=np.float64)],
            ),
        }

        group_schema = dataspace_creator.to_schema()
        assert isinstance(group_schema, dict)
        assert len(group_schema) == 3
        assert group_schema["A1"] == expected_schemas["A1"]
        assert group_schema["A2"] == expected_schemas["A2"]
        assert group_schema["A3"] == expected_schemas["A3"]

    def test_create_group(self, dataspace_creator, tmpdir_factory):
        expected_schemas = {
            "A1": tiledb.ArraySchema(
                domain=tiledb.Domain(
                    tiledb.Dim(
                        name="pressure.index",
                        domain=(0, 3),
                        tile=2,
                        dtype=np.uint64,
                        filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                    )
                ),
                attrs=[
                    tiledb.Attr(name="pressure.data", dtype=np.float64),
                    tiledb.Attr(name="b", dtype=np.float64),
                    tiledb.Attr(name="c", dtype=np.uint64),
                ],
            ),
            "A2": tiledb.ArraySchema(
                domain=tiledb.Domain(
                    tiledb.Dim(
                        name="pressure.index",
                        domain=(0, 3),
                        tile=2,
                        dtype=np.uint64,
                        filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                    ),
                    tiledb.Dim(
                        name="temperature",
                        domain=(1, 8),
                        tile=4,
                        dtype=np.uint64,
                        filters=tiledb.FilterList([tiledb.ZstdFilter(level=1)]),
                    ),
                ),
                sparse=True,
                attrs=[tiledb.Attr(name="d", dtype=np.uint64)],
            ),
            "A3": tiledb.ArraySchema(
                domain=tiledb.Domain(
                    tiledb.Dim(
                        name="temperature",
                        domain=(1, 8),
                        tile=8,
                        dtype=np.uint64,
                        filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                    ),
                ),
                attrs=[tiledb.Attr(name="e", dtype=np.float64)],
            ),
        }

        uri = str(tmpdir_factory.mktemp("output").join("dataspace_example_1"))
        dataspace_creator.create_group(uri, append=False)
        with tiledb.Group(uri, mode="r") as group:
            assert len(group) == 3
            for item in group:
                result_schema = tiledb.ArraySchema.load(item.uri)
                expected_schema = expected_schemas[item.name]
                assert result_schema == expected_schema


def test_create_array(tmpdir):
    # Generate dataspace creator
    creator = DataspaceCreator()
    creator.add_shared_dim("d1", (0, 1_000_000), np.uint32)
    creator.add_array_creator("A1", ("d1"))
    creator.get_array_creator("A1").add_attr_creator("a1", np.float64)

    # Create the array
    array_uri = str(tmpdir.mkdir("output").join("example_create_array"))
    creator.create_array(array_uri)

    # Load and check schema
    schema = tiledb.ArraySchema.load(array_uri)
    assert schema == tiledb.ArraySchema(
        tiledb.Domain(
            tiledb.Dim(
                "d1",
                domain=(0, 1_000_000),
                dtype=np.uint32,
                filters=tiledb.FilterList([tiledb.ZstdFilter()]),
            )
        ),
        attrs=[tiledb.Attr("a1", dtype=np.float64)],
    )


def test_repr_empty_dataspace():
    assert isinstance(repr(DataspaceCreator()), str)


def test_create_array_no_array_error():
    creator = DataspaceCreator()
    with pytest.raises(ValueError):
        creator.create_array("tmp")


def test_array_name_exists_error():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", (0, 3), np.int64)
    creator.add_array_creator("array1", ("row",))
    with pytest.raises(ValueError):
        creator.add_array_creator("array1", ("row",))


def test_add_attr_no_array_error():
    creator = DataspaceCreator()
    with pytest.raises(KeyError):
        creator.add_attr_creator("attr", "array1", np.float64)


def test_add_attr_name_exists_error():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", (0, 3), np.int64)
    creator.add_array_creator("array1", ("row",))
    creator.add_attr_creator("attr1", "array1", np.float64)
    with pytest.raises(ValueError):
        creator.add_attr_creator("attr1", "array1", np.float64)


def test_add_attr_dim_name_in_array_exists_error():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", (0, 3), np.uint64)
    creator.add_array_creator("array1", ("row",))
    with pytest.raises(ValueError):
        creator.add_attr_creator("row", "array1", np.float64)


def test_add_dim_name_exists_error():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", (1, 4), np.uint64)
    with pytest.raises(ValueError):
        creator.add_shared_dim("row", (0, 3), np.int64)


def test_get_array_creator_by_attr_not_uniquer():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", (1, 4), np.uint32)
    creator.add_array_creator("A1", ["row"])
    creator.get_array_creator("A1").add_attr_creator("attr", np.int32)
    creator.add_array_creator("A2", ["row"])
    creator.get_array_creator("A2").add_attr_creator("attr", np.int32)
    with pytest.raises(ValueError):
        creator.get_array_creator_by_attr("attr")


def test_remove_empty_array():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 10], np.int64)
    creator.add_array_creator("A1", ["row"])
    creator.remove_array_creator("A1")
    assert set(creator.array_creators()) == set()


def test_remove_array_with_attrs():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 10], np.int64)
    creator.add_array_creator("A1", ["row"])
    creator.add_array_creator("A2", ["row"])
    creator.add_attr_creator("x1", "A1", np.float64)
    creator.add_attr_creator("x2", "A1", np.float64)
    creator.add_attr_creator("y1", "A2", np.float64)
    creator.remove_array_creator("A2")
    array_creators = creator.array_creators()
    assert next(array_creators).name == "A1"
    assert tuple(array_creators) == tuple()


def test_remove_renamed_array():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 10], np.int64)
    creator.add_array_creator("A1", ["row"])
    creator.add_array_creator("A2", ["row"])
    creator.add_attr_creator("x1", "A1", np.float64)
    creator.add_attr_creator("x2", "A1", np.float64)
    creator.add_attr_creator("y1", "A2", np.float64)
    creator.get_array_creator("A2").name = "B1"
    creator.remove_array_creator("B1")
    array_names = {array_creator.name for array_creator in creator.array_creators()}
    assert array_names == {"A1"}


def test_remove_attr():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 7], np.int64)
    creator.add_array_creator("A1", ["row"])
    creator.add_attr_creator("x1", "A1", np.float64)
    array_creator = creator.get_array_creator("A1")
    assert array_creator.nattr == 1
    creator.remove_attr_creator("x1")
    assert array_creator.nattr == 0
    creator.add_array_creator("A2", ["row"])
    creator.add_attr_creator("x1", "A2", np.float64)
    assert creator.get_array_creator("A2").nattr == 1


def test_remove_attr_no_attr_error():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 7], np.int64)
    creator.add_array_creator("A1", ["row"])
    with pytest.raises(KeyError):
        creator.remove_attr_creator("x1")


def test_remove_dim():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 7], np.int32)
    dim_names = set(dim.name for dim in creator.shared_dims())
    assert dim_names == set(["row"])
    creator.remove_shared_dim("row")
    dim_names = set(dim.name for dim in creator.shared_dims())
    assert dim_names == set()


def test_remove_dim_axis_data():
    creator = DataspaceCreator()
    creator.add_shared_dim("row.data", [0.0, 100.0], np.float64)
    dim_names = set(dim.name for dim in creator.shared_dims())
    assert dim_names == {"row.data"}
    creator.remove_shared_dim("row.data")
    dim_names = set(dim.name for dim in creator.shared_dims())
    assert dim_names == set()


def test_remove_dim_in_use_error():
    creator = DataspaceCreator()
    creator.add_shared_dim("row.data", [0.0, 100.0], np.float64)
    creator.add_array_creator("A1", ["row.data"], sparse=True)
    creator.add_array_creator("A2", ["row.data"], sparse=True)
    with pytest.raises(ValueError):
        creator.remove_shared_dim("row.data")


def test_rename_array():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 7], np.int64)
    creator.add_array_creator("A1", ["row"])
    array_names = {array.name for array in creator.array_creators()}
    assert array_names == {"A1"}
    creator.get_array_creator("A1").name = "B1"
    array_names = {array.name for array in creator.array_creators()}
    assert array_names == {"B1"}


def test_rename_array_with_attrs():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 3], np.int32)
    creator.add_shared_dim("col", [0, 3], np.int32)
    creator.add_array_creator("A1", ["row", "col"], sparse=True, tiles=[2, 2])
    creator.add_attr_creator("x1", "A1", np.float64)
    creator.add_attr_creator("x2", "A1", np.float64)
    array_names = {array.name for array in creator.array_creators()}
    assert array_names == {"A1"}
    creator.get_array_creator("A1").name = "B1"
    array_names = {array.name for array in creator.array_creators()}
    assert array_names == {"B1"}


def test_rename_array_name_exists_error():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 7], np.int64)
    creator.add_array_creator("A1", ["row"])
    creator.add_array_creator("B1", ["row"])
    with pytest.raises(ValueError):
        creator.get_array_creator("A1").name = "B1"


def test_rename_attr():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 7], np.int64)
    creator.add_array_creator("A1", ["row"])
    creator.add_attr_creator("x1", "A1", np.float64)
    A1 = creator.get_array_creator("A1")
    assert A1.nattr == 1
    assert A1.attr_creator(0).name == "x1"
    A1.attr_creator(0).name = "y1"
    assert A1.attr_creator(0).name == "y1"
    group_schema = creator.to_schema()
    assert group_schema["A1"].has_attr("y1")


def test_rename_attr_name_exists_error():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 7], np.int64)
    creator.add_array_creator("A1", ["row"])
    creator.add_attr_creator("x1", "A1", np.float64)
    creator.add_attr_creator("y1", "A1", np.float64)
    with pytest.raises(ValueError):
        creator.get_array_creator("A1").attr_creator("x1").name = "y1"


def test_rename_attr_dim_name_in_array_exists_error():
    creator = DataspaceCreator()
    creator.add_shared_dim("y1", (0, 4), np.int32)
    creator.add_array_creator("A1", ["y1"])
    creator.add_attr_creator("x1", "A1", np.float64)
    with pytest.raises(ValueError):
        creator.get_array_creator("A1").attr_creator("x1").name = "y1"


def test_rename_dim_not_used():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 3], np.int32)
    dim_names = set(dim.name for dim in creator.shared_dims())
    assert dim_names == {"row"}
    creator.get_shared_dim("row").name = "col"
    dim_names = set(dim.name for dim in creator.shared_dims())
    assert dim_names == {"col"}


def test_rename_dim_dataspace_axis():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 3], np.float64)
    dim_names = set(dim.name for dim in creator.shared_dims())
    assert dim_names == {"row"}
    creator.get_shared_dim("row").name = "col"
    dim_names = set(dim.name for dim in creator.shared_dims())
    assert dim_names == {"col"}


def test_rename_dim_used():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 7], np.int64)
    creator.add_array_creator("A1", ["row"])
    dim_names = set(dim.name for dim in creator.shared_dims())
    assert dim_names == {"row"}
    creator.get_shared_dim("row").name = "unit"
    dim_names = set(dim.name for dim in creator.shared_dims())
    assert dim_names == {"unit"}
    creator.add_attr_creator("x1", "A1", np.int32)
    group_schema = creator.to_schema()
    assert group_schema["A1"].domain.has_dim("unit")


def test_array_no_dims():
    creator = DataspaceCreator()
    creator.add_array_creator("A1", [])
    assert creator.get_array_creator("A1").domain_creator.ndim == 0


def test_array_no_dims_to_schema_error():
    creator = DataspaceCreator()
    creator.add_array_creator("A1", [])
    with pytest.raises(ValueError):
        creator.to_schema()


def test_rename_dim_name_exists_in_dataspace_error():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 3], np.int32)
    creator.add_shared_dim("col", [0, 7], np.int32)
    with pytest.raises(NotImplementedError):
        creator.get_shared_dim("col").name = "row"


def test_rename_dim_no_merge_error():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, 3], np.int32)
    creator.add_shared_dim("col", [0, 3], np.int32)
    with pytest.raises(NotImplementedError):
        creator.get_shared_dim("col").name = "row"


def test_rename_dim_attr_name_in_array_exists_error():
    creator = DataspaceCreator()
    creator.add_shared_dim("y1", (0, 4), np.int32)
    creator.add_array_creator("A1", ["y1"])
    creator.add_attr_creator("x1", "A1", np.float64)
    with pytest.raises(ValueError):
        creator.get_shared_dim("y1").name = "x1"


def test_to_schema_bad_array():
    creator = DataspaceCreator()
    creator.add_shared_dim("row", [0, -1], np.int64)
    creator.add_array_creator("A1", ("row",))
    creator.add_attr_creator("x1", "A1", dtype=np.float64)
    with pytest.raises(RuntimeError):
        creator.to_schema()
