import numpy as np
import pytest

import tiledb
from tiledb.cf import GroupSchema, SharedDimension


class TestDataType:



class TestSharedDimension:

    _tiledb_dim = [
        tiledb.Dim(name="dim", domain=(1, 4), tile=4, dtype=np.int32),
    ]

    _bad_name_data = [
        ("bad.name", (1, 4), np.int32),
        (".badname", (1, 4), np.int32),
    ]

    _compare_data = [
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

    @pytest.mark.parametrize("dimension", _tiledb_dim)
    def test_create(self, dimension):
        shared_dimension = SharedDimension.create(dimension)
        assert shared_dimension.name == dimension.name
        assert shared_dimension.domain == dimension.domain
        assert shared_dimension.dtype == dimension.dtype
        assert repr(shared_dimension) is not None

    @pytest.mark.parametrize("name, domain, data_type", _bad_name_data)
    def test_bad_name(self, name, domain, data_type):
        with pytest.raises(ValueError):
            SharedDimension(name, domain, data_type)

    @pytest.mark.parametrize("dimension1, dimension2, is_equal", _compare_data)
    def test_compare_dimensions(self, dimension1, dimension2, is_equal):
        dim1 = SharedDimension.create(dimension1)
        dim2 = SharedDimension.create(dimension2)
        if is_equal:
            assert dim1 == dim2
        else:
            assert dim1 != dim2


class TestGroupSchema:

    _dim0 = tiledb.Dim(name="dim", domain=(0, 0), tile=1, dtype=np.int32)
    _row = tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.uint64)
    _col = tiledb.Dim(name="cols", domain=(1, 8), tile=2, dtype=np.uint64)
    _attr0 = tiledb.Attr(name="attr", dtype=np.int32)
    _attr_a = tiledb.Attr(name="a", dtype=np.uint64)
    _attr_b = tiledb.Attr(name="b", dtype=np.float64)
    _attr_c = tiledb.Attr(name="c", dtype=np.dtype("U"))
    _attr_d = tiledb.Attr(name="d", dtype=np.uint64)
    _empty_array_schema = tiledb.ArraySchema(
        domain=tiledb.Domain(_dim0),
        attrs=[_attr0],
        sparse=False,
    )
    _array_schema_1 = tiledb.ArraySchema(
        domain=tiledb.Domain(_row, _col),
        attrs=[_attr_a, _attr_b, _attr_c],
    )
    _array_schema_2 = tiledb.ArraySchema(
        domain=tiledb.Domain(_row, _col), sparse=True, attrs=[_attr_a]
    )
    _array_schema_3 = tiledb.ArraySchema(domain=tiledb.Domain(_row), attrs=[_attr_b])
    _array_schema_4 = tiledb.ArraySchema(
        domain=tiledb.Domain(_col), attrs=[_attr_b, _attr_d]
    )

    _scenarios = [
        {
            "array_schemas": None,
            "metadata_schema": None,
            "attribute_map": {},
            "num_schemas": 0,
        },
        {
            "array_schemas": [("A1", _array_schema_1)],
            "metadata_schema": _empty_array_schema,
            "attribute_map": {"a": ("A1",), "b": ("A1",), "c": ("A1",)},
            "num_schemas": 1,
        },
        {
            "array_schemas": [
                ("A1", _array_schema_1),
                ("A2", _array_schema_2),
                ("A3", _array_schema_3),
                ("A4", _array_schema_4),
            ],
            "metadata_schema": None,
            "attribute_map": {
                "a": ("A1", "A2"),
                "b": ("A1", "A3", "A4"),
                "c": ("A1",),
                "d": ("A4",),
            },
            "num_schemas": 4,
        },
    ]

    @pytest.mark.parametrize("scenario", _scenarios)
    def test_schema(self, scenario):
        array_schemas = scenario["array_schemas"]
        metadata_schema = scenario["metadata_schema"]
        attribute_map = scenario["attribute_map"]
        group_schema = GroupSchema(array_schemas, metadata_schema)
        group_schema.check()
        assert group_schema.metadata_schema == metadata_schema
        assert repr(group_schema) is not None
        assert len(group_schema) == scenario["num_schemas"]
        for attr_name, arrays in attribute_map.items():
            result = group_schema.get_all_attribute_arrays(attr_name)
            assert (
                result == arrays
            ), f"Get all arrays for attribute '{attr_name}' failed."
            if len(result) == 1:
                assert result[0] == group_schema.get_attribute_array(attr_name)

    def test_set_metadata_array(self):
        """Test setting default metadata schema."""
        group_schema = GroupSchema()
        assert group_schema.metadata_schema is None
        group_schema.set_default_metadata_schema()
        group_schema.metadata_schema.check()
        group_schema.set_default_metadata_schema()

    def test_repeat_name_error(self):
        """Test ValueError is raised when multiple array schemas have the same name."""
        array_schemas = [
            ("dense", self._array_schema_1),
            ("dense", self._array_schema_2),
        ]
        with pytest.raises(ValueError):
            GroupSchema(array_schemas)

    def test_dim_match_error(self):
        """Test ValueError is raised when two schemas have a dimension that doesn't
        match."""
        array_schemas = [
            ("dense1", self._array_schema_1),
            (
                "dense2",
                tiledb.ArraySchema(
                    domain=tiledb.Domain(
                        tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.int64),
                    ),
                    sparse=False,
                    attrs=[self._attr_a],
                ),
            ),
        ]
        with pytest.raises(ValueError):
            GroupSchema(array_schemas)

    def test_no_attr_error(self):
        """Test a KeyError is raised when querying for an attribute that isn't in
        schema"""
        group_schema = GroupSchema(
            [
                ("dense", self._array_schema_1),
            ]
        )
        with pytest.raises(KeyError):
            group_schema.get_attribute_array("missing")

    def test_multi_attr_array_error(self):
        """Test a ValueError is raised when calling `get_attribute_array` for an
        attribute that exists in multiple array schemas."""
        group_schema = GroupSchema(
            [
                ("dense", self._array_schema_1),
                ("sparse", self._array_schema_2),
            ]
        )
        with pytest.raises(ValueError):
            group_schema.get_attribute_array("a")
