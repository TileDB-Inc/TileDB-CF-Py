from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytest

import tiledb
from tiledb.cf import AttrMetadata, DimMetadata, from_netcdf, open_group_array
from tiledb.cf.netcdf_engine import NetCDF4ConverterEngine

netCDF4 = pytest.importorskip("netCDF4")


class ConvertNetCDFBase:
    """Base class for NetCDF converter tests of NetCDF files with a single group.

    Parameters:
        name: Short descriptive name for naming NetCDF file.
        dimension_args: Arguments to use as input for creating NetCDF dimensions.
        variable_kwargs: Keyword argurments to use as input for creating NetCDF
            variables.
        variable_data: Map for NetCDF variable name to variable data.
        variable_metadata: Map from NetCDF variable name to a dictionary of metadata.
        group_metadata: A dictionary of metadata for the NetCDF group.
        attr_to_var_map: Map from TileDB attribute name to NetCDF variable name when
            converting with ``coords_to_dims=False``.
    """

    name = "base"
    dimension_args: Sequence[Tuple[str, Optional[int]]] = []
    variable_kwargs: Sequence[Dict[str, Any]] = []
    variable_data: Dict[str, np.ndarray] = {}
    variable_metadata: Dict[str, Dict[str, Any]] = {}
    group_metadata: Dict[str, Any] = {}
    attr_to_var_map: Dict[str, str] = {}

    @pytest.fixture(scope="class")
    def netcdf_file(self, tmpdir_factory):
        """Returns a NetCDF file created from the group variables.

        Use group variables to create a NetCDF file with a single root group.
        """
        filepath = tmpdir_factory.mktemp("input_file").join(f"{self.name}.nc")
        with netCDF4.Dataset(filepath, mode="w") as dataset:
            if self.group_metadata:
                dataset.setncatts(self.group_metadata)
            for dim_args in self.dimension_args:
                dataset.createDimension(*dim_args)
            for var_kwargs in self.variable_kwargs:
                variable = dataset.createVariable(**var_kwargs)
                if variable.name in self.variable_data:
                    variable[...] = self.variable_data[variable.name]
                if variable.name in self.variable_metadata:
                    variable.setncatts(self.variable_metadata[variable.name])
        return filepath

    def check_attrs(self, group_uri):
        """Checks the values in each attribute match the values from the NetCDF
        variable it was copied from.
        """
        with tiledb.Group(group_uri) as group:
            for attr_name, var_name in self.attr_to_var_map.items():
                with open_group_array(group, attr=attr_name) as array:
                    nonempty_domain = array.nonempty_domain()
                    result = array.multi_index[nonempty_domain]
                np.testing.assert_equal(result[attr_name], self.variable_data[var_name])

    @pytest.mark.parametrize("collect_attrs", [True, False])
    def test_from_netcdf(self, netcdf_file, tmpdir, collect_attrs):
        """Integration test for `from_netcdf_file` function call."""
        uri = str(tmpdir.mkdir("output").join(self.name))
        from_netcdf(netcdf_file, uri, coords_to_dims=False, collect_attrs=collect_attrs)
        self.check_attrs(uri)

    @pytest.mark.parametrize("collect_attrs", [True, False])
    def test_converter_from_netcdf(self, netcdf_file, tmpdir, collect_attrs):
        """Integration test for converting using `from_file` and `convert_to_group`."""
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file, coords_to_dims=False, collect_attrs=collect_attrs
        )
        uri = str(tmpdir.mkdir("output").join(self.name))
        assert isinstance(repr(converter), str)
        converter.convert_to_group(uri)
        self.check_attrs(uri)

    def test_converter_html_repr(self, netcdf_file):
        """Test generated HTML is valid."""
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        try:
            tidylib = pytest.importorskip("tidylib")
            html_summary = converter._repr_html_()
            _, errors = tidylib.tidy_fragment(html_summary)
        except OSError:
            pytest.skip("unable to import libtidy backend")
        assert not bool(errors), str(errors)


class TestConverterSimpleNetCDF(ConvertNetCDFBase):
    """NetCDF conversion test cases for a simple NetCDF file.

    Dimensions:
        row (8)
    Variables:
        float x1(row)
    """

    name = "simple1"
    dimension_args = (("row", 8),)
    variable_kwargs = (
        {"varname": "x1", "datatype": np.float64, "dimensions": ("row",)},
    )
    variable_data = {"x1": np.linspace(1.0, 4.0, 8)}
    attr_to_var_map = {"x1": "x1"}

    def test_convert_dense_with_non_netcdf_dims(self, netcdf_file, tmpdir):
        """Test converting the NetCDF variable 'x1' into a TileDB array with
        an extra non-NetCDF dimension."""
        uri = str(tmpdir.mkdir("output").join("dense_assigned_dim_values"))
        converter = NetCDF4ConverterEngine()
        dim_dtype = np.dtype("uint32")
        converter.add_shared_dim("col", domain=(0, 4), dtype=dim_dtype)
        with netCDF4.Dataset(netcdf_file) as netcdf_group:
            converter.add_dim_to_dim_converter(
                netcdf_group.dimensions["row"], dtype=dim_dtype
            )
            converter.add_array_converter("array", ("row", "col"))
            converter.add_var_to_attr_converter(netcdf_group.variables["x1"], "array")
            converter.convert_to_array(
                uri, input_netcdf_group=netcdf_group, assigned_dim_values={"col": 2}
            )
        with tiledb.open(uri) as array:
            nonempty_domain = array.nonempty_domain()
            data = array[:, 2]
        assert nonempty_domain == ((0, 7), (2, 2))
        tiledb_array = data["x1"]
        original = self.variable_data["x1"]
        np.testing.assert_equal(tiledb_array, original)

    def test_append_to_group(self, netcdf_file, tmpdir):
        """Tests adding the arrays from the converter to an existing group."""
        uri = str(tmpdir.mkdir("output").join("append_example"))
        tiledb.group_create(uri)
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(tiledb.Dim("d0", dtype=np.int32, domain=(0, 4))),
            attrs=[tiledb.Attr("a2", dtype=np.float64)],
        )
        tiledb.Array.create(f"{uri}/original_array", schema)
        with tiledb.Group(uri, mode="w") as group:
            group.add(uri="original_array", name="original_array", relative=True)
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        converter.convert_to_group(uri, append=True)
        with tiledb.Group(uri) as group:
            group_items = {item.name for item in group}
            assert group_items == {"original_array", "array0"}

    def test_append_to_group_bad_name_error(self, netcdf_file, tmpdir):
        """Tests raising error when appending to group with an array name that is in
        the NetCDF4ConverterEngine."""
        uri = str(tmpdir.mkdir("output").join("append_example"))
        tiledb.group_create(uri)
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(tiledb.Dim("d0", dtype=np.int32, domain=(0, 4))),
            attrs=[tiledb.Attr("a2", dtype=np.float64)],
        )
        tiledb.Array.create(f"{uri}/array0", schema)
        with tiledb.Group(uri, mode="w") as group:
            group.add(uri="array0", name="array0", relative=True)
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        with pytest.raises(ValueError):
            converter.convert_to_group(uri, append=True)

    def test_convert_to_sparse_array(self, netcdf_file, tmpdir):
        uri = str(tmpdir.mkdir("output").join("sparse_example"))
        converter = NetCDF4ConverterEngine.from_file(netcdf_file, coords_to_dims=False)
        for array_creator in converter.array_creators():
            array_creator.sparse = True
        converter.convert_to_group(uri)
        with tiledb.Group(uri) as group:
            with open_group_array(group, attr="x1") as array:
                data = array[:]
        index = np.argsort(data["row"])
        x1 = data["x1"][index]
        expected = np.linspace(1.0, 4.0, 8)
        np.testing.assert_equal(x1, expected)

    def test_no_collect_tiles_by_var(self, netcdf_file):
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file,
            coords_to_dims=False,
            collect_attrs=False,
            tiles_by_var={"x1": (2,)},
        )
        group_schema = converter.to_schema()
        tiles = tuple(dim.tile for dim in group_schema["x1"].domain)
        assert tiles == (2,)

    def test_no_collect_tiles_by_dims(self, netcdf_file):
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file,
            coords_to_dims=False,
            collect_attrs=False,
            tiles_by_dims={("row",): (2,)},
        )
        group_schema = converter.to_schema()
        tiles = tuple(dim.tile for dim in group_schema["x1"].domain)
        assert tiles == (2,)

    @pytest.mark.parametrize("collect_attrs", (True, False))
    def test_offsets_filter(self, netcdf_file, collect_attrs):
        """Tests all arrays have offsets_filters set to desired value."""
        filters = tiledb.FilterList([tiledb.ZstdFilter(7)])
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file, collect_attrs=collect_attrs, offsets_filters=filters
        )
        for array_creator in converter.array_creators():
            assert array_creator.offsets_filters == filters

    @pytest.mark.parametrize("collect_attrs", (True, False))
    def test_attrs_filter(self, netcdf_file, collect_attrs):
        """Tests all arrays have attrs_filters set to desired value."""
        filters = tiledb.FilterList([tiledb.ZstdFilter(7)])
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file, collect_attrs=collect_attrs, attrs_filters=filters
        )
        for array_creator in converter.array_creators():
            for attr_creator in array_creator:
                assert attr_creator.filters == filters

    def test_rename_array(self, netcdf_file):
        converter = NetCDF4ConverterEngine.from_file(netcdf_file, coords_to_dims=False)
        converter.get_array_creator("array0").name = "A1"
        names = {array_creator.name for array_creator in converter.array_creators()}
        assert names == set(["A1"])

    def test_rename_attr(self, netcdf_file):
        converter = NetCDF4ConverterEngine.from_file(netcdf_file, coords_to_dims=False)
        converter.get_array_creator_by_attr("x1").attr_creator("x1").name = "y1"
        attr_names = {
            attr_creator.name for attr_creator in next(converter.array_creators())
        }
        assert attr_names == set(["y1"])

    def test_rename_dim(self, netcdf_file):
        converter = NetCDF4ConverterEngine.from_file(netcdf_file, coords_to_dims=False)
        converter.get_shared_dim("row").name = "col"
        dim_names = {shared_dim.name for shared_dim in converter.shared_dims()}
        assert dim_names == set(["col"])

    def test_non_netcdf_attr(self, netcdf_file, tmpdir):
        """Tests converting a NetCDF file with an external attribute."""
        converter = NetCDF4ConverterEngine.from_file(netcdf_file, coords_to_dims=False)
        converter.add_attr_creator("a1", "array0", np.float64)
        a1_data = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
        uri = str(tmpdir.mkdir("non-netcdf-attr-test"))
        converter.convert_to_group(uri, assigned_attr_values={"a1": a1_data})
        with tiledb.Group(uri) as group:
            with open_group_array(group, attr="a1") as array:
                result = array[:]
        np.testing.assert_equal(a1_data, result)

    def test_non_netcdf_attr_missing_data_error(self, netcdf_file, tmpdir):
        """Tests error converting a NetCDF file with an external attribute when data is
        missing."""
        converter = NetCDF4ConverterEngine.from_file(netcdf_file, coords_to_dims=False)
        converter.add_attr_creator("a1", "array0", np.float64)
        uri = str(tmpdir.mkdir("non-netcdf-attr-test"))
        with pytest.raises(KeyError):
            converter.convert_to_group(uri)

    def test_non_netcdf_attr_bad_shape_error(self, netcdf_file, tmpdir):
        """Tests error converting a NetCDF file with an external attribue when the size
        of the provided data is mismatched."""
        converter = NetCDF4ConverterEngine.from_file(netcdf_file, coords_to_dims=False)
        converter.add_attr_creator("a1", "array0", np.float64)
        a1_data = np.linspace(-1.0, 1.0, 4, dtype=np.float64)
        uri = str(tmpdir.mkdir("non-netcdf-attr-test"))
        with pytest.raises(tiledb.libtiledb.TileDBError):
            converter.convert_to_group(uri, assigned_attr_values={"a1": a1_data})

    def test_inject_dim_creator(self, netcdf_file):
        """Tests injecting a dimension into a NetCDF4ConverterArray."""
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        converter.add_shared_dim("dim0", domain=(0, 4), dtype=np.uint32)
        array_converter = converter.get_array_creator("array0")
        array_converter.domain_creator.inject_dim_creator("dim0", 0)
        dim_names = tuple(
            dim_creator.name for dim_creator in array_converter.domain_creator
        )
        assert dim_names == ("dim0", "row")

    def test_inject_dim_mismatch_attr_dims_error(self, netcdf_file):
        """Tests injecting a dimension into a NetCDF4ConverterArray."""
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        with netCDF4.Dataset("inmemory", mode="w", diskless=True) as dataset:
            dim = dataset.createDimension("dim0", 10)
            converter.add_dim_to_dim_converter(dim)
        array_converter = converter.get_array_creator("array0")
        with pytest.raises(ValueError):
            array_converter.domain_creator.inject_dim_creator("dim0", 0)

    def test_remove_dim_needed_by_attr_error(self, netcdf_file):
        """Tests raising an error when attempting to remove a dimension in a
        NetCDF4ConverterArray that is needed by an AttrCreator."""
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        array_converter = converter.get_array_creator("array0")
        with pytest.raises(ValueError):
            array_converter.domain_creator.remove_dim_creator("row")

    def test_bad_array_name_error(self, netcdf_file):
        converter = NetCDF4ConverterEngine.from_file(netcdf_file, coords_to_dims=False)
        with pytest.raises(ValueError):
            converter.add_array_converter("array0", tuple())


class TestConvertNetCDFSimpleCoord1(ConvertNetCDFBase):
    """NetCDF conversion test cases for a NetCDF file with a coordinate variable.

    Dimensions:
        x (4)
    Variables:
        real x (x)
        real y (x)
    """

    name = "simple_coord_1"
    group_metadata = {"name": name}
    dimension_args = (("x", 4),)
    variable_kwargs = (
        {"varname": "x", "datatype": np.float64, "dimensions": ("x",)},
        {"varname": "y", "datatype": np.float64, "dimensions": ("x",)},
    )
    variable_data = {
        "x": np.array([2.0, 5.0, -1.0, 4.0]),
        "y": np.array([4.0, 25.0, 1.0, 16.0]),
    }
    variable_metadata = {
        "x": {"description": "x array"},
        "y": {"description": "y array"},
    }
    attr_to_var_map = {"x.data": "x", "y": "y"}

    @pytest.mark.parametrize("collect_attrs", [True, False])
    def test_convert_coordinate(self, netcdf_file, tmpdir, collect_attrs):
        uri = str(tmpdir.mkdir("output").join("coordinate_example"))
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file,
            coords_to_dims=True,
            collect_attrs=collect_attrs,
        )
        shared_x = converter.get_shared_dim("x")
        shared_x.domain = (None, None)
        converter.convert_to_group(uri)
        with tiledb.Group(uri) as group:
            with open_group_array(group, attr="y") as array:
                schema = array.schema
                assert schema.sparse
                data = array[:]
                attr_meta = AttrMetadata(array.meta, "y")
                assert (
                    attr_meta["description"] == "y array"
                ), "attribute metadata not correctly copied."
                dim_meta = DimMetadata(array.meta, "x")
                assert (
                    dim_meta["description"] == "x array"
                ), "dim metadata not correctly copied."
                assert group.meta["name"] == self.name
        index = np.argsort(data["x"])
        x = data["x"][index]
        y = data["y"][index]
        np.testing.assert_equal(x, np.array([-1.0, 2.0, 4.0, 5.0]))
        np.testing.assert_equal(y, np.array([1.0, 4.0, 16.0, 25.0]))

    def test_convert_to_array(self, netcdf_file, tmpdir):
        uri = str(tmpdir.mkdir("output").join("array_example"))
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file,
            coords_to_dims=True,
            collect_attrs=True,
        )
        shared_x = converter.get_shared_dim("x")
        shared_x.domain = (None, None)
        converter.convert_to_array(uri)
        with tiledb.open(uri, attr="y") as array:
            schema = array.schema
            assert schema.sparse
            data = array[:]
            metadata_name = array.meta["name"]
            assert metadata_name == self.name
        index = np.argsort(data["x"])
        x = data["x"][index]
        y = data["y"][index]
        np.testing.assert_equal(x, np.array([-1.0, 2.0, 4.0, 5.0]))
        np.testing.assert_equal(y, np.array([1.0, 4.0, 16.0, 25.0]))

    @pytest.mark.parametrize(
        "collect_attrs, array_name", [(True, "array0"), (False, "y")]
    )
    def test_coordinate_tiles_by_var(self, netcdf_file, collect_attrs, array_name):
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file,
            coords_to_dims=True,
            collect_attrs=collect_attrs,
            tiles_by_var={"y": (100.0,)},
        )
        domain_creator = converter.get_array_creator(array_name).domain_creator
        tile = domain_creator.dim_creator(0).tile
        assert tile == 100.0

    @pytest.mark.parametrize(
        "collect_attrs, array_name", [(True, "array0"), (False, "y")]
    )
    def test_coordinate_tiles_by_dims(self, netcdf_file, collect_attrs, array_name):
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file,
            coords_to_dims=True,
            collect_attrs=collect_attrs,
            tiles_by_dims={("x",): (100.0,)},
        )
        domain_creator = converter.get_array_creator(array_name).domain_creator
        tile = domain_creator.dim_creator(0).tile
        assert tile == 100.0


class TestConvertNetCDFMultiCoords(ConvertNetCDFBase):
    """NetCDF conversion test cases for a NetCDF file with a coordinate variable.

    Dimensions:
        x (4)
    Variables:
        real x (x)
        real y (y)
    """

    name = "multicoords"
    dimension_args = (("x", 2), ("y", 2))
    variable_kwargs = (
        {"varname": "x", "datatype": np.float64, "dimensions": ("x",)},
        {"varname": "y", "datatype": np.float64, "dimensions": ("y",)},
        {"varname": "z", "datatype": np.float64, "dimensions": ("x", "y")},
    )
    variable_data = {
        "x": np.array([2.0, 5.0]),
        "y": np.array([-1.0, 4.0]),
        "z": np.array([[4.0, 25.0], [1.0, 16.0]]),
    }
    attr_to_var_map = {"x.data": "x", "y.data": "y", "z": "z"}

    @pytest.mark.parametrize("collect_attrs", [True, False])
    def test_convert_coordinate(self, netcdf_file, tmpdir, collect_attrs):
        uri = str(tmpdir.mkdir("output").join("coordinate_example"))
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file,
            coords_to_dims=True,
            collect_attrs=collect_attrs,
        )
        shared_x = converter.get_shared_dim("x")
        shared_x.domain = (None, None)
        shared_y = converter.get_shared_dim("y")
        shared_y.domain = (None, None)
        converter.convert_to_group(uri)
        with tiledb.Group(uri) as group:
            with open_group_array(group, attr="z") as array:
                schema = array.schema
                assert schema.sparse
                data = array[:]
        result = tuple(zip(data["x"], data["y"], data["z"]))
        expected = (
            (2.0, -1.0, 4.0),
            (2.0, 4.0, 25.0),
            (5.0, -1.0, 1.0),
            (5.0, 4.0, 16.0),
        )
        assert result == expected


class TestConvertNetCDFCoordWithTiles(ConvertNetCDFBase):
    """NetCDF conversion test cases for a NetCDF file with a coordinate variable.

    Dimensions:
        index (4)
    Variables:
        int index (index)
        real y (index)
    """

    name = "coord_with_chunks"
    dimension_args = (("index", 4),)
    variable_kwargs = (
        {
            "varname": "index",
            "datatype": np.int32,
            "dimensions": ("index",),
        },
        {
            "varname": "y",
            "datatype": np.float64,
            "dimensions": ("index",),
            "chunksizes": (4,),
        },
    )
    variable_data = {
        "index": np.array([1, 2, 3, 4]),
        "y": np.array([4.0, 25.0, 1.0, 16.0]),
    }
    attr_to_var_map = {"index.data": "index", "y": "y"}

    @pytest.mark.parametrize("collect_attrs", [True, False])
    def test_convert_coordinate(self, netcdf_file, tmpdir, collect_attrs):
        uri = str(tmpdir.mkdir("output").join("coordinate_example"))
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file,
            coords_to_dims=True,
            collect_attrs=collect_attrs,
        )
        index_dim = converter.get_shared_dim("index")
        index_dim.domain = (1, 4)
        converter.convert_to_group(uri)
        with tiledb.Group(uri) as group:
            with open_group_array(group, attr="y") as array:
                schema = array.schema
                assert schema.sparse
                data = array[:]
        index_order = np.argsort(data["index"])
        index = data["index"][index_order]
        y = data["y"][index_order]
        np.testing.assert_equal(index, self.variable_data["index"])
        np.testing.assert_equal(y, self.variable_data["y"])


class TestConvertNetCDFUnlimitedDim(ConvertNetCDFBase):
    """NetCDF conversion test cases for a NetCDF file with an unlimited dimension.

    Dimensions:
        row (None)
        col (4)
    Variables:
        uint16 x (row)
        uint16 y (col)
        uint16 data (row, col)
    """

    name = "simple_unlim_dim"
    dimension_args = (("row", None), ("col", 4))
    variable_kwargs = [
        {
            "varname": "data",
            "datatype": np.dtype("uint16"),
            "dimensions": ("row", "col"),
        },
        {"varname": "x", "datatype": np.dtype("uint16"), "dimensions": ("row",)},
        {"varname": "y", "datatype": np.dtype("uint16"), "dimensions": ("col",)},
    ]
    variable_data = {
        "data": np.array(
            ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16])
        ),
        "x": np.array([1, 2, 3, 4]),
        "y": np.array([5, 6, 7, 8]),
    }
    attr_to_var_map = {"data": "data", "x": "x", "y": "y"}

    def test_convert_dense_with_non_netcdf_dims(self, netcdf_file, tmpdir):
        """Test converting the NetCDF variable 'x1' into a TileDB array with
        an extra non-NetCDF dimension."""
        uri = str(tmpdir.mkdir("output").join("dense_assigned_dim_values"))
        converter = NetCDF4ConverterEngine()
        dim_dtype = np.dtype("uint32")
        converter.add_shared_dim("extra", domain=(0, 4), dtype=dim_dtype)
        with netCDF4.Dataset(netcdf_file) as netcdf_group:
            converter.add_dim_to_dim_converter(
                netcdf_group.dimensions["row"], dtype=dim_dtype
            )
            converter.add_dim_to_dim_converter(
                netcdf_group.dimensions["col"], dtype=dim_dtype
            )
            converter.add_array_converter("array", ("row", "extra", "col"))
            converter.add_var_to_attr_converter(netcdf_group.variables["data"], "array")
            converter.convert_to_array(
                uri, input_netcdf_group=netcdf_group, assigned_dim_values={"extra": 2}
            )
        with tiledb.open(uri) as array:
            nonempty_domain = array.nonempty_domain()
            data = array[:, 2, :]
        assert nonempty_domain == ((0, 3), (2, 2), (0, 3))
        tiledb_array = data["data"]
        original = self.variable_data["data"]
        np.testing.assert_equal(tiledb_array, original)

    def test_convert_sparse_with_non_netcdf_dims(self, netcdf_file, tmpdir):
        """Test converting the NetCDF variable 'x1' into a TileDB array with
        an extra non-NetCDF dimension."""
        uri = str(tmpdir.mkdir("output").join("dense_assigned_dim_values"))
        converter = NetCDF4ConverterEngine()
        dim_dtype = np.dtype("uint32")
        converter.add_shared_dim("extra", domain=(0, 4), dtype=dim_dtype)
        with netCDF4.Dataset(netcdf_file) as netcdf_group:
            converter.add_dim_to_dim_converter(
                netcdf_group.dimensions["row"], dtype=dim_dtype
            )
            converter.add_dim_to_dim_converter(
                netcdf_group.dimensions["col"], dtype=dim_dtype
            )
            converter.add_array_converter("array", ("row", "extra", "col"), sparse=True)
            converter.add_var_to_attr_converter(netcdf_group.variables["data"], "array")
            converter.convert_to_array(
                uri, input_netcdf_group=netcdf_group, assigned_dim_values={"extra": 2}
            )
        with tiledb.open(uri) as array:
            nonempty_domain = array.nonempty_domain()
            data = array[:, 2, :]
        assert nonempty_domain == ((0, 3), (2, 2), (0, 3))
        tiledb_array = data["data"]
        original = self.variable_data["data"].reshape(-1)
        np.testing.assert_equal(tiledb_array, original)

    def test_copy_missing_dim(self, netcdf_file, tmpdir):
        """Test converting the NetCDF variable 'x1' into a TileDB array with
        an extra non-NetCDF dimension."""
        uri = str(tmpdir.mkdir("output").join("dense_assigned_dim_values"))
        converter = NetCDF4ConverterEngine()
        dim_dtype = np.dtype("uint32")
        converter.add_shared_dim("extra", domain=(0, 4), dtype=dim_dtype)
        with netCDF4.Dataset(netcdf_file) as netcdf_group:
            converter.add_dim_to_dim_converter(
                netcdf_group.dimensions["row"], dtype=dim_dtype
            )
            converter.add_dim_to_dim_converter(
                netcdf_group.dimensions["col"], dtype=dim_dtype
            )
            converter.add_array_converter("array", ("row", "extra", "col"))
            converter.add_var_to_attr_converter(netcdf_group.variables["data"], "array")
            with pytest.raises(KeyError):
                converter.convert_to_array(uri, input_netcdf_group=netcdf_group)


class TestConvertUnpackVariables(ConvertNetCDFBase):
    """NetCDF conversion test cases for a packed NetCDF variable.

    Dimensions:
        x (4)
    Variables:
        real x (x)
        real y (x)
    """

    name = "simple_coord_1"
    group_metadata = {"name": name}
    dimension_args = (("x", 4),)
    variable_kwargs = (
        {"varname": "x", "datatype": np.int32, "dimensions": ("x",)},
        {"varname": "y", "datatype": np.float64, "dimensions": ("x",)},
    )
    variable_data = {
        "x": np.array([1, 2, 3, 4]),
        "y": np.array([0.0, 0.5, 1.0, 1.5]),
    }
    variable_metadata = {
        "x": {"scale_factor": np.float64(0.5), "add_offset": np.float64(1.0)},
    }
    attr_to_var_map = {"x.data": "x", "y": "y"}

    @pytest.mark.parametrize("collect_attrs", [True, False])
    def test_unpack_to_attr(self, netcdf_file, tmpdir, collect_attrs):
        """Tests NetCDF variable is correctly unpacked when mapping to a TileDB
        attribute."""
        uri = str(tmpdir.mkdir("output").join("coordinate_example"))
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file,
            collect_attrs=collect_attrs,
            unpack_vars=True,
        )
        converter.convert_to_group(uri)
        with tiledb.Group(uri) as group:
            with open_group_array(group, attr="x.data") as array:
                x = array[:]
                meta = AttrMetadata(array.meta, "x.data")
                assert "scale_factor" not in meta
                assert "add_offset" not in meta
        assert x.dtype == np.float64
        np.testing.assert_equal(x, np.array([1.5, 2.0, 2.5, 3.0]))

    @pytest.mark.parametrize("collect_attrs", [True, False])
    def test_unpack_to_dim(self, netcdf_file, tmpdir, collect_attrs):
        """Tests NetCDF variable is correctly unpacked when mapping to a TileDB
        dimension."""
        uri = str(tmpdir.mkdir("output").join("coordinate_example"))
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file,
            coords_to_dims=True,
            collect_attrs=collect_attrs,
            unpack_vars=True,
        )
        shared_x = converter.get_shared_dim("x")
        shared_x.domain = (-10.0, 10.0)
        shared_x.tile = 2.0
        converter.convert_to_group(uri)
        with tiledb.Group(uri) as group:
            with open_group_array(group, attr="y") as array:
                data = array[:]
                meta = DimMetadata(array.meta, "x")
                assert "scale_factor" not in meta
                assert "add_offset" not in meta
        x = data["x"]
        y = data["y"]
        assert x.dtype == np.float64
        np.testing.assert_equal(x, np.array([1.5, 2.0, 2.5, 3.0]))
        np.testing.assert_equal(y, np.array([0.0, 0.5, 1.0, 1.5]))


class TestConvertNetCDFMultipleScalarVariables(ConvertNetCDFBase):
    """NetCDF conversion test cases for NetCDF with multiple scalar variables.

    Variables:
        int32 x () = [1]
        int32 y () = [5]
    """

    name = "scalar_variables"
    variable_kwargs = [
        {"varname": "x", "datatype": np.dtype("int32")},
        {"varname": "y", "datatype": np.dtype("int32")},
    ]
    variable_data = {
        "x": np.array([1]),
        "y": np.array([5]),
    }
    attr_to_var_map = {"x": "x", "y": "y"}

    def test_sparse_array(self, netcdf_file, tmpdir):
        uri = str(tmpdir.mkdir("output").join("sparse_scalar_example"))
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file, coords_to_dims=False, collect_attrs=False
        )
        for array_creator in converter.array_creators():
            array_creator.sparse = True
        converter.convert_to_group(uri)
        with tiledb.Group(uri) as group:
            with open_group_array(group, array="scalars") as array:
                data = array[0]
        np.testing.assert_equal(data["x"], self.variable_data["x"])
        np.testing.assert_equal(data["y"], self.variable_data["y"])

    def test_scalar_assigned_dim_value(self, netcdf_file, tmpdir):
        """Tests setting the value for a scalar dimension when converting NetCDF
        scalars to TileDB."""
        uri = str(tmpdir.mkdir("output").join("scalar_assign_value_example"))
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        scalar_dim = converter.get_shared_dim("__scalars")
        scalar_dim.name = "dim0"
        scalar_dim.domain = (0, 10)
        converter.convert_to_group(uri, assigned_dim_values={"dim0": 1})
        with tiledb.Group(uri) as group:
            with open_group_array(group, array="array0") as array:
                data = array[1:2]
        np.testing.assert_equal(data["x"], self.variable_data["x"])
        np.testing.assert_equal(data["y"], self.variable_data["y"])

    def test_inject_dim(self, netcdf_file, tmpdir):
        """Tests converting NetCDF scalar dimensions to TileDB with an extra dimension
        added."""
        uri = str(tmpdir.mkdir("output").join("scalar_extra_dim_example"))
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        converter.add_shared_dim(dim_name="t", domain=(1, 8), dtype=np.dtype("uint64"))
        array_creator = converter.get_array_creator("array0")
        array_creator.domain_creator.inject_dim_creator("t", 1)
        converter.convert_to_group(uri, assigned_dim_values={"t": 1})
        with tiledb.Group(uri) as group:
            with open_group_array(group, array="array0") as array:
                data = array[:, 1]
        np.testing.assert_equal(data["x"], self.variable_data["x"])
        np.testing.assert_equal(data["y"], self.variable_data["y"])


class TestConvertNetCDFMatchingChunks(ConvertNetCDFBase):
    """NetCDF conversion test cases for a NetCDF file with two variables over the
    same diemnsions with the same chunksizes.

    Dimensions:
        row (8)
        col (8)
    Variables:
        int32 x1(row, col) with chunksizes (4, 4)
        int32 x2(row, col) with chunksizes (4, 4)
    """

    name = "matching_chunks"
    dimension_args = [("row", 8), ("col", 8)]
    variable_kwargs = [
        {
            "varname": "x1",
            "datatype": np.int32,
            "dimensions": ("row", "col"),
            "chunksizes": (4, 4),
        },
        {
            "varname": "x2",
            "datatype": np.int32,
            "dimensions": ("row", "col"),
            "chunksizes": (4, 4),
        },
    ]
    variable_data = {
        "x1": np.arange(64).reshape(8, 8),
        "x2": np.arange(64, 128).reshape(8, 8),
    }
    attr_to_var_map = {"x1": "x1", "x2": "x2"}

    def test_tile(self, netcdf_file):
        converter = NetCDF4ConverterEngine.from_file(netcdf_file, coords_to_dims=False)
        group_schema = converter.to_schema()
        tiles = tuple(dim.tile for dim in group_schema["array0"].domain)
        assert tiles == (4, 4)

    def test_collect_attrs_tiles_by_dims(self, netcdf_file):
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file,
            tiles_by_dims={("row", "col"): (2, 4)},
            coords_to_dims=False,
        )
        group_schema = converter.to_schema()
        tiles = tuple(dim.tile for dim in group_schema["array0"].domain)
        assert tiles == (2, 4)


class TestConvertNetCDFMismatchingChunks(ConvertNetCDFBase):
    """NetCDF conversion test cases for a NetCDF file with two variables over the same
    dimensions and different chunksizes.

    Dimensions:
        row (8)
        col (8)
    Variables:
        int32 x1 (row, col) with chunksizes (4, 4)
        int32 x2 (row, col) with chunksizes (2, 2)
    """

    name = "mismatching_chunks"
    dimension_args = (("row", 8), ("col", 8))
    variable_kwargs = [
        {
            "varname": "x1",
            "datatype": np.int32,
            "dimensions": ("row", "col"),
            "chunksizes": (4, 4),
        },
        {
            "varname": "x2",
            "datatype": np.int32,
            "dimensions": ("row", "col"),
            "chunksizes": (2, 2),
        },
    ]
    variable_data = {
        "x1": np.arange(64).reshape(8, 8),
        "x2": np.arange(64, 128).reshape(8, 8),
    }
    attr_to_var_map = {"x1": "x1", "x2": "x2"}

    def test_tile(self, netcdf_file):
        converter = NetCDF4ConverterEngine.from_file(netcdf_file, coords_to_dims=False)
        group_schema = converter.to_schema()
        tiles = tuple(dim.tile for dim in group_schema["array0"].domain)
        assert tiles == (8, 8)

    @pytest.mark.parametrize("collect_attrs", [True, False])
    def test_convert_sparse_arrays(self, tmpdir, netcdf_file, collect_attrs):
        uri = str(tmpdir.mkdir("output").join("sparse_multidim_example"))
        converter = NetCDF4ConverterEngine.from_file(
            netcdf_file,
            coords_to_dims=False,
        )
        for array_creator in converter.array_creators():
            array_creator.sparse = True
        converter.convert_to_group(uri)
        with tiledb.Group(uri) as group:
            with open_group_array(group, attr="x1") as array:
                x1_result = array[:, :]["x1"]
            x1_expected = np.arange(64, dtype=np.int32)
            np.testing.assert_equal(x1_result, x1_expected)
            with open_group_array(group, attr="x2") as array:
                x2_result = array[:, :]["x2"]
        x2_expected = np.arange(64, 128, dtype=np.int32)
        np.testing.assert_equal(x2_result, x2_expected)


class TestConvertNetCDFSingleVariableChunk(ConvertNetCDFBase):
    """NetCDF conversion test cases for a NetCDF file with two variables: one with the
    chunksize defined and the other with no chunksize specified.

    Dimensions:
        row (8)
        col (8)
    Variables:
        int32 x1 (row, col) with chunksize=(4,4)
        int32 x2 (row, col)
    """

    name = "single_chunk_variable"
    dimension_args = (("row", 8), ("col", 8))
    variable_kwargs = (
        {
            "varname": "x1",
            "datatype": np.int32,
            "dimensions": ("row", "col"),
            "chunksizes": (4, 4),
        },
        {
            "varname": "x2",
            "datatype": np.int32,
            "dimensions": ("row", "col"),
        },
    )
    variable_data = {
        "x1": np.arange(64).reshape(8, 8),
        "x2": np.arange(64, 128).reshape(8, 8),
    }
    attr_to_var_map = {"x1": "x1", "x2": "x2"}

    def test_tile_from_single_variable_chunks(self, netcdf_file):
        converter = NetCDF4ConverterEngine.from_file(netcdf_file, coords_to_dims=False)
        group_schema = converter.to_schema()
        tiles = tuple(dim.tile for dim in group_schema["array0"].domain)
        assert tiles == (4, 4)


class TestConverterNetCDFVariabelWithFill(ConvertNetCDFBase):
    """NetCDF conversion test cases for NetCDF variables with explicitly set fill
    value.

    Dimensions:
        t (4)
    Variables:
        int64 x(t) with fill = -1
        float64 scalar with fill = 0.0
    """

    name = "test_fill_values"
    dimension_args = (("t", 4),)
    variable_kwargs = (
        {"varname": "x", "datatype": np.int64, "dimensions": ("t",), "fill_value": -1},
        {
            "varname": "a",
            "datatype": np.float64,
            "dimensions": tuple(),
            "fill_value": 0.0,
        },
    )
    variable_data = {"x": np.array((-1, 2, -1, -1)), "a": np.array([0.0])}
    attr_to_var_map = {"x": "x", "a": "a"}

    @pytest.mark.parametrize("sparse", [True, False])
    def test_change_fill(self, netcdf_file, tmpdir, sparse):
        """Test changing the fill value for a standard NetCDF variable."""
        uri = str(tmpdir.mkdir("output").join(self.name))
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        x_array = converter.get_array_creator_by_attr("x")
        x_array.sparse = sparse
        x_array.attr_creator("x").fill = 0
        converter.convert_to_group(uri)
        with tiledb.Group(uri) as group:
            with open_group_array(group, attr="x") as array:
                result = array.multi_index[:]["x"]
                expected = np.array((0, 2, 0, 0))
                np.testing.assert_equal(result, expected)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_change_fill_scalar(self, netcdf_file, tmpdir, sparse):
        """Test changing the fill value for a NetCDF scalar variable."""
        uri = str(tmpdir.mkdir("output").join(self.name))
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        # Test change fill for standard variable
        scalar_array = converter.get_array_creator_by_attr("a")
        scalar_array.sparse = sparse
        scalar_array.attr_creator("a").fill = np.nan
        converter.convert_to_group(uri)
        with tiledb.Group(uri) as group:
            with open_group_array(group, attr="a") as array:
                result = array.multi_index[:]["a"]
                expected = np.array([np.nan])
                np.testing.assert_equal(result, expected)

    def test_fill_value_not_in_metadata(self, netcdf_file, tmpdir):
        """Test that NetCDF attribute `_FillValue` is not copied to TileDB."""
        uri = str(tmpdir.mkdir("output").join(self.name))
        converter = NetCDF4ConverterEngine.from_file(netcdf_file)
        converter.convert_to_group(uri)
        with tiledb.Group(uri) as group:
            with open_group_array(group, attr="x") as array:
                assert "_FillValue" not in array.meta


def test_group_metadata(tmpdir):
    filepath = str(tmpdir.mkdir("data").join("test_group_metadata.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dataset.setncattr("name", "Group metadata example")
        dataset.setncattr("array", [0.0, 1.0, 2.0])
    uri = str(tmpdir.mkdir("output").join("test_group_metadata"))
    from_netcdf(filepath, uri, coords_to_dims=False)
    with tiledb.Group(uri) as group:
        assert group.meta["name"] == "Group metadata example"
        assert group.meta["array"] == (0.0, 1.0, 2.0)


def test_group_metadata_no_copy(tmpdir):
    filepath = str(tmpdir.mkdir("data").join("test_group_metadata.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dataset.setncattr("name", "Group metadata example")
        dataset.setncattr("array", [0.0, 1.0, 2.0])
    uri = str(tmpdir.mkdir("output").join("test_group_metadata"))
    from_netcdf(filepath, uri, coords_to_dims=False, copy_metadata=False)
    with tiledb.Group(uri) as group:
        assert "name" not in group.meta
        assert "array" not in group.meta


def test_variable_metadata(tmpdir):
    filepath = str(tmpdir.mkdir("data").join("test_variable_metadata.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dataset.createDimension("row", 4)
        variable = dataset.createVariable("x1", np.float64, ("row",))
        variable[:] = np.array([1.0, 2.0, 3.0, 4.0])
        variable.setncattr("fullname", "Example variable")
        variable.setncattr("array", [1, 2])
        variable.setncattr("singleton", [1.0])
    uri = str(tmpdir.mkdir("output").join("test_variable_metadata"))
    from_netcdf(filepath, uri, coords_to_dims=False)
    with tiledb.Group(uri) as group:
        with open_group_array(group, attr="x1") as array:
            attr_meta = AttrMetadata(array.meta, "x1")
            assert attr_meta is not None
            assert attr_meta["fullname"] == "Example variable"
            assert attr_meta["array"] == (1, 2)
            assert attr_meta["singleton"] == 1.0


def test_variable_metadata_no_copy(tmpdir):
    filepath = str(tmpdir.mkdir("data").join("test_variable_metadata.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dataset.createDimension("row", 4)
        variable = dataset.createVariable("x1", np.float64, ("row",))
        variable[:] = np.array([1.0, 2.0, 3.0, 4.0])
        variable.setncattr("fullname", "Example variable")
        variable.setncattr("array", [1, 2])
        variable.setncattr("singleton", [1.0])
    uri = str(tmpdir.mkdir("output").join("test_variable_metadata"))
    from_netcdf(filepath, uri, coords_to_dims=False, copy_metadata=False)
    with tiledb.Group(uri) as group:
        with open_group_array(group, attr="x1") as array:
            attr_meta = AttrMetadata(array.meta, "x1")
            assert attr_meta is not None
            assert "fullname" not in attr_meta
            assert "array" not in attr_meta
            assert "singleton" not in attr_meta


def test_nested_groups(tmpdir, group1_netcdf_file):
    root_uri = str(tmpdir.mkdir("output").join("test_example_group1"))
    from_netcdf(group1_netcdf_file, root_uri, coords_to_dims=False)
    x = np.linspace(-1.0, 1.0, 8)
    y = np.linspace(-1.0, 1.0, 4)
    # Test root
    with tiledb.Group(root_uri) as group:
        with open_group_array(group, attr="x1") as array:
            x1 = array[:]
    np.testing.assert_equal(x1, x)
    # Test group 1
    with tiledb.Group(root_uri + "/group1") as group:
        with open_group_array(group, attr="x2") as array:
            x2 = array[:]
    np.testing.assert_equal(x2, 2.0 * x)
    # Test group 2
    with tiledb.Group(root_uri + "/group1/group2") as group:
        with open_group_array(group, attr="y1") as array:
            y1 = array[:]
    np.testing.assert_equal(y1, y)
    # Test group 3
    with tiledb.open(root_uri + "/group3/array0") as array:
        array0 = array[:, :]
        A1 = array0["A1"]
        A2 = array0["A2"]
        A3 = array0["A3"]
    np.testing.assert_equal(A1, np.outer(y, y))
    np.testing.assert_equal(A2, np.zeros((4, 4), dtype=np.float64))
    np.testing.assert_equal(A3, np.identity(4, dtype=np.int32))


def test_variable_fill(tmpdir):
    """Test converting a NetCDF variable with the _FillValue NetCDF attribute set."""
    filepath = str(tmpdir.mkdir("sample_netcdf").join("test_fill.nc"))
    with netCDF4.Dataset(filepath, mode="w") as dataset:
        dataset.createDimension("row", 4)
        dataset.createVariable("x1", np.dtype("int64"), ("row",), fill_value=-1)
        converter = NetCDF4ConverterEngine.from_group(dataset, coords_to_dims=False)
        attr_creator = converter._core.get_attr_creator("x1")
        assert attr_creator.fill == -1


def test_collect_attrs_tile_by_var(simple2_netcdf_file):
    converter = NetCDF4ConverterEngine.from_file(
        simple2_netcdf_file.filepath,
        tiles_by_var={"x1": (4,)},
        coords_to_dims=False,
    )
    group_schema = converter.to_schema()
    tiles = tuple(dim.tile for dim in group_schema["array0"].domain)
    assert tiles == (4,)


def test_copy_no_var_error(tmpdir, simple1_netcdf_file, simple2_netcdf_file):
    converter = NetCDF4ConverterEngine.from_file(
        simple2_netcdf_file.filepath,
        coords_to_dims=False,
    )
    print(converter)
    uri = str(tmpdir.mkdir("output").join("test_copy_error"))
    converter.create_group(uri)
    with pytest.raises(KeyError):
        converter.copy_to_group(uri, input_file=simple1_netcdf_file.filepath)


def test_mismatched_netcdf_dims():
    with netCDF4.Dataset("example.nc", mode="w", diskless=True) as dataset:
        x_dim = dataset.createDimension("x")
        y_dim = dataset.createDimension("y")
        var = dataset.createVariable("value", np.float64, ("y", "x"))
        converter = NetCDF4ConverterEngine()
        converter.add_dim_to_dim_converter(x_dim)
        converter.add_dim_to_dim_converter(y_dim)
        converter.add_array_converter("array", ("x", "y"))
        with pytest.raises(ValueError):
            converter.add_var_to_attr_converter(array_name="array", ncvar=var)
