# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Classes for converting NetCDF4 files to TileDB."""

import os
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import netCDF4
import numpy as np

import tiledb
from tiledb.cf.core import METADATA_ARRAY_NAME, DType
from tiledb.cf.creator import DataspaceCreator

from ._array_converters import NetCDF4ArrayConverter
from ._dim_converters import (
    NetCDF4CoordToDimConverter,
    NetCDF4DimToDimConverter,
    NetCDF4ScalarToDimConverter,
)
from ._utils import (
    _DEFAULT_INDEX_DTYPE,
    copy_group_metadata,
    get_variable_chunks,
    open_netcdf_group,
)


class NetCDF4ConverterEngine(DataspaceCreator):
    """Converter for NetCDF to TileDB using netCDF4.

    This class is used to generate and copy data to a TileDB group or array from a
    NetCDF file. The converter can be auto-generated from a NetCDF group, or it can
    be manually defined.

    This is a subclass of :class:`tiledb.cf.DataspaceCreator`. See
    :class:`tiledb.cf.DataspaceCreator` for documentation of additional properties and
    methods.
    """

    @classmethod
    def from_file(
        cls,
        input_file: Union[str, Path],
        group_path: str = "/",
        unlimited_dim_size: Optional[int] = None,
        dim_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
        tiles_by_var: Optional[Dict[str, Optional[Sequence[int]]]] = None,
        tiles_by_dims: Optional[Dict[Sequence[str], Optional[Sequence[int]]]] = None,
        coords_to_dims: bool = False,
        collect_attrs: bool = True,
        unpack_vars: bool = False,
        coords_filters: Optional[tiledb.FilterList] = None,
        offsets_filters: Optional[tiledb.FilterList] = None,
        attrs_filters: Optional[tiledb.FilterList] = None,
    ):
        """Returns a :class:`NetCDF4ConverterEngine` from a group in a NetCDF file.

        Parameters:
            input_file: The input NetCDF file to generate the converter engine from.
            group_path: The path to the NetCDF group to copy data from. Use ``'/'`` for
                the root group.
            unlimited_dim_size: The size of the domain for TileDB dimensions created
                from unlimited NetCDF dimensions. If ``None``, the current size of the
                NetCDF dimension will be used.
            dim_dtype: The numpy dtype for TileDB dimensions.
            tiles_by_var: A map from the name of a NetCDF variable to the tiles of the
                dimensions of the variable in the generated TileDB array.
            tiles_by_dims: A map from the name of NetCDF dimensions defining a variable
                to the tiles of those dimensions in the generated TileDB array.
            coords_to_dims: If ``True``, convert the NetCDF coordinate variable into a
                TileDB dimension for sparse arrays. Otherwise, convert the coordinate
                dimension into a TileDB dimension and the coordinate variable into a
                TileDB attribute.
            collect_attrs: If True, store all attributes with the same dimensions
                in the same array. Otherwise, store each attribute in a scalar array.
            unpack_vars: Unpack NetCDF variables with NetCDF attributes ``scale_factor``
                or ``add_offset`` using the transformation ``scale_factor * value +
                unpack``.
            coords_filters: Default filters for all dimensions.
            offsets_filters: Default filters for all offsets for variable attributes
                and dimensions.
            attrs_filters: Default filters for all attributes.
        """
        with open_netcdf_group(input_file=input_file, group_path=group_path) as group:
            return cls.from_group(
                group,
                unlimited_dim_size,
                dim_dtype,
                tiles_by_var,
                tiles_by_dims,
                default_input_file=input_file,
                default_group_path=group_path,
                coords_to_dims=coords_to_dims,
                collect_attrs=collect_attrs,
                unpack_vars=unpack_vars,
                coords_filters=coords_filters,
                offsets_filters=offsets_filters,
                attrs_filters=attrs_filters,
            )

    @classmethod
    def from_group(
        cls,
        netcdf_group: netCDF4.Group,
        unlimited_dim_size: Optional[int] = None,
        dim_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
        tiles_by_var: Optional[Dict[str, Optional[Sequence[int]]]] = None,
        tiles_by_dims: Optional[Dict[Sequence[str], Optional[Sequence[int]]]] = None,
        coords_to_dims: bool = False,
        collect_attrs: bool = True,
        unpack_vars: bool = False,
        coords_filters: Optional[tiledb.FilterList] = None,
        offsets_filters: Optional[tiledb.FilterList] = None,
        attrs_filters: Optional[tiledb.FilterList] = None,
        default_input_file: Optional[Union[str, Path]] = None,
        default_group_path: Optional[str] = None,
    ):
        """Returns a :class:`NetCDF4ConverterEngine` from a :class:`netCDF4.Group`.

        Parameters:
            group: The NetCDF group to convert.
            unlimited_dim_size: The size of the domain for TileDB dimensions created
                from unlimited NetCDF dimensions. If ``None``, the current size of the
                NetCDF variable will be used.
            dim_dtype: The numpy dtype for TileDB dimensions.
            tiles_by_var: A map from the name of a NetCDF variable to the tiles of the
                dimensions of the variable in the generated TileDB array.
            tiles_by_dims: A map from the name of NetCDF dimensions defining a variable
                to the tiles of those dimensions in the generated TileDB array.
            coords_to_dims: If ``True``, convert the NetCDF coordinate variable into a
                TileDB dimension for sparse arrays. Otherwise, convert the coordinate
                dimension into a TileDB dimension and the coordinate variable into a
                TileDB attribute.
            collect_attrs: If ``True``, store all attributes with the same dimensions
                in the same array. Otherwise, store each attribute in a scalar array.
            unpack_vars: Unpack NetCDF variables with NetCDF attributes ``scale_factor``
                or ``add_offset`` using the transformation ``scale_factor * value +
                unpack``.
            coords_filters: Default filters for all dimensions.
            offsets_filters: Default filters for all offsets for variable attributes
                and dimensions.
            attrs_filters: Default filters for all attributes.
            default_input_file: If not ``None``, the default NetCDF input file to copy
                data from.
            default_group_path: If not ``None``, the default NetCDF group to copy data
                from. Use ``'/'`` to specify the root group.
        """
        if collect_attrs:
            return cls._from_group_to_collected_attrs(
                netcdf_group=netcdf_group,
                unlimited_dim_size=unlimited_dim_size,
                dim_dtype=dim_dtype,
                tiles_by_var=tiles_by_var,
                tiles_by_dims=tiles_by_dims,
                coords_to_dims=coords_to_dims,
                unpack_vars=unpack_vars,
                coords_filters=coords_filters,
                offsets_filters=offsets_filters,
                attrs_filters=attrs_filters,
                default_input_file=default_input_file,
                default_group_path=default_group_path,
            )
        return cls._from_group_to_attr_per_array(
            netcdf_group=netcdf_group,
            unlimited_dim_size=unlimited_dim_size,
            dim_dtype=dim_dtype,
            tiles_by_var=tiles_by_var,
            tiles_by_dims=tiles_by_dims,
            coords_to_dims=coords_to_dims,
            unpack_vars=unpack_vars,
            scalar_array_name="scalars",
            coords_filters=coords_filters,
            offsets_filters=offsets_filters,
            attrs_filters=attrs_filters,
            default_input_file=default_input_file,
            default_group_path=default_group_path,
        )

    @classmethod  # noqa: C901
    def _from_group_to_attr_per_array(  # noqa: C901
        cls,
        netcdf_group: netCDF4.Group,
        unlimited_dim_size: Optional[int],
        dim_dtype: np.dtype,
        tiles_by_var: Optional[Dict[str, Optional[Sequence[int]]]],
        tiles_by_dims: Optional[Dict[Sequence[str], Optional[Sequence[int]]]],
        coords_to_dims: bool,
        scalar_array_name: str,
        unpack_vars: bool,
        coords_filters: Optional[tiledb.FilterList],
        offsets_filters: Optional[tiledb.FilterList],
        attrs_filters: Optional[tiledb.FilterList],
        default_input_file: Optional[Union[str, Path]],
        default_group_path: Optional[str],
    ):
        """Returns a :class:`NetCDF4ConverterEngine` from a :class:`netCDF4.Group`.

        Parameters:
            group: The NetCDF group to convert.
            unlimited_dim_size: The size of the domain for TileDB dimensions created
                from unlimited NetCDF dimensions.
            dim_dtype: The numpy dtype for TileDB dimensions.
            tiles_by_var: A map from the name of a NetCDF variable to the tiles of the
                dimensions of the variable in the generated TileDB array. The tile
                sizes defined by this dict take priority over the ``tiles_by_dims``
                parameter and the NetCDF variable chunksize.
            tiles_by_dims: A map from the name of NetCDF dimensions defining a variable
                to the tiles of those dimensions in the generated TileDB array. The
                tile size defined by this dict are used if they are not defined by the
                parameter ``tiles_by_var``.
            coords_to_dims: If ``True``, convert the NetCDF coordinate variable into a
                TileDB dimension for sparse arrays. Otherwise, convert the coordinate
                dimension into a TileDB dimension and the coordinate variable into a
                TileDB attribute.
            scalar_array_name: Name for the array the stores all NetCDF scalar
                variables. Cannot be the same name as any of the NetCDF variables in
                the provided NetCDF group.
            unpack_vars: Unpack NetCDF variables with NetCDF attributes ``scale_factor``
                or ``add_offset`` using the transformation ``scale_factor * value +
                unpack``.
            coords_filters: Default filters for all dimensions.
            offsets_filters: Default filters for all offsets for variable attributes
                and dimensions.
            attrs_filters: Default filters for all attributes.
            default_input_file: If not ``None``, the default NetCDF input file to copy
                data from.
            default_group_path: If not ``None``, the default NetCDF group to copy data
                from. Use ``'/'`` to specify the root group.
        """
        converter = cls(default_input_file, default_group_path)
        coord_names = set()
        tiles_by_var = {} if tiles_by_var is None else tiles_by_var
        tiles_by_dims = {} if tiles_by_dims is None else tiles_by_dims
        if coords_to_dims:
            for ncvar in netcdf_group.variables.values():
                if ncvar.ndim == 1 and ncvar.dimensions[0] == ncvar.name:
                    converter.add_coord_to_dim_converter(ncvar, unpack=unpack_vars)
                    coord_names.add(ncvar.name)
        for ncvar in netcdf_group.variables.values():
            if ncvar.name in coord_names:
                continue
            if not ncvar.dimensions:
                if scalar_array_name in netcdf_group.variables:
                    raise ValueError(
                        f"Cannot name array of scalar values `{scalar_array_name}`. An"
                        f" array with that name already exists."
                    )
                if scalar_array_name not in {
                    array_creator.name for array_creator in converter.array_creators()
                }:
                    converter.add_scalar_to_dim_converter("__scalars", dim_dtype)
                    converter.add_array_converter(
                        scalar_array_name,
                        ("__scalars",),
                        coords_filters=coords_filters,
                        offsets_filters=offsets_filters,
                        attrs_filters=attrs_filters,
                    )
                converter.add_var_to_attr_converter(ncvar, scalar_array_name)
            else:
                for dim in ncvar.get_dims():
                    if dim.name not in {
                        shared_dim.name for shared_dim in converter.shared_dims()
                    }:
                        converter.add_dim_to_dim_converter(
                            dim,
                            unlimited_dim_size,
                            dim_dtype,
                        )
                array_name = ncvar.name
                has_coord_dim = any(
                    dim_name in coord_names for dim_name in ncvar.dimensions
                )
                tiles = tiles_by_var.get(
                    ncvar.name,
                    tiles_by_dims.get(
                        ncvar.dimensions,
                        None if has_coord_dim else get_variable_chunks(ncvar),
                    ),
                )
                converter.add_array_converter(
                    array_name,
                    ncvar.dimensions,
                    tiles=tiles,
                    sparse=has_coord_dim,
                    coords_filters=coords_filters,
                    offsets_filters=offsets_filters,
                    attrs_filters=attrs_filters,
                )
                converter.add_var_to_attr_converter(
                    ncvar, array_name, unpack=unpack_vars
                )
        return converter

    @classmethod
    def _from_group_to_collected_attrs(
        cls,
        netcdf_group: netCDF4.Group,
        unlimited_dim_size: Optional[int],
        dim_dtype: np.dtype,
        tiles_by_var: Optional[Dict[str, Optional[Sequence[int]]]],
        tiles_by_dims: Optional[Dict[Sequence[str], Optional[Sequence[int]]]],
        coords_to_dims: bool,
        unpack_vars: bool,
        coords_filters: Optional[tiledb.FilterList],
        offsets_filters: Optional[tiledb.FilterList],
        attrs_filters: Optional[tiledb.FilterList],
        default_input_file: Optional[Union[str, Path]],
        default_group_path: Optional[str],
    ):
        """Returns a :class:`NetCDF4ConverterEngine` from a :class:`netCDF4.Group`.

        Parameters:
            group: The NetCDF group to convert.
            unlimited_dim_size: The size of the domain for TileDB dimensions created
                from unlimited NetCDF dimensions. If ``None``, the current size of the
                NetCDf dimension will be used.
            dim_dtype: The numpy dtype for TileDB dimensions.
            tiles_by_var: A map from the name of a NetCDF variable to the tiles of the
                dimensions of the variable in the generated TileDB array. This will
                take priority over NetCDF variable chunksize.
            tiles_by_dims: A map from the name of NetCDF dimensions defining a variable
                to the tiles of those dimensions in the generated TileDB array. This
                will take priority over tile sizes defined by the ``tiles_by_var``
                parameter and the NetCDF variable chunksize.
            coords_to_dims: If ``True``, convert the NetCDF coordinate variable into a
                TileDB dimension for sparse arrays. Otherwise, convert the coordinate
                dimension into a TileDB dimension and the coordinate variable into a
                TileDB attribute.
            unpack_vars: Unpack NetCDF variables with NetCDF attributes ``scale_factor``
                or ``add_offset`` using the transformation ``scale_factor * value +
                unpack``.
            coords_filters: Default filters for all dimensions.
            offsets_filters: Default filters for all offsets for variable attributes
                and dimensions.
            attrs_filters: Default filters for all attributes.
            default_input_file: If not ``None``, the default NetCDF input file to copy
                data from.
            default_group_path: If not ``None``, the default NetCDF group to copy data
                from. Use ``'/'`` to specify the root group.
        """
        converter = cls(default_input_file, default_group_path)
        coord_names = set()
        dims_to_vars: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
        autotiles: Dict[Sequence[str], Optional[Sequence[int]]] = {}
        tiles_by_dims = {} if tiles_by_dims is None else tiles_by_dims
        tiles_by_var = {} if tiles_by_var is None else tiles_by_var
        # Add data/coordinate dimension to converter, partition variables into arrays,
        # and compute the tile sizes for array dimensions.
        for ncvar in netcdf_group.variables.values():
            if coords_to_dims and ncvar.ndim == 1 and ncvar.dimensions[0] == ncvar.name:
                converter.add_coord_to_dim_converter(ncvar, unpack=unpack_vars)
                coord_names.add(ncvar.name)
            else:
                if not ncvar.dimensions and "__scalars" not in {
                    shared_dim.name for shared_dim in converter.shared_dims()
                }:
                    converter.add_scalar_to_dim_converter("__scalars", dim_dtype)
                dim_names = ncvar.dimensions if ncvar.dimensions else ("__scalars",)
                dims_to_vars[dim_names].append(ncvar.name)
                chunks = tiles_by_var.get(
                    ncvar.name,
                    None
                    if any(dim_name in coord_names for dim_name in ncvar.dimensions)
                    else get_variable_chunks(ncvar),
                )
                if chunks is not None:
                    autotiles[dim_names] = (
                        None
                        if dim_names in autotiles and chunks != autotiles[dim_names]
                        else chunks
                    )
        autotiles.update(tiles_by_dims)
        # Add index dimensions to converter.
        for ncvar in netcdf_group.variables.values():
            for dim in ncvar.get_dims():
                if dim.name not in {
                    shared_dim.name for shared_dim in converter.shared_dims()
                }:
                    converter.add_dim_to_dim_converter(
                        dim,
                        unlimited_dim_size,
                        dim_dtype,
                    )
        # Add arrays and attributes to the converter.
        for count, dim_names in enumerate(sorted(dims_to_vars.keys())):
            has_coord_dim = any(dim_name in coord_names for dim_name in dim_names)
            chunks = autotiles.get(dim_names)
            converter.add_array_converter(
                f"array{count}",
                dim_names,
                tiles=chunks,
                sparse=has_coord_dim,
                coords_filters=coords_filters,
                offsets_filters=offsets_filters,
                attrs_filters=attrs_filters,
            )
            for var_name in dims_to_vars[dim_names]:
                converter.add_var_to_attr_converter(
                    netcdf_group.variables[var_name],
                    f"array{count}",
                    unpack=unpack_vars,
                )
        return converter

    def __init__(
        self,
        default_input_file: Optional[Union[str, Path]] = None,
        default_group_path: Optional[str] = None,
    ):
        self.default_input_file = default_input_file
        self.default_group_path = default_group_path
        super().__init__()

    def __repr__(self):
        output = StringIO()
        output.write(f"{super().__repr__()}\n")
        if self.default_input_file is not None:
            output.write(f"Default NetCDF file: {self.default_input_file}\n")
        if self.default_group_path is not None:
            output.write(f"Deault NetCDF group path: {self.default_group_path}\n")
        return output.getvalue()

    def _copy_data(
        self,
        group_uri: str,
        array_uris: Dict[str, str],
        key: Optional[str],
        ctx: Optional[tiledb.Ctx],
        input_netcdf_group: Optional[netCDF4.Group],
        input_file: Optional[Union[str, Path]],
        input_group_path: Optional[str],
        assigned_dim_values: Optional[Dict[str, Any]],
        assigned_attr_values: Optional[Dict[str, np.ndarray]],
        copy_metadata: bool,
        timestamp: Optional[int],
    ):
        if input_netcdf_group is None:
            input_file = (
                input_file if input_file is not None else self.default_input_file
            )
            input_group_path = (
                input_group_path
                if input_group_path is not None
                else self.default_group_path
            )
        with open_netcdf_group(
            input_netcdf_group, input_file, input_group_path
        ) as netcdf_group:
            if copy_metadata:
                with tiledb.open(
                    group_uri, mode="w", key=key, ctx=ctx, timestamp=timestamp
                ) as group_array:
                    copy_group_metadata(netcdf_group, group_array.meta)
            for array_creator in self._registry.array_creators():
                if isinstance(array_creator, NetCDF4ArrayConverter):
                    array_creator.copy(
                        netcdf_group=netcdf_group,
                        tiledb_uri=array_uris[array_creator.name],
                        tiledb_key=key,
                        tiledb_ctx=ctx,
                        tiledb_timestamp=timestamp,
                        assigned_dim_values=assigned_dim_values,
                        assigned_attr_values=assigned_attr_values,
                        copy_metadata=copy_metadata,
                    )

    def _repr_html_(self):
        output = StringIO()
        output.write(f"{super()._repr_html_()}\n")
        output.write("<ul>\n")
        output.write(f"<li>Default NetCDF file: '{self.default_input_file}'</li>\n")
        output.write(
            f"<li>Default NetCDF group path: '{self.default_group_path}'</li>\n"
        )
        output.write("</ul>\n")
        return output.getvalue()

    def add_array_converter(
        self,
        array_name: str,
        dims: Sequence[str],
        cell_order: str = "row-major",
        tile_order: str = "row-major",
        capacity: int = 0,
        tiles: Optional[Sequence[int]] = None,
        coords_filters: Optional[tiledb.FilterList] = None,
        dim_filters: Optional[Dict[str, tiledb.FilterList]] = None,
        offsets_filters: Optional[tiledb.FilterList] = None,
        attrs_filters: Optional[tiledb.FilterList] = None,
        allows_duplicates: bool = False,
        sparse: bool = False,
    ):
        """Adds a new NetCDF to TileDB array converter to the CF dataspace.

        The name of each array must be unique. All properties must match the normal
        requirements for a ``TileDB.ArraySchema``.

        Parameters:
            array_name: Name of the new array to be created.
            dims: An ordered list of the names of the shared dimensions for the domain
                of this array.
            cell_order: The order in which TileDB stores the cells on disk inside a
                tile. Valid values are: ``row-major`` (default) or ``C`` for row major;
                ``col-major`` or ``F`` for column major; or ``Hilbert`` for a Hilbert
                curve.
            tile_order: The order in which TileDB stores the tiles on disk. Valid values
                are: ``row-major`` or ``C`` (default) for row major; or ``col-major`` or
                ``F`` for column major.
            capacity: The number of cells in a data tile of a sparse fragment.
            tiles: An optional ordered list of tile sizes for the dimensions of the
                array. The length must match the number of dimensions in the array.
            coords_filters: Filters for all dimensions that are not otherwise set by
                ``dim_filters.``
            dim_filters: A dict from dimension name to a ``FilterList`` for dimensions
                in the array. Overrides the values set in ``coords_filters``.
            offsets_filters: Filters for the offsets for variable length attributes or
                dimensions.
            attrs_filters: Default filters to use when adding an attribute to the array.
            allows_duplicates: Specifies if multiple values can be stored at the same
                 coordinate. Only allowed for sparse arrays.
            sparse: Specifies if the array is a sparse TileDB array (true) or dense
                TileDB array (false).
        """
        NetCDF4ArrayConverter(
            dataspace_registry=self._registry,
            name=array_name,
            dims=dims,
            cell_order=cell_order,
            tile_order=tile_order,
            capacity=capacity,
            tiles=tiles,
            coords_filters=coords_filters,
            dim_filters=dim_filters,
            offsets_filters=offsets_filters,
            attrs_filters=attrs_filters,
            allows_duplicates=allows_duplicates,
            sparse=sparse,
        )

    def add_coord_to_dim_converter(
        self,
        ncvar: netCDF4.Variable,
        dim_name: Optional[str] = None,
        domain: Optional[Tuple[DType, DType]] = None,
        dtype: Optional[np.dtype] = None,
        unpack: bool = False,
    ):
        """Adds a new NetCDF coordinate to TileDB dimension converter.

        Parameters:
            var: NetCDF coordinate variable to be converted.
            dim_name: If not ``None``, name to use for the TileDB dimension.
            domain: If not ``None``, the domain the TileDB dimension is valid on.
            dtype: If not ``None``, the data type the TileDB dimension will be set to.
            unpack: Unpack NetCDF data that has NetCDF attributes ``scale_factor`` or
                ``add_offset`` using the transformation ``scale_factor * value +
                unpack``.

        """
        NetCDF4CoordToDimConverter.from_netcdf(
            dataspace_registry=self._registry,
            ncvar=ncvar,
            name=dim_name,
            domain=domain,
            dtype=dtype,
            unpack=unpack,
        )

    def add_dim_to_dim_converter(
        self,
        ncdim: netCDF4.Dimension,
        unlimited_dim_size: Optional[int] = None,
        dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
        dim_name: Optional[str] = None,
    ):
        """Adds a new NetCDF dimension to TileDB dimension converter.

        Parameters:
            ncdim: NetCDF dimension to be converted.
            unlimited_dim_size: The size to use if the dimension is unlimited. If
                ``None``, the current size of the NetCDF dimension will be used.
            dtype: Numpy type to use for the NetCDF dimension.
            dim_name: If not ``None``, output name of the TileDB dimension.
        """
        NetCDF4DimToDimConverter.from_netcdf(
            dataspace_registry=self._registry,
            dim=ncdim,
            unlimited_dim_size=unlimited_dim_size,
            dtype=dtype,
            name=dim_name,
        )

    def add_scalar_to_dim_converter(
        self,
        dim_name: str = "__scalars",
        dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
    ):
        """Adds a new TileDB dimension for NetCDF scalar variables.

        Parameters:
            dim_name: Output name of the dimension.
            dtype: Numpy type to use for the scalar dimension
        """
        NetCDF4ScalarToDimConverter.create(
            dataspace_registry=self._registry, dim_name=dim_name, dtype=dtype
        )

    def add_var_to_attr_converter(
        self,
        ncvar: netCDF4.Variable,
        array_name: str,
        attr_name: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
        fill: Optional[Union[int, float, str]] = None,
        var: bool = False,
        nullable: bool = False,
        filters: Optional[tiledb.FilterList] = None,
        unpack: bool = False,
    ):
        """Adds a new variable to attribute converter to an array in the CF dataspace.

        The attribute's 'dataspace name' (name after dropping the suffix ``.data`` or
        ``.index``) must be unique.

        Parameters:
            ncvar: NetCDF variable to convert to a TileDB attribute.
            name: Name of the new attribute that will be added. If ``None``, the name
                will be copied from the NetCDF variable.
            dtype: Numpy dtype of the new attribute. If ``None``, the data type will be
                copied from the variable.
            fill: Fill value for unset cells. If ``None``, the fill value will be
                copied from the NetCDF variable if it has a fill value.
            var: Specifies if the attribute is variable length (automatic for
                byte/strings).
            nullable: Specifies if the attribute is nullable using validity tiles.
            filters: Specifies compression filters for the attribute.
            unpack: Unpack NetCDF data that has NetCDF attributes ``scale_factor`` or
                ``add_offset`` using the transformation ``scale_factor * value +
                unpack``.
        """
        try:
            array_creator = self._registry.get_array_creator(array_name)
        except KeyError as err:  # pragma: no cover
            raise KeyError(
                f"Cannot add attribute to array '{array_name}'. No array named "
                f"'{array_name}' exists."
            ) from err
        array_creator.add_var_to_attr_converter(
            ncvar=ncvar,
            name=attr_name,
            dtype=dtype,
            fill=fill,
            var=var,
            nullable=nullable,
            filters=filters,
            unpack=unpack,
        )

    def convert_to_array(
        self,
        output_uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        timestamp: Optional[int] = None,
        input_netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        input_group_path: Optional[str] = None,
        assigned_dim_values: Optional[Dict[str, Any]] = None,
        assigned_attr_values: Optional[Dict[str, np.ndarray]] = None,
        copy_metadata: bool = True,
    ):
        """Creates a TileDB arrays for a CF dataspace with only one array and copies
        data into it using the NetCDF converter engine.

        Parameters:
            output_uri: Uniform resource identifier for the TileDB array to be created.
            key: If not ``None``, encryption key to encrypt and decrypt output arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            timestamp: If not ``None``, the TileDB timestamp to write the NetCDF data to
                TileDB at.
            input_netcdf_group: If not ``None``, the NetCDF group to copy data from.
                This will be prioritized over ``input_file`` if both are provided.
            input_file: If not ``None``, the NetCDF file to copy data from. This will
                not be used if ``netcdf_group`` is not ``None``.
            input_group_path: If not ``None``, the path to the NetCDF group to copy data
                from.
            assigned_dim_values: Mapping from dimension name to value for dimensions
                that are not converter from the NetCDF group.
            assigned_attr_values: Mapping from attribute name to numpy array of values
                for attributes that are not converted from the NetCDF group.
            copy_metadata: If  ``True`` copy NetCDF group and variable attributes to
                TileDB metadata. If ``False`` do not copy metadata.
        """
        self.create_array(output_uri, key, ctx)
        self.copy_to_array(
            output_uri=output_uri,
            key=key,
            ctx=ctx,
            timestamp=timestamp,
            input_netcdf_group=input_netcdf_group,
            input_file=input_file,
            input_group_path=input_group_path,
            assigned_dim_values=assigned_dim_values,
            assigned_attr_values=assigned_attr_values,
            copy_metadata=copy_metadata,
        )

    def convert_to_group(
        self,
        output_uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        timestamp: Optional[int] = None,
        input_netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        input_group_path: Optional[str] = None,
        assigned_dim_values: Optional[Dict[str, Any]] = None,
        assigned_attr_values: Optional[Dict[str, np.ndarray]] = None,
        copy_metadata: bool = True,
        append: bool = False,
    ):
        """Creates a TileDB group and its arrays from the defined CF dataspace and
        copies data into them using the converter engine.

        Parameters:
            output_uri: Uniform resource identifier for the TileDB group to be created.
            key: If not ``None``, encryption key to encrypt and decrypt output arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            timestamp: If not ``None``, the TileDB timestamp to write the NetCDF data to
                TileDB at.
            input_netcdf_group: If not ``None``, the NetCDF group to copy data from.
                This will be prioritized over ``input_file`` if both are provided.
            input_file: If not ``None``, the NetCDF file to copy data from. This will
                not be used if ``netcdf_group`` is not ``None``.
            input_group_path: If not ``None``, the path to the NetCDF group to copy data
                from.
            assigned_dim_values: Mapping from dimension name to value for dimensions
                that are not converted from the NetCDF group.
            assigned_attr_values: Mapping from attribute name to numpy array of values
                for attributes that are not converted from the NetCDF group.
            copy_metadata: If  ``True`` copy NetCDF group and variable attributes to
                TileDB metadata. If ``False`` do not copy metadata.
            append: If ``True``, add arrays in the dataspace to an already existing
                group. The arrays in the dataspace cannot be in the group that is being
                append to.
        """
        self.create_group(output_uri, key, ctx, append=append)
        self.copy_to_group(
            output_uri=output_uri,
            key=key,
            ctx=ctx,
            input_netcdf_group=input_netcdf_group,
            input_file=input_file,
            input_group_path=input_group_path,
            assigned_dim_values=assigned_dim_values,
            assigned_attr_values=assigned_attr_values,
            copy_metadata=copy_metadata,
        )

    def convert_to_virtual_group(
        self,
        output_uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        timestamp: Optional[int] = None,
        input_netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        input_group_path: Optional[str] = None,
        assigned_dim_values: Optional[Dict[str, Any]] = None,
        assigned_attr_values: Optional[Dict[str, np.ndarray]] = None,
        copy_metadata: bool = True,
    ):
        """Creates a TileDB group and its arrays from the defined CF dataspace and
        copies data into them using the converter engine.

        Parameters:
            output_uri: Uniform resource identifier for the TileDB group to be created.
            key: If not ``None``, encryption key to encrypt and decrypt output arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            timestamp: If not ``None``, the TileDB timestamp to write the NetCDF data to
                TileDB at.
            input_netcdf_group: If not ``None``, the NetCDF group to copy data from.
                This will be prioritized over ``input_file`` if both are provided.
            input_file: If not ``None``, the NetCDF file to copy data from. This will
                not be used if ``netcdf_group`` is not ``None``.
            input_group_path: If not ``None``, the path to the NetCDF group to copy data
                from.
            assigned_dim_values: Mapping from dimension name to value for dimensions
                that are not converted from the NetCDF group.
            assigned_attr_values: Mapping from attribute name to numpy array of values
                for attributes that are not converted from the NetCDF group.
            copy_metadata: If  ``True`` copy NetCDF group and variable attributes to
                TileDB metadata. If ``False`` do not copy metadata.
        """
        self.create_virtual_group(output_uri, key, ctx)
        self.copy_to_virtual_group(
            output_uri=output_uri,
            key=key,
            ctx=ctx,
            timestamp=timestamp,
            input_netcdf_group=input_netcdf_group,
            input_file=input_file,
            input_group_path=input_group_path,
            assigned_dim_values=assigned_dim_values,
            assigned_attr_values=assigned_attr_values,
            copy_metadata=copy_metadata,
        )

    def copy_to_array(
        self,
        output_uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        timestamp: Optional[int] = None,
        input_netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        input_group_path: Optional[str] = None,
        assigned_dim_values: Optional[Dict[str, Any]] = None,
        assigned_attr_values: Optional[Dict[str, np.ndarray]] = None,
        copy_metadata: bool = True,
    ):
        """Copies data from a NetCDF group to a TileDB array.

        This will copy data from a NetCDF group that is defined either by a
        :class:`netCDF4.Group` or by an input_file and group path. If neither the
        ``netcdf_group`` or ``input_file`` is specified, this will copy data from the
        input file ``self.default_input_file``.  If both ``netcdf_group`` and
        ``input_file`` are set, this method will prioritize using the NetCDF group set
        by ``netcdf_group``.

        Parameters:
            output_uri: Uniform resource identifier for the TileDB array data is being
                copied to.
            key: If not ``None``, encryption key to decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            timestamp: If not ``None``, the TileDB timestamp to write the NetCDF data to
                TileDB at.
            input_netcdf_group: If not ``None``, the NetCDF group to copy data from.
                This will be prioritized over ``input_file`` if both are provided.
            input_file: If not ``None``, the NetCDF file to copy data from. This will
                not be used if ``netcdf_group`` is not ``None``.
            input_group_path: If not ``None``, the path to the NetCDF group to copy data
                from.
            assigned_dim_values: Mapping from dimension name to value for dimensions
                that are not copied from the NetCDF group.
            assigned_attr_values: Mapping from attribute name to numpy array of values
                for attributes that are not copied from the NetCDF group.
            copy_metadata: If  ``True`` copy NetCDF group and variable attributes to
                TileDB metadata. If ``False`` do not copy metadata.
        """
        if self._registry.narray != 1:  # pragma: no cover
            raise ValueError(
                f"Can only use `copy_to_array` for a {self.__class__.__name__} with "
                f"exactly 1 array creator. This {self.__class__.__name__} contains "
                f"{self._registry.narray} array creators."
            )
        array_creator = next(self._registry.array_creators())
        self._copy_data(
            group_uri=output_uri,
            array_uris={array_creator.name: output_uri},
            key=key,
            ctx=ctx,
            timestamp=timestamp,
            input_netcdf_group=input_netcdf_group,
            input_file=input_file,
            input_group_path=input_group_path,
            assigned_dim_values=assigned_dim_values,
            assigned_attr_values=assigned_attr_values,
            copy_metadata=copy_metadata,
        )

    def copy_to_group(
        self,
        output_uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        timestamp: Optional[int] = None,
        input_netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        input_group_path: Optional[str] = None,
        assigned_dim_values: Optional[Dict[str, Any]] = None,
        assigned_attr_values: Optional[Dict[str, np.ndarray]] = None,
        copy_metadata: bool = True,
    ):
        """Copies data from a NetCDF group to a TileDB CF dataspace.

        This will copy data from a NetCDF group that is defined either by a
        :class:`netCDF4.Group` or by an input_file and group path. If neither the
        ``netcdf_group`` or ``input_file`` is specified, this will copy data from the
        input file ``self.default_input_file``.  If both ``netcdf_group`` and
        ``input_file`` are set, this method will prioritize using the NetCDF group set
        by ``netcdf_group``.

        Parameters:
            output_uri: Uniform resource identifier for the TileDB group data is being
                copied to.
            key: If not ``None``, encryption key to decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            timestamp: If not ``None``, the TileDB timestamp to write the NetCDF data to
                TileDB at.
            input_netcdf_group: If not ``None``, the NetCDF group to copy data from.
                This will be prioritized over ``input_file`` if both are provided.
            input_file: If not ``None``, the NetCDF file to copy data from. This will
                not be used if ``netcdf_group`` is not ``None``.
            input_group_path: If not ``None``, the path to the NetCDF group to copy data
                from.
            assigned_dim_values: Mapping from dimension name to value for dimensions
                that are not copied from the NetCDF group.
            assigned_attr_values: Mapping from attribute name to numpy array of values
                for the attributes that are not copied from the NetCDF group.
            copy_metadata: If  ``True`` copy NetCDF group and variable attributes to
                TileDB metadata. If ``False`` do not copy metadata.
        """
        group_uri = os.path.join(output_uri, METADATA_ARRAY_NAME)
        array_uris = {
            array_creator.name: os.path.join(output_uri, array_creator.name)
            for array_creator in self._registry.array_creators()
        }
        self._copy_data(
            group_uri=group_uri,
            array_uris=array_uris,
            key=key,
            ctx=ctx,
            timestamp=timestamp,
            input_netcdf_group=input_netcdf_group,
            input_file=input_file,
            input_group_path=input_group_path,
            assigned_dim_values=assigned_dim_values,
            assigned_attr_values=assigned_attr_values,
            copy_metadata=copy_metadata,
        )

    def copy_to_virtual_group(
        self,
        output_uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        timestamp: Optional[int] = None,
        input_netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        input_group_path: Optional[str] = None,
        assigned_dim_values: Optional[Dict[str, Any]] = None,
        assigned_attr_values: Optional[Dict[str, np.ndarray]] = None,
        copy_metadata: bool = True,
    ):
        """Copies data from a NetCDF group to a TileDB CF dataspace.

        This will copy data from a NetCDF group that is defined either by a
        :class:`netCDF4.Group` or by an input_file and group path. If neither the
        ``netcdf_group`` or ``input_file`` is specified, this will copy data from the
        input file ``self.default_input_file``.  If both ``netcdf_group`` and
        ``input_file`` are set, this method will prioritize using the NetCDF group set
        by ``netcdf_group``.

        Parameters:
            output_uri: Uniform resource identifier for the TileDB group data is being
                copied to.
            key: If not ``None``, encryption key to decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            timestamp: If not ``None``, the TileDB timestamp to write the NetCDF data to
                TileDB at.
            input_netcdf_group: If not ``None``, the NetCDF group to copy data from.
                This will be prioritized over ``input_file`` if both are provided.
            input_file: If not ``None``, the NetCDF file to copy data from. This will
                not be used if ``netcdf_group`` is not ``None``.
            input_group_path: If not ``None``, the path to the NetCDF group to copy data
                from.
            assigned_dim_values: Mapping from dimension name to value for dimensions
                that are not from the NetCDF group.
            assigned_attr_values: Mapping from attribute name to numpy array of values
                for attributes that are not copied from the NetCDF group.
            copy_metadata: If  ``True`` copy NetCDF group and variable attributes to
                TileDB metadata. If ``False`` do not copy metadata.
        """
        array_uris = {
            array_creator.name: f"{output_uri}_{array_creator.name}"
            for array_creator in self._registry.array_creators()
        }
        self._copy_data(
            group_uri=output_uri,
            array_uris=array_uris,
            key=key,
            ctx=ctx,
            timestamp=timestamp,
            input_netcdf_group=input_netcdf_group,
            input_file=input_file,
            input_group_path=input_group_path,
            assigned_dim_values=assigned_dim_values,
            assigned_attr_values=assigned_attr_values,
            copy_metadata=copy_metadata,
        )
