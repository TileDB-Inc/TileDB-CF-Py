# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Classes for converting NetCDF4 files to TileDB."""

import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import netCDF4
import numpy as np

import tiledb

from ..core import Group
from ..creator import (
    ArrayCreator,
    AttrCreator,
    DataspaceCreator,
    SharedDim,
    dataspace_name,
)

_DEFAULT_INDEX_DTYPE = np.dtype("uint64")
COORDINATE_SUFFIX = ".data"


@dataclass
class NetCDFDimensionConverter(SharedDim):
    """Data for converting from a NetCDF dimension to a TileDB dimension.

    Parameters:
        name: Name of the TileDB dimension.
        domain: The (inclusive) interval on which the dimension is valid.
        dtype: The numpy dtype of the values and domain of the dimension.
        input_name: Name of the input NetCDF variable.
        input_size: Size of the input NetCDF variable.
        is_unlimited: If True, the input NetCDF variable is unlimited.

    Attributes:
        name: Name of the TileDB dimension.
        domain: The (inclusive) interval on which the dimension is valid.
        dtype: The numpy dtype of the values and domain of the dimension.
        input_name: Name of the input NetCDF variable.
        input_size: Size of the input NetCDF variable.
        is_unlimited: If True, the input NetCDF variable is unlimited.
    """

    input_name: str
    input_size: int
    is_unlimited: bool

    def __repr__(self):
        size_str = "unlimited" if self.is_unlimited else str(self.input_size)
        return (
            f"Dimension(name={self.input_name}, size={size_str}) -> "
            f"{super().__repr__()}"
        )

    @classmethod
    def from_netcdf(
        cls,
        dim: netCDF4.Dimension,
        unlimited_dim_size: int,
        dtype: np.dtype,
        dim_name: Optional[str] = None,
    ):
        """Returns a :class:`NetCDFDimensionConverter` from a
        :class:`netcdf4.Dimension`.

        Parameters:
            dim: The input netCDF4 dimension.
            unlimited_dim_size: The size of the domain of the output TileDB dimension
                when the input NetCDF dimension is unlimited.
            dtype: The numpy dtype of the values and domain of the output TileDB
                dimension.
            dim_name: The name of the output TileDB dimension. If ``None``, the name
                will be the same as the name of the input NetCDF dimension.
        """
        size = dim.size if not dim.isunlimited() else unlimited_dim_size
        return cls(
            dim_name if dim_name is not None else dim.name,
            (0, size - 1),
            np.dtype(dtype),
            dim.name,
            dim.size,
            dim.isunlimited(),
        )


@dataclass
class NetCDFVariableConverter(AttrCreator):
    """Data for converting from a NetCDF variable to a TileDB attribute.

    Parameters:
        name: Name of the new attribute.
        dtype: Numpy dtype of the attribute.
        fill: Fill value for unset cells.
        var: Specifies if the attribute is variable length (automatic for
            byte/strings).
        nullable: Specifies if the attribute is nullable using validity tiles.
        filters: Specifies compression filters for the attribute.
        input_name: Name of the input NetCDF variable that will be converted.
        input_dtype: Numpy dtype of the input NetCDF variable.

    Attributes:
        name: Name of the new attribute.
        dtype: Numpy dtype of the attribute.
        fill: Fill value for unset cells.
        var: Specifies if the attribute is variable length (automatic for
            byte/strings).
        nullable: Specifies if the attribute is nullable using validity tiles.
        filters: Specifies compression filters for the attribute.
        input_name: Name of the input NetCDF variable that will be converted.
        input_dtype: Numpy dtype of the input NetCDF variable.
    """

    input_name: Optional[str] = None
    input_dtype: Optional[np.dtype] = None

    def __repr__(self):
        return (
            f"Variable(name={self.input_name}, dtype={self.input_dtype}) -> "
            f"{super().__repr__()}"
        )

    @classmethod
    def from_netcdf(
        cls,
        ncvar: netCDF4.Variable,
        attr_name: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
        fill: Optional[Union[int, float, str]] = None,
        var: bool = False,
        nullable: bool = False,
        filters: Optional[tiledb.FilterList] = None,
    ):
        """Returns a :class:`NetCDFVariableConverter` from a :class:`netCDF4.Variable`.

        Parameters:
            ncvar: The input netCDF4 variable.
            attr_name: The name of the output TileDB attribute. If ``None``, the name
                will be generated from the name of the NetCDF variable.
            dtype: The numpy dtype of the output TileDB attribute. If ``None``, the name
                will be generated from the NetCDF variable.
            fill: Fill value for unset values in the input NetCDF variable. If ``None``,
                the fill value will be generated from the NetCDF variable.
            var: Specifies if the attribute is variable length (automatic for
                byte/strings).
            nullable: Specifies if the attribute is nullable using validity tiles.
            filters: Specifies compression filters for the attribute.
        """
        if fill is None and "_FillValue" in ncvar.ncattrs():
            fill = ncvar.getncattr("_FillValue")
        if attr_name is None:
            attr_name = (
                ncvar.name
                if ncvar.name not in ncvar.dimensions
                else ncvar.name + COORDINATE_SUFFIX
            )
        dtype = np.dtype(dtype) if dtype is not None else np.dtype(ncvar.dtype)
        return cls(
            attr_name,
            dtype,
            fill,
            var,
            nullable,
            filters,
            input_name=ncvar.name,
            input_dtype=ncvar.dtype,
        )


class NetCDFArrayConverter(ArrayCreator):
    """Converter for a TileDB array from a collection of NetCDF variables.

    Parameters:
        dims: An ordered list of the shared dimensions for the domain of this array.
        cell_order: The order in which TileDB stores the cells on disk inside a
            tile. Valid values are: ``row-major`` (default) or ``C`` for row major;
            ``col-major`` or ``F`` for column major; or ``Hilbert`` for a Hilbert curve.
        tile_order: The order in which TileDB stores the tiles on disk. Valid values
            are: ``row-major`` or ``C`` (default) for row major; or ``col-major`` or
            ``F`` for column major.
        capacity: The number of cells in a data tile of a sparse fragment.
        tiles: An optional ordered list of tile sizes for the dimensions of the
            array. The length must match the number of dimensions in the array.
        coords_filters: Filters for all dimensions that are not specified explicitly by
            ``dim_filters``.
        dim_filters: A dict from dimension name to a ``FilterList`` for dimensions in
            the array. Overrides the values set in ``coords_filters``.
        offsets_filters: Filters for the offsets for variable length attributes or
            dimensions.
        allows_duplicates: Specifies if multiple values can be stored at the same
             coordinate. Only allowed for sparse arrays.
        sparse: Specifies if the array is a sparse TileDB array (true) or dense
            TileDB array (false).

    Attributes:
        cell_order: The order in which TileDB stores the cells on disk inside a
            tile. Valid values are: ``row-major`` (default) or ``C`` for row major;
            ``col-major`` or ``F`` for column major; or ``Hilbert`` for a Hilbert curve.
        tile_order: The order in which TileDB stores the tiles on disk. Valid values
            are: ``row-major`` or ``C`` (default) for row major; or ``col-major`` or
            ``F`` for column major.
        capacity: The number of cells in a data tile of a sparse fragment.
        coords_filters: Filters for all dimensions that are not specified explicitly by
            ``dim_filters``.
        offsets_filters: Filters for the offsets for variable length attributes or
            dimensions.
        allows_duplicates: Specifies if multiple values can be stored at the same
             coordinate. Only allowed for sparse arrays.
    """

    def add_attr(self, attr_creator: AttrCreator):
        """Adds a new attribute to an array in the CF dataspace.

        Each attribute name must be unique. It also cannot conflict with the name of a
        dimension in the array.

        Parameters:
            attr_name: Name of the new attribute that will be added.
            dtype: Numpy dtype of the new attribute.
            fill: Fill value for unset cells.
            var: Specifies if the attribute is variable length (automatic for
                byte/strings).
            nullable: Specifies if the attribute is nullable using validity tiles.
            filters: Specifies compression filters for the attribute.
        """
        if not isinstance(attr_creator, NetCDFVariableConverter):
            raise NotImplementedError(
                "Support for non-NetCDF attributes in a NetCDFArrayConverter has not "
                "yet been implemented."
            )
        super().add_attr(attr_creator)

    def copy(
        self,
        netcdf_group: netCDF4.Group,
        tiledb_group: Group,
    ):
        """Copies data from a NetCDF group to a TileDB CF dataspace.

        Parameters:
            netcdf_group: The NetCDF group to copy data from.
            tiledb_group: The tiledb group to copy data into. Must be open into the
                array the data is being copied into.
        """
        data = {}
        for attr_converter in self._attr_creators.values():
            if not isinstance(attr_converter, NetCDFVariableConverter):
                raise RuntimeError(
                    "Cannot assign value for attribute {attr_converter.name} that is "
                    "of type {type(attr_converter)}."
                )
            try:
                variable = netcdf_group.variables[attr_converter.input_name]
            except KeyError as err:
                raise KeyError(
                    f"Variable {attr_converter.input_name} not found in "
                    f"requested NetCDF group."
                ) from err
            data[attr_converter.name] = variable[...]
            attr_meta = tiledb_group.get_attr_metadata(attr_converter.name)
            for meta_key in variable.ncattrs():
                copy_metadata_item(attr_meta, variable, meta_key)
        dim_slice = tuple(slice(dim.size) for dim in variable.get_dims())
        tiledb_group.array[dim_slice or slice(None)] = data


@dataclass
class NetCDF4ConverterEngine(DataspaceCreator):
    """Converter for NetCDF to TileDB using netCDF4."""

    @classmethod
    def from_file(
        cls,
        input_file: Union[str, Path],
        group_path: str = "/",
        unlimited_dim_size: int = 10000,
        dim_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
        tiles: Optional[Dict[Tuple[str, ...], Optional[Tuple[int, ...]]]] = None,
    ):
        """Returns a :class:`NetCDF4ConverterEngine` from a group in a NetCDF file.

        Parameters:
            input_file: The input NetCDF file to generate the converter engine from.
            group_path: The path to the NetCDF group to copy data from. Use ``'/'`` for
                the root group.
            unlimited_dim_size: The size of the domain for TileDB dimensions created
                from unlimited NetCDF dimensions.
            dim_dtype: The numpy dtype for TileDB dimensions.
            tiles: A map from the name of NetCDF dimensions defining a variable to the
                tiles of those dimensions in the generated NetCDF array.
        """
        with open_netcdf_group(
            input_file=input_file,
            group_path=group_path,
        ) as group:
            return cls.from_group(
                group,
                unlimited_dim_size,
                dim_dtype,
                tiles,
                input_file,
                group_path,
            )

    @classmethod
    def from_group(
        cls,
        netcdf_group: netCDF4.Group,
        unlimited_dim_size: int = 10000,
        dim_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
        tiles: Optional[Dict[Tuple[str, ...], Optional[Tuple[int, ...]]]] = None,
        default_input_file: Optional[Union[str, Path]] = None,
        default_group_path: Optional[str] = None,
    ):
        """Returns a :class:`NetCDF4ConverterEngine` from a :class:`netCDF4.Group`.

        Parameters:
            group: The NetCDF group to convert.
            unlimited_dim_size: The size of the domain for TileDB dimensions created
                from unlimited NetCDF dimensions.
            dim_dtype: The numpy dtype for TileDB dimensions.
            tiles: A map from the name of NetCDF dimensions defining a variable to the
                tiles of those dimensions in the generated NetCDF array.
            default_input_file: If not ``None``, the default NetCDF input file to copy
                data from.
            default_group_path: If not ``None``, the default NetCDF group to copy data
                from. Use ``'/'`` to specify the root group.
        """
        converter = cls(default_input_file, default_group_path)
        dims_to_vars: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
        autotiles: Dict[Tuple[str, ...], Optional[Tuple[int, ...]]] = {}
        for ncvar in netcdf_group.variables.values():
            for dim in ncvar.get_dims():
                if dim.name not in converter.dim_names:
                    converter.add_ncdim_to_dim_converter(
                        dim,
                        unlimited_dim_size,
                        dim_dtype,
                    )
            dims_to_vars[ncvar.dimensions].append(ncvar.name)
            chunks = ncvar.chunking()
            if not (chunks is None or chunks == "contiguous"):
                autotiles[ncvar.dimensions] = (
                    None
                    if ncvar.dimensions in autotiles
                    and chunks != autotiles.get(ncvar.dimensions)
                    else tuple(chunks)
                )
        if tiles is not None:
            autotiles.update(tiles)
        sorted_keys = sorted(dims_to_vars.keys())
        for count, dim_names in enumerate(sorted_keys):
            converter.add_array(
                f"array{count}", dim_names, tiles=autotiles.get(dim_names)
            )
            for var_name in dims_to_vars[dim_names]:
                converter.add_ncvar_to_attr_converter(
                    netcdf_group.variables[var_name], f"array{count}"
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
            output.write("Default NetCDF file: {self.default_input_file}\n")
        if self.default_group_path is not None:
            output.write("Deault NetCDF group path: {self.default_group_path}\n")
        return output.getvalue()

    def add_array(
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
        allows_duplicates: bool = False,
        sparse: bool = False,
    ):
        """Adds a new array to the CF dataspace.

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
            allows_duplicates: Specifies if multiple values can be stored at the same
                 coordinate. Only allowed for sparse arrays.
            sparse: Specifies if the array is a sparse TileDB array (true) or dense
                TileDB array (false).

        Raises:
            ValueError: Cannot add new array with given name.
        """
        try:
            self._check_new_array_name(array_name)
        except ValueError as err:
            raise ValueError(
                f"Cannot add new array with name '{array_name}'. {str(err)}"
            ) from err
        if not dims:
            dims = ("__scalars",)
            self.add_dim("__scalars", (0, 0), np.dtype(np.uint32))
        array_dims = tuple(self._dims[dim_name] for dim_name in dims)
        self._array_creators[array_name] = NetCDFArrayConverter(
            array_dims,
            cell_order,
            tile_order,
            capacity,
            tiles,
            coords_filters,
            dim_filters,
            offsets_filters,
            allows_duplicates,
            sparse,
        )
        for dim_name in dims:
            self._dim_to_arrays[dim_name].append(array_name)

    def add_ncdim_to_dim_converter(
        self,
        ncdim: netCDF4,
        unlimited_dim_size: int = 10000,
        dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
        dim_name: Optional[str] = None,
    ):
        dim_converter = NetCDFDimensionConverter.from_netcdf(
            ncdim,
            unlimited_dim_size,
            dtype,
            dim_name,
        )
        try:
            self._check_new_dim_name(dim_converter)
        except ValueError as err:
            raise ValueError(
                f"Cannot add new dimension '{dim_converter.name}'. {str(err)}"
            ) from err
        self._dims[dim_converter.name] = dim_converter
        if dim_converter.is_data_dim:
            self._data_dim_dataspace_names[
                dataspace_name(dim_converter.name)
            ] = dim_converter.name
        else:
            self._index_dim_dataspace_names[
                dataspace_name(dim_converter.name)
            ] = dim_converter.name

    def add_ncvar_to_attr_converter(
        self,
        ncvar: netCDF4.Variable,
        array_name: str,
        attr_name: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
        fill: Optional[Union[int, float, str]] = None,
        var: bool = False,
        nullable: bool = False,
        filters: Optional[tiledb.FilterList] = None,
    ):
        """Adds a new variable to attribute converter to an array in the CF dataspace.

        Each attribute name must be unique. It also cannot conflict with the name of a
        dimension in the array it is being added to, and the attribute's
        'dataspace name' (name after dropping the suffix ``.data`` or ``.index``) cannot
        conflict with the dataspace name of an existing attribute.

        Parameters:
            attr_name: Name of the new attribute that will be added.
            array_name: Name of the array the attribute will be added to.
            dtype: Numpy dtype of the new attribute.
            fill: Fill value for unset cells.
            var: Specifies if the attribute is variable length (automatic for
                byte/strings).
            nullable: Specifies if the attribute is nullable using validity tiles.
            filters: Specifies compression filters for the attribute.

        Raises:
            KeyError: The provided ``array_name`` does not correspond to an array in the
                dataspace.
            ValueError: Cannot create a new attribute with the provided ``attr_name``.
        """
        try:
            array_creator = self._array_creators[array_name]
        except KeyError as err:
            raise KeyError(
                f"Cannot add attribute to array '{array_name}'. No array named "
                f"'{array_name}' exists."
            ) from err
        if ncvar.ndim != 0 and ncvar.ndim != array_creator.ndim:
            raise ValueError(
                f"Cannot convert a NetCDF variable with {ncvar.ndim} dimensions to an "
                f"array with {array_creator.ndim} dimensions."
            )
        if ncvar.ndim == 0 and array_creator.ndim != 1:
            raise ValueError(
                f"Cannot add a scalar NetCDF variable to an array with "
                f"{array_creator.ndim} dimensions > 1."
            )
        ncvar_converter = NetCDFVariableConverter.from_netcdf(
            ncvar,
            attr_name,
            dtype,
            fill,
            var,
            nullable,
            filters,
        )
        try:
            self._check_new_attr_name(ncvar_converter.name)
        except ValueError as err:
            raise ValueError(
                f"Cannot add new attribute '{ncvar_converter.name}'. {str(err)}"
            ) from err
        array_creator.add_attr(ncvar_converter)
        self._attr_to_array[ncvar_converter.name] = array_name
        self._attr_dataspace_names[
            dataspace_name(ncvar_converter.name)
        ] = ncvar_converter.name

    def convert(
        self,
        group_uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        group_path: Optional[str] = None,
    ):
        """Creates a TileDB group and its arrays from the defined CF dataspace and
        copies data into them using the converter engine.

        Parameters:
            group_uri: Uniform resource identifier for the TileDB group to be created.
            key: If not ``None``, encryption key to decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            kwargs: Keyword arguments for the specific converter engine
        """
        self.create(group_uri, key, ctx)
        self.copy(group_uri, key, ctx, netcdf_group, input_file, group_path)

    def copy(
        self,
        group_uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        group_path: Optional[str] = None,
    ):
        """Copies data from a NetCDF group to a TileDB CF dataspace.

        This will copy data from a NetCDF group that is defined either by a
        :class:`netCDF4.Group` or by an input_file and group path. If neither the
        ``netcdf_group`` or ``input_file`` is specified, this will copy data from the
        input file ``self.default_input_file``.  If both ``netcdf_group`` and
        ``input_file`` are set, this method will prioritize using the NetCDF group set
        by ``netcdf_group``.

        Parameters:
            group_uri: Uniform resource identifier for the TileDB group data is being
                copied to.
            key: If not ``None``, encryption key to decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            input_file: If not ``None``, the NetCDF file to copy data from.
            group_path: If not ``None``, the path to the NetCDF group to copy data from.
            netcdf_group: If not ``None``, the NetCDF group to copy data from.
        """
        if netcdf_group is None:
            input_file = (
                input_file if input_file is not None else self.default_input_file
            )
            group_path = (
                group_path if group_path is not None else self.default_group_path
            )
        with open_netcdf_group(netcdf_group, input_file, group_path) as nc_group:
            # Copy group metadata
            with Group(group_uri, mode="w", key=key, ctx=ctx) as group:
                for group_key in nc_group.ncattrs():
                    copy_metadata_item(group.meta, nc_group, group_key)
            # Copy variables and variable metadata to arrays
            for array_name, array_creator in self._array_creators.items():
                if isinstance(array_creator, NetCDFArrayConverter):
                    with Group(
                        group_uri, mode="w", array=array_name, key=key, ctx=ctx
                    ) as tiledb_group:
                        array_creator.copy(nc_group, tiledb_group)


def copy_metadata_item(meta, netcdf_item, key):
    """Copies a NetCDF attribute from a NetCDF group or variable to TileDB metadata.

    Parameters:
        meta: TileDB metadata object to copy to.
        netcdf_item: NetCDF variable or group to copy from.
        key: Name of the NetCDF attribute that is being copied.
    """
    value = netcdf_item.getncattr(key)
    if isinstance(value, np.ndarray):
        value = tuple(value.tolist())
    elif isinstance(value, np.generic):
        value = (value.tolist(),)
    try:
        meta[key] = value
    except ValueError as err:
        warnings.warn(f"Failed to set group metadata {value} with error: " f"{err}")


@contextmanager
def open_netcdf_group(
    group: Optional[Union[netCDF4.Dataset, netCDF4.Group]] = None,
    input_file: Optional[Union[str, Path]] = None,
    group_path: Optional[str] = None,
):
    """Context manager for opening a NetCDF group.

    If both an input file and group are provided, this function will prioritize
    opening from the group.

    Parameters:
        group: A NetCDF group to read from.
        input_file: A NetCDF file to read from.
        group_path: The path to the NetCDF group to read from in a NetCDF file. Use
            ``'/'`` to specify the root group.
    """
    if group is not None:
        if not isinstance(group, (netCDF4.Dataset, netCDF4.Group)):
            raise TypeError(
                f"Invalid input: group={group} of type {type(group)} is not a netCDF "
                f"Group or or Dataset."
            )
        yield group
    else:
        if input_file is None:
            raise ValueError(
                "An input file must be provided; no default input file was set."
            )
        if group_path is None:
            raise ValueError(
                "A group path must be provided; no default group path was set. Use "
                "``'/'`` for the root group."
            )
        root_group = netCDF4.Dataset(input_file)
        root_group.set_auto_maskandscale(False)
        try:
            netcdf_group = root_group
            if group_path != "/":
                for child_group_name in group_path.strip("/").split("/"):
                    netcdf_group = netcdf_group.groups[child_group_name]
            yield netcdf_group
        finally:
            root_group.close()
