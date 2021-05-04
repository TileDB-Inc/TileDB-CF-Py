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
from ..creator import AttrCreator, DataspaceCreator, SharedDim, dataspace_name

_DEFAULT_INDEX_DTYPE = np.dtype("uint64")
COORDINATE_SUFFIX = ".data"


@dataclass
class NetCDFDimensionConverter(SharedDim):
    """Data for converting from a NetCDF dimension to a TileDB dimension.

    Parameters:
        input_name: The name of the input NetCDF dimension.
        input_size: The size of the input NetCDF dimension.
        is_unlimited: If the input NetCDF dimension is unlimited (can grow in size).
        output_name: The name of output TileDB dimension.
        output_domain: The interval the output TileDB dimension is defined on.
        output_dtype: The numpy dtype of the values and domain of the output TileDB
            dimension.

    Attributes:
        input_name: The name of the input NetCDF dimension.
        input_size: The size of the input NetCDF dimension.
        is_unlimited: If the input NetCDF dimension is unlimited (can grow in size).
        output_name: The name of output TileDB dimension.
        output_domain: The interval the output TileDB dimension is defined on.
        output_dtype: The numpy dtype of the values and domain of the output TileDB
            dimension.
    """

    input_name: str
    input_size: int
    is_unlimited: bool

    def __repr__(self):
        if self.is_unlimited:
            return (
                f"Dimension(name={self.input_name}, size=unlimited) -> SharedDim(name="
                f"{self.name}, domain=[{self.domain[0]}, {self.domain[1]}], dtype="
                f"'{self.dtype!s}')"
            )
        else:
            return (
                f"Dimension(name={self.input_name}, size={self.input_size}) -> Dim(name"
                f"={self.name}, domain=[{self.domain[0]}, {self.domain[1]}], dtype="
                f"'{self.dtype!s}')"
            )

    @classmethod
    def from_netcdf(
        cls,
        dim: netCDF4.Dimension,
        unlimited_dim_size: int,
        dtype: np.dtype,
    ):
        """Returns a :class:`NetCDFDimensionConverter` from a
        :class:`netcdf4.Dimension`.

        Parameters:
            dim: The input netCDF4 dimension.
            inlimited_dim_size: The size of the domain of the output TileDB dimension
                when the input NetCDF dimension is unlimited.
            dtype: The numpy dtype of the values and domain of the output TileDB
                dimension.
        """
        size = dim.size if not dim.isunlimited() else unlimited_dim_size
        return cls(
            dim.name,
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
        input_name: The name of the input NetCDF variable.
        input_dtype: The numpy dtype of the input NetCDF variable.
        input_fill: Fill value for unset values in the input NetCDF variable.
        input_chunks: If not ``None``, the chunk size of the input NetCDF variable.
        output_name: The name of the output TileDB attribute.
        output_dtype: The numpy dtype of the output TileDB attribute.
        output_fill: Fill value for unset values i the output TileDB attribute.

    Attributes:
        input_name: The name of the input NetCDF variable.
        input_dtype: The numpy dtype of the input NetCDF variable.
        input_fill: Fill value for unset values in the input NetCDF variable.
        input_chunks: If not ``None``, the chunk size of the input NetCDF variable.
        output_name: The name of the output TileDB attribute.
        output_dtype: The numpy dtype of the output TileDB attribute.
        output_fill: Fill value for unset values i the output TileDB attribute.
    """

    input_name: Optional[str] = None
    input_dtype: Optional[np.dtype] = None
    input_fill: Optional[Union[int, float, str]] = None
    input_chunks: Optional[Tuple[int, ...]] = None

    def __repr__(self):
        return (
            f"Variable(name={self.input_name}, dtype={self.input_dtype}, _FillValue="
            f"{self.input_fill}, chunks={self.input_chunks}) ->  {super().__repr__()}"
        )

    @classmethod
    def from_netcdf(cls, var: netCDF4.Variable):
        """Returns a :class:`NetCDFVariableConverter` from a :class:`netCDF4.Variable`.

        Parameters:
            var: The input netCDF4 variable.
        """
        fill = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
        chunks = var.chunking()
        chunks = None if chunks == "contiguous" or chunks is None else tuple(chunks)
        attr_name = (
            var.name if var.name not in var.dimensions else var.name + COORDINATE_SUFFIX
        )
        return cls(
            attr_name,
            var.dtype,
            fill,
            input_name= var.name,
            input_dtype=var.dtype,
            input_fill=fill,
            input_chunks=chunks,
        )


@dataclass
class NetCDFArrayConverter:
    """Data for converting a collection of NetCDF variables defined over the same
    NetCDF dimensions to a TileDB array.

    On initialization, if the tile size is not explicitly set, it will be updated from
    the NetCDF variable chunk sizes. The tile size will be set to the chunk size of the
    input NetCDF variables if:
        * the input for ``tiles=None``,
        * at least one variable has chunks not equal to ``None``,
        * all variables with chunks not equal to ``None`` have the same chunks.
    To skip checking the size of tiles on initialization, use ``tiles=tuple()``. This
    will set ``tiles=None`` after initialization.

    Parameters:
        dimensions: An ordered tuple of the converter for the dimensions the NetCDF
            variables are defined on.
        variables: A list of converters from NetCDF variables to TileDB attributes
            where the variables are all defined on the same dimensions.
        tiles: If not ``None``, the tile sizes for the TileDB dimensions.

    Attributes:
        dimensions: An ordered tuple of the converter for the dimensions the NetCDF
            variables are defined on.
        variables: A list of converters from NetCDF variables to TileDB attributes
            where the variables are all defined on the same dimensions.
        tiles: If not ``None``, the tile sizes for the TileDB dimensions.
    """

    dimensions: Tuple[NetCDFDimensionConverter, ...]
    variables: List[NetCDFVariableConverter]
    tiles: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.tiles is not None:
            if not self.tiles:
                self.tiles = None
        else:
            chunks = [
                var.input_chunks
                for var in self.variables
                if var.input_chunks is not None
            ]
            if len(set(chunks)) == 1:
                self.tiles = chunks[0]


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
        tiles: Dict[Tuple[str, ...], Tuple[int, ...]] = None,
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
        tiles: Dict[Tuple[str, ...], Tuple[int, ...]] = None,
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
        dimensions = {}
        variables = {}
        tiles = tiles if tiles is not None else {}
        partition = defaultdict(list)  # ignore: type
        for var in netcdf_group.variables.values():
            for dim in var.get_dims():
                if dim.name not in dimensions:
                    dimensions[dim.name] = NetCDFDimensionConverter.from_netcdf(
                        dim,
                        unlimited_dim_size,
                        dim_dtype,
                    )
            variable_converter = NetCDFVariableConverter.from_netcdf(var)
            variables[variable_converter.name] = variable_converter
            partition[var.dimensions].append(variable_converter)
        sorted_keys = sorted(partition.keys())
        arrays = {
            f"array{count}": NetCDFArrayConverter(
                tuple(dimensions[name] for name in dim_names),
                partition[dim_names],
                tiles.get(dim_names),
            )
            for count, dim_names in enumerate(sorted_keys)
        }
        return cls(
            dimensions,
            variables,
            arrays,
            default_input_file,
            default_group_path,
        )

    def __init__(
        self,
        dimensions: Dict[str, SharedDim],
        variables: Dict[str, NetCDFVariableConverter],
        arrays: Dict[str, NetCDFArrayConverter],
        default_input_file: Optional[Union[str, Path]] = None,
        default_group_path: Optional[str] = None,
    ):
        self._variables = variables
        self._arrays = arrays
        self._default_input_file = default_input_file
        self._default_group_path = default_group_path
        super().__init__()
        self._dims = dimensions
        self._index_dim_dataspace_names = {
            dataspace_name(dim.name): dim.name for dim in self._dims.values()
        }
        if len(self._index_dim_dataspace_names) != len(self._dims):
            raise ValueError(
                "Failed to create NetCDF4EngineConverter; there is a dimension with a "
                "duplicate dataspace name."
            )
        for array_name, array_converter in self._arrays.items():
            super().add_array(
                array_name=array_name,
                dims=tuple(dim.name for dim in array_converter.dimensions),
                tiles=array_converter.tiles,
            )
            for var_converter in array_converter.variables:
                super().add_attr(
                    var_converter.name,
                    array_name,
                    dtype=var_converter.dtype,
                    fill=var_converter.fill,
                )

    def __repr__(self):
        output = StringIO()
        output.write(f"{super().__repr__()}\n")
        if self._default_input_file is not None:
            output.write("Default NetCDF file: {self._default_input_file}\n")
        if self._default_group_path is not None:
            output.write("Deault NetCDF group path: {self._default_group_path}\n")
        return output.getvalue()

    def add_array(
        self,
        array_name: str,
        dims: Union[str, Sequence[str]],
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
        """Adding external arrays is not yet supported."""
        raise NotImplementedError("Adding external arrays is not yet supported.")

    def add_attr(
        self,
        attr_name: str,
        array_name: str,
        dtype: np.dtype,
        fill: Optional[Union[int, float, str]] = None,
        var: bool = False,
        nullable: bool = False,
        filters: Optional[tiledb.FilterList] = None,
    ):
        """Adding external attributes is not yet supported."""
        raise NotImplementedError("Adding external attributes is not yet supported.")

    def convert(
        self,
        group_uri: str,
        key: Optional[Union[Dict[str, str], str]] = None,
        ctx: Optional[tiledb.Ctx] = None,
        netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        group_path: Optional[str] = None,
    ):
        """Creates a TileDB group and its arrays from the defined CF dataspace and
        copies data into them using the converter engine.

        Parameters:
            group_uri: Uniform resource identifier for the TileDB group to be created.
            key: If not ``None``, encryption key, or dictionary of encryption keys, to
                decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            kwargs: Keyword arguments for the specific converter engine
        """
        self.create(group_uri, key, ctx)
        self.copy(group_uri, key, ctx, netcdf_group, input_file, group_path)

    def copy(
        self,
        group_uri: str,
        key: Optional[Union[Dict[str, str], str]] = None,
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
            key: If not ``None``, encryption key, or dictionary of encryption keys, to
                decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            input_file: If not ``None``, the NetCDF file to copy data from.
            group_path: If not ``None``, the path to the NetCDF group to copy data from.
            netcdf_group: If not ``None``, the NetCDF group to copy data from.
        """
        if netcdf_group is None:
            input_file = (
                input_file if input_file is not None else self._default_input_file
            )
            group_path = (
                group_path if group_path is not None else self._default_group_path
            )
        with open_netcdf_group(netcdf_group, input_file, group_path) as nc_group:
            # Copy group metadata
            with Group(group_uri, mode="w", key=key, ctx=ctx) as group:
                for group_key in nc_group.ncattrs():
                    copy_metadata_item(group.meta, nc_group, group_key)
            # Copy variables and variable metadata to arrays
            for array_name, array_converter in self._arrays.items():
                with Group(
                    group_uri, mode="w", array=array_name, key=key, ctx=ctx
                ) as group:
                    data = {}
                    if not array_converter.variables:
                        continue
                    for var_converter in array_converter.variables:
                        try:
                            variable = nc_group.variables[var_converter.input_name]
                        except KeyError as err:
                            raise KeyError(
                                f"Variable {var_converter.input_name} not found in "
                                f"requested NetCDF group."
                            ) from err
                        data[var_converter.name] = variable[...]
                        attr_meta = group.get_attr_metadata(var_converter.name)
                        for meta_key in variable.ncattrs():
                            copy_metadata_item(attr_meta, variable, meta_key)
                    dim_slice = tuple(slice(dim.size) for dim in variable.get_dims())
                    group.array[dim_slice or slice(None)] = data

    def rename_array(self, original_name: str, new_name: str):
        """Renames an array in the output TileDB CF dataspace.

        Parameters:
            original_name: Current name of the array to be renamed.
            new_name: New name the array will be renamed to.
        """
        if new_name in self._arrays:
            raise ValueError(
                f"Cannot rename array '{original_name}' to '{new_name}'. An array with "
                f"that name already exists."
            )
        super().rename_array(original_name, new_name)
        self._arrays[new_name] = self._arrays.pop(original_name)

    def rename_attr(self, original_name: str, new_name: str):
        """Renames an attribute in the output TileDB CF dataspace.

        Parameters:
            original_name: Current name of the attribute to be renamed.
            new_name: New name the attribute will be renamed to.
        """
        if new_name in self._variables:
            raise ValueError(
                f"Cannot rename attribute '{original_name}' to '{new_name}'. An "
                f"attribute with that name already exists."
            )
        super().rename_attr(original_name, new_name)
        self._variables[new_name] = self._variables.pop(original_name)
        self._variables[new_name].name = new_name


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
