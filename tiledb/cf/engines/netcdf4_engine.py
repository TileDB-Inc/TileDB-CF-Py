# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Classes for converting NetCDF4 files to TileDB."""

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import netCDF4
import numpy as np

import tiledb

from ..core import AttrMetadata, Group
from ..creator import (
    ArrayCreator,
    AttrCreator,
    DataspaceCreator,
    DType,
    SharedDim,
    dataspace_name,
)

_DEFAULT_INDEX_DTYPE = np.dtype("uint64")
COORDINATE_SUFFIX = ".data"


class NetCDFDimConverter(ABC):
    @abstractmethod
    def get_values(
        self, netcdf_group: netCDF4.Dataset, sparse: bool
    ) -> Union[np.ndarray, slice]:
        """Returns the values of the NetCDF dimension that is being copied, or None if
        the dimension is of size 0.

        Parameters:
            netcdf_group: NetCDF group to get the dimension values from.
            sparse: ``True`` if copying into a sparse array and ``False`` if copying
                into a dense array.

        Returns:
            The coordinates needed for querying the create TileDB dimension in the form
                of a numpy array if sparse is ``True`` and a slice otherwise.
        """


@dataclass
class NetCDFCoordToDimConverter(SharedDim, NetCDFDimConverter):

    input_name: str
    input_dtype: np.dtype
    input_add_offset: Optional[DType] = None
    input_scale_factor: Optional[DType] = None
    input_unsigned: Optional[bool] = None

    def __repr__(self):
        return (
            f"Variable(name={self.input_name}, dtype={self.input_dtype}) -> "
            f"{super().__repr__()}"
        )

    @classmethod
    def from_netcdf(
        cls,
        var: netCDF4.Variable,
        dim_name: Optional[str] = None,
        domain: Optional[Tuple[DType, DType]] = None,
    ):
        """Returns a :class:`NetCDFCoordToDimConverter` from a
        :class:`netcdf4.Variable`.

        Parameters:
            var: The input netCDF4 variable to convert.
            dim_name: The name of the output TileDB dimension. If ``None``, the name
                will be the same as the name of the input NetCDF variable.
        """
        if len(var.dimensions) != 1:
            raise ValueError(
                f"Cannot create dimension from variable '{var.name}' with shape "
                f"{var.shape}. Coordinate variables must have only one dimension."
            )
        add_offset = get_ncattr(var, "add_offset")
        scale_factor = get_ncattr(var, "scale_factor")
        unsigned = get_ncattr(var, "_Unsigned")
        if add_offset is not None or scale_factor is not None or unsigned is not None:
            raise NotImplementedError(
                f"Cannot convert variable {var.name} into a TileDB dimension. Support "
                f"for converting scaled coordinates has not yet been implemented."
            )
        dtype = np.dtype(var.dtype)
        if domain is None:
            domain = (None, None)
        return cls(
            dim_name if dim_name is not None else var.name,
            (None, None),
            dtype,
            input_name=var.name,
            input_dtype=dtype,
            input_add_offset=add_offset,
            input_scale_factor=scale_factor,
            input_unsigned=unsigned,
        )

    def get_values(
        self,
        netcdf_group: netCDF4.Dataset,
        sparse: bool,
    ):
        """Returns the values of the NetCDF coordinate that is being copied, or
        None if the coordinate is of size 0.

        Parameters:
            netcdf_group: NetCDF group to get the coordinate values from.
            sparse: ``True`` if copying into a sparse array and ``False`` if copying
                into a dense array.

        Returns:
            The coordinate values needed for querying the TileDB dimension in the
                form a numpy array.
        """
        if not sparse:
            raise NotImplementedError(
                "Support for copying NetCDF coordinates to dense arrays has not "
                "been implemented."
            )
        try:
            variable = netcdf_group.variables[self.input_name]
        except KeyError as err:
            raise KeyError(
                f"The variable '{self.input_name}' was not found in the provided "
                f"NetCDF group. Cannot copy date from variable '{self.input_name}' to "
                f"TileDB dimension '{self.name}'."
            ) from err
        if variable.ndim != 1:
            raise ValueError(
                f"The variable '{self.input_name}' with {variable.ndim} dimensions is "
                f"not a valid NetCDF coordinate. Cannot copy data from variable "
                f"'{self.input_name}' to TileDB dimension '{self.name}'."
            )
        if variable.get_dims()[0].size < 1:
            return None
        return variable[:]

    @property
    def is_data_dim(self) -> bool:
        return True

    @property
    def is_index_dim(self) -> bool:
        return False


@dataclass
class NetCDFDimToDimConverter(SharedDim, NetCDFDimConverter):
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
        """Returns a :class:`NetCDFDimToDimConverter` from a
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

    def get_values(
        self, netcdf_group: netCDF4.Dataset, sparse: bool
    ) -> Union[np.ndarray, slice]:
        """Returns the values of the NetCDF dimension that is being copied, or None if
        the dimension is of size 0.

        Parameters:
            netcdf_group: NetCDF group to get the dimension values from.
            sparse: ``True`` if copying into a sparse array and ``False`` if copying
                into a dense array.

        Returns:
            The coordinates needed for querying the create TileDB dimension in the form
                of a numpy array if sparse is ``True`` and a slice otherwise.
        """
        group = netcdf_group
        while group is not None:
            if self.input_name in group.dimensions:
                dim = group.dimensions[self.input_name]
                if dim.size == 0:
                    raise ValueError(
                        f"Cannot copy dimension data from NetCDF dimension "
                        f"'{self.input_name}' to TileDB dimension '{self.name}'. The "
                        f"NetCDF dimension is of size 0; there is no data to copy."
                    )
                if self.domain[1] is not None and dim.size - 1 > self.domain[1]:
                    raise IndexError(
                        f"Cannot copy dimension data from NetCDF dimension "
                        f"'{self.input_name}' to TileDB dimension '{self.name}'. The "
                        f"NetCDF dimension size of {dim.size} does not fit in the "
                        f"domain {self.domain} of the TileDB dimension."
                    )
                if sparse:
                    return np.arange(dim.size)
                return slice(dim.size)
            group = group.parent
        raise KeyError(
            f"Unable to copy NetCDF dimension '{self.input_name}' to the TileDB "
            f"dimension '{self.name}'. No NetCDF dimension with that name exists in "
            f"the NetCDF group '{netcdf_group.path}' or its parent groups."
        )


@dataclass
class NetCDFScalarDimConverter(SharedDim, NetCDFDimConverter):
    """Data for converting from a NetCDF dimension to a TileDB dimension.

    Parameters:
        name: Name of the TileDB dimension.
        domain: The (inclusive) interval on which the dimension is valid.
        dtype: The numpy dtype of the values and domain of the dimension.

    Attributes:
        name: Name of the TileDB dimension.
        domain: The (inclusive) interval on which the dimension is valid.
        dtype: The numpy dtype of the values and domain of the dimension.
    """

    def __repr__(self):
        return f" Scalar dimensions -> {super().__repr__()}"

    def get_values(
        self, netcdf_group: netCDF4.Dataset, sparse: bool
    ) -> Union[np.ndarray, slice]:
        """Get dimension values from a NetCDF group.

        Parameters:
            netcdf_group: NetCDF group to get the dimension values from.
            sparse: ``True`` if copying into a sparse array and ``False`` if copying
                into a dense array.

        Returns:
            The coordinates needed for querying the create TileDB dimension in the form
                of a numpy array if sparse is ``True`` and a slice otherwise.
        """
        if sparse:
            return np.array([0])
        return slice(1)

    @classmethod
    def create(cls, dim_name: str, dtype: np.dtype):
        """Returns a :class:`NetCDFDimToDimConverter` from a
        :class:`netcdf4.Dimension`.

        Parameters:
            dim_name: The name of the output TileDB dimension.
            dtype: The numpy dtype of the values and domain of the output TileDB
                dimension.
        """
        return cls(dim_name, (0, 0), np.dtype(dtype))


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

    def __post_init__(self):
        for dim_creator in self._dim_creators:
            if not isinstance(dim_creator.base, NetCDFDimConverter):
                raise TypeError(
                    f"Cannot create NetCDFArrayConverter with dimension creator "
                    f"{repr(dim_creator)}. The dimension does not describe a NetCDF "
                    f"dimension."
                )

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
        tiledb_array: tiledb.Array,
    ):
        """Copies data from a NetCDF group to a TileDB CF array.

        Parameters:
            netcdf_group: The NetCDF group to copy data from.
            tiledb_arary: The TileDB array to copy data into. The array must be open
                in write mode.
        """
        dim_query = []
        for dim_creator in self._dim_creators:
            assert isinstance(dim_creator.base, NetCDFDimConverter)
            dim_query.append(
                dim_creator.base.get_values(netcdf_group, sparse=self.sparse)
            )
        data = {}
        for attr_converter in self._attr_creators.values():
            assert isinstance(attr_converter, NetCDFVariableConverter)
            try:
                variable = netcdf_group.variables[attr_converter.input_name]
            except KeyError as err:
                raise KeyError(
                    f"Variable {attr_converter.input_name} not found in "
                    f"requested NetCDF group."
                ) from err
            data[attr_converter.name] = variable[...]
            attr_meta = AttrMetadata(tiledb_array.meta, attr_converter.name)
            for meta_key in variable.ncattrs():
                copy_metadata_item(attr_meta, variable, meta_key)
        tiledb_array[tuple(dim_query)] = data


class NetCDF4ConverterEngine(DataspaceCreator):
    """Converter for NetCDF to TileDB using netCDF4."""

    @classmethod
    def from_file(
        cls,
        input_file: Union[str, Path],
        group_path: str = "/",
        unlimited_dim_size: int = 10000,
        dim_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
        tiles_by_var: Optional[Dict[str, Optional[Sequence[int]]]] = None,
        tiles_by_dims: Optional[Dict[Sequence[str], Optional[Sequence[int]]]] = None,
        coords_to_dims: bool = True,
        collect_attrs: bool = True,
    ):
        """Returns a :class:`NetCDF4ConverterEngine` from a group in a NetCDF file.

        Parameters:
            input_file: The input NetCDF file to generate the converter engine from.
            group_path: The path to the NetCDF group to copy data from. Use ``'/'`` for
                the root group.
            unlimited_dim_size: The size of the domain for TileDB dimensions created
                from unlimited NetCDF dimensions.
            dim_dtype: The numpy dtype for TileDB dimensions.
            tiles_by_var: A map from the name of a NetCDF variable to the tiles of the
                dimensions of the variable in the generated NetCDF array.
            tiles_by_dims: A map from the name of NetCDF dimensions defining a variable
                to the tiles of those dimensions in the generated NetCDF array.
            coords_to_dims: If ``True``, convert the NetCDF coordinate variable into a
                TileDB dimension for sparse arrays. Otherwise, convert the coordinate
                dimension into a TileDB dimension and the coordinate variable into a
                TileDB attribute.
            collect_attrs: If True, store all attributes with the same dimensions
                in the same array. Otherwise, store each attribute in a scalar array.
        """
        with open_netcdf_group(
            input_file=input_file,
            group_path=group_path,
        ) as group:
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
            )

    @classmethod
    def from_group(
        cls,
        netcdf_group: netCDF4.Group,
        unlimited_dim_size: int = 10000,
        dim_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
        tiles_by_var: Optional[Dict[str, Optional[Sequence[int]]]] = None,
        tiles_by_dims: Optional[Dict[Sequence[str], Optional[Sequence[int]]]] = None,
        coords_to_dims: bool = True,
        collect_attrs: bool = True,
        default_input_file: Optional[Union[str, Path]] = None,
        default_group_path: Optional[str] = None,
    ):
        """Returns a :class:`NetCDF4ConverterEngine` from a :class:`netCDF4.Group`.

        Parameters:
            group: The NetCDF group to convert.
            unlimited_dim_size: The size of the domain for TileDB dimensions created
                from unlimited NetCDF dimensions.
            dim_dtype: The numpy dtype for TileDB dimensions.
            tiles_by_var: A map from the name of a NetCDF variable to the tiles of the
                dimensions of the variable in the generated NetCDF array.
            tiles_by_dims: A map from the name of NetCDF dimensions defining a variable
                to the tiles of those dimensions in the generated NetCDF array.
            coords_to_dims: If ``True``, convert the NetCDF coordinate variable into a
                TileDB dimension for sparse arrays. Otherwise, convert the coordinate
                dimension into a TileDB dimension and the coordinate variable into a
                TileDB attribute.
            collect_attrs: If ``True``, store all attributes with the same dimensions
                in the same array. Otherwise, store each attribute in a scalar array.
            default_input_file: If not ``None``, the default NetCDF input file to copy
                data from.
            default_group_path: If not ``None``, the default NetCDF group to copy data
                from. Use ``'/'`` to specify the root group.
        """
        if collect_attrs:
            return cls._from_group_to_collected_attrs(
                netcdf_group,
                unlimited_dim_size,
                dim_dtype,
                tiles_by_var,
                tiles_by_dims,
                coords_to_dims=coords_to_dims,
                default_input_file=default_input_file,
                default_group_path=default_group_path,
            )
        return cls._from_group_to_attr_per_array(
            netcdf_group,
            unlimited_dim_size,
            dim_dtype,
            tiles_by_var,
            tiles_by_dims,
            coords_to_dims=coords_to_dims,
            default_input_file=default_input_file,
            default_group_path=default_group_path,
        )

    @classmethod
    def _from_group_to_attr_per_array(
        cls,
        netcdf_group: netCDF4.Group,
        unlimited_dim_size: int = 10000,
        dim_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
        tiles_by_var: Optional[Dict[str, Optional[Sequence[int]]]] = None,
        tiles_by_dims: Optional[Dict[Sequence[str], Optional[Sequence[int]]]] = None,
        coords_to_dims: bool = True,
        default_input_file: Optional[Union[str, Path]] = None,
        default_group_path: Optional[str] = None,
    ):
        """Returns a :class:`NetCDF4ConverterEngine` from a :class:`netCDF4.Group`.

        Parameters:
            group: The NetCDF group to convert.
            unlimited_dim_size: The size of the domain for TileDB dimensions created
                from unlimited NetCDF dimensions.
            dim_dtype: The numpy dtype for TileDB dimensions.
            tiles_by_var: A map from the name of a NetCDF variable to the tiles of the
                dimensions of the variable in the generated NetCDF array. The tile
                sizes defined by this dict take priority over the ``tiles_by_dims``
                parameter and the NetCDF variable chunksize.
            tiles_by_dims: A map from the name of NetCDF dimensions defining a variable
                to the tiles of those dimensions in the generated NetCDF array. The
                tile size defined by this dict are used if they are not defined by the
                parameter ``tiles_by_var``.
            coords_to_dims: If ``True``, convert the NetCDF coordinate variable into a
                TileDB dimension for sparse arrays. Otherwise, convert the coordinate
                dimension into a TileDB dimension and the coordinate variable into a
                TileDB attribute.
            default_input_file: If not ``None``, the default NetCDF input file to copy
                data from.
            default_group_path: If not ``None``, the default NetCDF group to copy data
                from. Use ``'/'`` to specify the root group.
        """
        converter = cls(default_input_file, default_group_path)
        if coords_to_dims:
            raise NotImplementedError
        for ncvar in netcdf_group.variables.values():
            for dim in ncvar.get_dims():
                if dim.name not in converter.dim_names:
                    converter.add_dim_to_dim_converter(
                        dim,
                        unlimited_dim_size,
                        dim_dtype,
                    )
            if not ncvar.dimensions:
                array_name = (
                    "scalars" if "scalars" not in netcdf_group.variables else "_scalars"
                )
                if array_name not in converter.array_names:
                    converter.add_scalar_dim_converter("__scalars", dim_dtype)
                    converter.add_array("scalars", ("__scalars",))
            else:
                if tiles_by_var is not None and ncvar.name in tiles_by_var:
                    array_tiles = tiles_by_var[ncvar.name]
                elif tiles_by_dims is not None and ncvar.dimensions in tiles_by_dims:
                    array_tiles = tiles_by_dims[ncvar.dimensions]
                else:
                    chunks = ncvar.chunking()
                    array_tiles = (
                        None
                        if chunks is None or chunks == "contiguous"
                        else tuple(chunks)
                    )
                array_name = ncvar.name
                converter.add_array(
                    array_name,
                    ncvar.dimensions,
                    tiles=array_tiles,
                )
            converter.add_var_to_attr_converter(ncvar, array_name)
        return converter

    @classmethod
    def _from_group_to_collected_attrs(
        cls,
        netcdf_group: netCDF4.Group,
        unlimited_dim_size: int = 10000,
        dim_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
        tiles_by_var: Optional[Dict[str, Optional[Sequence[int]]]] = None,
        tiles_by_dims: Optional[Dict[Sequence[str], Optional[Sequence[int]]]] = None,
        coords_to_dims: bool = True,
        default_input_file: Optional[Union[str, Path]] = None,
        default_group_path: Optional[str] = None,
    ):
        """Returns a :class:`NetCDF4ConverterEngine` from a :class:`netCDF4.Group`.

        Parameters:
            group: The NetCDF group to convert.
            unlimited_dim_size: The size of the domain for TileDB dimensions created
                from unlimited NetCDF dimensions.
            dim_dtype: The numpy dtype for TileDB dimensions.
            tiles_by_var: A map from the name of a NetCDF variable to the tiles of the
                dimensions of the variable in the generated NetCDF array. This will
                take priority over NetCDF variable chunksize.
            tiles_by_dims: A map from the name of NetCDF dimensions defining a variable
                to the tiles of those dimensions in the generated NetCDF array. This
                will take priority over tile sizes defined by the ``tiles_by_var``
                parameter and the NetCDF variable chunksize.
            coords_to_dims: If ``True``, convert the NetCDF coordinate variable into a
                TileDB dimension for sparse arrays. Otherwise, convert the coordinate
                dimension into a TileDB dimension and the coordinate variable into a
                TileDB attribute.
            default_input_file: If not ``None``, the default NetCDF input file to copy
                data from.
            default_group_path: If not ``None``, the default NetCDF group to copy data
                from. Use ``'/'`` to specify the root group.
        """
        converter = cls(default_input_file, default_group_path)
        coord_names = []
        dims_to_vars: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
        autotiles: Dict[Sequence[str], Optional[Sequence[int]]] = {}
        tiles_by_dims = {} if tiles_by_dims is None else tiles_by_dims
        tiles_by_var = {} if tiles_by_var is None else tiles_by_var
        # Add data/coordinate dimension to converter, partition variables into arrays,
        # and compute the tile sizes for array dimensions.
        for ncvar in netcdf_group.variables.values():
            if coords_to_dims and ncvar.ndim == 1 and ncvar.dimensions[0] == ncvar.name:
                converter.add_coord_to_dim_converter(ncvar)
                coord_names.append(ncvar.name)
            else:
                if not ncvar.dimensions and "__scalars" not in converter.dim_names:
                    converter.add_scalar_dim_converter("__scalars", dim_dtype)
                dim_names = ncvar.dimensions if ncvar.dimensions else ("__scalars",)
                dims_to_vars[dim_names].append(ncvar.name)
                chunks = tiles_by_var.get(ncvar.name, ncvar.chunking())
                if not (chunks is None or chunks == "contiguous"):
                    chunks = tuple(chunks)
                    autotiles[dim_names] = (
                        None
                        if dim_names in autotiles and chunks != autotiles.get(dim_names)
                        else chunks
                    )
        autotiles.update(tiles_by_dims)
        # Add index dimensions to converter.
        for ncvar in netcdf_group.variables.values():
            for dim in ncvar.get_dims():
                if dim.name not in converter.dim_names:
                    converter.add_dim_to_dim_converter(
                        dim,
                        unlimited_dim_size,
                        dim_dtype,
                    )
        # Add arrays and attributes to the converter.
        for count, dim_names in enumerate(sorted(dims_to_vars.keys())):
            is_sparse = any(dim_name in coord_names for dim_name in dim_names)
            converter.add_array(
                f"array{count}",
                dim_names,
                tiles=autotiles.get(dim_names),
                sparse=is_sparse,
            )
            for var_name in dims_to_vars[dim_names]:
                converter.add_var_to_attr_converter(
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
            output.write(f"Default NetCDF file: {self.default_input_file}\n")
        if self.default_group_path is not None:
            output.write(f"Deault NetCDF group path: {self.default_group_path}\n")
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

    def add_coord_to_dim_converter(
        self,
        var: netCDF4.Variable,
        dim_name: Optional[str] = None,
    ):
        """Adds a new NetCDF coordinate to TileDB dimension converter.

        Parameters:
            var: NetCDF coordinate variable to be converted.
            dim_name: If not ``None``, name to use for the TileDB dimension.
        """
        dim_converter = NetCDFCoordToDimConverter.from_netcdf(var, dim_name=dim_name)
        if dim_converter.name == "__scalars":
            raise NotImplementedError(
                "Support for converting a NetCDF file with reserved dimension "
                "name '__scalars' is not implemented."
            )
        super()._add_shared_dimension(dim_converter)

    def add_dim_to_dim_converter(
        self,
        ncdim: netCDF4.Dimension,
        unlimited_dim_size: int = 10000,
        dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
        dim_name: Optional[str] = None,
    ):
        """Adds a new NetCDF dimension to TileDB dimension converter.

        Parameters:
            ncdim: NetCDF dimension to be converted.
            unlimited_dim_size: Size for an unlimited dimension type.
            dtype: Numpy type to use for the NetCDF dimension.
            dim_name: If not ``None``, output name of the TileDB dimension.
        """
        dim_converter = NetCDFDimToDimConverter.from_netcdf(
            ncdim,
            unlimited_dim_size,
            dtype,
            dim_name,
        )
        if dim_converter.name == "__scalars":
            raise NotImplementedError(
                "Support for converting a NetCDF file with reserved dimension "
                "name '__scalars' is not implemented."
            )
        super()._add_shared_dimension(dim_converter)

    def add_scalar_dim_converter(
        self,
        dim_name: str = "__scalars",
        dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
    ):
        """Adds a new NetCDF scalar dimension.

        Parameters:
            dim_name: Output name of the dimension.
            dtype: Numpy type to use for the scalar dimension
        """
        dim_converter = NetCDFScalarDimConverter.create(dim_name, dtype)
        super()._add_shared_dimension(dim_converter)

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
        except KeyError as err:  # pragma: no cover
            raise KeyError(
                f"Cannot add attribute to array '{array_name}'. No array named "
                f"'{array_name}' exists."
            ) from err
        if ncvar.ndim != 0 and ncvar.ndim != array_creator.ndim:  # pragma: no cover
            raise ValueError(
                f"Cannot convert a NetCDF variable with {ncvar.ndim} dimensions to an "
                f"array with {array_creator.ndim} dimensions."
            )
        if ncvar.ndim == 0 and array_creator.ndim != 1:  # pragma: no cover
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
        except ValueError as err:  # pragma: no cover
            raise ValueError(
                f"Cannot add new attribute '{ncvar_converter.name}'. {str(err)}"
            ) from err
        array_creator.add_attr(ncvar_converter)
        self._attr_to_array[ncvar_converter.name] = array_name
        self._attr_dataspace_names[
            dataspace_name(ncvar_converter.name)
        ] = ncvar_converter.name

    def convert_to_group(
        self,
        output_uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        input_netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        input_group_path: Optional[str] = None,
    ):
        """Creates a TileDB group and its arrays from the defined CF dataspace and
        copies data into them using the converter engine.

        Parameters:
            output_uri: Uniform resource identifier for the TileDB group to be created.
            key: If not ``None``, encryption key to encrypt and decrypt output arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            input_netcdf_group: If not ``None``, the NetCDF group to copy data from.
                This will be prioritized over ``input_file`` if both are provided.
            input_file: If not ``None``, the NetCDF file to copy data from. This will
                not be used if ``netcdf_group`` is not ``None``.
            input_group_path: If not ``None``, the path to the NetCDF group to copy data
                from.
        """
        self.create_group(output_uri, key, ctx)
        self.copy_to_group(
            output_uri,
            key,
            ctx,
            input_netcdf_group,
            input_file,
            input_group_path,
        )

    def convert_to_virtual_group(
        self,
        output_uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        input_netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        input_group_path: Optional[str] = None,
    ):
        """Creates a TileDB group and its arrays from the defined CF dataspace and
        copies data into them using the converter engine.

        Parameters:
            output_uri: Uniform resource identifier for the TileDB group to be created.
            key: If not ``None``, encryption key to encrypt and decrypt output arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            input_netcdf_group: If not ``None``, the NetCDF group to copy data from.
                This will be prioritized over ``input_file`` if both are provided.
            input_file: If not ``None``, the NetCDF file to copy data from. This will
                not be used if ``netcdf_group`` is not ``None``.
            input_group_path: If not ``None``, the path to the NetCDF group to copy data
                from.
        """
        self.create_virtual_group(output_uri, key, ctx)
        self.copy_to_virtual_group(
            output_uri,
            key,
            ctx,
            input_netcdf_group,
            input_file,
            input_group_path,
        )

    def copy_to_group(
        self,
        output_uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        input_netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        input_group_path: Optional[str] = None,
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
            input_netcdf_group: If not ``None``, the NetCDF group to copy data from.
                This will be prioritized over ``input_file`` if both are provided.
            input_file: If not ``None``, the NetCDF file to copy data from. This will
                not be used if ``netcdf_group`` is not ``None``.
            input_group_path: If not ``None``, the path to the NetCDF group to copy data
                from.
            use_virtual_groups: If ``True``, create a virtual group using ``output_uri``
                as the name for the group metadata array. All other arrays will be named
                using the convention ``{uri}_{array_name}`` where ``array_name`` is the
                name of the array.
        """
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
            input_netcdf_group,
            input_file,
            input_group_path,
        ) as netcdf_group:
            # Copy group metadata
            with Group(output_uri, mode="w", key=key, ctx=ctx) as group:
                for group_key in netcdf_group.ncattrs():
                    copy_metadata_item(group.meta, netcdf_group, group_key)
            # Copy variables and variable metadata to arrays
            for array_name, array_creator in self._array_creators.items():
                if isinstance(array_creator, NetCDFArrayConverter):
                    with Group(
                        output_uri, mode="w", array=array_name, key=key, ctx=ctx
                    ) as tiledb_group:
                        array_creator.copy(netcdf_group, tiledb_group.array)

    def copy_to_virtual_group(
        self,
        output_uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        input_netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        input_group_path: Optional[str] = None,
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
            input_netcdf_group: If not ``None``, the NetCDF group to copy data from.
                This will be prioritized over ``input_file`` if both are provided.
            input_file: If not ``None``, the NetCDF file to copy data from. This will
                not be used if ``netcdf_group`` is not ``None``.
            input_group_path: If not ``None``, the path to the NetCDF group to copy data
                from.
        """
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
            input_netcdf_group,
            input_file,
            input_group_path,
        ) as netcdf_group:
            # Copy group metadata
            with tiledb.Array(output_uri, mode="w", key=key, ctx=ctx) as array:
                for group_key in netcdf_group.ncattrs():
                    copy_metadata_item(array.meta, netcdf_group, group_key)
            # Copy variables and variable metadata to arrays
            for array_name, array_creator in self._array_creators.items():
                array_uri = output_uri + "_" + array_name
                if isinstance(array_creator, NetCDFArrayConverter):
                    with tiledb.open(array_uri, mode="w", key=key, ctx=ctx) as array:
                        array_creator.copy(netcdf_group, array)


def copy_metadata_item(meta, netcdf_item, key):
    """Copies a NetCDF attribute from a NetCDF group or variable to TileDB metadata.

    Parameters:
        meta: TileDB metadata object to copy to.
        netcdf_item: NetCDF variable or group to copy from.
        key: Name of the NetCDF attribute that is being copied.
    """
    import time

    value = netcdf_item.getncattr(key)
    if key == "history":
        value = value + " - TileDB array created on " + time.ctime(time.time())
    elif isinstance(value, np.ndarray):
        value = tuple(value.tolist())
    elif isinstance(value, np.generic):
        value = (value.tolist(),)
    try:
        meta[key] = value
    except ValueError as err:  # pragma: no cover
        warnings.warn(f"Failed to set group metadata {value} with error: " f"{err}")


def get_ncattr(netcdf_item, key: str) -> Any:
    if key in netcdf_item.ncattrs():
        return netcdf_item.getncattr(key)
    return None


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
