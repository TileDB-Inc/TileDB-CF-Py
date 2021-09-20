# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Classes for converting NetCDF4 objects to TileDB attributes."""

import warnings
from abc import abstractmethod
from typing import Optional, Tuple, Union

import netCDF4
import numpy as np

import tiledb
from tiledb.cf.core import DimMetadata, DType
from tiledb.cf.creator import DataspaceRegistry, SharedDim

from ._utils import get_ncattr, safe_set_metadata


class NetCDF4ToDimConverter(SharedDim):
    """Abstract base class for classes that copy data from objects in a NetCDF group to
    a TileDB dimension.
    """

    def copy_metadata(self, netcdf_group: netCDF4.Dataset, tiledb_array: tiledb.Array):
        """Copy the metadata data from NetCDF to TileDB.

        Parameters:
            netcdf_group: NetCDF group to get the metadata items from.
            tiledb_array: TileDB array to copy the metadata items to.
        """

    @abstractmethod
    def get_query_size(self, netcdf_group: netCDF4.Dataset):
        """Returns the number of coordinates to copy from NetCDF to TileDB.

        Parameters:
            netcdf_group: NetCDF group to copy the data from.
        """

    @abstractmethod
    def get_values(
        self, netcdf_group: netCDF4.Dataset, sparse: bool
    ) -> Union[np.ndarray, slice]:
        """Returns values from a NetCDF group that will be copied to TileDB.

        Parameters:
            netcdf_group: NetCDF group to get the values from.
            sparse: ``True`` if copying into a sparse array and ``False`` if copying
                into a dense array.

        Returns:
            The coordinates needed for querying the create TileDB dimension in the form
                of a numpy array if sparse is ``True`` and a slice otherwise.
        """


class NetCDF4CoordToDimConverter(NetCDF4ToDimConverter):
    """Converter for a NetCDF variable/dimension pair to a TileDB dimension.

    Attributes:
        name: Name of the TileDB dimension.
        domain: The (inclusive) interval on which the dimension is valid.
        dtype: The numpy dtype of the values and domain of the dimension.
        input_dim_name: The name of input NetCDF dimension.
        input_var_name: The name of input NetCDF variable.
        input_var_dtype: The numpy dtype of the input NetCDF variable.
    """

    def __init__(
        self,
        dataspace_registry: DataspaceRegistry,
        name: str,
        domain: Optional[Tuple[Optional[DType], Optional[DType]]],
        dtype: np.dtype,
        input_dim_name: str,
        input_var_name: str,
        input_var_dtype: np.dtype,
    ):
        super().__init__(dataspace_registry, name, domain, dtype)
        self.input_dim_name = input_dim_name
        self.input_var_name = input_var_name
        self.input_var_dtype = input_var_dtype

    def __eq__(self, other):
        return (
            super.__eq__(self, other)
            and self.input_dim_name == other.input_dim_name
            and self.input_var_name == other.input_var_name
            and self.input_var_dtype == other.input_var_dtype
        )

    def __repr__(self):
        return (
            f"NetCDFVariable(name={self.input_var_name}, dtype={self.input_var_dtype}) "
            f" -> {super().__repr__()}"
        )

    def _get_ncvar(self, netcdf_group: netCDF4.Dataset) -> netCDF4.Variable:
        try:
            variable = netcdf_group.variables[self.input_var_name]
        except KeyError as err:
            raise KeyError(
                f"The variable '{self.input_var_name}' was not found in the provided "
                f"NetCDF group."
            ) from err
        if variable.ndim != 1:
            raise ValueError(
                f"The variable '{self.input_var_name}' with {variable.ndim} dimensions "
                f"is not a valid NetCDF coordinate. A coordinate must have exactly 1 "
                f"dimension."
            )
        return variable

    def copy_metadata(self, netcdf_group: netCDF4.Dataset, tiledb_array: tiledb.Array):
        """Copy the metadata data from NetCDF to TileDB.

        Parameters:
            netcdf_group: NetCDF group to get the metadata items from.
            tiledb_array: TileDB array to copy the metadata items to.
        """
        variable = self._get_ncvar(netcdf_group)
        dim_meta = DimMetadata(tiledb_array.meta, self.name)
        for key in variable.ncattrs():
            safe_set_metadata(dim_meta, key, variable.getncattr(key))

    @classmethod
    def from_netcdf(
        cls,
        dataspace_registry: DataspaceRegistry,
        var: netCDF4.Variable,
        name: Optional[str] = None,
        domain: Optional[Tuple[DType, DType]] = None,
    ):
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
        return cls(
            dataspace_registry=dataspace_registry,
            name=name if name is not None else var.name,
            domain=domain,
            dtype=dtype,
            input_dim_name=var.dimensions[0],
            input_var_name=var.name,
            input_var_dtype=dtype,
        )

    def get_query_size(self, netcdf_group: netCDF4.Dataset):
        """Returns the number of coordinates to copy from NetCDF to TileDB.

        Parameters:
            netcdf_group: NetCDF group to copy the data from.
        """
        variable = self._get_ncvar(netcdf_group)
        return variable.get_dims()[0].size

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
        variable = self._get_ncvar(netcdf_group)
        if not sparse:
            raise NotImplementedError(
                "Support for copying NetCDF coordinates to dense arrays has not "
                "been implemented."
            )
        if variable.get_dims()[0].size < 1:
            raise ValueError(
                f"Cannot copy dimension data from NetCDF variable "
                f"'{self.input_var_name}' to TileDB dimension '{self.name}'. "
                f"There is no data to copy."
            )
        return variable[:]

    def html_input_summary(self):
        """Returns a HTML string summarizing the input for the dimension."""
        return (
            f"NetCDFVariable(name={self.input_var_name}, dtype={self.input_var_dtype})"
        )

    @property
    def input_dtype(self) -> np.dtype:
        """(DEPRECATED) Name of the input NetCDF variable and dimension."""
        with warnings.catch_warnings():
            warnings.warn(
                "Deprecated. Use `input_var_dtype` instead.",
                DeprecationWarning,
            )
        return self.input_var_dtype

    @property
    def input_name(self) -> str:
        """(DEPRECATED) Name of the input NetCDF variable and dimension."""
        with warnings.catch_warnings():
            warnings.warn(
                "Deprecated. Use `input_var_name` to get the name of the input NetCDF "
                "variable and `input_dim_name` to get the input NetCDF dimension.",
                DeprecationWarning,
            )
        if self.input_var_name != self.input_dim_name:
            raise ValueError(
                "Input name is ambiguous. The input variable and input dimension have "
                "different names."
            )
        return self.input_var_name

    @property
    def is_index_dim(self) -> bool:
        return False


class NetCDF4DimToDimConverter(NetCDF4ToDimConverter):
    """Converter for a NetCDF dimension to a TileDB dimension.

    Attributes:
        name: Name of the TileDB dimension.
        domain: The (inclusive) interval on which the dimension is valid.
        dtype: The numpy dtype of the values and domain of the dimension.
        input_dim_name: Name of the input NetCDF variable.
        input_dim_size: Size of the input NetCDF variable.
        is_unlimited: If True, the input NetCDF variable is unlimited.
    """

    def __init__(
        self,
        dataspace_registry: DataspaceRegistry,
        name: str,
        domain: Optional[Tuple[Optional[DType], Optional[DType]]],
        dtype: np.dtype,
        input_dim_name: str,
        input_dim_size: int,
        is_unlimited: bool,
    ):
        super().__init__(dataspace_registry, name, domain, dtype)
        self.input_dim_name = input_dim_name
        self.input_dim_size = input_dim_size
        self.is_unlimited = is_unlimited

    def __eq__(self, other):
        return (
            super.__eq__(self, other)
            and self.input_dim_name == other.input_dim_name
            and self.input_dim_size == other.input_dim_size
            and self.is_unlimited == other.is_unlimited
        )

    def __repr__(self):
        size_str = "unlimited" if self.is_unlimited else str(self.input_dim_size)
        return (
            f"NetCDFDimension(name={self.input_dim_name}, size={size_str}) -> "
            f"{super().__repr__()}"
        )

    def _get_ncdim(self, netcdf_group: netCDF4.Dataset) -> netCDF4.Dimension:
        group = netcdf_group
        while group is not None:
            if self.input_dim_name in group.dimensions:
                return group.dimensions[self.input_dim_name]
            group = group.parent
        raise KeyError(
            f"Unable to copy NetCDF dimension '{self.input_dim_name}' to the TileDB "
            f"dimension '{self.name}'. No NetCDF dimension with that name exists in "
            f"the NetCDF group '{netcdf_group.path}' or its parent groups."
        )

    @classmethod
    def from_netcdf(
        cls,
        dataspace_registry: DataspaceRegistry,
        dim: netCDF4.Dimension,
        unlimited_dim_size: Optional[int],
        dtype: np.dtype,
        name: Optional[str] = None,
    ):
        size = (
            unlimited_dim_size
            if dim.isunlimited() and unlimited_dim_size is not None
            else dim.size
        )
        return cls(
            dataspace_registry=dataspace_registry,
            name=name if name is not None else dim.name,
            domain=(0, size - 1),
            dtype=dtype,
            input_dim_name=dim.name,
            input_dim_size=dim.size,
            is_unlimited=dim.isunlimited(),
        )

    def get_query_size(self, netcdf_group: netCDF4.Dataset):
        """Returns the number of coordinates to copy from NetCDF to TileDB.

        Parameters:
            netcdf_group: NetCDF group to copy the data from.
        """
        dim = self._get_ncdim(netcdf_group)
        return dim.size

    def get_values(
        self, netcdf_group: netCDF4.Dataset, sparse: bool
    ) -> Union[np.ndarray, slice]:
        """Returns the values of the NetCDF dimension that is being copied.

        Parameters:
            netcdf_group: NetCDF group to get the dimension values from.
            sparse: ``True`` if copying into a sparse array and ``False`` if copying
                into a dense array.

        Returns:
            The coordinates needed for querying the created TileDB dimension in the form
                of a numpy array if sparse is ``True`` and a slice otherwise.
        """
        dim = self._get_ncdim(netcdf_group)
        if dim.size == 0:
            raise ValueError(
                f"Cannot copy dimension data from NetCDF dimension "
                f"'{self.input_dim_name}' to TileDB dimension '{self.name}'. "
                f"The NetCDF dimension is of size 0; there is no data to copy."
            )
        if self.domain is not None and (
            self.domain[1] is not None and dim.size - 1 > self.domain[1]
        ):
            raise IndexError(
                f"Cannot copy dimension data from NetCDF dimension "
                f"'{self.input_dim_name}' to TileDB dimension '{self.name}'. "
                f"The NetCDF dimension size of {dim.size} does not fit in the "
                f"domain {self.domain} of the TileDB dimension."
            )
        if sparse:
            return np.arange(dim.size)
        return slice(dim.size)

    def html_input_summary(self):
        """Returns a HTML string summarizing the input for the dimension."""
        size_str = "unlimited" if self.is_unlimited else str(self.input_dim_size)
        return f"NetCDFDimension(name={self.input_dim_name}, size={size_str})"

    @property
    def input_name(self) -> str:
        """(DEPRECATED) Name of the input NetCDF dimension."""
        with warnings.catch_warnings():
            warnings.warn(
                "Deprecated. Use `input_dim_name` instead.",
                DeprecationWarning,
            )
        return self.input_dim_name

    @property
    def input_size(self) -> int:
        """(DEPRECATED) Size of the input NetCDF dimension."""
        with warnings.catch_warnings():
            warnings.warn(
                "Deprecated. Use `input_dim_size` instead.",
                DeprecationWarning,
            )
        return self.input_dim_size


class NetCDF4ScalarToDimConverter(NetCDF4ToDimConverter):
    """Converter for NetCDF scalar (empty) dimensions to a TileDB Dimension.

    Attributes:
        name: Name of the TileDB dimension.
        domain: The (inclusive) interval on which the dimension is valid.
        dtype: The numpy dtype of the values and domain of the dimension.
    """

    def __repr__(self):
        return f" Scalar dimensions -> {super().__repr__()}"

    @classmethod
    def create(
        cls, dataspace_registry: DataspaceRegistry, dim_name: str, dtype: np.dtype
    ):
        return cls(dataspace_registry, dim_name, (0, 0), dtype)

    def get_query_size(self, netcdf_group: netCDF4.Dataset):
        """Returns the number of coordinates to copy from NetCDF to TileDB.

        Parameters:
            netcdf_group: NetCDF group to copy the data from.
        """
        return 1

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

    def html_input_summary(self):
        """Returns a string HTML summary."""
        return "NetCDF empty dimension"
