# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Classes for converting NetCDF4 objects to TileDB attributes."""

import warnings
from abc import abstractmethod
from typing import Optional, Sequence, Union

import netCDF4
import numpy as np

import tiledb
from tiledb.cf.core import AttrMetadata
from tiledb.cf.creator import ArrayRegistry, AttrCreator

from ._utils import COORDINATE_SUFFIX, safe_set_metadata


class NetCDF4ToAttrConverter(AttrCreator):
    """Abstract base class for classes that copy data from objects in a NetCDF group to
    a TileDB attribute.
    """

    def copy_metadata(self, netcdf_group: netCDF4.Dataset, tiledb_array: tiledb.Array):
        """Copy the metadata data from NetCDF to TileDB.

        Parameters:
            netcdf_group: NetCDF group to get the metadata items from.
            tiledb_array: TileDB array to copy the metadata items to.
        """

    @abstractmethod
    def get_values(
        self,
        netcdf_group: netCDF4.Dataset,
        sparse: bool,
        shape: Optional[Union[int, Sequence[int]]] = None,
    ) -> np.ndarray:
        """Returns TileDB attribute values from a NetCDF group.

        Parameters:
            netcdf_group: NetCDF group to get the dimension values from.
            sparse: ``True`` if copying into a sparse array and ``False`` if copying
                into a dense array.
            shape: If not ``None``, the shape to return the numpy array as.

        Returns:
            The values needed to set an attribute in a TileDB array. If the array
        is sparse the values will be returned as an 1D array; otherwise, they will
        be returned as an ND array.
        """


class NetCDF4VarToAttrConverter(NetCDF4ToAttrConverter):
    """Converter for a NetCDF variable to a TileDB attribute.

    Attributes:
        name: Name of the new attribute.
        dtype: Numpy dtype of the attribute.
        fill: Fill value for unset cells.
        var: Specifies if the attribute is variable length (automatic for
            byte/strings).
        nullable: Specifies if the attribute is nullable using validity tiles.
        filters: Specifies compression filters for the attribute.
        input_var_name: Name of the input NetCDF variable that will be converted.
        input_var_dtype: Numpy dtype of the input NetCDF variable.
    """

    def __init__(
        self,
        array_registry: ArrayRegistry,
        name: str,
        dtype: np.dtype,
        fill: Optional[Union[int, float, str]],
        var: bool,
        nullable: bool,
        filters: Optional[tiledb.FilterList],
        input_var_name: str,
        input_var_dtype: np.dtype,
    ):
        super().__init__(
            array_registry=array_registry,
            name=name,
            dtype=dtype,
            fill=fill,
            var=var,
            nullable=nullable,
            filters=filters,
        )
        self.input_var_name = input_var_name
        self.input_var_dtype = input_var_dtype

    def __repr__(self):
        return (
            f"NetCDFVariable(name={self.input_var_name}, dtype={self.input_var_dtype})"
            f" -> {super().__repr__()}"
        )

    def html_summary(self) -> str:
        """Returns a string HTML summary of the :class:`AttrCreator`."""
        return (
            f"NetCDFVariable(name={self.input_var_name}, dtype={self.input_var_dtype})"
            f"{super().html_summary()}"
        )

    def copy_metadata(self, netcdf_group: netCDF4.Dataset, tiledb_array: tiledb.Array):
        """Copy the metadata data from NetCDF to TileDB.

        Parameters:
            netcdf_group: NetCDF group to get the metadata items from.
            tiledb_array: TileDB array to copy the metadata items to.
        """
        try:
            variable = netcdf_group.variables[self.input_var_name]
        except KeyError as err:
            raise KeyError(
                f"The variable '{self.input_var_name}' was not found in the provided "
                f"NetCDF group."
            ) from err
        attr_meta = AttrMetadata(tiledb_array.meta, self.name)
        for key in variable.ncattrs():
            safe_set_metadata(attr_meta, key, variable.getncattr(key))

    @classmethod
    def from_netcdf(
        cls,
        array_registry: ArrayRegistry,
        ncvar: netCDF4.Variable,
        name: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
        fill: Optional[Union[int, float, str]] = None,
        var: bool = False,
        nullable: bool = False,
        filters: Optional[tiledb.FilterList] = None,
    ):
        if fill is None and "_FillValue" in ncvar.ncattrs():
            fill = ncvar.getncattr("_FillValue")
        if name is None:
            name = (
                ncvar.name
                if ncvar.name not in ncvar.dimensions
                else ncvar.name + COORDINATE_SUFFIX
            )
        dtype = np.dtype(dtype) if dtype is not None else np.dtype(ncvar.dtype)
        return cls(
            array_registry=array_registry,
            name=name,
            dtype=dtype,
            fill=fill,
            var=var,
            nullable=nullable,
            filters=filters,
            input_var_name=ncvar.name,
            input_var_dtype=ncvar.dtype,
        )

    def get_values(
        self,
        netcdf_group: netCDF4.Dataset,
        sparse: bool,
        shape: Optional[Union[int, Sequence[int]]] = None,
    ) -> np.ndarray:
        """Returns TileDB attribute values from a NetCDF group.

        Parameters:
            netcdf_group: NetCDF group to get the dimension values from.
            sparse: ``True`` if copying into a sparse array and ``False`` if copying
                into a dense array.
            shape: If not ``None``, the shape to return the numpy array as.

        Returns:
            The values needed to set an attribute in a TileDB array. If the array
        is sparse the values will be returned as an 1D array; otherwise, they will
        be returned as an ND array.
        """
        try:
            variable = netcdf_group.variables[self.input_var_name]
        except KeyError as err:
            raise KeyError(
                f"The variable '{self.input_var_name}' was not found in the provided "
                f"NetCDF group."
            ) from err
        if shape is not None:
            return np.reshape(variable[...], shape)
        return variable[...]

    @property
    def input_dtype(self) -> np.dtype:
        """(DEPRECATED) Size of the input NetCDF dimension."""
        with warnings.catch_warnings():
            warnings.warn(
                "Deprecated. Use `input_var_dtype` instead.",
                DeprecationWarning,
            )
        return self.input_var_dtype

    @property
    def input_name(self) -> str:
        """(DEPRECATED) Name of the input NetCDF variable."""
        with warnings.catch_warnings():
            warnings.warn(
                "Deprecated. Use `input_var_name` instead.",
                DeprecationWarning,
            )
        return self.input_var_name
