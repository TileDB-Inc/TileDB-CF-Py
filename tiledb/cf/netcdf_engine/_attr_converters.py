"""Classes for converting NetCDF4 objects to TileDB attributes."""

from abc import abstractmethod
from typing import Optional, Sequence, Union

import netCDF4
import numpy as np
from typing_extensions import Self

import tiledb
from tiledb.cf.core import AttrMetadata

from ..core._attr_creator import AttrCreator
from ..core.registry import Registry
from ._utils import (
    COORDINATE_SUFFIX,
    get_netcdf_metadata,
    get_unpacked_dtype,
    get_variable_values,
    safe_set_metadata,
)


class NetCDF4ToAttrConverter(AttrCreator):
    """Abstract base class for classes that copy data from objects in a NetCDF group to
    a TileDB attribute.
    """

    def copy_metadata(self, netcdf_group: netCDF4.Dataset, tiledb_array: tiledb.Array):
        """Copy the metadata data from NetCDF to TileDB.

        Parameters
        ----------
        netcdf_group
            NetCDF group to get the metadata items from.
        tiledb_array
            TileDB array to copy the metadata items to.
        """

    @abstractmethod
    def get_values(
        self,
        netcdf_group: netCDF4.Dataset,
        indexer: Sequence[slice],
    ) -> np.ndarray:
        """Returns TileDB attribute values from a NetCDF group.

        Parameters
        ----------
        netcdf_group
            NetCDF group to get the dimension values from.
        indexer
            Region to get data for.

        Returns
        -------
        np.ndarray
            The values needed to set an attribute in a TileDB array. If the array is
            sparse the values will be returned as an 1D array; otherwise, they will be
            returned as an ND array.
        """


class NetCDF4VarToAttrConverter(NetCDF4ToAttrConverter):
    """Converter for a NetCDF variable to a TileDB attribute.

    Parameters
    ----------
    name
        The name of the attribute.
    dtype
        The datatype of the attribute that will be created.
    fill
        Optional fill value for the attribute that will be created.
    var
        Specifies if the attribute that will be created will be variable length
        (automatic for byte/strings).
    nullable
        Specifies if the attribute that will be created will be nullable using
        validity tiles.
    filters
        Filter pipeline to apply to the attribute.
    input_var_name
        Name of the input netCDF variable to convert.
    input_var_dtype
        Datatype of the netCDF variable to convert.
    unpack
        If ``True``, unpack the data before converting.
    registry
        Registry for this attribute creator.
    """

    def __init__(
        self,
        name: str,
        dtype: np.dtype,
        fill: Optional[Union[int, float, str]],
        var: bool,
        nullable: bool,
        filters: Optional[tiledb.FilterList],
        input_var_name: str,
        input_var_dtype: np.dtype,
        unpack: bool,
        *,
        registry: Optional[Registry[Self]] = None,
    ):
        super().__init__(
            registry=registry,
            name=name,
            dtype=dtype,
            fill=fill,
            var=var,
            nullable=nullable,
            filters=filters,
        )
        self.input_var_name = input_var_name
        self.input_var_dtype = input_var_dtype
        self.unpack = unpack

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

        Parameters
        ----------
        netcdf_group
            NetCDF group to get the metadata items from.
        tiledb_array
            TileDB array to copy the metadata items to.
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
            if key == "_FillValue":
                continue
            if self.unpack and key in {"scale_factor", "add_offset"}:
                continue
            safe_set_metadata(attr_meta, key, variable.getncattr(key))

    @classmethod
    def from_netcdf(
        cls,
        ncvar: netCDF4.Variable,
        *,
        name: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
        fill: Optional[Union[int, float, str]] = None,
        var: bool = False,
        nullable: bool = False,
        filters: Optional[tiledb.FilterList] = None,
        unpack: bool = False,
        registry: Optional[Registry[Self]] = None,
    ):
        """Create from a NetCDF variable.

        Parameters
        ----------
        ncvar
            Input netCDF variable.
        name
            Name to use for the attribute. If ``None``, use the NetCDF variable name.
        dtype
            The datatype of the attribute. If ``None``, use the NetCDF variable dtype.
        fill
            Optional fill value for the attribute. If ``None, use the NetCDF variable
            fill.
        var
            Specifies if the attribute that will be created will be variable length
            (automatic for byte/strings).
        nullable
            Specifies if the attribute that will be created will be nullable using
            validity tiles.
        filters
            Filter pipeline to apply to the attribute.
        unpack
            If ``True``, unpack the NetCDF variable before converting.
        registry
            The registry for the attribute creator.
        """
        if fill is None:
            fill = get_netcdf_metadata(ncvar, "_FillValue")
        if name is None:
            name = (
                ncvar.name
                if ncvar.name not in ncvar.dimensions
                else ncvar.name + COORDINATE_SUFFIX
            )
        if dtype is None:
            dtype = get_unpacked_dtype(ncvar) if unpack else ncvar.dtype
        dtype = np.dtype(dtype)
        return cls(
            registry=registry,
            name=name,
            dtype=dtype,
            fill=fill,
            var=var,
            nullable=nullable,
            filters=filters,
            input_var_name=ncvar.name,
            input_var_dtype=ncvar.dtype,
            unpack=unpack,
        )

    def get_values(
        self,
        netcdf_group: netCDF4.Dataset,
        indexer: Sequence[slice],
    ) -> np.ndarray:
        """Returns TileDB attribute values from a NetCDF group.

        Parameters
        ----------
        netcdf_group
            NetCDF group to get the dimension values from.
        indexer
            Slice to query the NetCDF variable on.

        Returns
        -------
        np.ndarray
            The values needed to set an attribute in a TileDB array. If the array is
            sparse the values will be returned as an 1D array; otherwise, they will be
            returned as an ND array.
        """
        try:
            variable = netcdf_group.variables[self.input_var_name]
        except KeyError as err:  # pragma: no cover
            raise KeyError(
                f"The variable '{self.input_var_name}' was not found in the provided "
                f"NetCDF group."
            ) from err
        return get_variable_values(
            variable, indexer, fill=self.fill, unpack=self.unpack
        )
