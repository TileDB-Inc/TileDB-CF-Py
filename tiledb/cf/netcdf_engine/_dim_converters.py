# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Classes for converting NetCDF4 objects to TileDB attributes."""

import itertools
from abc import abstractmethod
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import netCDF4
import numpy as np

import tiledb
from tiledb.cf.core import DimMetadata, DType
from tiledb.cf.creator import DataspaceRegistry, DimCreator, SharedDim

from ._utils import get_unpacked_dtype, get_variable_values, safe_set_metadata


class NetCDF4ToDimConverter(DimCreator):
    """Converter from NetCDF to a TileDB dimension in a :class:`NetCDF4ArrayConverter`
    using a :class:`SharedDim` for the base dimension.

    Attributes:
        tile: The tile size for the dimension.
        filters: Specifies compression filters for the dimension.
    """

    def __init__(
        self,
        base: SharedDim,
        tile: Optional[Union[int, float]] = None,
        filters: Optional[tiledb.FilterList] = None,
        max_fragment_length: Optional[int] = None,
    ):
        self._base = base
        self.tile = tile
        self.filters = filters
        self.max_fragment_length = max_fragment_length

    def __repr__(self):
        filters_str = ""
        if self.filters:
            filters_str = ", filters=FilterList(["
            for dim_filter in self.filters:
                filters_str += repr(dim_filter) + ", "
            filters_str += "])"
        return (
            f"DimCreator({repr(self._base)}, tile={self.tile}, "
            f"max_fragment_length={self.max_fragment_length}{filters_str})"
        )

    def get_fragment_indices(self, netcdf_group: netCDF4.Dataset) -> Iterable[slice]:
        """Returns a sequence of slices for copying chunks of the dimension data for
        each TileDB fragment.

        Parameters:
            netcdf_group: NetCDF group to copy the data from.

        Returns:
            A sequence of slices for copying chunks of the dimension data for each
            TileDB fragment.
        """
        if not isinstance(self._base, NetCDF4ToDimBase):
            return (slice(None),)
        size = self.base.get_query_size(netcdf_group)
        if self.max_fragment_length is None:
            return (slice(0, size),)
        indices = np.arange(
            0, size + self.max_fragment_length, self.max_fragment_length
        )
        indices[-1] = size
        return itertools.starmap(slice, zip(indices, indices[1:]))

    def get_query_coordinates(
        self,
        netcdf_group: netCDF4.Group,
        sparse: bool,
        indexer: slice,
        assigned_dim_values: Optional[Dict[str, Any]],
    ):
        if assigned_dim_values is not None and self.name in assigned_dim_values:
            if self.is_from_netcdf:
                raise NotImplementedError(
                    "Support for over-writing dimension coordinate values copied from "
                    "NetCDF is not yet implemented."
                )
            return assigned_dim_values[self.name]
        if isinstance(self._base, NetCDF4ToDimBase):
            return self._base.get_values(netcdf_group, sparse=sparse, indexer=indexer)
        raise KeyError(f"Missing value for dimension '{self.name}'.")

    def html_summary(self) -> str:
        """Returns a string HTML summary of the :class:`NetCDF4ToDimConverter`."""
        filters_str = ""
        if self.filters:
            filters_str = ", filters=FilterList(["
            for dim_filter in self.filters:
                filters_str += repr(dim_filter) + ", "
            filters_str += "])"
        return (
            f"{self._base.html_input_summary()} &rarr; tiledb.Dim("
            f"{self._base.html_output_summary()}, tile={self.tile}, "
            f"max_fragment_length={self.max_fragment_length}{filters_str})"
        )

    @property
    def is_from_netcdf(self) -> bool:
        """Returns ``True`` if the dimension is converted from a NetCDF variable or
        dimension."""
        return hasattr(self._base, "from_netcdf")

    @property
    def max_fragment_length(self) -> Optional[int]:
        """The maximum number of elements to copy at a time. If ``None``, there is no
        maximum.
        """
        return self._max_fragment_length

    @max_fragment_length.setter
    def max_fragment_length(self, value: Optional[int]):
        if value is not None and value < 1:
            raise ValueError("The maximum fragment length must be a positive value.")
        self._max_fragment_length = value


class NetCDF4ToDimBase(SharedDim):
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
        self, netcdf_group: netCDF4.Dataset, sparse: bool, indexer: slice
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


class NetCDF4CoordToDimConverter(NetCDF4ToDimBase):
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
        unpack: bool,
    ):
        super().__init__(dataspace_registry, name, domain, dtype)
        self.input_dim_name = input_dim_name
        self.input_var_name = input_var_name
        self.input_var_dtype = input_var_dtype
        self.unpack = unpack

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
            if self.unpack and key in {"scale_factor", "add_offset"}:
                continue
            safe_set_metadata(dim_meta, key, variable.getncattr(key))

    @classmethod
    def from_netcdf(
        cls,
        dataspace_registry: DataspaceRegistry,
        ncvar: netCDF4.Variable,
        name: Optional[str] = None,
        domain: Optional[Tuple[DType, DType]] = None,
        dtype: Optional[np.dtype] = None,
        unpack: bool = False,
    ):
        if len(ncvar.dimensions) != 1:
            raise ValueError(
                f"Cannot create dimension from variable '{ncvar.name}' with shape "
                f"{ncvar.shape}. Coordinate variables must have only one dimension."
            )
        if dtype is None:
            dtype = get_unpacked_dtype(ncvar) if unpack else ncvar.dtype
        dtype = np.dtype(dtype)
        return cls(
            dataspace_registry=dataspace_registry,
            name=name if name is not None else ncvar.name,
            domain=domain,
            dtype=dtype,
            input_dim_name=ncvar.dimensions[0],
            input_var_name=ncvar.name,
            input_var_dtype=dtype,
            unpack=unpack,
        )

    def get_query_size(self, netcdf_group: netCDF4.Dataset):
        """Returns the number of coordinates to copy from NetCDF to TileDB.

        Parameters:
            netcdf_group: NetCDF group to copy the data from.
        """
        variable = self._get_ncvar(netcdf_group)
        return variable.get_dims()[0].size

    def get_values(self, netcdf_group: netCDF4.Dataset, sparse: bool, indexer: slice):
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
        if indexer.step not in {1, None}:
            raise ValueError("Dimension indexer must have step size of 1.")
        if not sparse:
            raise NotImplementedError(
                "Support for copying NetCDF coordinates to dense arrays has not "
                "been implemented."
            )
        variable = self._get_ncvar(netcdf_group)
        if variable.get_dims()[0].size < 1:
            raise ValueError(
                f"Cannot copy dimension data from NetCDF variable "
                f"'{self.input_var_name}' to TileDB dimension '{self.name}'. "
                f"There is no data to copy."
            )
        return get_variable_values(variable, indexer, fill=None, unpack=self.unpack)

    def html_input_summary(self):
        """Returns a HTML string summarizing the input for the dimension."""
        return (
            f"NetCDFVariable(name={self.input_var_name}, dtype={self.input_var_dtype})"
        )

    @property
    def is_index_dim(self) -> bool:
        return False


class NetCDF4DimToDimConverter(NetCDF4ToDimBase):
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
        self, netcdf_group: netCDF4.Dataset, sparse: bool, indexer: slice
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
        if indexer.step not in {1, None}:
            raise ValueError("Dimension indexer must have step size of 1.")
        start = 0 if indexer.start is None else indexer.start
        stop = indexer.stop
        if stop is None:
            dim = self._get_ncdim(netcdf_group)
            stop = dim.size
        if stop <= start:
            raise IndexError(
                f"Cannot copy dimension data from NetCDF dimension "
                f"'{self.input_dim_name}' of length {stop} less than starting ."
                f"fragment index {start}."
            )
        if self.domain is not None and (
            self.domain[1] is not None and stop - 1 > self.domain[1]
        ):
            raise IndexError(
                f"Cannot copy dimension data from NetCDF dimension "
                f"'{self.input_dim_name}' to TileDB dimension '{self.name}'. "
                f"The NetCDF chunk ending at {stop} is out of bounds "
                f"of the TileDB dimensions's domain {self.domain}."
            )
        return np.arange(start, stop) if sparse else slice(start, stop)

    def html_input_summary(self):
        """Returns a HTML string summarizing the input for the dimension."""
        size_str = "unlimited" if self.is_unlimited else str(self.input_dim_size)
        return f"NetCDFDimension(name={self.input_dim_name}, size={size_str})"


class NetCDF4ScalarToDimConverter(NetCDF4ToDimBase):
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
        self, netcdf_group: netCDF4.Dataset, sparse: bool, indexer: slice
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
        if indexer.step not in {1, None}:
            raise ValueError("Dimension indexer must have step size of 1.")
        if indexer.start not in {None, 0}:
            raise IndexError(
                f"Cannot copy scalar data to TileDB dimension '{self.name}'. The NetCDF"
                f" chunk starting at {indexer.start} is out of bounds."
            )
        if indexer.stop not in {None, 1}:
            raise IndexError(
                f"Cannot copy scalar data to TileDB dimension '{self.name}'. The NetCDF"
                f" chunk ending at {indexer.stop} is out of bounds."
            )
        return np.array([0]) if sparse else slice(0, 1)

    def html_input_summary(self):
        """Returns a string HTML summary."""
        return "NetCDF empty dimension"
