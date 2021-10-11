# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Classes for converting NetCDF4 files to TileDB."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import netCDF4
import numpy as np

import tiledb
from tiledb.cf.creator import (
    ArrayCreator,
    ArrayRegistry,
    DataspaceRegistry,
    DimCreator,
    DomainCreator,
)

from ._attr_converters import NetCDF4ToAttrConverter, NetCDF4VarToAttrConverter
from ._dim_converters import NetCDF4ToDimBase, NetCDF4ToDimConverter


class NetCDF4ArrayConverter(ArrayCreator):
    """Converter for a TileDB array from a collection of NetCDF variables.

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

    def _register(
        self, dataspace_registry: DataspaceRegistry, name: str, dim_names: Sequence[str]
    ):
        shared_dims = (
            dataspace_registry.get_shared_dim(dim_name) for dim_name in dim_names
        )
        dim_creators = tuple(
            (
                NetCDF4ToDimConverter(shared_dim)
                if isinstance(shared_dim, NetCDF4ToDimBase)
                else DimCreator(shared_dim)
            )
            for shared_dim in shared_dims
        )
        array_registry = ArrayRegistry(dataspace_registry, name, dim_creators)
        return (
            array_registry,
            NetCDF4DomainConverter(array_registry, dataspace_registry),
        )

    def add_var_to_attr_converter(
        self,
        ncvar: netCDF4.Variable,
        name: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
        fill: Optional[Union[int, float, str]] = None,
        var: bool = False,
        nullable: bool = False,
        filters: Optional[tiledb.FilterList] = None,
    ):
        """Adds a new variable to attribute converter to the array creator.

        The attribute's 'dataspace name' (name after dropping the suffix ``.data`` or
        ``.index``) be unique.

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
        """
        if ncvar.dimensions != self.domain_creator.netcdf_dims:
            raise ValueError(
                f"Cannot add NetCDF variable converter with NetCDF dimensions "
                f"that do not match the array NetCDF dimension converters. Variable "
                f"dimensions={ncvar.dimensions}, array NetCDF dimensions="
                f"{self.domain_creator.netcdf_dims}."
            )
        NetCDF4VarToAttrConverter.from_netcdf(
            array_registry=self._registry,
            ncvar=ncvar,
            name=name,
            dtype=dtype,
            fill=fill,
            var=var,
            nullable=nullable,
            filters=filters,
        )

    def copy(
        self,
        netcdf_group: netCDF4.Group,
        tiledb_uri: str,
        tiledb_key: Optional[str] = None,
        tiledb_ctx: Optional[str] = None,
        assigned_dim_values: Optional[Dict[str, Any]] = None,
        assigned_attr_values: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Copies data from a NetCDF group to a TileDB CF array.

        Parameters:
            netcdf_group: The NetCDF group to copy data from.
            tiledb_uri: The TileDB array uri to copy data into.
            tiledb_key: If not ``None``, the encryption key for the TileDB array.
            tiledb_ctx: If not ``None``, the TileDB context wrapper for a TileDB
                storage manager to use when opening the TileDB array.
            assigned_dim_values: Mapping from dimension name to value for dimensions
                that are not copied from the NetCDF group.
            assigned_attr_values: Mapping from attribute name to numpy array of values
                for attributes that are not copied from the NetCDF group.
        """
        with tiledb.open(
            tiledb_uri, mode="w", key=tiledb_key, ctx=tiledb_ctx
        ) as tiledb_array:
            # Copy metadata for TileDB dimensions and attributes.
            for attr_creator in self:
                if isinstance(attr_creator, NetCDF4ToAttrConverter):
                    attr_creator.copy_metadata(netcdf_group, tiledb_array)
            for dim_creator in self._domain_creator:
                if isinstance(dim_creator.base, NetCDF4ToDimBase):
                    dim_creator.base.copy_metadata(netcdf_group, tiledb_array)
            # Copy array data to TileDB.
            if self.sparse:
                shape: Optional[Union[int, Sequence[int]]] = -1
            else:
                shape = (
                    None
                    if all(
                        isinstance(dim_creator.base, NetCDF4ToDimBase)
                        for dim_creator in self._domain_creator
                    )
                    else self.domain_creator.get_dense_query_shape(netcdf_group)
                )
            data = {}
            for attr_creator in self:
                if isinstance(attr_creator, NetCDF4ToAttrConverter):
                    data[attr_creator.name] = attr_creator.get_values(
                        netcdf_group, sparse=self.sparse, shape=shape
                    )
                else:
                    if (
                        assigned_attr_values is None
                        or attr_creator.name not in assigned_attr_values
                    ):
                        raise KeyError(
                            f"Missing value for attribute '{attr_creator.name}'."
                        )
                    data[attr_creator.name] = assigned_attr_values[attr_creator.name]
            coord_values = self._domain_creator.get_query_coordinates(
                netcdf_group, self.sparse, assigned_dim_values
            )
            tiledb_array[coord_values] = data


class NetCDF4DomainConverter(DomainCreator):
    """Converter for NetCDF dimensions to a TileDB domain."""

    def get_dense_query_shape(self, netcdf_group: netCDF4.Dataset) -> Tuple[int, ...]:
        """Returns the shape of the coordinates for copying from the requested NetCDF
        group to a dense array.

        Parameters:
            netcdf_group: Group to query the data from.
        """
        return tuple(
            dim_creator.base.get_query_size(netcdf_group)
            if isinstance(dim_creator.base, NetCDF4ToDimBase)
            else 1
            for dim_creator in self
        )

    def get_query_coordinates(
        self,
        netcdf_group: netCDF4.Group,
        sparse: bool,
        assigned_dim_values: Optional[Dict[str, Any]] = None,
    ):
        """Returns the coordinates used to copy data from a NetCDF group.

        Parameters:
            netcdf_group: Group to query the data from.
            sparse: If ``True``, return coordinates for a sparse write. If ``False``,
                return coordinates for a dense write.
            assigned_dim_values: Values for any non-NetCDF dimensions.
        """
        query_coords = []
        for dim_creator in self:
            if isinstance(dim_creator.base, NetCDF4ToDimBase):
                query_coords.append(
                    dim_creator.base.get_values(netcdf_group, sparse=sparse)
                )
            else:
                if (
                    assigned_dim_values is None
                    or dim_creator.name not in assigned_dim_values
                ):
                    raise KeyError(f"Missing value for dimension '{dim_creator.name}'.")
                query_coords.append(assigned_dim_values[dim_creator.name])
        if sparse:
            return tuple(
                dim_data.reshape(-1)
                for dim_data in np.meshgrid(*query_coords, indexing="ij")
            )
        return tuple(query_coords)

    def inject_dim_creator(
        self,
        dim_name: str,
        position: int,
        tiles: Optional[Union[int, float]] = None,
        filters: Optional[Union[tiledb.FilterList]] = None,
    ):
        """Add an additional dimension into the domain of the array.

        Parameters:
            dim_name: Name of the shared dimension to add to the array's domain.
            position: Position of the shared dimension. Negative values count backwards
                from the end of the new number of dimensions.
            tiles: The size size for the dimension.
            filters: Compression filters for the dimension.
        """
        shared_dim = self._dataspace_registry.get_shared_dim(dim_name)
        if isinstance(shared_dim, NetCDF4ToDimBase):
            if any(
                isinstance(attr_creator, NetCDF4VarToAttrConverter)
                for attr_creator in self._array_registry.attr_creators()
            ):
                raise ValueError(
                    "Cannot add a new NetCDF dimension converter to an array that "
                    "already contains NetCDF variable to attribute converters."
                )
        if isinstance(shared_dim, NetCDF4ToDimBase):
            self._array_registry.inject_dim_creator(
                NetCDF4ToDimConverter(shared_dim, tiles, filters), position
            )
        else:
            self._array_registry.inject_dim_creator(
                DimCreator(shared_dim, tiles, filters), position
            )

    @property
    def netcdf_dims(self):
        """Ordered tuple of NetCDF dimension names for dimension converters."""
        return tuple(
            dim_creator.base.input_dim_name
            for dim_creator in self
            if hasattr(dim_creator.base, "input_dim_name")
        )

    def remove_dim_creator(self, dim_id: Union[str, int]):
        """Removes a dimension creator from the array creator.

        Parameters:
            dim_id: dimension index (int) or name (str)
        """
        index = (
            dim_id
            if isinstance(dim_id, int)
            else self._array_registry.get_dim_position_by_name(dim_id)
        )
        dim_creator = self._array_registry.get_dim_creator(index)
        if isinstance(dim_creator.base, NetCDF4ToDimBase):
            if any(
                isinstance(attr_creator, NetCDF4VarToAttrConverter)
                for attr_creator in self._array_registry.attr_creators()
            ):
                raise ValueError(
                    "Cannot remove NetCDF dimension converter from an array that "
                    "contains NetCDF variable to attribute converters."
                )
        self._array_registry.remove_dim_creator(index)
