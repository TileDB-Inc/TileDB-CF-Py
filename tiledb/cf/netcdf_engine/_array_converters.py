# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Classes for converting NetCDF4 files to TileDB."""

import itertools
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import netCDF4
import numpy as np

import tiledb
from tiledb.cf.creator import (
    ArrayCreator,
    ArrayRegistry,
    DataspaceRegistry,
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
        attrs_filters: Default filters to use when adding an attribute to the array.
        allows_duplicates: Specifies if multiple values can be stored at the same
             coordinate. Only allowed for sparse arrays.
    """

    def _copy_to_array(
        self,
        netcdf_group: netCDF4.Group,
        tiledb_array: tiledb.Array,
        indexer: Tuple[slice, ...],
        assigned_dim_values: Optional[Dict[str, Any]],
        assigned_attr_values: Optional[Dict[str, np.ndarray]],
    ):
        """Copies data from a NetCDF group to a TileDB CF array.

        Parameters:
            netcdf_group: The NetCDF group to copy data from.
            tiledb_array: The TileDB array to  copy data to.
            indexer: Slices defining what values to copy for each dimension.
            assigned_dim_values: Mapping from dimension name to value for dimensions
                that are not copied from the NetCDF group.
            assigned_attr_values: Mapping from attribute name to numpy array of values
                for attributes that are not copied from the NetCDF group.
        """
        assert len(indexer) == self.ndim, "indexer has incorrect number of values"
        netcdf_indexer = tuple(
            index_slice
            for index_slice, dim_creator in zip(indexer, self.domain_creator)
            if dim_creator.is_from_netcdf
        )
        data = {}
        for attr_creator in self:
            attr_name = attr_creator.name
            if isinstance(attr_creator, NetCDF4ToAttrConverter):
                data[attr_name] = attr_creator.get_values(
                    netcdf_group=netcdf_group, indexer=netcdf_indexer
                )
            else:
                if (
                    assigned_attr_values is None
                    or attr_name not in assigned_attr_values
                ):
                    raise KeyError(f"Missing value for attribute '{attr_name}'.")

                data[attr_name] = assigned_attr_values[attr_name][indexer]
        coord_values = self._domain_creator.get_query_coordinates(
            netcdf_group, self.sparse, indexer, assigned_dim_values
        )
        tiledb_array[coord_values] = data

    def _register(
        self, dataspace_registry: DataspaceRegistry, name: str, dim_names: Sequence[str]
    ):
        shared_dims = map(dataspace_registry.get_shared_dim, dim_names)
        dim_creators = tuple(map(NetCDF4ToDimConverter, shared_dims))
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
        unpack: bool = False,
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
            filters: Specifies compression filters for the attribute. If ``None``, use
                array's ``attrs_filters`` property.
            unpack: Unpack NetCDF data that has NetCDF attributes ``scale_factor`` or
                ``add_offset`` using the transformation ``scale_factor * value +
                unpack``.
        """
        if ncvar.dimensions != self.domain_creator.netcdf_dims:
            raise ValueError(
                f"Cannot add NetCDF variable converter with NetCDF dimensions "
                f"that do not match the array NetCDF dimension converters. Variable "
                f"dimensions={ncvar.dimensions}, array NetCDF dimensions="
                f"{self.domain_creator.netcdf_dims}."
            )
        if filters is None:
            filters = self.attrs_filters
        NetCDF4VarToAttrConverter.from_netcdf(
            array_registry=self._registry,
            ncvar=ncvar,
            name=name,
            dtype=dtype,
            fill=fill,
            var=var,
            nullable=nullable,
            filters=filters,
            unpack=unpack,
        )

    def copy(
        self,
        netcdf_group: netCDF4.Group,
        tiledb_uri: str,
        tiledb_key: Optional[str] = None,
        tiledb_ctx: Optional[str] = None,
        tiledb_timestamp: Optional[int] = None,
        assigned_dim_values: Optional[Dict[str, Any]] = None,
        assigned_attr_values: Optional[Dict[str, np.ndarray]] = None,
        copy_metadata: bool = True,
    ):
        """Copies data from a NetCDF group to a TileDB CF array.

        Parameters:
            netcdf_group: The NetCDF group to copy data from.
            tiledb_uri: The TileDB array uri to copy data into.
            tiledb_key: If not ``None``, the encryption key for the TileDB array.
            tiledb_ctx: If not ``None``, the TileDB context wrapper for a TileDB
                storage manager to use when opening the TileDB array.
            tiledb_timestamp: If not ``None``, the timestamp to write the TileDB data
                at.
            assigned_dim_values: Mapping from dimension name to value for dimensions
                that are not copied from the NetCDF group.
            assigned_attr_values: Mapping from attribute name to numpy array of values
                for attributes that are not copied from the NetCDF group.
        """
        dim_slices = tuple(
            dim_creator.get_fragment_indices(netcdf_group)
            for dim_creator in self.domain_creator
        )
        with tiledb.open(
            tiledb_uri,
            mode="w",
            key=tiledb_key,
            ctx=tiledb_ctx,
            timestamp=tiledb_timestamp,
        ) as tiledb_array:
            if copy_metadata:
                for attr_creator in self:
                    if isinstance(attr_creator, NetCDF4ToAttrConverter):
                        attr_creator.copy_metadata(netcdf_group, tiledb_array)
                for dim_creator in self._domain_creator:
                    if isinstance(dim_creator.base, NetCDF4ToDimBase):
                        dim_creator.base.copy_metadata(netcdf_group, tiledb_array)
            # Copy array data.
            for indexer in itertools.product(*dim_slices):
                self._copy_to_array(
                    netcdf_group,
                    tiledb_array,
                    indexer,
                    assigned_dim_values,
                    assigned_attr_values,
                )


class NetCDF4DomainConverter(DomainCreator):
    """Converter for NetCDF dimensions to a TileDB domain."""

    def get_query_coordinates(
        self,
        netcdf_group: netCDF4.Group,
        sparse: bool,
        indexer: Sequence[slice],
        assigned_dim_values: Optional[Dict[str, Any]] = None,
    ):
        """Returns the coordinates used to copy data from a NetCDF group.

        Parameters:
            netcdf_group: Group to query the data from.
            sparse: If ``True``, return coordinates for a sparse write. If ``False``,
                return coordinates for a dense write.
            assigned_dim_values: Values for any non-NetCDF dimensions.
        """
        if len(indexer) != self.ndim:
            raise ValueError(
                f"Indexer must be the same length as the domain of the array. Indexer "
                f"of length {len(indexer)} provide for an array with {self.ndim} "
                f"dimensions."
            )
        query_coords = tuple(
            dim_creator.get_query_coordinates(
                netcdf_group, sparse, index_slice, assigned_dim_values
            )
            for index_slice, dim_creator in zip(indexer, self)
        )
        if sparse:
            return tuple(
                dim_data.reshape(-1)
                for dim_data in np.meshgrid(*query_coords, indexing="ij")
            )
        return query_coords

    def inject_dim_creator(self, dim_name: str, position: int, **dim_kwargs):
        """Add an additional dimension into the domain of the array.

        Parameters:
            dim_name: Name of the shared dimension to add to the array's domain.
            position: Position of the shared dimension. Negative values count backwards
                from the end of the new number of dimensions.
            dim_kwargs: Keyword arguments to pass to :class:`NetCDF4ToDimConverter`.
        """
        dim_creator = NetCDF4ToDimConverter(
            self._dataspace_registry.get_shared_dim(dim_name), **dim_kwargs
        )
        if dim_creator.is_from_netcdf:
            if any(
                isinstance(attr_creator, NetCDF4VarToAttrConverter)
                for attr_creator in self._array_registry.attr_creators()
            ):
                raise ValueError(
                    "Cannot add a new NetCDF dimension converter to an array that "
                    "already contains NetCDF variable to attribute converters."
                )
        self._array_registry.inject_dim_creator(dim_creator, position)

    @property
    def max_fragment_shape(self):
        """Maximum shape of a fragment when copying from NetCDF to TileDB.

        For a dense array, this is the shape of dense fragment. For a sparse array,
        it is the maximum number of coordinates copied for each dimension.
        """
        return tuple(dim_creator.max_fragment_length for dim_creator in self)

    @max_fragment_shape.setter
    def max_fragment_shape(self, value: Sequence[Optional[int]]):
        if len(value) != self.ndim:
            raise ValueError(
                f"The length of the max_fragment_shape must match the number of "
                f"dimensions. Input of length {len(value)} provide for an array with "
                f"{self.ndim} dimensions."
            )
        for max_fragment_length, dim_creator in zip(value, self):
            dim_creator.max_fragment_length = max_fragment_length

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
