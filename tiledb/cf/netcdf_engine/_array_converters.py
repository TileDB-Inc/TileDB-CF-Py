"""Classes for converting NetCDF4 files to TileDB."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import netCDF4
import numpy as np

import tiledb
from tiledb.cf.core import (
    ArrayCreator,
    AttrCreator,
    CFSourceConnector,
    DomainCreator,
    SharedDim,
)

from ..core._array_creator import ArrayCreatorCore, DomainDimRegistry
from ..core.registry import Registry
from ._attr_converters import NetCDF4ToAttrConverter
from ._dim_converters import NetCDF4ToDimConverter
from ._utils import COORDINATE_SUFFIX
from .source import NetCDF4VariableSource, NetCDFGroupReader


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
        offsets_filters: Filters for the offsets for variable length attributes or
            dimensions.
        attrs_filters: Default filters to use when adding an attribute to the array.
        allows_duplicates: Specifies if multiple values can be stored at the same
             coordinate. Only allowed for sparse arrays.
    """

    def __init__(self, netcdf_group=None, **kwargs):
        super().__init__(**kwargs)
        self._netcdf_group_reader = (
            NetCDFGroupReader() if netcdf_group is None else netcdf_group
        )

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

    def _new_core(
        self, sparse: bool, dim_registry: Registry[SharedDim], dim_names: Sequence[str]
    ):
        return NetCDF4ArrayConverterCore(sparse, dim_registry, dim_names)

    def _new_domain_creator(self):
        return NetCDF4DomainConverter(self._core)

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

        if self._core.nfragment == 0:
            if len(ncvar.shape) == self.ndim:
                target_region = tuple((0, dim_size - 1) for dim_size in ncvar.shape)
            else:
                dim_sizes = {dim.name: dim.size for dim in ncvar.get_dims()}
                target_region = tuple(
                    (0, dim_sizes.get(dim.name, 1) - 1) for dim in self._domain_creator
                )
            self._core.add_dense_fragment_writer(target_region=target_region)
        if filters is None:
            filters = self.attrs_filters

        source = CFSourceConnector(
            NetCDF4VariableSource.from_variable(
                netcdf_variable=ncvar,
                unpack=unpack,
                netcdf_group=self._netcdf_group_reader,
            ),
            dtype=dtype,
            fill=fill,
        )
        if name is None:
            name = (
                ncvar.name
                if ncvar.name not in ncvar.dimensions
                else ncvar.name + COORDINATE_SUFFIX
            )
        attr_creator = AttrCreator(
            registry=self._attr_registry,
            name=name,
            dtype=source.dtype,
            fill=source.fill,
            var=var,
            nullable=nullable,
            filters=filters,
        )
        attr_creator.set_fragment_data(0, source)

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
        self._netcdf_group_reader.open(input_netcdf_group=netcdf_group)
        if assigned_dim_values is not None:
            for dim_creator in self.domain_creator:
                if dim_creator.name in assigned_dim_values:
                    dim_creator.set_fragment_data(
                        0, assigned_dim_values[dim_creator.name]
                    )
        if assigned_attr_values is not None:
            for attr_creator in self:
                if attr_creator.name in assigned_attr_values:
                    attr_creator.set_fragment_data(
                        0, assigned_attr_values[attr_creator.name]
                    )
        # TODO: Implement
        if tiledb_timestamp is not None:
            raise NotImplementedError()
        self.write(
            uri=tiledb_uri,
            key=tiledb_key,
            ctx=tiledb_ctx,
            append=True,
            skip_metadata=not copy_metadata,
        )


class NetCDF4ArrayConverterCore(ArrayCreatorCore):
    def _new_dim_creator(self, dim_name: str, **kwargs):
        return NetCDF4ToDimConverter(
            self._dim_registry[dim_name], registry=DomainDimRegistry(self), *kwargs
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

    @property
    def max_fragment_shape(self):
        """Maximum shape of a fragment when copying from NetCDF to TileDB.

        For a dense array, this is the shape of dense fragment. For a sparse array,
        it is the maximum number of coordinates copied for each dimension.
        """
        # TODO: Fix this
        raise NotImplementedError()

    @max_fragment_shape.setter
    def max_fragment_shape(self, value: Sequence[Optional[int]]):
        # TODO Fix this
        raise NotImplementedError()

    @property
    def netcdf_dims(self):
        """Ordered tuple of NetCDF dimension names for dimension converters."""
        return tuple(
            dim_creator.base.input_dim_name
            for dim_creator in self
            if hasattr(dim_creator.base, "input_dim_name")
        )
