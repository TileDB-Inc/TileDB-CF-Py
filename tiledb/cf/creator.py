# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Classes for creating a dataspace."""

from __future__ import annotations

from abc import ABCMeta
from collections import OrderedDict
from io import StringIO
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import tiledb

from .core import METADATA_ARRAY_NAME, Group, GroupSchema, VirtualGroup

DType = Union[int, float, str, None]
DATA_SUFFIX = ".data"
INDEX_SUFFIX = ".index"


def dataspace_name(full_name: str):
    """Returns dataspace name for from full dimension or attribute name.

    Parameters:
        full_name: The full name of the dimension or attribute as it will be written in
            TileDB.

    Returns:
        The name of the dimension or attribute as it would be written in the NetCDF data
        model.
    """
    if full_name.endswith(DATA_SUFFIX):
        return full_name[: -len(DATA_SUFFIX)]
    if full_name.endswith(INDEX_SUFFIX):
        return full_name[: -len(INDEX_SUFFIX)]
    return full_name


class DataspaceCreator:
    """Creator for a group of arrays that satify the CF Dataspace Convention.


    This class can be used directly to create a TileDB group that follows the
    TileDB CF Dataspace convention. It is also useful as a super class for
    converters/ingesters of data from sources that follow a NetCDF or NetCDF-like
    data model to TileDB.
    """

    def __init__(self):
        self._registry = DataspaceRegistry()

    def __repr__(self):
        output = StringIO()
        output.write("DataspaceCreator(")
        if self._registry.ndim > 0:
            output.write("\n Shared Dimensions:\n")
            for dim in self._registry.shared_dims():
                output.write(f"  '{dim.name}':  {repr(dim)},\n")
        if self._registry.narray > 0:
            output.write("\n Array Creators:\n")
            for array_creator in self._registry.array_creators():
                output.write(f"  '{array_creator.name}':{repr(array_creator)}\n")
        output.write(")")
        return output.getvalue()

    def _repr_html_(self):
        output = StringIO()
        output.write(f"<h4>{self.__class__.__name__}</h4>\n")
        output.write("<ul>\n")
        output.write("<li>\n")
        output.write("Shared Dimensions\n")
        if self._registry.ndim > 0:
            output.write("<table>\n")
            for dim in self._registry.shared_dims():
                output.write(
                    f'<tr><td style="text-align: left;">{dim.html_input_summary()} '
                    f"&rarr; SharedDim({dim.html_output_summary()})</td>\n</tr>\n"
                )
            output.write("</table>\n")
        output.write("</li>\n")
        output.write("<li>\n")
        output.write("Array Creators\n")
        for array_creator in self._registry.array_creators():
            output.write("<details>\n")
            output.write("<summary>\n")
            output.write(
                f"{array_creator.__class__.__name__} <em>{array_creator.name}</em>("
                f"{', '.join(map(lambda x: str(x.name), array_creator.domain_creator))}"
                f")\n"
            )
            output.write("</summary>\n")
            output.write(f"{array_creator.html_summary()}\n")
            output.write("</details>\n")
        output.write("</li>\n")
        output.write("</ul>\n")
        return output.getvalue()

    def add_array_creator(
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
        attrs_filters: Optional[tiledb.FilterList] = None,
        allows_duplicates: bool = False,
        sparse: bool = False,
    ):
        """Adds a new array to the CF dataspace.

        The name of each array must be unique. All other properties should satisfy
        the same requirements as a ``tiledb.ArraySchema``.

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
            dim_filters: A dict from dimension name to a :class:`tiledb.FilterList`
                for dimensions in the array. Overrides the values set in
                ``coords_filters``.
            offsets_filters: Filters for the offsets for variable length attributes or
                dimensions.
            attrs_filters: Default filters to use when adding an attribute to the
                array.
            allows_duplicates: Specifies if multiple values can be stored at the same
                 coordinate. Only allowed for sparse arrays.
            sparse: Specifies if the array is a sparse TileDB array (true) or dense
                TileDB array (false).
        """
        ArrayCreator(
            dataspace_registry=self._registry,
            name=array_name,
            dims=dims,
            cell_order=cell_order,
            tile_order=tile_order,
            capacity=capacity,
            tiles=tiles,
            coords_filters=coords_filters,
            dim_filters=dim_filters,
            offsets_filters=offsets_filters,
            attrs_filters=attrs_filters,
            allows_duplicates=allows_duplicates,
            sparse=sparse,
        )

    def add_attr_creator(
        self,
        attr_name: str,
        array_name: str,
        dtype: np.dtype,
        fill: Optional[Union[int, float, str]] = None,
        var: bool = False,
        nullable: bool = False,
        filters: Optional[tiledb.FilterList] = None,
    ):
        """Adds a new attribute to an array in the CF dataspace.

        The 'dataspace name' (name after dropping the suffix ``.data`` or ``.index``)
        must be unique.

        Parameters:
            attr_name: Name of the new attribute that will be added.
            array_name: Name of the array the attribute will be added to.
            dtype: Numpy dtype of the new attribute.
            fill: Fill value for unset cells.
            var: Specifies if the attribute is variable length (automatic for
                byte/strings).
            nullable: Specifies if the attribute is nullable using validity tiles.
            filters: Specifies compression filters for the attribute.
        """
        array_creator = self._registry.get_array_creator(array_name)
        array_creator.add_attr_creator(attr_name, dtype, fill, var, nullable, filters)

    def add_shared_dim(self, dim_name: str, domain: Tuple[Any, Any], dtype: np.dtype):
        """Adds a new dimension to the CF dataspace.

        Each dimension name must be unique. Adding a dimension where the name, domain,
        and dtype matches a current dimension does nothing.

        Parameters:
            dim_name: Name of the new dimension to be created.
            domain: The (inclusive) interval on which the dimension is valid.
            dtype: The numpy dtype of the values and domain of the dimension.
        """
        SharedDim(self._registry, dim_name, domain, dtype)

    def array_creators(self):
        """Iterates over array creators in the CF dataspace."""
        return self._registry.array_creators()

    def create_array(
        self,
        uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Creates a TileDB array for a CF dataspace with only one array.

        Parameters:
            uri: Uniform resource identifier for the TileDB array to be created.
            key: If not ``None``, encryption key to decrypt the array.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        if self._registry.narray != 1:
            raise ValueError(
                f"Can only use `create_array` for a {self.__class__.__name__} with "
                f"exactly 1 array creator."
            )
        array_creator = next(self._registry.array_creators())
        array_creator.create(uri, key=key, ctx=ctx)

    def create_group(
        self,
        uri: str,
        key: Optional[Union[Dict[str, str], str]] = None,
        ctx: Optional[tiledb.Ctx] = None,
        append: bool = False,
    ):
        """Creates a TileDB group and arrays for the CF dataspace.

        Parameters:
            uri: Uniform resource identifier for the TileDB group to be created.
            key: If not ``None``, encryption key, or dictionary of encryption keys, to
                decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            append: If ``True``, add arrays in the dataspace to an already existing
                group. The arrays in the dataspace cannot be in the group that is being
                append to.
        """
        schema = self.to_schema(ctx)
        Group.create(uri, schema, key, ctx, append=append)

    def create_virtual_group(
        self,
        uri: str,
        key: Optional[Union[Dict[str, str], str]] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Creates TileDB arrays for the CF dataspace.

        Parameters:
            uri: Prefix for the uniform resource identifier for the TileDB arrays that
                will be created.
            key: If not ``None``, encryption key, or dictionary of encryption keys, to
                decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        VirtualGroup.create(uri, self.to_schema(ctx), key, ctx)

    def get_array_creator(self, array_name: str):
        """Returns the array creator with the requested name.

        Parameters:
            array_name: Name of the array to return.
        """
        return self._registry.get_array_creator(array_name)

    def get_array_creator_by_attr(self, attr_name: str):
        """Returns the array creator with the requested attribute in it.

        Parameters:
            attr_name: Name of the attribute to return the array creator with.
        """
        return self._registry.get_array_creator_by_attr(attr_name)

    def get_shared_dim(self, dim_name: str):
        """Returns the shared dimension with the requested name.

        Parameters:
            array_name: Name of the array to return.
        """
        return self._registry.get_shared_dim(dim_name)

    def remove_array_creator(self, array_name: str):
        """Removes the specified array and all its attributes from the CF dataspace.

        Parameters:
            array_name: Name of the array that will be removed.
        """
        self._registry.deregister_array_creator(array_name)

    def remove_attr_creator(self, attr_name: str):
        """Removes the specified attribute from the CF dataspace.

        Parameters:
            attr_name: Name of the attribute that will be removed.
        """
        array_creator = self._registry.get_array_creator_by_attr(attr_name=attr_name)
        array_creator.remove_attr_creator(attr_name)

    def remove_shared_dim(self, dim_name: str):
        """Removes the specified dimension from the CF dataspace.

        This can only be used to remove dimensions that are not currently being used in
        an array.

        Parameters:
            dim_name: Name of the dimension to be removed.
        """
        self._registry.deregister_shared_dim(dim_name)

    def shared_dims(self):
        """Iterators over shared dimensions in the CF dataspace."""
        return self._registry.shared_dims()

    def to_schema(self, ctx: Optional[tiledb.Ctx] = None) -> GroupSchema:
        """Returns a group schema for the CF dataspace.

        Parameters:
           ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        array_schemas = {}
        for array_creator in self._registry.array_creators():
            try:
                array_schemas[array_creator.name] = array_creator.to_schema(ctx)
            except tiledb.libtiledb.TileDBError as err:
                raise RuntimeError(
                    f"Failed to create an ArraySchema for array '{array_creator.name}'."
                ) from err
        group_schema = GroupSchema(array_schemas)
        return group_schema


class DataspaceRegistry:
    def __init__(self):
        self._shared_dims: Dict[str, SharedDim] = {}
        self._array_creators: Dict[str, ArrayCreator] = {}
        self._attr_to_array: Dict[str, str] = {}

    def array_creators(self):
        """Iterator over array creators in the CF dataspace."""
        return iter(self._array_creators.values())

    def check_new_array_name(self, array_name: str):
        if array_name in self._array_creators:
            raise ValueError(f"An array with name '{array_name}' already exists.")
        if array_name == METADATA_ARRAY_NAME:
            raise ValueError(f"The array name '{METADATA_ARRAY_NAME}' is reserved.")

    def check_new_attr_name(self, attr_name: str):
        if attr_name in self._attr_to_array:
            raise ValueError(f"An attribute with name '{attr_name}' already exists.")
        ds_name = dataspace_name(attr_name)
        ds_names = {ds_name, ds_name + DATA_SUFFIX, ds_name + INDEX_SUFFIX}
        if not ds_names.isdisjoint(self._attr_to_array.keys()):
            raise ValueError(
                f"An attribute named with the dataspace name '{ds_name}' already "
                f"exists."
            )

    def check_new_dim(self, shared_dim: SharedDim):
        if (
            shared_dim.name in self._shared_dims
            and shared_dim != self._shared_dims[shared_dim.name]
        ):
            raise ValueError(
                f"A different dimension with name '{shared_dim.name}' already exists."
            )

    def check_rename_shared_dim(self, original_name: str, new_name: str):
        if new_name in self._shared_dims:
            raise NotImplementedError(
                f"Cannot rename dimension '{original_name}' to '{new_name}'. A "
                f"dimension with the same name already exists, and merging dimensions "
                f"has not yet been implemented."
            )
        if new_name in self._attr_to_array:
            array_creator = self.get_array_creator_by_attr(new_name)
            if original_name in (
                dim_creator.name for dim_creator in array_creator.domain_creator
            ):
                raise ValueError(
                    f"Cannot rename dimension '{original_name}' to '{new_name}'. An"
                    f" attribute with the same name already exists in the array "
                    f"'{array_creator.name}' that uses this dimension."
                )

    def deregister_array_creator(self, array_name: str):
        """Removes the specified array and all its attributes from the CF dataspace.

        Parameters:
            array_name: Name of the array that will be removed.
        """
        array_creator = self._array_creators.pop(array_name)
        for attr_creator in array_creator:
            del self._attr_to_array[attr_creator.name]

    def deregister_attr_creator(self, attr_name: str):
        del self._attr_to_array[attr_name]

    def deregister_shared_dim(self, dim_name: str):
        array_list = [
            array_creator.name
            for array_creator in self.array_creators()
            if dim_name
            in (dim_creator.name for dim_creator in array_creator.domain_creator)
        ]
        if array_list:
            raise ValueError(
                f"Cannot remove dimension '{dim_name}'. Dimension is being used in "
                f"arrays: {array_list}."
            )
        del self._shared_dims[dim_name]

    def get_array_creator(self, array_name: str) -> ArrayCreator:
        """Returns the array creator with the requested name."""
        return self._array_creators[array_name]

    def get_array_creator_by_attr(self, attr_name: str) -> ArrayCreator:
        """Returns an array creator that contains the requested attribute."""
        try:
            array_name = self._attr_to_array[attr_name]
        except KeyError as err:
            raise KeyError(f"No attribute with the name '{attr_name}'.") from err
        return self._array_creators[array_name]

    def get_attr_creator(self, attr_name: str) -> AttrCreator:
        """Returns the attribute creator with the requested name."""
        array_creator = self.get_array_creator_by_attr(attr_name=attr_name)
        return array_creator.attr_creator(attr_name)

    def get_shared_dim(self, dim_name: str) -> SharedDim:
        """Returns the dim creator with the requested name."""
        return self._shared_dims[dim_name]

    @property
    def narray(self) -> int:
        return len(self._array_creators)

    @property
    def ndim(self) -> int:
        return len(self._shared_dims)

    def register_array_creator(self, array_creator: ArrayCreator):
        """Registers a new array creator with the CF dataspace."""
        self.check_new_array_name(array_creator.name)
        self._array_creators[array_creator.name] = array_creator

    def register_attr_to_array(self, array_name: str, attr_name: str):
        """Registers a new attribute name to an array creator."""
        if array_name not in self._array_creators:  # pragma: no cover
            raise KeyError(f"No array named '{array_name}' exists.")
        self.check_new_attr_name(attr_name)
        self._attr_to_array[attr_name] = array_name

    def register_shared_dim(self, shared_dim: SharedDim):
        """Registers a new shared dimension to the CF dataspace.

        Parameters:
            shared_dim: The new shared dimension to register.
        """
        self.check_new_dim(shared_dim)
        self._shared_dims[shared_dim.name] = shared_dim

    def shared_dims(self):
        """Iterates over shared dimensions in the CF dataspace."""
        return iter(self._shared_dims.values())

    def update_array_creator_name(self, original_name: str, new_name: str):
        self._array_creators[new_name] = self._array_creators.pop(original_name)
        for attr_creator in self._array_creators[new_name]:
            self._attr_to_array[attr_creator.name] = new_name

    def update_attr_creator_name(self, original_name: str, new_name: str):
        self._attr_to_array[new_name] = self._attr_to_array.pop(original_name)

    def update_shared_dim_name(self, original_name: str, new_name: str):
        self._shared_dims[new_name] = self._shared_dims.pop(original_name)


class ArrayCreator:
    """Creator for a TileDB array using shared dimension definitions.

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
        sparse: If ``True``, creates a sparse array. Otherwise, create
    """

    def __init__(
        self,
        dataspace_registry: DataspaceRegistry,
        name: str,
        dims: Sequence[str],
        cell_order: str = "row-major",
        tile_order: str = "row-major",
        capacity: int = 0,
        tiles: Optional[Sequence[int]] = None,
        coords_filters: Optional[tiledb.FilterList] = None,
        dim_filters: Optional[Dict[str, tiledb.FilterList]] = None,
        offsets_filters: Optional[tiledb.FilterList] = None,
        attrs_filters: Optional[tiledb.FilterList] = None,
        allows_duplicates: bool = False,
        sparse: bool = False,
    ):
        if isinstance(dims, str):
            dims = (dims,)
        if len(set(dims)) != len(dims):
            raise ValueError(
                "Cannot create array; the array has repeating dimensions. All "
                "dimensions must have a unique name."
            )
        self._registry, self._domain_creator = self._register(
            dataspace_registry, name, dims
        )
        self.cell_order = cell_order
        self.tile_order = tile_order
        self.capacity = capacity
        if tiles is not None:
            self._domain_creator.tiles = tiles
        self.coords_filters = coords_filters
        if dim_filters is not None:
            for dim_name, filters in dim_filters.items():
                self._domain_creator.dim_creator(dim_name).filters = filters
        self.offsets_filters = offsets_filters
        self.attrs_filters = attrs_filters
        self.allows_duplicates = allows_duplicates
        self.sparse = sparse
        self._name = name
        dataspace_registry.register_array_creator(self)

    def __iter__(self):
        """Returns iterator over attribute creators."""
        return self._registry.attr_creators()

    def __repr__(self) -> str:
        output = StringIO()
        output.write("  ArrayCreator(\n")
        output.write("     domain=Domain(*[\n")
        for dim_creator in self._domain_creator:
            output.write(f"       {repr(dim_creator)},\n")
        output.write("     ]),\n")
        output.write("     attrs=[\n")
        for attr_creator in self:
            output.write(f"       {repr(attr_creator)},\n")
        output.write("     ],\n")
        output.write(
            f"     cell_order='{self.cell_order}',\n"
            f"     tile_order='{self.tile_order}',\n"
        )
        output.write(f"     capacity={self.capacity},\n")
        output.write(f"     sparse={self.sparse},\n")
        if self.sparse:
            output.write(f"     allows_duplicates={self.allows_duplicates},\n")
        if self.coords_filters is not None:
            output.write("     coords_filters=FilterList([")
            for index, coord_filter in enumerate(self.coords_filters):
                output.write(f"{repr(coord_filter)}")
                if index < len(self.coords_filters):
                    output.write(", ")
            output.write("])\n")
        output.write("  )")
        return output.getvalue()

    def _register(
        self, dataspace_registry: DataspaceRegistry, name: str, dim_names: Sequence[str]
    ):
        dim_creators = tuple(
            DimCreator(dataspace_registry.get_shared_dim(dim_name))
            for dim_name in dim_names
        )

        array_registry = ArrayRegistry(dataspace_registry, name, dim_creators)
        return array_registry, DomainCreator(array_registry, dataspace_registry)

    def attr_creator(self, key: Union[int, str]) -> AttrCreator:
        """Returns the requested attribute creator

        Parameters:
            key: The attribute creator index (int) or name (str).

        Returns:
            The attribute creator at the given index of name.
        """
        return self._registry.get_attr_creator(key)

    def add_attr_creator(
        self,
        name: str,
        dtype: np.dtype,
        fill: Optional[Union[int, float, str]] = None,
        var: bool = False,
        nullable: bool = False,
        filters: Optional[tiledb.FilterList] = None,
    ):
        """Adds a new attribute to an array in the CF dataspace.

        The attribute's 'dataspace name' (name after dropping the suffix ``.data`` or
        ``.index``) must be unique.

        Parameters:
            name: Name of the new attribute that will be added.
            dtype: Numpy dtype of the new attribute.
            fill: Fill value for unset cells.
            var: Specifies if the attribute is variable length (automatic for
                byte/strings).
            nullable: Specifies if the attribute is nullable using validity tiles.
            filters: Specifies compression filters for the attribute. If ``None``, use
                the array's ``attrs_filters`` property.
        """
        if filters is None:
            filters = self.attrs_filters
        AttrCreator(self._registry, name, dtype, fill, var, nullable, filters)

    def create(
        self, uri: str, key: Optional[str] = None, ctx: Optional[tiledb.Ctx] = None
    ):
        """Creates a TileDB array at the provided URI.

        Parameters:
            uri: Uniform resource identifier for the array to be created.
            key: If not ``None``, encryption key to decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        tiledb.Array.create(uri, self.to_schema(ctx), key, ctx)

    @property
    def domain_creator(self) -> DomainCreator:
        """Domain creator that creates the domain for the TileDB array."""
        return self._domain_creator

    @property
    def name(self) -> str:
        """Name of the array."""
        return self._registry.name

    @name.setter
    def name(self, name: str):
        self._registry.name = name

    @property
    def nattr(self) -> int:
        """Number of attributes in the array."""
        return self._registry.nattr

    @property
    def ndim(self) -> int:
        """Number of dimensions in the array."""
        return self._registry.ndim

    def html_summary(self) -> str:
        """Returns a string HTML summary of the :class:`ArrayCreator`."""
        cell_style = 'style="text-align: left;"'
        output = StringIO()
        output.write("<ul>\n")
        output.write("<li>\n")
        output.write("Domain\n")
        output.write("<table>\n")
        for dim_creator in self._domain_creator:
            output.write(
                f"<tr><td {cell_style}>{dim_creator.html_summary()}</td></tr>\n"
            )
        output.write("</table>\n")
        output.write("</li>\n")
        output.write("<li>\n")
        output.write("Attributes\n")
        output.write("<table>\n")
        for attr_creator in self:
            output.write(
                f"<tr><td {cell_style}>{attr_creator.html_summary()}</td></tr>\n"
            )
        output.write("</table>\n")
        output.write("</li>\n")
        output.write("<li>\n")
        output.write("Array Properties\n")
        output.write(
            f"<table>\n"
            f"<tr><td {cell_style}>cell_order={self.cell_order}</td></tr>\n"
            f"<tr><td {cell_style}>tile_order={self.tile_order}</td></tr>\n"
            f"<tr><td {cell_style}>capacity={self.capacity}</td></tr>\n"
            f"<tr><td {cell_style}>sparse={self.sparse}</td></tr>\n"
        )
        if self.sparse:
            output.write(
                f"<tr><td {cell_style}>allows_duplicates"
                f"={self.allows_duplicates}</td></tr>\n"
            )
        output.write(
            f"<tr><td {cell_style}>coords_filters={self.coords_filters}</td></tr>\n"
        )
        output.write("</table>\n")
        output.write("</li>\n")
        output.write("</ul>\n")
        return output.getvalue()

    def remove_attr_creator(self, attr_name):
        """Removes the requested attribute from the array.

        Parameters:
            attr_name: Name of the attribute to remove.
        """
        return self._registry.deregister_attr_creator(attr_name)

    def to_schema(self, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
        """Returns an array schema for the array.

        Parameters:
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        if self._registry.nattr == 0:
            raise ValueError("Cannot create schema for array with no attributes.")
        domain = self._domain_creator.to_tiledb(ctx)
        attrs = tuple(attr_creator.to_tiledb(ctx) for attr_creator in self)
        return tiledb.ArraySchema(
            domain=domain,
            attrs=attrs,
            cell_order=self.cell_order,
            tile_order=self.tile_order,
            capacity=self.capacity,
            coords_filters=self.coords_filters,
            offsets_filters=self.offsets_filters,
            allows_duplicates=self.allows_duplicates,
            sparse=self.sparse,
            ctx=ctx,
        )


class ArrayRegistry:
    def __init__(
        self,
        dataspace_registry: DataspaceRegistry,
        name: str,
        dim_creators: Tuple[DimCreator, ...],
    ):
        self._dataspace_registry = dataspace_registry
        self._name = name
        self._dim_creators = dim_creators
        self._attr_creators: Dict[str, AttrCreator] = OrderedDict()

    def attr_creators(self):
        """Iterates over attribute creators in the array creator."""
        return iter(self._attr_creators.values())

    def check_new_attr_name(self, attr_name):
        if attr_name in self._attr_creators:
            raise ValueError(
                f"An attribute with the name '{attr_name}' already exists in this "
                f"array."
            )
        for dim_creator in self.dim_creators():
            if attr_name == dim_creator.name:
                raise ValueError(
                    f"A dimension with the name '{attr_name}' already exists in this "
                    f"array."
                )

    def deregister_attr_creator(self, attr_name: str):
        """Removes an attribute from the group."""
        self._dataspace_registry.deregister_attr_creator(attr_name)
        del self._attr_creators[attr_name]

    def dim_creators(self):
        """Iterates over dimension creators in the array creator."""
        return iter(self._dim_creators)

    def get_attr_creator(self, key: Union[str, int]) -> AttrCreator:
        """Returns the requested attribute creator.

        Parameters:
            attr_name: Name of the attribute to return.
        """
        if isinstance(key, int):
            return tuple(self._attr_creators.values())[key]
        return self._attr_creators[key]

    def get_dim_creator(self, key: Union[int, str]) -> DimCreator:
        """Returns the requested dimension creator.

        Parameters:
            key: Name (string) or index (integer) of the dimension to return.

        Returns:
            The requested dimension creator.
        """
        if isinstance(key, int):
            return self._dim_creators[key]
        index = self.get_dim_position_by_name(key)
        return self._dim_creators[index]

    def get_dim_position_by_name(self, dim_name: str) -> int:
        """Returns the dimension position of the requested dimension name.

        Parameters:
            dim_name: Name of the dimension to get the position of.

        Returns:
            The position of the requested dimension in the array domain.
        """
        for index, dim_creator in enumerate(self._dim_creators):
            if dim_creator.name == dim_name:
                return index
        raise KeyError(f"Dimension creator with name '{dim_name}' not found.")

    def inject_dim_creator(self, dim_creator: DimCreator, position: int):
        """Add an additional dimension into the domain of the array.

        Parameters:
            dim_creator: The dimension creator to add.
            position: Position of the shared dimension. Negative values count backwards
                from the end of the new number of dimensions.
        """
        if dim_creator.name in {dim_creator.name for dim_creator in self._dim_creators}:
            raise ValueError(
                f"Cannot add dimension creator `{dim_creator.name}` to this array. "
                f"That dimension is already in use."
            )
        if dim_creator.name in self._attr_creators:
            raise ValueError(
                f"Cannot add dimension creator `{dim_creator.name}` to this array. An "
                f"attribute creator with that name already exists."
            )
        index = self.ndim + 1 + position if position < 0 else position
        if index < 0 or index > self.ndim:
            raise IndexError(
                f"Cannot add dimension to position {position} for an array with "
                f"{self.ndim} dimensions."
            )
        self._dim_creators = (
            self._dim_creators[:index] + (dim_creator,) + self._dim_creators[index:]
        )

    @property
    def name(self) -> str:
        """Name of the array."""
        return self._name

    @name.setter
    def name(self, name: str):
        self._dataspace_registry.check_new_array_name(name)
        self._dataspace_registry.update_array_creator_name(self._name, name)
        self._name = name

    @property
    def nattr(self) -> int:
        return len(self._attr_creators)

    @property
    def ndim(self) -> int:
        return len(self._dim_creators)

    def register_attr_creator(self, attr_creator):
        self.check_new_attr_name(attr_creator.name)
        attr_name = attr_creator.name
        if self._dataspace_registry is not None:
            self._dataspace_registry.register_attr_to_array(self._name, attr_name)
        self._attr_creators[attr_name] = attr_creator

    def remove_dim_creator(self, dim_index: int):
        """Remove a dim creator from the array.

        Parameters:
            dim_creator: The dimension creator to add.
            position: Position of the shared dimension. Negative values count backwards
                from the end of the new number of dimensions.
        """
        index = dim_index + self.ndim if dim_index < 0 else dim_index
        if index < 0 or index >= self.ndim:
            raise IndexError(
                f"Dimension index {dim_index} is outside the bounds of the domain."
            )
        self._dim_creators = (
            self._dim_creators[:dim_index] + self._dim_creators[dim_index + 1 :]
        )

    def update_attr_creator_name(self, original_name: str, new_name: str):
        """Renames an attribute in the array.

        Parameters:
            original_name: Current name of the attribute to be renamed.
            new_name: New name the attribute will be renamed to.
        """
        self._dataspace_registry.update_attr_creator_name(original_name, new_name)
        self._attr_creators[new_name] = self._attr_creators.pop(original_name)


class AttrCreator(metaclass=ABCMeta):
    """Creator for a TileDB attribute.

    Attributes:
        dtype: Numpy dtype of the attribute.
        fill: Fill value for unset cells.
        var: Specifies if the attribute is variable length (automatic for
            byte/strings).
        nullable: Specifies if the attribute is nullable using validity tiles.
        filters: Specifies compression filters for the attribute.
    """

    def __init__(
        self,
        array_registry: ArrayRegistry,
        name: str,
        dtype: np.dtype,
        fill: Optional[DType] = None,
        var: bool = False,
        nullable: bool = False,
        filters: Optional[tiledb.FilterList] = None,
    ):
        self._array_registry = array_registry
        self._name = name
        self.dtype = np.dtype(dtype)
        self.fill = fill
        self.var = var
        self.nullable = nullable
        self.filters = filters
        self._array_registry.register_attr_creator(self)

    def __repr__(self):
        filters_str = f", filters=FilterList({self.filters})" if self.filters else ""
        return (
            f"AttrCreator(name={self.name}, dtype='{self.dtype!s}', var={self.var}, "
            f"nullable={self.nullable}{filters_str})"
        )

    def html_summary(self) -> str:
        """Returns a string HTML summary of the :class:`AttrCreator`."""
        filters_str = f", filters=FilterList({self.filters})" if self.filters else ""
        return (
            f" &rarr; tiledb.Attr(name={self.name}, dtype='{self.dtype!s}', "
            f"var={self.var}, nullable={self.nullable}{filters_str})"
        )

    @property
    def name(self) -> str:
        """Name of the attribute."""
        return self._name

    @name.setter
    def name(self, name: str):
        self._array_registry.check_new_attr_name(name)
        self._array_registry.update_attr_creator_name(self._name, name)
        self._name = name

    def to_tiledb(self, ctx: Optional[tiledb.Ctx] = None) -> tiledb.Attr:
        """Returns a :class:`tiledb.Attr` using the current properties.

        Parameters:
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.

        Returns:
            Returns an attribute with the set properties.
        """
        return tiledb.Attr(
            name=self.name,
            dtype=self.dtype,
            fill=self.fill,
            var=self.var,
            nullable=self.nullable,
            filters=self.filters,
            ctx=ctx,
        )


class DomainCreator:
    """Creator for a TileDB domain."""

    def __init__(self, array_registry, dataspace_registry):
        self._array_registry = array_registry
        self._dataspace_registry = dataspace_registry

    def __iter__(self):
        return self._array_registry.dim_creators()

    def __len__(self):
        return self.ndim

    def inject_dim_creator(self, dim_name: str, position: int, **dim_kwargs):
        """Adds a new dimension creator at a specified location.

        Parameters:
            dim_name: Name of the shared dimension to add to the array's domain.
            position: Position of the shared dimension. Negative values count backwards
                from the end of the new number of dimensions.
            dim_kwargs: Keyword arguments to pass to :class:`DimCreator`.
        """
        self._array_registry.inject_dim_creator(
            DimCreator(self._dataspace_registry.get_shared_dim(dim_name), **dim_kwargs),
            position,
        )

    @property
    def ndim(self):
        """Number of dimensions in the domain."""
        return self._array_registry.ndim

    def dim_creator(self, dim_id):
        """Returns a dimension creator from the domain creator given the dimension's
        index or name.

        Parameter:
            dim_id: dimension index (int) or name (str)

        Returns:
            The dimension creator with the requested key.
        """
        return self._array_registry.get_dim_creator(dim_id)

    def remove_dim_creator(self, dim_id: Union[str, int]):
        """Removes a dimension creator from the array creator.

        Parameters:
            dim_id: dimension index (int) or name (str)
        """
        if isinstance(dim_id, int):
            self._array_registry.remove_dim_creator(dim_id)
        else:
            index = self._array_registry.get_dim_position_by_name(dim_id)
            self._array_registry.remove_dim_creator(index)

    @property
    def tiles(self):
        """Tiles for the dimension creators in the domain."""
        return tuple(dim_creator.tile for dim_creator in self)

    @tiles.setter
    def tiles(self, tiles: Sequence[Optional[int]]):
        if len(tiles) != self.ndim:
            raise ValueError(
                f"Cannot set tiles. Got {len(tiles)} tile(s) for an array with "
                f"{self.ndim} dimension(s)."
            )
        for dim_creator, tile in zip(self, tiles):
            dim_creator.tile = tile

    def to_tiledb(self, ctx: Optional[tiledb.Ctx] = None) -> tiledb.Domain:
        """Returns a TileDB domain from the contained dimension creators."""
        if self.ndim == 0:
            raise ValueError("Cannot create schema for array with no dimensions.")
        tiledb_dims = [dim_creator.to_tiledb() for dim_creator in self]
        return tiledb.Domain(tiledb_dims, ctx=ctx)


class DimCreator:
    """Creator for a TileDB dimension using a SharedDim.

    Attributes:
        tile: The tile size for the dimension.
        filters: Specifies compression filters for the dimension.
    """

    def __init__(
        self,
        base: SharedDim,
        tile: Optional[Union[int, float]] = None,
        filters: Optional[Union[tiledb.FilterList]] = None,
    ):
        self._base = base
        self.tile = tile
        self.filters = filters

    def __repr__(self):
        filters_str = ""
        if self.filters:
            filters_str = ", filters=FilterList(["
            for dim_filter in self.filters:
                filters_str += repr(dim_filter) + ", "
            filters_str += "])"
        return f"DimCreator({repr(self._base)}, tile={self.tile}{filters_str})"

    @property
    def base(self) -> SharedDim:
        """Shared definition for the dimensions name, domain, and dtype."""
        return self._base

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the values and domain of the dimension."""
        return self._base.dtype

    @property
    def domain(self) -> Optional[Tuple[Optional[DType], Optional[DType]]]:
        """The (inclusive) interval on which the dimension is valid."""
        return self._base.domain

    def html_summary(self) -> str:
        """Returns a string HTML summary of the :class:`DimCreator`."""
        filters_str = ""
        if self.filters:
            filters_str = ", filters=FilterList(["
            for dim_filter in self.filters:
                filters_str += repr(dim_filter) + ", "
            filters_str += "])"
        return (
            f"{self._base.html_input_summary()} &rarr; tiledb.Dim("
            f"{self._base.html_output_summary()}, tile={self.tile}{filters_str})"
        )

    @property
    def name(self) -> str:
        """Name of the dimension."""
        return self._base.name

    def to_tiledb(self, ctx: Optional[tiledb.Ctx] = None) -> tiledb.Domain:
        """Returns a :class:`tiledb.Dim` using the current properties.

        Parameters:
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.

        Returns:
            A tiledb dimension with the set properties.
        """
        if self.domain is None:
            raise ValueError(
                f"Cannot create a TileDB dimension for dimension '{self.name}'. No "
                f"domain is set."
            )
        return tiledb.Dim(
            name=self.name,
            domain=self.domain,
            tile=self.tile,
            filters=self.filters,
            dtype=self.dtype,
            ctx=ctx,
        )


class SharedDim(metaclass=ABCMeta):
    """Definition for the name, domain and data type of a collection of dimensions."""

    def __init__(
        self,
        dataspace_registry: DataspaceRegistry,
        name: str,
        domain: Optional[Tuple[Optional[DType], Optional[DType]]],
        dtype: np.dtype,
    ):
        self._name = name
        self.domain = domain
        self.dtype = np.dtype(dtype)
        self._dataspace_registry = dataspace_registry
        self._dataspace_registry.register_shared_dim(self)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if not isinstance(self, other.__class__):
            return False
        return (
            self.name == other.name
            and self.domain == other.domain
            and self.dtype == other.dtype
        )

    def __repr__(self) -> str:
        return (
            f"SharedDim(name={self.name}, domain={self.domain}, dtype='{self.dtype!s}')"
        )

    def html_input_summary(self) -> str:
        """Returns a HTML string summarizing the input for the dimension."""
        return ""

    def html_output_summary(self) -> str:
        """Returns a string HTML summary of the :class:`SharedDim`."""
        return f"name={self.name}, domain={self.domain}, dtype='{self.dtype!s}'"

    @property
    def is_index_dim(self) -> bool:
        """Returns ``True`` if this is an `index dimension` and ``False`` otherwise.

        An index dimension is a dimension with an integer data type and whose domain
        starts at 0.
        """
        if self.domain:
            return np.issubdtype(self.dtype, np.integer) and self.domain[0] == 0
        return False

    @property
    def name(self) -> str:
        """Name of the shared dimension."""
        return self._name

    @name.setter
    def name(self, name: str):
        self._dataspace_registry.check_rename_shared_dim(self._name, name)
        self._dataspace_registry.update_shared_dim_name(self._name, name)
        self._name = name
