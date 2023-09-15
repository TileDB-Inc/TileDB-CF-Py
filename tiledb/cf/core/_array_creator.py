from __future__ import annotations

from collections import OrderedDict
from io import StringIO
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
from typing_extensions import Self

import tiledb

from ._attr_creator import AttrCreator
from ._dim_creator import DimCreator
from ._shared_dim import SharedDim
from .registry import RegisteredByNameMixin, Registry


class ArrayDimRegistry:
    def __init__(self):
        self._shared_dims: Dict[str, SharedDim] = dict()

    def __delitem__(self, name: str):
        del self._shared_dims[name]

    def __getitem__(self, name: str) -> SharedDim:
        return self._shared_dims[name]

    def __setitem__(self, name: str, value: SharedDim):
        if value.is_registered:
            raise ValueError(f"Shared dimension '{value.name}' is already registered.")
        old_name = value.name
        if name != value.name:
            value.name = name
        if value.name in self._shared_dims and value != self._shared_dims[value.name]:
            if value.name != old_name:
                value.name = old_name
            raise ValueError(
                f"Cannot add shared dimension '{value.name}'. A different shared "
                f"dimension with that name already exists."
            )
        self._shared_dims[name] = value

    def rename(self, old_name: str, new_name: str):
        if new_name in self._shared_dims:
            raise NotImplementedError(
                f"Cannot rename dimension '{old_name}' to '{new_name}'. A "
                f"dimension with the same name already exists, and merging dimensions "
                f"has not yet been implemented."
            )
        self._shared_dims[new_name] = self._shared_dims.pop(old_name)


class ArrayCreator(RegisteredByNameMixin):
    """Creator for a TileDB array using shared dimension definitions.

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
        sparse: If ``True``, creates a sparse array. Otherwise, create
    """

    def __init__(
        self,
        *,
        name: str = "array",
        dim_order: Optional[Sequence[str]] = None,
        cell_order: str = "row-major",
        tile_order: str = "row-major",
        capacity: int = 0,
        tiles: Optional[Sequence[int]] = None,
        dim_filters: Optional[Dict[str, tiledb.FilterList]] = None,
        offsets_filters: Optional[tiledb.FilterList] = None,
        attrs_filters: Optional[tiledb.FilterList] = None,
        allows_duplicates: bool = False,
        sparse: bool = False,
        registry: Optional[Registry[Self]] = None,
        dim_registry: Optional[Registry[SharedDim]] = None,
        shared_dims: Optional[Iterable[SharedDim]] = None,
    ):
        # Check all dimension names are unique.
        if dim_order is None:
            dim_order = tuple()
        elif isinstance(dim_order, str):
            dim_order = (dim_order,)
        if len(set(dim_order)) != len(dim_order):
            raise ValueError(
                "Cannot create array; the array has repeating dimensions. All "
                "dimensions must have a unique name."
            )

        # Get dimension registry and add any new dimensions.
        if dim_registry is None:
            dim_registry = ArrayDimRegistry()
        if shared_dims is not None:
            for dim in shared_dims:
                dim_registry[dim.name] = dim

        # Initialize the core implementation, the domain creator, and the
        # attribute registry.
        self._core = self._new_core(dim_registry, dim_order)
        self._domain_creator = self._new_domain_creator()
        self._attr_registry = ArrayAttrRegistry(self._core)

        # Set array properties.
        self.cell_order = cell_order
        self.tile_order = tile_order
        self.capacity = capacity
        if tiles is not None:
            self._domain_creator.tiles = tiles
        if dim_filters is not None:
            for dim_name, filters in dim_filters.items():
                self._domain_creator.dim_creator(dim_name).filters = filters
        self.offsets_filters = offsets_filters
        self.attrs_filters = attrs_filters
        self.allows_duplicates = allows_duplicates
        self.sparse = sparse

        # Set name and registry for the array creator.
        super().__init__(name, registry)

    def __iter__(self):
        """Returns iterator over attribute creators."""
        return self._core.attr_creators()

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
        output.write("  )")
        return output.getvalue()

    def _new_core(self, dataspace_registry, dim_names: Sequence[str]):
        return ArrayCreatorCore(dataspace_registry, dim_names)

    def _new_domain_creator(self):
        return DomainCreator(self._core)

    def attr_creator(self, key: Union[int, str]) -> AttrCreator:
        """Returns the requested attribute creator

        Parameters:
            key: The attribute creator index (int) or name (str).

        Returns:
            The attribute creator at the given index of name.
        """
        return self._core.get_attr_creator(key)

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
        AttrCreator(
            registry=self._attr_registry,
            name=name,
            dtype=dtype,
            fill=fill,
            var=var,
            nullable=nullable,
            filters=filters,
        )

    def create(
        self, uri: str, key: Optional[str] = None, ctx: Optional[tiledb.Ctx] = None
    ):
        """Creates a TileDB array at the provided URI.

        Parameters:
            uri: Uniform resource identifier for the array to be created.
            key: If not ``None``, encryption key to decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        tiledb.Array.create(uri=uri, schema=self.to_schema(ctx), key=key, ctx=ctx)

    @property
    def domain_creator(self) -> DomainCreator:
        """Domain creator that creates the domain for the TileDB array."""
        return self._domain_creator

    def has_attr_creator(self, name: str) -> bool:
        """Returns if an attribute creator with the requested name is in the array
        creator

        Parameters:
            name: The name of the attribute creator to check for.
        """
        return self._core.has_attr_creator(name)

    @property
    def nattr(self) -> int:
        """Number of attributes in the array."""
        return self._core.nattr

    @property
    def ndim(self) -> int:
        """Number of dimensions in the array."""
        return self._core.ndim

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
        output.write("</table>\n")
        output.write("</li>\n")
        output.write("</ul>\n")
        return output.getvalue()

    def remove_attr_creator(self, attr_name):
        """Removes the requested attribute from the array.

        Parameters:
            attr_name: Name of the attribute to remove.
        """
        return self._core.deregister_attr_creator(attr_name)

    def to_schema(self, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
        """Returns an array schema for the array.

        Parameters:
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        if self._core.nattr == 0:
            raise ValueError("Cannot create schema for array with no attributes.")
        domain = self._domain_creator.to_tiledb(ctx)
        attrs = tuple(attr_creator.to_tiledb(ctx) for attr_creator in self)
        return tiledb.ArraySchema(
            domain=domain,
            attrs=attrs,
            cell_order=self.cell_order,
            tile_order=self.tile_order,
            capacity=self.capacity,
            offsets_filters=self.offsets_filters,
            allows_duplicates=self.allows_duplicates,
            sparse=self.sparse,
            ctx=ctx,
        )


class ArrayCreatorCore:
    def __init__(self, dim_registry: Registry[SharedDim], array_dims: Sequence[str]):
        self._dim_registry = dim_registry
        self._dim_creators = tuple(
            self._new_dim_creator(dim_name) for dim_name in array_dims
        )
        self._attr_creators: Dict[str, AttrCreator] = OrderedDict()

    def _new_dim_creator(self, dim_name: str, **kwargs):
        return DimCreator(self._dim_registry[dim_name], **kwargs)

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

    def has_attr_creator(self, name: str) -> bool:
        """Returns if an attribute creator with the requested name is in the array
        creator

        Parameters:
            name: The name of the attribute creator to check for.
        """
        return name in self._attr_creators

    def inject_dim_creator(self, dim_name: str, position: int, **dim_kwargs):
        """Add an additional dimension into the domain of the array."""
        dim_creator = self._new_dim_creator(dim_name, **dim_kwargs)
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
    def nattr(self) -> int:
        return len(self._attr_creators)

    @property
    def ndim(self) -> int:
        return len(self._dim_creators)

    def register_attr_creator(self, attr_creator):
        self.check_new_attr_name(attr_creator.name)
        attr_name = attr_creator.name
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
        self._attr_creators[new_name] = self._attr_creators.pop(original_name)


class ArrayAttrRegistry:
    def __init__(self, array_core: ArrayCreatorCore):
        self._core = array_core

    def __delitem__(self, name: str):
        self._core.deregister_attr_creator(name)

    def __getitem__(self, name: str) -> AttrCreator:
        return self._core.get_attr_creator(name)

    def __setitem__(self, name: str, value: AttrCreator):
        if value.is_registered:
            raise ValueError("AttrCreator '{value.name}' is already registered.")
        if name != value.name:
            value.name = name
        self._core.register_attr_creator(value)

    def rename(self, old_name: str, new_name: str):
        self._core.check_new_attr_name(new_name)
        self._core.update_attr_creator_name(old_name, new_name)


class DomainCreator:
    """Creator for a TileDB domain."""

    def __init__(self, array_core: ArrayCreatorCore):
        self._core = array_core

    def __iter__(self):
        return self._core.dim_creators()

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
        self._core.inject_dim_creator(dim_name, position, **dim_kwargs)

    @property
    def ndim(self):
        """Number of dimensions in the domain."""
        return self._core.ndim

    def dim_creator(self, dim_id):
        """Returns a dimension creator from the domain creator given the dimension's
        index or name.

        Parameter:
            dim_id: dimension index (int) or name (str)

        Returns:
            The dimension creator with the requested key.
        """
        return self._core.get_dim_creator(dim_id)

    def remove_dim_creator(self, dim_id: Union[str, int]):
        """Removes a dimension creator from the array creator.

        Parameters:
            dim_id: dimension index (int) or name (str)
        """
        if isinstance(dim_id, int):
            self._core.remove_dim_creator(dim_id)
        else:
            index = self._core.get_dim_position_by_name(dim_id)
            self._core.remove_dim_creator(index)

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
