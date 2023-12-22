from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from io import StringIO
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from typing_extensions import Self

import tiledb

from ._attr_creator import AttrCreator
from ._dim_creator import DimCreator
from ._fragment_writer import FragmentWriter
from ._shared_dim import SharedDim
from .registry import RegisteredByNameMixin, Registry
from .source import FieldData

DenseRange = Union[Tuple[int, int], Tuple[np.datetime64, np.datetime64]]


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

    Attributes
    ----------
    cell_order
        The order in which TileDB stores the cells on disk inside a
        tile. Valid values are: ``row-major`` (default) or ``C`` for row major;
        ``col-major`` or ``F`` for column major; or ``Hilbert`` for a Hilbert curve.
    tile_order
        The order in which TileDB stores the tiles on disk. Valid values are:
        ``row-major`` or ``C`` (default) for row major; or ``col-major`` or
        ``F`` for column major.
    capacity
        The number of cells in a data tile of a sparse fragment.
    tiles
        The tile extents to set on each dimension. The length must match the number
        of dimensions.
    dim_filters
        A dictionary from dimension name to TileDB filters to apply to the dimension.
    offsets_filters
        Filters for the offsets for variable length attributes or dimensions.
    attrs_filters
        Default filters to use when adding an attribute to the array.
    allows_duplicates
        Specifies if multiple values can be stored at the same
        coordinate. Only allowed for sparse arrays.
    sparse
        If ``True``, creates a sparse array. Otherwise, creates a dense array.
    registry
        Registry this array will belong to.
    dim_registry
        Registry for the shared dimensions this array will use. If none is provided,
        a registry will be created.
    shared_dims
        An ordered list of shared dimensions to use as the dimensions this array.
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
        self._core = self._new_core(sparse, dim_registry, dim_order)
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

    def __getitem__(self, key: Union[int, str]) -> AttrCreator:
        return self._core.get_attr_creator(key)

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

    def _new_core(
        self, sparse: bool, dim_registry: Registry[SharedDim], dim_names: Sequence[str]
    ):
        return ArrayCreatorCore(sparse, dim_registry, dim_names)

    def _new_domain_creator(self):
        return DomainCreator(self._core)

    def attr_creator(self, key: Union[int, str]) -> AttrCreator:
        """Returns the requested attribute creator

        Parameters
        ----------
        key
            The attribute creator index (int) or name (str).

        Returns
        -------
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

        Parameters
        ----------
        name
            Name of the new attribute that will be added.
        dtype
            Numpy dtype of the new attribute.
        fill
            Fill value for unset cells.
        var
            Specifies if the attribute is variable length (automatic for
            byte/strings).
        nullable
            Specifies if the attribute is nullable using validity tiles.
        filters
            Specifies compression filters for the attribute. If ``None``, use
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

    def add_dense_fragment_writer(
        self,
        target_region: Optional[Tuple[DenseRange, ...]] = None,
    ):
        """Add a writer for dense fragments.

        Parameters
        ----------
        target_region
            Region the fragments are written on. If ``None``, the region is
            set to the entire domain of the array.
        """
        self._core.add_dense_fragment_writer(target_region)

    def add_sparse_fragment_writer(
        self,
        *,
        size: Optional[int] = None,
        shape: Optional[Tuple[int, ...]] = None,
        form: str = "coo",
    ):
        """Add a writer for sparse fragments.

        There are two valid forms for the sparse writer: "coo" and "row-major".

        For "coo" form, the size is used to define the footprint of the data. This
        supports a general sparse writes. The full expanded data for each dimension
        must be provided.

        Example input data for "coo" form on a 2D array:
           dim1 = [1, 2, 1, 2]
           dim2 = [3, 3, 4, 4]
           attr = [1, 2, 3, 4]

        For "row-major" form, a grid of data is provided. The data on each dimension
        is just the dimension for that part of the grid:

        Example input data for "row-major" form on a 2D array:
            dim1 = [1, 2]
            dim2 = [3, 4]
            attr = [1, 2, 3, 4]

        Parameters
        ----------
        size
            The number of elements the fragment stores.
        shape
            The shape of the fragment. Required for "row-major" form.
        form
            The form for the dimension data. Can either be "coo" (coordinate form) or
            "row-major".
        """
        if size is not None and shape is not None and np.prod(shape) != size:
            raise ValueError("Mismatch between shape={shape} and size={size}.")

        if form == "coo":
            if size is None:
                if shape is None:
                    raise TypeError(
                        "Must provided shape or size for writing in 'coo' form."
                    )
                size = np.prod(shape)
            self._core.add_sparse_coo_fragment_writer(size)
        elif form == "row-major":
            if shape is None:
                raise ValueError("Must set shape when using form 'row-major'.")
            self._core.add_sparse_row_major_fragment_writer(shape)
        else:
            raise ValueError(
                f"'{form}' is not a valid value for 'form'. Valid options include: "
                f"'coo', 'row-major'."
            )

    def create(
        self, uri: str, key: Optional[str] = None, ctx: Optional[tiledb.Ctx] = None
    ):
        """Creates a TileDB array at the provided URI.

        Parameters
        ----------
        uri
            Uniform resource identifier for the array to be created.
        key
            If not ``None``, encryption key to decrypt arrays.
        ctx
            If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        tiledb.Array.create(uri=uri, schema=self.to_schema(ctx), key=key, ctx=ctx)

    def write(
        self,
        uri: str,
        *,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        timestamp: Optional[int] = None,
        append: bool = False,
        skip_metadata: bool = False,
        writer_indices: Optional[Iterable[int]] = None,
    ):
        """Writes data to a TileDB array at the provided URI.

        If ``apend=True``, a new TileDB array will be created at the URI. Otherwise,
        the data will be written to an existing array.

        Parameters
        ----------
        uri
            Uniform resource identifier for the array.
        key
            If not ``None``, encryption key to decrypt arrays.
        timestamp
            If not ``None``, the timestamp to write new data at.
        append
            If ``True``, write data to an existing array. Otherwise, create a new array.
        skip_metadata
            If ``True``, do not write metadata.
        writer_indices
            If not ``None``, an iterable list of fragment writers to write from.
        """
        if not append:
            self.create(uri, key=key, ctx=ctx)
        with tiledb.open(uri, key=key, ctx=ctx, timestamp=timestamp, mode="w") as array:
            if writer_indices is None:
                for frag_writer in self._core.fragment_writers():
                    frag_writer.write(array, skip_metadata=skip_metadata)
            else:
                for index in writer_indices:
                    frag_writer = self._core.get_fragment_writer(index)
                    frag_writer.write(array, skip_metadata=skip_metadata)

    @property
    def domain_creator(self) -> DomainCreator:
        """Domain creator that creates the domain for the TileDB array."""
        return self._domain_creator

    def has_attr_creator(self, name: str) -> bool:
        """Returns if an attribute creator with the requested name is in the array
        creator

        Parameters
        ----------
        name
            The name of the attribute creator to check for.

        Returns
        -------
        bool
            If an attribute creator with the requested name is in the array creator.
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

        Parameters
        ----------
        attr_name
            Name of the attribute to remove.
        """
        return self._core.deregister_attr_creator(attr_name)

    @property
    def sparse(self) -> bool:
        """If the array creator is sparse."""
        return self._core.sparse

    @sparse.setter
    def sparse(self, is_sparse: bool):
        self._core.sparse = is_sparse

    def to_schema(self, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
        """Returns an array schema for the array.

        Parameters
        ----------
        ctx
            If not ``None``, TileDB context wrapper for a TileDB storage manager.


        Returns
        -------
        tiledb.ArraySchema
            An array schema for the array from the array creator properties.
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
    def __init__(
        self, sparse: bool, dim_registry: Registry[SharedDim], array_dims: Sequence[str]
    ):
        self._sparse = sparse
        self._dim_registry = dim_registry
        self._dim_creators = tuple(
            self._new_dim_creator(dim_name) for dim_name in array_dims
        )
        self._attr_creators: Dict[str, AttrCreator] = OrderedDict()
        self._fragment_writers: List[FragmentWriter] = list()

    def _new_dim_creator(self, dim_name: str, **kwargs):
        return DimCreator(
            self._dim_registry[dim_name], registry=DomainDimRegistry(self), **kwargs
        )

    def add_dense_fragment_writer(
        self,
        target_region: Optional[Tuple[Tuple[int, int], ...]],
    ):
        """Add a writer for dense fragments.

        Parameters
        ----------
        target_region
            Region the fragments are written on. If ``None``, the region is
            set to the entire domain of the array.
        """
        self._fragment_writers.append(
            FragmentWriter.create_dense(
                dims=tuple(dim.base for dim in self._dim_creators),
                attr_names=self._attr_creators.keys(),
                target_region=target_region,
            )
        )

    def add_sparse_coo_fragment_writer(self, size: int):
        """Adds a spare writer for a COO fragment.

        For "coo" form, the size is used to define the footprint of the data. This
        supports a general sparse writes. The full expanded data for each dimension
        must be provided.

        Example input data for "coo" form on a 2D array:
           dim1 = [1, 2, 1, 2]
           dim2 = [3, 3, 4, 4]
           attr = [1, 2, 3, 4]

        Parameters
        ----------
        size
            The number of elements the fragment stores.
        """
        self._fragment_writers.append(
            FragmentWriter.create_sparse_coo(
                dims=tuple(dim.base for dim in self._dim_creators),
                attr_names=self._attr_creators.keys(),
                size=size,
            )
        )

    def add_sparse_row_major_fragment_writer(self, shape: Tuple[int, ...]):
        """Adds a sparse writer for a row-major grament.

        For "row-major" form, a grid of data is provided. The data on each dimension
        is just the dimension for that part of the grid:

        Example input data for "row-major" form on a 2D array:
            dim1 = [1, 2]
            dim2 = [3, 4]
            attr = [1, 2, 3, 4]

        Parameters
        ----------
        shape
            The shape of the fragment.
        """
        self._fragment_writers.append(
            FragmentWriter.create_sparse_row_major(
                dims=tuple(dim.base for dim in self._dim_creators),
                attr_names=self._attr_creators.keys(),
                shape=shape,
            )
        )

    def attr_creators(self):
        """Iterates over attribute creators in the array creator."""
        return iter(self._attr_creators.values())

    def check_new_attr_name(self, attr_name):
        """Raises an error if the provided name is not a valid new attribute name.

        Parameters
        ----------
        attr_name
            The attribute name to check.
        """
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
        """Removes an attribute from the group.

        Parameters
        ----------
        attr_name
            The name of the attribute creator to remove.
        """
        del self._attr_creators[attr_name]
        for frag_writer in self._fragment_writers:
            frag_writer.remove_attr(attr_name)

    def dim_creators(self):
        """Iterates over dimension creators in the array creator."""
        return iter(self._dim_creators)

    def fragment_writers(self):
        """Iterates over fragmetn writers in the array creator."""
        return iter(self._fragment_writers)

    def get_attr_creator(self, key: Union[str, int]) -> AttrCreator:
        """Returns the requested attribute creator.

        Parameters
        ----------
        attr_name
            Name of the attribute creator to return.
        """
        if isinstance(key, int):
            return tuple(self._attr_creators.values())[key]
        return self._attr_creators[key]

    def get_dim_creator(self, key: Union[int, str]) -> DimCreator:
        """Returns the requested dimension creator.

        Parameters
        ----------
        key
            Name (string) or index (integer) of the dimension to return.

        Returns
        -------
        DimCreator
            The requested dimension creator.
        """
        if isinstance(key, int):
            return self._dim_creators[key]
        index = self.get_dim_position_by_name(key)
        return self._dim_creators[index]

    def get_dim_position_by_name(self, dim_name: str) -> int:
        """Returns the dimension position of the requested dimension name.

        Parameters
        ----------
        dim_name
            Name of the dimension to get the position of.

        Returns
        -------
        int
            The position of the requested dimension in the array domain.
        """
        for index, dim_creator in enumerate(self._dim_creators):
            if dim_creator.name == dim_name:
                return index
        raise KeyError(f"Dimension creator with name '{dim_name}' not found.")

    def get_fragment_writer(self, index: int) -> FragmentWriter:
        """Returns the fragment writer at the requested index.

        Parameters
        ----------
        index
            The index of the fragment writer to return.

        Returns
        -------
        FragmentWriter
            The requested fragment writer.
        """
        return self._fragment_writers[index]

    def has_attr_creator(self, name: str) -> bool:
        """Returns if an attribute creator with the requested name is in the array
        creator.

        Parameters
        ----------
        name
            The name of the attribute creator to check for.

        Returns
        -------
        bool
            If an attribute creator with the requested name is in the array creator.
        """
        return name in self._attr_creators

    def inject_dim_creator(self, dim_name: str, position: int, **dim_kwargs):
        """Add an additional dimension into the domain of the array.

        Parameters
        ----------
        dim_name
            Name of the dimension creator that will be added.
        position
            Index the dimension creator will be added at.
        **dim_kwargs: dict, optional
            Keyword arguments to pass to the dimension creator.
        """
        if len(self._fragment_writers) > 0:
            raise NotImplementedError(
                "Injecting a dimension on an array that already has fragment writers "
                "is not implemented."
            )
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
        """Number of attribute creators."""
        return len(self._attr_creators)

    @property
    def ndim(self) -> int:
        """Number of dimension creators."""
        return len(self._dim_creators)

    @property
    def nwriter(self) -> int:
        """Number of writers."""
        return len(self._fragment_writers)

    def register_attr_creator(self, attr_creator):
        """Register an attribute creator to this array creator."""
        self.check_new_attr_name(attr_creator.name)
        attr_name = attr_creator.name
        self._attr_creators[attr_name] = attr_creator
        for frag_writer in self._fragment_writers:
            frag_writer.add_attr(attr_name)

    def remove_dim_creator(self, dim_index: int):
        """Remove a dim creator from the array.

        Parameters
        ----------
        dim_creator
            The location of the dimension creator to remove.
        """
        if len(self._fragment_writers) > 0:
            raise NotImplementedError(
                "Removing a dimension on an array that already has fragment writers "
                "is not implemented."
            )
        index = dim_index + self.ndim if dim_index < 0 else dim_index
        if index < 0 or index >= self.ndim:
            raise IndexError(
                f"Dimension index {dim_index} is outside the bounds of the domain."
            )
        self._dim_creators = (
            self._dim_creators[:dim_index] + self._dim_creators[dim_index + 1 :]
        )

    def set_writer_attr_data(
        self, writer_index: Optional[int], attr_name: str, data: FieldData
    ):
        """Sets attribute data on a fragment writer.

        Parameters
        ----------
        writer_index
            Index of the fragment writer to add attribute data to.
        attr_name
            The name of the attribute creator to add data for.
        data
            The data that is being added.
        """
        if writer_index is None:
            if self.nwriter > 1:
                raise ValueError(
                    "Must specify `writer_index` for array with multiple writers."
                )
            writer_index = 0
        self._fragment_writers[writer_index].set_attr_data(attr_name, data)

    def set_writer_dim_data(
        self, writer_index: Optional[int], dim_name: str, data: FieldData
    ):
        """Sets dimension data on a fragment writer.

        Parameters
        ----------
        writer_index
            The index of the fragment writer to add dimension data to.
        dim_name
            The name of the dimension creator to add data for.
        data
            The data that is being added.
        """
        if writer_index is None:
            if self.nwriter > 1:
                raise ValueError(
                    "Must specify `writer_index` for array with multiple writers."
                )
            writer_index = 0
        self._fragment_writers[writer_index].set_dim_data(dim_name, data)

    @property
    def sparse(self) -> bool:
        """If the array creator is sparse."""
        return self._sparse

    @sparse.setter
    def sparse(self, is_sparse: bool):
        if is_sparse is self._sparse:
            # No-op
            return
        if is_sparse is False and any(
            not writer.is_dense_region for writer in self._fragment_writers
        ):
            raise ValueError(
                "Cannot convert an array with a sparse fragment writer to a dense "
                "array."
            )
        self._sparse = is_sparse

    def update_attr_creator_name(self, original_name: str, new_name: str):
        """Renames an attribute in the array.

        Parameters
        ----------
        original_name
            Current name of the attribute to be renamed.
        new_name
            New name the attribute will be renamed to.
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

    def set_writer_data(
        self, writer_index: Optional[int], attr_name: str, data: FieldData
    ):
        self._core.set_writer_attr_data(writer_index, attr_name, data)

    def rename(self, old_name: str, new_name: str):
        self._core.check_new_attr_name(new_name)
        self._core.update_attr_creator_name(old_name, new_name)


class DomainDimRegistry:
    def __init__(self, array_core: ArrayCreatorCore):
        self._core = array_core

    def set_writer_data(
        self, writer_index: Optional[int], dim_name: str, data: FieldData
    ):
        self._core.set_writer_dim_data(writer_index, dim_name, data)


class DomainCreator:
    """Creator for a TileDB domain."""

    def __init__(self, array_core: ArrayCreatorCore):
        self._core = array_core
        self._dim_registry = DomainDimRegistry(self._core)

    def __getitem__(self, key: Union[int, str]):
        return self._core.get_dim_creator(key)

    def __iter__(self):
        return self._core.dim_creators()

    def __len__(self):
        return self.ndim

    def inject_dim_creator(self, dim_name: str, position: int, **dim_kwargs):
        """Adds a new dimension creator at a specified location.

        Parameters
        ----------
        dim_name
            Name of the shared dimension to add to the array's domain.
        position
            Position of the shared dimension. Negative values count backwards
            from the end of the new number of dimensions.
        dim_kwargs: dict, optional
            Keyword arguments to pass to ``DimCreator``.
        """
        self._core.inject_dim_creator(dim_name, position, **dim_kwargs)

    @property
    def ndim(self):
        """Number of dimensions in the domain."""
        return self._core.ndim

    def dim_creator(self, dim_id) -> DimCreator:
        """Returns a dimension creator from the domain creator given the dimension's
        index or name.

        Parameters
        ----------
        dim_id
            dimension index (int) or name (str)

        Returns
        -------
        DimCreator
            The dimension creator with the requested key.
        """
        return self._core.get_dim_creator(dim_id)

    def remove_dim_creator(self, dim_id: Union[str, int]):
        """Removes a dimension creator from the array creator.

        Parameters
        ----------
        dim_id
            dimension index (int) or name (str)
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
        """Returns a TileDB domain from the contained dimension creators.

        Parameters
        ----------
        ctx
            If not ``None``, the context to use when creating the domain.
        """
        if self.ndim == 0:
            raise ValueError("Cannot create schema for array with no dimensions.")
        tiledb_dims = [dim_creator.to_tiledb() for dim_creator in self]
        return tiledb.Domain(tiledb_dims, ctx=ctx)
