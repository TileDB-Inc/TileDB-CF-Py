# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Classes for creating a dataspace."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from io import StringIO
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

import tiledb

from .core import METADATA_ARRAY_NAME, Group, GroupSchema

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

    Examples:
       The :class:`DataspaceCreator` can be used to create a group of TileDB arrays
       that follow the conventions of a CF dataspace. The dataspace is created by adding
       dimensions, arrays, and attributes to the dataspace creator, then using the
       :meth:`DataspaceCreator.create` routine. For example::

           creator = DataspaceCreator()
           creator.add_dim("row", (0, 4), np.uint32)
           creator.add_dim("col", (0, 7), np.uint32)
           creator.add_array("row_vectors", ("row",))
           creator.add_array("matrices", ("row", "col"))
           creator.add_array("column_vectors", ("col",))
           creator.add_attr("x1", "row_vectors", np.dtype("float64"))
           creator.add_attr("x2", "row_vectors", np.dtype("float64"))
           creator.add_attr("A1", "matrices", np.dtype("float64"))
           creator.add_attr("A2", "matrices", np.dtype("float64"))
           creator.add_attr("y1", "column_vectors", np.dtype("float64"))
           creator.create("example_group")
    """

    def __init__(self):
        """Constructs a :class:`DataspaceCreator`."""
        self._dims: MutableMapping[str, SharedDim] = {}
        self._array_creators: Dict[str, ArrayCreator] = {}
        self._dim_to_arrays: Dict[str, List[str]] = defaultdict(list)
        self._attr_to_array: Dict[str, str] = {}
        self._attr_dataspace_names: Dict[str, str] = {}
        self._data_dim_dataspace_names: Dict[str, str] = {}
        self._index_dim_dataspace_names: Dict[str, str] = {}

    def __repr__(self):
        output = StringIO()
        if self._dims or self._array_creators:
            output.write("DataspaceCreator(\n")
            output.write(" Shared Dimensions:\n")
            for dim_name, dim in self._dims.items():
                output.write(f"  '{dim_name}':  {repr(dim)},\n")
            output.write("\n")
            output.write(" Array Creators:\n")
            for array_name, array_creator in self._array_creators.items():
                output.write(f"  '{array_name}':{repr(array_creator)}\n")
            output.write(")")
        else:
            output.write("DataspaceCreator()")
        return output.getvalue()

    def _check_new_array_name(self, array_name: str):
        if array_name in self._array_creators:
            raise ValueError(f"An array with name '{array_name}' already exists.")
        if array_name == METADATA_ARRAY_NAME:
            raise ValueError(f"The array name '{METADATA_ARRAY_NAME}' is reserved.")

    def _check_new_attr_name(self, attr_name: str):
        if attr_name in self._attr_to_array:
            raise ValueError(f"An attribute with name '{attr_name}' already exists.")
        ds_name = dataspace_name(attr_name)
        if ds_name in self._attr_dataspace_names:
            raise ValueError(
                f"An attribute named '{self._attr_dataspace_names[ds_name]}' with "
                f"dataspace name '{ds_name}' already exists."
            )
        if ds_name in self._data_dim_dataspace_names:
            raise NotImplementedError(
                f"A data dimension with dataspace name '{ds_name}' already exists. "
                f"Support for data dimension and attributes with the same name is "
                f"not yet implemented."
            )

    def _check_new_dim_name(self, dim: SharedDim):
        if dim.name in self._dims and dim != self._dims[dim.name]:
            raise ValueError(
                f"A different dimension with name '{dim.name}' already exists."
            )
        ds_name = dataspace_name(dim.name)
        if dim.is_data_dim:
            if (
                ds_name in self._data_dim_dataspace_names
                and dim.name != self._data_dim_dataspace_names.get(ds_name)
            ):
                raise ValueError(
                    f"A data dimension named '{self._data_dim_dataspace_names[ds_name]}"
                    f"' with dataspace name '{ds_name}' already exists."
                )
            if ds_name in self._attr_dataspace_names:
                raise NotImplementedError(
                    f"An attribute with dataspace name '{ds_name}' already exists. "
                    f"Support for data dimension and attributes with the same name is "
                    f"not yet implemented."
                )
        else:
            if (
                ds_name in self._index_dim_dataspace_names
                and dim.name != self._index_dim_dataspace_names.get(ds_name)
            ):
                raise ValueError(
                    f"An index dimension named "
                    f"'{self._index_dim_dataspace_names[ds_name]}' with dataspace name "
                    f"'{ds_name}' already exists."
                )

    def add_array(
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
        allows_duplicates: bool = False,
        sparse: bool = False,
    ):
        """Adds a new array to the CF dataspace.

        The name of each array must be unique. All properties must match the normal
        requirements for a ``TileDB.ArraySchema``.

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
            dim_filters: A dict from dimension name to a ``FilterList`` for dimensions
                in the array. Overrides the values set in ``coords_filters``.
            offsets_filters: Filters for the offsets for variable length attributes or
                dimensions.
            allows_duplicates: Specifies if multiple values can be stored at the same
                 coordinate. Only allowed for sparse arrays.
            sparse: Specifies if the array is a sparse TileDB array (true) or dense
                TileDB array (false).

        Raises:
            ValueError: Cannot add new array with given name.
        """
        try:
            self._check_new_array_name(array_name)
        except ValueError as err:
            raise ValueError(
                f"Cannot add new array with name '{array_name}'. {str(err)}"
            ) from err
        if not dims:
            dims = ("__scalars",)
            self.add_dim("__scalars", (0, 0), np.dtype(np.uint32))
        array_dims = tuple(self._dims[dim_name] for dim_name in dims)
        self._array_creators[array_name] = ArrayCreator(
            array_dims,
            cell_order,
            tile_order,
            capacity,
            tiles,
            coords_filters,
            dim_filters,
            offsets_filters,
            allows_duplicates,
            sparse,
        )
        for dim_name in dims:
            self._dim_to_arrays[dim_name].append(array_name)

    def add_attr(
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

        Each attribute name must be unique. It also cannot conflict with the name of a
        dimension in the array it is being added to, and the attribute's
        'dataspace name' (name after dropping the suffix ``.data`` or ``.index``) cannot
        conflict with the dataspace name of an existing attribute.

        Parameters:
            attr_name: Name of the new attribute that will be added.
            array_name: Name of the array the attribute will be added to.
            dtype: Numpy dtype of the new attribute.
            fill: Fill value for unset cells.
            var: Specifies if the attribute is variable length (automatic for
                byte/strings).
            nullable: Specifies if the attribute is nullable using validity tiles.
            filters: Specifies compression filters for the attribute.

        Raises:
            KeyError: The provided ``array_name`` does not correspond to an array in the
                dataspace.
            ValueError: Cannot create a new attribute with the provided ``attr_name``.
        """
        try:
            array_creator = self._array_creators[array_name]
        except KeyError as err:
            raise KeyError(
                f"Cannot add attribute to array '{array_name}'. No array named "
                f"'{array_name}' exists."
            ) from err
        try:
            self._check_new_attr_name(attr_name)
        except ValueError as err:
            raise ValueError(
                f"Cannot add new attribute '{attr_name}'. {str(err)}"
            ) from err
        attr_creator = AttrCreator(
            attr_name,
            np.dtype(dtype),
            fill,
            var,
            nullable,
            filters,
        )
        array_creator.add_attr(attr_creator)
        self._attr_to_array[attr_name] = array_name
        self._attr_dataspace_names[dataspace_name(attr_name)] = attr_name

    def add_dim(self, dim_name: str, domain: Tuple[Any, Any], dtype: np.dtype):
        """Adds a new dimension to the CF dataspace.

        Each dimension name must be unique. Adding a dimension where the name, domain,
        and dtype matches a current dimension does nothing.

        Parameters:
            dim_name: Name of the new dimension to be created.
            domain: The (inclusive) interval on which the dimension is valid.
            dtype: The numpy dtype of the values and domain of the dimension.

        Raises:
            ValueError: Cannot create a new dimension with the provided ``dim_name``.
        """
        shared_dim: SharedDim = SharedDim(dim_name, domain, np.dtype(dtype))
        try:
            self._check_new_dim_name(shared_dim)
        except ValueError as err:
            raise ValueError(
                f"Cannot add new dimension '{dim_name}'. {str(err)}"
            ) from err
        self._dims[dim_name] = shared_dim
        if shared_dim.is_data_dim:
            self._data_dim_dataspace_names[dataspace_name(dim_name)] = dim_name
        else:
            self._index_dim_dataspace_names[dataspace_name(dim_name)] = dim_name

    @property
    def array_names(self):
        """A view of the names of arrays in the CF dataspace."""
        return self._array_creators.keys()

    @property
    def attr_names(self):
        """A view of the names of attributes in the CF dataspace."""
        return self._attr_to_array.keys()

    def create_group(
        self,
        uri: str,
        key: Optional[Union[Dict[str, str], str]] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Creates a TileDB group and arrays for the CF dataspace.

        Parameters:
            uri: Uniform resource identifier for the TileDB group to be created, or
                prefix URIs for the TileDB arrays that will be created if
                ``use_virtual_groups=True``.
            key: If not ``None``, encryption key, or dictionary of encryption keys, to
                decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        Group.create(uri, self.to_schema(ctx), key, ctx)

    def create_virtual_group(
        self,
        uri: str,
        key: Optional[Union[Dict[str, str], str]] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Creates TileDB arrays for the CF dataspace.

        Parameters:
            uri: Uniform resource identifier for the TileDB group to be created, or
                prefix URIs for the TileDB arrays that will be created if
                ``use_virtual_groups=True``.
            key: If not ``None``, encryption key, or dictionary of encryption keys, to
                decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        Group.create_virtual(uri, self.to_schema(ctx), key, ctx)

    @property
    def dim_names(self):
        """A view of the names of dimensions in the CF dataspace."""
        return self._dims.keys()

    def remove_array(self, array_name: str):
        """Removes the specified array and all its attributes from the CF dataspace.

        Parameters:
            array_name: Name of the array that will be removed.
        """
        array = self._array_creators[array_name]
        for attr_name in array.attr_names:
            del self._attr_dataspace_names[dataspace_name(attr_name)]
            del self._attr_to_array[attr_name]
        for dim_name in array.dim_names:
            self._dim_to_arrays[dim_name].remove(array_name)
        del self._array_creators[array_name]

    def remove_attr(self, attr_name: str):
        """Removes the specified attribute from the CF dataspace.

        Parameters:
            attr_name: Name of the attribute that will be removed.
        """
        try:
            array_name = self._attr_to_array[attr_name]
        except KeyError as err:
            raise KeyError(
                f"Cannot remove attribute '{attr_name}'. No attribute with name "
                f"'{attr_name}' exists."
            ) from err
        array_creator = self._array_creators[array_name]
        array_creator.remove_attr(attr_name)
        del self._attr_dataspace_names[dataspace_name(attr_name)]
        del self._attr_to_array[attr_name]

    def remove_dim(self, dim_name: str):
        """Removes the specified dimension from the CF dataspace.

        Parameters:
            dim_name: Name of the dimension to be removed.

        Raises:
            ValueError: Dimension is being using in array(s).
        """
        array_list = self._dim_to_arrays.get(dim_name)
        if array_list:
            raise ValueError(
                f"Cannot remove dimension '{dim_name}'. Dimension is being used in "
                f"arrays: {array_list}."
            )
        dim = self._dims.pop(dim_name)
        if dim.is_data_dim:
            del self._data_dim_dataspace_names[dataspace_name(dim.name)]
        else:
            del self._index_dim_dataspace_names[dataspace_name(dim.name)]

    def rename_array(self, original_name: str, new_name: str):
        """Renames an array in the CF dataspace.

        Parameters:
            original_name: Current name of the array to be renamed.
            new_name: New name the array will be renamed to.

        Raises:
            ValueError: Cannot rename an array to the provided name ``new_name``.
        """
        try:
            self._check_new_array_name(new_name)
        except ValueError as err:
            raise ValueError(
                f"Cannot rename array '{original_name}' to '{new_name}'. {str(err)}"
            ) from err
        self._array_creators[new_name] = self._array_creators.pop(original_name)
        for attr_name in self._array_creators[new_name].attr_names:
            self._attr_to_array[attr_name] = new_name
        for dim_name in self._array_creators[new_name].dim_names:
            self._dim_to_arrays[dim_name].remove(original_name)
            self._dim_to_arrays[dim_name].append(new_name)

    def rename_attr(self, original_name: str, new_name: str):
        """Renames an attribute in the CF dataspace.

        Parameters:
            original_name: Current name of the attribute to be renamed.
            new_name: New name the attribute will be renamed to.

        Raises:
            ValueError: Cannot rename the attribute to provided name ``attr_name``.
        """
        try:
            self._check_new_attr_name(new_name)
        except ValueError as err:
            raise ValueError(
                f"Cannot rename attribute '{original_name}' to '{new_name}'. {str(err)}"
            ) from err
        array = self._array_creators[self._attr_to_array[original_name]]
        array.rename_attr(original_name, new_name)
        self._attr_to_array[new_name] = self._attr_to_array.pop(original_name)
        del self._attr_dataspace_names[dataspace_name(original_name)]
        self._attr_dataspace_names[dataspace_name(new_name)] = new_name

    def rename_dim(self, original_name: str, new_name: str):
        """Renames a dimension in the CF dataspace.

        Parameters:
            original_name: Current name of the dimension to be renamed.
            new_name: New name the dimension will be renamed to.

        Raises:
            ValueError: Cannot rename the dimension to the provided name ``new_name``.
        """
        try:
            dim = self._dims[original_name]
            self._check_new_dim_name(SharedDim(new_name, dim.domain, dim.dtype))
        except ValueError as err:
            raise ValueError(
                f"Cannot rename dimension '{original_name}' to '{new_name}'. {str(err)}"
            ) from err
        if new_name in self._dims:
            raise NotImplementedError(
                f"Cannot rename dimension '{original_name}' to '{new_name}'. A "
                f"dimension with the same name already exists, and merging dimensions "
                f"has not yet been implemented."
            )
        if self._attr_to_array.get(new_name) in self._dim_to_arrays[original_name]:
            raise ValueError(
                f"Cannot rename dimension '{original_name}' to '{new_name}'. An "
                f"attribute with the same name already exists in the array "
                f"'{self._attr_to_array[new_name]}' that uses this dimension."
            )
        self._dims[new_name] = self._dims.pop(original_name)
        self._dims[new_name].name = new_name
        if dim.is_data_dim:
            del self._data_dim_dataspace_names[dataspace_name(original_name)]
            self._data_dim_dataspace_names[dataspace_name(new_name)] = new_name
        else:
            del self._index_dim_dataspace_names[dataspace_name(original_name)]
            self._index_dim_dataspace_names[dataspace_name(new_name)] = new_name

    def set_array_properties(self, array_name: str, **properties):
        """Sets properties for an array in the CF dataspace.

        Valid properties are:

            * ``cell_order``: The order in which TileDB stores the cells on disk inside
              a tile. Valid values are: ``row-major`` (default) or ``C`` for row
              major; ``col-major`` or ``F`` for column major; or ``Hilbert`` for a
              Hilbert curve.
            * ``tile_order``: The order in which TileDB stores the tiles on disk. Valid
              values are: ``row-major`` or ``C`` (default) for row major; or
              ``col-major`` or ``F`` for column major.
            * ``capacity``: The number of cells in a data tile of a sparse fragment.
            * ``tiles``: An optional ordered list of tile sizes for the dimensions of
              the array. The length must match the number of dimensions in the array.
            * ``coords_filters``: Filters for all dimensions that do not otherwise have
              a specified filter list.
            * ``dim_filters``: A dict from dimension name to a ``FilterList`` for
              dimensions in the array. Overrides the values set in ``coords_filters``.
            * ``offsets_filters``: Filters for the offsets for variable length
              attributes or dimensions.
            * ``allows_duplicates``: Specifies if multiple values can be stored at the
              same coordinate. Only allowed for sparse arrays.
            * ``sparse``: Specifies if the array is a sparse TileDB array (true) or
              dense TileDB array (false).

        Parameters:
            array_name: Name of the array to set properties for.
            properties: Keyword arguments for array properties.
        """
        array_creator = self._array_creators[array_name]
        for property_name, value in properties.items():
            setattr(array_creator, property_name, value)

    def set_attr_properties(self, attr_name: str, **properties):
        """Sets properties for an attribute in the CF dataspace.

        Valid properties are:
            * ``name``: The name of the attribute.
            * ``dtype``: Numpy dtype of the attribute.
            * ``fill``: Fill value for unset cells.
            * ``var``: Specifies if the attribute is variable length (automatic for
              bytes/strings).
            * ``nullable``: Specifies if the attribute is nullable using validity tiles.
            * ``filters``: Specifies compression filters for the attributes.

        Parameters:
            attr_name: Name of the attribute to set properties for.
            properties: Keyword arguments for attribute properties.
        """
        if "name" in properties:
            old_name = attr_name
            attr_name = properties.pop("name")
            self.rename_attr(old_name, attr_name)
        try:
            array_name = self._attr_to_array[attr_name]
        except KeyError as err:
            raise KeyError(
                f"Attribute with name '{attr_name}' does not exist."
            ) from err
        self._array_creators[array_name].set_attr_properties(
            attr_name,
            **properties,
        )

    def to_schema(self, ctx: Optional[tiledb.Ctx] = None) -> GroupSchema:
        """Returns a group schema for the CF dataspace.

        Parameters:
           ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        array_schemas = {
            array_name: array_creator.to_schema(ctx)
            for array_name, array_creator in self._array_creators.items()
        }
        group_schema = GroupSchema(array_schemas)
        group_schema.set_default_metadata_schema(ctx)
        return group_schema


class ArrayCreator:
    """Creator for a TileDB array using shared dimension definitions.

    Parameters:
        dims: An ordered list of the shared dimensions for the domain of this array.
        cell_order: The order in which TileDB stores the cells on disk inside a
            tile. Valid values are: ``row-major`` (default) or ``C`` for row major;
            ``col-major`` or ``F`` for column major; or ``Hilbert`` for a Hilbert curve.
        tile_order: The order in which TileDB stores the tiles on disk. Valid values
            are: ``row-major`` or ``C`` (default) for row major; or ``col-major`` or
            ``F`` for column major.
        capacity: The number of cells in a data tile of a sparse fragment.
        tiles: An optional ordered list of tile sizes for the dimensions of the
            array. The length must match the number of dimensions in the array.
        coords_filters: Filters for all dimensions that are not specified explicitly by
            ``dim_filters``.
        dim_filters: A dict from dimension name to a ``FilterList`` for dimensions in
            the array. Overrides the values set in ``coords_filters``.
        offsets_filters: Filters for the offsets for variable length attributes or
            dimensions.
        allows_duplicates: Specifies if multiple values can be stored at the same
             coordinate. Only allowed for sparse arrays.
        sparse: Specifies if the array is a sparse TileDB array (true) or dense
            TileDB array (false).

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

    def __init__(
        self,
        dims: Sequence[SharedDim],
        cell_order: str = "row-major",
        tile_order: str = "row-major",
        capacity: int = 0,
        tiles: Optional[Sequence[int]] = None,
        coords_filters: Optional[tiledb.FilterList] = None,
        dim_filters: Optional[Dict[str, tiledb.FilterList]] = None,
        offsets_filters: Optional[tiledb.FilterList] = None,
        allows_duplicates: bool = False,
        sparse: bool = False,
    ):
        """Constructor for a ArrayCreator object."""
        self._dim_creators = tuple(DimCreator(dim) for dim in dims)
        if not self._dim_creators:
            raise ValueError(
                "Cannot create array. Array must have at lease one dimension."
            )
        self._attr_creators: Dict[str, AttrCreator] = OrderedDict()
        self.cell_order = cell_order
        self.tile_order = tile_order
        self.capacity = capacity
        if tiles is not None:
            self.tiles = tiles
        self.coords_filters = coords_filters
        if dim_filters is not None:
            self.dim_filters = dim_filters
        self.offsets_filters = offsets_filters
        self.allows_duplicates = allows_duplicates
        self.sparse = sparse

    def __repr__(self) -> str:
        output = StringIO()
        output.write("  ArrayCreator(\n")
        output.write("     domain=Domain(*[\n")
        for dim_creator in self._dim_creators:
            output.write(f"       {repr(dim_creator)},\n")
        output.write("     ]),\n")
        output.write("     attrs=[\n")
        for attr_creator in self._attr_creators.values():
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

    def add_attr(self, attr_creator: AttrCreator):
        """Adds a new attribute to an array in the CF dataspace.

        Each attribute name must be unique. It also cannot conflict with the name of a
        dimension in the array.

        Parameters:
            attr_name: Name of the new attribute that will be added.
            dtype: Numpy dtype of the new attribute.
            fill: Fill value for unset cells.
            var: Specifies if the attribute is variable length (automatic for
                byte/strings).
            nullable: Specifies if the attribute is nullable using validity tiles.
            filters: Specifies compression filters for the attribute.
        """
        attr_name = attr_creator.name
        if attr_name in self._attr_creators:
            raise ValueError(
                f"Cannot create new attribute with name '{attr_name}'. An attribute "
                f"with that name already exists in this array."
            )
        if attr_name in self.dim_names:
            raise ValueError(
                f"Cannot create new attribute with name '{attr_name}'. A dimension with"
                f" that name already exists in this array."
            )
        self._attr_creators[attr_name] = attr_creator

    @property
    def attr_names(self):
        """A view of the names of attributes in the array."""
        return self._attr_creators.keys()

    def create(
        self,
        uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Creates a TileDB array at the provided URI.

        Parameters:
            uri: Uniform resource identifier for the array to be created.
            key: If not ``None``, encryption key to decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """

        tiledb.Array.create(uri, self.to_schema(ctx), key, ctx)

    @property
    def dim_filters(self) -> Mapping[str, Optional[tiledb.FilterList]]:
        """A dict from dimension name to a ``FilterList`` for dimensions in the array.
        Overrides the values set in ``coords_filters``.
        """
        return {
            dim_creator.name: dim_creator.filters for dim_creator in self._dim_creators
        }

    @dim_filters.setter
    def dim_filters(
        self,
        dim_filters: Mapping[str, Optional[tiledb.FilterList]],
    ):
        dim_map = {dim_creator.name: dim_creator for dim_creator in self._dim_creators}
        for dim_name, filters in dim_filters.items():
            dim_map[dim_name].filters = filters

    @property
    def dim_names(self) -> Tuple[str, ...]:
        """A static snapshot of the names of dimensions of the array."""
        return tuple(dim_creator.name for dim_creator in self._dim_creators)

    @property
    def ndim(self) -> int:
        """Number of dimensions in the array."""
        return len(self._dim_creators)

    def rename_attr(self, original_name: str, new_name: str):
        """Renames an attribute in the array.

        Parameters:
            original_name: Current name of the attribute to be renamed.
            new_name: New name the attribute will be renamed to.
        """
        if new_name in self.dim_names:
            raise ValueError(
                f"Cannot rename attr '{original_name}' to '{new_name}'. A dimension "
                f"with that name already exists in this array."
            )
        attr = self._attr_creators.pop(original_name)
        attr.name = new_name
        self._attr_creators[new_name] = attr

    def remove_attr(self, attr_name: str):
        """Removes the specified attribute from the array.

        Parameters:
            attr_name: Name of the attribute that will be removed.
        """
        del self._attr_creators[attr_name]

    def set_attr_properties(self, attr_name: str, **properties):
        """Sets properties for an attribute in the array.

        Valid properties are:
            * ``name``: The name of the attribute.
            * ``dtype``: Numpy dtype of the attribute.
            * ``fill``: Fill value for unset cells.
            * ``var``: Specifies if the attribute is variable length (automatic for
              bytes/strings).
            * ``nullable``: Specifies if the attribute is nullable using validity tiles.
            * ``filters``: Specifies compression filters for the attributes.

        Parameters:
            attr_name: Name of the attribute to set properties for.
            properties: Keyword arguments for attribute properties.
        """
        if "name" in properties:
            old_name = attr_name
            attr_name = properties.pop("name")
            self.rename_attr(old_name, attr_name)
        attr_creator = self._attr_creators[attr_name]
        for property_name, value in properties.items():
            setattr(attr_creator, property_name, value)

    @property
    def sparse(self) -> bool:
        """Specifies if the array is a sparse TileDB array or dense TileDB array."""
        return self._sparse

    @sparse.setter
    def sparse(self, sparse):
        if not sparse:
            dim_dtype = np.dtype(self._dim_creators[0].dtype)
            if dim_dtype.kind not in ("i", "u", "M"):
                raise ValueError(
                    f"Cannot set sparse=False. Array contains dimension with type "
                    f"{dim_dtype} not supported in dense arrays."
                )
            for dim in self._dim_creators:
                if dim_dtype != dim.dtype:
                    raise ValueError(
                        "Cannot set sparse=False. Array contains dimensions with "
                        "different dimension types."
                    )
        self._sparse = sparse

    @property
    def tiles(self) -> Collection[Union[int, float, None]]:
        """An optional ordered list of tile sizes for the dimensions of the
        array. The length must match the number of dimensions in the array."""
        return tuple(dim_creator.tile for dim_creator in self._dim_creators)

    @tiles.setter
    def tiles(self, tiles: Collection[Union[int, float, None]]):
        if len(tiles) != self.ndim:
            raise ValueError(
                f"Cannot set tiles. Got {len(tiles)} tile(s) for an array with "
                f"{self.ndim} dimension(s)."
            )
        for dim_creator, tile in zip(self._dim_creators, tiles):
            dim_creator.tile = tile

    def to_schema(self, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
        """Returns an array schema for the array.

        Parameters:
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        assert self.ndim > 0, "Must have at least one dimension."
        if len(self._attr_creators) == 0:
            raise ValueError("Cannot create schema for array with no attributes.")
        tiledb_dims = [dim_creator.to_tiledb() for dim_creator in self._dim_creators]
        domain = tiledb.Domain(tiledb_dims, ctx=ctx)
        attrs = tuple(
            attr_creator.to_tiledb(ctx) for attr_creator in self._attr_creators.values()
        )
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


@dataclass
class AttrCreator:
    """Creator for a TileDB attribute.

    Parameters:
        name: Name of the new attribute.
        dtype: Numpy dtype of the attribute.
        fill: Fill value for unset cells.
        var: Specifies if the attribute is variable length (automatic for
            byte/strings).
        nullable: Specifies if the attribute is nullable using validity tiles.
        filters: Specifies compression filters for the attribute.

    Attributes:
        name: Name of the new attribute.
        dtype: Numpy dtype of the attribute.
        fill: Fill value for unset cells.
        var: Specifies if the attribute is variable length (automatic for
            byte/strings).
        nullable: Specifies if the attribute is nullable using validity tiles.
        filters: Specifies compression filters for the attribute.
    """

    name: str
    dtype: np.dtype
    fill: Optional[DType] = None
    var: bool = False
    nullable: bool = False
    filters: Optional[tiledb.FilterList] = None

    def __repr__(self):
        filters_str = ""
        if self.filters:
            filters_str = ", filters=FilterList(["
            for attr_filter in self.filters:
                filters_str += repr(attr_filter) + ", "
            filters_str += "])"
        return (
            f"AttrCreator(name={self.name}, dtype='{self.dtype!s}', var={self.var}, "
            f"nullable={self.nullable}{filters_str})"
        )

    def to_tiledb(self, ctx: Optional[tiledb.Ctx] = None) -> tiledb.Attr:
        """Returns a :class:`tiledb.Attr` using the current properties.

        Parameters:
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
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


@dataclass
class DimCreator:
    """Creator for a TileDB dimension using a SharedDim.

    Attributes:
        base: Shared definition for the dimensions name, domain, and dtype.
        tile: The tile size for the dimension.
        filters: Specifies compression filters for the dimension.
    """

    base: SharedDim
    tile: Optional[Union[int, float]] = None
    filters: Optional[Union[tiledb.FilterList]] = None

    def __repr__(self):
        filters_str = ""
        if self.filters:
            filters_str = ", filters=FilterList(["
            for dim_filter in self.filters:
                filters_str += repr(dim_filter) + ", "
            filters_str += "])"
        return f"DimCreator({repr(self.base)}, tile={self.tile}{filters_str})"

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the values and domain of the dimension."""
        return self.base.dtype

    @property
    def domain(self) -> Tuple[Optional[DType], Optional[DType]]:
        """The (inclusive) interval on which the dimension is valid."""
        return self.base.domain

    @property
    def name(self) -> str:
        """Name of the dimension."""
        return self.base.name

    def to_tiledb(self, ctx: Optional[tiledb.Ctx] = None) -> tiledb.Domain:
        """Returns a :class:`tiledb.Dim` using the current properties.

        Parameters:
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        return tiledb.Dim(
            name=self.name,
            domain=self.domain,
            tile=self.tile,
            filters=self.filters,
            dtype=self.dtype,
            ctx=ctx,
        )


@dataclass
class SharedDim:
    """A class for a shared one-dimensional dimension.

    Parameters:
        name: Name of the :class:`SharedDim`.
        domain: The (inclusive) interval on which the :class:`SharedDim` is valid.
        dtype: The numpy dtype of the values and domain of the dimension.
    """

    name: str
    domain: Tuple[Optional[DType], Optional[DType]]
    dtype: np.dtype

    @classmethod
    def from_tiledb_dim(cls, dim: tiledb.Dim):
        """Converts a tiledb.Dim to a :class:`SharedDim`

        Parameters:
            dim: TileDB dimension that will be used to create the shared
                dimension.

        Returns:
            A :class:`SharedDim` that has the name, domain, and dtype of the
                tiledb.Dim.
        """
        return cls(dim.name, dim.domain, dim.dtype)

    def __repr__(self) -> str:
        return (
            f"SharedDim(name={self.name}, domain={self.domain}, dtype='{self.dtype!s}')"
        )

    @property
    def is_data_dim(self) -> bool:
        """Returns if the :class:`SharedDim` is a 'data dimension'

        A data dimension is a dimension that is not an integer type or starts at a value
        other than zero. This is compared to an 'index axis' that is a simple integer
        index with default offset.
        """
        return not self.is_index_dim

    @property
    def is_index_dim(self) -> bool:
        """Returns if the :class:`SharedDim` is a 'index dimension'

        An index dimension is a dimension that is of an integer type and whose domain
        starts at 0.
        """
        return np.issubdtype(self.dtype, np.integer) and self.domain[0] == 0
