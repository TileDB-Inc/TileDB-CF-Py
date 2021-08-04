# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""Classes for creating a dataspace."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from io import StringIO
from typing import Any, Collection, Dict, Mapping, Optional, Sequence, Tuple, Union

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
                f"{array_creator.__class__.__name__} <em>{array_creator.name}</em>"
                f"({', '.join(map(str, array_creator.dim_names))})\n"
            )
            output.write("</summary>\n")
            output.write(f"{array_creator.html_summary()}\n")
            output.write("</details>\n")
        output.write("</li>\n")
        output.write("</ul>\n")
        return output.getvalue()

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
            allows_duplicates: Specifies if multiple values can be stored at the same
                 coordinate. Only allowed for sparse arrays.
            sparse: Specifies if the array is a sparse TileDB array (true) or dense
                TileDB array (false).
        """
        array_dims = tuple(self._registry.get_shared_dim(dim_name) for dim_name in dims)
        self._registry.add_array_creator(
            ArrayCreator(
                dims=array_dims,
                cell_order=cell_order,
                tile_order=tile_order,
                capacity=capacity,
                tiles=tiles,
                coords_filters=coords_filters,
                dim_filters=dim_filters,
                offsets_filters=offsets_filters,
                allows_duplicates=allows_duplicates,
                sparse=sparse,
                name=array_name,
                registry=self._registry,
            )
        )

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
        # deprecate in favor of get_array_creator[name].add_attr_creator
        try:
            array_creator = self._registry.get_array_creator(array_name)
        except KeyError as err:
            raise KeyError(
                f"Cannot add attribute to array '{array_name}'. No array named "
                f"'{array_name}' exists."
            ) from err
        array_creator.add_attr_creator(attr_name, dtype, fill, var, nullable, filters)

    def add_dim(self, dim_name: str, domain: Tuple[Any, Any], dtype: np.dtype):
        """Adds a new dimension to the CF dataspace.

        Each dimension name must be unique. Adding a dimension where the name, domain,
        and dtype matches a current dimension does nothing.

        Parameters:
            dim_name: Name of the new dimension to be created.
            domain: The (inclusive) interval on which the dimension is valid.
            dtype: The numpy dtype of the values and domain of the dimension.
        """
        self._registry.add_shared_dimension(
            SharedDim(dim_name, domain, np.dtype(dtype), self._registry)
        )

    @property
    def array_names(self):
        """A view of the names of arrays in the CF dataspace."""
        # TODO: deprecate this function
        return self._registry._array_creators.keys()

    @property
    def attr_names(self):
        """A view of the names of attributes in the CF dataspace."""
        # TODO: deprecate this function
        return self._registry._attr_to_array.keys()

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
        if self._registry.ndim != 1:
            raise ValueError(
                f"Can only use `create_array` for a {self.__class__.__name__} with "
                f"exactly 1 array creator. This {self.__class__.__name__} contains "
                f"{self._registry.ndim} array creators."
            )
        array_creator = tuple(self._registry.array_creators())[0]
        array_creator.create(uri, key=key, ctx=ctx)

    def create_group(
        self,
        uri: str,
        key: Optional[Union[Dict[str, str], str]] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Creates a TileDB group and arrays for the CF dataspace.

        Parameters:
            uri: Uniform resource identifier for the TileDB group to be created.
            key: If not ``None``, encryption key, or dictionary of encryption keys, to
                decrypt arrays.
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        schema = self.to_schema(ctx)
        Group.create(uri, schema, key, ctx)

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
        Group.create_virtual(uri, self.to_schema(ctx), key, ctx)

    @property
    def dim_names(self):
        """A view of the names of dimensions in the CF dataspace."""
        # TODO: deprecate this function
        return self._registry._shared_dims.keys()

    def get_array_property(self, array_name: str, property_name: str) -> Any:
        """Returns a requested property from an array in the CF dataspace.

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
            array_name: Name of the array to get the property from.
            property_name: Name of the requested property.
        """
        # TODO: deprecate this function
        array_creator = self._registry.get_array_creator(array_name)
        return getattr(array_creator, property_name)

    def get_attr_property(self, attr_name: str, property_name: str) -> Any:
        """Returns a requested property for an attribute in the CF dataspace.

        Valid properties are:
            * ``name``: The name of the attribute.
            * ``dtype``: Numpy dtype of the attribute.
            * ``fill``: Fill value for unset cells.
            * ``var``: Specifies if the attribute is variable length (automatic for
              bytes/strings).
            * ``nullable``: Specifies if the attribute is nullable using validity tiles.
            * ``filters``: Specifies compression filters for the attributes.

        Parameters:
            attr_name: Name of the attribute to get the property from.
            property_name: Name of the requested property.
        """
        # TODO: deprecate this function
        try:
            array_creator = self._registry.lookup_array_creator(attr_name)
        except KeyError as err:
            raise KeyError(
                f"Attribute with name '{attr_name}' does not exist."
            ) from err
        return array_creator.get_attr_property(attr_name, property_name)

    def get_dim_property(self, dim_name: str, property_name: str) -> Any:
        """Returns a requested property for a dimension in the CF dataspace.

        Valid properties are:
            * ``name``: The name of the dimension.
            * ``domain``: The (inclusive) valid range for the dimensions.
            * ``dtype``: The Numpy data type of the dimension.
        """
        # TODO: deprecate this function
        dim = self._registry.get_shared_dim(dim_name)
        return getattr(dim, property_name)

    def remove_array(self, array_name: str):
        """Removes the specified array and all its attributes from the CF dataspace.

        Parameters:
            array_name: Name of the array that will be removed.
        """
        self._registry.remove_array_creator(array_name)

    def remove_attr(self, attr_name: str):
        """Removes the specified attribute from the CF dataspace.

        Parameters:
            attr_name: Name of the attribute that will be removed.
        """
        # TODO: deprecate and replace with function directly in array_creator.
        self._registry.remove_attr_creator(attr_name)

    def remove_dim(self, dim_name: str):
        """Removes the specified dimension from the CF dataspace.

        This can only be used to remove dimensions that are not currently being used in
        an array.

        Parameters:
            dim_name: Name of the dimension to be removed.
        """
        self._registry.remove_shared_dim(dim_name)

    def rename_array(self, original_name: str, new_name: str):
        """Renames an array in the CF dataspace.

        Parameters:
            original_name: Current name of the array to be renamed.
            new_name: New name the array will be renamed to.
        """
        # TODO: deprecate and replace with direct call to array name
        self._registry.get_array_creator(original_name).name = new_name

    def rename_attr(self, original_name: str, new_name: str):
        """Renames an attribute in the CF dataspace.

        Parameters:
            original_name: Current name of the attribute to be renamed.
            new_name: New name the attribute will be renamed to.
        """
        # TODO: deprecate and replace with direct call to attribute name
        array_creator = self._registry.lookup_array_creator(attr_name=original_name)
        array_creator.rename_attr(original_name, new_name)

    def rename_dim(self, original_name: str, new_name: str):
        """Renames a dimension in the CF dataspace.

        Parameters:
            original_name: Current name of the dimension to be renamed.
            new_name: New name the dimension will be renamed to.
        """
        # TODO: deprecate and replace with direct call to dimension name
        self._registry.get_shared_dim(original_name).name = new_name

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
        # TODO: deprecate this function
        array_creator = self._registry.get_array_creator(array_name)
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
        # TODO: deprecate this function
        array_creator = self._registry.lookup_array_creator(attr_name=attr_name)
        array_creator.set_attr_properties(attr_name, **properties)

    def set_dim_properties(self, dim_name: str, **properties):
        """Sets properties for a shared dimension in the CF dataspace.

        Valid properties are:
            * ``name``: The name of the dimension.
            * ``domain``: The (inclusive) inverval on which the dimension is valid.
            * ``dtype``: The data type of the dimension.

        Parameters:
            dim_name: Name of the dimension to set properties for.
            properties: Keyword arguments for dimension properties.
        """
        # TODO: deprecate this function
        dim = self._registry.get_shared_dim(dim_name)
        for property_name, value in properties.items():
            setattr(dim, property_name, value)

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

    def add_array_creator(self, array_creator: ArrayCreator):
        """Registers a new array creator with the CF dataspace."""
        try:
            self.check_new_array_name(array_creator.name)
        except ValueError as err:
            raise ValueError(
                f"Cannot add new array with name '{array_creator.name}'. {str(err)}"
            ) from err
        for shared_dim in array_creator.shared_dims():
            self.check_new_dim(shared_dim)
        for attr_name in array_creator.attr_names:
            self.check_new_attr_name(attr_name)
        self._array_creators[array_creator.name] = array_creator
        for attr_name in array_creator.attr_names:
            self._attr_to_array[attr_name] = array_creator.name

    def add_attr_to_array(self, array_name: str, attr_name: str):
        """Registers a new attribute name to an array creator."""
        if array_name not in self._array_creators:
            raise KeyError(
                f"Cannot add attribute to array '{array_name}'. No array named "
                f"'{array_name}' exists."
            )
        try:
            self.check_new_attr_name(attr_name)
        except ValueError as err:
            raise ValueError(
                f"Cannot add new attribute '{attr_name}'. {str(err)}"
            ) from err
        self._attr_to_array[attr_name] = array_name

    def add_shared_dimension(self, shared_dim: SharedDim):
        """Registers a new shared dimension to the CF dataspace.

        Parameters:
            shared_dim: The new shared dimension to register.
        """
        try:
            self.check_new_dim(shared_dim)
        except ValueError as err:
            raise ValueError(
                f"Cannot add new dimension '{shared_dim.name}'. {str(err)}"
            ) from err
        self._shared_dims[shared_dim.name] = shared_dim
        print(f"DIMS: {self._shared_dims.keys()}")

    def array_creators(self):
        """Iterator over array creators in the CF dataspace."""
        for array_creator in self._array_creators.values():
            yield array_creator

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

    def get_array_creator(self, array_name: str) -> ArrayCreator:
        """Returns the array creator with the requested name."""
        return self._array_creators[array_name]

    def get_shared_dim(self, dim_name: str) -> SharedDim:
        """Returns the dim creator with the requested name."""
        return self._shared_dims[dim_name]

    def lookup_array_creator(self, attr_name: str) -> ArrayCreator:
        """Returns an array creator that contains the requested attribute."""
        try:
            array_name = self._attr_to_array[attr_name]
        except KeyError as err:
            raise KeyError(f"No attribute with the name '{attr_name}'.") from err
        return self._array_creators[array_name]

    @property
    def narray(self) -> int:
        return len(self._array_creators)

    @property
    def nattr(self) -> int:
        return len(self._attr_to_array)

    @property
    def ndim(self) -> int:
        return len(self._shared_dims)

    def remove_array_creator(self, array_name: str):
        """Removes the specified array and all its attributes from the CF dataspace.

        Parameters:
            array_name: Name of the array that will be removed.
        """
        array = self._array_creators[array_name]
        for attr_name in array.attr_names:
            del self._attr_to_array[attr_name]
        del self._array_creators[array_name]

    def remove_attr_creator(self, attr_name: str):
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
        del self._attr_to_array[attr_name]

    def remove_shared_dim(self, dim_name: str):
        """Removes the specified dimension from the CF dataspace.
        an array.

        Parameters:
            dim_name: Name of the dimension to be removed.
        """
        array_list = [
            array_creator.name
            for array_creator in self.array_creators()
            if dim_name in array_creator.dim_names
        ]
        if array_list:
            raise ValueError(
                f"Cannot remove dimension '{dim_name}'. Dimension is being used in "
                f"arrays: {array_list}."
            )
        del self._shared_dims[dim_name]

    def rename_array_creator(self, original_name: str, new_name: str):
        """Renames an array in the CF dataspace.

        Parameters:
            original_name: Current name of the array to be renamed.
            new_name: New name the array will be renamed to.
        """
        try:
            self.check_new_array_name(new_name)
        except ValueError as err:
            raise ValueError(
                f"Cannot rename array '{original_name}' to '{new_name}'. {str(err)}"
            ) from err
        self._array_creators[new_name] = self._array_creators.pop(original_name)
        for attr_name in self._array_creators[new_name].attr_names:
            self._attr_to_array[attr_name] = new_name

    def rename_attr_creator(self, original_name: str, new_name: str):
        """Renames an attribute in the CF dataspace.

        Parameters:
            original_name: Current name of the attribute to be renamed.
            new_name: New name the attribute will be renamed to.
        """
        try:
            self.check_new_attr_name(new_name)
        except ValueError as err:
            raise ValueError(
                f"Cannot rename attribute '{original_name}' to '{new_name}'. {str(err)}"
            ) from err
        self._attr_to_array[new_name] = self._attr_to_array.pop(original_name)

    def rename_shared_dim(self, original_name: str, new_name: str):
        """Renames a dimension in the CF dataspace.

        Parameters:
            original_name: Current name of the dimension to be renamed.
            new_name: New name the dimension will be renamed to.
        """
        try:
            dim = self._shared_dims[original_name]
            self.check_new_dim(SharedDim(new_name, dim.domain, dim.dtype))
        except ValueError as err:
            raise ValueError(
                f"Cannot rename dimension '{original_name}' to '{new_name}'. {str(err)}"
            ) from err
        if new_name in self._shared_dims:
            raise NotImplementedError(
                f"Cannot rename dimension '{original_name}' to '{new_name}'. A "
                f"dimension with the same name already exists, and merging dimensions "
                f"has not yet been implemented."
            )
        if new_name in self._attr_to_array:
            array_name = self._attr_to_array[new_name]
            if original_name in self._array_creators[array_name].dim_names:
                raise ValueError(
                    f"Cannot rename dimension '{original_name}' to '{new_name}'. An "
                    f"attribute with the same name already exists in the array "
                    f"'{array_name}' that uses this dimension."
                )
        self._shared_dims[new_name] = self._shared_dims.pop(original_name)

    def shared_dims(self):
        """Iterates over shared dimensions in the CF dataspace."""
        for shared_dim in self._shared_dims.values():
            yield shared_dim


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
        name: Name of the array. Used for registering to a group or dataspace.
        registry: Dataspace registry for the dataspace the array belongs to.

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
        name: str = "",
        registry: Optional[DataspaceRegistry] = None,
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
        self._name = name
        self._registry = registry
        self.__post_init__()

    def __post_init__(self):
        pass

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
        if self._registry is not None:
            print(f"ADDING ATTR {attr_name}")
            self._registry.add_attr_to_array(self._name, attr_name)
        self._attr_creators[attr_name] = attr_creator

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
        self.add_attr(AttrCreator(name, np.dtype(dtype), fill, var, nullable, filters))

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

    def shared_dims(self):
        """Iterators over shared dimensions in this array."""
        for dim_creator in self._dim_creators:
            yield dim_creator.base

    @property
    def dim_names(self) -> Tuple[str, ...]:
        """A static snapshot of the names of dimensions of the array."""
        return tuple(dim_creator.name for dim_creator in self._dim_creators)

    def get_attr_property(self, attr_name: str, property_name: str) -> Any:
        """Returns a requested property for an attribute in the array.

        Valid properties are:
            * ``name``: The name of the attribute.
            * ``dtype``: Numpy dtype of the attribute.
            * ``fill``: Fill value for unset cells.
            * ``var``: Specifies if the attribute is variable length (automatic for
              bytes/strings).
            * ``nullable``: Specifies if the attribute is nullable using validity tiles.
            * ``filters``: Specifies compression filters for the attributes.

        Parameters:
            attr_name: Name of the attribute to get the property from.
            property_name: Name of requested property.
        """
        attr_creator = self._attr_creators[attr_name]
        return getattr(attr_creator, property_name)

    @property
    def name(self) -> str:
        """Name of the array."""
        return self._name

    @name.setter
    def name(self, name: str):
        if self._registry is not None:
            self._registry.rename_array_creator(self._name, name)
        self._name = name

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
        if self._registry is not None:
            self._registry.rename_attr_creator(original_name, new_name)
        attr = self._attr_creators.pop(original_name)
        attr.name = new_name
        self._attr_creators[new_name] = attr

    def remove_attr(self, attr_name: str):
        """Removes the specified attribute from the array.

        Parameters:
            attr_name: Name of the attribute that will be removed.
        """
        del self._attr_creators[attr_name]

    def html_summary(self) -> str:
        """Returns a string HTML summary of the :class:`ArrayCreator`."""
        cell_style = 'style="text-align: left;"'
        output = StringIO()
        output.write("<ul>\n")
        output.write("<li>\n")
        output.write("Domain\n")
        output.write("<table>\n")
        for dim_creator in self._dim_creators:
            output.write(
                f"<tr><td {cell_style}>{dim_creator.html_summary()}</td></tr>\n"
            )
        output.write("</table>\n")
        output.write("</li>\n")
        output.write("<li>\n")
        output.write("Attributes\n")
        output.write("<table>\n")
        for attr_creator in self._attr_creators.values():
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
        sparst: Specifies if the array is a sparse TileDB array (true) or dense
            TileDB array (false).


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

    def to_schema(
        self, ctx: Optional[tiledb.Ctx] = None, key: Optional[str] = None
    ) -> tiledb.ArraySchema:
        """Returns an array schema for the array.

        Parameters:
            ctx: If not ``None``, TileDB context wrapper for a TileDB storage manager.
            key: If not ``None``, encryption key to decrypt the array.
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


class DimCreator:
    """Creator for a TileDB dimension using a SharedDim.

    Attributes:
        base: Shared definition for the dimensions name, domain, and dtype.
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


class SharedDim:
    """A class for a shared one-dimensional dimension.

    Parameters:
        name: Name of the shared dimension.
        domain: The (inclusive) interval on which the :class:`SharedDim` is valid.
        dtype: The numpy dtype of the values and domain of the dimension.
    """

    def __init__(
        self,
        name: str,
        domain: Optional[Tuple[Optional[DType], Optional[DType]]],
        dtype: np.dtype,
        registry: Optional[DataspaceRegistry] = None,
    ):
        self._name = name
        self.domain = domain
        self.dtype = dtype
        self._registry = registry

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.name == other.name
            and self.domain == other.domain
            and self.dtype == other.dtype
        )

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

    def html_input_summary(self) -> str:
        """Returns a HTML string summarizing the input for the dimension."""
        return ""

    def html_output_summary(self) -> str:
        """Returns a string HTML summary of the :class:`SharedDim`."""
        return f"name={self.name}, domain={self.domain}, dtype='{self.dtype!s}'"

    @property
    def is_index_dim(self) -> bool:
        """Returns if the :class:`SharedDim` is a 'index dimension'

        An index dimension is a dimension that is of an integer type and whose domain
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
        if self._registry is not None:
            self._registry.rename_shared_dim(self._name, name)
        self._name = name
