"""Classes for creating a dataspace."""

from __future__ import annotations

from collections.abc import MutableMapping
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import tiledb

from ._array_creator import ArrayCreator
from ._shared_dim import SharedDim
from .api import create_group

if TYPE_CHECKING:
    from ._attr_creator import AttrCreator


class DataspaceCreator:
    """Creator for a group of arrays that satify the CF Dataspace Convention.

    This class can be used directly to create a TileDB group that follows the
    TileDB CF Dataspace convention. It is also useful as a super class for
    converters/ingesters of data from sources that follow a NetCDF or NetCDF-like
    data model to TileDB.
    """

    def __init__(self):
        self._core = DataspaceCreatorCore()
        self._domain = DataspaceDomain(self._core)
        self._array_registry = DataspaceArrayRegistry(self._core)

    def __repr__(self):
        output = StringIO()
        output.write("DataspaceCreator(")
        if self._core.ndim > 0:
            output.write("\n Shared Dimensions:\n")
            for dim in self._core.shared_dims():
                output.write(f"  '{dim.name}':  {repr(dim)},\n")
        if self._core.narray > 0:
            output.write("\n Array Creators:\n")
            for array_creator in self._core.array_creators():
                output.write(f"  '{array_creator.name}':{repr(array_creator)}\n")
        output.write(")")
        return output.getvalue()

    def _repr_html_(self):
        output = StringIO()
        output.write(f"<h4>{self.__class__.__name__}</h4>\n")
        output.write("<ul>\n")
        output.write("<li>\n")
        output.write("Shared Dimensions\n")
        if self._core.ndim > 0:
            output.write("<table>\n")
            for dim in self._core.shared_dims():
                output.write(
                    f'<tr><td style="text-align: left;">{dim.html_input_summary()} '
                    f"&rarr; SharedDim({dim.html_output_summary()})</td>\n</tr>\n"
                )
            output.write("</table>\n")
        output.write("</li>\n")
        output.write("<li>\n")
        output.write("Array Creators\n")
        for array_creator in self._core.array_creators():
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
        dim_filters: Optional[Dict[str, tiledb.FilterList]] = None,
        offsets_filters: Optional[tiledb.FilterList] = None,
        attrs_filters: Optional[tiledb.FilterList] = None,
        allows_duplicates: bool = False,
        sparse: bool = False,
    ):
        """Adds a new array to the CF dataspace.

        The name of each array must be unique. All other properties should satisfy
        the same requirements as a ``tiledb.ArraySchema``.

        Parameters
        ----------
        array_name
            Name of the new array to be created.
        dims
            An ordered list of the names of the shared dimensions for the domain
            of this array.
        cell_order
            The order in which TileDB stores the cells on disk inside a tile. Valid
            values are: ``row-major`` (default) or ``C`` for row major; ``col-major`` or
            ``F`` for column major; or ``Hilbert`` for a Hilbert curve.
        tile_order
            The order in which TileDB stores the tiles on disk. Valid values are:
            ``row-major`` or ``C`` (default) for row major; or ``col-major`` or  ``F``
            for column major.
        capacity
            The number of cells in a data tile of a sparse fragment.
        tiles
            An optional ordered list of tile sizes for the dimensions of the array. The
            length must match the number of dimensions in the array.
        dim_filters
            A dict from dimension name to a ``tiledb.FilterList`` for dimensions in the
            array.
        offsets_filters
            Filters for the offsets for variable length attributes or dimensions.
        attrs_filters
            Default filters to use when adding an attribute to the array.
        allows_duplicates
            Specifies if multiple values can be stored at the same coordinate. Only
            allowed for sparse arrays.
        sparse
            Specifies if the array is a sparse TileDB array (true) or dense TileDB
            array (false).
        """
        ArrayCreator(
            registry=self._array_registry,
            dim_registry=self._domain,
            name=array_name,
            dim_order=dims,
            cell_order=cell_order,
            tile_order=tile_order,
            capacity=capacity,
            tiles=tiles,
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

        Parameters
        ----------
        attr_name
            Name of the new attribute that will be added.
        array_name
            Name of the array the attribute will be added to.
        dtype
            Numpy dtype of the new attribute.
        fill
            Fill value for unset cells.
        var
            Specifies if the attribute is variable length (automatic for byte/strings).
        nullable
            Specifies if the attribute is nullable using validity tiles.
        filters
            Specifies compression filters for the attribute.
        """
        array_creator = self._core.get_array_creator(array_name)
        array_creator.add_attr_creator(attr_name, dtype, fill, var, nullable, filters)

    def add_shared_dim(self, dim_name: str, domain: Tuple[Any, Any], dtype: np.dtype):
        """Adds a new dimension to the CF dataspace.

        Each dimension name must be unique. Adding a dimension where the name, domain,
        and dtype matches a current dimension does nothing.

        Parameters
        ----------
        dim_name
            Name of the new dimension to be created.
        domain
            The (inclusive) interval on which the dimension is valid.
        dtype
            The numpy dtype of the values and domain of the dimension.
        """
        SharedDim(dim_name, domain, dtype, registry=self._domain)

    def array_creators(self):
        """Iterates over array creators in the CF dataspace."""
        return self._core.array_creators()

    def create_array(
        self,
        uri: str,
        key: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """Creates a TileDB array for a CF dataspace with only one array.

        Parameters
        ---------
        uri
            Uniform resource identifier for the TileDB array to be created.
        key
            If not ``None``, encryption key to decrypt the array.
        ctx
            If not ``None``, TileDB context wrapper for a TileDB storage manager.
        """
        if self._core.narray != 1:
            raise ValueError(
                f"Can only use `create_array` for a {self.__class__.__name__} with "
                f"exactly 1 array creator."
            )
        array_creator = next(self._core.array_creators())
        array_creator.create(uri, key=key, ctx=ctx)

    def create_group(
        self,
        uri: str,
        key: Optional[Union[Dict[str, str], str]] = None,
        ctx: Optional[tiledb.Ctx] = None,
        append: bool = False,
    ):
        """Creates a TileDB group and arrays for the CF dataspace.

        Parameters
        ----------
        uri
            Uniform resource identifier for the TileDB group to be created.
        key
            If not ``None``, encryption key, or dictionary of encryption keys, to
            decrypt arrays.
        ctx
            If not ``None``, TileDB context wrapper for a TileDB storage manager.
        append
            If ``True``, add arrays in the dataspace to an already existing group. The
            arrays in the dataspace cannot be in the group that is being append to.
        """
        schema = self.to_schema(ctx)
        create_group(uri, schema, key=key, ctx=ctx, append=append)

    def get_array_creator(self, array_name: str):
        """Returns the array creator with the requested name.

        Parameters
        ----------
        array_name
            Name of the array to return.
        """
        return self._core.get_array_creator(array_name)

    def get_array_creator_by_attr(self, attr_name: str):
        """Returns the array creator with the requested attribute in it.

        Parameters
        ----------
        attr_name
            Name of the attribute to return the array creator with.
        """
        return self._core.get_array_creator_by_attr(attr_name)

    def get_shared_dim(self, dim_name: str):
        """Returns the shared dimension with the requested name.

        Parameters
        ----------
        dim_name
            Name of the array to return.
        """
        return self._core.get_shared_dim(dim_name)

    def remove_array_creator(self, array_name: str):
        """Removes the specified array and all its attributes from the CF dataspace.

        Parameters
        ----------
        array_name
            Name of the array that will be removed.
        """
        self._core.deregister_array_creator(array_name)

    def remove_attr_creator(self, attr_name: str):
        """Removes the specified attribute from the CF dataspace.

        Parameters
        ----------
        attr_name
            Name of the attribute that will be removed.
        """
        array_creator = self._core.get_array_creator_by_attr(attr_name=attr_name)
        array_creator.remove_attr_creator(attr_name)

    def remove_shared_dim(self, dim_name: str):
        """Removes the specified dimension from the CF dataspace.

        This can only be used to remove dimensions that are not currently being used in
        an array.

        Parameters
        ----------
        dim_name
            Name of the dimension to be removed.
        """
        self._core.deregister_shared_dim(dim_name)

    def shared_dims(self):
        """Iterators over shared dimensions in the CF dataspace."""
        return self._core.shared_dims()

    def to_schema(
        self, ctx: Optional[tiledb.Ctx] = None
    ) -> Dict[str, tiledb.ArraySchema]:
        """Returns a dictionary of array schemas for the CF dataspace.

        Parameters
        ----------
        ctx
            If not ``None``, TileDB context wrapper for a TileDB storage manager.

        Returns
        -------
        Dict[str, tiledb.ArraySchema]
            A dictionary of array schemas for the CF dataspace.
        """
        array_schemas = {}
        for array_creator in self._core.array_creators():
            try:
                array_schemas[array_creator.name] = array_creator.to_schema(ctx)
            except tiledb.libtiledb.TileDBError as err:
                raise RuntimeError(
                    f"Failed to create an ArraySchema for array '{array_creator.name}'."
                    f" {str(err)}"
                ) from err
        return array_schemas


class DataspaceCreatorCore:
    def __init__(self):
        self._shared_dims: Dict[str, SharedDim] = {}
        self._array_creators: Dict[str, ArrayCreator] = {}

    def array_creators(self):
        """Iterator over array creators in the CF dataspace."""
        return iter(self._array_creators.values())

    def check_new_array_name(self, array_name: str):
        """Raises an error if the input name is not a valid new array name.

        Parameters
        ----------
        array_name
            The name to check.
        """
        if array_name in self._array_creators:
            raise ValueError(f"An array with name '{array_name}' already exists.")

    def check_new_dim(self, shared_dim: SharedDim):
        """Raises an error if the input dimension is not a valid new shared dimension.

        Parameters
        ----------
        shared_dim
            Input shared dimension to check.
        """
        if (
            shared_dim.name in self._shared_dims
            and shared_dim != self._shared_dims[shared_dim.name]
        ):
            raise ValueError(
                f"A different dimension with name '{shared_dim.name}' already exists."
            )

    def check_rename_shared_dim(self, original_name: str, new_name: str):
        """ "Raise error if the shared dimension cannot be renamed to the new name.

        Parameters
        ----------
        original_name
            The original name for the shared dimension to rename.
        new_name
            The new name for the shared dimension.
        """
        if new_name in self._shared_dims:
            raise NotImplementedError(
                f"Cannot rename dimension '{original_name}' to '{new_name}'. A "
                f"dimension with the same name already exists, and merging dimensions "
                f"has not yet been implemented."
            )
        for array_creator in self.array_creators():
            if array_creator.has_attr_creator(new_name) and original_name in set(
                dim_creator.name for dim_creator in array_creator.domain_creator
            ):
                raise ValueError(
                    f"Cannot rename dimension '{original_name}' to '{new_name}'. An"
                    f" attribute with the same name already exists in the array "
                    f"'{array_creator.name}' that uses this dimension."
                )

    def deregister_array_creator(self, array_name: str):
        """Removes the specified array and all its attributes from the CF dataspace.

        Parameters
        ----------
        array_name
            Name of the array that will be removed.
        """
        del self._array_creators[array_name]

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
        """Returns the array creator with the requested name.

        Parameters
        ----------
        array_name
            Name of the array creator to return.
        """
        return self._array_creators[array_name]

    def get_array_creator_by_attr(self, attr_name: str) -> ArrayCreator:
        """Returns an array creator that contains the requested attribute.

        Parameters
        ----------
        attr_name
            Name of the attribute to get the array creator for.
        """
        requested = None
        for array_creator in self.array_creators():
            if array_creator.has_attr_creator(attr_name):
                if requested is not None:
                    raise ValueError(
                        f"Multiple array creators have an attribute named "
                        f"'{attr_name}'."
                    )
                requested = array_creator
        if requested is None:
            raise KeyError(f"No attribute with the name '{attr_name}'.")
        return requested

    def get_attr_creator(self, attr_name: str) -> AttrCreator:
        """Returns the attribute creator with the requested name.

        Parameters
        ----------
        attr_name
            The name of the attribute creator to return.
        """
        array_creator = self.get_array_creator_by_attr(attr_name=attr_name)
        return array_creator.attr_creator(attr_name)

    def get_shared_dim(self, dim_name: str) -> SharedDim:
        """Returns the dim creator with the requested name.

        Parameters
        ----------
        dim_name
            The name of the dimension to return.
        """
        return self._shared_dims[dim_name]

    @property
    def narray(self) -> int:
        """The number of array creators."""
        return len(self._array_creators)

    @property
    def ndim(self) -> int:
        """The number of shared dimensions."""
        return len(self._shared_dims)

    def register_array_creator(self, array_creator: ArrayCreator):
        """Registers a new array creator with the CF dataspace.

        Parameters
        ----------
        array_creator
            The new array creator to register.
        """
        self.check_new_array_name(array_creator.name)
        self._array_creators[array_creator.name] = array_creator

    def register_shared_dim(self, shared_dim: SharedDim):
        """Registers a new shared dimension to the CF dataspace.

        Parameters
        ----------
        shared_dim
            The new shared dimension to register.
        """
        self.check_new_dim(shared_dim)
        self._shared_dims[shared_dim.name] = shared_dim

    def shared_dims(self):
        """Iterates over shared dimensions in the CF dataspace."""
        return iter(self._shared_dims.values())

    def update_array_creator_name(self, original_name: str, new_name: str):
        """Update the name of the array creator.

        Parameters
        ----------
        original_name
            The original name of the array creator.
        new_name
            The new name of the array creator.
        """
        self._array_creators[new_name] = self._array_creators.pop(original_name)

    def update_shared_dim_name(self, original_name: str, new_name: str):
        """Update the name of the shared dimension.

        Parameters
        ----------
        original_name
            The original name of the shared dimension.
        new_name
            The new name of the shared dimension.
        """
        self._shared_dims[new_name] = self._shared_dims.pop(original_name)


class DataspaceArrayRegistry(MutableMapping):
    def __init__(self, core: DataspaceCreatorCore):
        self._core = core

    def __delitem__(self, name: str):
        self._core.deregister_array_creator(name)

    def __getitem__(self, name: str) -> ArrayCreator:
        return self._core.get_array_creator(name)

    def __iter__(self):
        return self._core.array_creators()

    def __len__(self) -> int:
        return self._core.narray

    def __setitem__(self, name: str, value: ArrayCreator):
        if value.is_registered:
            raise ValueError(f"Array creator '{value.name}' is already registered.")
        if name != value.name:
            value.name = name
        self._core.register_array_creator(value)

    def rename(self, old_name: str, new_name: str):
        self._core.check_new_array_name(new_name)
        self._core.update_array_creator_name(old_name, new_name)


class DataspaceDomain(MutableMapping):
    def __init__(self, core: DataspaceCreatorCore):
        self._core = core

    def __delitem__(self, name: str):
        self._core.deregister_shared_dim(name)

    def __getitem__(self, name: str) -> SharedDim:
        return self._core.get_shared_dim(name)

    def __iter__(self):
        return self._core.shared_dims()

    def __len__(self) -> int:
        return self._core.ndim

    def __setitem__(self, name: str, value: SharedDim):
        if value.is_registered:
            raise ValueError(f"Shared dimension '{value.name}' is already registered.")
        if name != value.name:
            value.name = name
        self._core.register_shared_dim(value)

    def rename(self, old_name: str, new_name: str):
        self._core.check_rename_shared_dim(old_name, new_name)
        self._core.update_shared_dim_name(old_name, new_name)
