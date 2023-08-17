TileDB-CF Python API Reference
******************************

Warning:

  The TileDB-CF Python library is still initial development and the
  API may change rapidly.


Group and Metadata Support
**************************

"tiledb.cf" is the core module for the TileDB-CF-Py library.

This module contains core classes and functions for supporting the
NetCDF data model in the TileDB storage engine. To use this module
simply import using:

   import tiledb.cf


Groups
======

class tiledb.cf.Group(uri, mode='r', key=None, timestamp=None, ctx=None)

   Class for accessing group metadata and arrays in a TileDB group.

   The group class is a context manager for accessing the arrays,
   group metadata, and attributes in a TileDB group. It can be used to
   access group-level metadata and open arrays inside the group.

   Parameters:
      * **uri** ("str") – Uniform resource identifier for TileDB group
        or array.

      * **mode** ("str") – Mode the array and metadata objects are
        opened in. Either read ‘r’ or write ‘w’ mode.

      * **key** ("Union"["Dict"["str", "str"], "str", "None"]) – If
        not "None", encryption key, or dictionary of encryption keys
        by array name, to decrypt arrays.

      * **timestamp** ("Optional"["int"]) – If not "None", timestamp
        to open the group metadata and array at.

      * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
        wrapper for a TileDB storage manager.

   close()

      Closes this Group, flushing all buffered data.

   close_array(array=None, attr=None)

      Closes one of the open arrays in the group, chosen by providing
      array name or attr name.

      Parameters:
         * **array** ("Optional"["str"]) – If not "None", close the
           array with this name. Overrides attr if both are provided.

         * **attr** ("Optional"["str"]) – If not "None", close the
           array that contains this attr. Attr must be in only one of
           the group arrays.

   classmethod create(uri, group_schema, key=None, ctx=None, append=False)

      Creates a TileDB group and the arrays inside the group from a
      group schema.

      This method creates a TileDB group at the provided URI and
      creates arrays inside the group with the names and array schemas
      from the provided group schema.

      Parameters:
         * **uri** ("str") – Uniform resource identifier for TileDB
           group or array.

         * **group_schema** ("GroupSchema") – Schema that defines the
           group to be created.

         * **key** ("Union"["Dict"["str", "str"], "str", "None"]) – If
           not "None", encryption key, or dictionary of encryption
           keys to decrypt arrays.

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

         * **append** ("bool") – If "True", add arrays from the
           provided group schema to an already existing group. The
           names for the arrays in the group schema cannot already
           exist in the group being append to.

   property has_metadata_array: bool

      Flag that is true if there a metadata array for storing group
      metadata.

   property meta: tiledb.Metadata | None

      Metadata object for the group, or "None" if no array to store
      group metadata exists.

   open_array(array=None, attr=None, mode=None)

      Opens one of the arrays in the group, chosen by providing array
      name or attr name, with an optional setting for a mode different
      from the default group mode.

      Parameters:
         * **array** ("Optional"["str"]) – If not "None", open the
           array with this name. Overrides attr if both are provided.

         * **attr** ("Optional"["str"]) – If not "None", open the
           array that contains this attr. Attr must be in only one of
           the group arrays.

         * **mode** ("str") – mode the array is opened in. Either read
           ‘r’ or write ‘w’. If not provided, defaults to group mode.

      Return type:
         "Array"

      Returns:
         tiledb.Array opened in the specified mode

class tiledb.cf.VirtualGroup(array_uris, mode='r', key=None, timestamp=None, ctx=None)

   Class for accessing group metadata and arrays in a virtual TileDB
   group.

   This is a subclass of "tiledb.cf.Group" that treats a dictionary of
   arrays like a TileDB group. If there is an array named
   "__tiledb_group", it will be treated as the group metadata array.

   See "tiledb.cf.Group" for documentation on the methods and
   properties available in this class.

   Parameters:
      * **array_uris** ("Dict"["str", "str"]) – Mapping from array
        names to array uniform resource identifiers.

      * **mode** ("str") – Mode the array and metadata objects are
        opened in. Either read ‘r’ or write ‘w’ mode.

      * **key** ("Union"["Dict"["str", "str"], "str", "None"]) – If
        not "None", encryption key, or dictionary of encryption keys,
        to decrypt arrays.

      * **timestamp** ("Optional"["int"]) – If not "None", timestamp
        to open the group metadata and array at.

      * **array** – DEPRECACTED: use group.open_array instead.

      * **attr** – DEPRECATED: use group.open_array instead.

      * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
        wrapper for a TileDB storage manager.

   classmethod create(uri, group_schema, key=None, ctx=None, append=False)

      Create the arrays in a group schema.

      This will create arrays for a group in a flat directory
      structure. The group metadata array is created at the provided
      URI, and all other arrays are created at "{uri}_{array_name}"
      where "{uri}" is the provided URI and "{array_name}" is the name
      of the array as stored in the group schema.

      Parameters:
         * **uri** ("str") – Uniform resource identifier for group
           metadata and prefix for arrays.

         * **group_schema** ("GroupSchema") – Schema that defines the
           group to be created.

         * **key** ("Union"["Dict"["str", "str"], "str", "None"]) – If
           not "None", encryption key, or dictionary of encryption
           keys to decrypt arrays.

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

         * **is_virtual** – (DEPRECATED) If "True", create arrays in a
           flat directory without creating a TileDB group.

         * **append** ("bool") – If "True", add to existing group. Not
           valid for virtual groups.


Group Schema
============

class tiledb.cf.GroupSchema(array_schemas=None, metadata_schema=None, use_default_metadata_schema=True, ctx=None)

   Schema for a TileDB group.

   A TileDB group is completely defined by the arrays in the group.
   This class is a mapping from array names to array schemas. It also
   contains an optional array schema for an array to store group-level
   metadata.

   Parameters:
      * **array_schemas** ("Optional"["Dict"["str", "ArraySchema"]]) –
        A dict of array names to array schemas in the group.

      * **metadata_schema** ("Optional"["ArraySchema"]) – If not
        "None", a schema for the group metadata array.

      * **use_default_metadata_schema** ("bool") – If "True" and
        "metadata_schema=None" a default schema will be created for
        the metadata array.

      * **ctx** ("Optional"["Ctx"]) – TileDB Context used for
        generating default metadata schema.

   arrays_with_attr(attr_name)

      Returns a tuple of the names of all arrays with a matching
      attribute.

      Parameter:
         attr_name: Name of the attribute to look up arrays for.

      Return type:
         "Optional"["List"["str"]]

      Returns:
         A tuple of the name of all arrays with a matching attribute,
         or *None* if no
            such array.

   check()

      Checks the correctness of each array in the GroupSchema.

   classmethod load(uri, ctx=None, key=None)

      Loads a schema for a TileDB group from a TileDB URI.

      Parameters:
         * **uri** ("str") – uniform resource identifier for the
           TileDB group

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

         * **key** ("Union"["Dict"["str", "str"], "str", "None"]) – If
           not "None", encryption key, or dictionary of encryption
           keys, to decrypt arrays.

   classmethod load_virtual(array_uris, ctx=None, key=None)

      Loads a schema for a TileDB group from a mapping of array names
      to array URIs.

      Parameters:
         * **array_uris** ("Dict"["str", "str"]) – Mapping from array
           names to array uniform resource identifiers.

         * **metadata_uri** – Array uniform resource identifier for
           array where metadata is stored.

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

         * **key** ("Union"["Dict"["str", "str"], "str", "None"]) – If
           not "None", encryption key, or dictionary of encryption
           keys, to decrypt arrays.

   property metadata_schema: ArraySchema | None

      ArraySchema for the group-level metadata.


Metadata Wrappers
=================

class tiledb.cf.ArrayMetadata(metadata)

   Class for accessing array-related metadata from a TileDB metadata
   object.

   This class provides a way for accessing the TileDB array metadata
   that excludes attribute and dimension specific metadata.

   Parameters:
      **metadata** (*tiledb.Metadata*) – TileDB array metadata object
      for the desired array.

class tiledb.cf.AttrMetadata(metadata, attr)

   Metadata wrapper for accessing attribute metadata.

   This class allows access to the metadata for an attribute stored in
   the metadata for a TileDB array.

   Parameters:
      * **metadata** (*tiledb.Metadata*) – TileDB array metadata for
        the array containing the desired attribute.

      * **attr** (*str*) – Name or index of the arrary attribute being
        requested.

class tiledb.cf.DimMetadata(metadata, dim)

   Metadata wrapper for accessing dimension metadata.

   This class allows access to the metadata for a dimension stored in
   the metadata for a TileDB array.

   Parameters:
      * **metadata** (*tiledb.Metadata*) – TileDB array metadata for
        the array containing the desired attribute.

      * **dim** (*str*) – Name or index of the arrary attribute being
        requested.


Dataspace Creator
*****************


Dataspace Creator
=================

class tiledb.cf.DataspaceCreator

   Creator for a group of arrays that satify the CF Dataspace
   Convention.

   This class can be used directly to create a TileDB group that
   follows the TileDB CF Dataspace convention. It is also useful as a
   super class for converters/ingesters of data from sources that
   follow a NetCDF or NetCDF-like data model to TileDB.

   add_array_creator(array_name, dims, cell_order='row-major', tile_order='row-major', capacity=0, tiles=None, dim_filters=None, offsets_filters=None, attrs_filters=None, allows_duplicates=False, sparse=False)

      Adds a new array to the CF dataspace.

      The name of each array must be unique. All other properties
      should satisfy the same requirements as a "tiledb.ArraySchema".

      Parameters:
         * **array_name** ("str") – Name of the new array to be
           created.

         * **dims** ("Sequence"["str"]) – An ordered list of the names
           of the shared dimensions for the domain of this array.

         * **cell_order** ("str") – The order in which TileDB stores
           the cells on disk inside a tile. Valid values are: "row-
           major" (default) or "C" for row major; "col-major" or "F"
           for column major; or "Hilbert" for a Hilbert curve.

         * **tile_order** ("str") – The order in which TileDB stores
           the tiles on disk. Valid values are: "row-major" or "C"
           (default) for row major; or "col-major" or "F" for column
           major.

         * **capacity** ("int") – The number of cells in a data tile
           of a sparse fragment.

         * **tiles** ("Optional"["Sequence"["int"]]) – An optional
           ordered list of tile sizes for the dimensions of the array.
           The length must match the number of dimensions in the
           array.

         * **dim_filters** ("Optional"["Dict"["str", "FilterList"]]) –
           A dict from dimension name to a "tiledb.FilterList" for
           dimensions in the array.

         * **offsets_filters** ("Optional"["FilterList"]) – Filters
           for the offsets for variable length attributes or
           dimensions.

         * **attrs_filters** ("Optional"["FilterList"]) – Default
           filters to use when adding an attribute to the array.

         * **allows_duplicates** ("bool") – Specifies if multiple
           values can be stored at the same coordinate. Only allowed
           for sparse arrays.

         * **sparse** ("bool") – Specifies if the array is a sparse
           TileDB array (true) or dense TileDB array (false).

   add_attr_creator(attr_name, array_name, dtype, fill=None, var=False, nullable=False, filters=None)

      Adds a new attribute to an array in the CF dataspace.

      The ‘dataspace name’ (name after dropping the suffix ".data" or
      ".index") must be unique.

      Parameters:
         * **attr_name** ("str") – Name of the new attribute that will
           be added.

         * **array_name** ("str") – Name of the array the attribute
           will be added to.

         * **dtype** ("dtype") – Numpy dtype of the new attribute.

         * **fill** ("Union"["int", "float", "str", "None"]) – Fill
           value for unset cells.

         * **var** ("bool") – Specifies if the attribute is variable
           length (automatic for byte/strings).

         * **nullable** ("bool") – Specifies if the attribute is
           nullable using validity tiles.

         * **filters** ("Optional"["FilterList"]) – Specifies
           compression filters for the attribute.

   add_shared_dim(dim_name, domain, dtype)

      Adds a new dimension to the CF dataspace.

      Each dimension name must be unique. Adding a dimension where the
      name, domain, and dtype matches a current dimension does
      nothing.

      Parameters:
         * **dim_name** ("str") – Name of the new dimension to be
           created.

         * **domain** ("Tuple"["Any", "Any"]) – The (inclusive)
           interval on which the dimension is valid.

         * **dtype** ("dtype") – The numpy dtype of the values and
           domain of the dimension.

   array_creators()

      Iterates over array creators in the CF dataspace.

   create_array(uri, key=None, ctx=None)

      Creates a TileDB array for a CF dataspace with only one array.

      Parameters:
         * **uri** ("str") – Uniform resource identifier for the
           TileDB array to be created.

         * **key** ("Optional"["str"]) – If not "None", encryption key
           to decrypt the array.

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

   create_group(uri, key=None, ctx=None, append=False)

      Creates a TileDB group and arrays for the CF dataspace.

      Parameters:
         * **uri** ("str") – Uniform resource identifier for the
           TileDB group to be created.

         * **key** ("Union"["Dict"["str", "str"], "str", "None"]) – If
           not "None", encryption key, or dictionary of encryption
           keys, to decrypt arrays.

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

         * **append** ("bool") – If "True", add arrays in the
           dataspace to an already existing group. The arrays in the
           dataspace cannot be in the group that is being append to.

   create_virtual_group(uri, key=None, ctx=None)

      Creates TileDB arrays for the CF dataspace.

      Parameters:
         * **uri** ("str") – Prefix for the uniform resource
           identifier for the TileDB arrays that will be created.

         * **key** ("Union"["Dict"["str", "str"], "str", "None"]) – If
           not "None", encryption key, or dictionary of encryption
           keys, to decrypt arrays.

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

   get_array_creator(array_name)

      Returns the array creator with the requested name.

      Parameters:
         **array_name** ("str") – Name of the array to return.

   get_array_creator_by_attr(attr_name)

      Returns the array creator with the requested attribute in it.

      Parameters:
         **attr_name** ("str") – Name of the attribute to return the
         array creator with.

   get_shared_dim(dim_name)

      Returns the shared dimension with the requested name.

      Parameters:
         **array_name** – Name of the array to return.

   remove_array_creator(array_name)

      Removes the specified array and all its attributes from the CF
      dataspace.

      Parameters:
         **array_name** ("str") – Name of the array that will be
         removed.

   remove_attr_creator(attr_name)

      Removes the specified attribute from the CF dataspace.

      Parameters:
         **attr_name** ("str") – Name of the attribute that will be
         removed.

   remove_shared_dim(dim_name)

      Removes the specified dimension from the CF dataspace.

      This can only be used to remove dimensions that are not
      currently being used in an array.

      Parameters:
         **dim_name** ("str") – Name of the dimension to be removed.

   shared_dims()

      Iterators over shared dimensions in the CF dataspace.

   to_schema(ctx=None)

      Returns a group schema for the CF dataspace.

      Parameters:
         **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
         wrapper for a TileDB storage manager.

      Return type:
         "GroupSchema"


Shared Dimension
================

class tiledb.cf.creator.SharedDim(dataspace_registry, name, domain, dtype)

   Definition for the name, domain and data type of a collection of
   dimensions.

   html_input_summary()

      Returns a HTML string summarizing the input for the dimension.

      Return type:
         "str"

   html_output_summary()

      Returns a string HTML summary of the "SharedDim".

      Return type:
         "str"

   property is_index_dim: bool

      Returns "True" if this is an *index dimension* and "False"
      otherwise.

      An index dimension is a dimension with an integer data type and
      whose domain starts at 0.

   property name: str

      Name of the shared dimension.


Array Creator
=============

class tiledb.cf.creator.ArrayCreator(dataspace_registry, name, dims, cell_order='row-major', tile_order='row-major', capacity=0, tiles=None, dim_filters=None, offsets_filters=None, attrs_filters=None, allows_duplicates=False, sparse=False)

   Creator for a TileDB array using shared dimension definitions.

   cell_order

      The order in which TileDB stores the cells on disk inside a
      tile. Valid values are: "row-major" (default) or "C" for row
      major; "col-major" or "F" for column major; or "Hilbert" for a
      Hilbert curve.

   tile_order

      The order in which TileDB stores the tiles on disk. Valid values
      are: "row-major" or "C" (default) for row major; or "col-major"
      or "F" for column major.

   capacity

      The number of cells in a data tile of a sparse fragment.

   offsets_filters

      Filters for the offsets for variable length attributes or
      dimensions.

   attrs_filters

      Default filters to use when adding an attribute to the array.

   allows_duplicates

      Specifies if multiple values can be stored at the same
      coordinate. Only allowed for sparse arrays.

   sparse

      If "True", creates a sparse array. Otherwise, create

   add_attr_creator(name, dtype, fill=None, var=False, nullable=False, filters=None)

      Adds a new attribute to an array in the CF dataspace.

      The attribute’s ‘dataspace name’ (name after dropping the suffix
      ".data" or ".index") must be unique.

      Parameters:
         * **name** ("str") – Name of the new attribute that will be
           added.

         * **dtype** ("dtype") – Numpy dtype of the new attribute.

         * **fill** ("Union"["int", "float", "str", "None"]) – Fill
           value for unset cells.

         * **var** ("bool") – Specifies if the attribute is variable
           length (automatic for byte/strings).

         * **nullable** ("bool") – Specifies if the attribute is
           nullable using validity tiles.

         * **filters** ("Optional"["FilterList"]) – Specifies
           compression filters for the attribute. If "None", use the
           array’s "attrs_filters" property.

   attr_creator(key)

      Returns the requested attribute creator

      Parameters:
         **key** ("Union"["int", "str"]) – The attribute creator index
         (int) or name (str).

      Return type:
         "AttrCreator"

      Returns:
         The attribute creator at the given index of name.

   create(uri, key=None, ctx=None)

      Creates a TileDB array at the provided URI.

      Parameters:
         * **uri** ("str") – Uniform resource identifier for the array
           to be created.

         * **key** ("Optional"["str"]) – If not "None", encryption key
           to decrypt arrays.

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

   property domain_creator: DomainCreator

      Domain creator that creates the domain for the TileDB array.

   html_summary()

      Returns a string HTML summary of the "ArrayCreator".

      Return type:
         "str"

   property name: str

      Name of the array.

   property nattr: int

      Number of attributes in the array.

   property ndim: int

      Number of dimensions in the array.

   remove_attr_creator(attr_name)

      Removes the requested attribute from the array.

      Parameters:
         **attr_name** – Name of the attribute to remove.

   to_schema(ctx=None)

      Returns an array schema for the array.

      Parameters:
         **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
         wrapper for a TileDB storage manager.

      Return type:
         "ArraySchema"


Domain Creator
==============

class tiledb.cf.creator.DomainCreator(array_registry, dataspace_registry)

   Creator for a TileDB domain.

   dim_creator(dim_id)

      Returns a dimension creator from the domain creator given the
      dimension’s index or name.

      Parameter:
         dim_id: dimension index (int) or name (str)

      Returns:
         The dimension creator with the requested key.

   inject_dim_creator(dim_name, position, **dim_kwargs)

      Adds a new dimension creator at a specified location.

      Parameters:
         * **dim_name** ("str") – Name of the shared dimension to add
           to the array’s domain.

         * **position** ("int") – Position of the shared dimension.
           Negative values count backwards from the end of the new
           number of dimensions.

         * **dim_kwargs** – Keyword arguments to pass to "DimCreator".

   property ndim

      Number of dimensions in the domain.

   remove_dim_creator(dim_id)

      Removes a dimension creator from the array creator.

      Parameters:
         **dim_id** ("Union"["str", "int"]) – dimension index (int) or
         name (str)

   property tiles

      Tiles for the dimension creators in the domain.

   to_tiledb(ctx=None)

      Returns a TileDB domain from the contained dimension creators.

      Return type:
         "Domain"


Dimension Creator
=================

class tiledb.cf.creator.DimCreator(base, tile=None, filters=None)

   Creator for a TileDB dimension using a SharedDim.

   tile

      The tile size for the dimension.

   filters

      Specifies compression filters for the dimension.

   property base: SharedDim

      Shared definition for the dimensions name, domain, and dtype.

   property domain: Tuple[int | float | str | None, int | float | str | None] | None

      The (inclusive) interval on which the dimension is valid.

   property dtype: dtype

      The numpy dtype of the values and domain of the dimension.

   html_summary()

      Returns a string HTML summary of the "DimCreator".

      Return type:
         "str"

   property name: str

      Name of the dimension.

   to_tiledb(ctx=None)

      Returns a "tiledb.Dim" using the current properties.

      Parameters:
         **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
         wrapper for a TileDB storage manager.

      Return type:
         "Domain"

      Returns:
         A tiledb dimension with the set properties.


Attribute Creator
=================

class tiledb.cf.creator.AttrCreator(array_registry, name, dtype, fill=None, var=False, nullable=False, filters=None)

   Creator for a TileDB attribute.

   dtype

      Numpy dtype of the attribute.

   fill

      Fill value for unset cells.

   var

      Specifies if the attribute is variable length (automatic for
      byte/strings).

   nullable

      Specifies if the attribute is nullable using validity tiles.

   filters

      Specifies compression filters for the attribute.

   html_summary()

      Returns a string HTML summary of the "AttrCreator".

      Return type:
         "str"

   property name: str

      Name of the attribute.

   to_tiledb(ctx=None)

      Returns a "tiledb.Attr" using the current properties.

      Parameters:
         **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
         wrapper for a TileDB storage manager.

      Return type:
         "Attr"

      Returns:
         Returns an attribute with the set properties.


NetCDF to TileDB Conversion
***************************


Auto-convert Function
=====================

tiledb.cf.from_netcdf(input_file, output_uri, input_group_path='/', recursive=True, output_key=None, output_ctx=None, unlimited_dim_size=10000, dim_dtype=dtype('uint64'), tiles_by_var=None, tiles_by_dims=None, coords_to_dims=False, collect_attrs=True, unpack_vars=False, offsets_filters=None, attrs_filters=None, copy_metadata=True, use_virtual_groups=False)

   Converts a NetCDF input file to nested TileDB CF dataspaces.

   See "NetCDF4ConverterEngine" for more information on the backend
   converter engine used for the conversion.

   Parameters:
      * **input_file** ("Union"["str", "Path"]) – The input NetCDF
        file to generate the converter engine from.

      * **output_uri** ("str") – The uniform resource identifier for
        the TileDB group to be created.

      * **input_group_path** ("str") – The path to the NetCDF group to
        copy data from. Use "'/'" for the root group.

      * **recursive** ("bool") – If "True", recursively convert groups
        in a NetCDF file. Otherwise, only convert group provided.

      * **output_key** ("Optional"["str"]) – If not "None", encryption
        key to decrypt arrays.

      * **output_ctx** ("Optional"["Ctx"]) – If not "None", TileDB
        context wrapper for a TileDB storage manager.

      * **dim_dtype** ("dtype") – The numpy dtype for the TileDB
        dimensions created from NetCDF dimensions.

      * **unlimited_dim_size** ("int") – The size of the domain for
        TileDB dimensions created from unlimited NetCDF dimensions.

      * **dim_dtype** – The numpy dtype for TileDB dimensions.

      * **tiles_by_var** ("Optional"["Dict"["str", "Dict"["str",
        "Optional"["Sequence"["int"]]]]]) – A map from the name of a
        NetCDF variable to the tiles of the dimensions of the variable
        in the generated TileDB array.

      * **tiles_by_dims** ("Optional"["Dict"["str",
        "Dict"["Sequence"["str"], "Optional"["Sequence"["int"]]]]]) –
        A map from the name of NetCDF dimensions defining a variable
        to the tiles of those dimensions in the generated TileDB
        array.

      * **coords_to_dims** ("bool") – If "True", convert the NetCDF
        coordinate variable into a TileDB dimension for sparse arrays.
        Otherwise, convert the coordinate dimension into a TileDB
        dimension and the coordinate variable into a TileDB attribute.

      * **collect_attrs** ("bool") – If "True", store all attributes
        with the same dimensions in the same array. Otherwise, store
        each attribute in a scalar array.

      * **unpack_vars** ("bool") – Unpack NetCDF variables with NetCDF
        attributes "scale_factor" or "add_offset" using the
        transformation "scale_factor * value + unpack".

      * **offsets_filters** ("Optional"["FilterList"]) – Default
        filters for all offsets for variable attributes and
        dimensions.

      * **attrs_filters** ("Optional"["FilterList"]) – Default filters
        for all attributes.

      * **copy_metadata** ("bool") – If  "True" copy NetCDF group and
        variable attributes to TileDB metadata. If "False" do not copy
        metadata.

      * **use_virtual_groups** ("bool") – If "True", create a virtual
        group using "output_uri" as the name for the group metadata
        array. All other arrays will be named using the convention
        "{uri}_{array_name}" where "array_name" is the name of the
        array.


NetCDF4 Converter Engine
========================

class tiledb.cf.NetCDF4ConverterEngine(default_input_file=None, default_group_path=None)

   Converter for NetCDF to TileDB using netCDF4.

   This class is used to generate and copy data to a TileDB group or
   array from a NetCDF file. The converter can be auto-generated from
   a NetCDF group, or it can be manually defined.

   This is a subclass of "tiledb.cf.DataspaceCreator". See
   "tiledb.cf.DataspaceCreator" for documentation of additional
   properties and methods.

   add_array_converter(array_name, dims, cell_order='row-major', tile_order='row-major', capacity=0, tiles=None, dim_filters=None, offsets_filters=None, attrs_filters=None, allows_duplicates=False, sparse=False)

      Adds a new NetCDF to TileDB array converter to the CF dataspace.

      The name of each array must be unique. All properties must match
      the normal requirements for a "TileDB.ArraySchema".

      Parameters:
         * **array_name** ("str") – Name of the new array to be
           created.

         * **dims** ("Sequence"["str"]) – An ordered list of the names
           of the shared dimensions for the domain of this array.

         * **cell_order** ("str") – The order in which TileDB stores
           the cells on disk inside a tile. Valid values are: "row-
           major" (default) or "C" for row major; "col-major" or "F"
           for column major; or "Hilbert" for a Hilbert curve.

         * **tile_order** ("str") – The order in which TileDB stores
           the tiles on disk. Valid values are: "row-major" or "C"
           (default) for row major; or "col-major" or "F" for column
           major.

         * **capacity** ("int") – The number of cells in a data tile
           of a sparse fragment.

         * **tiles** ("Optional"["Sequence"["int"]]) – An optional
           ordered list of tile sizes for the dimensions of the array.
           The length must match the number of dimensions in the
           array.

         * **dim_filters** ("Optional"["Dict"["str", "FilterList"]]) –
           A dict from dimension name to a "FilterList" for dimensions
           in the array.

         * **offsets_filters** ("Optional"["FilterList"]) – Filters
           for the offsets for variable length attributes or
           dimensions.

         * **attrs_filters** ("Optional"["FilterList"]) – Default
           filters to use when adding an attribute to the array.

         * **allows_duplicates** ("bool") – Specifies if multiple
           values can be stored at the same coordinate. Only allowed
           for sparse arrays.

         * **sparse** ("bool") – Specifies if the array is a sparse
           TileDB array (true) or dense TileDB array (false).

   add_coord_to_dim_converter(ncvar, dim_name=None, domain=None, dtype=None, unpack=False)

      Adds a new NetCDF coordinate to TileDB dimension converter.

      Parameters:
         * **var** – NetCDF coordinate variable to be converted.

         * **dim_name** ("Optional"["str"]) – If not "None", name to
           use for the TileDB dimension.

         * **domain** ("Optional"["Tuple"["TypeVar"("DType",
           covariant=True), "TypeVar"("DType", covariant=True)]]) – If
           not "None", the domain the TileDB dimension is valid on.

         * **dtype** ("Optional"["dtype"]) – If not "None", the data
           type the TileDB dimension will be set to.

         * **unpack** ("bool") – Unpack NetCDF data that has NetCDF
           attributes "scale_factor" or "add_offset" using the
           transformation "scale_factor * value + unpack".

   add_dim_to_dim_converter(ncdim, unlimited_dim_size=None, dtype=dtype('uint64'), dim_name=None)

      Adds a new NetCDF dimension to TileDB dimension converter.

      Parameters:
         * **ncdim** ("Dimension") – NetCDF dimension to be converted.

         * **unlimited_dim_size** ("Optional"["int"]) – The size to
           use if the dimension is unlimited. If "None", the current
           size of the NetCDF dimension will be used.

         * **dtype** ("dtype") – Numpy type to use for the NetCDF
           dimension.

         * **dim_name** ("Optional"["str"]) – If not "None", output
           name of the TileDB dimension.

   add_scalar_to_dim_converter(dim_name='__scalars', dtype=dtype('uint64'))

      Adds a new TileDB dimension for NetCDF scalar variables.

      Parameters:
         * **dim_name** ("str") – Output name of the dimension.

         * **dtype** ("dtype") – Numpy type to use for the scalar
           dimension

   add_var_to_attr_converter(ncvar, array_name, attr_name=None, dtype=None, fill=None, var=False, nullable=False, filters=None, unpack=False)

      Adds a new variable to attribute converter to an array in the CF
      dataspace.

      The attribute’s ‘dataspace name’ (name after dropping the suffix
      ".data" or ".index") must be unique.

      Parameters:
         * **ncvar** ("Variable") – NetCDF variable to convert to a
           TileDB attribute.

         * **name** – Name of the new attribute that will be added. If
           "None", the name will be copied from the NetCDF variable.

         * **dtype** ("Optional"["dtype"]) – Numpy dtype of the new
           attribute. If "None", the data type will be copied from the
           variable.

         * **fill** ("Union"["int", "float", "str", "None"]) – Fill
           value for unset cells. If "None", the fill value will be
           copied from the NetCDF variable if it has a fill value.

         * **var** ("bool") – Specifies if the attribute is variable
           length (automatic for byte/strings).

         * **nullable** ("bool") – Specifies if the attribute is
           nullable using validity tiles.

         * **filters** ("Optional"["FilterList"]) – Specifies
           compression filters for the attribute.

         * **unpack** ("bool") – Unpack NetCDF data that has NetCDF
           attributes "scale_factor" or "add_offset" using the
           transformation "scale_factor * value + unpack".

   convert_to_array(output_uri, key=None, ctx=None, timestamp=None, input_netcdf_group=None, input_file=None, input_group_path=None, assigned_dim_values=None, assigned_attr_values=None, copy_metadata=True)

      Creates a TileDB arrays for a CF dataspace with only one array
      and copies data into it using the NetCDF converter engine.

      Parameters:
         * **output_uri** ("str") – Uniform resource identifier for
           the TileDB array to be created.

         * **key** ("Optional"["str"]) – If not "None", encryption key
           to encrypt and decrypt output arrays.

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

         * **timestamp** ("Optional"["int"]) – If not "None", the
           TileDB timestamp to write the NetCDF data to TileDB at.

         * **input_netcdf_group** ("Optional"["Group"]) – If not
           "None", the NetCDF group to copy data from. This will be
           prioritized over "input_file" if both are provided.

         * **input_file** ("Union"["str", "Path", "None"]) – If not
           "None", the NetCDF file to copy data from. This will not be
           used if "netcdf_group" is not "None".

         * **input_group_path** ("Optional"["str"]) – If not "None",
           the path to the NetCDF group to copy data from.

         * **assigned_dim_values** ("Optional"["Dict"["str", "Any"]])
           – Mapping from dimension name to value for dimensions that
           are not converter from the NetCDF group.

         * **assigned_attr_values** ("Optional"["Dict"["str",
           "ndarray"]]) – Mapping from attribute name to numpy array
           of values for attributes that are not converted from the
           NetCDF group.

         * **copy_metadata** ("bool") – If  "True" copy NetCDF group
           and variable attributes to TileDB metadata. If "False" do
           not copy metadata.

   convert_to_group(output_uri, key=None, ctx=None, timestamp=None, input_netcdf_group=None, input_file=None, input_group_path=None, assigned_dim_values=None, assigned_attr_values=None, copy_metadata=True, append=False)

      Creates a TileDB group and its arrays from the defined CF
      dataspace and copies data into them using the converter engine.

      Parameters:
         * **output_uri** ("str") – Uniform resource identifier for
           the TileDB group to be created.

         * **key** ("Optional"["str"]) – If not "None", encryption key
           to encrypt and decrypt output arrays.

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

         * **timestamp** ("Optional"["int"]) – If not "None", the
           TileDB timestamp to write the NetCDF data to TileDB at.

         * **input_netcdf_group** ("Optional"["Group"]) – If not
           "None", the NetCDF group to copy data from. This will be
           prioritized over "input_file" if both are provided.

         * **input_file** ("Union"["str", "Path", "None"]) – If not
           "None", the NetCDF file to copy data from. This will not be
           used if "netcdf_group" is not "None".

         * **input_group_path** ("Optional"["str"]) – If not "None",
           the path to the NetCDF group to copy data from.

         * **assigned_dim_values** ("Optional"["Dict"["str", "Any"]])
           – Mapping from dimension name to value for dimensions that
           are not converted from the NetCDF group.

         * **assigned_attr_values** ("Optional"["Dict"["str",
           "ndarray"]]) – Mapping from attribute name to numpy array
           of values for attributes that are not converted from the
           NetCDF group.

         * **copy_metadata** ("bool") – If  "True" copy NetCDF group
           and variable attributes to TileDB metadata. If "False" do
           not copy metadata.

         * **append** ("bool") – If "True", add arrays in the
           dataspace to an already existing group. The arrays in the
           dataspace cannot be in the group that is being append to.

   convert_to_virtual_group(output_uri, key=None, ctx=None, timestamp=None, input_netcdf_group=None, input_file=None, input_group_path=None, assigned_dim_values=None, assigned_attr_values=None, copy_metadata=True)

      Creates a TileDB group and its arrays from the defined CF
      dataspace and copies data into them using the converter engine.

      Parameters:
         * **output_uri** ("str") – Uniform resource identifier for
           the TileDB group to be created.

         * **key** ("Optional"["str"]) – If not "None", encryption key
           to encrypt and decrypt output arrays.

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

         * **timestamp** ("Optional"["int"]) – If not "None", the
           TileDB timestamp to write the NetCDF data to TileDB at.

         * **input_netcdf_group** ("Optional"["Group"]) – If not
           "None", the NetCDF group to copy data from. This will be
           prioritized over "input_file" if both are provided.

         * **input_file** ("Union"["str", "Path", "None"]) – If not
           "None", the NetCDF file to copy data from. This will not be
           used if "netcdf_group" is not "None".

         * **input_group_path** ("Optional"["str"]) – If not "None",
           the path to the NetCDF group to copy data from.

         * **assigned_dim_values** ("Optional"["Dict"["str", "Any"]])
           – Mapping from dimension name to value for dimensions that
           are not converted from the NetCDF group.

         * **assigned_attr_values** ("Optional"["Dict"["str",
           "ndarray"]]) – Mapping from attribute name to numpy array
           of values for attributes that are not converted from the
           NetCDF group.

         * **copy_metadata** ("bool") – If  "True" copy NetCDF group
           and variable attributes to TileDB metadata. If "False" do
           not copy metadata.

   copy_to_array(output_uri, key=None, ctx=None, timestamp=None, input_netcdf_group=None, input_file=None, input_group_path=None, assigned_dim_values=None, assigned_attr_values=None, copy_metadata=True)

      Copies data from a NetCDF group to a TileDB array.

      This will copy data from a NetCDF group that is defined either
      by a "netCDF4.Group" or by an input_file and group path. If
      neither the "netcdf_group" or "input_file" is specified, this
      will copy data from the input file "self.default_input_file".
      If both "netcdf_group" and "input_file" are set, this method
      will prioritize using the NetCDF group set by "netcdf_group".

      Parameters:
         * **output_uri** ("str") – Uniform resource identifier for
           the TileDB array data is being copied to.

         * **key** ("Optional"["str"]) – If not "None", encryption key
           to decrypt arrays.

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

         * **timestamp** ("Optional"["int"]) – If not "None", the
           TileDB timestamp to write the NetCDF data to TileDB at.

         * **input_netcdf_group** ("Optional"["Group"]) – If not
           "None", the NetCDF group to copy data from. This will be
           prioritized over "input_file" if both are provided.

         * **input_file** ("Union"["str", "Path", "None"]) – If not
           "None", the NetCDF file to copy data from. This will not be
           used if "netcdf_group" is not "None".

         * **input_group_path** ("Optional"["str"]) – If not "None",
           the path to the NetCDF group to copy data from.

         * **assigned_dim_values** ("Optional"["Dict"["str", "Any"]])
           – Mapping from dimension name to value for dimensions that
           are not copied from the NetCDF group.

         * **assigned_attr_values** ("Optional"["Dict"["str",
           "ndarray"]]) – Mapping from attribute name to numpy array
           of values for attributes that are not copied from the
           NetCDF group.

         * **copy_metadata** ("bool") – If  "True" copy NetCDF group
           and variable attributes to TileDB metadata. If "False" do
           not copy metadata.

   copy_to_group(output_uri, key=None, ctx=None, timestamp=None, input_netcdf_group=None, input_file=None, input_group_path=None, assigned_dim_values=None, assigned_attr_values=None, copy_metadata=True)

      Copies data from a NetCDF group to a TileDB CF dataspace.

      This will copy data from a NetCDF group that is defined either
      by a "netCDF4.Group" or by an input_file and group path. If
      neither the "netcdf_group" or "input_file" is specified, this
      will copy data from the input file "self.default_input_file".
      If both "netcdf_group" and "input_file" are set, this method
      will prioritize using the NetCDF group set by "netcdf_group".

      Parameters:
         * **output_uri** ("str") – Uniform resource identifier for
           the TileDB group data is being copied to.

         * **key** ("Optional"["str"]) – If not "None", encryption key
           to decrypt arrays.

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

         * **timestamp** ("Optional"["int"]) – If not "None", the
           TileDB timestamp to write the NetCDF data to TileDB at.

         * **input_netcdf_group** ("Optional"["Group"]) – If not
           "None", the NetCDF group to copy data from. This will be
           prioritized over "input_file" if both are provided.

         * **input_file** ("Union"["str", "Path", "None"]) – If not
           "None", the NetCDF file to copy data from. This will not be
           used if "netcdf_group" is not "None".

         * **input_group_path** ("Optional"["str"]) – If not "None",
           the path to the NetCDF group to copy data from.

         * **assigned_dim_values** ("Optional"["Dict"["str", "Any"]])
           – Mapping from dimension name to value for dimensions that
           are not copied from the NetCDF group.

         * **assigned_attr_values** ("Optional"["Dict"["str",
           "ndarray"]]) – Mapping from attribute name to numpy array
           of values for the attributes that are not copied from the
           NetCDF group.

         * **copy_metadata** ("bool") – If  "True" copy NetCDF group
           and variable attributes to TileDB metadata. If "False" do
           not copy metadata.

   copy_to_virtual_group(output_uri, key=None, ctx=None, timestamp=None, input_netcdf_group=None, input_file=None, input_group_path=None, assigned_dim_values=None, assigned_attr_values=None, copy_metadata=True)

      Copies data from a NetCDF group to a TileDB CF dataspace.

      This will copy data from a NetCDF group that is defined either
      by a "netCDF4.Group" or by an input_file and group path. If
      neither the "netcdf_group" or "input_file" is specified, this
      will copy data from the input file "self.default_input_file".
      If both "netcdf_group" and "input_file" are set, this method
      will prioritize using the NetCDF group set by "netcdf_group".

      Parameters:
         * **output_uri** ("str") – Uniform resource identifier for
           the TileDB group data is being copied to.

         * **key** ("Optional"["str"]) – If not "None", encryption key
           to decrypt arrays.

         * **ctx** ("Optional"["Ctx"]) – If not "None", TileDB context
           wrapper for a TileDB storage manager.

         * **timestamp** ("Optional"["int"]) – If not "None", the
           TileDB timestamp to write the NetCDF data to TileDB at.

         * **input_netcdf_group** ("Optional"["Group"]) – If not
           "None", the NetCDF group to copy data from. This will be
           prioritized over "input_file" if both are provided.

         * **input_file** ("Union"["str", "Path", "None"]) – If not
           "None", the NetCDF file to copy data from. This will not be
           used if "netcdf_group" is not "None".

         * **input_group_path** ("Optional"["str"]) – If not "None",
           the path to the NetCDF group to copy data from.

         * **assigned_dim_values** ("Optional"["Dict"["str", "Any"]])
           – Mapping from dimension name to value for dimensions that
           are not from the NetCDF group.

         * **assigned_attr_values** ("Optional"["Dict"["str",
           "ndarray"]]) – Mapping from attribute name to numpy array
           of values for attributes that are not copied from the
           NetCDF group.

         * **copy_metadata** ("bool") – If  "True" copy NetCDF group
           and variable attributes to TileDB metadata. If "False" do
           not copy metadata.

   classmethod from_file(input_file, group_path='/', unlimited_dim_size=None, dim_dtype=dtype('uint64'), tiles_by_var=None, tiles_by_dims=None, coords_to_dims=False, collect_attrs=True, unpack_vars=False, offsets_filters=None, attrs_filters=None)

      Returns a "NetCDF4ConverterEngine" from a group in a NetCDF
      file.

      Parameters:
         * **input_file** ("Union"["str", "Path"]) – The input NetCDF
           file to generate the converter engine from.

         * **group_path** ("str") – The path to the NetCDF group to
           copy data from. Use "'/'" for the root group.

         * **unlimited_dim_size** ("Optional"["int"]) – The size of
           the domain for TileDB dimensions created from unlimited
           NetCDF dimensions. If "None", the current size of the
           NetCDF dimension will be used.

         * **dim_dtype** ("dtype") – The numpy dtype for TileDB
           dimensions.

         * **tiles_by_var** ("Optional"["Dict"["str",
           "Optional"["Sequence"["int"]]]]) – A map from the name of a
           NetCDF variable to the tiles of the dimensions of the
           variable in the generated TileDB array.

         * **tiles_by_dims** ("Optional"["Dict"["Sequence"["str"],
           "Optional"["Sequence"["int"]]]]) – A map from the name of
           NetCDF dimensions defining a variable to the tiles of those
           dimensions in the generated TileDB array.

         * **coords_to_dims** ("bool") – If "True", convert the NetCDF
           coordinate variable into a TileDB dimension for sparse
           arrays. Otherwise, convert the coordinate dimension into a
           TileDB dimension and the coordinate variable into a TileDB
           attribute.

         * **collect_attrs** ("bool") – If True, store all attributes
           with the same dimensions in the same array. Otherwise,
           store each attribute in a scalar array.

         * **unpack_vars** ("bool") – Unpack NetCDF variables with
           NetCDF attributes "scale_factor" or "add_offset" using the
           transformation "scale_factor * value + unpack".

         * **offsets_filters** ("Optional"["FilterList"]) – Default
           filters for all offsets for variable attributes and
           dimensions.

         * **attrs_filters** ("Optional"["FilterList"]) – Default
           filters for all attributes.

   classmethod from_group(netcdf_group, unlimited_dim_size=None, dim_dtype=dtype('uint64'), tiles_by_var=None, tiles_by_dims=None, coords_to_dims=False, collect_attrs=True, unpack_vars=False, offsets_filters=None, attrs_filters=None, default_input_file=None, default_group_path=None)

      Returns a "NetCDF4ConverterEngine" from a "netCDF4.Group".

      Parameters:
         * **group** – The NetCDF group to convert.

         * **unlimited_dim_size** ("Optional"["int"]) – The size of
           the domain for TileDB dimensions created from unlimited
           NetCDF dimensions. If "None", the current size of the
           NetCDF variable will be used.

         * **dim_dtype** ("dtype") – The numpy dtype for TileDB
           dimensions.

         * **tiles_by_var** ("Optional"["Dict"["str",
           "Optional"["Sequence"["int"]]]]) – A map from the name of a
           NetCDF variable to the tiles of the dimensions of the
           variable in the generated TileDB array.

         * **tiles_by_dims** ("Optional"["Dict"["Sequence"["str"],
           "Optional"["Sequence"["int"]]]]) – A map from the name of
           NetCDF dimensions defining a variable to the tiles of those
           dimensions in the generated TileDB array.

         * **coords_to_dims** ("bool") – If "True", convert the NetCDF
           coordinate variable into a TileDB dimension for sparse
           arrays. Otherwise, convert the coordinate dimension into a
           TileDB dimension and the coordinate variable into a TileDB
           attribute.

         * **collect_attrs** ("bool") – If "True", store all
           attributes with the same dimensions in the same array.
           Otherwise, store each attribute in a scalar array.

         * **unpack_vars** ("bool") – Unpack NetCDF variables with
           NetCDF attributes "scale_factor" or "add_offset" using the
           transformation "scale_factor * value + unpack".

         * **offsets_filters** ("Optional"["FilterList"]) – Default
           filters for all offsets for variable attributes and
           dimensions.

         * **attrs_filters** ("Optional"["FilterList"]) – Default
           filters for all attributes.

         * **default_input_file** ("Union"["str", "Path", "None"]) –
           If not "None", the default NetCDF input file to copy data
           from.

         * **default_group_path** ("Optional"["str"]) – If not "None",
           the default NetCDF group to copy data from. Use "'/'" to
           specify the root group.


NetCDF4 to TileDB Shared Dimension Converters
=============================================

class tiledb.cf.netcdf_engine.NetCDF4CoordToDimConverter(dataspace_registry, name, domain, dtype, input_dim_name, input_var_name, input_var_dtype, unpack)

   Converter for a NetCDF variable/dimension pair to a TileDB
   dimension.

   name

      Name of the TileDB dimension.

   domain

      The (inclusive) interval on which the dimension is valid.

   dtype

      The numpy dtype of the values and domain of the dimension.

   input_dim_name

      The name of input NetCDF dimension.

   input_var_name

      The name of input NetCDF variable.

   input_var_dtype

      The numpy dtype of the input NetCDF variable.

   copy_metadata(netcdf_group, tiledb_array)

      Copy the metadata data from NetCDF to TileDB.

      Parameters:
         * **netcdf_group** ("Dataset") – NetCDF group to get the
           metadata items from.

         * **tiledb_array** ("Array") – TileDB array to copy the
           metadata items to.

   get_query_size(netcdf_group)

      Returns the number of coordinates to copy from NetCDF to TileDB.

      Parameters:
         **netcdf_group** ("Dataset") – NetCDF group to copy the data
         from.

   get_values(netcdf_group, sparse, indexer)

      Returns the values of the NetCDF coordinate that is being
      copied, or None if the coordinate is of size 0.

      Parameters:
         * **netcdf_group** ("Dataset") – NetCDF group to get the
           coordinate values from.

         * **sparse** ("bool") – "True" if copying into a sparse array
           and "False" if copying into a dense array.

      Returns:
         The coordinate values needed for querying the TileDB
         dimension in the
            form a numpy array.

   html_input_summary()

      Returns a HTML string summarizing the input for the dimension.

   property is_index_dim: bool

      Returns "True" if this is an *index dimension* and "False"
      otherwise.

      An index dimension is a dimension with an integer data type and
      whose domain starts at 0.

class tiledb.cf.netcdf_engine.NetCDF4DimToDimConverter(dataspace_registry, name, domain, dtype, input_dim_name, input_dim_size, is_unlimited)

   Converter for a NetCDF dimension to a TileDB dimension.

   name

      Name of the TileDB dimension.

   domain

      The (inclusive) interval on which the dimension is valid.

   dtype

      The numpy dtype of the values and domain of the dimension.

   input_dim_name

      Name of the input NetCDF variable.

   input_dim_size

      Size of the input NetCDF variable.

   is_unlimited

      If True, the input NetCDF variable is unlimited.

   get_query_size(netcdf_group)

      Returns the number of coordinates to copy from NetCDF to TileDB.

      Parameters:
         **netcdf_group** ("Dataset") – NetCDF group to copy the data
         from.

   get_values(netcdf_group, sparse, indexer)

      Returns the values of the NetCDF dimension that is being copied.

      Parameters:
         * **netcdf_group** ("Dataset") – NetCDF group to get the
           dimension values from.

         * **sparse** ("bool") – "True" if copying into a sparse array
           and "False" if copying into a dense array.

      Return type:
         "Union"["ndarray", "slice"]

      Returns:
         The coordinates needed for querying the created TileDB
         dimension in the form
            of a numpy array if sparse is "True" and a slice
            otherwise.

   html_input_summary()

      Returns a HTML string summarizing the input for the dimension.

class tiledb.cf.netcdf_engine.NetCDF4ScalarToDimConverter(dataspace_registry, name, domain, dtype)

   Converter for NetCDF scalar (empty) dimensions to a TileDB
   Dimension.

   name

      Name of the TileDB dimension.

   domain

      The (inclusive) interval on which the dimension is valid.

   dtype

      The numpy dtype of the values and domain of the dimension.

   get_query_size(netcdf_group)

      Returns the number of coordinates to copy from NetCDF to TileDB.

      Parameters:
         **netcdf_group** ("Dataset") – NetCDF group to copy the data
         from.

   get_values(netcdf_group, sparse, indexer)

      Get dimension values from a NetCDF group.

      Parameters:
         * **netcdf_group** ("Dataset") – NetCDF group to get the
           dimension values from.

         * **sparse** ("bool") – "True" if copying into a sparse array
           and "False" if copying into a dense array.

      Return type:
         "Union"["ndarray", "slice"]

      Returns:
         The coordinates needed for querying the create TileDB
         dimension in the form
            of a numpy array if sparse is "True" and a slice
            otherwise.

   html_input_summary()

      Returns a string HTML summary.


NetCDF4 to TileDB Array Converter
=================================

class tiledb.cf.netcdf_engine.NetCDF4ArrayConverter(dataspace_registry, name, dims, cell_order='row-major', tile_order='row-major', capacity=0, tiles=None, dim_filters=None, offsets_filters=None, attrs_filters=None, allows_duplicates=False, sparse=False)

   Converter for a TileDB array from a collection of NetCDF variables.

   cell_order

      The order in which TileDB stores the cells on disk inside a
      tile. Valid values are: "row-major" (default) or "C" for row
      major; "col-major" or "F" for column major; or "Hilbert" for a
      Hilbert curve.

   tile_order

      The order in which TileDB stores the tiles on disk. Valid values
      are: "row-major" or "C" (default) for row major; or "col-major"
      or "F" for column major.

   capacity

      The number of cells in a data tile of a sparse fragment.

   offsets_filters

      Filters for the offsets for variable length attributes or
      dimensions.

   attrs_filters

      Default filters to use when adding an attribute to the array.

   allows_duplicates

      Specifies if multiple values can be stored at the same
      coordinate. Only allowed for sparse arrays.

   add_var_to_attr_converter(ncvar, name=None, dtype=None, fill=None, var=False, nullable=False, filters=None, unpack=False)

      Adds a new variable to attribute converter to the array creator.

      The attribute’s ‘dataspace name’ (name after dropping the suffix
      ".data" or ".index") be unique.

      Parameters:
         * **ncvar** ("Variable") – NetCDF variable to convert to a
           TileDB attribute.

         * **name** ("Optional"["str"]) – Name of the new attribute
           that will be added. If "None", the name will be copied from
           the NetCDF variable.

         * **dtype** ("Optional"["dtype"]) – Numpy dtype of the new
           attribute. If "None", the data type will be copied from the
           variable.

         * **fill** ("Union"["int", "float", "str", "None"]) – Fill
           value for unset cells. If "None", the fill value will be
           copied from the NetCDF variable if it has a fill value.

         * **var** ("bool") – Specifies if the attribute is variable
           length (automatic for byte/strings).

         * **nullable** ("bool") – Specifies if the attribute is
           nullable using validity tiles.

         * **filters** ("Optional"["FilterList"]) – Specifies
           compression filters for the attribute. If "None", use
           array’s "attrs_filters" property.

         * **unpack** ("bool") – Unpack NetCDF data that has NetCDF
           attributes "scale_factor" or "add_offset" using the
           transformation "scale_factor * value + unpack".

   copy(netcdf_group, tiledb_uri, tiledb_key=None, tiledb_ctx=None, tiledb_timestamp=None, assigned_dim_values=None, assigned_attr_values=None, copy_metadata=True)

      Copies data from a NetCDF group to a TileDB CF array.

      Parameters:
         * **netcdf_group** ("Group") – The NetCDF group to copy data
           from.

         * **tiledb_uri** ("str") – The TileDB array uri to copy data
           into.

         * **tiledb_key** ("Optional"["str"]) – If not "None", the
           encryption key for the TileDB array.

         * **tiledb_ctx** ("Optional"["str"]) – If not "None", the
           TileDB context wrapper for a TileDB storage manager to use
           when opening the TileDB array.

         * **tiledb_timestamp** ("Optional"["int"]) – If not "None",
           the timestamp to write the TileDB data at.

         * **assigned_dim_values** ("Optional"["Dict"["str", "Any"]])
           – Mapping from dimension name to value for dimensions that
           are not copied from the NetCDF group.

         * **assigned_attr_values** ("Optional"["Dict"["str",
           "ndarray"]]) – Mapping from attribute name to numpy array
           of values for attributes that are not copied from the
           NetCDF group.


NetCDF4 to TileDB Domain Converter
==================================

class tiledb.cf.netcdf_engine.NetCDF4DomainConverter(array_registry, dataspace_registry)

   Converter for NetCDF dimensions to a TileDB domain.

   get_query_coordinates(netcdf_group, sparse, indexer, assigned_dim_values=None)

      Returns the coordinates used to copy data from a NetCDF group.

      Parameters:
         * **netcdf_group** ("Group") – Group to query the data from.

         * **sparse** ("bool") – If "True", return coordinates for a
           sparse write. If "False", return coordinates for a dense
           write.

         * **assigned_dim_values** ("Optional"["Dict"["str", "Any"]])
           – Values for any non-NetCDF dimensions.

   inject_dim_creator(dim_name, position, **dim_kwargs)

      Add an additional dimension into the domain of the array.

      Parameters:
         * **dim_name** ("str") – Name of the shared dimension to add
           to the array’s domain.

         * **position** ("int") – Position of the shared dimension.
           Negative values count backwards from the end of the new
           number of dimensions.

         * **dim_kwargs** – Keyword arguments to pass to
           "NetCDF4ToDimConverter".

   property max_fragment_shape

      Maximum shape of a fragment when copying from NetCDF to TileDB.

      For a dense array, this is the shape of dense fragment. For a
      sparse array, it is the maximum number of coordinates copied for
      each dimension.

   property netcdf_dims

      Ordered tuple of NetCDF dimension names for dimension
      converters.

   remove_dim_creator(dim_id)

      Removes a dimension creator from the array creator.

      Parameters:
         **dim_id** ("Union"["str", "int"]) – dimension index (int) or
         name (str)


NetCDF4 to TileDB Dimension Converters
======================================

class tiledb.cf.netcdf_engine.NetCDF4ToDimConverter(base, tile=None, filters=None, max_fragment_length=None)

   Converter from NetCDF to a TileDB dimension in a
   "NetCDF4ArrayConverter" using a "SharedDim" for the base dimension.

   tile

      The tile size for the dimension.

   filters

      Specifies compression filters for the dimension.


NetCDF4 to TileDB Attribute Converters
======================================

class tiledb.cf.netcdf_engine.NetCDF4VarToAttrConverter(array_registry, name, dtype, fill, var, nullable, filters, input_var_name, input_var_dtype, unpack)

   Converter for a NetCDF variable to a TileDB attribute.

   name

      Name of the new attribute.

   dtype

      Numpy dtype of the attribute.

   fill

      Fill value for unset cells.

   var

      Specifies if the attribute is variable length (automatic for
      byte/strings).

   nullable

      Specifies if the attribute is nullable using validity tiles.

   filters

      Specifies compression filters for the attribute.

   input_var_name

      Name of the input NetCDF variable that will be converted.

   input_var_dtype

      Numpy dtype of the input NetCDF variable.

   copy_metadata(netcdf_group, tiledb_array)

      Copy the metadata data from NetCDF to TileDB.

      Parameters:
         * **netcdf_group** ("Dataset") – NetCDF group to get the
           metadata items from.

         * **tiledb_array** ("Array") – TileDB array to copy the
           metadata items to.

   get_values(netcdf_group, indexer)

      Returns TileDB attribute values from a NetCDF group.

      Parameters:
         * **netcdf_group** ("Dataset") – NetCDF group to get the
           dimension values from.

         * **indexer** ("Sequence"["slice"]) – Slice to query the
           NetCDF variable on.

      Return type:
         "ndarray"

      Returns:
         The values needed to set an attribute in a TileDB array. If
         the array

      is sparse the values will be returned as an 1D array; otherwise,
      they will be returned as an ND array.

   html_summary()

      Returns a string HTML summary of the "AttrCreator".

      Return type:
         "str"


TileDB Backend for xarray
*************************

TODO: Add documentation for the TileDB backend for xarray.
