# History

## Unreleased

### Bug fixes

### Breaking Behavior

### New Features

### Improvements

### Deprecation

##  TileDB-CF-Py Release O.4.1

### Bug fixes

* Fixes initialization of `tile_order` in `NetCDF4ArrayConverter`.

### New Features

* Adds `tiles` property to `DomainCreator` to make getting/setting tiles easier.
* Adds support for adding non-NetCDF dimensions in the `NetCDF4ConverterArray`.

### Improvements

* Updates `NetCDF4CreatorEngine` to copy variable metadata when copying a NetCDF coordinate to a TileDB dimension.

### Deprecation

* Deprecates `input_name` in favor of `input_dim_name` and `input_var_name` and `input_dtype` in favor of `input_var_dtype` in `NetCDF4CoordToDimConverter`.
* Deprecates `input_name` and `input_size` in favor of `input_dim_name` and `input_dim_size` in `NetCDF4DimToDimConverter`.
* Deprecates `input_name` and `input_dtype` in favor of `input_var_name` and `input_var_dtype` in `NetCDF4VarToAttrConverter`.
* Deprecates module `tiledb.cf.engines.netcdf4_engine` in favor of `tiledb.cf.netcdf_engine`.

## TileDB-CF-Py Release 0.4.0

### Bug fixes

* Fix missing context in `GroupSchema.load`.

### Breaking Behavior

* Replaces array/attr setting in `Group` class initialization with `open` and `close` methods for opening any array in the group.
* `NetCDF4ConverterEngine.add_array_converter` adds a `NetCDF4ArrayConverter` and `NetCDF4ConverterEngine.add_array` inherits from `DataspaceCreator`.
* Updates `ArrayMetadata` to skip `DimMetadata`.

### New Features

* Adds `create_array` to `DataspaceCreator` for dataspaces with 1 array.
* Adds `convert_to_array` and `copy_to_array` to `NetCDF4ConverterEngine` for converters with 1 array.
* Adds `DimMetadata` class for handling dimension metadata.
* Adds `get_array_creator` and `get_shared_dim` methods to `DataspaceCreator` for direct access to `DataspaceCreator` components.
* Adds `array_creators` and `shared_dims` methods to `DataspaceCreator` for iterating over `DataspaceCreator` components.
* Adds `open_array` to `Group` to open any array in a group.
* Adds `close_array` to `Group` to close any open array in a group.

### Improvements

* Supports opening and explicitly closing multiple arrays with a `Group` object.

### Deprecation

* Deprecates `Group.create_virtual` in favor of `VirtualGroup.create`.
* Deprecates `NetCDF4ConverterEngine.add_scalar_dim_converter` in favor of `NetCDF4ConverterEngine.add_scalar_to_dim_converter`.
* Deprecates `Dataspace.add_array` in favor of `Dataspace.add_array_creator`.
* Deprecates `Dataspace.add_attr` in favor of `Dataspace.add_attr_creator`.
* Deprecates `Dataspace.add_dim` in favor of `Dataspace.add_shared_dim`.
* Deprecates `Dataspace.remove_array` in favor of `Dataspace.remove_array_creator`.
* Deprecates `Dataspace.remove_attr` in favor of `Dataspace.remove_attr_creator`.
* Deprecates `Dataspace.remove_dim` in favor of `Dataspace.remove_shared_dim`.
* Deprecates `Dataspace.array_names`, `Dataspace.dim_names`, and `Dataspace.attr_names`
* Deprecates `Dataspace.get_array_properties`, `Dataspace.get_attr_properties`, and `Dataspace.get_dim_properties`
* Deprecates `Dataspace.set_array_properties`, `Dataspace.set_attr_properties`, and `Dataspace.set_dim_properties`
* Deprecates `Dataspace.rename_array`, `Dataspace.rename_attr`, and `Dataspace.rename_dim`

## TileDB-CF-Py Release 0.3.0

### Breaking Behavior

* Makes `NetCDF4ConverterEngine` methods `add_ncvar_to_attr` and `add_ncdim_to_dim` private.
* Renames method `create` in `DataspaceConverter` to `create_group` and changes method parameters.
* Renames method `convert` in `NetCDF4ConverterEngine` to `convert_to_group` and changes method parameters.
* Renames method `copy` in `NetCDF4ConverterEninge` to `copy_group` and changes method parameters.
* Adds `use_virtual_groups` parameter to `from_netcdf` and `from_netcdf_group` functions.I
* Replaces parameter `tiles` with `tiles_by_dims` and `tiles_by_var` in `from_netcdf` and `from_netcdf_group` functions.
* Renames method `get_all_attr_arrays` in `GroupSchema` to `arrays_with_attr`.
* Removes methods `get_attr_array` and `set_default_metadata_schema` from `GroupSchema` class.
* Change `from_group` and `from_file` in `NetCDF4ConverterEngine` to default to convertering NetCDF coordinates to dimensions.
* Update TileDB-CF standard to version 0.2.0 and implement changes in `DataspaceCreator` class.
* Remove support for arrays with no dimensions in `DataspaceCreator.add_array` method.
* Increase minimum TileDB-Py version to 0.9.3.

### New Features

* Adds the parameter `is_virtual` to classmethod `Group.create` for flagging if the created group should be a virtual group.
* Adds the classmethod `Group.create_virtual` that creates a virtual group from a mapping of array names to URIs.
* Adds a classmethod `GroupSchema.load_virtual` for loading a virtual group defined by a mapping from array names to URIs.
* Adds method `create_virtual_group` to `DataspaceConverter`.
* Adds method `convert_to_virtual_group` in `NetCDF4ConverterEngine`.
* Adds method `copy_to_virtual_group` in `NetCDF4ConverterEngine`.
* Adds TileDB backend engine for xarray (previously in TileDB-xarray package).
* Adds methods to convert NetCDF group where all attributes are stored in separate arrays.
* Adds parameter to set default metadata schema in `GroupSchema` instance in not otherwise specified.
* Adds ability to convert NetCDF coordinates to TileDB dimensions.
* Adds the parameter `ctx` to classmethod `TileDBBackendEntrypoint.open_dataset` for using a TileDB context other than the default context.

### Bug fixes

* Fixes detection of tiles from NetCDF variables with matching chunk sizes.
* Fixes ouput in NetCDF4ConverterEngine and GroupSchema `__repr__` methods.
* Fixes build error when installing with `python setup.py install`.

## TileDB-CF-Py Release 0.2.0

The TileDB-CF-Py v0.2.0 release is the initial release of TileDB-CF-Py.

### New Features

* Initial release of the [TileDB CF dataspace specification](tiledb-cf-spec.md) for defining a data model compatible with the NetCDF data model.
* Adds a `Group` class for reading and writing to arrays in a TileDB group.
* Adds a `GroupSchema` class for loading the array schemas for ararys in a TileDB group.
* Adds `AttrMetadata` and `ArrayMetadata` class for managing attribute specific metadata.
* Adds a `DataspaceCreator` class for creating groups compatible with the TileDB CF dataspace specification.
* Adds a `NetCDF4ConverterEngine` for converting NetCDF files to TileDB with the `netCDF4` library.
* Adds functions and a command-line interface for converting NetCDF files to TileDB.
