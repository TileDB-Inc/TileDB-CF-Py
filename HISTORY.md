# History

## In Progress

### Breaking Behavior

* Makes `NetCDF4ConverterEngine` methods `add_ncvar_to_attr` and `add_ncdim_to_dim` private.
* Renames method `create` in `DataspaceConverter` to `create_group` and changes method parameters.
* Renames method `convert` in `NetCDF4ConverterEngine` to `convert_to_group` and changes method parameters.
* Renames method `copy` in `NetCDF4ConverterEninge` to `copy_group` and changes method parameters.
* Adds `use_virtual_groups` parameter to `from_netcdf` and `from_netcdf_group` functions.

### New Features

* Adds the parameter `is_virtual` to classmethod `Group.create` for flagging if the created group should be a virtual group.
* Adds the classmethod `Group.create_virtual` that creates a virtual group from a mapping of array names to URIs.
* Adds a classmethod `GroupSchema.load_virtual` for loading a virtual group defined by a mapping from array names to URIs.
* Adds method `create_virtual_group` to `DataspaceConverter`.
* Adds method `convert_to_virtual_group` in `NetCDF4ConverterEngine`.
* Adds method `copy_to_virtual_group` in NetCDF4ConverterEngine`.
* Adds TileDB backend engine for xarray (previously in TileDB-xarray package).

### Improvements

### Deprecation

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
