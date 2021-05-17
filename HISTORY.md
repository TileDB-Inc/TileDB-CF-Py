# History

## In Progress

### Breaking Behavior

* Makes `NetCDF4ConverterEngine` methods `add_ncvar_to_attr` and `add_ncdim_to_dim` private.

### New Features

* Adds the parameter `is_virtual` to classmethod `Group.create` for flagging if the created group should be a virtual group.
* Adds the classmethod `Group.create_virtual` that creates a virtual group from a mapping of array names to URIs.
* Adds a classmethod `GroupSchema.load_virtual` for loading a virtual group defined by a mapping from array names to URIs.

### Improvements

### Deprecation

### Bug fixes

* Fixes detection of tiles from NetCDF variables with matching chunk sizes.
* Fixes f-strings in NetCDF4ConverterEngine `__repr__` method

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
