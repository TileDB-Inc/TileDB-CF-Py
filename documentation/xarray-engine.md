---
title: TileDB-CF Xarray Engine
---


## Reading from TileDB with Xarray

Xarray uses a plugin infrastructure that allows third-party developers to create their own backend engines for reading data into xarray. TileDB-CF contains one such backend. To use the backend, make sure `tiledb-cf` is installed in your current Python environment, and use the `tiledb` engine:


```python
import xarray as xr

xr.open_dataset(tiledb_uri, engine="tiledb")
```

The TileDB engine can be used to open either a TileDB array or a TileDB group. See the requirements on the arrays below.

The backend engine will open the group or array as a dataset with TileDB dimensions mapping to dataset dimensions, TileDB attributes mapping to dataset variables/DataArrays, and TileDB metadata mapping to dataset attributes.


For a TileDB array to be readable by xarray, the following must be satisfied:

* The array must be dense.
* All dimensions on the array must be either signed or unsigned integers.
* Add dimensions must have a domain that starts at `0`.

For a TileDB group to be readable by xarray, the following must be satisfied:

* All arrays in the group satisfy the above requirements for the array to be readable.
* Each attribute has a unique "variable name".

The TileDB backend engine can be used with the standard xarray keyword arguments. It supports the additional TileDB-specific arguments:

* `config`: An optional TileDB configuration object to use in arrays and groups.
* `ctx`: An optional TileDB context object to use for all TileDB operations.
* `timestamp`: An optional timestamp to open the TileDB array at (not supported on groups).


## Writing from Xarray to TileDB

The xarray writer is stricter than the xarray backend engine (reader). While the reader will attempt to open arrays with multiple attributes, the xarray writer only creates arrays with one attribute per name.

There are two sets of functions for writing to xarray:

1. Single dataset ingestion.

    * Functions used: `from_xarray`
    * Useful when copying an entire xarray dataset to a TileDB group in a single function call.
    * Creates the group and copies all data and metadata to the new group in a single function call.

2. Multi-dataset ingestion.

    * Main functions: `create_group_from_xarray` and `copy_data_from_xarray`.
    * Additional helper function: `copy_metadata_from_xarray`.
    * Useful when copying multiple xarray datasets to a single TileDB group.
    * Creates the group and copies data to the group in separate API calls.

The xarray to TileDB writer will copy the dataset in the following way:

* One group is created for the dataset.
* Dataset "attributes" are copied to group level metadata.
* Each xarray variable is copied to its own dense TileDB array with a single TileDB attribute.

The array schema for an xarray variable is generated as follows:

* TileDB array properties:

  - The TileDB array is dense.

* TileDB Domain:

  - All dimensions have the same datatype determined by the `dim_dtype` encoding.

  - The dimension names in the TileDB array match the dimension names in the xarray variable.

  - The dimension tiles are determined by the `tiles` encoding.

  - The domain of each dimension is set to `[0, max_size - 1]` where `max_size` is computed as follows:

    1. Use the corresponding element of the  `max_shape` encoding if provided.

    2. If the `max_shape` encoding is not provided and the xarray dimension is "unlimited", use the largest possible size for this integer type.

    3. If the `max_shape` encoding is not provided and the xarray dimension is not "unlimited", use the size of the xarray dimension.

* TileDB Attribute:

  - The attribute datatype is the same as the variable datatype (after applying xarray encodings).

  - The attribute name is set using the following:

    1. Use the name provided by `attr_name` encoding.

    2. If the `attr_name` encoding is not provided and there is no dimension on this variable with the same name as the variable, use the name of the variable.

    3. If the `attr_name` encoding is provided and there is a dimension on this variable with the same name as the variable, use the variable name appended with `_`.

  - The attribute filters are determined by the `filters` encoding.



### TileDB Encoding

The writer takes a dictionary from dataset variable names to a dictionary of encodings for setting TileDB properties. The possible encoding keywords are provided in the table below.

+------------------+-----------------------------------------------+--------------------+
| Encoding Keyword | Details                                       | Type               |
+==================+===============================================+====================+
| `attr_name`      | Name to use for the TileDB attribute.         | str                |
+------------------+-----------------------------------------------+--------------------+
| `filters`        | Filter list to apply to the TileDB attribute. | tiledb.FilterList  |
+------------------+-----------------------------------------------+--------------------+
| `tiles`          | Tile sizes to apply to the TileDB dimensions. | tuple of ints      |
+------------------+-----------------------------------------------+--------------------+
| `max_shape`      | Maximum possible size of the TileDB array.    | tuple of ints      |
+------------------+-----------------------------------------------+--------------------+
| `dim_dtype`      | Datatype to use for the TileDB dimensions.    | str or numpy.dtype |
+------------------+-----------------------------------------------+--------------------+


### Region to Write

If the creating TileDB array's with either unlimited dimensions or with encoded `max_shape` larger than the current size of the xarray variable, then the region to write the data to needs to be provided. This is input as a dictionary from dimension names to slices. The slice uses xarray/numpy conventions and will write to a region that does **not** include the upper bound of the slice.


### Creating Multiple Fragments

When copying data with either the `from_xarray` or `copy_data_from_xarray` functions, the copy routine will use Xarray chunks for separate writes - creating multiple fragments.
