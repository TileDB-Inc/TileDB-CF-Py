.. _tutorial:

******************
TileDB-CF Tutorial
******************


NetCDF Converter Engine
=======================

NetCDF-to-TileDB Compatibility
------------------------------

The TileDB-CF package provides an interface for generating TileDB groups from NetCDF datasets using the TileDB CF Dataspace convention. The CF Dataspace model supports the classic NetCDF-4 data model by mapping:

* NetCDF groups to TileDB groups;
* NetCDF dimensions to TileDB dimensions;
* NetCDF variables to TileDB attributes or TileDB dimensions;
* NetCDF attributes to TileDB array metadata.

Some features and use cases do not directly transfer or may need to be modified before use in TileDB.

* **Coordinates**: In NetCDF, it is a common convention to name a one-dimensional variable with the same name as its dimension to signify it as a "coordinate" or independent variable other variables are defined on. In TileDB, a variable and dimension in the same array cannot have the same name. This is handled by using the `.index` and `.data` suffixes.

* **Unlimited Dimensions**: TileDB can support unlimited dimensions by using fill values, sparse arrays, or nullable attributes. However, for dataset consisting of multiple attributes stored in multiple arrays, it may be cumbersome to determine the current "size" (maximum coordinate data has been added for) of the unlimited dimension. Storing attributes defined on the same dimensions in the same array helps partially mitigate this issue.

* **Compound data types**: As of TileDB version 2.2, compound data types are not directly supported in TileDB. Compound data types can be broken into their constituent parts; however, this breaks storage locality (TileDB attributes are stored in a [columnar format](https://docs.tiledb.com/main/basic-concepts/data-format)). Variable, opaque, and string data types are supported.


NetCDF Conversion Quick Start
-----------------------------

TODO


Command-Line Interface
----------------------

TileDB-CF provides a command line interface to the NetCDF converter engine. It contains the following options:

.. code:: bash

    Usage: tiledb-cf netcdf-convert [OPTIONS]

        Converts a NetCDF input file to nested TileDB groups.

    Options:
        -i, --input-file TEXT           The path or URI to the NetCDF file that will be converted.  [required]

        -o, --output-uri TEXT           The URI for the output TileDB group. [required]

        --input-group-path TEXT         The path in the input NetCDF for the root group that will be converted.  [default: /]

        --recursive / --no-recursive    Recursively convert all groups contained in the input group path.  [default: True]

        -k, --output-key TEXT           Key for the generated TileDB arrays.

        --unlimited-dim-size INTEGER    Size to convert unlimited dimensions to. [default: 10000]

        --dim-dtype [int8|int16|int32|int64|uint8|uint16|uint32|uint64]
                                  The data type for TileDB dimensions created from converted NetCDF.  [default: uint64]

        --help                          Show this message and exit.


Xarray Support
==============

The TileDB-CF package provides an optional interface to xarray. To install

Reading from TileDB with Xarray
-------------------------------

Xarray uses a plugin infrastructure that allows third-party developers to create their own backend engines for reading and writing. TileDB-CF contains one such backend. To use the backend, make sure `tiledb-cf` is installed in your current Python environment, and use the `tiledb` engine:

.. code:: python

    import xarray as xr

    xr.open_dataset(tiledb_uri, engine="tiledb")

The TileDB engine can be used to open either a TileDB array or a TileDB group. See the requirements on the arrays below.

The backend engine will open the group or array as a dataset with TileDB dimensions mapping to dataset dimensions, TileDB attributes mapping to dataset variables/DataArrays, and TileDB metadata mapping to dataset attributes.


For a TileDB array to be readable by xarray, the following must be satisfied:

* The array must be dense.
* All dimensions on the array must be either signed or unsigned integers.
* Add dimensions must have a domain that starts at ``0``.

For a TileDB group to be readable by xarray, the following must be satisfied:

* All arrays in the group satisfy the above requirements for the array to be readable.
* Each attribute has a unique "variable name".

Special TileDB Keyword Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Writing from Xarray to TileDB
-----------------------------

There are two sets of functions for writing to xarray:

1. ``from_xarray``

    * Useful when copying an entire xarray dataset to a TileDB group in a single function call.
    * Creates the group and copies all data and metadata to the new group.

2. ``create_group_from_xarray``, ``copy_data_from_xarray``, ``copy_metadata_from_xarray``:

    * Useful when copying multiple xarray datasets to a single TileDB group.
    * Creates the group and copies data to the group in separate API calls.

The xarray writer is stricter than the xarray backend engine (reader). While the reader will attempt to open arrays with multiple attributes, the xarray writer only creates arrays with one attribute per name.


TileDB Encoding
^^^^^^^^^^^^^^^

TODO

Region to Write
^^^^^^^^^^^^^^^

TODO

Creating Multiple Fragments
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When copying data with either the ``from_xarray`` or ``copy_data_from_xarray`` functions, the copy routine will use Xarray chunks for separate writes - creating multiple fragments.

About fragments: TODO

Handling Coordinates
^^^^^^^^^^^^^^^^^^^^

TODO

**Writer Restrictions**

The xarray writer has all the restrictions of the xarray backend. In addition, the writer will only write to groups where each array has exactly one attribute.
