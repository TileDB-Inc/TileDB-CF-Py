.. _overview:

******************
TileDB-CF Overview
******************

.. warning::
    The TileDB-CF library is still under initial development and changes may not be backward compatible.

About
=====

TileDB-CF is a python package intended to aid in modeling and analyzing complex multi-dimensional data in TileDB. It currently contains the following components:

* **Core**: High-level API for common TileDB group and metadata actions.

* **Creator**: Support for generating a group following the TileDB-CF dataspace specification.

* **NetCDF Engine**: Support for creating a TileDB group or array from NetCDF data and copying the data into the new group or array.

* **Xarray Engine**:

  - Backend engine that can be used with `xarray.open_dataset`.
  - Support for create a TileDB from an xarray dataset and copying the data into the new group.


Installation
============

This project is available from `PyPI`_ and may be installed with ``pip``:

.. code:: bash

    pip install tiledb-cf

TileDB-CF contains optional features that will be enabled if the required python packages are included in the python environment. These include:

* ``docs``: support for generating documentation with sphinx,
* ``netCDF4``: support for the NetCDF engine,
* ``xarray``: support for the xarray engine,
* ``parallel``: support for dask operations (used with the xarray engine),
* ``examples``: additional packages needed to run the example notebooks.


To install tiledb-cf with additional dependencies use:

.. code:: bash

    pip install tiledb-cf[<optional dependecies>]

For example, to install xarray with dask support and the NetCDF converter engine use:

.. code:: bash

    pip install tiledb-cf[xarray,parallel,netCDF4]


.. _PyPI: https://pypi.org/project/tiledb-cf

TileDB Data Model
=================

`TileDB`_ is a powerful open-source engine for storing and accessing dense and sparse multi-dimensional arrays.  A complete description of the TileDB data model can be found at the `TileDB website`_.

TileDB stores data as dense or sparse multi-dimensional arrays. The arrays can be grouped together in TileDB groups. A brief summary:

* **Group**: A group is a TileDB object that stores metadata, arrays, and other groups. The groups use URIs to track members, so multiple groups can store the same assets.

* **Array**: A set of attributes and dimensions that can be queried together:

    * **Dimensions**: The dimensions along with their domains orient a multi-dimensional space of cells. A dimension is defined by its name, domain, and data type along with additional data that specifies data storage and compression. The dimension values is called the cell coordinates. There can be any number of dimensions in an array.

    * **Attributes**: In each cell in the logical layout, TileDB stores a tuple comprised of any number of attributes, each of any data type (fixed- or variable-sized).

* **Metadata**: This is (typically small) key-value data associated with an array or a group.

* **Dimension labels** (experimental): Dimension labels store either increasing of decreasing data in a one-dimensional TileDB array that can be used to indirectly query other dimensions.


.. _TileDB: https://github.com/TileDB-Inc/TileDB

.. _TileDB website: https://docs.tiledb.com/


NetCDF Converter Engine
=======================

NetCDF Data Model
-----------------
The NetCDF data model is a common choice for multi-dimensional data, especially in the climate and weather space. NetCDF and TileDB use over lapping terminology to refer to concepts in their respective data model.

 A complete description of the NetCDF data model can be found at the `UCAR website`_.

A NetCDF file consists of **groups**, **dimensions**, **variables**, and **attributes**. Each NetCDF file has at least one root group that contains all other objects. Additional subgroups can be added to heirarchically organize the data.

* **Dimensions**: A dimension is a name-size pair that describes an axis of a multi-dimension array. The size of the dimension may be "unlimited" (allowed to grow). The NetCDF dimension is roughly ananlogous to a TileDB dimension in a dense TileDB array.

* **Variables**: A variable is a multi-dimensional array with a NetCDF dimension associated to each axis of the array. The size of the dimensions must match the shape of the multi-dimensional array. A NetCDF variable is roughly equivalent to a TileDB attribute in a sparse or dense TileDB array or a TileDB dimension in a sparse TileDB array.

* **Attribute**: An attribute is a key-value pair that is associated with either a group or variable. Attributes are used to store (typically small) metadata. NetCDF attributes are roughly equivalent to TileDB metadata.

* **Group**: A NetCDF group is a collection of dimensions, variables, and attributes. A simple NetCDF group might map to a TileDB array. A more complex group would need to be mapped to a TileDB group.



.. _UCAR website: https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_data_model.html

NetCDF-to-TileDB Compatibility
------------------------------

The TileDB-CF package provides an interface for generating TileDB groups from NetCDF datasets using the TileDB CF Dataspace convention. The CF Dataspace model supports the classic NetCDF-4 data model by mapping:

* NetCDF groups to TileDB groups;
* NetCDF dimensions to TileDB dimensions;
* NetCDF variables to TileDB attributes or TileDB dimensions;
* NetCDF attributes to TileDB group or array metadata.

Some features and use cases do not directly transfer or may need to be modified before use in TileDB.

* **Coordinates**: In NetCDF, it is a common convention to name a one-dimensional variable with the same name as its dimension to signify it as a "coordinate" or independent variable other variables are defined on. In TileDB, a variable and dimension in the same array cannot have the same name. This can be handled by renaming either the dimension or the variable when copying to TileDB.

* **Unlimited Dimensions**: TileDB can support unlimited dimensions by creating the domain on a dimension larger than the initial data. The domain must be set at creation time, and cannot be modified after array creation.

* **Compound data types**: As of TileDB version 2.16, compound data types are not directly supported in TileDB. Compound data types can be broken into their constituent parts; however, this breaks storage locality. Variable, opaque, and string data types are supported.


Programmatic Interface
----------------------

The ``NetCDFConverterEngine`` is a configurable tool for ingesting data from NetCDF into TileDB. The class can be manually constructed, or it can be auto-generated from a NetCDF file or group.


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

The TileDB backend engine can be used with the standard xarray keyword arguments. It supports the additional TileDB-specific arguments:

* TODO


Writing from Xarray to TileDB
-----------------------------

The xarray writer is stricter than the xarray backend engine (reader). While the reader will attempt to open arrays with multiple attributes, the xarray writer only creates arrays with one attribute per name.

There are two sets of functions for writing to xarray:

1. Single dataset ingestion.

    * Functions used: ``from_xarray``
    * Useful when copying an entire xarray dataset to a TileDB group in a single function call.
    * Creates the group and copies all data and metadata to the new group.

2. Multi-dataset ingestion.

    * Main functions: ``create_group_from_xarray`` and ``copy_data_from_xarray``.
    * Additional helper function: ``copy_metadata_from_xarray``.
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

  - All dimensions have the same datatype determined by the ``dim_dtype`` encoding.

  - The dimension names in the TileDB array match the dimension names in the xarray variable.

  - The dimension tiles are determined by the ``tiles`` encoding.

  - The domain of each dimension is set to ``[0, max_size - 1]`` where ``max_size`` is computed as follows:

    1. Use the corresponding element of the  ``max_shape`` encoding if provided.

    2. If the ``max_shape`` encoding is not provided and the xarray dimension is "unlimited", use the largest possible size for this integer type.

    3. If the ``max_shape`` encoding is not provided and the xarray dimension is not "unlimited", use the size of the xarray dimension.

* TileDB Attribute:

  - The attribute datatype is the same as the variable datatype (after applying xarray encodings).

  - The attribute name is set using the following:

    1. Use the name provided by ``attr_name`` encoding.

    2. If the ``attr_name`` encoding is not provided and there is no dimension on this variable with the same name as the variable, use the name of the variable.

    3. If the ``attr_name`` encoding is provided and there is a dimension on this variable with the same name as the variable, use the variable name appended with `_`.

  - The attribute filters are determined by the ``filters`` encoding.


TileDB Encoding
^^^^^^^^^^^^^^^

The writer takes a dictionary from dataset variable names to a dictionary of encodings for setting TileDB properties. The possible encoding keywords are provided in the table below.

+------------------+-----------------------------------------------+--------------------+
| Encoding Keyword | Details                                       | Type               |
+==================+===============================================+====================+
| ``attr_name``    | Name to use for the TileDB attribute.         | str                |
+------------------+-----------------------------------------------+--------------------+
| ``filters``      | Filter list to apply to the TileDB attribute. | tiledb.FilterList  |
+------------------+-----------------------------------------------+--------------------+
| ``tiles``        | Tile sizes to apply to the TileDB dimensions. | tuple of ints      |
+------------------+-----------------------------------------------+--------------------+
| ``max_shape``    | Maximum possible size of the TileDB array.    | tuple of ints      |
+------------------+-----------------------------------------------+--------------------+
| ``dim_dtype``    | Datatype to use for the TileDB dimensions.    | str or numpy.dtype |
+------------------+-----------------------------------------------+--------------------+


Region to Write
^^^^^^^^^^^^^^^

If the creating TileDB array's with either unlimited dimensions or with encoded ``max_shape`` larger than the current size of the xarray variable, then the region to write the data to needs to be provided. This is input as a dictionary from dimension names to slices. The slice uses xarray/numpy conventions and will write to a region that does **not** include the upper bound of the slice.


Creating Multiple Fragments
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When copying data with either the ``from_xarray`` or ``copy_data_from_xarray`` functions, the copy routine will use Xarray chunks for separate writes - creating multiple fragments.
