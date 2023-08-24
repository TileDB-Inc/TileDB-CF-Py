.. _overview:


******************
TileDB-CF Overview
******************

About
=====

Current compoents include:

* 

Installation
============

This project is available from [PyPI](https://pypi.org/project/tiledb-cf/) and may be installed with ``pip``:

.. code:: bash

    pip install tiledb-cf

TileDB-CF contains optional features that will be enabled if the required python packages are included in the python environment it is called from:

* 


To install tiledb-cf with additional dependencies use:

.. code:: bash

    pip install tiledb-cf[<optional dependecies>]

For example, to install xarray with dask support and the NetCDF converter engine use:

.. code:: bash

    pip install tiledb-cf[xarray,distributed,netcdf4]


TileDB Data Model
=================

[TileDB](https://github.com/TileDB-Inc/TileDB) is a powerful open-source engine for storing and accessing dense and sparse multi-dimensional arrays.  A complete description of the TileDB data model can be found at the [TileDB website](https://docs.tiledb.com/main/basic-concepts/data-model).

TileDB stores data as dense or sparse multi-dimensional arrays. The arrays can be grouped together in TileDB groups. A brief summary:

* **Group**: A group is a TileDB object that stores metadata, arrays, and other groups. The groups use URIs to track members, so multiple groups can store the same assets.

* **Array**: A set of attributes and dimensions that can be queried together:

    * **Dimensions**: The dimensions along with their domains orient a multi-dimensional space of cells. A dimension is defined by its name, domain, and data type along with additional data that specifies data storage and compression. A tuple of dimension values is called the **cell coordinates**. There can be any number of dimensions in an array.

    * **Attributes**: In each cell in the logical layout, TileDB stores a tuple comprised of any number of attributes, each of any data type (fixed- or variable-sized).

* **Metadata**: This is (typically small) key-value data associated with an array or a group.

* **Dimension labels** (experimental): Dimension labels store either increasing of decreasing data in a one-dimensional TileDB array that can be used to indirectly query other dimensions. 

 

**Comparison: NetCDF Data Model**

The NetCDF data model is a common choice for multi-dimensional data, especially in the climate and weather space. NetCDF and TileDB use over lapping terminology to refer to concepts in their respective data model.

 A complete description of the NetCDF data model can be found at the [UCAR website](https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_data_model.html).

A NetCDF file consists of **groups**, **dimensions**, **variables**, and **attributes**. Each NetCDF file has at least one root group that contains all other objects. Additional subgroups can be added to heirarchically organize the data.

* **Dimensions**: A dimension is a name-size pair that describes an axis of a multi-dimension array. The size of the dimension may be "unlimited" (allowed to grow). The NetCDF dimension is roughly ananlogous to a TileDB dimension.

* **Variables**: A variable is a multi-dimensional array with a NetCDF dimension associated to each axis of the array. The size of the dimensions must match the shape of the multi-dimensional array. A NetCDF variable is roughly equivalent to a TileDB attribute.

* **Attribute**: An attribute is a key-value pair that is associated with either a group or variable. Attributes are used to store (typically small) metadata. NetCDF attributes are roughly equivalent to TileDB metadata.

* **Group**: A NetCDF group is a collection of dimensions, variables, and attributes. A simple NetCDF group might map to a TileDB array. A more complex group would need to be mapped to a TileDB group.


