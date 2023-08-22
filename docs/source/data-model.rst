
***********
Data Model
***********

Introduction
############

[TileDB](https://github.com/TileDB-Inc/TileDB) is a powerful open-source engine for storing and accessing dense and sparse multi-dimensional arrays. The objective of this document is to define the TileDB _CF Dataspace_ specification: a specification for TileDB groups that is compatible with the [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) data model using the [CF Metadata Conventions](http://cfconventions.org).

NetCDF and TileDB use over lapping terminology to refer to concepts in their respective data model. A complete description of the TileDB data model can be found at the [TileDB website](https://docs.tiledb.com/main/basic-concepts/data-model). A complete description of the NetCDF data model can be found at the [UCAR website](https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_data_model.html).

TileDB Data Model
#################

TileDB stores data as dense or sparse multi-dimensional arrays. An array (either dense or sparse) consists of:

* **Dimensions**: The dimensions along with their domains orient a multi-dimensional space of cells. A dimension is defined by its name, domain, and data type along with additional data that specifies data storage and compression. A tuple of dimension values is called the **cell coordinates**. There can be any number of dimensions in an array.
* **Attributes**: In each cell in the logical layout, TileDB stores a tuple comprised of any number of attributes, each of any data type (fixed- or variable-sized).
* **Array metadata**: This is (typically small) key-value data associated with an array.
* **Axes labels**: These are practically other (dense or sparse) arrays attached to each dimension, which facilitate slicing multi-dimensional ranges on conditions other than array positional indices.

Multiple arrays can be stored together in a **TileDB group**.


Comparison: NetCDF Data Model
-----------------------------

The NetCDF data model is a common choice for multi-dimensional data. 


A NetCDF file consists of **groups**, **dimensions**, **variables**, and **attributes**. Each NetCDF file has at least one root group that contains all other objects. Additional subgroups can be added to heirarchically organize the data.

* **Dimensions**: A dimension is a name-size pair that describes an axis of a multi-dimension array. The size of the dimension may be "unlimited" (allowed to grow).
* **Variables**: A variable is a multi-dimensional array with a NetCDF dimension associated to each axis of the array. The size of the dimensions must match the shape of the multi-dimensional array.
* **Attribute**: An attribute is a key-value pair that is associated with either a group or variable. Attributes are used to store (typically small) metadata.

