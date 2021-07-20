# TileDB CF Dataspace

:information_source: **Notes:**

* The current TileDB-CF format version number is **0.2.0**.

## Introduction

[TileDB](https://github.com/TileDB-Inc/TileDB) is a powerful open-source engine for storing and accessing dense and sparse multi-dimensional arrays. The objective of this document is to define the TileDB _CF Dataspace_ specification: a specification for TileDB groups that is compatible with the [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) data model using the [CF Metadata Conventions](http://cfconventions.org).

NetCDF and TileDB use over lapping terminology to refer to concepts in their respective data model. A complete description of the TileDB data model can be found at the [TileDB website](https://docs.tiledb.com/main/basic-concepts/data-model). A complete description of the NetCDF data model can be found at the [UCAR website](https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_data_model.html).

### TileDB Data Model

TileDB stores data as dense or sparse multi-dimensional arrays. An array (either dense or sparse) consists of:

* **Dimensions**: The dimensions along with their domains orient a multi-dimensional space of cells. A dimension is defined by its name, domain, and data type along with additional data that specifies data storage and compression. A tuple of dimension values is called the **cell coordinates**. There can be any number of dimensions in an array.
* **Attributes**: In each cell in the logical layout, TileDB stores a tuple comprised of any number of attributes, each of any data type (fixed- or variable-sized).
* **Array metadata**: This is (typically small) key-value data associated with an array.
* **Axes labels**: These are practically other (dense or sparse) arrays attached to each dimension, which facilitate slicing multi-dimensional ranges on conditions other than array positional indices.

Multiple arrays can be stored together in a **TileDB group**.

#### NetCDF Data Model

A NetCDF file consists of **groups**, **dimensions**, **variables**, and **attributes**. Each NetCDF file has at least one root group that contains all other objects. Additional subgroups can be added to heirarchically organize the data.

* **Dimensions**: A dimension is a name-size pair that describes an axis of a multi-dimension array. The size of the dimension may be "unlimited" (allowed to grow).
* **Variables**: A variable is a multi-dimensional array with a NetCDF dimension associated to each axis of the array. The size of the dimensions must match the shape of the multi-dimensional array.
* **Attribute**: An attribute is a key-value pair that is associated with either a group or variable. Attributes are used to store (typically small) metadata.

## Specification

A TileDB CF dataspace is a TileDB group with arrays, attributes, and dimensions that satisfy the following rules.

### Terminology

* **Dataspace name**: The name of an attribute or dimension stripped of an optional suffix of `.index` or `.data`.
* **Collection of dimensions**: A set of TileDB dimensions with the same name, data type, and domain.

### CF Dataspace

A CF Dataspace is a TileDB group that follows certain requirements in order to provide additional relational context to dimensions and attributes using naming conventions. In a CF Dataspace, TileDB attributes within the entire group are unique and TileDB dimensions that share the same name are considered the same object.

#### Requirements for Attributes and Dimensions

1. All attributes and dimension must be named (there must not be any anonymous attributes or dimensions).
2. All dimensions that share a name must belong to the same collection (they must have the same domain and data type).
3. All attributes must have a unique dataspace name.

#### Requirements for Metadata

1. Group metadata is stored in a special metadata array named `__tiledb_group` inside the TileDB group.
2. Attribute metadata is stored in the same array the attribute is stored in. The metadata key must use the prefix `__tiledb_attr.{attr_name}.` where `{attr_name}` is the full name of the attribute.
3. If the metadata key `_FillValue` exists for an attribute; it must have the same value as the fill value for the attribute.

### Simple CF Dataspace

A simple CF dataspace is a direct implementation of the NetCDF data model in TileDB. It follows the same rules as a CF dataspace along with the following requirements:

#### Additional Requirements for Dimensions

1. All dimensions use integer indices and have a domain with lower bound of 0.
2. All collections of dimensions must have a unique dataspace name.

## Compatibility with the NetCDF Data Model

The CF Dataspace model supports the NetCDF data model by mapping:

* NetCDF groups to TileDB groups;
* NetCDF dimensions to TileDB dimensions;
* NetCDF variables to TileDB attributes or TileDB dimensions;
* NetCDF attributes to TileDB array metadata.

### Read a Simple TileDB CF Dataspaces into the NetCDF Data Model

A simple TileDB CF Dataspace maps directly to the NetCDF Data Model.

To read TileDB dimensions as NetCDF dimensions:

* Use the dataspace name for the NetCDF dimension.
* Treat TileDB dimensions with the same name as the same NetCDF dimension.

To read TileDB attributes as NetCDF variables:

* Use the dataspace name for the NetCDF variable.

To read TileDB metadata as NetCDF attributes:

* Assign metadata in the group metadata array as group attributes.
* Assign metadata with the `__tiledb_attr.{attr_name}.` prefix to the TileDB attribute with that name.


### Supported NetCDF Features

The TileDB CF Dataspace fully supports the classic NetCDF Data Model and most features in the NetCDF-4 data model. Some features and use cases do not directly transfer or may need to be modified before use in TileDB.

* **Coordinates**: In NetCDF, it is a common convention to name a one-dimensional variable with the same name as its dimension to signify it as a "coordinate" or independent variable other variables are defined on. In TileDB, a variable and dimension in the same array cannot have the same name. This is handled by using the `.index` and `.data` suffixes.
* **Unlimited Dimensions**: TileDB can support unlimited dimensions by using fill values, sparse arrays, or nullable attributes. However, for dataset consisting of multiple attributes stored in multiple arrays, it may be cumbersome to determine the current "size" (maximum coordinate data has been added for) of the unlimited dimension. Storing attributes defined on the same dimensions in the same array helps partially mitigate this issue.
* **Compound data types**: As of TileDB version 2.2, compound data types are not directly supported in TileDB. Compound data types can be broken into their constituent parts; however, this breaks storage locality (TileDB attributes are stored in a [columnar format](https://docs.tiledb.com/main/basic-concepts/data-format)). Variable, opaque, and string data types are supported.
