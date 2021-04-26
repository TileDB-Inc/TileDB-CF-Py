# TileDB CF Specification

:information_source: **Notes:**

* The current TileDB-CF format version number is **0.1.0**.

## Introduction

[TileDB](https://github.com/TileDB-Inc/TileDB) is a powerful open-source engine for storing and accessing dense and sparse multi-dimensional arrays. The objective of this document is to define the TileDB _CF Dataspace_ specification: a specification for TileDB groups that is compatible with the [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) data model using the [CF Metadata Conventions](http://cfconventions.org/latest.html).

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

## TileDB CF Dataspace Specification

A TileDB CF dataspace is a TileDB group with arrays, attributes, and dimensions that satisfy the following rules.

### Terminology

* **Index dimension**: A TileDB dimension with an integer data type and domain with `0` as its lower bound.
* **Data dimension**: Any TileDB dimension that is not an index dimension.
* **Dataspace name**: The name of an attribute or dimension stripped of an optional suffix of `.index` or `.data`.

#### Requirements for Dimensions

1. All dimension must be named (no anonymous dimensions).
2. All dimensions that share a name must have the same domain and data type.
3. An index dimension may optionally end in the suffix `.index`.
4. A data dimension may optionally end in the suffix `.data`.
5. All data dimensions must have an axis label that maps an index dimension with the same dataspace name as the data dimension to an attribute with the same full name and data type as the data dimension.

#### Requirements for Attributes

1. All attributes must be named (no anonymous attributes).
2. All attributes must have unique dataspace names.
3. An index dimension may optionally end in the suffix `.index`.
4. Attributes may optionally end in the suffix `.index` or `.data`.

#### Requirements for Metadata

1. Group metadata is stored in a special metadata array named `__tiledb_group` inside the TileDB group.
2. Attribute metadata is stored in the array the attribute is in using the prefix `__tiledb_attr.{attr_name}.` for the attribute key where `{attr_name}` is the full name of the attribute.
3. If the metadata key `_FillValue` exists for an attribute; it must have the same value as the fill value for the attribute.

## Compatibility with the NetCDF Data Model

The CF Dataspace model supports the NetCDF data model by mapping:

* NetCDF groups to TileDB groups;
* NetCDF dimensions to TileDB dimensions;
* NetCDF variables to TileDB attributes;
* NetCDF attributes to TileDB array metadata.

The conversion from NetCDF to TileDB is one-to-many. This is to allow flexibility when picking an appropriate TileDB layout that will still be able to interface with tooling that uses the NetCDF data model. For example, NetCDF variables that share the same NetCDF dimensions can be split into multiple TileDB arrays or combined into a single TileDB array.

### Direct Conversion of a NetCDF File to TileDB

This is a suggestion on how to convert a NetCDF file into a collection of TileDB CF Dataspaces in a way that preserves the NetCDF data model. Alternative conversion schemes may be appropriate depending on factors including the sparsity of the data, planned reading/writing access, and plans to combine multiple datasets. In some cases, it may be worth moving away from the NetCDF data model entirely, and refactoring the data storage to something that is more natural for its intended use case. This is especially true of datasets that consist of large amounts of sparse data.

1. Recursively map each NetCDF group to a TileDB CF Dataspace preserving group hierarchy.
2. In each NetCDF group, collect variables in a group defined on the same dimensions into an arbitrarily named array defined using the following.
    * Each NetCDF dimension is mapped to a TileDB dimension with an integer data type and domain `[0, size-1]` where size is the NetCDF dimension size for standard dimensions and a sufficiently large value for unlimited dimensions.
    * Each variable is mapped to an attribute in the array. If one of the dimensions is unlimited, make sure at least one of the following is true: each attribute is nullable; each attribute has an appropriate fill value; or the array is sparse. If the variable has the same name as one of the dimensions the variable is defined on, add the suffix `.data` to the TileDB attribute.
3. Set tile sizes, sparse/dense, filters, and other TileDB properties appropriately based on planned data access.

### Read TileDB CF Dataspaces into the NetCDF Data Model

This is a suggestion on how to convert a collection of TileDB CF Dataspaces from TileDB to a tool that uses a NetCDF data model for its data storage or backend. It can be modified as appropriate to work best with a particular tool.

1. Recursively map each TileDB CF Dataspace to a NetCDF group preserving group hierachry.
2. Map each index dimension in TileDB to a NetCDF dimension where:
    * the name and size of the NetCDF dimension is the dataspace name and size of the TileDB dimension.
3. Map each attribute to a variable where:
    * the dimension names of the NetCDF variable is the dimension dataspace names of the array the TileDB attribute is stored in.

### Supported NetCDF Features

The TileDB CF Dataspace fully supports the classic NetCDF Data Model and most features in the NetCDF-4 data model. Some features and use cases do not directly transfer or may need to be modified before use in TileDB.

* **Unlimited Dimensions**: TileDB can support unlimited dimensions by using fill values, sparse arrays, or nullable attributes. However, for dataset consisting of multiple attributes stored in multiple arrays, it may be difficult to determine the current size of the unlimited dimension.
* **Coordinates**: In NetCDF, it is a common convention to name a variable with the same name as the dimension it is defined on to signify it as a "coordinate" or independent variable other variables are defined on. In TileDB, a variable and dimension in the same array cannot have the same name. This is handle by using the `.index` and `.data` suffixes.
* **Compound data types**: As of TileDB version 2.2, compound data types are not directly supported in TileDB. Compound data types can be broken into their constituent parts; however, this breaks storage locality (TileDB attributes are stored in a [columnar format](https://docs.tiledb.com/main/basic-concepts/data-format)). Variable, opaque, and string data types are supported.