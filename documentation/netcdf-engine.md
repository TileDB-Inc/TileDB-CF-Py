---
title: TileDB-CF NetCDF Engine
---

## NetCDF Data Model
The NetCDF data model is a common choice for multi-dimensional data, especially in the climate and weather space. NetCDF and TileDB use over lapping terminology to refer to concepts in their respective data model.

A complete description of the NetCDF data model can be found at the [UCAR website](https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_data_model.html).

A NetCDF file consists of **groups**, **dimensions**, **variables**, and **attributes**. Each NetCDF file has at least one root group that contains all other objects. Additional subgroups can be added to heirarchically organize the data.

* **Dimensions**: A dimension is a name-size pair that describes an axis of a multi-dimension array. The size of the dimension may be "unlimited" (allowed to grow). The NetCDF dimension is roughly ananlogous to a TileDB dimension in a dense TileDB array.

* **Variables**: A variable is a multi-dimensional array with a NetCDF dimension associated to each axis of the array. The size of the dimensions must match the shape of the multi-dimensional array. A NetCDF variable is roughly equivalent to either a TileDB attribute in a sparse or dense TileDB array or a TileDB dimension in a sparse TileDB array.

* **Attribute**: An attribute is a key-value pair that is associated with either a group or variable. Attributes are used to store (typically small) metadata. NetCDF attributes are roughly equivalent to TileDB metadata.

* **Group**: A NetCDF group is a collection of dimensions, variables, and attributes. A simple NetCDF group might map to a TileDB array. A more complex group would need to be mapped to a TileDB group.


## NetCDF-to-TileDB Compatibility

The TileDB-CF package provides an interface for generating TileDB groups from NetCDF datasets using the TileDB-CF Dataspace convention. The CF Dataspace model supports the classic NetCDF-4 data model by mapping:

* NetCDF groups to TileDB groups;
* NetCDF dimensions to TileDB dimensions;
* NetCDF variables to TileDB attributes or TileDB dimensions;
* NetCDF attributes to TileDB group or array metadata.

Some features and use cases do not directly transfer or may need to be modified before use in TileDB.

* **Coordinates**: In NetCDF, it is a common convention to name a one-dimensional variable with the same name as its dimension to signify it as a "coordinate" or independent variable other variables are defined on. In TileDB, a variable and dimension in the same array cannot have the same name. This can be handled by renaming either the dimension or the variable when copying to TileDB.

* **Unlimited Dimensions**: TileDB can support unlimited dimensions by creating the domain on a dimension larger than the initial data. The domain must be set at creation time, and cannot be modified after array creation.

* **Compound data types**: As of TileDB version 2.16, compound data types are not directly supported in TileDB. Compound data types can be broken into their constituent parts; however, this breaks storage locality. Variable, opaque, and string data types are supported.


## Programmatic Interface

The `NetCDFConverterEngine` is a configurable tool for ingesting data from NetCDF into TileDB. The class can be manually constructed, or it can be auto-generated from a NetCDF file or group.

## Command-Line Interface

TileDB-CF provides a command line interface to the NetCDF converter engine. It contains the following options:

```bash
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
```
