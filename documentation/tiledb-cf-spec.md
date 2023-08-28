---
title: TileDB-CF Dataspace
---

::: {.warning}
The current dataspace specification is not stable. Backwards compatibility is not guaranteed for specification less than 1.0.0.
:::

## Current TileDB-CF Dataspace Specification

* The current TileDB-CF format version number is **0.2.0**.

### TileDB-CF Dataspace 0.2.0

A TileDB CF dataspace is a TileDB group with arrays, attributes, and dimensions that satisfy the following rules.

#### Terminology

* **Dataspace name**: The name of an attribute or dimension stripped of an optional suffix of `.index` or `.data`.
* **Collection of dimensions**: A set of TileDB dimensions with the same name, data type, and domain.

#### CF Dataspace

A CF Dataspace is a TileDB group that follows certain requirements in order to provide additional relational context to dimensions and attributes using naming conventions. In a CF Dataspace, TileDB attributes within the entire group are unique and TileDB dimensions that share the same name are considered the same object.

**Requirements for Attributes and Dimensions**

1. All attributes and dimension must be named (there must not be any anonymous attributes or dimensions).
2. All dimensions that share a name must belong to the same collection (they must have the same domain and data type).
3. All attributes must have a unique dataspace name.

**Requirements for Metadata**

1. Group metadata is stored in a special metadata array named `__tiledb_group` inside the TileDB group.
2. Attribute metadata is stored in the same array the attribute is stored in. The metadata key must use the prefix `__tiledb_attr.{attr_name}.` where `{attr_name}` is the full name of the attribute.
3. If the metadata key `_FillValue` exists for an attribute; it must have the same value as the fill value for the attribute.

### Simple CF Dataspace

A simple CF dataspace is a direct implementation of the NetCDF data model in TileDB. It follows the same rules as a CF dataspace along with the following requirements:

**Additional Requirements for Dimensions**

1. All dimensions use integer indices and have a domain with lower bound of 0.
2. All collections of dimensions must have a unique dataspace name.


## Specification Q&A

1. Why have a special specification for the TileDB-CF library?

    The TileDB data model is very general and can be used to support a wide-range of applications. However, there is always a push-and-pull between how general your data model is and enabling specific behavior or interpretations for the data. The purpose of the TileDB-CF specification is to handle the case where we have multiple TileDB arrays defined on the same underlying dimensions. By creating a specificiation we make our assumptions explicit and let users know exactly what they must do to use this tool.


2. Is the specification backwards compatible?

    Not yet. This library and data model are still under initial development. When the data model has stabalized we will release a 1.0.0 version.

3. Why is there both a library version and a specification version?

    The TileDB-CF python package will update much more frequently the specification. The specification is more-or-less just a summary of the conventions the TileDB-CF library is using. As such, a change to the specification version will always coincide to a change to the library version, but the library version can update without effecting the specification.

4. What version is my current data?

    The TileDB-CF dataspace specification is fairly minimal. Your data may satisfy multiple versions. Currently, we do not provide support for checking your data satisfies the TileDB-CF dataspace convention, but some such tooling will be implemented before the 1.0.0 release of this specification.


## Changelog

### Version 0.1.0

- Initial release. See :ref:`cf-spec-0.1.0`


## Appendix

### TileDB-CF Dataspace 0.1.0

#### Terminology

* **Index dimension**: A TileDB dimension with an integer data type and domain with `0` as its lower bound.
* **Data dimension**: Any TileDB dimension that is not an index dimension.
* **Dataspace name**: The name of an attribute or dimension stripped of an optional suffix of `.index` or `.data`.

#### CF Dataspace

**Requirements for Attributes and Dimensions**

1. All attributes and dimension must be named (there must not be any anonymous attributes or dimensions).
2. All dimensions that share a name must have the same domain and data type.
3. All attributes must have a unique dataspace name.
4. If an attribute and data dimension share the same dataspace name, they must share the same full name and data type.

**Requirements for Metadata**

1. Group metadata is stored in a special metadata array named `__tiledb_group` inside the TileDB group.
2. Attribute metadata is stored in the array the attribute is in using the prefix `__tiledb_attr.{attr_name}.` for the attribute key where `{attr_name}` is the full name of the attribute.
3. If the metadata key `_FillValue` exists for an attribute; it must have the same value as the fill value for the attribute.

#### Indexable CF Dataspace

A CF Dataspace is said to be indexable if it satisfies all requirements of a CF Dataspace along with the following condition:

* All data dimensions must have an axis label that maps an index dimension with the same dataspace name as the data dimension to an attribute with the same full name and data type as the data dimension.
