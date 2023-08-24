# TileDB CF Dataspace

:information_source: **Notes:**

* The current TileDB-CF format version number is **0.2.0**.

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