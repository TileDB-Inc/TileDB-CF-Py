# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
"""``tiledb.cf`` is the core module for the TileDB-CF-Py library.

This module contains core classes and functions for supporting the NetCDF data model in
the `TileDB storage engine <https://github.com/TileDB-Inc/TileDB>`_. To use this module
simply import using:

.. code-block:: python

    import tiledb.cf
"""

from .cli import cli
from .core import (
    ATTR_METADATA_FLAG,
    DIM_METADATA_FLAG,
    METADATA_ARRAY_NAME,
    ArrayMetadata,
    AttrMetadata,
    DimMetadata,
    Group,
    GroupSchema,
    VirtualGroup,
)
from .creator import DATA_SUFFIX, INDEX_SUFFIX, DataspaceCreator, dataspace_name
from .netcdf_engine import from_netcdf, has_netCDF4
from .xarray_engine import has_xarray

if has_netCDF4:
    from .netcdf_engine import NetCDF4ConverterEngine
