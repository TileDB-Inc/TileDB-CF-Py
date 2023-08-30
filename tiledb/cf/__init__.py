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
    ArrayMetadata,
    AttrMetadata,
    DataspaceCreator,
    DimMetadata,
    create_group,
    open_group_array,
)
from .netcdf_engine import from_netcdf, has_netCDF4
from .xarray_engine import (
    copy_data_from_xarray,
    copy_metadata_from_xarray,
    create_group_from_xarray,
    from_xarray,
    has_xarray,
)

if has_netCDF4:
    from .netcdf_engine import NetCDF4ConverterEngine
