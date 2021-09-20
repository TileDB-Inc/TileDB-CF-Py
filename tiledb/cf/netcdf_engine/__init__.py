# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.

try:
    import netCDF4

    has_netCDF4 = True

except ImportError:
    has_netCDF4 = False

from .api import from_netcdf

__all__ = ["has_netCDF4", "from_netcdf"]  # type: ignore

if has_netCDF4:
    from .converter import (
        NetCDF4ArrayConverter,
        NetCDF4ConverterEngine,
        NetCDF4CoordToDimConverter,
        NetCDF4DimToDimConverter,
        NetCDF4DomainConverter,
        NetCDF4ScalarToDimConverter,
        NetCDF4VarToAttrConverter,
        open_netcdf_group,
    )

    __all__.append("NetCDF4ConverterEngine")
