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
    from ._array_converters import NetCDF4ArrayConverter, NetCDF4DomainConverter
    from ._attr_converters import NetCDF4VarToAttrConverter
    from ._dim_converters import (
        NetCDF4CoordToDimConverter,
        NetCDF4DimToDimConverter,
        NetCDF4ScalarToDimConverter,
        NetCDF4ToDimConverter,
    )
    from ._utils import open_netcdf_group
    from .converter import NetCDF4ConverterEngine

    __all__.append("NetCDF4ConverterEngine")
