# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.

try:
    import xarray

    has_xarray = True

except ImportError:
    has_xarray = False
