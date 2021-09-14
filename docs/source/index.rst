TileDB-CF Python API Reference
##############################

.. warning::

    The TileDB-CF Python library is still initial development and the API may change rapidly.


Group and Metadata Support
##########################
.. automodule:: tiledb.cf
   :noindex:

Groups
======

.. autoclass:: tiledb.cf.Group
  :members:

.. autoclass:: tiledb.cf.VirtualGroup
  :members:

Group Schema
============

.. autoclass:: tiledb.cf.GroupSchema
  :members:

Metadata Wrappers
=================

.. autoclass:: tiledb.cf.ArrayMetadata

.. autoclass:: tiledb.cf.AttrMetadata

Dataspace Creator
#################

Dataspace Creator
=================

.. autoclass:: tiledb.cf.DataspaceCreator
  :members:

Shared Dimension
================

.. autoclass:: tiledb.cf.creator.SharedDim
  :members:

Array Creator
=============

.. autoclass:: tiledb.cf.creator.ArrayCreator
  :members:

Domain Creator
==============

.. autoclass:: tiledb.cf.creator.DomainCreator
  :members:

Dimension Creator
=================

.. autoclass:: tiledb.cf.creator.DimCreator
  :members:

Attribute Creator
=================

.. autoclass:: tiledb.cf.creator.AttrCreator
  :members:


NetCDF4 to TileDB Conversion
############################

Auto-convert Function
=====================

.. autofunction:: tiledb.cf.from_netcdf

NetCDF4 Converter Engine
========================

.. autoclass:: tiledb.cf.engines.netcdf4_engine.NetCDF4ConverterEngine
  :members:

NetCDF4 to TileDB Array Converter
=================================

.. autoclass:: tiledb.cf.engines.netcdf4_engine.NetCDF4ArrayConverter
  :members:

NetCDF4 to TileDB Dimension Converters
=======================================

.. autoclass:: tiledb.cf.engines.netcdf4_engine.NetCDF4CoordToDimConverter
  :members:

.. autoclass:: tiledb.cf.engines.netcdf4_engine.NetCDF4DimToDimConverter
  :members:

.. autoclass:: tiledb.cf.engines.netcdf4_engine.NetCDF4ScalarToDimConverter
  :members:

NetCDF4 to TileDB Attribute Converters
=======================================

.. autoclass:: tiledb.cf.engines.netcdf4_engine.NetCDF4VarToAttrConverter
  :members:

TileDB Backend for xarray
#########################

TODO: Add documentation for the TileDB backend for xarray.
