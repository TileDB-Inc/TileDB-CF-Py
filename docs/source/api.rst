.. _api:

*************
API Reference
*************

.. warning::

    The TileDB-CF Python library is still initial development and the API may change rapidly.


.. automodule:: tiledb.cf
   :noindex:

Core
====

.. autoclass:: tiledb.cf.Group
  :members:

.. autoclass:: tiledb.cf.GroupSchema
  :members:

.. autoclass:: tiledb.cf.ArrayMetadata

.. autoclass:: tiledb.cf.AttrMetadata

.. autoclass:: tiledb.cf.DimMetadata


Dataspace Creator
=================

.. autoclass:: tiledb.cf.DataspaceCreator
  :members:

.. autoclass:: tiledb.cf.creator.SharedDim
  :members:

.. autoclass:: tiledb.cf.creator.ArrayCreator
  :members:

.. autoclass:: tiledb.cf.creator.DomainCreator
  :members:

.. autoclass:: tiledb.cf.creator.DimCreator
  :members:

.. autoclass:: tiledb.cf.creator.AttrCreator
  :members:

NetCDF-to-TileDB
================

.. autofunction:: tiledb.cf.from_netcdf

.. autoclass:: tiledb.cf.NetCDF4ConverterEngine
  :members:

.. autoclass:: tiledb.cf.netcdf_engine.NetCDF4CoordToDimConverter
  :members:

.. autoclass:: tiledb.cf.netcdf_engine.NetCDF4DimToDimConverter
  :members:

.. autoclass:: tiledb.cf.netcdf_engine.NetCDF4ScalarToDimConverter
  :members:

.. autoclass:: tiledb.cf.netcdf_engine.NetCDF4ArrayConverter
  :members:

.. autoclass:: tiledb.cf.netcdf_engine.NetCDF4DomainConverter
  :members:

.. autoclass:: tiledb.cf.netcdf_engine.NetCDF4ToDimConverter

.. autoclass:: tiledb.cf.netcdf_engine.NetCDF4VarToAttrConverter
  :members:

TileDB-xarray
=============

.. autofunction:: tiledb.cf.from_xarray

.. autofunction:: tiledb.cf.create_group_from_xarray

.. autofunction:: tiledb.cf.copy_data_from_xarray

.. autofunction:: tiledb.cf.copy_metadata_from_xarray
