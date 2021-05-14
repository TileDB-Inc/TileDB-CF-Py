TileDB-CF Python API Reference
==============================

.. warning::

   The TileDB-CF Python library is still under development and the API may change rapidly.


Modules
-------

The TileDB-CF library can be imported using the ``tiledb.cf`` module, .eg.

.. code-block:: python

   import tiledb.cf

All external class and functions in the TileDB-CF library are available in this module.


NetCDF Auto-convert Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: tiledb.cf.from_netcdf
   :noindex:

.. autofunction:: tiledb.cf.from_netcdf_group
   :noindex:



Creator Classes
---------------

DataspaceCreator
^^^^^^^^^^^^^^^^

.. autoclass:: tiledb.cf.DataspaceCreator
   :members:
   :noindex:

Converter Classes
-----------------

NetCDF4ConverterEngine
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: tiledb.cf.engines.netcdf4_engine.NetCDF4ConverterEngine
   :members:
   :inherited-members:
   :noindex:


Core Classes
------------

Group
^^^^^

.. autoclass:: tiledb.cf.Group
   :members:
   :noindex:

VirtualGroup
^^^^^^^^^^^^

.. autoclass:: tiledb.cf.VirtualGroup
   :members:
   :inherited-members:
   :noindex:


GroupSchema
^^^^^^^^^^^

.. autoclass:: tiledb.cf.GroupSchema
   :members:
   :noindex:


Array Metadata
^^^^^^^^^^^^^^

.. autoclass:: tiledb.cf.ArrayMetadata
   :members: __getitem__, __setitem__, __delitem__
   :noindex:


AttrMetadata
^^^^^^^^^^^^

.. autoclass:: tiledb.cf.AttrMetadata
   :members: __getitem__, __setitem__, __delitem__
   :noindex:
