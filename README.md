<a href="https://tiledb.com"><img src="https://github.com/TileDB-Inc/TileDB/raw/dev/doc/source/_static/tiledb-logo_color_no_margin_@4x.png" alt="TileDB logo" width="400"></a>

# TileDB-CF-Py

The TileDB-CF-Py library is a Python library for supporting standardized storage of climate and forecast datasets in [TileDB](https://tiledb.com). TileDB-CF-Py provides readers, writers, and an in-memory data model for viewing and manipulating TileDB Arrays and Groups using the TileDB-CF Standard.

Intended use cases for this library include using the readers and writers for converting NetCDF datasets that follow the [CF Convention](http://cfconventions.org/latest.html) to TileDB and reading/writing TileDB datasets to [xarray](http://xarray.pydata.org/en/stable/).

## Quick Links

* TileDB-CF-Py
  * [Documentation](https://docs.tiledb.com/geospatial)
  * API Documentation (TBD)
  * TileDB-CF Standard (TBD)

* TileDB
  * [Homepage](https://tiledb.com)
  * [Documentation](https://docs.tiledb.com/main/)
  * [Forum](https://forum.tiledb.io/)
  * [Organization](https://github.com/TileDB-Inc/)

## Quick Installation

This project is currently in early development, and is only available on [GitHub](https://github.com/).

Upon release it will be available from either [conda-forge](https://anaconda.org/conda-forge/tiledb-py) with
[conda](https://conda.io/docs/):

```bash
conda install -c conda-forge tiledb-cf
```

or from [PyPI](https://pypi.org/project/tiledb/) with ``pip``:

```bash
pip install tiledb-cf
```

## Development

For information on contributing to this project see the [CONTRIBUTING](CONTRIBUTING.md) document.
