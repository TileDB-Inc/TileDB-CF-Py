<a href="https://tiledb.com"><img src="https://github.com/TileDB-Inc/TileDB/raw/dev/doc/source/_static/tiledb-logo_color_no_margin_@4x.png" alt="TileDB logo" width="400"></a>

# TileDB-CF-Py

The TileDB-CF-Py library is a Python library for supporting the NetCDF data model in the [TileDB storage engine](https://github.com/TileDB-Inc/TileDB). TileDB-CF-Py provides readers and writers for viewing and manipulating TileDB arrays and groups using TileDB CF Dataspaces - a special TileDB group that follows the requirements in [tiledb-cf-spec.md](tiledb-cf-spec.md).

## TileDB Quick Links

  * [Homepage](https://tiledb.com)
  * [Documentation](https://docs.tiledb.com/main/)
  * [Forum](https://forum.tiledb.io/)
  * [Organization](https://github.com/TileDB-Inc/)

## Getting Started

### Quick Installation

This project is available from [PyPI](https://pypi.org/project/tiledb/) and may be installed with ``pip``:

```bash
pip install tiledb-cf
```

### Documentation

#### API Documentation

To build the API documentation do the following from this projects root directory:

1. Install required packages:
   ```bash
   python3 -m pip install tiledb-cf[docs]
   ```
2. Make the HTML document:
   ```bash
   make -C docs/ html
   ```
3. Open [docs/_build/html/index.html](./docs/_build/html/index.html) in a web browser of your choice.

#### Example Notebooks

Example Jupyter notebooks are available in the [examples](./examples) folder.

#### Command Line Interface

TileDB-CF provides a command line interface. Currently, it has the following commands:

```bash
Usage: tiledb-cf netcdf-convert [OPTIONS]

  Converts a NetCDF input file to nested TileDB groups.

Options:
  -i, --input-file TEXT           The path or URI to the NetCDF file that will be converted.  [required]

  -o, --output-uri TEXT           The URI for the output TileDB group. [required]

  --input-group-path TEXT         The path in the input NetCDF for the root group that will be converted.  [default: /]

  --recursive / --no-recursive    Recursively convert all groups contained in the input group path.  [default: True]

  -k, --output-key TEXT           Key for the generated TileDB arrays.

  --unlimited-dim-size INTEGER    Size to convert unlimited dimensions to. [default: 10000]

  --dim-dtype [int8|int16|int32|int64|uint8|uint16|uint32|uint64]
                                  The data type for TileDB dimensions created from converted NetCDF.  [default: uint64]

  --help                          Show this message and exit.
```

## Development

For information on contributing to this project see the [CONTRIBUTING](CONTRIBUTING.md) document.
