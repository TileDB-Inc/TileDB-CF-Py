import numpy as np
from click.testing import CliRunner

import tiledb
import tiledb.cf


def test_netcdf_convert_collect(tmpdir, simple1_netcdf_file):
    uri = str(tmpdir.mkdir("output").join("simple1"))
    runner = CliRunner()
    result = runner.invoke(
        tiledb.cf.cli,
        [
            "netcdf-convert",
            "-i",
            simple1_netcdf_file.filepath,
            "-o",
            uri,
            "--collect-attrs",
        ],
    )
    assert result.exit_code == 0
    array_schema = tiledb.ArraySchema.load(uri + "/array0")
    attr_names = [attr.name for attr in array_schema]
    dim_names = [dim.name for dim in array_schema.domain]
    assert attr_names == ["x1"]
    assert dim_names == ["row"]
    with tiledb.open(uri + "/array0", attr="x1") as array:
        x1 = array[:]
    np.testing.assert_equal(x1, np.linspace(1.0, 4.0, 8))


def test_netcdf_convert_separate(tmpdir, simple1_netcdf_file):
    uri = str(tmpdir.mkdir("output").join("simple1"))
    runner = CliRunner()
    result = runner.invoke(
        tiledb.cf.cli,
        [
            "netcdf-convert",
            "-i",
            simple1_netcdf_file.filepath,
            "-o",
            uri,
            "--array-per-attr",
        ],
    )
    assert result.exit_code == 0
    array_schema = tiledb.ArraySchema.load(uri + "/x1")
    attr_names = [attr.name for attr in array_schema]
    dim_names = [dim.name for dim in array_schema.domain]
    assert attr_names == ["x1"]
    assert dim_names == ["row"]
    with tiledb.open(uri + "/x1", attr="x1") as array:
        x1 = array[:]
    np.testing.assert_equal(x1, np.linspace(1.0, 4.0, 8))
