# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.

import pytest

xr = pytest.importorskip("xarray")  # isort:skip
dask = pytest.importorskip("dask")  # isort:skip
distributed = pytest.importorskip("distributed")  # isort:skip

from dask.distributed import Client
from distributed.utils_test import cluster, loop
from xarray.tests import assert_allclose

da = pytest.importorskip("dask.array")

loop = loop  # loop is an imported fixture, which flake8 has issues ack-ing


def test_dask_distributed_tiledb_integration_test(loop, create_tiledb_example):
    array_uri, expected = create_tiledb_example
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            ds = xr.open_dataset(array_uri, chunks={"time": 1}, engine="tiledb-cf")
            assert isinstance(ds["pressure"].data, da.Array)
            actual = ds.compute()
            assert_allclose(actual, expected)


def test_dask_distributed_tiledb_datetime_integration_test(
    loop,
    create_tiledb_datetime_example,
):
    array_uri, expected = create_tiledb_datetime_example
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            ds = xr.open_dataset(array_uri, chunks={"date": 1}, engine="tiledb-cf")
            assert isinstance(ds["temperature"].data, da.Array)
            actual = ds.compute()
            assert_allclose(actual, expected)
