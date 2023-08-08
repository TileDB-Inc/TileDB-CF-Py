import numpy as np
import pytest

import tiledb


@pytest.fixture
def create_tiledb_group_example(tmpdir):
    xr = pytest.importorskip("xarray")

    # Define data
    data = {
        "pressure": np.linspace(
            -1.0, 1.0, num=32, endpoint=True, dtype=np.float64
        ).reshape(8, 4),
        "count": np.arange(0, 32, dtype=np.int32).reshape(8, 4),
    }

    # Create expected dataset
    expected = xr.Dataset(
        data_vars={
            "pressure": xr.DataArray(
                data=data["pressure"],
                dims=["time", "x"],
                attrs={"long_name": "example float data"},
            ),
            "count": xr.DataArray(
                data=data["count"],
                dims=["time", "x"],
                attrs={"long_name": "example int data"},
            ),
        },
        attrs={"global_1": "value1", "global_2": "value2"},
    )

    # Create the TileDB group
    group_uri = str(tmpdir.join("tiledb_group_example_1"))
    count_uri = str(tmpdir.join("count_array"))
    pressure_uri = str(tmpdir.join("pressure_array"))
    count_schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="time", domain=(0, 7), tile=4, dtype=np.int32),
            tiledb.Dim(name="x", domain=(0, 3), tile=4, dtype=np.int32),
        ),
        sparse=False,
        attrs=[tiledb.Attr(name="count", dtype=np.int32)],
    )

    pressure_schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="time", domain=(0, 7), tile=4, dtype=np.int32),
            tiledb.Dim(name="x", domain=(0, 3), tile=4, dtype=np.int32),
        ),
        sparse=False,
        attrs=[tiledb.Attr(name="pressure", dtype=np.float64)],
    )

    # Create and write to arrays.
    tiledb.Array.create(count_uri, count_schema)
    with tiledb.open(count_uri, mode="w") as array:
        array[:, :] = data["count"]
        array.meta["__tiledb_attr.count.long_name"] = "example int data"
    tiledb.Array.create(pressure_uri, pressure_schema)
    with tiledb.open(pressure_uri, mode="w") as array:
        array[:, :] = data["pressure"]
        array.meta["__tiledb_attr.pressure.long_name"] = "example float data"

    # Create group and add arrays and metadata.
    tiledb.Group.create(group_uri)
    with tiledb.Group(group_uri, mode="w") as group:
        group.add(pressure_uri)
        group.add(count_uri)
        group.meta["global_1"] = "value1"
        group.meta["global_2"] = "value2"
    return group_uri, expected


@pytest.fixture
def create_tiledb_example(tmpdir):
    xr = pytest.importorskip("xarray")
    # Define data
    float_data = np.linspace(
        -1.0, 1.0, num=32, endpoint=True, dtype=np.float64
    ).reshape(8, 4)
    int_data = np.arange(0, 32, dtype=np.int32).reshape(8, 4)
    # Create expected dataset
    expected = xr.Dataset(
        data_vars={
            "pressure": xr.DataArray(
                data=float_data,
                dims=["time", "x"],
                attrs={"long_name": "example float data"},
            ),
            "count": xr.DataArray(
                data=int_data,
                dims=["time", "x"],
                attrs={"long_name": "example int data"},
            ),
        },
        attrs={"global_1": "value1", "global_2": "value2"},
    )
    array_uri = str(tmpdir.join("tiledb_example_1"))
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="time", domain=(0, 7), tile=4, dtype=np.int32),
            tiledb.Dim(name="x", domain=(0, 3), tile=4, dtype=np.int32),
        ),
        sparse=False,
        attrs=[
            tiledb.Attr(name="count", dtype=np.int32),
            tiledb.Attr(name="pressure", dtype=np.float64),
        ],
    )
    tiledb.Array.create(array_uri, schema)
    with tiledb.open(array_uri, mode="w") as array:
        array[:, :] = {
            "pressure": float_data,
            "count": int_data,
        }
        array.meta["global_1"] = "value1"
        array.meta["global_2"] = "value2"
        array.meta["__tiledb_attr.pressure.long_name"] = "example float data"
        array.meta["__tiledb_attr.count.long_name"] = "example int data"
    return array_uri, expected


@pytest.fixture
def create_tiledb_datetime_example(tmpdir):
    xr = pytest.importorskip("xarray")
    data = np.linspace(-1.0, 20.0, num=16, endpoint=True, dtype=np.float64)
    date = np.arange(np.datetime64("2000-01-01"), np.datetime64("2000-01-17"))
    # Create expected dataset
    expected = xr.Dataset(
        data_vars={"temperature": xr.DataArray(data=data, dims="date")},
        coords={"date": date},
    )
    # Create TileDB array
    array_uri = str(tmpdir.join("tiledb_example_2"))
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(
                name="date",
                domain=(np.datetime64("2000-01-01"), np.datetime64("2000-01-16")),
                tile=np.timedelta64(4, "D"),
                dtype=np.datetime64("", "D"),
            ),
        ),
        attrs=[tiledb.Attr(name="temperature", dtype=np.float64)],
    )
    tiledb.DenseArray.create(array_uri, schema)
    with tiledb.DenseArray(array_uri, mode="w") as array:
        array[:] = {"temperature": data}
    return array_uri, expected
