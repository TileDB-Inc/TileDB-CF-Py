import sys
from typing import Any, Dict, List

import numpy as np
import pytest

import tiledb
from tiledb.cf.xarray_engine import (
    copy_data_from_xarray,
    copy_metadata_from_xarray,
    create_group_from_xarray,
    from_xarray,
)

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 9), reason="xarray requires python3.9 or higher"
)


class TileDBXarrayWriterBase:
    """Base class for TileDB-xarray writes."""

    name = "base"
    kwargs: Dict[str, Any] = {}

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        return str(tmpdir_factory.mktemp("output").join(f"{self.name}.tiledb"))

    @pytest.fixture(scope="class")
    def input_dataset(self):
        return xr.Dataset()

    @pytest.fixture(scope="class")
    def expected_dataset(self):
        return xr.Dataset()

    @pytest.fixture(scope="class")
    def expected_schemas(self):
        return {}

    def test_create_dataset(
        self, group_uri, input_dataset, expected_dataset, expected_schemas
    ):
        # Write to TileDB and check the reloaded datests.
        from_xarray(input_dataset, group_uri, **self.kwargs)
        result = xr.open_dataset(group_uri, engine="tiledb")
        xr.testing.assert_equal(result, expected_dataset)

        # Check the array schemas are as expected.
        with tiledb.Group(group_uri) as group:
            for name, expected_schema in expected_schemas.items():
                array_uri = group[name].uri
                schema = tiledb.ArraySchema.load(array_uri)
                assert schema == expected_schema


class TileDBXarrayMultiWriterBase:
    """Base class for TileDB-xarray writes."""

    name = "base"

    create_kwargs: Dict[str, Any] = {}
    copy_kwargs: List[Dict[str, Any]] = []

    @pytest.fixture(scope="class")
    def input_datasets(self):
        return [xr.Dataset()]

    @pytest.fixture(scope="class")
    def expected_datasets(self):
        return xr.Dataset()

    @pytest.fixture(scope="class")
    def expected_schemas(self):
        return {}

    def test_create_datasets(
        self,
        tmpdir,
        input_datasets,
        expected_datasets,
        expected_schemas,
    ):
        # Create the initial group.
        group_uri = str(tmpdir.mkdir("output1").join(self.name))
        create_group_from_xarray(input_datasets[0], group_uri, **self.create_kwargs)

        # Check the array schemas are as expected.
        with tiledb.Group(group_uri) as group:
            for name, expected_schema in expected_schemas.items():
                array_uri = group[name].uri
                schema = tiledb.ArraySchema.load(array_uri)
                assert schema == expected_schema

        num_input = len(input_datasets)
        assert len(self.copy_kwargs) == num_input
        assert len(expected_datasets) == num_input

        for index, ds in enumerate(input_datasets):
            copy_data_from_xarray(ds, group_uri, **self.copy_kwargs[index])
            result = xr.open_dataset(group_uri, engine="tiledb")
            xr.testing.assert_equal(result, expected_datasets[index])

    def test_create_datasets_separate_copy_metadata(
        self,
        tmpdir,
        input_datasets,
        expected_datasets,
        expected_schemas,
    ):
        # Create the initial group.
        group_uri = str(tmpdir.mkdir("output2").join(self.name))
        create_group_from_xarray(
            input_datasets[0],
            group_uri,
            copy_group_metadata=False,
            copy_variable_metadata=False,
            **self.create_kwargs,
        )
        copy_metadata_from_xarray(input_datasets[0], group_uri)

        # Check the array schemas are as expected.
        with tiledb.Group(group_uri) as group:
            for name, expected_schema in expected_schemas.items():
                array_uri = group[name].uri
                schema = tiledb.ArraySchema.load(array_uri)
                assert schema == expected_schema

        num_input = len(input_datasets)
        assert len(self.copy_kwargs) == num_input
        assert len(expected_datasets) == num_input

        for index, ds in enumerate(input_datasets):
            copy_data_from_xarray(ds, group_uri, **self.copy_kwargs[index])
            result = xr.open_dataset(group_uri, engine="tiledb")
            xr.testing.assert_equal(result, expected_datasets[index])


class TestWriteEmpty(TileDBXarrayWriterBase):
    name = "empty"


class TestWriteSimple1D(TileDBXarrayWriterBase):
    name = "simple_1d"

    ds = xr.Dataset(
        {
            "example": xr.DataArray(
                np.linspace(-1.0, 1.0, 16, dtype=np.float32),
                dims="rows",
                attrs={"units": "inches"},
            ),
            "index": xr.DataArray(
                np.arange(16, dtype=np.uint32),
                dims="rows",
                attrs={"long name": "Rows in the data array"},
            ),
        },
        attrs={"description": "example dataset"},
    )

    kwargs = {
        "encoding": {
            "example": {"tiles": (8,)},
            "index": {"tiles": (8,)},
        }
    }

    @pytest.fixture(scope="class")
    def input_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_schemas(self):
        domain = tiledb.Domain(
            tiledb.Dim("rows", domain=(0, 15), tile=8, dtype=np.uint32)
        )
        default_filters = tiledb.FilterList([tiledb.ZstdFilter(level=5)])
        example_attr = tiledb.Attr("example", filters=default_filters, dtype=np.float32)
        index_attr = tiledb.Attr("index", filters=default_filters, dtype=np.uint32)
        return {
            "example": tiledb.ArraySchema(domain, (example_attr,)),
            "index": tiledb.ArraySchema(domain, (index_attr,)),
        }


class TestWriteCoord1D(TileDBXarrayWriterBase):
    name = "coord_1d"

    ds = xr.Dataset(
        {
            "example": xr.DataArray(
                np.linspace(-1.0, 1.0, 16, dtype=np.float32), dims="rows"
            ),
            "rows": xr.DataArray(np.arange(16, dtype=np.uint32), dims="rows"),
        }
    )

    kwargs = {
        "encoding": {
            "example": {"tiles": (8,)},
            "rows": {"tiles": (8,), "attr_name": "index"},
        }
    }

    @pytest.fixture(scope="class")
    def input_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_schemas(self):
        domain = tiledb.Domain(
            tiledb.Dim("rows", domain=(0, 15), tile=8, dtype=np.uint32)
        )
        default_filters = tiledb.FilterList([tiledb.ZstdFilter(level=5)])
        example_attr = tiledb.Attr("example", filters=default_filters, dtype=np.float32)
        index_attr = tiledb.Attr("index", filters=default_filters, dtype=np.uint32)
        return {
            "example": tiledb.ArraySchema(domain, (example_attr,)),
            "rows": tiledb.ArraySchema(domain, (index_attr,)),
        }


class TestWriteUnlimitedDim1D(TileDBXarrayWriterBase):
    name = "unlimited_1d"

    kwargs = {
        "encoding": {"example": {"tiles": (8,)}},
        "region": {"rows": slice(8)},
        "unlimited_dims": set(["rows"]),
    }

    ds = xr.Dataset(
        {"example": xr.DataArray(np.arange(8, dtype=np.uint32), dims="rows")}
    )

    @pytest.fixture(scope="class")
    def input_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_schemas(self):
        domain = tiledb.Domain(
            tiledb.Dim(
                "rows",
                domain=(0, np.iinfo(np.dtype("uint32")).max - 1),
                tile=8,
                dtype=np.uint32,
            )
        )
        default_filters = tiledb.FilterList([tiledb.ZstdFilter(level=5)])
        example_attr = tiledb.Attr("example", filters=default_filters, dtype=np.uint32)
        return {
            "example": tiledb.ArraySchema(domain, (example_attr,)),
        }


class TestWriteUnlimitedDimWithShape1D(TileDBXarrayWriterBase):
    name = "unlimited_1d"

    kwargs = {
        "encoding": {"example": {"tiles": (8,), "max_shape": (32,)}},
        "region": {"rows": slice(8)},
        "unlimited_dims": set(["rows"]),
    }

    ds = xr.Dataset(
        {"example": xr.DataArray(np.arange(8, dtype=np.uint32), dims="rows")}
    )

    @pytest.fixture(scope="class")
    def input_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_schemas(self):
        domain = tiledb.Domain(
            tiledb.Dim("rows", domain=(0, 31), tile=8, dtype=np.uint32)
        )
        default_filters = tiledb.FilterList([tiledb.ZstdFilter(level=5)])
        example_attr = tiledb.Attr("example", filters=default_filters, dtype=np.uint32)
        return {
            "example": tiledb.ArraySchema(domain, (example_attr,)),
        }


class TestWriteTimedelta1D(TileDBXarrayWriterBase):
    name = "timedelta_1d"

    kwargs = {"encoding": {"example": {"tiles": (8,)}}}

    ds = xr.decode_cf(
        xr.Dataset(
            {
                "example": xr.DataArray(
                    data=np.arange(8, dtype=np.float64),
                    dims="rows",
                    attrs={"units": "hours"},
                )
            }
        )
    )

    @pytest.fixture(scope="class")
    def input_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_schemas(self):
        domain = tiledb.Domain(
            tiledb.Dim("rows", domain=(0, 7), tile=8, dtype=np.uint32)
        )
        default_filters = tiledb.FilterList([tiledb.ZstdFilter(level=5)])
        example_attr = tiledb.Attr("example", filters=default_filters, dtype=np.int64)
        return {
            "example": tiledb.ArraySchema(domain, (example_attr,)),
        }


class TestWriteDatetime(TileDBXarrayWriterBase):
    name = "datetime_1d"

    ds = xr.decode_cf(
        xr.Dataset(
            {
                "example": xr.DataArray(
                    data=np.arange(8, dtype=np.float64),
                    dims="rows",
                    attrs={"units": "hours since 2000-01-01"},
                )
            }
        )
    )

    kwargs = {
        "encoding": {"example": {"tiles": (8,)}},
        "region": {"rows": slice(8)},
    }

    @pytest.fixture(scope="class")
    def input_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_schemas(self):
        domain = tiledb.Domain(
            tiledb.Dim("rows", domain=(0, 7), tile=8, dtype=np.uint32)
        )
        default_filters = tiledb.FilterList([tiledb.ZstdFilter(level=5)])
        example_attr = tiledb.Attr("example", filters=default_filters, dtype="int64")
        return {
            "example": tiledb.ArraySchema(domain, (example_attr,)),
        }


class TestWriteMixedUnlimitedDim12D(TileDBXarrayWriterBase):
    name = "mixed_unlimited_2d"

    kwargs = {
        "encoding": {"example": {"tiles": (8, 8), "max_shape": (32, 8)}},
        "region": {"rows": slice(8)},
        "unlimited_dims": set(["rows"]),
    }

    ds = xr.Dataset(
        {
            "example": xr.DataArray(
                np.reshape(np.arange(64, dtype=np.uint32), (8, 8)),
                dims=("rows", "cols"),
            )
        }
    )

    @pytest.fixture(scope="class")
    def input_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_schemas(self):
        domain = tiledb.Domain(
            tiledb.Dim("rows", domain=(0, 31), tile=8, dtype=np.uint32),
            tiledb.Dim("cols", domain=(0, 7), tile=8, dtype=np.uint32),
        )
        default_filters = tiledb.FilterList([tiledb.ZstdFilter(level=5)])
        example_attr = tiledb.Attr("example", filters=default_filters, dtype=np.uint32)
        return {
            "example": tiledb.ArraySchema(domain, (example_attr,)),
        }


class TestWriteSimpleDask(TileDBXarrayWriterBase):
    da = pytest.importorskip("dask.array")

    name = "simple_dask"

    kwargs = {"encoding": {"example": {"tiles": (8, 8)}}}

    ds = xr.Dataset(
        {
            "example": xr.DataArray(
                da.from_array(
                    np.reshape(np.arange(1024, dtype=np.float32), (32, 32)),
                    chunks=((8, 16, 8), (16, 16)),
                ),
                dims=("rows", "cols"),
            )
        }
    )

    @pytest.fixture(scope="class")
    def input_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_dataset(self):
        return self.ds

    @pytest.fixture(scope="class")
    def expected_schemas(self):
        domain = tiledb.Domain(
            tiledb.Dim("rows", domain=(0, 31), tile=8, dtype=np.uint32),
            tiledb.Dim("cols", domain=(0, 31), tile=8, dtype=np.uint32),
        )
        default_filters = tiledb.FilterList([tiledb.ZstdFilter(level=5)])
        example_attr = tiledb.Attr("example", filters=default_filters, dtype=np.float32)
        return {
            "example": tiledb.ArraySchema(domain, (example_attr,)),
        }


class TestMultWriteSimple1D(TileDBXarrayMultiWriterBase):
    name = "multiwrite_simple_1d"

    ds1 = xr.Dataset({"example": xr.DataArray(np.arange(16, dtype=np.int32), dims="x")})
    ds2 = xr.Dataset(
        {"example": xr.DataArray(np.arange(-16, 0, dtype=np.int32), dims="x")}
    )

    create_kwargs = {"encoding": {"example": {"tiles": (8,)}}}
    copy_kwargs = [{}, {}]

    @pytest.fixture(scope="class")
    def input_datasets(self):
        return [self.ds1, self.ds2]

    @pytest.fixture(scope="class")
    def expected_datasets(self):
        return [self.ds1, self.ds2]

    @pytest.fixture(scope="class")
    def expected_schemas(self):
        default_filters = tiledb.FilterList([tiledb.ZstdFilter(level=5)])
        return {
            "example": tiledb.ArraySchema(
                tiledb.Domain(tiledb.Dim("x", domain=(0, 15), tile=8, dtype=np.uint32)),
                [tiledb.Attr("example", filters=default_filters, dtype=np.int32)],
            )
        }


class TestMultRegionWriteSimple1D(TileDBXarrayMultiWriterBase):
    name = "multi_region_write_simple_1d"

    create_kwargs = {
        "encoding": {"example": {"tiles": (8,), "max_shape": (16,)}},
    }

    copy_kwargs = [{"region": {"x": slice(0, 8)}}, {"region": {"x": slice(8, 16)}}]

    @pytest.fixture(scope="class")
    def input_datasets(self):
        return [
            xr.Dataset(
                {"example": xr.DataArray(np.arange(0, 8, dtype=np.int32), dims="x")}
            ),
            xr.Dataset(
                {"example": xr.DataArray(np.arange(0, 8, dtype=np.int32), dims="x")}
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_datasets(self):
        data = np.arange(0, 16, dtype=np.int32)
        data[8:16] = np.arange(0, 8, dtype=np.int32)
        return [
            xr.Dataset({"example": xr.DataArray(data[0:8], dims="x")}),
            xr.Dataset({"example": xr.DataArray(data, dims="x")}),
        ]

    @pytest.fixture(scope="class")
    def expected_schemas(self):
        default_filters = tiledb.FilterList([tiledb.ZstdFilter(level=5)])
        return {
            "example": tiledb.ArraySchema(
                tiledb.Domain(tiledb.Dim("x", domain=(0, 15), tile=8, dtype=np.uint32)),
                [tiledb.Attr("example", filters=default_filters, dtype=np.int32)],
            )
        }
