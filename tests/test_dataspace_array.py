import numpy as np
import pytest

import tiledb
from tiledb.cf import DataspaceArray

_CF_COORDINATE_SUFFIX = ".axis_data"


class TestDataspaceArray:

    _array_schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="pressure", domain=(0, 3), tile=2, dtype=np.uint64)
        ),
        attrs=[
            tiledb.Attr(name="pressure" + _CF_COORDINATE_SUFFIX, dtype=np.float64),
            tiledb.Attr(name="temperature", dtype=np.float64),
        ],
    )

    _pressure = np.array([15.7, 22.1, 17.9, 24.2])
    _temperature = np.array([5.9, 6.3, 3.3, 5.4])

    @pytest.fixture(scope="class")
    def array_uri(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("simple")) + "/array"
        tiledb.Array.create(uri, self._array_schema)
        with tiledb.DenseArray(uri, "w") as array:
            array[:] = {
                "pressure" + _CF_COORDINATE_SUFFIX: self._pressure,
                "temperature": self._temperature,
            }
        return uri

    def test_read_array_cf_coordinate(self, array_uri):
        with DataspaceArray(array_uri) as array:
            pressure = array.base[:]["pressure" + _CF_COORDINATE_SUFFIX]
        assert np.array_equal(pressure, self._pressure)

    def test_read_array_attribute(self, array_uri):
        with DataspaceArray(array_uri) as array:
            temperature = array.base[:]["temperature"]
        assert np.array_equal(temperature, self._temperature)
