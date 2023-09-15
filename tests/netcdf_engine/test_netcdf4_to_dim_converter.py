import numpy as np
import pytest

from tiledb.cf.core._creator import SharedDim
from tiledb.cf.netcdf_engine import NetCDF4ToDimConverter

netCDF4 = pytest.importorskip("netCDF4")


class TestSharedDimBase:
    """This class tests the NetCDF4ToDimConverter for a non-NetCDF dimension."""

    shared_dim = SharedDim(
        name="dim0",
        dtype=np.dtype("int32"),
        domain=(0, 31),
        registry=None,
    )

    def test_default_properties(self):
        """Tests the default properties are correctly set."""
        dim_converter = NetCDF4ToDimConverter(self.shared_dim)
        assert dim_converter.tile is None
        assert dim_converter.filters is None
        assert dim_converter.max_fragment_length is None

    def test_set_max_fragment_length(self):
        """Tests setting max_fragment_length."""
        dim_converter = NetCDF4ToDimConverter(self.shared_dim)
        dim_converter.max_fragment_length = 1
        assert dim_converter.max_fragment_length == 1

    def test_bad_max_fragment_length_error(self):
        """Tests error when setting an invalid max_fragment_length."""
        dim_converter = NetCDF4ToDimConverter(self.shared_dim)
        with pytest.raises(ValueError):
            dim_converter.max_fragment_length = 0
