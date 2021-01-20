import numpy as np
import pytest

import tiledb
from tiledb.cf import DataspaceGroup


class TestCreateMetadata:
    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        """Creates a TileDB group and return URI."""
        uri = str(tmpdir_factory.mktemp("empty_group"))
        tiledb.group_create(uri)
        return uri

    def test_create_metadata(self, group_uri):
        uri = group_uri
        with DataspaceGroup(uri) as group:
            assert not group.has_metadata_array
            group.create_metadata_array()
            assert not group.has_metadata_array
            group.reopen()
            assert group.has_metadata_array


class TestSimpleGroup:

    _metadata_schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.uint64)
        ),
        attrs=[tiledb.Attr(name="a", dtype=np.uint64)],
        sparse=True,
    )

    @pytest.fixture(scope="class")
    def group_uri(self, tmpdir_factory):
        uri = str(tmpdir_factory.mktemp("group1"))
        tiledb.group_create(uri)
        metadata_uri = DataspaceGroup.metadata_uri(uri)
        tiledb.Array.create(metadata_uri, self._metadata_schema)
        return uri

    def test_has_metadata(self, group_uri):
        uri = group_uri
        with DataspaceGroup(uri) as group:
            assert group.has_metadata_array
