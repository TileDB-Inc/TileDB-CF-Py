"""Classes for additional group and metadata support useful for the TileDB-CF data
model."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Iterator, Optional, TypeVar, Union

import tiledb

DType = TypeVar("DType", covariant=True)
ATTR_METADATA_FLAG = "__tiledb_attr."
DIM_METADATA_FLAG = "__tiledb_dim."


class Metadata(MutableMapping):
    """Class for accessing Metadata using the standard MutableMapping API.

    Parameters
    ----------
    metadata
        TileDB array metadata object.
    """

    def __init__(self, metadata: tiledb.Metadata):
        self._metadata = metadata

    def __iter__(self) -> Iterator[str]:
        """Iterates over all metadata keys."""
        for tiledb_key in self._metadata.keys():
            key = self._from_tiledb_key(tiledb_key)
            if key is not None:
                yield key

    def __len__(self) -> int:
        """Returns the number of metadata items."""
        return sum(1 for _ in self)

    def __getitem__(self, key: str) -> Any:
        """Implementation of [key] -> val (dict item retrieval).

        Parameters
        ----------
        key
            Key to find value from.

        Returns
        -------
        Any
            Value stored with provided key.
        """
        return self._metadata[self._to_tiledb_key(key)]

    def __setitem__(self, key: str, value: Any):
        """Implementation of [key] <- val (dict item assignment).

        Paremeters
        ----------
        key
            Key to set
        value
            Corresponding value
        """
        self._metadata[self._to_tiledb_key(key)] = value

    def __delitem__(self, key):
        """Implementation of del [key] (dict item deletion).

        Parameters
        ----------
        key
            Key to remove.
        """
        del self._metadata[self._to_tiledb_key(key)]

    def _to_tiledb_key(self, key: str) -> str:
        """Map an external user metadata key to an internal tiledb key."""
        return key  # pragma: no cover

    def _from_tiledb_key(self, tiledb_key: str) -> Optional[str]:
        """Map an internal tiledb key to an external user metadata key.

        Parameters
        ----------
        tiledb_key
            Internal key to use for metadata.

        Returns
        -------
        Optional[str]
            The external user metadata key corresponding to `tiledb_key`,
            or None if there is no such corresponding key.
        """
        return tiledb_key  # pragma: no cover


class ArrayMetadata(Metadata):
    """Class for accessing array-related metadata from a TileDB metadata object.

    This class provides a way for accessing the TileDB array metadata that excludes
    attribute and dimension specific metadata.
    """

    def _to_tiledb_key(self, key: str) -> str:
        if key.startswith(ATTR_METADATA_FLAG):
            raise KeyError("Key is reserved for attribute metadata.")
        if key.startswith(DIM_METADATA_FLAG):
            raise KeyError("Key is reserved for dimension metadata.")
        return key

    def _from_tiledb_key(self, tiledb_key: str) -> Optional[str]:
        if not (
            tiledb_key.startswith(ATTR_METADATA_FLAG)
            or tiledb_key.startswith(DIM_METADATA_FLAG)
        ):
            return tiledb_key
        return None


class AttrMetadata(Metadata):
    """Metadata wrapper for accessing attribute metadata.

    This class allows access to the metadata for an attribute stored in the metadata
    for a TileDB array.

    Parameters
    ----------
    metadata
        TileDB array metadata for the array containing the desired attribute.
    attr
        Name or index of the arrary attribute being requested.
    """

    def __init__(self, metadata: tiledb.Metadata, attr: Union[str, int]):
        super().__init__(metadata)
        try:
            attr_name = metadata.array.attr(attr).name
        except tiledb.TileDBError as err:
            raise KeyError(f"Attribute `{attr}` not found in array.") from err
        self._key_prefix = ATTR_METADATA_FLAG + attr_name + "."

    def _to_tiledb_key(self, key: str) -> str:
        return self._key_prefix + key

    def _from_tiledb_key(self, tiledb_key: str) -> Optional[str]:
        if tiledb_key.startswith(self._key_prefix):
            return tiledb_key[len(self._key_prefix) :]
        return None


class DimMetadata(Metadata):
    """Metadata wrapper for accessing dimension metadata.

    This class allows access to the metadata for a dimension stored in the metadata
    for a TileDB array.

    Parameters
    ----------
    metadata
        TileDB array metadata for the array containing the desired attribute.
    dim
        Name or index of the arrary attribute being requested.
    """

    def __init__(self, metadata: tiledb.Metadata, dim: Union[str, int]):
        super().__init__(metadata)
        try:
            dim_name = metadata.array.dim(dim).name
        except tiledb.TileDBError as err:
            raise KeyError(f"Dimension `{dim}` not found in array.") from err
        self._key_prefix = DIM_METADATA_FLAG + dim_name + "."

    def _to_tiledb_key(self, key: str) -> str:
        return self._key_prefix + key

    def _from_tiledb_key(self, tiledb_key: str) -> Optional[str]:
        if tiledb_key.startswith(self._key_prefix):
            return tiledb_key[len(self._key_prefix) :]
        return None
