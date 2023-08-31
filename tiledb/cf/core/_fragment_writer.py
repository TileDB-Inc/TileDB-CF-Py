"""Regions for writing data to TileDB arrays"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
from typing_extensions import Protocol

import tiledb

from .._utils import safe_set_metadata
from ._metadata import AttrMetadata
from .source import BufferData


class FragmentRegionProtocol(Protocol):
    @property
    def shape(self) -> Union[None, Tuple[int, ...]]:
        ...

    @property
    def size(self) -> Union[None, Tuple[int, ...]]:
        ...

    @property
    def target_region(self) -> Union[None, Tuple[Tuple[int, int], ...]]:
        ...


class DenseFragmentRegion:
    def __init__(self, target_region: Tuple[Tuple[int, int], ...]):
        self._target_region = target_region
        self._shape = tuple(
            dim_range[1] - dim_range[0] + 1 for dim_range in target_region
        )
        self._size = sum(self._shape)

    @property
    def shape(self) -> Union[None, Tuple[int, ...]]:
        return self._shape

    @property
    def size(self) -> int:
        return sum(self._shape)

    @property
    def target_region(self) -> Union[None, Tuple[Tuple[int, int], ...]]:
        return self._target_region


class SparseFragmentRegion:
    def __init__(self, size: int):
        self._size

    @property
    def shape(self) -> Union[None, Tuple[int, ...]]:
        return None

    @property
    def size(self) -> int:
        return self._size

    @property
    def target_region(self) -> Union[None, Tuple[Tuple[int, int], ...]]:
        return None


class FragmentWriter:
    def __init__(
        self,
        *,
        sparse: bool,
        region: FragmentRegionProtocol,
        ndim: int,
        attr_names: Sequence[str],
    ):
        # TODO: Clean-up sparse vs. dense story
        self._sparse = sparse
        self._region = region
        if isinstance(self._region, SparseFragmentRegion):
            self._dim_data = [None] * ndim
        else:
            self._dim_data = None
        self._attr_data = {name: None for name in attr_names}

    def add_attr(self, attr_name: str):
        self._attr_data.setdefault(attr_name, None)

    def set_attr_data(self, attr_name: str, data: BufferData):
        if data.size != self._region.size:
            raise ValueError(
                f"Cannot set data with size {data.size} to a fragment region with size "
                f"{self._region.size}"
            )
        if not self._sparse and data.shape != self._region.shape:
            data.shape = self._region.shape

    def set_dim_data(self, dim_index: int, data: BufferData):
        if not self._sparse:
            raise ValueError("Cannot set dimension data on a dense fragment.")
        if data.size != self._region.size:
            raise ValueError(
                f"Cannot set data with size {data.size} to a fragment region with size "
                f"{self._region.size}"
            )
        self._dim_data[dim_index] = data

    def write_fragment(self, array: tiledb.libtiledb.Array):
        for name, data in self._attr_data.items():
            if data is None:
                raise ValueError(
                    f"Cannot write fragment. Missing data for attribute '{name}'."
                )

        # Write buffer data.
        if array.sparse:
            # TODO: Support dense region
            for idim, data in enumerate(self._dim_data):
                if data is None:
                    raise ValueError(
                        f"Cannot write sparse fragment. Missing data for dimension "
                        f"with index {idim}."
                    )
            array[*self._dim_data.data()] = {
                name: buffer_data.data()
                for name, buffer_data in self._attr_data.items()
            }
        else:
            subarray = [
                (
                    tuple((dim_range[0], dim_range[1] + 1))
                    if np.issubdtype(type(dim_range[1]), np.integer)
                    else dim_range
                )
                for dim_range in self._region.target_region
            ]
            array[subarray] = {
                name: buffer_data.data()
                for name, buffer_data in self._attr_data.items()
            }

        # Write metadata.
        # TODO: Add dimension metadata
        for name, data in self._attr_data.items():
            meta = AttrMetadata(array.meta, name)
            for key, val in data.metadata.items():
                safe_set_metadata(meta, key, val)
