"""Regions for writing data to TileDB arrays"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

import tiledb

from .._utils import safe_set_metadata
from ._metadata import AttrMetadata
from ._shared_dim import SharedDim
from .source import FieldData

DenseRange = Union[Tuple[int, int], Tuple[np.datetime64, np.datetime64]]


class FragmentWriter(metaclass=ABCMeta):
    def __init__(self, attr_names: Sequence[str]):
        self._attr_data = {name: None for name in attr_names}

    def add_attr(self, attr_name: str):
        self._attr_data.setdefault(attr_name, None)

    def remove_attr(self, attr_name: str):
        del self._attr_data[attr_name]

    @abstractmethod
    def set_attr_data(self, attr_name: str, data: FieldData):
        ...

    @abstractmethod
    def set_dim_data(self, dim_index: int, data: FieldData):
        ...

    @abstractmethod
    def write(self, array: tiledb.libtiledb.Array):
        ...

    def write_attr_metadata(self, array: tiledb.libtiledb.Array):
        for name, data in self._attr_data.items():
            meta = AttrMetadata(array.meta, name)
            for key, val in data.metadata.items():
                safe_set_metadata(meta, key, val)


class DenseRegion:
    def __init__(
        self,
        dims: Tuple[SharedDim],
        region: Tuple[DenseRange, ...],
    ):
        # TODO: Check all dimensions are ints or dataetime
        self._dims = dims
        if region is None:
            self._region = tuple(dim.domain for dim in self._dims)
        else:
            self._region = tuple(region)
        if len(self._region) != len(self._dims):
            # TODO: Add error message
            raise ValueError()
        self._shape = tuple(
            int(dim_range[1] - dim_range[0]) + 1 for dim_range in self._region
        )
        self._size = sum(self._shape)

    def coordinates(self):
        def create_coords(dim_range, dtype):
            if dtype.kind in {"u", "i"}:
                dt = 1
            elif dtype.kind == "M":
                dt = np.timedelta(1, np.datetime_data(dtype)[0])
            else:
                raise ValueError(f"Unsupported datatype {dtype}.")
            return np.arange(dim_range[0], dim_range[1] + dt, dtype=dtype)

        values = tuple(
            create_coords(dim_range, dim.dtype)
            for dim, dim_range in zip(self._dims, self._region)
        )
        return np.meshgrid(*values, indexing="ij")

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    def subarray(self) -> List[slice, ...]:
        return [
            (
                slice(dim_range[0], dim_range[1] + 1)
                if np.issubdtype(type(dim_range[1]), np.integer)
                else slice(dim_range[0], dim_range[1])
            )
            for dim_range in self._region
        ]

    @property
    def region(self):
        return self._region


class DenseArrayFragmentWriter(FragmentWriter):
    def __init__(
        self,
        *,
        dims: Tuple[SharedDim],
        attr_names: Sequence[str],
        target_region: Optional[Tuple[DenseRange, ...]],
    ):
        self._target_region = DenseRegion(dims, target_region)
        super().__init__(attr_names)

    def set_attr_data(self, attr_name: str, data: FieldData):
        if data.size != self._target_region.size:
            raise ValueError(
                f"Cannot set data with size {data.size} to a fragment region with size "
                f"{self._target_region.size}"
            )
        if data.shape != self._target_region.shape:
            data.shape = self._target_region.shape
        self._attr_data[attr_name] = data

    def set_dim_data(self, dim_index: int, data: FieldData):
        raise ValueError("Cannot set dimension data on a dense fragment.")

    def write(self, array: tiledb.libtiledb.Array):
        # Check data is set.
        for name, data in self._attr_data.items():
            if data is None:
                raise ValueError(
                    f"Cannot write fragment. Missing data for attribute '{name}'."
                )

        # Write buffer data.
        array[*self._target_region.subarray()] = {
            name: data.values for name, data in self._attr_data.items()
        }
        self.write_attr_metadata(array)


class SparseArrayFragmentWriter:
    def __init__(
        self,
        *,
        dims: Tuple[SharedDim],
        attr_names: Sequence[str],
        size: Optional[int] = None,
        target_region: Optional[Tuple[DenseRange, ...]] = None,
    ):
        if target_region is not None or size is None:
            self._target_region = DenseRegion(dims, target_region)
            self._size = self._target_region.size
            self._dim_data = None
        else:
            self._target_region = None
            self._dim_data = [None] * len(dims)
            self._size = size
        self._attr_data = {name: None for name in attr_names}

    def set_attr_data(self, attr_name: str, data: FieldData):
        if data.size != self._size:
            raise ValueError(
                f"Cannot set data with size {data.size} to a fragment region with size "
                f"{self._region.size}"
            )
        self._attr_data[attr_name] = data

    def set_dim_data(self, dim_index: int, data: FieldData):
        if self._target_region is not None:
            raise ValueError(
                "Cannot set dimension data. Fragment writer is set on a dense region."
            )
        if data.size != self._size:
            raise ValueError(
                f"Cannot set data with size {data.size} to a fragment region with size "
                f"{self._size}"
            )
        self._dim_data[dim_index] = data

    def write(self, array: tiledb.libtiledb.Array):
        # Check data is set.
        if self._target_region is None:
            for idim, data in enumerate(self._dim_data):
                if data is None:
                    raise ValueError(
                        f"Cannot write sparse fragment. Missing data for dimension "
                        f"with index {idim}."
                    )
        for name, data in self._attr_data.items():
            if data is None:
                raise ValueError(
                    f"Cannot write fragment. Missing data for attribute '{name}'."
                )

        # Get dimension data.
        if self._target_region is None:
            coords = tuple(data.values for data in self._dim_data)
        else:
            coords = self._target_region.coordinates()

        # Write the data.
        array[*coords] = {name: data.values for name, data in self._attr_data.items()}
