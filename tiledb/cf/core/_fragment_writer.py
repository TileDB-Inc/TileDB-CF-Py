"""Regions for writing data to TileDB arrays"""

from __future__ import annotations

from abc import ABCMeta
from typing import Optional, Sequence, Tuple, Union

import numpy as np

import tiledb

from .._utils import safe_set_metadata
from ._metadata import AttrMetadata, DimMetadata
from ._shared_dim import SharedDim
from .source import FieldData

DenseRange = Union[Tuple[int, int], Tuple[np.datetime64, np.datetime64]]


class FragmentWriter(metaclass=ABCMeta):
    @classmethod
    def create_dense(
        cls,
        dims: Tuple[SharedDim],
        attr_names: Sequence[str],
        target_region: Optional[Tuple[DenseRange, ...]],
    ):
        return cls(DenseRegion(dims, target_region), attr_names)

    @classmethod
    def create_sparse_coo(
        cls, dims: Tuple[SharedDim], attr_names: Sequence[str], size: int
    ):
        return cls(SparseRegion(dims, size), attr_names)

    @classmethod
    def create_sparse_row_major(
        cls, dims: Tuple[SharedDim], attr_names: Sequence[str], shape: Tuple[int, ...]
    ):
        return cls(SparseRowMajorRegion(dims, shape), attr_names)

    def __init__(
        self,
        target_region: Union[DenseRegion, SparseRegion, SparseRowMajorRegion],
        attr_names: Sequence[str],
    ):
        self._attr_data = {name: None for name in attr_names}
        self._target_region = target_region
        if isinstance(self._target_region, DenseRegion):
            self._is_dense_region = True
        elif isinstance(self._target_region, SparseRegion) or isinstance(
            self._target_region, SparseRowMajorRegion
        ):
            self._is_dense_region = False
        else:
            raise TypeError(
                f"Type {type(self._target_region)} is not a valid target region."
            )  # pragma: no cover

    def add_attr(self, attr_name: str):
        self._attr_data.setdefault(attr_name, None)

    @property
    def is_dense_region(self) -> bool:
        return self._is_dense_region

    @property
    def nattr(self) -> int:
        return len(self._attr_data)

    @property
    def ndim(self) -> int:
        return self._target_region.ndim

    def remove_attr(self, attr_name: str):
        del self._attr_data[attr_name]

    def set_attr_data(self, attr_name: str, data: FieldData):
        if attr_name not in self._attr_data:
            raise KeyError(f"Array has no attribute '{attr_name}'.")
        if data.size != self._target_region.size:
            raise ValueError(
                f"Cannot set data with size {data.size} to a fragment region with size "
                f"{self._target_region.size}"
            )
        if (
            self._target_region.shape is not None
            and data.shape != self._target_region.shape
        ):
            data.shape = self._target_region.shape
        self._attr_data[attr_name] = data

    def set_dim_data(self, dim_name: str, data: FieldData):
        self._target_region.set_dim_data(dim_name, data)

    def write(self, array: tiledb.libtiledb.Array, *, skip_metadata: bool = False):
        # Check data is set.
        for name, data in self._attr_data.items():
            if data is None:
                raise ValueError(
                    f"Cannot write fragment. Missing data for attribute '{name}'."
                )

        # Get the data for what region to write data: coordinates for a sparse array
        # and a subarray for a dense array.
        if array.schema.sparse:
            region = self._target_region.coordinates()
        else:
            if not self._is_dense_region:
                raise RuntimeError("Cannot write a sparse fragment in a dense array.")
            region = self._target_region.subarray()

        for attr in array.schema:
            attr_data = self._attr_data[attr.name]
            if hasattr(attr_data, "fill") and attr_data.fill != attr.fill[0]:
                attr_data.fill = attr.fill

        array[region] = {name: data.values for name, data in self._attr_data.items()}

        if not skip_metadata:
            for name, data in self._attr_data.items():
                meta = AttrMetadata(array.meta, name)
                for key, val in data.metadata.items():
                    safe_set_metadata(meta, key, val)
            self._target_region.write_metadata(array)


class DenseRegion:
    def __init__(
        self,
        dims: Tuple[SharedDim],
        region: Tuple[DenseRange, ...],
    ):
        self._dims = dims
        for dim in self._dims:
            if dim.dtype.kind not in {"u", "i", "M"}:
                raise ValueError(
                    f"Cannot create a dense region for array with dimension "
                    f"'{dim.name}' with dtype={dim.dtype}."
                )
        if region is None:
            self._region = tuple(dim.domain for dim in self._dims)
        else:
            self._region = tuple(region)
        if len(self._region) != len(self._dims):
            raise ValueError(
                f"Cannot set {len(self._region)} regions on an array with "
                f"{len(self._dims)} dimensions."
            )
        self._shape = tuple(
            int(dim_range[1] - dim_range[0]) + 1 for dim_range in self._region
        )
        self._size = np.prod(self._shape)

    def coordinates(self) -> Tuple[np.ndarray]:
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
        return tuple(
            dim_data.reshape(-1) for dim_data in np.meshgrid(*values, indexing="ij")
        )

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def set_dim_data(self, _dim_name: str, _data: FieldData):
        raise RuntimeError(
            "Cannot set dimension data on fragment that is being written to a dense "
            "region."
        )

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        return self._shape

    @property
    def size(self) -> int:
        return self._size

    def subarray(self) -> Tuple[slice, ...]:
        return tuple(
            (
                slice(dim_range[0], dim_range[1] + 1)
                if np.issubdtype(type(dim_range[1]), np.integer)
                else slice(dim_range[0], dim_range[1])
            )
            for dim_range in self._region
        )

    def write_metadata(self, array: tiledb.libtiledb.Array):
        """Write any metadata associated with this region."""


class SparseRegion:
    def __init__(self, dims: Tuple[SharedDim], size: int):
        # Set dimensions.
        self._dims = dims
        self._dim_data = [None] * len(dims)

        # Set the size of the data.
        self._size = size

    def coordinates(self) -> Tuple[np.ndarray]:
        for idim, data in enumerate(self._dim_data):
            if data is None:
                raise ValueError(
                    f"Cannot construct dimension coordinates. Missing data for "
                    f"dimension '{self._dims[idim].name}'."
                )
        return tuple(data.values for data in self._dim_data)

    @property
    def ndim(self) -> int:
        return len(self._dims)

    def set_dim_data(self, dim_name: str, data: FieldData):
        if data.size != self._size:
            raise ValueError(
                f"Cannot set data with size {data.size} to dimension '{dim_name}' on "
                f"a fragment with target region size {self._size}"
            )
        for index, dim in enumerate(self._dims):
            if dim.name == dim_name:
                dim_index = index
                break
        else:
            raise KeyError("No dimension with name '{dim_name}'")

        self._dim_data[dim_index] = data

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        return None

    @property
    def size(self) -> int:
        return self._size

    def subarray(self) -> Tuple[slice, ...]:
        raise RuntimeError("Cannot construct a subarray for a sparse region.")

    def write_metadata(self, array: tiledb.libtiledb.Array):
        """Write any metadata associated with this region."""
        for dim, data in zip(self._dims, self._dim_data):
            meta = DimMetadata(array.meta, dim.name)
            for key, val in data.metadata.items():
                safe_set_metadata(meta, key, val)


class SparseRowMajorRegion:
    def __init__(self, dims: Tuple[SharedDim], shape: Tuple[int, ...]):
        # Check input.
        if len(dims) != len(shape):
            raise ValueError(
                f"Cannot set a fragment with shape={shape} for an array with "
                f"{len(dims)} dimensions."
            )

        # Set dimensions.
        self._dims = dims
        self._dim_data = [None] * len(dims)

        # Set the size of the data.
        self._shape = shape
        self._size = np.prod(shape)

    def coordinates(self) -> Tuple[np.ndarray]:
        for idim, data in enumerate(self._dim_data):
            if data is None:
                raise ValueError(
                    f"Cannot construct dimension coordinates. Missing data for "
                    f"dimension '{self._dims[idim].name}'."
                )
        coords = tuple(data.values for data in self._dim_data)
        return tuple(
            dim_data.reshape(-1) for dim_data in np.meshgrid(*coords, indexing="ij")
        )

    @property
    def ndim(self) -> int:
        return len(self._dims)

    def set_dim_data(self, dim_name: str, data: FieldData):
        for index, dim in enumerate(self._dims):
            if dim.name == dim_name:
                dim_index = index
                break
        else:
            raise KeyError("No dimension with name '{dim_name}'")

        if data.size != self._shape[dim_index]:
            raise ValueError(
                f"Cannot set data with size {data.size} to dimension the "
                f"{dim_index + 1}-th dimension '{dim_name}' on a fragment with target "
                f"region shape {self._shape}."
            )
        self._dim_data[dim_index] = data

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        return self._shape

    @property
    def size(self) -> int:
        return self._size

    def subarray(self) -> Tuple[slice, ...]:
        raise RuntimeError("Cannot construct a subarray for a sparse region.")

    def write_metadata(self, array: tiledb.libtiledb.Array):
        """Write any metadata associated with this region."""
        for dim, data in zip(self._dims, self._dim_data):
            meta = DimMetadata(array.meta, dim.name)
            for key, val in data.metadata.items():
                safe_set_metadata(meta, key, val)
