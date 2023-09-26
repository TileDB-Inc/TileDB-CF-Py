from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Tuple, Union

import numpy as np

# TODO: Update return type for numeric types
NumericValue = Any


class CFDataSource(Protocol):
    def add_offset(self) -> Optional[NumericValue]:
        ...

    @property
    def dtype(self) -> np.dtype:
        ...

    @property
    def fill(self) -> Optional[NumericValue]:
        ...

    def get_metadata(self) -> Mapping[str, Any]:
        ...

    def get_values(self) -> np.ndarray:
        ...

    @property
    def scale_factor(self) -> NumericValue:
        ...

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        ...

    @property
    def size(self) -> int:
        ...


class FieldData(Protocol):
    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the data."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        """A mapping of metadata string-to-value pairs."""

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of the data, or `None` if no shape."""

    @shape.setter
    def shape(self, new_shape: Tuple[int, ...]):
        """Set the shape to `new_shape`."""

    @property
    def size(self) -> int:
        """Size of the data."""

    @property
    def values(self) -> np.array:
        """Data values."""


class NumpyData:
    def __init__(
        self, input: np.array, *, metadata: Optional[Mapping[str, Any]] = None
    ):
        self._source_data = input
        self._metadata = dict() if metadata is None else dict(metadata)

    @property
    def dtype(self):
        return self._source_data.dtype

    @property
    def metadata(self):
        return self._metadata

    @property
    def shape(self):
        return self._source_data.shape

    @shape.setter
    def shape(self, new_shape):
        self._source_data = np.reshape(self._source_data, new_shape)

    @property
    def size(self):
        return self._source_data.size

    @property
    def values(self):
        return self._source_data


class NumpyRegion:
    def __init__(self, region: Tuple[Tuple[int, int], ...], max_shape=Tuple[int, ...]):
        self._region = region
        self._max_shape = max_shape
        self._shape = tuple(
            dim_range[1] - dim_range[0] + 1 for dim_range in self._region
        )

        # Check number of dimensions
        if len(self._region) != len(self._max_shape):
            raise ValueError()

        # Check valid ranges
        for dim_range, dim_size in zip(self._region, self._max_shape):
            if dim_range[1] < dim_range[0]:
                raise ValueError()
            if not 0 <= dim_range[0] < dim_size:
                raise KeyError()
            if dim_range[1] >= dim_size:
                raise KeyError()

    def as_slices(self):
        return tuple(
            slice(dim_range[0], dim_range[1] + 1) for dim_range in self._region
        )

    @property
    def ndim(self) -> int:
        return len(self._region)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape


class CFSourceConnector:
    def __init__(
        self,
        data_source: CFDataSource,
        *,
        dtype=None,
        shape=None,
        fill=None,
    ):
        # CF datasource
        self._source = data_source

        # Transformation information
        self._dtype = dtype
        self._shape = shape
        self._fill = fill

        # Loaded data
        self._metadata = None
        self._values = None

    @property
    def dtype(self) -> np.dtype:
        return self._source.dtype if self._dtype is None else self._dtype

    @property
    def fill(self) -> NumericValue:
        return self._source.fill if self._fill is None else self._fill

    @fill.setter
    def fill(self, new_fill: NumericValue):
        self._fill = new_fill

    def load(self):
        self.load_data()

    def load_metadata(self):
        if self._metadata is None:
            self._metadata = self._source.get_metadata()

    def load_values(self):
        if self._values is None:
            self.reload_values()

    @property
    def metadata(self) -> Mapping[str, Any]:
        if self._metadata is None:
            self._metadata = self._source.get_metadata()
        return self._metadata

    def reload(self):
        self.reload_values()
        self.reload_metadata()

    def reload_metadata(self):
        self._metadata = self._source.get_metadata()

    def reload_values(self):
        self._values = self._source.get_values()

        # Unpack
        if self._source.scale_factor is not None:
            self._values = self._source.scale_factor * self._values

        if self._source.add_offset is not None:
            self._values = self._values + self._source.add_offset

        # Convert datatype
        if self._dtype is not None and self._dtype != self._values.dtype:
            self._values = self._values.astype(self._dtype)

        # Update fill
        if (
            self._fill is not None
            and self._source.fill is not None
            and self._fill != self._source.fill
        ):
            np.putmask(self._values, self._values == self._source.fill, self._fill)

        # Reshape
        if self._shape is not None:
            self._values = self._values.reshape(self._shape)

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        return self._source.shape if self._shape is None else self._shape

    @shape.setter
    def shape(self, new_shape: Tuple[int, ...]):
        if sum(new_shape) != self._source.size:
            raise ValueError(
                f"Cannot reshape a variable with size={self._size} to the shape "
                f"shape={new_shape}."
            )
        self._shape = new_shape

    @property
    def size(self) -> int:
        return self._source.size

    @property
    def values(self) -> np.array:
        self.load_values()
        return self._values


def create_field_data(
    source: Union[np.ndarray, int, FieldData], dtype: np.dtype
) -> FieldData:
    if isinstance(source, np.ndarray):
        field_data = NumpyData(source.astype(dtype))
    elif isinstance(source, int):
        field_data = NumpyData(np.ndarray(source, dtype=dtype))
    else:
        field_data = source
    return field_data
