from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
from typing_extensions import Protocol


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
