from typing import Any, Mapping, Optional, Protocol, Tuple

import numpy as np


class BufferData(Protocol):
    @property
    def data(self) -> np.array:
        ...

    @property
    def dtype(self) -> np.dtype:
        ...

    @property
    def metadata(self) -> Mapping[str, Any]:
        ...

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        ...

    @shape.setter
    def shape(self, new_shape: Tuple[int, ...]):
        ...

    @property
    def size(self) -> int:
        ...


class NumpyBuffer:
    def __init__(
        self, input: np.array, *, metadata: Optional[Mapping[str, Any]] = None
    ):
        self._source_data = input
        self._metadata = dict() if metadata is None else dict(metadata)

    @property
    def data(self):
        return self._source_data

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
