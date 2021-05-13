# Copyright 2021 TileDB Inc.
# Licensed under the MIT License.
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class NetCDF4TestCase:
    """Dataclass the holds values required to generate NetCDF test cases

    name: name of the test case
    dimension_args: sequence of arguments required to create NetCDF4 dimensions
    variable_args: sequence of arguments required to create NetCDF4 variables
    variable_data: dict of variable data by variable name
    variable_matadata: dict of variable metadata key-value pairs by variable name
    group_metadata: group metadata key-value pairs
    """

    name: str
    dimension_args: Sequence[Tuple[str, Optional[int]]]
    variable_args: Sequence[Tuple[str, np.dtype, Tuple[str, ...]]]
    variable_data: Dict[str, np.ndarray]
    variable_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    group_metadata: Dict[str, Any] = field(default_factory=dict)
