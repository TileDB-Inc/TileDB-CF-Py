from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple, Union

import netCDF4
import numpy as np

from tiledb.cf.core.source import NumericValue, NumpyRegion

from ._utils import get_netcdf_metadata, open_netcdf_group

if TYPE_CHECKING:
    from pathlib import Path


class NetCDFGroupReader:
    def __init__(
        self,
        default_input_file: Optional[Union[str, Path]] = None,
        default_group_path: Optional[str] = None,
    ):
        self.default_input_file = default_input_file
        self.default_group_path = default_group_path
        self._root_group: Optional[netCDF4.Group] = None
        self._netcdf_group: Optional[netCDF4.Group] = None

    def close(self):
        if self._netcdf_group is not None:
            self._root_group.close()
            self._root_group = None
            self._netcdf_group = None

    def open(
        self,
        input_netcdf_group: Optional[netCDF4.Group] = None,
        input_file: Optional[Union[str, Path]] = None,
        input_group_path: Optional[str] = None,
    ):
        if input_netcdf_group is not None:
            if not isinstance(input_netcdf_group, (netCDF4.Dataset, netCDF4.Group)):
                raise TypeError(
                    f"Invalid input: group={input_netcdf_group} of type "
                    f"{type(input_netcdf_group)} is not a netCDF4 Group or or Dataset."
                )
            self._netcdf_group = input_netcdf_group
            return

        # Get the input file.
        input_file = input_file if input_file is not None else self.default_input_file
        if input_file is None:
            raise ValueError(
                "An input file must be provided; no default input file was set."
            )

        # Get the group path
        group_path = (
            input_group_path
            if input_group_path is not None
            else self.default_group_path
        )
        if group_path is None:
            raise ValueError(
                "A group path must be provided; no default group path was set. Use "
                "``'/'`` for the root group."
            )

        self._root_group = netCDF4.Dataset(input_file)
        self._root_group.set_auto_maskandscale(False)

        self._netcdf_group = self._root_group
        if group_path != "/":
            for child_group_name in group_path.strip("/").split("/"):
                self._netcdf_group = self._netcdf_group.groups[child_group_name]

    def __getitem__(self, variable_name: str):
        if self._netcdf_group is None:
            raise ValueError("NetCDF group is not opened.")
        try:
            variable = self._netcdf_group.variables[self.input_var_name]
        except KeyError as err:  # pragma: no cover
            raise KeyError(
                f"The variable '{self.input_var_name}' was not found in the provided "
                f"NetCDF group."
            ) from err
        return variable

    @property
    def dimensions(self):
        return self.netcdf4.dimensions

    def getncattr(self, key):
        return self.netcdf4.getncattr(key)

    @property
    def parent(self):
        return self.netcdf4.parent

    @property
    def netcdf4(self):
        if self._netcdf_group is None:
            raise ValueError("NetCDF group is not opened.")
        return self._netcdf_group

    def ncattrs(self):
        return self.netcdf4.ncattrs()

    @property
    def variables(self):
        return self.netcdf4.variables


class NetCDF4VariableSource:
    # TODO: from_file
    # TODO: from_group

    # TODO: Make from variable just save the variable, and use from group for
    # NetCDF converter engine
    @classmethod
    def from_variable(
        cls,
        netcdf_variable: netCDF4.Variable,
        netcdf_group: netCDF4.Group,
        *,
        region=None,
        unpack=False,
        copy_metadata=True,
    ):
        fill = get_netcdf_metadata(netcdf_variable, "_FillValue", is_number=True)
        dtype = netcdf_variable.dtype

        if unpack:
            # Get scale factor and add offset.
            scale_factor = get_netcdf_metadata(
                netcdf_variable, "scale_factor", is_number=True
            )
            add_offset = get_netcdf_metadata(
                netcdf_variable, "add_offset", is_number=True
            )

            # Update dtype
            test = np.array(0, dtype=dtype)
            if scale_factor is not None:
                test = scale_factor * test
            if add_offset is not None:
                test = test + add_offset
            dtype = test.dtype

            # Update fill
            if fill is not None:
                fill = scale_factor * fill + add_offset

            # Set encoding keys
            encoding_keys = {"_FillValue", "scale_factor", "add_offset"}
        else:
            scale_factor = None
            add_offset = None
            encoding_keys = {"_FillValue"}

        return cls(
            variable_name=netcdf_variable.name,
            dtype=dtype,
            shape=netcdf_variable.shape,
            netcdf_group=netcdf_group,
            region=region,
            scale_factor=scale_factor,
            add_offset=add_offset,
            fill=fill,
            copy_metadata=copy_metadata,
            encoding_keys=encoding_keys,
        )

    def __init__(
        self,
        variable_name: str,
        dtype: np.dtype,
        shape: Optional[Tuple[Tuple[int, int], ...]],
        *,
        netcdf_group: Optional[netCDF4.Group] = None,
        region: Optional[Tuple[Tuple[int, int], ...]] = None,
        unpack: bool = False,
        add_offset: Optional[NumericValue] = None,
        scale_factor: Optional[NumericValue] = None,
        fill: Optional[NumericValue] = None,
        copy_metadata: bool = True,
        encoding_keys=None,
    ):
        # Variable access information
        self._netcdf_group = netcdf_group
        self._variable_name = variable_name

        # Metadata
        self._copy_metadata = copy_metadata
        self._encoding_keys = set() if encoding_keys is None else encoding_keys

        # CF Conventions
        self._unpack = unpack
        self._scale_factor = scale_factor
        self._add_offset = add_offset
        self._fill = fill

        # Variable information
        self._region = region
        self._dtype = dtype

        if shape is None:
            if region is not None:
                raise ValueError("Cannot specify a region on a scalar variable.")
            self._is_scalar = True
            self._size = 1
            self._shape = (1,)
            self._region = None
        else:
            self._is_scalar = False
            if region is None:
                self._region = None
                self._shape = shape
                self._size = sum(self._shape)
            else:
                self._region = NumpyRegion(region, shape)
                self._size = self._region.size
                self._shape = self._region.shape

    @property
    def add_offset(self) -> NumericValue:
        return self._add_offset

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def fill(self) -> NumericValue:
        self._fill

    def get_metadata(self) -> Mapping[str, Any]:
        variable = self._netcdf_group.variables[self._variable_name]
        if self._copy_metadata:
            self._metadata = {
                key: get_netcdf_metadata(variable, key)
                for key in variable.ncattrs()
                if key not in self._encoding_keys
            }
        else:
            self._metadata = dict()

        if self._metadata is None:
            self.reload()
        return self._metadata

    def get_values(self) -> np.ndarray:
        variable = self._netcdf_group.variables[self._variable_name]
        if self._region is None:
            return variable[...]
        return variable[*self._region.as_slices()]

    @property
    def scale_factor(self) -> NumericValue:
        return self._scale_factor

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        return self._shape

    @property
    def size(self) -> int:
        return self._size
