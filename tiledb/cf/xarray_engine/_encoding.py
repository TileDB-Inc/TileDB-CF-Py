import numpy as np

import tiledb

# Generic xarray encoding
_UNLIMITED_DIMS_ENCODING = "unlimited_dims"
_FILL_VALUE_ENCODING = "_FillValue"

# TileDB specific variable encoding
_ATTR_NAME_ENCODING = "attr_name"
_ATTR_FILTERS_ENCODING = "filters"
_TILE_SIZES_ENCODING = "tiles"
_MAX_SHAPE_ENCODING = "max_shape"
_DIM_DTYPE_ENCODING = "dim_dtype"


class TileDBVariableEncoder:
    """Class for encoding array variables.

    Parameters
    ----------
    name
        Name of the variable.
    variable
        Xarray variable to encode.
    encoding
        Dictionary of TileDB encoding keywords.
    unlimited_dims
        Unlimited dimensions. Only used if max_shape is not provided in the encoding.
    ctx
        Context object for TileDB operations.
    """

    valid_encoding_keys = {
        _ATTR_FILTERS_ENCODING,
        _ATTR_NAME_ENCODING,
        _DIM_DTYPE_ENCODING,
        _MAX_SHAPE_ENCODING,
        _TILE_SIZES_ENCODING,
    }

    def __init__(self, name, variable, encoding, unlimited_dims, ctx):
        # Set initial class properties.
        self._ctx = ctx
        self._name = name
        self._variable = variable
        self._encoding = dict()

        # Check the input encoding data is valid.
        for key in encoding:
            if key not in self.valid_encoding_keys:
                raise KeyError(
                    "Encoding error on variable '{self._name}'. Invalid encoding key "
                    f"``{key}``."
                )

        # Initialize encoding values.
        try:
            # Set attribute encodings: attr_name and attr_filters.
            self.attr_name = encoding.get(
                _ATTR_NAME_ENCODING,
                f"{self._name}_" if self._name in variable.dims else self._name,
            )
            self.filters = encoding.get(
                _ATTR_FILTERS_ENCODING,
                tiledb.FilterList(
                    (tiledb.ZstdFilter(level=5, ctx=self._ctx),), ctx=self._ctx
                ),
            )

            # Set domain encodings: dim_dtype, tiles, and max_shape.
            self.dim_dtype = encoding.get(_DIM_DTYPE_ENCODING, np.dtype(np.uint32))
            if _MAX_SHAPE_ENCODING in encoding:
                self.max_shape = encoding.get(_MAX_SHAPE_ENCODING)
            else:
                # Set unlimited dimensions to max possible size for datatype and
                # remaining dimensions to the size of the variable dimension.
                unlimited_dims = {
                    dim_name for dim_name in variable.dims if dim_name in unlimited_dims
                }
                unlim = np.iinfo(self.dim_dtype).max
                self.max_shape = tuple(
                    unlim if dim_name in unlimited_dims else dim_size
                    for dim_name, dim_size in zip(variable.dims, variable.shape)
                )
            self.tiles = encoding.get(_TILE_SIZES_ENCODING, None)
        except ValueError as err:
            raise ValueError(f"Encoding error for variable '{self._name}'.") from err

    @property
    def attr_name(self):
        return self._encoding.get(_ATTR_NAME_ENCODING, self._name)

    @attr_name.setter
    def attr_name(self, name):
        if name in self._variable.dims:
            raise ValueError(
                f"Attribute name '{name}' is already a dimension name. Attribute names "
                f"must be unique."
            )
        self._encoding[_ATTR_NAME_ENCODING] = name

    @property
    def dim_dtype(self):
        return self._encoding[_DIM_DTYPE_ENCODING]

    @dim_dtype.setter
    def dim_dtype(self, dim_dtype):
        if dim_dtype.kind not in ("i", "u"):
            raise ValueError(
                f"Dimension dtype ``{dim_dtype}`` is not a valid signed or unsigned "
                f"integer dtype."
            )
        self._encoding[_DIM_DTYPE_ENCODING] = dim_dtype

    @property
    def dtype(self):
        return self._variable.dtype

    @property
    def fill(self):
        fill = self._variable.encoding.get(_FILL_VALUE_ENCODING, None)
        if fill is np.nan:
            return None
        return fill

    @property
    def filters(self):
        return self._encoding[_ATTR_FILTERS_ENCODING]

    @filters.setter
    def filters(self, filters):
        self._encoding[_ATTR_FILTERS_ENCODING] = filters

    def create_array_schema(self):
        """Returns a TileDB attribute from the provided variable and encodings."""
        attr = tiledb.Attr(
            name=self.attr_name,
            dtype=self.dtype,
            fill=self.fill,
            filters=self.filters,
            ctx=self._ctx,
        )
        tiles = self.tiles
        max_shape = self.max_shape
        dims = tuple(
            tiledb.Dim(
                name=dim_name,
                dtype=self.dim_dtype,
                domain=(0, max_shape[index] - 1),
                tile=None if tiles is None else tiles[index],
                ctx=self._ctx,
            )
            for index, dim_name in enumerate(self._variable.dims)
        )
        return tiledb.ArraySchema(
            domain=tiledb.Domain(*dims, ctx=self._ctx),
            attrs=(attr,),
            ctx=self._ctx,
        )

    def get_encoding_metadata(self):
        meta = dict()
        return meta

    @property
    def max_shape(self):
        return self._encoding[_MAX_SHAPE_ENCODING]

    @max_shape.setter
    def max_shape(self, max_shape):
        if len(max_shape) != self._variable.ndim:
            raise ValueError(
                f"Incompatible shape {max_shape} for variable with "
                f"{self._variable.ndim} dimensions."
            )
        if any(
            dim_size < var_size
            for dim_size, var_size in zip(max_shape, self._variable.shape)
        ):
            raise ValueError(
                f"Incompatible max shape {max_shape} for variable with shape "
                f"{self._variable.shape}. Max shape must be greater than or equal "
                f"to the variable shape for all dimensions."
            )
        if (
            _TILE_SIZES_ENCODING in self._encoding
            and self.tiles is not None
            and any(
                dim_size < tile_size
                for tile_size, dim_size in zip(self.tiles, max_shape)
            )
        ):
            raise ValueError(
                f"Incompatible max shape {max_shape} provied for a variable with tiles "
                f"{self.tiles}. Each tile must be less than or equal to the "
                f"max size of the dimension it is on."
            )

        self._encoding[_MAX_SHAPE_ENCODING] = max_shape

    @property
    def encoding(self):
        return self._encoding

    @property
    def tiles(self):
        return self._encoding[_TILE_SIZES_ENCODING]

    @tiles.setter
    def tiles(self, tiles):
        if tiles is not None:
            if len(tiles) != self._variable.ndim:
                raise ValueError(
                    f"Incompatible number of tiles given. {len(tiles)} tiles provided "
                    f"for a variable with {self._variable.ndim} dimensions. There must "
                    f"be exactly one tile per dimension."
                )
            if _MAX_SHAPE_ENCODING in self._encoding and any(
                dim_size < tile_size
                for tile_size, dim_size in zip(tiles, self.max_shape)
            ):
                raise ValueError(
                    f"Incompatible tiles {tiles} provied for a variable with max shape "
                    f"{self.max_shape}. Each tile must be less than or equal to the "
                    f"max size of the dimension it is on."
                )
        self._encoding[_TILE_SIZES_ENCODING] = tiles

    @property
    def variable_name(self):
        return self._name
