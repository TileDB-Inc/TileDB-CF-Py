"""Module for xarray backend plugin using the TileDB-Xarray Convention.

This plugin will only open groups using the TileDB-Xarray Convention. It has
stricter requirements for the TileDB group and array structures than standard
TileDB. See spec `tiledb-xr-spec.md` in project root.

Example:
  Open a TileDB group with the xarray engine::

    import xarray as xr
    dataset = xr.open_dataset(
        "dataset.tiledb-xr",
        backend_kwargs={"Ctx": ctx},
        engine="tiledb"
    )


"""

import os
import warnings
from typing import Iterable, ClassVar

import tiledb
from xarray.backends.common import BACKEND_ENTRYPOINTS, BackendEntrypoint
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.utils import close_on_error
from xarray.core.dataset import Dataset

from .deprecated_backend_store import TileDBDataStore
from .backend_store import TileDBXarrayStore


class TileDBXarrayBackendEntrypoint(BackendEntrypoint):
    """
    TODO: Add docs for TileDBXarrayBackendEntrypoint
    """

    open_dataset_parameters: ClassVar[tuple | None] = [
        "filename_or_obj",
        "drop_variables",
        "config",
        "ctx",
    ]
    description: ClassVar[
        str
    ] = "TileDB backend for xarray using the TileDB-Xarray specification"
    url: ClassVar[str] = "https://github.com/TileDB-Inc/TileDB-CF-Py"

    def open_dataset(
        self,
        filename_or_obj,
        *,
        config=None,
        ctx=None,
        timestamp=None,
        open_full_domain=False,
        use_deprecated_engine=False,
        key=None,
        encode_fill=False,
        coord_dims=None,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables: str | Iterable[str] | None = None,
        use_cftime=None,
        decode_timedelta=None,
    ) -> Dataset:
        """
        Open a TileDB group or array as an xarray dataset.


        TODO: Full description of usage.

        Parameters
        ----------
        filename_or_obj: TileDB URI for the group or array to open in xarray.
        config: TileDB config object to pass to TileDB objects.
        ctx: TileDB context to use for TileDB operations.
        timestamp: Timestamp to use opening a TileDB array. Not supported on groups.
        open_full_domain: If ``True``, use the full dimension to define the size of
            dimensions that aren't otherwise specified. If ``False``, use the non-empty
            domain of the array (computed when the dataset is first loaded).
        key: [Deprecated] Encryption key to use for the backend array.
        encode_fill: [Deprecated] Encode the TileDB fill using `_FillValue` if ``True``.
        coord_dims: [Deprecated] Interpret the following dimension as coordinates.
        """

        if use_deprecated_engine:
            warnings.warn(
                "Using deprecated TileDB-Xarray plugin",
                DeprecationWarning,
                stacklevel=1,
            )
            datastore = TileDBDataStore(
                uri=filename_or_obj,
                key=key,
                timestamp=timestamp,
                ctx=ctx,
                encode_fill=encode_fill,
                open_full_domain=open_full_domain,
                coord_dims=coord_dims,
            )
            # Xarray indirection to open dataset defined in a plugin.
            store_entrypoint = StoreBackendEntrypoint()
            with close_on_error(datastore):
                dataset = store_entrypoint.open_dataset(
                    datastore,
                    mask_and_scale=mask_and_scale,
                    decode_times=decode_times,
                    concat_characters=concat_characters,
                    decode_coords=decode_coords,
                    drop_variables=drop_variables,
                    use_cftime=use_cftime,
                    decode_timedelta=decode_timedelta,
                )
            return dataset

        # Warn if a deprecated keyword was used.
        if key is not None:
            warnings.warn(
                "Deprecated keyword 'key' provided. To use the deprecated backend "
                "engine set `use_deprecated_engine=True`",
                DeprecationWarning,
                stacklevel=1,
            )
        if encode_fill:
            warnings.warn(
                "Deprecated keyword 'encode_fill' provided. To use the deprecated "
                "backend engine set `use_deprecated_engine=True`",
                DeprecationWarning,
                stacklevel=1,
            )
        if coord_dims is not None:
            warnings.warn(
                "Deprecated keyword 'coord_dims' provided. To use the deprecated  "
                "backend engine set `use_deprecated_engine=True`",
                DeprecationWarning,
                stacklevel=1,
            )

        try:
            datastore = TileDBXarrayStore(
                filename_or_obj, config=config, ctx=ctx, timestamp=timestamp
            )

            # Xarray indirection to open dataset defined in a plugin.
            store_entrypoint = StoreBackendEntrypoint()
            with close_on_error(datastore):
                dataset = store_entrypoint.open_dataset(
                    datastore,
                    mask_and_scale=mask_and_scale,
                    decode_times=decode_times,
                    concat_characters=concat_characters,
                    decode_coords=decode_coords,
                    drop_variables=drop_variables,
                    use_cftime=use_cftime,
                    decode_timedelta=decode_timedelta,
                )
            return dataset
        except ValueError as err:
            raise ValueError(
                "Failed to open with current TileDB-xarray backend. To use the "
                "old TileDB-xarray backend set `use_deprecated_engine=True`"
            ) from err

    def guess_can_open(self, filename_or_obj) -> bool:
        """Check for datasets that can be opened with this backend."""
        if isinstance(filename_or_obj, (str, os.PathLike)):
            _, ext = os.path.splitext(filename_or_obj)
            if ext in {".tiledb", ".tdb", ".tdb-xr"}:
                return True
        try:
            return tiledb.object_type(filename_or_obj) in {"array", "group"}
        except tiledb.TileDBError:
            return False


BACKEND_ENTRYPOINTS["tiledb"] = ("tiledb", TileDBXarrayBackendEntrypoint)
