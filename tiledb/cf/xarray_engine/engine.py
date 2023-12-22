"""Module for xarray backend plugin using the TileDB-Xarray Convention.

Example:
  Open a TileDB group with the xarray engine::

    import xarray as xr
    dataset = xr.open_dataset(
        "dataset.tiledb",
        backend_kwargs={"Ctx": ctx},
        engine="tiledb"
    )


"""
from __future__ import annotations

import os
import warnings
from typing import ClassVar, Iterable

from xarray.backends.common import BACKEND_ENTRYPOINTS, BackendEntrypoint
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.dataset import Dataset
from xarray.core.utils import close_on_error

import tiledb

from ._backend_store import TileDBXarrayStore
from ._deprecated_backend_store import TileDBDataStore


class TileDBXarrayBackendEntrypoint(BackendEntrypoint):
    """TileDB backend for xarray."""

    open_dataset_parameters: ClassVar[tuple | None] = [
        "filename_or_obj",
        "config",
        "ctx",
        "timestamp",
    ]
    description: ClassVar[
        str
    ] = "TileDB backend for xarray for opening TileDB arrays and groups"
    url: ClassVar[str] = "https://github.com/TileDB-Inc/TileDB-CF-Py"

    def open_dataset(
        self,
        filename_or_obj,
        *,
        config=None,
        ctx=None,
        timestamp=None,
        use_deprecated_engine=None,
        key=None,
        encode_fill=None,
        coord_dims=None,
        open_full_domain=None,
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

        Parameters
        ----------
        filename_or_obj
            TileDB URI for the group or array to open in xarray.
        config
            TileDB config object to pass to TileDB objects.
        ctx
            TileDB context to use for TileDB operations.
        timestamp
            Timestamp to open the TileDB array at. Not valid for groups.
        key
            [Deprecated] Encryption key to use for the backend array.
        encode_fill
            [Deprecated] Encode the TileDB fill value.
        coord_dims
            [Deprecated] List of dimensions to convert to coordinates.
        open_full_domain
            [Deprecated] Open the full TileDB domain instead of the non-empty domain.
        mask_and_scale
            xarray decoder that masks fill value and applies float-scale filter using
            variable metadata.
        decode_times
            xarray decoder that converts variables with NetCDF CF-Convention time
            metadata to a numpy.datetime64 datatype.
        concat_characters
            xarray decoder not supported by TileDB.
        decode_coords
            xarray decoder that controls which variables are set as coordinate
            variables.
        drop_variables
            A variable or list of variables to exclude from being opened from the
            dataset.
        use_cftime
            xarray decoder option. Uses cftime for datetime decoding.
        decode_timedelta
            xarray decoder that converts variables with time units to a
            numpy.timedelta64 datatype.
        """

        deprecated_kwargs = {
            "key": key,
            "encode_fill": encode_fill,
            "open_full_domain": open_full_domain,
        }

        # If deprecated keyword aguments were set, then switch to the deprecated engine.
        if use_deprecated_engine is None:

            def check_use_deprecated(key_name, key_value):
                if key_value is not None:
                    warnings.warn(
                        f"Deprecated keyword '{key_name}' provided; deprecated engine "
                        f"is enabled.",
                        DeprecationWarning,
                        stacklevel=1,
                    )
                    return True

            use_deprecated_engine = any(
                check_use_deprecated(key, val)
                for (key, val) in deprecated_kwargs.items()
            )

        # Use the deprecated xarray engine for opening the array.
        if use_deprecated_engine:
            warnings.warn(
                "Using deprecated TileDB-Xarray plugin",
                DeprecationWarning,
                stacklevel=1,
            )

            # Create the deprecated store.
            encode_fill = False if encode_fill is None else encode_fill
            open_full_domain = False if open_full_domain is None else open_full_domain
            datastore = TileDBDataStore(
                uri=filename_or_obj,
                key=key,
                timestamp=timestamp,
                ctx=ctx,
                encode_fill=encode_fill,
                open_full_domain=open_full_domain,
                coord_dims=coord_dims,
            )

            # Use xarray indirection to open dataset defined in a plugin.
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

        # Using new engine: warn if any deprecated keyword arguments were set.
        for arg_name, arg_value in deprecated_kwargs.items():
            if arg_value is not None:
                warnings.warn(
                    f"Skipping deprecated keyword '{arg_name}' used when "
                    f"`use_deprecated_engine=False`.",
                    DeprecationWarning,
                    stacklevel=1,
                )

        # Create the TileDB backend store.
        datastore = TileDBXarrayStore(
            filename_or_obj, config=config, ctx=ctx, timestamp=timestamp
        )

        # Use xarray indirection to open dataset defined in a plugin.
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

    def guess_can_open(self, filename_or_obj) -> bool:
        """Check for datasets that can be opened with this backend."""
        if isinstance(filename_or_obj, (str, os.PathLike)):
            _, ext = os.path.splitext(filename_or_obj)
            if ext in {".tiledb", ".tdb"}:
                return True
        try:
            return tiledb.object_type(filename_or_obj) in {"array", "group"}
        except tiledb.TileDBError:
            return False


BACKEND_ENTRYPOINTS["tiledb"] = ("tiledb", TileDBXarrayBackendEntrypoint)
