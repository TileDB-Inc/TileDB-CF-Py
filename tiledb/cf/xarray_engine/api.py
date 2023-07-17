from typing import Iterable, Mapping, Optional

import tiledb


def from_xarray(
    dataset,
    group_uri,
    *,
    config: Optional[tiledb.Config] = None,
    ctx: Optional[tiledb.Ctx] = None,
    encoding: Optional[Mapping] = None,
    compute: bool = True,
    create_group: bool = True,
    create_arrays: bool = True,
    copy_group_metadata: bool = True,
    copy_variable_data: bool = True,
    copy_variable_metadata: bool = True,
    region: Optional[Mapping[str, slice]] = None,
    unlimited_dims: Optional[Iterable[str]] = None,
):
    """Writes an xarray dataset to a TileDB group.

    dataset: The xarray Dataset to write.
    group_uri: The URI to the TileDB group to create or append to.
    config: A TileDB config object to use for TileDB objects.
    ctx: A TileDB context object to use for TileDB operations.
    encoding: A nested diction with variable names as keys and dictionaries
        of specific encodings as values. For variables that do not have an
        encoding set here, any encodings already set on the variable will
        be used.
    compute: If ``True`` write array data immediately, otherwise return a
        ``dask.delated.Delayed`` object that can be computed to write
        array data later.

    TODO: Add additional parameters

    region: Optional mapping from dimension names to integer slices along the
        dataset dimensions to indicate the region to write thhis dataset's data in.
        TODO: Fill in details about how region works.
    chunkmanager_store_kwargs: Additional arguments passed on to the
        `ChunkManager.store`.
    unlimited_dims: Optional set of dimensions to treat as unlimited.
    """
    from .writer import from_xarray as from_xarray_impl

    return from_xarray_impl(
        dataset,
        group_uri,
        config=config,
        ctx=ctx,
        encoding=encoding,
        compute=compute,
        create_group=create_group,
        create_arrays=create_arrays,
        copy_group_metadata=copy_group_metadata,
        copy_variable_data=copy_variable_data,
        copy_variable_metadata=copy_variable_metadata,
        region=region,
        unlimited_dims=unlimited_dims,
    )
