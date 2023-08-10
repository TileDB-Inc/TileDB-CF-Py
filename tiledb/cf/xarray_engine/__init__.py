try:
    import xarray

    has_xarray = True

except ImportError:
    has_xarray = False


from .api import (
    copy_data_from_xarray,
    copy_metadata_from_xarray,
    create_group_from_xarray,
    from_xarray,
)

__all__ = [
    "has_xarray",
    "copy_data_from_xarray",
    "copy_metadata_from_xarray",
    "create_group_from_xarray",
    "from_xarray",
]  # type: ignore
