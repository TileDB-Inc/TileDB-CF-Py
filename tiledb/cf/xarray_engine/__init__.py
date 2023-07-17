try:
    import xarray

    has_xarray = True

except ImportError:
    has_xarray = False


from .api import from_xarray

__all__ = ["has_xarray", "from_xarray"]  # type: ignore
