try:
    # Prefer system installed version if available
    from cached_property import cached_property
except ImportError:
    from arch.vendor.cached_property import cached_property

__all__ = ['cached_property']
