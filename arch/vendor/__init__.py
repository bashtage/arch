import sys

if sys.version_info >= (3, 8):
    from functools import cached_property
else:
    try:
        # Prefer system installed version if available
        from property_cached import cached_property
    except ImportError:
        from arch.vendor.property_cached import cached_property


__all__ = ["cached_property"]

