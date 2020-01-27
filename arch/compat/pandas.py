try:
    from pandas.api.types import is_datetime64_any_dtype
except ImportError:
    from pandas.core.common import is_datetime64_any_dtype

__all__ = ["is_datetime64_any_dtype"]
