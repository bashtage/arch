from __future__ import absolute_import

try:
    from pandas.api.types import is_datetime64_dtype
except ImportError:
    from pandas.core.common import is_datetime64_dtype

__all__ = ['is_datetime64_dtype']
