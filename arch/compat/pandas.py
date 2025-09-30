from typing import TYPE_CHECKING

from packaging.version import parse
import pandas as pd

if TYPE_CHECKING:
    from pandas.api.types import is_datetime64_any_dtype
else:
    try:
        from pandas.api.types import is_datetime64_any_dtype
    except ImportError:
        from pandas.core.common import is_datetime64_any_dtype

PD_LT_22 = parse(pd.__version__) < parse("2.1.99")
MONTH_END = "M" if PD_LT_22 else "ME"

__all__ = ["MONTH_END", "is_datetime64_any_dtype"]
