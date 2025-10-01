from typing import overload
import warnings

import numpy as np
import pandas as pd

from arch._typing import Float64Array, Literal, NDArrayOrFrame


class ColumnNameConflict(Warning):
    pass


column_name_conflict_doc: str = """\
Some of the column names being added were not unique and have been renamed.

             {0}
"""


def _enforce_unique_col_name(existing: list[str], new: list[str]) -> list[str]:
    converted_names = []
    unique_names = list(new[:])
    for i, n in enumerate(new):
        if n in existing:
            original_name = n
            fixed_name = n
            duplicate_count = 0
            while fixed_name in existing:
                fixed_name = n + "_" + str(duplicate_count)
                duplicate_count += 1
            unique_names[i] = fixed_name
            converted_names.append(f"{original_name}   ->   {fixed_name}")
    if converted_names:

        ws = column_name_conflict_doc.format("\n    ".join(converted_names))
        warnings.warn(ws, ColumnNameConflict, stacklevel=2)

    return unique_names


@overload
def add_trend(
    x: None = ...,
    trend: Literal["n", "c", "t", "ct", "ctt"] = ...,
    prepend: bool = ...,
    nobs: int = ...,
    has_constant: Literal["raise", "add", "skip"] = ...,
) -> Float64Array:  # pragma: no cover
    ...  # pragma: no cover


@overload
def add_trend(
    x: Float64Array,
    trend: Literal["n", "c", "t", "ct", "ctt"] = ...,
    prepend: bool = ...,
    nobs: None = ...,
    has_constant: Literal["raise", "add", "skip"] = ...,
) -> Float64Array:  # pragma: no cover
    ...  # pragma: no cover


@overload
def add_trend(
    x: pd.DataFrame,
    trend: Literal["n", "c", "t", "ct", "ctt"] = ...,
    prepend: bool = ...,
    nobs: None = ...,
    has_constant: Literal["raise", "add", "skip"] = ...,
) -> pd.DataFrame:  # pragma: no cover
    ...  # pragma: no cover


def add_trend(
    x: NDArrayOrFrame | None = None,
    trend: Literal["n", "c", "t", "ct", "ctt"] = "c",
    prepend: bool = False,
    nobs: int | None = None,
    has_constant: Literal["raise", "add", "skip"] = "skip",
) -> Float64Array | pd.DataFrame:
    """
    Adds a trend and/or constant to an array.

    Parameters
    ----------
    x : {ndarray, DataFrame}
        Original array of data. If None, then nobs must be a positive integer
    trend : str {"n", "c","t","ct","ctt"}
        The trend(s) to add. Supported options are:

        * "n" no trend (no-op)
        * "c" add constant only
        * "t" add trend only
        * "ct" add constant and linear trend
        * "ctt" add constant and linear and quadratic trend.
    prepend : bool
        If True, prepends the new data to the columns of x.
    nobs : int, positive
        Positive integer containing the length of the trend series.  Only used
        if x is none.
    has_constant : str {'raise', 'add', 'skip'}
        Controls what happens when trend is 'c' and a constant already
        exists in X. 'raise' will raise an error. 'add' will duplicate a
        constant. 'skip' will return the data without change. 'skip' is the
        default.

    Notes
    -----
    Returns columns as ["ctt","ct","t","c"] whenever applicable. There is
    currently no checking for an existing trend.
    """
    if trend not in ("n", "c", "ct", "ctt", "t"):
        raise ValueError(f"trend {trend} not understood")
    trend_name = trend.lower()
    if (x is None and nobs is None) or (x is not None and nobs is not None):
        raise ValueError("One and only one of x or nobs must be provided.")
    if trend_name == "n":
        if x is not None:
            return x
        assert nobs is not None
        return np.empty((nobs, 0))
    elif trend_name == "c":
        trend_order = 0
    elif trend_name in {"ct", "t"}:
        trend_order = 1
    else:  # trend_name == "ctt":
        trend_order = 2

    if x is not None:
        nobs = len(np.asanyarray(x))
    elif nobs is None or nobs <= 0:
        raise ValueError("nobs must be a positive integer if x is None")
    trend_array = np.vander(np.arange(1, nobs + 1, dtype=np.double), trend_order + 1)
    # put in order ctt
    trend_array = np.fliplr(trend_array)
    if trend_name == "t":
        trend_array = trend_array[:, 1:]
        # check for constant
    if x is None:
        return np.asarray(trend_array, dtype=float)
    x_array = np.asarray(x)
    if "c" in trend_name and np.any(
        np.logical_and(np.ptp(x_array, axis=0) == 0, np.all(x_array != 0, axis=0))
    ):
        if has_constant == "raise":
            raise ValueError("x already contains a constant")
        elif has_constant == "add":
            pass
        elif has_constant == "skip" and trend in ("c", "ct", "ctt"):
            trend_array = trend_array[:, 1:]
    if isinstance(x, pd.DataFrame):
        columns: list[str] = ["const", "trend", "quadratic_trend"]
        if trend_name == "t":
            columns = [columns[1]]
        else:
            columns = columns[0 : trend_order + 1]
        columns = _enforce_unique_col_name([str(col) for col in x.columns], columns)
        trend_array_df = pd.DataFrame(trend_array, index=x.index, columns=columns)
        if prepend:
            x = trend_array_df.join(x)
        else:
            x = x.join(trend_array_df)
    elif prepend:
        x = np.column_stack((trend_array, x))
    else:
        x = np.column_stack((x, trend_array))
    assert x is not None
    return x
