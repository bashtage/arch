"""
Utility functions that do not explicitly relate to Volatility modeling
"""

from arch.compat.pandas import is_datetime64_any_dtype

from abc import ABCMeta
from collections.abc import Hashable, Sequence
import datetime as dt
from functools import cached_property
from typing import Any, Literal, Union, cast, overload

import numpy as np
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Series,
    Timestamp,
    concat,
    to_datetime,
)

from arch._typing import (
    AnyArray,
    AnyArray1D,
    AnyPandas,
    ArrayLike,
    DateLike,
    Float64Array1D,
    NDArray,
)

__all__ = [
    "AbstractDocStringInheritor",
    "DocStringInheritor",
    "cutoff_to_index",
    "date_to_index",
    "ensure1d",
    "ensure2d",
    "find_index",
    "parse_dataframe",
    "to_array_1d",
]

deprecation_doc: str = """
{func} has been moved.  Please use {new_location}.{func}.
"""


def to_array_1d(x: AnyArray | Series) -> Float64Array1D:
    """
    Ensure array is 1D and float64

    Parameters
    ----------
    x : {ndarray, Series}
        Array to convert

    Returns
    -------
    ndarray
        1D float64 array
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 1 and x.dtype == np.float64:
            return cast("Float64Array1D", x)
        _x = x.squeeze()
        if _x.ndim == 1:
            return cast("Float64Array1D", _x.astype(float))
        elif _x.ndim == 0:
            return cast("Float64Array1D", np.atleast_1d(_x).astype(float, copy=False))
        else:
            raise ValueError("x must be 1D or 1D convertible")
    elif isinstance(x, Series):
        return x.to_numpy().astype(float, copy=False)
    else:
        raise TypeError("x must be a Series or ndarray")


@overload
def ensure1d(
    x: float | Sequence[int | float] | ArrayLike,
    name: Hashable | None,
    series: Literal[True] = ...,
) -> Series:  # pragma: no cover
    ...  # pragma: no cover


@overload
def ensure1d(
    x: float | Sequence[int | float] | ArrayLike,
    name: Hashable | None,
    series: Literal[False],
) -> AnyArray1D:  # pragma: no cover
    ...  # pragma: no cover


def ensure1d(
    x: float | Sequence[int | float] | ArrayLike,
    name: Hashable | None,
    series: bool = False,
) -> AnyArray1D | Series:
    if isinstance(x, Series):
        if not isinstance(x.name, str):
            x.name = str(x.name)
        if series:
            return x
        else:
            return x.to_numpy()

    if isinstance(x, DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} must be squeezable to 1 dimension")
        if not series:
            return x.iloc[:, 0].to_numpy()
        x_series = Series(x.iloc[:, 0], x.index)
        if not isinstance(x_series.name, str):
            x_series.name = str(x_series.name)
        return x_series

    x_arr = np.asarray(x)
    if sum([s > 1 for s in x_arr.shape]) > 1:
        raise ValueError(f"{name} must be squeezable to 1 dimension")
    x_arr = x_arr.ravel()
    if series:
        return Series(x_arr, name=name)
    else:
        return x_arr.ravel()


def ensure2d(
    x: Sequence[float | int] | Sequence[Sequence[float | int]] | ArrayLike,
    name: str,
) -> DataFrame | NDArray:
    if isinstance(x, Series):
        return DataFrame(x)
    elif isinstance(x, DataFrame):
        return x
    elif isinstance(x, np.ndarray):
        if x.ndim == 0:
            return np.asarray([[x]])
        elif x.ndim == 1:
            return x[:, None]
        elif x.ndim == 2:
            return x
        else:
            raise ValueError(f"Variable {name} must be 2d or reshapable to 2d")
    else:
        raise TypeError(f"Variable {name} must be a Series, DataFrame or ndarray.")


def parse_dataframe(
    x: ArrayLike | None, name: str | list[str]
) -> (
    tuple[Index, Index]
    | tuple[list[Hashable | None], Index]
    | tuple[list[str], NDArray]
):
    if x is None:
        assert isinstance(name, str)
        return [name], np.empty(0)
    if isinstance(x, DataFrame):
        return x.columns, x.index
    elif isinstance(x, Series):
        return [x.name], x.index
    else:
        if not isinstance(name, list):
            name = [name]
        return name, np.arange(np.squeeze(x).shape[0])


class DocStringInheritor(type):
    """
    A variation on
    https://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
    by Paul McGuire
    """

    def __new__(
        mcs, name: str, bases: tuple[type, ...], clsdict: dict[str, Any]
    ) -> Any:
        if not (clsdict.get("__doc__")):
            for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()):
                doc = mro_cls.__doc__
                if doc:
                    clsdict["__doc__"] = doc
                    break
        for attr, attribute in clsdict.items():
            if not attribute.__doc__:
                for mro_cls in (
                    mro_cls
                    for base in bases
                    for mro_cls in base.mro()
                    if hasattr(mro_cls, attr)
                ):
                    doc = getattr(mro_cls, attr).__doc__
                    if doc:
                        if isinstance(attribute, cached_property):
                            attribute.func.__doc__ = doc
                            clsdict[attr] = cached_property(attribute.func)
                        elif isinstance(attribute, property):
                            clsdict[attr] = property(
                                attribute.__get__,
                                attribute.__set__,
                                attribute.__delete__,
                                doc,
                            )
                        else:
                            attribute.__doc__ = doc
                        break
        return type.__new__(mcs, name, bases, clsdict)


class ConcreteClassMeta(ABCMeta):
    def __init__(cls, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        missing: list[str] = getattr(cls, "__abstractmethods__", [])
        if missing:
            missing_meth = ", ".join(missing)
            raise TypeError(
                f"{cls.__name__} has not implemented abstract methods {missing_meth}"
            )


class AbstractDocStringInheritor(ConcreteClassMeta, DocStringInheritor):
    pass


def date_to_index(
    date: str | dt.date | dt.datetime | np.datetime64 | Timestamp,
    date_index: Union[DatetimeIndex, NDArray, "Series[Timestamp]"],
) -> int:
    """
    Looks up a date in an array of dates

    Parameters
    ----------
    date : {str, datetime, datetime64, Timestamp}
        Date to use when returning the index
    date_index : {DatetimeIndex, ndarray}
        Index data containing datetime64 values

    Returns
    -------
    int
        Index location

    Notes
    -----
    Assumes dates are increasing and unique.
    """
    if not is_datetime64_any_dtype(date_index):
        raise ValueError("date_index must be a datetime64 array")
    values = (
        date_index.values
        if isinstance(date_index, DatetimeIndex)
        else np.asarray(date_index)
    )
    if not np.all((np.diff(values).astype(dtype=np.int64)) > 0):
        raise ValueError("date_index is not monotonic and unique")
    if not isinstance(date, (dt.datetime, np.datetime64, str, Timestamp)):
        raise ValueError("date must be a datetime, datetime64, Timestamp or string")

    if isinstance(date, Timestamp):
        assert isinstance(date_index, DatetimeIndex)
        if date_index.tzinfo is not None:
            date = date.tz_convert("GMT").tz_localize(None)
        date_64 = date.to_datetime64()
    elif isinstance(date, dt.datetime):
        date_64 = np.datetime64(date)
    elif isinstance(date, str):
        pd_dt = to_datetime(date, errors="coerce")
        if pd_dt is NaT:
            raise ValueError(f"date: {date} cannot be parsed to a date.")
        assert isinstance(pd_dt, Timestamp)
        date_64 = pd_dt.to_datetime64()
    else:
        assert isinstance(date, np.datetime64)
        date_64 = date

    if isinstance(date_index, DatetimeIndex):
        if date_index.tzinfo is not None:
            date_index = date_index.tz_convert("GMT").tz_localize(None)
        date_index = date_index.to_numpy()

    date_index = np.asarray(date_index)

    locs = np.nonzero(date_index <= date_64)[0]
    if locs.shape[0] == 0:
        return 0

    loc = locs.max()
    in_array = np.any(date_index == date_64)
    if not in_array:
        loc += 1

    return int(loc)


def cutoff_to_index(cutoff: int | DateLike | None, index: Index, default: int) -> int:
    """
    Converts a cutoff to a numerical index

    Parameters
    ----------
    cutoff : {None, str, datetime, datetime64, Timestamp)
        The cutoff point to use
    index : DatetimeIndex
        Pandas index
    default : int
        The value to return if cutoff is None

    Returns
    -------
    int
        Integer value where
    """
    int_index = default
    if isinstance(cutoff, (str, dt.datetime, np.datetime64, Timestamp)):
        assert isinstance(index, DatetimeIndex)
        int_index = date_to_index(cutoff, index)
    elif isinstance(cutoff, int) or issubclass(cutoff.__class__, np.integer):
        assert cutoff is not None
        int_index = int(cutoff)

    return int_index


def find_index(s: AnyPandas, index: int | DateLike) -> int:
    """
    Returns the numeric index for a string or datetime

    Parameters
    ----------
    s : Series or DataFrame
        Series or DataFrame to use in lookup
    index : datetime-like, str
        Index value, either a string convertible to a datetime or a datetime

    Returns
    -------
    int
        Integer location of index value
    """
    if isinstance(index, (int, np.int64)):
        return int(index)
    assert isinstance(index, (str, dt.datetime, np.datetime64, Timestamp))
    date_index = to_datetime(index, errors="coerce")
    # TODO: Bug in pandas-stubs does not return correct types when errors=coerce
    if date_index is NaT:
        raise ValueError(f"{index} cannot be converted to datetime")
    loc = np.argwhere(s.index == date_index).squeeze()
    if loc.size == 0:
        raise ValueError("index not found")
    return int(loc)


def _is_real_scalar(value: Any) -> bool:
    """Check for a real scalar. Booleans are not treated as scalars here"""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _conform_new_to_array(original: NDArray, new: Any) -> NDArray:
    """
    Reshape new observations so that they can be stacked below original

    Parameters
    ----------
    original : ndarray
        The existing data. Must be 1-d or 2-d.
    new : scalar or array_like
        The new data. Scalars are treated as a single observation, 1-d data
        is treated as multiple observations unless original has more than one
        column, in which case it is treated as one observation of all columns.

    Returns
    -------
    ndarray
        New data with the same number of dimensions as original, and, if 2-d,
        the same number of columns as original.
    """
    new_arr = np.asarray(new)
    if original.ndim == 1:
        if sum(s > 1 for s in new_arr.shape) > 1:
            raise ValueError(
                "The appended data must be 1-dimensional since the original data "
                f"is 1-dimensional. Got an array with shape {new_arr.shape}."
            )
        return new_arr.ravel()
    ncol = original.shape[1]
    if new_arr.ndim == 0 or (new_arr.ndim == 1 and ncol == 1):
        new_arr = new_arr.reshape((-1, 1))
    elif new_arr.ndim == 1:
        # A single observation containing a value for every column
        new_arr = new_arr.reshape((1, -1))
    if new_arr.ndim != 2 or new_arr.shape[1] != ncol:
        raise ValueError(
            "The appended data must have the same number of columns as the "
            f"original data ({ncol}). Got an array with shape {new_arr.shape}."
        )
    return new_arr


def append_same_type(
    original: ArrayLike | list[Any] | tuple[Any, ...],
    new: ArrayLike | list[Any] | tuple[Any, ...] | float,
) -> Any:
    """
    Append observations to existing data, preserving the type of the data

    Parameters
    ----------
    original : {ndarray, Series, DataFrame, list, tuple}
        The existing data.
    new : {ndarray, Series, DataFrame, list, tuple, float}
        The data to append. This must have the same type as original, unless
        original is an ndarray, list or tuple, in which case a scalar can be
        appended as a single observation. 1-d data appended to a 2-d ndarray
        with more than one column is treated as a single observation.

    Returns
    -------
    {ndarray, Series, DataFrame, list, tuple}
        The extended data. Has the same type as original. The inputs are never
        modified.

    Raises
    ------
    TypeError
        If the types of original and new are not compatible.
    ValueError
        If new does not contain any observations, has a different number of
        columns from original, or, when using a DataFrame, has different
        column labels.
    """
    if not isinstance(original, (Series, DataFrame, list, tuple, np.ndarray)):
        raise TypeError(
            "The original data must be a pandas Series, DataFrame, numpy ndarray, "
            f"list or tuple. Got {type(original)}."
        )
    scalar = _is_real_scalar(new)
    scalar_ok = isinstance(original, (list, tuple, np.ndarray))
    if not ((scalar and scalar_ok) or isinstance(new, type(original))):
        raise TypeError(
            "Input data must be the same type as the original data, unless the "
            "original was an ndarray, list or tuple, in which case the input data "
            f"can be a scalar. Got {type(new)}, expected {type(original)}."
        )
    if not scalar and np.size(new) == 0:
        raise ValueError("Input data must contain at least one observation.")

    extended: Any
    if isinstance(original, Series):
        extended = concat([original, cast("Series", new)], axis=0)
        # concat drops the name when the names differ, e.g., if new is unnamed
        extended.name = original.name
    elif isinstance(original, DataFrame):
        new_frame = cast("DataFrame", new)
        new_columns = new_frame.columns
        if not original.columns.equals(new_columns):
            raise ValueError(
                "The columns of the appended data must match the columns of the "
                f"original data. Got {list(new_columns)}, expected "
                f"{list(original.columns)}."
            )
        extended = concat([original, new_frame], axis=0)
    elif isinstance(original, np.ndarray):
        _original = np.atleast_1d(original)
        if _original.ndim > 2:
            raise ValueError("The original data must be 1-d or 2-d.")
        extended = np.concatenate(
            (_original, _conform_new_to_array(_original, new)), axis=0
        )
    else:
        # list or tuple, converted to an array to validate shapes
        _original = np.asarray(original, dtype=float)
        if _original.ndim > 2:
            raise ValueError("The original data must be 1-d or 2-d.")
        new_rows = _conform_new_to_array(_original, np.asarray(new, dtype=float))
        rows = new_rows.tolist()
        if _original.ndim == 2:
            rows = [type(original)(row) for row in rows]
        extended = type(original)(list(original) + rows)
    return extended
