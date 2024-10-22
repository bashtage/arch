"""
Utility functions that do not explicitly relate to Volatility modeling
"""

from arch.compat.pandas import is_datetime64_any_dtype

from abc import ABCMeta
from collections.abc import Hashable, Sequence
import datetime as dt
from functools import cached_property
from typing import Any, Literal, Optional, Union, overload

import numpy as np
from pandas import DataFrame, DatetimeIndex, Index, NaT, Series, Timestamp, to_datetime

from arch.typing import AnyPandas, ArrayLike, DateLike, NDArray

__all__ = [
    "ensure1d",
    "parse_dataframe",
    "DocStringInheritor",
    "date_to_index",
    "cutoff_to_index",
    "ensure2d",
    "AbstractDocStringInheritor",
    "find_index",
]

deprecation_doc: str = """
{func} has been moved.  Please use {new_location}.{func}.
"""


@overload
def ensure1d(
    x: Union[int, float, Sequence[Union[int, float]], ArrayLike],
    name: Optional[Hashable],
    series: Literal[True] = ...,
) -> Series:  # pragma: no cover
    ...  # pragma: no cover


@overload
def ensure1d(
    x: Union[int, float, Sequence[Union[int, float]], ArrayLike],
    name: Optional[Hashable],
    series: Literal[False],
) -> np.ndarray:  # pragma: no cover
    ...  # pragma: no cover


def ensure1d(
    x: Union[int, float, Sequence[Union[int, float]], ArrayLike],  # noqa: E231
    name: Optional[Hashable],
    series: bool = False,
) -> Union[NDArray, Series]:
    if isinstance(x, Series):
        if not isinstance(x.name, str):
            x.name = str(x.name)
        if series:
            return x
        else:
            return np.asarray(x)

    if isinstance(x, DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} must be squeezable to 1 dimension")
        if not series:
            return np.asarray(x.iloc[:, 0])
        x_series = Series(x.iloc[:, 0], x.index)
        if not isinstance(x_series.name, str):
            x_series.name = str(x_series.name)
        return x_series

    x_arr = np.squeeze(np.asarray(x))
    if x_arr.ndim == 0:
        x_arr = x_arr[None]
    elif x_arr.ndim != 1:
        raise ValueError(f"{name} must be squeezable to 1 dimension")

    if series:
        return Series(x_arr, name=name)
    else:
        return x_arr


def ensure2d(
    x: Union[
        Sequence[Union[float, int]], Sequence[Sequence[Union[float, int]]], ArrayLike
    ],
    name: str,
) -> Union[DataFrame, NDArray]:
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
            raise ValueError("Variable " + name + "must be 2d or reshapable to 2d")
    else:
        raise TypeError("Variable " + name + "must be a Series, DataFrame or ndarray.")


def parse_dataframe(x: Optional[ArrayLike], name: Union[str, list[str]]) -> Union[
    tuple[Index, Index],
    tuple[list[Optional[Hashable]], Index],
    tuple[list[str], NDArray],
]:
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
        if not ("__doc__" in clsdict and clsdict["__doc__"]):
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
    date: Union[str, dt.date, dt.datetime, np.datetime64, Timestamp],
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


def cutoff_to_index(
    cutoff: Union[None, int, DateLike], index: Index, default: int
) -> int:
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


def find_index(s: AnyPandas, index: Union[int, DateLike]) -> int:
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
