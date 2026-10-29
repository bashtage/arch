"""
Utility functions that do not explicitly relate to Volatility modeling.
"""

from arch.compat.pandas import is_datetime64_any_dtype

from abc import ABCMeta
from collections.abc import Hashable, Sequence
import datetime as dt
from functools import cached_property
from typing import Any, Literal, NoReturn, Union, cast, overload

import numpy as np
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Period,
    PeriodIndex,
    Series,
    Timestamp,
    concat,
    to_datetime,
)
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_float_dtype,
    is_integer_dtype,
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


def to_array_1d(x: AnyArray | Series) -> Float64Array1D:
    """
    Ensure an array is 1D and float64.

    Parameters
    ----------
    x : ndarray or Series
        Array to convert. Integer, Boolean and float32 data are converted to
        float64.

    Returns
    -------
    ndarray
        1D float64 array.

    Raises
    ------
    ValueError
        If x is an ndarray that cannot be converted to 1D.
    TypeError
        If x is not a Series or an ndarray, or its data are float16, complex or
        cannot be converted to float64.
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 1 and x.dtype == np.float64 and x.dtype.char == "d":
            return cast("Float64Array1D", x)
        _x = x.squeeze()
        if _x.ndim == 1:
            return cast("Float64Array1D", as_float64(_x, "x"))
        elif _x.ndim == 0:
            return cast("Float64Array1D", np.atleast_1d(as_float64(_x, "x")))
        else:
            # Imported here because the validation module imports this module
            from arch.utility._validation import _describe, _fail  # noqa: PLC0415

            _fail(ValueError, "x", "1D or 1D convertible", _describe(x))
    elif isinstance(x, Series):
        return as_float64(x.to_numpy(), "x")
    else:
        from arch.utility._validation import _describe, _fail  # noqa: PLC0415

        _fail(TypeError, "x", "a Series or ndarray", _describe(x))


_REAL_NUMBERS = "real numbers (integer, Boolean, float32 or float64 data)"
_INEXACT_NUMBERS = (
    "real or complex numbers (integer, Boolean, float32, float64, complex64 or "
    "complex128 data)"
)


def _numbers_rule(allow_complex: bool) -> str:
    """The rule for the data in a message: the numbers and the data types accepted."""
    return _INEXACT_NUMBERS if allow_complex else _REAL_NUMBERS


def _rejected_dtype(dtype: Any, allow_complex: bool) -> bool:
    """
    Whether the data type of numeric data is not accepted.

    This is the only definition of which numeric data types are rejected: float16,
    which has 11 bits of precision and is never accepted, and complex types, which
    are accepted only if ``allow_complex``. Every other numeric type is converted:
    integer, unsigned integer, Boolean and real floating point types to float64
    (float32 is promoted, since a calculation in float32 loses accuracy that the
    estimators need), and complex types to complex128.
    """
    dtype = getattr(dtype, "numpy_dtype", dtype)  # pandas extension types
    kind = getattr(dtype, "kind", "")
    if kind == "f":
        return bool(dtype.itemsize < 4)
    return kind == "c" and not allow_complex


def _dtype_error(
    name: str, dtype: Any, shape: tuple[int, ...], x: Any, allow_complex: bool
) -> NoReturn:
    """Raise the TypeError of data that have a data type that is not accepted."""
    dtype = getattr(dtype, "numpy_dtype", dtype)
    raise TypeError(
        f"{name} must be array_like of {_numbers_rule(allow_complex)}; got "
        f"{dtype} data with shape {shape} ({type(x).__name__})"
    )


def _check_pandas_dtypes(
    x: Series | DataFrame, dtypes: Sequence[Any], name: str, allow_complex: bool
) -> None:
    """Reject a Series, or a DataFrame with a column, of a data type not accepted."""
    for dtype in dtypes:
        if _rejected_dtype(dtype, allow_complex):
            _dtype_error(name, dtype, x.shape, x, allow_complex)


def as_float64(x: Any, name: str, *, allow_complex: bool = False) -> Any:
    """
    Convert array_like data to a float64 ndarray.

    Parameters
    ----------
    x : array_like
        The data. Integer, unsigned integer, Boolean, float32 and other real
        floating point data are converted to float64. Object data are converted
        if every element is a number, and are real if no element has an
        imaginary part different from 0, even if the elements are complex
        numbers. float16 data are not accepted.
    name : str
        The name of the variable. Used in error messages.
    allow_complex : bool, optional
        If True, complex data (complex64, complex128 or object data with an
        element that has an imaginary part different from 0) are converted to
        complex128. Real data are still converted to float64. If False, complex
        data are not accepted.

    Returns
    -------
    ndarray
        The data as float64, or as complex128 if allow_complex and the data
        are complex, with the shape of x. This is x if x is an ndarray with
        dtype float64 or, if allow_complex, complex128.

    Raises
    ------
    TypeError
        If the data cannot be converted to float64 (or complex128 if
        allow_complex), e.g., if the data are None or contain None, strings,
        dates or objects that are not numbers, or if they are float16, or
        complex unless allow_complex.

    Notes
    -----
    NumPy converts None to NaN when it converts an object array, so that
    ``np.asarray(None).astype(float)`` is NaN. None is rejected here, as a value
    and as an element of an object array, so that a missing argument or value
    is not turned into a NaN.

    float32 data are converted to float64 and the calculations of the
    estimators are done in float64, so that their results do not depend on
    whether the data were stored in 32 or 64 bits. float16 data are rejected,
    since its 11 bits of precision (about 3 decimal digits) are not enough for
    any estimator; convert it with ``astype(np.float32)`` or
    ``astype(np.float64)`` if that is intended.
    """
    if x is None:
        raise TypeError(
            f"{name} must be array_like data that can be converted to float64; "
            "got None (NoneType)"
        )
    if isinstance(x, DataFrame):
        # The columns can have different data types, which np.asarray merges
        _check_pandas_dtypes(x, list(x.dtypes), name, allow_complex)
    arr = np.asarray(x)
    # The character is "d" for float64 only. np.longdouble is equal to np.float64 where
    # it has 64 bits (Windows) but it is a different type that LAPACK does not accept.
    if arr.dtype == np.float64 and arr.dtype.char == "d":
        return arr
    kind = arr.dtype.kind
    if kind in "fc":
        if _rejected_dtype(arr.dtype, allow_complex):
            _dtype_error(name, arr.dtype, arr.shape, x, allow_complex)
        if _is_exactly(arr.dtype, np.complex128):
            return arr
        return arr.astype(np.complex128 if kind == "c" else np.float64)
    if kind in "biu":
        return arr.astype(np.float64)
    if kind == "O":
        return _object_values(arr, x, name, allow_complex)
    raise TypeError(
        f"{name} must be array_like with integer, Boolean or floating point "
        f"values that can be converted to float64. Received an array with shape "
        f"{arr.shape} and data type {arr.dtype}."
    )


def _object_values(
    values: np.ndarray,
    x: Any,
    name: str,
    allow_complex: bool,
    none_is_nan: bool = False,
) -> np.ndarray:
    """
    Convert object data to float64, or to complex128 if an element is complex.

    This is the only conversion of object data. It is used for object arrays and for
    the object columns of a Series or DataFrame (``x`` is used in the messages).

    The values are cast to complex128, which accepts every real and complex number,
    such as Python, NumPy and Decimal numbers, so that an element that is not a number
    raises. If no imaginary part is different from 0, the real parts are returned as
    float64. Otherwise the data are complex, and they are returned as complex128 if
    ``allow_complex`` and are rejected if not.

    None is rejected, so that a missing argument or value does not become a NaN,
    unless ``none_is_nan``. pandas data use it, since pandas treats None as missing
    and converts it to NaN. NumPy casts None to a complex NaN, which would be complex
    data, so None is found before the cast.
    """
    if isinstance(x, (Series, DataFrame)):
        source = f"a {type(x).__name__} with shape {x.shape}"
    else:
        source = f"an object array with shape {values.shape}"
    try:
        missing = np.equal(values, None).astype(bool)
    except (TypeError, ValueError):
        # An element with an unusual comparison is not None, and the cast rejects it
        missing = np.zeros(values.shape, dtype=bool)
    if np.any(missing):
        if not none_is_nan:
            raise TypeError(
                f"{name} must contain real numbers that can be converted to "
                f"float64; got None in {source} ({type(x).__name__})"
            )
        values = np.where(missing, np.nan, values)
    try:
        as_complex = values.astype(np.complex128)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must contain real numbers that can be converted to float64. "
            f"Received {source} that contains elements that cannot be converted "
            f"({exc})."
        ) from exc
    if not np.any(as_complex.imag):
        return np.ascontiguousarray(as_complex.real)
    if not allow_complex:
        raise TypeError(
            f"{name} must contain real numbers that can be converted to float64; "
            f"got complex numbers in {source} ({type(x).__name__})"
        )
    return as_complex


def _is_exactly(dtype: Any, target: type) -> bool:
    """
    Whether a data type is float64 (or complex128), and not just equal to it.

    ``np.longdouble`` is equal to ``np.float64`` where long double has 64 bits
    (Windows), but it is a different type that pandas does not convert, and LAPACK
    only accepts float64. The type character is "d" for float64 and "D" for
    complex128.
    """
    return bool(
        isinstance(dtype, np.dtype)
        and dtype == target
        and dtype.char == np.dtype(target).char
    )


def _astype_pandas(
    x: Series | DataFrame, target: type, allow_complex: bool, name: str
) -> tuple[Any, type]:
    """
    Convert a Series or DataFrame using astype.

    The object columns are converted by ``_object_values``, and the target is
    complex128 if one of them is complex. Returns the converted object and the data
    type that it was converted to.
    """
    dtypes = list(x.dtypes) if isinstance(x, DataFrame) else [x.dtype]
    objects = {
        i: _object_values(
            (x if isinstance(x, Series) else x.iloc[:, i]).to_numpy(),
            x,
            name,
            allow_complex,
            none_is_nan=True,
        )
        for i, dtype in enumerate(dtypes)
        if isinstance(dtype, np.dtype) and dtype.kind == "O"
    }
    if any(values.dtype.kind == "c" for values in objects.values()):
        target = np.complex128
    try:
        if not objects:
            return x.astype(target), target
        if isinstance(x, Series):
            return Series(objects[0].astype(target), index=x.index, name=x.name), target
        # Columns by position, since the labels of the columns can be duplicated
        columns = [
            objects[i] if i in objects else x.iloc[:, i].astype(target).to_numpy()
            for i in range(x.shape[1])
        ]
        values: np.ndarray = np.column_stack(columns).astype(target)
        return DataFrame(values, index=x.index, columns=x.columns), target
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must contain real numbers that can be converted to float64. "
            f"Received a {type(x).__name__} with shape {x.shape} that contains "
            f"elements that cannot be converted ({exc})."
        ) from exc


def _float64_pandas(
    x: Series | DataFrame, name: str, allow_complex: bool = False
) -> Any:
    """
    Convert a Series or DataFrame to float64, keeping the index and labels.

    Parameters
    ----------
    x : Series or DataFrame
        The data.
    name : str
        The name of the variable. Used in error messages.
    allow_complex : bool, optional
        If True, complex data are converted to complex128, and all the columns of
        a DataFrame with a complex column are converted to complex128.

    Returns
    -------
    Series or DataFrame
        x if every column is a float64 NumPy dtype (complex128 if allow_complex and
        a column is complex). Otherwise a new object, so that the input is never
        modified.

    Raises
    ------
    TypeError
        If the data cannot be converted to real float64 values (or complex128 if
        allow_complex), or they are float16.
    """
    dtypes: list[Any] = list(x.dtypes) if isinstance(x, DataFrame) else [x.dtype]
    if all(_is_exactly(dtype, np.float64) for dtype in dtypes):
        return x
    _check_pandas_dtypes(x, dtypes, name, allow_complex)
    target: type = (
        np.complex128
        if allow_complex
        and any(isinstance(dtype, np.dtype) and dtype.kind == "c" for dtype in dtypes)
        else np.float64
    )
    if all(_is_exactly(dtype, target) for dtype in dtypes):
        return x
    if not all(
        is_float_dtype(dtype)
        or is_integer_dtype(dtype)
        or is_bool_dtype(dtype)
        or (allow_complex and is_complex_dtype(dtype))
        or (isinstance(dtype, np.dtype) and dtype.kind == "O")
        for dtype in dtypes
    ):
        raise TypeError(
            f"{name} must contain integer, Boolean or floating point values that "
            f"can be converted to float64. Received a {type(x).__name__} with "
            f"shape {x.shape} and data type(s) {', '.join(map(str, set(dtypes)))}."
        )
    converted, target = _astype_pandas(x, target, allow_complex, name)
    new_dtypes = (
        list(converted.dtypes) if isinstance(x, DataFrame) else [converted.dtype]
    )
    if not all(_is_exactly(dtype, target) for dtype in new_dtypes):
        # pandas does not convert longdouble to float64 when they are equal
        values = converted.to_numpy().astype(target)
        if isinstance(converted, DataFrame):
            return DataFrame(values, index=converted.index, columns=converted.columns)
        return Series(values, index=converted.index, name=converted.name)
    return converted


@overload
def ensure1d(
    x: float | Sequence[int | float] | ArrayLike,
    name: Hashable | None,
    series: Literal[False] = ...,
    *,
    allow_complex: bool = ...,
) -> AnyArray1D:  # pragma: no cover
    ...  # pragma: no cover


@overload
def ensure1d(
    x: float | Sequence[int | float] | ArrayLike,
    name: Hashable | None,
    series: Literal[True],
    *,
    allow_complex: bool = ...,
) -> Series:  # pragma: no cover
    ...  # pragma: no cover


def ensure1d(
    x: float | Sequence[int | float] | ArrayLike,
    name: Hashable | None,
    series: bool = False,
    *,
    allow_complex: bool = False,
) -> AnyArray1D | Series:
    """
    Convert data to a 1-d float64 ndarray or Series.

    Parameters
    ----------
    x : array_like, Series or DataFrame
        The data to convert. A DataFrame must have a single column, and any
        other input must have at most one dimension with more than one element.
        Integer, unsigned integer, Boolean and float32 data are converted to
        float64. float16 data are not accepted.
    name : Hashable or None
        The name of the variable. Used in error messages and as the name of
        the Series if x is not a Series or a DataFrame and series is True.
    series : bool, optional
        If True, return a Series. If False, return an ndarray.
    allow_complex : bool, optional
        If True, complex data are converted to complex128 and real data to
        float64. If False, complex data are not accepted.

    Returns
    -------
    ndarray or Series
        The 1-d data. A Series if series is True and an ndarray otherwise. The
        dtype is float64, or complex128 if allow_complex and x is complex. The
        index of a Series is kept. x is returned, or is a view of x, if x is
        float64 data (or complex128 data if allow_complex). The name of a Series
        that is not a str is converted to a str.

    Raises
    ------
    ValueError
        If x cannot be squeezed to 1 dimension.
    TypeError
        If x cannot be converted to real float64 values (None cannot be
        converted), or to complex128 values if allow_complex, if x is float16 or
        complex data that are not allowed, name is not hashable or series is not
        a bool.

    Notes
    -----
    x is never modified. NaN and inf are not checked, because the function
    is used by code that handles them; the callers that need finite data
    check it.
    """
    if series is not True and series is not False:
        # Slow path only. ensure1d is called by the volatility models in the
        # likelihood, so the default must not cost anything
        from arch.utility._validation import check_bool  # noqa: PLC0415

        series = check_bool("series", series)
    if name is not None and not isinstance(name, str):
        from arch.utility._validation import check_instance  # noqa: PLC0415

        check_instance("name", name, Hashable)
    if isinstance(x, Series):
        x_ser = _float64_pandas(x, str(name), allow_complex)
        if not isinstance(x_ser.name, str):
            # Rename a new Series so that the Series of the caller is unchanged
            x_ser = x_ser.rename(str(x_ser.name))
        if series:
            return x_ser
        else:
            return x_ser.to_numpy()

    if isinstance(x, DataFrame):
        if x.shape[1] != 1:
            raise ValueError(
                f"{name} must be squeezable to 1 dimension. Received a DataFrame "
                f"with shape {x.shape}, and a DataFrame must have a single column."
            )
        x_frame = _float64_pandas(x, str(name), allow_complex)
        if not series:
            return x_frame.iloc[:, 0].to_numpy()
        x_series = Series(x_frame.iloc[:, 0], x_frame.index)
        if not isinstance(x_series.name, str):
            x_series.name = str(x_series.name)
        return x_series

    x_arr = as_float64(x, str(name), allow_complex=allow_complex)
    if sum([s > 1 for s in x_arr.shape]) > 1:
        raise ValueError(
            f"{name} must be squeezable to 1 dimension. Received an array with "
            f"shape {x_arr.shape}, and at most one dimension can have more than "
            "one element."
        )
    x_arr = x_arr.ravel()
    if series:
        return Series(x_arr, name=name)
    else:
        return x_arr.ravel()


def ensure2d(
    x: Sequence[float | int] | Sequence[Sequence[float | int]] | ArrayLike,
    name: str,
    *,
    allow_complex: bool = False,
) -> DataFrame | NDArray:
    """
    Convert data to a 2-d float64 DataFrame or ndarray.

    Parameters
    ----------
    x : array_like, Series or DataFrame
        The data to convert. A Series is converted to a DataFrame with a
        single column. A scalar is converted to an array with a single row and
        column, and a 1-d array_like to an array with a single column.
        Integer, unsigned integer, Boolean and float32 data are converted to
        float64. float16 data are not accepted.
    name : str
        The name of the variable. Used in error messages.
    allow_complex : bool, optional
        If True, complex data are converted to complex128 and real data to
        float64. If False, complex data are not accepted.

    Returns
    -------
    DataFrame or ndarray
        The 2-d data with dtype float64, or complex128 if allow_complex and x is
        complex. A DataFrame if x is a Series or a DataFrame and an ndarray
        otherwise. The index and the column labels are kept. x is returned if it
        is a DataFrame or an ndarray with float64 data (or complex128 data if
        allow_complex).

    Raises
    ------
    ValueError
        If x has more than 2 dimensions.
    TypeError
        If x cannot be converted to real float64 values (None cannot be
        converted), or to complex128 values if allow_complex, if x is float16 or
        complex data that are not allowed, or name is not a str.

    Notes
    -----
    x is never modified. NaN and inf are not checked, because the function
    is used by code that handles them; the callers that need finite data
    check it.
    """
    if not isinstance(name, str):
        from arch.utility._validation import check_str  # noqa: PLC0415

        check_str("name", name)
    if isinstance(x, Series):
        return DataFrame(_float64_pandas(x, name, allow_complex))
    elif isinstance(x, DataFrame):
        return _float64_pandas(x, name, allow_complex)
    arr = as_float64(x, name, allow_complex=allow_complex)
    if arr.ndim == 0:
        return np.asarray([[arr]])
    elif arr.ndim == 1:
        return arr[:, None]
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(
            f"Variable {name} must be 2d or reshapable to 2d. Received an array "
            f"with {arr.ndim} dimensions and shape {arr.shape}."
        )


def parse_dataframe(
    x: ArrayLike | None, name: str | list[str]
) -> (
    tuple[Index, Index]
    | tuple[list[Hashable | None], Index]
    | tuple[list[str], NDArray]
):
    """
    Extract the column labels and the index of data.

    Parameters
    ----------
    x : array_like, Series, DataFrame or None
        The data. Pandas objects provide their own labels.
    name : str or list of str
        The column labels to use if x is not a Series or a DataFrame. Must be a
        str if x is None.

    Returns
    -------
    columns : Index or list
        The columns of x if x is a DataFrame, a list containing the name of x
        if x is a Series, and otherwise a list of the labels in name.
    index : Index or ndarray
        The index of x if x is a Series or a DataFrame, an empty ndarray if x
        is None, and otherwise the integers 0, 1, ..., nobs - 1 where nobs is
        the number of observations in x.

    Raises
    ------
    TypeError
        If name is not a str or a list, or is not a str when x is None.
    """
    if not isinstance(name, (str, list)) or (x is None and not isinstance(name, str)):
        from arch.utility._validation import check_instance  # noqa: PLC0415

        check_instance("name", name, str if x is None else (str, list))
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


class DocStringInheritor(type):  # numpydoc ignore=PR02
    """
    Metaclass that copies the missing docstrings of a class from its bases.

    A variation on
    https://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
    by Paul McGuire.

    Parameters
    ----------
    name : str
        The name of the class.
    bases : tuple of type
        The base classes of the class.
    clsdict : dict
        The namespace of the class.
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
    """
    Metaclass that requires a class to implement all abstract methods.

    Parameters
    ----------
    *args : tuple, optional
        The name, base classes and namespace of the class.
    **kwargs : dict, optional
        Keyword arguments passed to ``ABCMeta``.

    Raises
    ------
    TypeError
        If the class has abstract methods that are not implemented.
    """

    def __init__(cls, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        missing: list[str] = getattr(cls, "__abstractmethods__", [])
        if missing:
            missing_meth = ", ".join(missing)
            raise TypeError(
                f"{cls.__name__} has not implemented abstract methods {missing_meth}"
            )


class AbstractDocStringInheritor(  # numpydoc ignore=PR02
    ConcreteClassMeta, DocStringInheritor
):
    """
    Metaclass that copies missing docstrings and requires concrete classes.

    Combines `DocStringInheritor` and `ConcreteClassMeta`.

    Parameters
    ----------
    *args : tuple, optional
        The name, base classes and namespace of the class.
    **kwargs : dict, optional
        Keyword arguments passed to ``ABCMeta``.
    """


def date_to_index(
    date: str | dt.date | dt.datetime | np.datetime64 | Timestamp,
    date_index: Union[DatetimeIndex, NDArray, "Series[Timestamp]"],
) -> int:
    """
    Look up a date in an array of dates.

    Parameters
    ----------
    date : str, datetime, datetime64 or Timestamp
        Date to use when returning the index.
    date_index : DatetimeIndex or ndarray
        Index data containing datetime64 values.

    Returns
    -------
    int
        Index location.

    Raises
    ------
    TypeError
        If date is not a str, a datetime, a datetime64 or a Timestamp, or
        date_index does not contain datetime64 values.
    ValueError
        If date_index is empty, is not increasing and unique, or date is a str
        that cannot be parsed to a date.

    Notes
    -----
    Assumes dates are increasing and unique. The result is the location of date
    if it is in date_index, and otherwise the location where it would be
    inserted: 0 for a date before the first date and the length of date_index
    for a date after the last date. `cutoff_to_index` rejects the dates that
    are not within the index.
    """
    # Imported here because the validation module imports this module
    from arch.utility._validation import _describe, _fail  # noqa: PLC0415

    if not is_datetime64_any_dtype(date_index):
        _fail(
            TypeError,
            "date_index",
            "a datetime64 array",
            _describe(date_index),
        )
    if isinstance(date_index, DatetimeIndex):
        if date_index.tzinfo is not None:
            # The same UTC values without a time zone that .values returned. Using
            # .values is deprecated for time zone aware indices.
            date_index = date_index.tz_convert("GMT").tz_localize(None)
        values = date_index.to_numpy()
    else:
        values = np.asarray(date_index)
    if values.shape[0] == 0:
        _fail(
            ValueError,
            "date_index",
            "a non-empty array of dates",
            _describe(date_index),
        )
    if not np.all((np.diff(values).astype(dtype=np.int64)) > 0):
        _fail(
            ValueError,
            "date_index",
            "increasing and unique",
            _describe(date_index),
        )
    if not isinstance(date, (dt.datetime, np.datetime64, str, Timestamp)):
        _fail(
            TypeError,
            "date",
            "a datetime, datetime64, Timestamp or string",
            _describe(date),
        )

    if isinstance(date, Timestamp):
        # A naive Timestamp is used as is, like a date string
        if date.tzinfo is not None:
            date = date.tz_convert("GMT").tz_localize(None)
        date_64 = date.to_datetime64()
    elif isinstance(date, dt.datetime):
        date_64 = np.datetime64(date)
    elif isinstance(date, str):
        pd_dt = to_datetime(date, errors="coerce")
        if pd_dt is NaT:
            _fail(
                ValueError,
                "date",
                "a string that can be parsed to a date",
                _describe(date),
            )
        assert isinstance(pd_dt, Timestamp)
        date_64 = pd_dt.to_datetime64()
    else:
        assert isinstance(date, np.datetime64)
        date_64 = date

    if np.isnat(date_64):
        _fail(ValueError, "date", "a date, not NaT", _describe(date))
    locs = np.nonzero(values <= date_64)[0]
    if locs.shape[0] == 0:
        return 0
    loc = locs.max()
    in_array = np.any(values == date_64)
    if not in_array:
        loc += 1

    return int(loc)


def cutoff_to_index(cutoff: int | DateLike | None, index: Index, default: int) -> int:
    """
    Convert a cutoff to a numerical index.

    Parameters
    ----------
    cutoff : int, str, date, datetime, datetime64, Timestamp, Period or None
        The cutoff point to use. An int is the location of the observation, and
        must be between -len(index) and len(index), inclusive. A negative int
        counts from the end, as in Python slicing, so that -1 is len(index) - 1
        and the result is never negative. Dates, strings and periods can only
        be used when index is a DatetimeIndex or a
        PeriodIndex, and must not be before the first or after the last date
        in index. A Period is converted to the first instant in the period.
    index : Index
        Pandas index. Must be a DatetimeIndex or a PeriodIndex if cutoff is a
        date, a datetime, a datetime64, a Timestamp, a Period or a string.
    default : int
        The value to return if cutoff is None. Must be between 0 and the length
        of index, inclusive.

    Returns
    -------
    int
        The integer location of the cutoff, which is not negative.

    Raises
    ------
    TypeError
        If cutoff is not None, an integer or a date-like value, index is not an
        Index, or default is not an integer.
    ValueError
        If default is an integer that is not between 0 and the length of index,
        cutoff is an integer that is not between -len(index) and len(index), or
        cutoff is a date-like value and the index is not a
        DatetimeIndex or a PeriodIndex, or the date is not within the index.
    """
    # Imported here because the validation module imports this module
    from arch.utility._validation import (  # noqa: PLC0415
        _date_value,
        _date_values,
        _describe,
        _fail,
        check_instance,
        check_int,
    )

    check_instance("index", index, Index)
    nobs = len(index)
    default = check_int("default", default, lower=0, upper=nobs)
    if cutoff is None:
        return default
    if isinstance(cutoff, (int, np.integer)) and not isinstance(
        cutoff, (bool, np.bool_)
    ):
        cutoff = check_int("cutoff", cutoff, lower=-nobs, upper=nobs)
        return cutoff + nobs if cutoff < 0 else cutoff
    if isinstance(cutoff, Period):
        cutoff = cutoff.start_time
    elif isinstance(cutoff, dt.date) and not isinstance(cutoff, dt.datetime):
        cutoff = Timestamp(cutoff)
    if isinstance(cutoff, (str, dt.datetime, np.datetime64)):  # includes Timestamp
        if isinstance(index, PeriodIndex):
            index = index.to_timestamp()
        if not isinstance(index, DatetimeIndex):
            _fail(
                ValueError,
                "cutoff",
                "an integer when the data does not have a DatetimeIndex or a "
                "PeriodIndex",
                _describe(cutoff),
            )
        when = _date_value(cutoff)
        if nobs == 0 or when is None:
            first = last = "-"
        else:
            values = _date_values(index)
            first, last = values[0], values[-1]
            if first <= when <= last:
                return date_to_index(cutoff, index)
        _fail(
            ValueError,
            "cutoff",
            f"a date within the index, from {first} to {last}",
            _describe(cutoff),
        )
    _fail(
        TypeError,
        "cutoffs",
        "None, an integer, a string, a date, a datetime, a datetime64, a Timestamp "
        "or a Period",
        _describe(cutoff),
    )


def find_index(s: AnyPandas, index: int | DateLike) -> int:
    """
    Return the numeric index for a string or datetime.

    Parameters
    ----------
    s : Series or DataFrame
        Series or DataFrame to use in lookup.
    index : int, str, datetime, datetime64 or Timestamp
        Index value, either an integer location between -nobs and nobs, where
        nobs is the number of observations, a string convertible to a datetime
        or a datetime. A negative location counts from the end, as in Python
        indexing, so that -1 is nobs - 1. A non-negative location is returned
        unchanged.

    Returns
    -------
    int
        Integer location of index value, which is not negative.

    Raises
    ------
    TypeError
        If s is not a Series or a DataFrame, or index is not an integer, a str
        or a date-like value.
    ValueError
        If index is an integer that is not between -nobs and nobs, is a str
        that cannot be converted to a date, or is a
        date that is not in the index of s.
    """
    # Imported here because the validation module imports this module
    from arch.utility._validation import (  # noqa: PLC0415
        _describe,
        _fail,
        check_instance,
        check_int,
    )

    check_instance("s", s, (Series, DataFrame))
    if isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)):
        nobs = s.shape[0]
        loc = check_int("index", index, lower=-nobs, upper=nobs)
        return loc + nobs if loc < 0 else loc
    if not isinstance(index, (str, dt.datetime, np.datetime64, Timestamp)):
        _fail(
            TypeError,
            "index",
            "an integer, a string, a datetime, a datetime64 or a Timestamp",
            _describe(index),
        )
    date_index = to_datetime(index, errors="coerce")
    # TODO: Bug in pandas-stubs does not return correct types when errors=coerce
    if date_index is NaT:
        _fail(
            ValueError,
            "index",
            "a value that can be converted to a date",
            _describe(index),
        )
    loc = np.argwhere(s.index == date_index).squeeze()
    if loc.size == 0:
        _fail(ValueError, "index", "a date in the index of s", _describe(index))
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


def _append_sequence(original: NDArray, new: Any) -> NDArray:
    """
    Append to a single sequence of observations in any squeezable layout

    Parameters
    ----------
    original : ndarray
        The existing observations. At most one dimension can have more than one
        element, e.g., (n,), (n, 1) or (1, n).
    new : scalar or array_like
        The new observations. At most one dimension can have more than one
        element.

    Returns
    -------
    ndarray
        The extended observations with the same layout as original, so that
        a row stays a row and a column stays a column.
    """
    original = np.atleast_1d(original)
    new_arr = np.asarray(new)
    if sum(s > 1 for s in original.shape) > 1 or sum(s > 1 for s in new_arr.shape) > 1:
        raise ValueError(
            "The original and appended data must both be 1-dimensional, or "
            "squeezable to 1 dimension, since they contain a single sequence of "
            f"observations. Got shapes {original.shape} and {new_arr.shape}."
        )
    long_axes = [i for i, size in enumerate(original.shape) if size > 1]
    axis = long_axes[0] if long_axes else 0
    new_shape = [1] * original.ndim
    new_shape[axis] = -1
    return np.concatenate((original, new_arr.reshape(new_shape)), axis=axis)


def _nested_container(container: type, values: list[Any]) -> Any:
    """Convert nested lists to list or tuple, using tuples at every level"""
    if container is list:
        return values
    return tuple(
        _nested_container(container, v) if isinstance(v, list) else v for v in values
    )


def append_same_type(
    original: ArrayLike | list[Any] | tuple[Any, ...],
    new: ArrayLike | list[Any] | tuple[Any, ...] | float,
    sequence: bool = False,
) -> Any:
    """
    Append observations to existing data, preserving the type of the data.

    Parameters
    ----------
    original : ndarray, Series, DataFrame, list or tuple
        The existing data.
    new : ndarray, Series, DataFrame, list, tuple or float
        The data to append. This must have the same type as original, unless
        original is an ndarray, list or tuple, in which case a scalar can be
        appended as a single observation. 1-d data appended to a 2-d ndarray
        with more than one column is treated as a single observation.
    sequence : bool, optional
        If True, ndarray, list and tuple data are a single sequence of
        observations that can be stored in any layout that can be squeezed to
        1 dimension, e.g., (n,), (n, 1) or (1, n), and the layout of original
        is preserved. If False, 2-d data are observations (rows) by variables
        (columns), and new must have the same number of columns as original.

    Returns
    -------
    ndarray, Series, DataFrame, list or tuple
        The extended data. Has the same type as original. The inputs are never
        modified.

    Raises
    ------
    TypeError
        If the types of original and new are not compatible.
    ValueError
        If new does not contain any observations, has a different number of
        columns from original, or, when using a DataFrame, has different
        column labels. If sequence is True, also raised if original or new
        cannot be squeezed to 1 dimension.
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
        if sequence:
            extended = _append_sequence(original, new)
        else:
            _original = np.atleast_1d(original)
            if _original.ndim > 2:
                raise ValueError("The original data must be 1-d or 2-d.")
            extended = np.concatenate(
                (_original, _conform_new_to_array(_original, new)), axis=0
            )
    else:
        # list or tuple, converted to an array to validate shapes
        _original = np.asarray(original, dtype=float)
        new_arr = np.asarray(new, dtype=float)
        if sequence:
            stacked = _append_sequence(_original, new_arr)
            if _original.ndim <= 1:
                rows = stacked.ravel()[_original.size :].tolist()
                extended = type(original)(list(original) + rows)
            else:
                # Row or column layouts, e.g., [[1.0, 2.0, 3.0]]
                extended = _nested_container(type(original), stacked.tolist())
        else:
            if _original.ndim > 2:
                raise ValueError("The original data must be 1-d or 2-d.")
            rows = _conform_new_to_array(_original, new_arr).tolist()
            if _original.ndim == 2:
                rows = [type(original)(row) for row in rows]
            extended = type(original)(list(original) + rows)
    return extended
