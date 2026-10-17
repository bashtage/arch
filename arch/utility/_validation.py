"""
Validation of the arguments of the public API.

The helpers take the name of the argument first, so that the error messages name
the argument, then say what is accepted and show the value that was received::

    <name> must be <rule>; got <value> (<type>)

The exception type follows the Python convention: ``TypeError`` for a value of the wrong
type (including data that cannot be converted to numbers) and ``ValueError`` for a value
of the right type that is not allowed (out of range, not one of the options, wrong shape
or length, NaN or inf where finite values are needed).

The rules that decide which values are accepted are in the small private functions named
``_is_*`` and ``_float_in_range``. They are the only code to change if the policy
changes (decisions 66, 67 and 68 in the release plan are provisional):

* flags accept ``bool`` and ``np.bool_`` only (``_is_bool``);
* integers are accepted by ``operator.index``, which rejects floats, including 5.0, and
  Booleans (``_is_integer``);
* floats must be real numbers that are finite, unless ``allow_nan`` or ``allow_inf``
  (``_is_real`` and ``_float_in_range``);
* NumPy scalars are accepted wherever the matching Python type is accepted.

The messages are built only when a check fails, so a successful check is cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import datetime as dt
import math
import numbers
import operator
from typing import Any, Literal, NoReturn, cast, overload

import numpy as np
from numpy.random import Generator, RandomState
import pandas as pd

from arch.utility.array import as_float64, cutoff_to_index

__all__ = [
    "check_array",
    "check_bool",
    "check_callable",
    "check_float",
    "check_index",
    "check_instance",
    "check_int",
    "check_ndarray",
    "check_option",
    "check_seed",
    "check_str",
]

_MAX_REPR = 40

Inclusive = Literal["both", "neither", "left", "right"]


def _short_repr(value: Any) -> str:
    """Return the repr of a value, shortened to keep error messages readable."""
    if isinstance(value, np.generic) and not isinstance(value, (str, bytes)):
        text = str(value)
    else:
        text = repr(value)
    if len(text) > _MAX_REPR:
        text = text[: _MAX_REPR - 3] + "..."
    return text


def _describe(value: Any) -> str:
    """Describe the value received as ``<value> (<type>)``."""
    if isinstance(value, (np.ndarray, pd.Series, pd.DataFrame, pd.Index)):
        return f"shape {tuple(value.shape)} ({type(value).__name__})"
    return f"{_short_repr(value)} ({type(value).__name__})"


def _describe_array(arr: np.ndarray, value: Any) -> str:
    """Describe an array by the shape of the converted data and the type received."""
    return f"shape {arr.shape} ({type(value).__name__})"


def _fail(exc: type[Exception], name: str, rule: str, received: str) -> NoReturn:
    """Raise ``exc`` with the message ``<name> must be <rule>; got <received>``."""
    raise exc(f"{name} must be {rule}; got {received}")


def _fmt(number: float) -> str:
    """Format a bound for the message."""
    if isinstance(number, numbers.Integral):
        return str(int(number))
    return f"{number:g}"


def _format_choices(choices: Sequence[str]) -> str:
    quoted = [f"'{choice}'" for choice in choices]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " or " + quoted[-1]


def _with_none(rule: str, allow_none: bool) -> str:
    return f"{rule} or None" if allow_none else rule


def _is_bool(value: Any) -> bool:
    """Flags: bool and np.bool_ only. Change here to accept 0/1 or any truth value."""
    return isinstance(value, (bool, np.bool_))


def _is_integer(value: Any) -> bool:
    """Integers accepted by operator.index, but not bool, np.bool_ or a bool array."""
    if _is_bool(value):
        return False
    if isinstance(value, np.ndarray) and value.dtype.kind not in "iu":
        return False
    try:
        operator.index(value)
    except TypeError:
        return False
    return True


def _is_real(value: Any) -> bool:
    """Real numbers, including NumPy scalars, but not bool, np.bool_ or Decimal."""
    if _is_bool(value):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 0 and value.dtype.kind in "iuf"
    return isinstance(value, numbers.Real)


@overload
def check_bool(
    name: str, value: Any, *, allow_none: Literal[False] = ...
) -> bool:  # pragma: no cover
    ...  # pragma: no cover


@overload
def check_bool(
    name: str, value: Any, *, allow_none: Literal[True]
) -> bool | None:  # pragma: no cover
    ...  # pragma: no cover


def check_bool(name: str, value: Any, *, allow_none: bool = False) -> bool | None:
    """
    Check a flag.

    Parameters
    ----------
    name : str
        The name of the argument, which is used in the error message.
    value : object
        The value passed by the user. Only ``bool`` and ``np.bool_`` are accepted, so
        0, 1, "no" and arrays are rejected.
    allow_none : bool, optional
        If True, None is accepted and returned unchanged. Use this only where None is
        the documented default.

    Returns
    -------
    bool or None
        The flag as a Python bool, or None if ``allow_none`` and ``value`` is None.

    Raises
    ------
    TypeError
        If the value is not a bool (or None if ``allow_none``).
    """
    if _is_bool(value):
        return bool(value)
    if allow_none and value is None:
        return None
    _fail(TypeError, name, _with_none("True or False", allow_none), _describe(value))


def _int_rule(lower: int | None, upper: int | None, allow_none: bool) -> str:
    if lower is not None and upper is not None:
        rule = f"an integer in [{_fmt(lower)}, {_fmt(upper)}]"
    elif lower is not None:
        rule = f"an integer >= {_fmt(lower)}"
    elif upper is not None:
        rule = f"an integer <= {_fmt(upper)}"
    else:
        rule = "an integer"
    return _with_none(rule, allow_none)


@overload
def check_int(
    name: str,
    value: Any,
    *,
    lower: int | None = ...,
    upper: int | None = ...,
    allow_none: Literal[False] = ...,
) -> int:  # pragma: no cover
    ...  # pragma: no cover


@overload
def check_int(
    name: str,
    value: Any,
    *,
    lower: int | None = ...,
    upper: int | None = ...,
    allow_none: Literal[True],
) -> int | None:  # pragma: no cover
    ...  # pragma: no cover


def check_int(
    name: str,
    value: Any,
    *,
    lower: int | None = None,
    upper: int | None = None,
    allow_none: bool = False,
) -> int | None:
    """
    Check an integer.

    Parameters
    ----------
    name : str
        The name of the argument, which is used in the error message.
    value : object
        The value passed by the user. Python and NumPy integers are accepted, using
        ``operator.index``. Floats, including 5.0, and Booleans are rejected.
    lower : int, optional
        The smallest accepted value (inclusive).
    upper : int, optional
        The largest accepted value (inclusive).
    allow_none : bool, optional
        If True, None is accepted and returned unchanged.

    Returns
    -------
    int or None
        The value as a Python int, or None if ``allow_none`` and ``value`` is None.

    Raises
    ------
    TypeError
        If the value is not an integer (or None if ``allow_none``).
    ValueError
        If the value is smaller than ``lower`` or larger than ``upper``.
    """
    if allow_none and value is None:
        return None
    exc: type[Exception] = TypeError
    if _is_integer(value):
        result = operator.index(value)
        if (lower is None or result >= lower) and (upper is None or result <= upper):
            return result
        exc = ValueError
    _fail(exc, name, _int_rule(lower, upper, allow_none), _describe(value))


def _float_rule(
    lower: float | None,
    upper: float | None,
    inclusive: Inclusive,
    allow_nan: bool,
    allow_inf: bool,
    allow_none: bool,
) -> str:
    rule = "a float" if (allow_nan or allow_inf) else "a finite float"
    left = "[" if inclusive in ("both", "left") else "("
    right = "]" if inclusive in ("both", "right") else ")"
    if lower is not None and upper is not None:
        rule += f" in {left}{_fmt(lower)}, {_fmt(upper)}{right}"
    elif lower is not None:
        rule += f" {'>=' if left == '[' else '>'} {_fmt(lower)}"
    elif upper is not None:
        rule += f" {'<=' if right == ']' else '<'} {_fmt(upper)}"
    if allow_nan and allow_inf:
        rule += " (NaN and inf allowed)"
    elif allow_nan:
        rule += " (NaN allowed)"
    elif allow_inf:
        rule += " (inf allowed)"
    return _with_none(rule, allow_none)


def _float_in_range(
    value: float,
    lower: float | None,
    upper: float | None,
    inclusive: Inclusive,
    *,
    allow_nan: bool,
    allow_inf: bool,
) -> bool:
    """Check NaN, inf and the bounds of a float."""
    if math.isnan(value):
        return allow_nan
    if math.isinf(value) and not allow_inf:
        return False
    if lower is not None and (
        value < lower or (value == lower and inclusive in ("neither", "right"))
    ):
        return False
    return not (
        upper is not None
        and (value > upper or (value == upper and inclusive in ("neither", "left")))
    )


@overload
def check_float(
    name: str,
    value: Any,
    *,
    lower: float | None = ...,
    upper: float | None = ...,
    inclusive: Inclusive = ...,
    allow_nan: bool = ...,
    allow_inf: bool = ...,
    allow_none: Literal[False] = ...,
) -> float:  # pragma: no cover
    ...  # pragma: no cover


@overload
def check_float(
    name: str,
    value: Any,
    *,
    lower: float | None = ...,
    upper: float | None = ...,
    inclusive: Inclusive = ...,
    allow_nan: bool = ...,
    allow_inf: bool = ...,
    allow_none: Literal[True],
) -> float | None:  # pragma: no cover
    ...  # pragma: no cover


def check_float(
    name: str,
    value: Any,
    *,
    lower: float | None = None,
    upper: float | None = None,
    inclusive: Inclusive = "both",
    allow_nan: bool = False,
    allow_inf: bool = False,
    allow_none: bool = False,
) -> float | None:
    """
    Check a real number.

    Parameters
    ----------
    name : str
        The name of the argument, which is used in the error message.
    value : object
        The value passed by the user. Python and NumPy integers and floats are
        accepted. Booleans, strings and complex numbers are rejected.
    lower : float, optional
        The lower bound.
    upper : float, optional
        The upper bound.
    inclusive : {"both", "neither", "left", "right"}, optional
        Whether ``lower`` ("left") and ``upper`` ("right") are accepted values.
    allow_nan : bool, optional
        If True, NaN is accepted, and the bounds are not applied to it.
    allow_inf : bool, optional
        If True, inf and -inf are accepted if they satisfy the bounds.
    allow_none : bool, optional
        If True, None is accepted and returned unchanged.

    Returns
    -------
    float or None
        The value as a Python float, or None if ``allow_none`` and ``value`` is None.

    Raises
    ------
    TypeError
        If the value is not a real number (or None if ``allow_none``).
    ValueError
        If the value is NaN or infinite where that is not allowed, or is outside the
        bounds.
    """
    if allow_none and value is None:
        return None
    exc: type[Exception] = TypeError
    if _is_real(value):
        exc = ValueError
        try:
            result = float(value)
        except OverflowError:
            pass  # A Python int that is too large for a float
        else:
            if _float_in_range(
                result,
                lower,
                upper,
                inclusive,
                allow_nan=allow_nan,
                allow_inf=allow_inf,
            ):
                return result
    rule = _float_rule(lower, upper, inclusive, allow_nan, allow_inf, allow_none)
    _fail(exc, name, rule, _describe(value))


@overload
def check_seed(
    name: str, value: Any, *, allow_none: Literal[True] = ...
) -> int | Generator | RandomState | None:  # pragma: no cover
    ...  # pragma: no cover


@overload
def check_seed(
    name: str, value: Any, *, allow_none: Literal[False]
) -> int | Generator | RandomState:  # pragma: no cover
    ...  # pragma: no cover


def check_seed(
    name: str, value: Any, *, allow_none: bool = True
) -> int | Generator | RandomState | None:
    """
    Check a seed or a random number generator.

    Parameters
    ----------
    name : str
        The name of the argument, which is used in the error message.
    value : object
        The value passed by the user. None, a non-negative integer, a NumPy
        ``Generator`` or a NumPy ``RandomState`` are accepted. Booleans, floats and
        negative integers are rejected.
    allow_none : bool, optional
        If True (the default), None is accepted and returned unchanged, which means
        that the generator is seeded from system entropy.

    Returns
    -------
    int, Generator, RandomState or None
        An integer as a Python int, and anything else unchanged.

    Raises
    ------
    TypeError
        If the value is not an integer, a Generator or a RandomState (or None if
        ``allow_none``).
    ValueError
        If the value is a negative integer.
    """
    if value is None and allow_none:
        return None
    if isinstance(value, (Generator, RandomState)):
        return value
    exc: type[Exception] = TypeError
    if _is_integer(value):
        result = operator.index(value)
        if result >= 0:
            return result
        exc = ValueError
    rule = "a non-negative integer, a NumPy Generator or a NumPy RandomState"
    _fail(exc, name, f"None, {rule}" if allow_none else rule, _describe(value))


@overload
def check_str(
    name: str, value: Any, *, allow_none: Literal[False] = ...
) -> str:  # pragma: no cover
    ...  # pragma: no cover


@overload
def check_str(
    name: str, value: Any, *, allow_none: Literal[True]
) -> str | None:  # pragma: no cover
    ...  # pragma: no cover


def check_str(name: str, value: Any, *, allow_none: bool = False) -> str | None:
    """
    Check a string that is not one of a fixed set of options.

    Use `check_option` for an option.

    Parameters
    ----------
    name : str
        The name of the argument, which is used in the error message.
    value : object
        The value passed by the user. Only ``str`` (including ``np.str_``) is accepted.
    allow_none : bool, optional
        If True, None is accepted and returned unchanged.

    Returns
    -------
    str or None
        The string, or None if ``allow_none`` and ``value`` is None.

    Raises
    ------
    TypeError
        If the value is not a string (or None if ``allow_none``).
    """
    if isinstance(value, str):
        return str(value)
    if allow_none and value is None:
        return None
    _fail(TypeError, name, _with_none("a string", allow_none), _describe(value))


def check_callable(
    name: str, value: Any, *, allow_none: bool = False
) -> Callable[..., Any] | None:
    """
    Check that a value can be called.

    Parameters
    ----------
    name : str
        The name of the argument, which is used in the error message.
    value : object
        The value passed by the user.
    allow_none : bool, optional
        If True, None is accepted and returned unchanged.

    Returns
    -------
    callable or None
        The value, or None if ``allow_none`` and ``value`` is None.

    Raises
    ------
    TypeError
        If the value is not callable (or None if ``allow_none``).
    """
    if callable(value):
        return value
    if allow_none and value is None:
        return None
    _fail(TypeError, name, _with_none("callable", allow_none), _describe(value))


def check_instance(
    name: str,
    value: Any,
    types: type | tuple[type, ...],
    *,
    allow_none: bool = False,
) -> Any:
    """
    Check that a value is an instance of a class.

    Parameters
    ----------
    name : str
        The name of the argument, which is used in the error message.
    value : object
        The value passed by the user.
    types : type or tuple of type
        The accepted classes. Their names are used in the error message.
    allow_none : bool, optional
        If True, None is accepted and returned unchanged.

    Returns
    -------
    object
        The value, unchanged.

    Raises
    ------
    TypeError
        If the value is not an instance of ``types`` (or None if ``allow_none``).
    """
    if isinstance(value, types):
        return value
    if allow_none and value is None:
        return None
    classes = types if isinstance(types, tuple) else (types,)
    names = [cls.__name__ for cls in classes]
    listing = ", ".join(names[:-1]) + " or " + names[-1] if len(names) > 1 else names[0]
    rule = _with_none(f"an instance of {listing}", allow_none)
    _fail(TypeError, name, rule, _describe(value))


def _shape_text(shape: Sequence[int | None]) -> str:
    entries = ["n" if size is None else str(size) for size in shape]
    return f"({', '.join(entries)}{',' if len(entries) == 1 else ''})"


def _array_rule(
    ndim: int | None,
    shape: Sequence[int | None] | None,
    min_length: int | None,
    allow_nan: bool,
    allow_inf: bool,
    allow_none: bool,
) -> str:
    rule = "an array" if ndim is None else f"a {ndim}-d array"
    if allow_nan and not allow_inf:
        rule += " of numbers without inf"
    elif allow_inf and not allow_nan:
        rule += " of numbers without NaN"
    elif not allow_nan and not allow_inf:
        rule += " of finite numbers"
    if shape is not None:
        rule += f" with shape {_shape_text(shape)}"
    if min_length is not None:
        unit = "element" if ndim == 1 else "row"
        rule += f" with at least {min_length} {unit}{'' if min_length == 1 else 's'}"
    return _with_none(rule, allow_none)


def _nonfinite_description(arr: np.ndarray, allow_nan: bool, allow_inf: bool) -> str:
    """Count the values that are not allowed, for example ``2 NaN and 1 inf``."""
    parts = []
    if not allow_nan and (count := int(np.isnan(arr).sum())):
        parts.append(f"{count} NaN")
    if not allow_inf and (count := int(np.isinf(arr).sum())):
        parts.append(f"{count} inf")
    return " and ".join(parts)


def _to_float64(name: str, value: Any, rule: str) -> np.ndarray:
    """
    Convert to a float64 ndarray, raising a TypeError in the house form.

    ``as_float64`` converts None, as an element of an object array or as the value,
    to NaN. Missing values are NaN in a float array, and None is not a number.
    """
    data = value.to_numpy() if isinstance(value, (pd.Series, pd.DataFrame)) else value
    if data is not None:
        try:
            arr = np.asarray(data)
            if arr.dtype != object or not any(item is None for item in arr.flat):
                return as_float64(arr, name)
        except TypeError:
            pass
    raise TypeError(f"{name} must be {rule}; got {_describe(value)}")


def _reshape(arr: np.ndarray, ndim: int, squeeze: bool) -> np.ndarray | None:
    """Reshape as ensure1d and ensure2d do, or return None if it is not possible."""
    if not squeeze:
        return None
    if ndim == 1 and sum(size > 1 for size in arr.shape) <= 1:
        return arr.ravel()
    if ndim == 2 and arr.ndim < 2:
        return arr.reshape(1, 1) if arr.ndim == 0 else arr[:, None]
    return None


def check_array(
    name: str,
    value: Any,
    *,
    ndim: int | None = None,
    shape: Sequence[int | None] | None = None,
    min_length: int | None = None,
    squeeze: bool = True,
    allow_nan: bool = False,
    allow_inf: bool = False,
    allow_none: bool = False,
) -> Any:
    """
    Convert array_like data to float64 and check its dimensions and values.

    This uses the conversion of `arch.utility.array.as_float64`, which ``ensure1d``
    and ``ensure2d`` also use. The result is an ndarray. Call ``ensure1d`` or
    ``ensure2d`` first if the index of a pandas object must be kept, and use this
    function for the checks.

    Parameters
    ----------
    name : str
        The name of the argument, which is used in the error message.
    value : array_like, Series or DataFrame
        The data. Integer, unsigned integer and Boolean data are converted to float64.
    ndim : int, optional
        The number of dimensions of the result, 1 or 2. If ``squeeze`` is True, 1-d
        data can have any shape with a single dimension longer than 1 (a column, a row
        or a scalar) and 2-d data can be a scalar or 1-d (which becomes a column), as in
        ``ensure1d`` and ``ensure2d``.
    shape : sequence of int or None, optional
        The required shape of the result. None in a position accepts any size. The
        length of the sequence is the number of dimensions, and it replaces ``ndim``.
    min_length : int, optional
        The smallest accepted size of the first dimension.
    squeeze : bool, optional
        If False, ``ndim`` (or ``len(shape)``) must be the number of dimensions of the
        data exactly, and nothing is reshaped.
    allow_nan : bool, optional
        If True, NaN is accepted.
    allow_inf : bool, optional
        If True, inf and -inf are accepted.
    allow_none : bool, optional
        If True, None is accepted and returned unchanged.

    Returns
    -------
    ndarray or None
        The float64 data. This is ``value`` itself if it is a float64 ndarray with the
        required number of dimensions. None if ``allow_none`` and ``value`` is None.

    Raises
    ------
    TypeError
        If the data cannot be converted to float64 (strings, dates, complex numbers,
        objects that are not numbers, None).
    ValueError
        If the data do not have the required number of dimensions, shape or length, or
        contain NaN or inf where that is not allowed.
    """
    if allow_none and value is None:
        return None
    if shape is not None:
        ndim = len(shape)
    rule = _array_rule(ndim, shape, min_length, allow_nan, allow_inf, allow_none)
    arr = _to_float64(name, value, rule)
    if ndim is not None and arr.ndim != ndim:
        reshaped = _reshape(arr, ndim, squeeze)
        if reshaped is None:
            _fail(ValueError, name, rule, _describe_array(arr, value))
        arr = reshaped
    if shape is not None and any(
        size is not None and size != actual
        for size, actual in zip(shape, arr.shape, strict=False)
    ):
        _fail(ValueError, name, rule, _describe_array(arr, value))
    if min_length is not None and (arr.ndim == 0 or arr.shape[0] < min_length):
        _fail(ValueError, name, rule, _describe_array(arr, value))
    if arr.size and not (allow_nan and allow_inf):
        if allow_nan or allow_inf or not np.isfinite(arr).all():
            detail = _nonfinite_description(arr, allow_nan, allow_inf)
            if detail:
                _fail(ValueError, name, rule, f"{detail} ({type(value).__name__})")
    return arr


def check_ndarray(
    name: str,
    value: Any,
    *,
    ndim: int,
    shape: Sequence[int | None] | None = None,
    writeable: bool = False,
    allow_none: bool = False,
) -> Any:
    """
    Check an ndarray without converting it, before it is passed to compiled code.

    The arrays that the Cython recursions read and write must already be C contiguous
    float64 ndarrays. This function never copies, since an array that the callee fills
    in place must be the caller's array.

    Parameters
    ----------
    name : str
        The name of the argument, which is used in the error message.
    value : object
        The value passed by the user. It must be a float64 ndarray.
    ndim : int
        The required number of dimensions.
    shape : sequence of int or None, optional
        The required shape. None in a position accepts any size.
    writeable : bool, optional
        If True, the array must be writeable (an output array).
    allow_none : bool, optional
        If True, None is accepted and returned unchanged.

    Returns
    -------
    ndarray or None
        ``value`` unchanged, or None if ``allow_none`` and ``value`` is None.

    Raises
    ------
    TypeError
        If the value is not an ndarray, or its dtype is not float64.
    ValueError
        If the array is not C contiguous (a view with a stride, a transpose), is
        read-only but must be writeable, or has the wrong number of dimensions or
        shape.
    """
    if allow_none and value is None:
        return None
    if not isinstance(value, np.ndarray) or value.dtype != np.float64:
        exc: type[Exception] = TypeError
        received = (
            _describe(value)
            if not isinstance(value, np.ndarray)
            else f"{value.dtype} data ({type(value).__name__})"
        )
    elif value.ndim != ndim or (
        shape is not None
        and any(
            size is not None and size != actual
            for size, actual in zip(shape, value.shape, strict=False)
        )
    ):
        exc, received = ValueError, _describe(value)
    elif not value.flags.c_contiguous:
        exc, received = ValueError, "an array that is not C contiguous (ndarray)"
    elif writeable and not value.flags.writeable:
        exc, received = ValueError, "a read-only array (ndarray)"
    else:
        return value
    rule = f"a {ndim}-d C contiguous float64 ndarray"
    if shape is not None:
        rule += f" with shape {_shape_text(shape)}"
    _fail(exc, name, _with_none(rule, allow_none), received)


def _date_values(index: pd.Index) -> np.ndarray:
    """The values of a DatetimeIndex or PeriodIndex as naive datetime64, in UTC."""
    dates = index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
    assert isinstance(dates, pd.DatetimeIndex)
    if dates.tz is not None:
        dates = dates.tz_convert("GMT").tz_localize(None)
    return dates.to_numpy()


def _date_value(value: Any) -> np.datetime64 | None:
    """Convert a date-like value to naive datetime64 in UTC, or None if invalid."""
    if isinstance(value, np.datetime64) and np.isnat(value):
        return None
    if isinstance(value, pd.Period):
        value = value.start_time
    elif isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        value = pd.Timestamp(value)
    if isinstance(value, str):
        value = pd.to_datetime(value, errors="coerce")
        if value is pd.NaT:
            return None
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert("GMT").tz_localize(None)
        return value.to_datetime64()
    when = np.datetime64(value)
    return None if np.isnat(when) else when


_DATE_TYPES = (str, dt.date, np.datetime64, pd.Timestamp, pd.Period)


def check_index(
    name: str,
    value: Any,
    nobs: int,
    *,
    index: pd.Index | None = None,
    default: int | None = None,
    upper: int | None = None,
    allow_none: bool = True,
) -> int | None:
    """
    Check a position or a date that selects a point in a sample.

    This is the check for ``first_obs``, ``last_obs`` and ``start``. A position must be
    in ``[0, nobs]``, and a date must be within the index.

    Parameters
    ----------
    name : str
        The name of the argument, which is used in the error message.
    value : int, str, date, datetime, datetime64, Timestamp, Period or None
        The position of an observation, or, if ``index`` is a DatetimeIndex or a
        PeriodIndex, a date, a datetime, a datetime64, a Timestamp, a Period or a string
        that can be converted to a date. Negative integers (positions from the end) are
        not accepted.
    nobs : int
        The number of observations.
    index : Index, optional
        The index of the data. Needed to accept dates.
    default : int, optional
        The result when ``value`` is None.
    upper : int, optional
        The largest accepted position. The default is ``nobs``.
    allow_none : bool, optional
        If True (the default), None is accepted and ``default`` is returned. If False,
        None raises a TypeError.

    Returns
    -------
    int or None
        The position, or ``default`` if ``value`` is None, which is None if there is no
        default.

    Raises
    ------
    TypeError
        If the value is not an integer, a date or None (if ``allow_none``).
    ValueError
        If the position is not in ``[0, nobs]`` (or ``upper``), the date is outside the
        index or cannot be parsed, or a date is given without a date index.
    """
    if value is None and allow_none:
        return default
    stop = nobs if upper is None else upper
    if _is_integer(value):
        position = operator.index(value)
        if 0 <= position <= stop:
            return position
        exc: type[Exception] = ValueError
    elif isinstance(value, _DATE_TYPES):
        exc = ValueError
        if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
            try:
                when = _date_value(value)
                values = _date_values(index)
                if when is not None and values[0] <= when <= values[-1]:
                    return cutoff_to_index(cast("Any", value), index, 0)
            except ValueError:
                pass  # date_to_index requires a monotonic, unique index
    else:
        exc = TypeError
    rule = f"an integer position in [0, {stop}]"
    if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
        rule += (
            ", a date within the index" if allow_none else " or a date within the index"
        )
    _fail(exc, name, _with_none(rule, allow_none), _describe(value))


def check_option(
    name: str,
    value: Any,
    allowed: Iterable[str] | Mapping[str, str],
    *,
    display: Sequence[str] | None = None,
    normalize: Callable[[str], str] | None = None,
) -> str:
    """
    Compare an option to its allowed values without regard to case.

    Parameters
    ----------
    name : str
        The name of the argument, which is used in the error message.
    value : object
        The value passed by the user. It must be a string.
    allowed : iterable of str or mapping of str to str
        The lower case allowed values. A mapping maps each lower case spelling,
        including any alias, to the canonical value that is returned. The
        canonical value may use any case, for example "R" or "Za".
    display : sequence[str], optional
        The values listed in the error message. The default is the lower case
        spellings in ``allowed``.
    normalize : callable, optional
        A function applied to the lower case value before it is looked up, for
        example to ignore spaces and hyphens in the names of kernels.

    Returns
    -------
    str
        The canonical value. This is the lower case value when ``allowed`` is
        not a mapping.

    Raises
    ------
    TypeError
        If the value is not a string.
    ValueError
        If the value is a string that is not one of the allowed values.
    """
    table = dict(allowed) if isinstance(allowed, Mapping) else {a: a for a in allowed}
    if isinstance(value, str):
        key = value.lower()
        if normalize is not None:
            key = normalize(key)
        if key in table:
            return table[key]
    choices = list(table) if display is None else list(display)
    exc = ValueError if isinstance(value, str) else TypeError
    _fail(exc, name, f"one of {_format_choices(choices)}", _describe(value))
