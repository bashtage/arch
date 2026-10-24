"""
The data type rule of the helpers that convert data: ``as_float64``, ``ensure1d``,
``ensure2d``, ``check_array`` and ``to_array_1d``.

* bool, integer, unsigned integer, float32, float64 and longdouble data are converted
  to float64;
* float16 data are always rejected;
* complex data are rejected, unless ``allow_complex`` is True, and then they are
  converted to complex128 (``to_array_1d`` has no such option);
* object data are converted if every element is a number.

Every row is one data type, one container and one value of ``allow_complex``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest

from arch.utility._validation import check_array
from arch.utility.array import as_float64, ensure1d, ensure2d, to_array_1d

VALUES = np.array([0, 1, 2, 3, 4, 5, 6, 7])
REAL_DTYPES = [
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
    np.longdouble,
]
COMPLEX_DTYPES = [np.complex64, np.complex128, np.clongdouble]


def _data(dtype: type) -> np.ndarray:
    values = VALUES % 2 if dtype is np.bool_ else VALUES
    return values.astype(dtype)


def _label(dtype: type) -> str:
    return dtype.__name__


def _as_float64(x: Any, allow_complex: bool) -> Any:
    return as_float64(x, "my_data", allow_complex=allow_complex)


def _ensure1d(x: Any, allow_complex: bool) -> Any:
    return ensure1d(x, "my_data", allow_complex=allow_complex)


def _ensure2d(x: Any, allow_complex: bool) -> Any:
    return np.asarray(ensure2d(x, "my_data", allow_complex=allow_complex))


def _check_array(x: Any, allow_complex: bool) -> Any:
    return check_array("my_data", x, ndim=1, allow_complex=allow_complex)


HELPERS: list[tuple[str, Callable[[Any, bool], Any]]] = [
    ("as_float64", _as_float64),
    ("ensure1d", _ensure1d),
    ("ensure2d", _ensure2d),
    ("check_array", _check_array),
]
HELPER_IDS = [name for name, _ in HELPERS]
CONTAINERS: list[tuple[str, Callable[[np.ndarray], Any]]] = [
    ("ndarray", lambda x: x),
    ("series", lambda x: pd.Series(x, index=pd.RangeIndex(10, 10 + x.shape[0]))),
    (
        "frame",
        lambda x: pd.DataFrame({"a": x}, index=pd.RangeIndex(10, 10 + x.shape[0])),
    ),
]
CONTAINER_IDS = [name for name, _ in CONTAINERS]


def _values(result: Any) -> np.ndarray:
    return np.asarray(result).ravel()


@pytest.mark.parametrize("allow_complex", [False, True], ids=["real", "complex-ok"])
@pytest.mark.parametrize("container", [c for _, c in CONTAINERS], ids=CONTAINER_IDS)
@pytest.mark.parametrize("helper", [h for _, h in HELPERS], ids=HELPER_IDS)
@pytest.mark.parametrize("dtype", REAL_DTYPES, ids=_label)
def test_real_data_are_float64(dtype, helper, container, allow_complex):
    data = _data(dtype)
    original = data.copy()
    result = helper(container(data), allow_complex)
    assert np.asarray(result).dtype == np.float64
    assert np.asarray(result).dtype.char == "d"
    assert_array_equal(_values(result), original.astype(np.float64))
    assert_array_equal(data, original)


@pytest.mark.parametrize("allow_complex", [False, True], ids=["real", "complex-ok"])
@pytest.mark.parametrize("container", [c for _, c in CONTAINERS], ids=CONTAINER_IDS)
@pytest.mark.parametrize("helper", [h for _, h in HELPERS], ids=HELPER_IDS)
def test_float16_is_always_rejected(helper, container, allow_complex):
    data = _data(np.float16)
    match = (
        r"^my_data must be .*float32 or float64.*; got float16 data with shape "
        if not allow_complex
        else r"^my_data must be .*complex128.*; got float16 data with shape "
    )
    with pytest.raises(TypeError, match=match):
        helper(container(data), allow_complex)


@pytest.mark.parametrize("container", [c for _, c in CONTAINERS], ids=CONTAINER_IDS)
@pytest.mark.parametrize("helper", [h for _, h in HELPERS], ids=HELPER_IDS)
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES, ids=_label)
def test_complex_is_rejected_unless_allowed(dtype, helper, container):
    data = _data(dtype) + 1j
    with pytest.raises(TypeError, match=r"^my_data must be .*; got complex\w* data"):
        helper(container(data), False)


@pytest.mark.parametrize("container", [c for _, c in CONTAINERS], ids=CONTAINER_IDS)
@pytest.mark.parametrize("helper", [h for _, h in HELPERS], ids=HELPER_IDS)
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES, ids=_label)
def test_complex_is_complex128_if_allowed(dtype, helper, container):
    data = (_data(dtype) + 2j * _data(dtype)[::-1]).astype(dtype)
    result = helper(container(data), True)
    assert np.asarray(result).dtype == np.complex128
    assert np.asarray(result).dtype.char == "D"
    assert_array_equal(_values(result), data.astype(np.complex128))


@pytest.mark.parametrize(
    "helper", [_as_float64, _ensure2d], ids=["as_float64", "ensure2d"]
)
def test_real_and_complex_columns_of_a_frame(helper):
    frame = pd.DataFrame({"a": np.arange(4.0), "b": np.arange(4) + 1j})
    result = np.asarray(helper(frame, True))
    assert result.dtype == np.complex128
    assert_array_equal(result[:, 0], np.arange(4.0))
    assert_array_equal(result[:, 1], np.arange(4) + 1j)
    with pytest.raises(TypeError, match=r"^my_data must be .*; got complex128 data"):
        helper(frame, False)


def test_complex_128_and_float64_are_not_copied():
    real = np.arange(4.0)
    assert as_float64(real, "x") is real
    assert as_float64(real, "x", allow_complex=True) is real
    z = real + 1j
    assert as_float64(z, "x", allow_complex=True) is z
    assert ensure2d(
        z.reshape(2, 2), "x", allow_complex=True
    ).base is z or np.shares_memory(
        ensure2d(z.reshape(2, 2), "x", allow_complex=True), z
    )


def test_a_float32_array_is_converted_to_a_new_array():
    x = np.arange(4, dtype=np.float32)
    out = as_float64(x, "x")
    assert out is not x
    assert not np.shares_memory(out, x)
    assert out.dtype == np.float64


OBJECTS: list[tuple[str, Any, bool, type | None]] = [
    ("floats", np.array([1, 2.5, np.float32(3)], dtype=object), False, np.float64),
    ("floats-complex-ok", np.array([1, 2.5], dtype=object), True, np.float64),
    ("complex", np.array([1, 2.5j], dtype=object), True, np.complex128),
    ("complex-rejected", np.array([1, 2.5j], dtype=object), False, None),
    (
        "numpy-complex",
        np.array([np.complex128(1 + 2j)], dtype=object),
        True,
        np.complex128,
    ),
    (
        "numpy-complex-rejected",
        np.array([np.complex128(1 + 2j)], dtype=object),
        False,
        None,
    ),
    ("strings", np.array(["a", "b"], dtype=object), True, None),
    ("none", np.array([1.0, None], dtype=object), True, None),
]


@pytest.mark.parametrize(
    ("data", "allow_complex", "dtype"),
    [(d, a, t) for _, d, a, t in OBJECTS],
    ids=[name for name, *_ in OBJECTS],
)
def test_object_data(data, allow_complex, dtype):
    if dtype is None:
        with pytest.raises(TypeError, match=r"^my_data must "):
            as_float64(data, "my_data", allow_complex=allow_complex)
    else:
        result = as_float64(data, "my_data", allow_complex=allow_complex)
        assert result.dtype == dtype
        assert_array_equal(result, data.astype(dtype))


@pytest.mark.parametrize("allow_complex", [False, True])
@pytest.mark.parametrize("container", [c for _, c in CONTAINERS], ids=CONTAINER_IDS)
@pytest.mark.parametrize("helper", [h for _, h in HELPERS], ids=HELPER_IDS)
def test_numpy_complex_objects(helper, container, allow_complex):
    data = np.array([np.complex128(1 + 2j), 3.0], dtype=object)
    original = data.copy()
    if allow_complex:
        result = helper(container(data), True)
        assert np.asarray(result).dtype == np.complex128
        assert_array_equal(_values(result), data.astype(np.complex128))
    else:
        with pytest.raises(TypeError, match=r"^my_data must "):
            helper(container(data), False)
    assert_array_equal(data, original)


@pytest.mark.parametrize(
    "dtype", REAL_DTYPES + [np.float16] + COMPLEX_DTYPES, ids=_label
)
@pytest.mark.parametrize("shape", [(8,), (8, 1), (1, 8), ()], ids=str)
def test_to_array_1d(dtype, shape):
    data = _data(dtype)
    data = data[0] if shape == () else data.reshape(shape)
    if np.dtype(dtype).kind in "fc" and np.dtype(dtype).itemsize in (2,):
        with pytest.raises(TypeError, match=r"^x must be .*float16 data"):
            to_array_1d(np.asarray(data))
    elif np.dtype(dtype).kind == "c":
        with pytest.raises(TypeError, match=r"^x must be .*complex\w* data"):
            to_array_1d(np.asarray(data))
    else:
        result = to_array_1d(np.asarray(data))
        assert result.dtype == np.float64
        assert result.ndim == 1
        assert_array_equal(result, np.atleast_1d(np.asarray(data, dtype=float)).ravel())


def test_to_array_1d_series():
    series = pd.Series(_data(np.float16))
    with pytest.raises(TypeError, match=r"^x must be .*float16 data"):
        to_array_1d(series)
    assert to_array_1d(pd.Series(_data(np.int64))).dtype == np.float64
