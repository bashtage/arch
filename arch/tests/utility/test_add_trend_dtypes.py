"""add_trend with complex data: the trend columns are real, and the data are kept."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest

from arch.utility.timeseries import add_trend

NOBS = 12
T = np.arange(1.0, NOBS + 1)
# The columns of each trend, computed without add_trend
TRENDS = {
    "c": [np.ones(NOBS)],
    "t": [T],
    "ct": [np.ones(NOBS), T],
    "ctt": [np.ones(NOBS), T, T**2],
}


def _data():
    rng = np.random.default_rng(5)
    return rng.standard_normal((NOBS, 2)) + 1j * rng.standard_normal((NOBS, 2))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("prepend", [True, False], ids=["prepend", "append"])
@pytest.mark.parametrize("trend", list(TRENDS))
def test_complex_array(trend, prepend, dtype):
    z = _data().astype(dtype)
    z64 = z.astype(np.complex128)
    result = add_trend(z, trend, prepend=prepend)
    columns = np.column_stack(TRENDS[trend])
    expected = np.column_stack([columns, z64] if prepend else [z64, columns])
    assert result.dtype == np.complex128
    assert_array_equal(result, expected)
    assert_array_equal(z, _data().astype(dtype))


def test_complex_series_and_frame():
    z = _data()
    series = add_trend(pd.Series(z[:, 0], name="s"), "ct")
    assert series.dtype == np.complex128
    assert_array_equal(series[:, 0], z[:, 0])
    assert_array_equal(series[:, 1:].real, np.column_stack(TRENDS["ct"]))
    assert_array_equal(series[:, 1:].imag, 0)

    index = pd.date_range("2000-01-31", periods=NOBS, freq="D")
    frame = pd.DataFrame(z, columns=["a", "b"], index=index)
    result = add_trend(frame, "ct", prepend=True)
    assert list(result.columns) == ["const", "trend", "a", "b"]
    assert result.index.equals(index)
    assert result["a"].dtype == np.complex128
    assert result["const"].dtype == np.float64
    assert_array_equal(result[["a", "b"]].to_numpy(), z)
    assert_allclose(result["trend"], T)


def test_complex_constant_is_detected():
    z = _data()
    z[:, 0] = 2 + 1j
    with pytest.raises(ValueError, match="x already contains a constant"):
        add_trend(z, "c", has_constant="raise")
    skipped = add_trend(z, "ct", has_constant="skip")
    assert skipped.shape == (NOBS, 3)
    added = add_trend(z, "c", has_constant="add")
    assert added.shape == (NOBS, 3)
    # A column with a zero real part and a constant imaginary part is a constant, and
    # so is a column that is zero in the imaginary part, but a column of zeros is not
    z[:, 0] = 3j
    with pytest.raises(ValueError, match="x already contains a constant"):
        add_trend(z, "c", has_constant="raise")
    z[:, 0] = 0
    assert add_trend(z, "c", has_constant="raise").shape == (NOBS, 3)


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.uint8, np.bool_])
def test_real_data_are_float64(dtype):
    x = (np.arange(NOBS * 2).reshape(NOBS, 2) % 3).astype(dtype)
    result = add_trend(x, "ct")
    assert result.dtype == np.float64


def test_float16_is_rejected():
    x = np.arange(NOBS, dtype=np.float16)
    with pytest.raises(TypeError, match=r"^x must be .*; got float16 data with shape"):
        add_trend(x, "c")
