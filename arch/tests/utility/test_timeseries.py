import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from arch.utility.timeseries import ColumnNameConflict, add_trend


@pytest.fixture
def rng():
    return RandomState(12345)


def test_add_trend_err():
    with pytest.raises(ValueError, match=r"One and only one"):
        add_trend(x=None, trend="ctt", nobs=None)


def test_add_trend_prepend(rng):
    n = 10
    x = rng.randn(n, 1)
    trend_1 = add_trend(x, trend="ct", prepend=True)
    trend_2 = add_trend(x, trend="ct", prepend=False)
    assert_equal(trend_1[:, :2], trend_2[:, 1:])


def test_add_time_trend_dataframe(rng):
    n = 10
    x = rng.randn(n, 1)
    x = pd.DataFrame(x, columns=["col1"])
    trend_1 = add_trend(x, trend="t")
    assert_array_almost_equal(np.asarray(trend_1["trend"]), np.arange(1.0, n + 1))


def test_add_trend_prepend_dataframe(rng):
    n = 10
    x = rng.randn(n, 1)
    x = pd.DataFrame(x, columns=["col1"])
    trend_1 = add_trend(x, trend="ct", prepend=True)
    trend_2 = add_trend(x, trend="ct", prepend=False)
    assert_frame_equal(trend_1.iloc[:, :2], trend_2.iloc[:, 1:])


def test_add_trend_duplicate_name():
    x = pd.DataFrame(np.zeros((10, 1)), columns=["trend"])
    with pytest.warns(ColumnNameConflict, match=r"Some of the column names being"):
        _ = add_trend(x, trend="ct")
    with pytest.warns(ColumnNameConflict, match=r"Some of the column names being"):
        y = add_trend(x, trend="ct")

    assert "const" in y.columns
    assert "trend_0" in y.columns


def test_add_trend_c():
    x = np.zeros((10, 1))
    y = add_trend(x, trend="c")
    assert np.all(y[:, 1] == 1.0)


def test_add_trend_ct():
    n = 20
    x = np.zeros((20, 1))
    y = add_trend(x, trend="ct")
    assert np.all(y[:, 1] == 1.0)
    assert_equal(y[0, 2], 1.0)
    assert_array_almost_equal(np.diff(y[:, 2]), np.ones(n - 1))


def test_add_trend_ctt():
    n = 10
    x = np.zeros((n, 1))
    y = add_trend(x, trend="ctt")
    assert np.all(y[:, 1] == 1.0)
    assert y[0, 2] == 1.0
    assert_array_almost_equal(np.diff(y[:, 2]), np.ones(n - 1))
    assert y[0, 3] == 1.0
    assert_array_almost_equal(np.diff(y[:, 3]), np.arange(3.0, 2.0 * n, 2.0))


def test_add_trend_t():
    n = 20
    x = np.zeros((20, 1))
    y = add_trend(x, trend="t")
    assert y[0, 1] == 1.0
    assert_array_almost_equal(np.diff(y[:, 1]), np.ones(n - 1))


def test_add_trend_no_input():
    n = 100
    y = add_trend(x=None, trend="ct", nobs=n)
    assert np.all(y[:, 0] == 1.0)
    assert y[0, 1] == 1.0
    assert_array_almost_equal(np.diff(y[:, 1]), np.ones(n - 1))


def test_skip_constant():
    x = np.ones((100, 1))
    appended = add_trend(x, trend="c", has_constant="add")
    assert_array_equal(np.ones((100, 2)), appended)
    appended = add_trend(x, trend="c", has_constant="skip")
    assert_array_equal(np.ones((100, 1)), appended)


def test_errors():
    n = 100
    with pytest.raises(ValueError, match=r"trend unknown not understood"):
        add_trend(x=None, trend="unknown", nobs=n)
    with pytest.raises(
        ValueError, match=r"One and only one of x or nobs must be provided"
    ):
        add_trend(x=None, trend="ct")
    x = np.ones((100, 1))
    with pytest.raises(ValueError, match=r"x already contains a constant"):
        add_trend(x, trend="ct", has_constant="raise")


def test_trend_n_nobs():
    assert add_trend(nobs=100, trend="n").shape == (100, 0)
    assert add_trend(np.empty((100, 2)), trend="n").shape == (100, 2)


def test_addtrend_bad_nobs():
    with pytest.raises(ValueError, match=r"nobs must"):
        add_trend(None, trend="ct")
    with pytest.raises(ValueError, match=r"nobs must"):
        add_trend(None, trend="ct", nobs=-3)
