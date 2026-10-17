"""String options of the utility functions ignore case"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.utility.timeseries import add_trend

BAD_VALUES = [
    pytest.param("bogus", ValueError, id="bogus"),
    pytest.param("", ValueError, id="empty"),
    pytest.param(None, TypeError, id="none"),
    pytest.param(1, TypeError, id="int"),
]


@pytest.mark.parametrize("trend", ["n", "c", "t", "ct", "ctt"])
def test_add_trend_case(trend):
    x = np.arange(12.0).reshape(6, 2)
    expected = add_trend(x, trend=trend)
    assert_allclose(add_trend(x, trend=trend.upper()), expected)
    assert_allclose(
        add_trend(trend=trend.title(), nobs=6), add_trend(trend=trend, nobs=6)
    )


def test_add_trend_values():
    # Hand computation: columns are the constant, t and t^2
    out = add_trend(trend="CTT", nobs=4)
    t = np.arange(1.0, 5.0)
    assert_allclose(out, np.column_stack([np.ones(4), t, t**2]))


@pytest.mark.parametrize("has_constant", ["raise", "add", "skip"])
def test_add_trend_has_constant_case(has_constant):
    x = np.column_stack([np.ones(4), np.arange(4.0) + 1])
    expected = None
    try:
        expected = add_trend(x, trend="c", has_constant=has_constant)
    except ValueError:
        assert has_constant == "raise"
    for spelling in (has_constant.upper(), has_constant.capitalize()):
        if expected is None:
            with pytest.raises(ValueError, match="x already contains a constant"):
                add_trend(x, trend="c", has_constant=spelling)
        else:
            assert_allclose(add_trend(x, trend="c", has_constant=spelling), expected)


def test_add_trend_skip_is_not_add():
    x = np.column_stack([np.ones(4), np.arange(4.0) + 1])
    assert add_trend(x, trend="CT", has_constant="SKIP").shape == (4, 3)
    assert add_trend(x, trend="CT", has_constant="Add").shape == (4, 4)


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_add_trend_invalid(bad, exc):
    x = np.arange(12.0).reshape(6, 2)
    with pytest.raises(exc, match=r"trend must be one of 'n', 'c', 'ct', 'ctt' or 't'"):
        add_trend(x, trend=bad)
    # Any other value of has_constant used to be treated as "add"
    with pytest.raises(
        exc, match=r"has_constant must be one of 'raise', 'add' or 'skip'"
    ):
        add_trend(x, has_constant=bad)
