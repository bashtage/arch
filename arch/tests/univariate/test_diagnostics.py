import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats

from arch.univariate import excess_kurtosis, hill_estimator, var_ratio
from arch.univariate.distribution import Normal, StudentsT
from arch.univariate.mean import ZeroMean
from arch.univariate.volatility import ConstantVariance


class PositiveTailNormal(Normal):
    def ppf(self, pits, parameters=None):
        return 1.0


class NonfiniteTailNormal(Normal):
    def ppf(self, pits, parameters=None):
        return np.nan


def test_excess_kurtosis():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, np.nan])
    assert_allclose(excess_kurtosis(x), stats.kurtosis(x[:-1], fisher=True))


def test_excess_kurtosis_constant():
    assert_equal(np.isnan(excess_kurtosis(np.ones(10))), True)


def test_excess_kurtosis_no_finite():
    with pytest.raises(ValueError, match="at least one finite"):
        excess_kurtosis(np.array([np.nan, np.inf]))


def test_hill_estimator():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    expected = 1.0 / np.mean(np.log(np.array([8.0, 16.0])) - np.log(4.0))
    assert_allclose(hill_estimator(x, 2), expected)


def test_hill_estimator_default_k():
    x = np.arange(1.0, 10.0)
    expected = 1.0 / np.mean(np.log(np.array([7.0, 8.0, 9.0])) - np.log(6.0))
    assert_allclose(hill_estimator(x), expected)


def test_hill_estimator_invalid_k():
    x = np.arange(1.0, 10.0)
    with pytest.raises(ValueError, match=r"k must be an integer >= 1 or None; got 0"):
        hill_estimator(x, 0)
    with pytest.raises(ValueError, match=r"k must be an integer in \[1, 8\]; got 9"):
        hill_estimator(x, 9)
    with pytest.raises(TypeError, match="k must be an integer"):
        hill_estimator(x, 2.0)


def test_hill_estimator_no_positive():
    with pytest.raises(ValueError, match="at least two non-zero"):
        hill_estimator(np.zeros(10))


def test_hill_estimator_repeated_extremes():
    assert_equal(hill_estimator(np.array([1.0, 2.0, 2.0, 2.0]), 2), np.inf)


def test_var_ratio():
    assert_allclose(var_ratio(), 1.0)
    assert_allclose(var_ratio(distribution=Normal()), 1.0)

    # p is the probability of an exception: 0.01 is a 99% VaR
    p = 0.01
    nu = 8.0
    gaussian_var = -stats.norm.ppf(p)
    model_var = -StudentsT().ppf(p, [nu])
    assert_allclose(var_ratio(p, StudentsT(), [nu]), gaussian_var / model_var)
    # The default is the 99.9% VaR
    assert_allclose(
        var_ratio(distribution=StudentsT(), parameters=[nu]),
        stats.norm.ppf(0.001) / StudentsT().ppf(0.001, [nu]),
    )


def test_var_ratio_invalid():
    with pytest.raises(ValueError, match=r"p must be a finite float in \(0, 0.5\)"):
        var_ratio(0.5)
    # A confidence level is not a probability of an exception
    with pytest.raises(ValueError, match=r"p must be a finite float in \(0, 0.5\)"):
        var_ratio(0.99)
    with pytest.raises(
        TypeError, match=r"distribution must be an instance of Distribution or None"
    ):
        var_ratio(distribution="normal")


def test_var_ratio_no_level():
    # var_ratio is new in 9.0, and only takes p, the probability of an exception
    data = np.array([-1.5, -0.8, -0.4, 0.2, 0.7, 1.0, 1.8, 2.4])
    mod = ZeroMean(data, volatility=ConstantVariance(), distribution=StudentsT())
    res = mod.fix(np.array([1.0, 8.0]))
    for func in (var_ratio, res.var_ratio):
        with pytest.raises(TypeError, match="level"):
            func(level=0.99)  # type: ignore[call-arg]


def test_var_ratio_invalid_model_var():
    with pytest.raises(ValueError, match="positive finite"):
        var_ratio(distribution=PositiveTailNormal())
    with pytest.raises(ValueError, match="positive finite"):
        var_ratio(distribution=NonfiniteTailNormal())


def test_result_diagnostics():
    data = np.array([-1.5, -0.8, -0.4, 0.2, 0.7, 1.0, 1.8, 2.4])
    mod = ZeroMean(data, volatility=ConstantVariance(), distribution=StudentsT())
    res = mod.fix(np.array([1.0, 8.0]))

    assert_allclose(res.excess_kurtosis, excess_kurtosis(res.std_resid))
    assert_allclose(res.hill_estimator(2), hill_estimator(res.std_resid, 2))
    assert_allclose(res.var_ratio(0.01), var_ratio(0.01, StudentsT(), [8.0]))
    assert_allclose(res.var_ratio(), var_ratio(0.001, StudentsT(), [8.0]))
