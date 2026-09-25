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
    with pytest.raises(ValueError, match="k must satisfy"):
        hill_estimator(x, 0)
    with pytest.raises(ValueError, match="k must satisfy"):
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

    level = 0.99
    nu = 8.0
    gaussian_var = -stats.norm.ppf(1.0 - level)
    model_var = -StudentsT().ppf(1.0 - level, [nu])
    assert_allclose(var_ratio(level, StudentsT(), [nu]), gaussian_var / model_var)


def test_var_ratio_invalid():
    with pytest.raises(ValueError, match=r"larger than 0\.5"):
        var_ratio(0.5)
    with pytest.raises(TypeError, match="must inherit"):
        var_ratio(distribution="normal")


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
    assert_allclose(res.var_ratio(0.99), var_ratio(0.99, StudentsT(), [8.0]))
