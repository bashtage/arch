import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.bootstrap import IIDBootstrap
from arch.bootstrap.base import _loo_jackknife


@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize(
    "method", ["norm", "basic", "percentile", "bc", "bca", "studentized"]
)
def test_conf_int_reused_statistic_output(scalar, method):
    x = np.random.default_rng(0).standard_exponential((40, 2))
    if scalar:
        x = x[:, 0]
    output = np.empty(()) if scalar else np.empty(2)

    def statistic(y):
        return np.mean(y, axis=0, out=output)

    expected = IIDBootstrap(x, seed=1).conf_int(
        lambda y: y.mean(0), reps=100, method=method, studentize_reps=15
    )
    actual = IIDBootstrap(x, seed=1).conf_int(
        statistic, reps=100, method=method, studentize_reps=15
    )
    assert_allclose(actual, expected)


@pytest.mark.parametrize("method", ["cov", "var"])
@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("complex_statistic", [False, True])
def test_covariance_reused_statistic_output(method, scalar, complex_statistic):
    x = np.arange(40.0).reshape(20, 2)
    if scalar:
        x = x[:, 0]
    factor = 1.0 + 2.0j if complex_statistic else 1.0
    output = np.empty(() if scalar else 2, dtype=np.result_type(factor))

    def statistic(y):
        return np.multiply(np.mean(y, axis=0), factor, out=output)

    expected = getattr(IIDBootstrap(x, seed=1), method)(
        lambda y: y.mean(0) * factor, reps=100, recenter=False
    )
    actual = getattr(IIDBootstrap(x, seed=1), method)(
        statistic, reps=100, recenter=False
    )
    assert_allclose(actual, expected)


@pytest.mark.parametrize("scalar", [True, False])
def test_jackknife_reused_statistic_output(scalar):
    x = np.arange(40.0).reshape(20, 2)
    if scalar:
        x = x[:, 0]
    output = np.empty(()) if scalar else np.empty(2)

    def statistic(y):
        return np.mean(y, axis=0, out=output)

    expected = _loo_jackknife(lambda y: y.mean(0), len(x), (x,), {})
    actual = _loo_jackknife(statistic, len(x), (x,), {})
    assert_allclose(actual, expected)
