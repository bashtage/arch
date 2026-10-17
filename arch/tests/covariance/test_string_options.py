"""String options of the covariance estimators ignore case"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.covariance.kernel import KERNELS, get_kernel_estimator
from arch.covariance.var import PreWhitenedRecolored


@pytest.fixture(scope="module")
def data():
    return np.random.default_rng(5).standard_normal((200, 2))


@pytest.mark.parametrize("kernel", KERNELS)
def test_get_kernel_estimator_case(kernel):
    estimator = get_kernel_estimator(kernel)
    assert estimator.__name__ == kernel
    for spelling in (kernel.lower(), kernel.upper(), kernel.swapcase()):
        assert get_kernel_estimator(spelling) is estimator


@pytest.mark.parametrize("method", ["aic", "hqc", "bic"])
def test_var_method_case(data, method):
    expected = PreWhitenedRecolored(data, method=method, max_lag=3)
    for spelling in (method.upper(), method.capitalize()):
        est = PreWhitenedRecolored(data, method=spelling, max_lag=3)
        assert_allclose(est.cov.long_run, expected.cov.long_run)


@pytest.mark.parametrize("kernel", ["Bartlett", "Parzen", "QuadraticSpectral"])
def test_var_kernel_case(data, kernel):
    expected = PreWhitenedRecolored(data, lags=1, kernel=kernel, bandwidth=3.0)
    for spelling in (kernel.lower(), kernel.upper()):
        est = PreWhitenedRecolored(data, lags=1, kernel=spelling, bandwidth=3.0)
        assert_allclose(est.cov.long_run, expected.cov.long_run)


@pytest.mark.parametrize(
    ("bad", "exc"),
    [("bogus", ValueError), ("", ValueError), (None, TypeError), (1, TypeError)],
)
def test_var_method_invalid(data, bad, exc):
    match = r"method must be one of 'aic', 'hqc' or 'bic'; got "
    with pytest.raises(exc, match=match):
        PreWhitenedRecolored(data, method=bad)
