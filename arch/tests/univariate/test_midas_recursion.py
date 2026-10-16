"""
Tests of the MIDAS variance recursion against a direct (slow) implementation of
the model equations
"""

from arch.compat.numba import HAS_NUMBA

import types

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import gammaln

import arch.univariate.recursions_python as recpy

try:
    import arch.univariate.recursions as rec_cython

    MISSING_EXTENSION = False
except ImportError:
    MISSING_EXTENSION = True

if MISSING_EXTENSION:
    rec: types.ModuleType = recpy
else:
    rec = rec_cython

pytestmark = pytest.mark.filterwarnings("ignore::arch.compat.numba.PerformanceWarning")

# Pure Python implementation, always available. The JIT compiled version is
# the same function, so it is only added when numba is available
RECURSIONS = {"python": recpy.midas_recursion_python}
if HAS_NUMBA:
    RECURSIONS["numba"] = recpy.midas_recursion
if not MISSING_EXTENSION:
    RECURSIONS["cython"] = rec_cython.midas_recursion


def midas_weights(m, theta):
    j = np.arange(1.0, m + 1)
    w = np.exp(gammaln(theta + j) - gammaln(j + 1) - gammaln(theta))
    return w / w.sum()


def reference_variance(parameters, weights, resids, backcast):
    """
    sigma2_t = omega + sum_{i=1}^m w_i (alpha + gamma 1[e_{t-i} < 0]) e_{t-i}^2

    where the terms before the start of the sample are w_i (alpha + gamma / 2)
    times the backcast.
    """
    omega, alpha, gamma = parameters
    nobs = resids.shape[0]
    m = weights.shape[0]
    out = np.empty(nobs)
    for t in range(nobs):
        total = omega
        for i in range(m):
            if t - i - 1 >= 0:
                x = resids[t - i - 1]
                total += weights[i] * (alpha + gamma * (x < 0)) * x**2
            else:
                total += weights[i] * (alpha + 0.5 * gamma) * backcast
        out[t] = total
    return out


def run_recursion(func, parameters, weights, resids, backcast, bounds):
    nobs = resids.shape[0]
    sigma2 = np.full(nobs, np.nan)
    var_bounds = np.empty((nobs, 2))
    var_bounds[:, 0] = bounds[0]
    var_bounds[:, 1] = bounds[1]
    func(parameters, weights, resids, sigma2, nobs, backcast, var_bounds)
    return sigma2


@pytest.mark.parametrize("func", RECURSIONS.values(), ids=list(RECURSIONS))
@pytest.mark.parametrize("m", [1, 2, 3, 5, 8, 22], ids=lambda v: f"m{v}")
@pytest.mark.parametrize("gamma", [0.0, 0.25], ids=["sym", "asym"])
@pytest.mark.parametrize(
    "nobs",
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 23, 29, 40, 63],
    ids=lambda v: f"nobs{v}",
)
def test_midas_recursion_matches_reference(func, m, gamma, nobs):
    # Covers observation counts that are/are not multiples of the loop blocking,
    # shorter and longer than the number of lags
    rng = np.random.default_rng(nobs * 100 + m)
    resids = rng.standard_normal(nobs) * 1.3
    parameters = np.array([0.2, 0.55, gamma])
    weights = midas_weights(m, 0.7)
    sigma2 = run_recursion(func, parameters, weights, resids, 1.4, (1e-300, 1e300))
    expected = reference_variance(parameters, weights, resids, 1.4)
    assert_allclose(sigma2, expected, rtol=1e-12)


@pytest.mark.parametrize("func", RECURSIONS.values(), ids=list(RECURSIONS))
def test_midas_recursion_bounds(func):
    rng = np.random.default_rng(99)
    resids = rng.standard_normal(60) * 0.5
    weights = midas_weights(10, 0.7)
    parameters = np.array([0.0, 0.5, 0.2])
    unbounded = reference_variance(parameters, weights, resids, 0.3)
    # Bounds that are violated on both sides, including the first observations
    lower_bound, upper_bound = np.quantile(unbounded, [0.25, 0.75])
    sigma2 = run_recursion(
        func, parameters, weights, resids, 0.3, (lower_bound, upper_bound)
    )
    lower = unbounded < lower_bound
    upper = unbounded > upper_bound
    inside = ~(lower | upper)
    assert lower.sum() > 2
    assert upper.sum() > 2
    assert_allclose(sigma2[lower], lower_bound)
    assert_allclose(sigma2[upper], upper_bound + np.log(unbounded[upper] / upper_bound))
    assert_allclose(sigma2[inside], unbounded[inside])


@pytest.mark.skipif(MISSING_EXTENSION, reason="Cython extension not available")
def test_midas_recursion_python_cython_identical():
    # The compiled and pure Python recursions perform the same operations in the
    # same order
    rng = np.random.default_rng(5)
    resids = rng.standard_normal(333)
    parameters = np.array([0.1, 0.4, 0.2])
    weights = midas_weights(22, 0.8)
    args = (parameters, weights, resids, 1.1, (1e-300, 1e300))
    a = run_recursion(recpy.midas_recursion_python, *args)
    b = run_recursion(rec_cython.midas_recursion, *args)
    assert_allclose(a, b, rtol=1e-13)
