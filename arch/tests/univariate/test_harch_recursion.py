"""
Tests of the HARCH variance recursion and simulation against direct (slow)
implementations of the model equations
"""

from arch.compat.numba import HAS_NUMBA

import types

import numpy as np
from numpy.testing import assert_allclose
import pytest

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
RECURSIONS = {"python": recpy.harch_recursion_python}
if HAS_NUMBA:
    RECURSIONS["numba"] = recpy.harch_recursion
if not MISSING_EXTENSION:
    RECURSIONS["cython"] = rec_cython.harch_recursion

LAGS = {
    "1": [1],
    "1-5-22": [1, 5, 22],
    "3-7": [3, 7],
    "1-2-3-4-5": [1, 2, 3, 4, 5],
    "10": [10],
    "2-9": [2, 9],
}


def reference_variance(parameters, resids, lags, backcast):
    """
    sigma2_t = omega + sum_i alpha_i / lag_i * sum_{j=1}^{lag_i} e_{t-j}^2

    where e_s^2 is replaced by the backcast for s < 0.
    """
    nobs = resids.shape[0]
    resids2 = np.concatenate([np.full(max(lags), backcast), resids**2])
    offset = max(lags)
    out = np.empty(nobs)
    for t in range(nobs):
        total = parameters[0]
        for i, lag in enumerate(lags):
            total += (
                parameters[1 + i] / lag * np.sum(resids2[offset + t - lag : offset + t])
            )
        out[t] = total
    return out


def run_recursion(func, parameters, resids, lags, backcast, bounds):
    nobs = resids.shape[0]
    sigma2 = np.full(nobs, np.nan)
    var_bounds = np.empty((nobs, 2))
    var_bounds[:, 0] = bounds[0]
    var_bounds[:, 1] = bounds[1]
    func(
        parameters,
        resids,
        sigma2,
        np.asarray(lags, dtype=np.int32),
        nobs,
        backcast,
        var_bounds,
    )
    return sigma2


@pytest.mark.parametrize("func", RECURSIONS.values(), ids=list(RECURSIONS))
@pytest.mark.parametrize("lags", LAGS.values(), ids=list(LAGS))
@pytest.mark.parametrize(
    "nobs",
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 23, 29, 40, 63],
    ids=lambda v: f"nobs{v}",
)
def test_harch_recursion_matches_reference(func, lags, nobs):
    # Covers observation counts that are/are not multiples of the loop blocking,
    # shorter and longer than the longest lag
    rng = np.random.default_rng(nobs * 100 + sum(lags))
    resids = rng.standard_normal(nobs) * 1.3
    parameters = np.concatenate([[0.2], rng.uniform(0.05, 0.3, len(lags))])
    sigma2 = run_recursion(func, parameters, resids, lags, 1.4, (1e-300, 1e300))
    expected = reference_variance(parameters, resids, lags, 1.4)
    assert_allclose(sigma2, expected, rtol=1e-12)


@pytest.mark.parametrize("func", RECURSIONS.values(), ids=list(RECURSIONS))
def test_harch_recursion_bounds(func):
    rng = np.random.default_rng(99)
    resids = rng.standard_normal(60)
    lags = [1, 5, 22]
    parameters = np.array([0.1, 0.2, 0.3, 0.4])
    unbounded = reference_variance(parameters, resids, lags, 1.0)
    # Bounds that are violated on both sides, including the first observations
    lower_bound, upper_bound = np.quantile(unbounded, [0.25, 0.75])
    sigma2 = run_recursion(
        func, parameters, resids, lags, 1.0, (lower_bound, upper_bound)
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
def test_harch_recursion_python_cython_identical():
    # The compiled and pure Python recursions perform the same operations in the
    # same order
    rng = np.random.default_rng(5)
    resids = rng.standard_normal(333)
    parameters = np.array([0.1, 0.2, 0.3, 0.2])
    args = (parameters, resids, [1, 5, 22], 1.1, (1e-300, 1e300))
    a = run_recursion(recpy.harch_recursion_python, *args)
    b = run_recursion(rec_cython.harch_recursion, *args)
    assert_allclose(a, b, rtol=1e-13)
