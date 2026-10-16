"""
Tests of the FIGARCH variance recursion against a direct (slow) implementation of
the model equations
"""

from arch.compat.numba import HAS_NUMBA

import types

import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.univariate import ConstantMean
import arch.univariate.recursions_python as recpy
from arch.univariate.volatility import FIGARCH

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
RECURSIONS = {"python": recpy.figarch_recursion_python}
if HAS_NUMBA:
    RECURSIONS["numba"] = recpy.figarch_recursion
if not MISSING_EXTENSION:
    RECURSIONS["cython"] = rec_cython.figarch_recursion


def reference_variance(parameters, fresids, p, q, trunc_lag, backcast):
    """
    Direct evaluation of

        h_t = omega / (1 - beta)
              + sum_{i=1}^{trunc} lam_i f_{t-i}
              + (sum_{i=t+1}^{trunc} lam_i) * backcast

    where f_t = |e_t| ** power. Returns the unbounded value.
    """
    nobs = fresids.shape[0]
    lam = rec.figarch_weights(parameters[1:], p, q, trunc_lag)
    omega = parameters[0]
    beta = parameters[1 + p + q] if q else 0.0
    out = np.empty(nobs)
    for t in range(nobs):
        nlags = min(t, trunc_lag)
        # lam[i] multiplies the value i + 1 periods ago
        history = fresids[t - nlags : t][::-1]
        out[t] = (
            omega / (1 - beta) + lam[:nlags].dot(history) + lam[t:].sum() * backcast
        )
    return out


def run_recursion(func, parameters, fresids, p, q, trunc_lag, backcast, bounds):
    nobs = fresids.shape[0]
    sigma2 = np.full(nobs, np.nan)
    var_bounds = np.empty((nobs, 2))
    var_bounds[:, 0] = bounds[0]
    var_bounds[:, 1] = bounds[1]
    func(parameters, fresids, sigma2, p, q, nobs, trunc_lag, backcast, var_bounds)
    return sigma2


@pytest.mark.parametrize("func", RECURSIONS.values(), ids=list(RECURSIONS))
@pytest.mark.parametrize("p", [0, 1], ids=["p0", "p1"])
@pytest.mark.parametrize("q", [0, 1], ids=["q0", "q1"])
@pytest.mark.parametrize(
    "nobs", [1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 31, 40], ids=lambda v: f"nobs{v}"
)
@pytest.mark.parametrize("trunc_lag", [1, 2, 3, 4, 5, 8, 9, 25], ids=lambda v: f"T{v}")
def test_figarch_recursion_matches_reference(func, p, q, nobs, trunc_lag):
    # Covers observation counts that are/are not multiples of the loop blocking,
    # shorter and longer than the truncation lag
    rng = np.random.default_rng(nobs * 100 + trunc_lag)
    fresids = rng.standard_normal(nobs) ** 2
    parameters = [0.4]
    if p:
        parameters.append(0.15)
    parameters.append(0.35)
    if q:
        parameters.append(0.3)
    parameters = np.array(parameters)
    sigma2 = run_recursion(
        func, parameters, fresids, p, q, trunc_lag, 1.7, (1e-300, 1e300)
    )
    expected = reference_variance(parameters, fresids, p, q, trunc_lag, 1.7)
    assert_allclose(sigma2, expected, rtol=1e-12)


@pytest.mark.parametrize("func", RECURSIONS.values(), ids=list(RECURSIONS))
def test_figarch_recursion_bounds(func):
    rng = np.random.default_rng(99)
    nobs = 30
    fresids = rng.standard_normal(nobs) ** 2
    parameters = np.array([1.0, 0.1, 0.4, 0.3])
    unbounded = reference_variance(parameters, fresids, 1, 1, 10, 1.0)
    # Bounds that are violated on both sides
    lower_bound, upper_bound = np.quantile(unbounded, [0.25, 0.75])
    sigma2 = run_recursion(
        func, parameters, fresids, 1, 1, 10, 1.0, (lower_bound, upper_bound)
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
def test_figarch_recursion_python_cython_identical():
    # The compiled and pure Python recursions perform the same operations in the
    # same order
    rng = np.random.default_rng(5)
    fresids = rng.standard_normal(333) ** 2
    parameters = np.array([0.1, 0.2, 0.45, 0.3])
    args = (parameters, fresids, 1, 1, 70, 1.1, (1e-300, 1e300))
    a = run_recursion(recpy.figarch_recursion_python, *args)
    b = run_recursion(rec_cython.figarch_recursion, *args)
    assert_allclose(a, b, rtol=1e-13)


@pytest.mark.parametrize("power", [2.0, 1.0, 1.5], ids=lambda v: f"power{v}")
def test_figarch_compute_variance(power):
    # The volatility process applies the power to the output of the recursion
    y = np.random.default_rng(11).standard_normal(75)
    vol = FIGARCH(1, 1, power=power, truncation=12)
    params = np.array([0.3, 0.1, 0.3, 0.25])
    var_bounds = vol.variance_bounds(y)
    backcast = vol.backcast(y)
    sigma2 = np.empty(75)
    vol.compute_variance(params, y, sigma2, backcast, var_bounds)
    expected = reference_variance(params, np.abs(y) ** power, 1, 1, 12, backcast)
    assert np.all((expected > var_bounds[:, 0]) & (expected < var_bounds[:, 1]))
    assert_allclose(sigma2, expected ** (2.0 / power), rtol=1e-12)


@pytest.mark.parametrize("power", [2.0, 1.0], ids=lambda v: f"power{v}")
def test_figarch_simulation_forecast_first_step(power):
    # The first step of a simulation forecast depends only on observed data, so it
    # must match the one-step analytic forecast, which uses the recursion
    y = np.random.default_rng(3).standard_normal(300)
    vol = FIGARCH(1, 1, power=power, truncation=40)
    res = ConstantMean(y, volatility=vol).fix(np.array([0.0, 0.1, 0.1, 0.3, 0.3]))
    analytic = res.forecast(horizon=1, start=250, reindex=False)
    sim = res.forecast(
        horizon=3,
        start=250,
        method="simulation",
        simulations=20,
        reindex=False,
        random_state=np.random.RandomState(1),
    )
    first_step = sim.simulations.variances[:, :, 0]
    expected = np.broadcast_to(analytic.variance.to_numpy(), first_step.shape)
    assert_allclose(first_step, expected, rtol=1e-12)
