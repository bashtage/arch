"""
Tests of the FIAPARCH variance recursion, the updater, simulation and simulation
based forecasts against direct (slow) implementations of the model equations
"""

from arch.compat.numba import HAS_NUMBA

import pickle
import types

import numpy as np
from numpy.testing import assert_allclose
import pytest

import arch.univariate.recursions_python as recpy
from arch.univariate.volatility import FIAPARCH

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


def reference_variance(parameters, resids, p, q, trunc_lag, backcast, gamma, delta):
    """
    Direct evaluation of

        sigma_t^delta = omega / (1 - beta)
                        + sum_{i=1}^{trunc} lam_i s_{t-i}
                        + (sum_{i=t+1}^{trunc} lam_i) * backcast ** (delta / 2)

    where s_t = (|e_t| - gamma * e_t) ** delta.  Returns the unbounded variance.
    """
    nobs = resids.shape[0]
    lam = rec.figarch_weights(parameters[1:], p, q, trunc_lag)
    omega = parameters[0]
    beta = parameters[1 + p + q] if q else 0.0
    shocks = (np.abs(resids) - gamma * resids) ** delta
    sigma_delta = np.empty(nobs)
    for t in range(nobs):
        nlags = min(t, trunc_lag)
        # lam[i] multiplies the shock i + 1 periods ago
        history = shocks[t - nlags : t][::-1]
        sigma_delta[t] = (
            omega / (1 - beta)
            + lam[:nlags].dot(history)
            + lam[t:].sum() * backcast ** (delta / 2.0)
        )
    return sigma_delta ** (2.0 / delta)


# Pure Python implementation, always available. The JIT compiled version is
# the same function, so it is only added when numba is available
RECURSIONS = {"python": recpy.fiaparch_recursion_python}
if HAS_NUMBA:
    RECURSIONS["numba"] = recpy.fiaparch_recursion
if not MISSING_EXTENSION:
    RECURSIONS["cython"] = rec_cython.fiaparch_recursion


def run_recursion(func, parameters, resids, p, q, trunc_lag, backcast, gamma, delta):
    nobs = resids.shape[0]
    sigma2 = np.full(nobs, np.nan)
    sigma_delta = np.full(nobs, np.nan)
    var_bounds = np.empty((nobs, 2))
    var_bounds[:, 0] = 1e-300
    var_bounds[:, 1] = 1e300
    func(
        parameters,
        resids,
        np.abs(resids),
        sigma2,
        sigma_delta,
        p,
        q,
        nobs,
        trunc_lag,
        backcast,
        var_bounds,
        gamma,
        delta,
    )
    return sigma2, sigma_delta


@pytest.mark.parametrize("func", RECURSIONS.values(), ids=list(RECURSIONS))
@pytest.mark.parametrize("p", [0, 1], ids=["p0", "p1"])
@pytest.mark.parametrize("q", [0, 1], ids=["q0", "q1"])
@pytest.mark.parametrize(
    "nobs", [1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 31, 40], ids=lambda v: f"nobs{v}"
)
@pytest.mark.parametrize("trunc_lag", [1, 2, 3, 4, 5, 8, 9, 25], ids=lambda v: f"T{v}")
def test_fiaparch_recursion_matches_reference(func, p, q, nobs, trunc_lag):
    # Covers observation counts that are/are not multiples of the loop blocking,
    # shorter and longer than the truncation lag
    rng = np.random.default_rng(nobs * 100 + trunc_lag)
    resids = rng.standard_normal(nobs) * 1.3
    d = 0.35
    phi = 0.15 if p else None
    beta = 0.3 if q else None
    parameters = [0.4]
    if p:
        parameters.append(phi)
    parameters.append(d)
    if q:
        parameters.append(beta)
    parameters = np.array(parameters)
    gamma, delta, backcast = -0.3, 1.4, 1.7
    sigma2, sigma_delta = run_recursion(
        func, parameters, resids, p, q, trunc_lag, backcast, gamma, delta
    )
    expected = reference_variance(
        parameters, resids, p, q, trunc_lag, backcast, gamma, delta
    )
    assert_allclose(sigma2, expected, rtol=1e-12)
    assert_allclose(sigma_delta, expected ** (delta / 2.0), rtol=1e-12)


@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0, 3.5], ids=lambda v: f"delta{v}")
@pytest.mark.parametrize("gamma", [-0.9, 0.0, 0.9], ids=lambda v: f"gamma{v}")
def test_fiaparch_recursion_delta_gamma(delta, gamma):
    rng = np.random.default_rng(1234)
    resids = rng.standard_normal(60)
    parameters = np.array([0.2, 0.1, 0.4, 0.3])
    for func in RECURSIONS.values():
        sigma2, _ = run_recursion(func, parameters, resids, 1, 1, 17, 1.2, gamma, delta)
        expected = reference_variance(parameters, resids, 1, 1, 17, 1.2, gamma, delta)
        assert_allclose(sigma2, expected, rtol=1e-12)


def test_fiaparch_recursion_python_cython_identical():
    # The compiled and pure Python recursions perform the same operations in the
    # same order
    if MISSING_EXTENSION:
        pytest.skip("Cython extension not available")
    rng = np.random.default_rng(5)
    resids = rng.standard_normal(333)
    parameters = np.array([0.1, 0.2, 0.45, 0.3, 0.25, 1.6])
    a, _ = run_recursion(
        recpy.fiaparch_recursion_python,
        parameters[:4],
        resids,
        1,
        1,
        70,
        1.1,
        0.25,
        1.6,
    )
    b, _ = run_recursion(
        rec_cython.fiaparch_recursion, parameters[:4], resids, 1, 1, 70, 1.1, 0.25, 1.6
    )
    assert_allclose(a, b, rtol=1e-13)


@pytest.mark.parametrize("func", RECURSIONS.values(), ids=list(RECURSIONS))
def test_fiaparch_recursion_bounds(func):
    rng = np.random.default_rng(99)
    nobs = 30
    resids = rng.standard_normal(nobs)
    parameters = np.array([1.0, 0.1, 0.4, 0.3])
    sigma2 = np.zeros(nobs)
    sigma_delta = np.zeros(nobs)
    var_bounds = np.empty((nobs, 2))
    var_bounds[:, 0] = 1.5
    var_bounds[:, 1] = 2.0
    func(
        parameters,
        resids,
        np.abs(resids),
        sigma2,
        sigma_delta,
        1,
        1,
        nobs,
        10,
        1.0,
        var_bounds,
        0.1,
        1.5,
    )
    unbounded = reference_variance(parameters, resids, 1, 1, 10, 1.0, 0.1, 1.5)
    assert np.any(unbounded < 1.5) or np.any(unbounded > 2.0)
    lower = unbounded < 1.5
    upper = unbounded > 2.0
    inside = ~(lower | upper)
    assert_allclose(sigma2[lower], 1.5)
    assert_allclose(sigma2[upper], 2.0 + np.log(unbounded[upper] / 2.0))
    assert_allclose(sigma2[inside], unbounded[inside])
    # sigma_delta holds the bounded value on the sigma ** delta scale
    assert_allclose(sigma_delta, sigma2**0.75)


@pytest.mark.parametrize("module", [recpy, rec], ids=["python", "default"])
@pytest.mark.parametrize("est_delta", [True, False], ids=["est_delta", "fixed_delta"])
@pytest.mark.parametrize("nobs", [1, 6, 7, 40], ids=lambda v: f"nobs{v}")
def test_fiaparch_updater_matches_recursion(module, est_delta, nobs):
    # The updater is used by ARCH-in-mean, which calls it one observation at a time
    rng = np.random.default_rng(321 + nobs)
    trunc_lag = 9
    resids = rng.standard_normal(nobs)
    vol = FIAPARCH(1, 1, 1, delta=None if est_delta else 1.5, truncation=trunc_lag)
    parameters = np.array([0.2, 0.1, 0.4, 0.3, 0.2] + ([1.3] if est_delta else []))
    var_bounds = vol.variance_bounds(resids)
    backcast = vol.backcast(resids)
    sigma2_ref = np.zeros(nobs)
    vol.compute_variance(parameters, resids, sigma2_ref, backcast, var_bounds)

    sigma2 = np.full(nobs, np.nan)
    fu = module.FIAPARCHUpdater(1, 1, 1, trunc_lag, est_delta, 1.5)
    fu.initialize_update(parameters, backcast, nobs)
    for t in range(nobs):
        fu._update_tester(t, parameters, resids, sigma2, var_bounds)
        if t == nobs // 2:
            fu = pickle.loads(pickle.dumps(fu))
    assert_allclose(sigma2, sigma2_ref, rtol=1e-12)


def simulate_reference(vol, parameters, nobs, errors, burn, delta, gamma):
    """The original, lag-by-lag, implementation of the simulation"""
    truncation = vol.truncation
    p, q = vol.p, vol.q
    fig_params = parameters[: 1 + p + 1 + q]
    lam = rec.figarch_weights(fig_params[1:], p, q, truncation)
    beta = float(fig_params[-1]) if q else 0.0
    persistence = np.sum(lam)
    initial_value = float(fig_params[0]) / (1 - beta) / (1 - persistence)
    total = truncation + nobs + burn
    sigma2 = np.empty(total)
    data = np.empty(total)
    sigma_delta = np.empty(total)
    sigma_delta[:truncation] = initial_value
    sigma2[:truncation] = initial_value ** (2.0 / delta)
    data[:truncation] = np.sqrt(sigma2[:truncation]) * errors[:truncation]
    omega_tilde = float(fig_params[0]) / (1 - beta)
    for t in range(truncation, total):
        total_shock = 0.0
        for i in range(truncation):
            lagged = data[t - 1 - i]
            total_shock += lam[i] * (abs(lagged) - gamma * lagged) ** delta
        sigma_delta[t] = omega_tilde + total_shock
        sigma2[t] = sigma_delta[t] ** (2.0 / delta)
        data[t] = errors[t] * np.sqrt(sigma2[t])
    return data[truncation + burn :], sigma2[truncation + burn :]


@pytest.mark.parametrize("truncation", [1, 5, 40], ids=lambda v: f"T{v}")
@pytest.mark.parametrize("est_delta", [True, False], ids=["est_delta", "fixed_delta"])
def test_fiaparch_simulate_matches_reference(truncation, est_delta):
    vol = FIAPARCH(1, 1, 1, delta=None if est_delta else 1.2, truncation=truncation)
    delta = 1.7 if est_delta else 1.2
    parameters = np.array([0.1, 0.1, 0.3, 0.2, 0.3] + ([delta] if est_delta else []))
    nobs, burn = 50, 20
    errors = np.random.default_rng(8).standard_normal(truncation + nobs + burn)

    def rng(size):
        assert size == truncation + nobs + burn
        return errors

    data, sigma2 = vol.simulate(parameters, nobs, rng, burn=burn)
    ref_data, ref_sigma2 = simulate_reference(
        vol, parameters, nobs, errors, burn, delta, 0.3
    )
    assert_allclose(data, ref_data, rtol=1e-12)
    assert_allclose(sigma2, ref_sigma2, rtol=1e-12)
