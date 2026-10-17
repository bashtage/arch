"""
Tests of the argument checks of the volatility updaters and ARCHInMeanRecursion.

The compiled recursions read and write their arguments without bounds checks, so
None or an array that is too short used to crash the interpreter or to read and write
memory that does not belong to the array. The compiled and the pure Python
implementations must raise the same errors.
"""

from __future__ import annotations

import re
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

pytestmark = pytest.mark.filterwarnings("ignore::arch.compat.numba.PerformanceWarning")

MODULES = [pytest.param(recpy, id="python")]
if not MISSING_EXTENSION:
    MODULES.append(pytest.param(rec_cython, id="compiled"))

NOBS = 10
EWMA_PARAMS = np.array([0.0, 0.1, 0.9])
VECTOR = "a 1-d C contiguous float64 ndarray"
MATRIX = "a 2-d C contiguous float64 ndarray"


def _message(name: str, rule: str, received: str) -> str:
    return re.escape(f"{name} must be {rule}; got {received}")


@pytest.fixture(params=MODULES)
def rec(request) -> types.ModuleType:
    return request.param


def _inputs(rec: types.ModuleType, nobs: int = NOBS) -> dict:
    updater = rec.EWMAUpdater(0.9)
    updater.initialize_update(EWMA_PARAMS, 1.0, nobs)
    rs = np.random.default_rng(20240925)
    return {
        "recursion": rec.ARCHInMeanRecursion(updater),
        "y": rs.standard_normal(nobs),
        "x": np.ascontiguousarray(rs.standard_normal((nobs, 2))),
        "mean_parameters": np.array([0.1, -0.2, 0.05]),
        "variance_params": EWMA_PARAMS.copy(),
        "sigma2": np.zeros(nobs),
        "var_bounds": np.ones((nobs, 2)) * np.array([1e-8, 1e8]),
        "power": 1.0,
    }


def _recursion(inputs: dict) -> np.ndarray:
    args = dict(inputs)
    return args.pop("recursion").recursion(**args)


def _reference(inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    """The residuals and the EWMA variance, written out directly."""
    y, x, b = inputs["y"], inputs["x"], inputs["mean_parameters"]
    lam = 0.9
    nobs = y.shape[0]
    sigma2 = np.zeros(nobs)
    resids = np.zeros(nobs)
    for t in range(nobs):
        if t == 0:
            sigma2[t] = 1.0
        else:
            sigma2[t] = (1 - lam) * resids[t - 1] ** 2 + lam * sigma2[t - 1]
        resids[t] = y[t] - x[t] @ b[:2] - b[2] * sigma2[t]
    return resids, sigma2


def test_arch_in_mean_matches_reference(rec):
    # The checks accept an exact fit and a sigma2 that is longer than needed
    inputs = _inputs(rec)
    resids = _recursion(inputs)
    expected_resids, expected_sigma2 = _reference(inputs)
    assert_allclose(resids, expected_resids, rtol=1e-12, atol=1e-14)
    assert_allclose(inputs["sigma2"], expected_sigma2, rtol=1e-12, atol=1e-14)

    longer = _inputs(rec)
    longer["sigma2"] = np.zeros(NOBS + 5)
    assert_allclose(_recursion(longer), expected_resids, rtol=1e-12, atol=1e-14)


NONE_RULES = {
    "y": VECTOR,
    "x": f"{MATRIX} with at least 10 rows and 0 columns",
    "mean_parameters": f"{VECTOR} with at least 3 elements",
    "variance_params": VECTOR,
    "sigma2": f"{VECTOR} with at least 10 elements",
    "var_bounds": f"{MATRIX} with at least 10 rows and 2 columns",
}


@pytest.mark.parametrize("name", list(NONE_RULES))
def test_arch_in_mean_none(rec, name):
    inputs = _inputs(rec)
    inputs[name] = None
    if name == "y":
        # With y missing the sample size is unknown, so the others are not checked
        inputs["x"] = None
        rule = VECTOR
    else:
        rule = NONE_RULES[name]
    with pytest.raises(TypeError, match=_message(name, rule, "None (NoneType)")):
        _recursion(inputs)


@pytest.mark.parametrize("size", [0, 1, NOBS - 1])
def test_arch_in_mean_short_sigma2(rec, size):
    inputs = _inputs(rec)
    inputs["sigma2"] = np.zeros(size)
    rule = f"{VECTOR} with at least 10 elements"
    received = f"shape ({size},) (ndarray)"
    with pytest.raises(ValueError, match=_message("sigma2", rule, received)):
        _recursion(inputs)
    assert np.all(inputs["sigma2"] == 0.0)


def test_arch_in_mean_short_inputs(rec):
    # The sample size is the length of y
    inputs = _inputs(rec)
    inputs["y"] = inputs["y"][:3]
    assert _recursion(inputs).shape == (3,)

    inputs = _inputs(rec)
    inputs["x"] = inputs["x"][:4]
    rule = f"{MATRIX} with at least 10 rows and 0 columns"
    with pytest.raises(ValueError, match=_message("x", rule, "shape (4, 2)")):
        _recursion(inputs)

    inputs = _inputs(rec)
    inputs["mean_parameters"] = inputs["mean_parameters"][:2]
    rule = f"{VECTOR} with at least 3 elements"
    received = "shape (2,) (ndarray)"
    with pytest.raises(ValueError, match=_message("mean_parameters", rule, received)):
        _recursion(inputs)

    rule = f"{MATRIX} with at least 10 rows and 2 columns"
    inputs = _inputs(rec)
    inputs["var_bounds"] = inputs["var_bounds"][:6]
    with pytest.raises(ValueError, match=_message("var_bounds", rule, "shape (6, 2)")):
        _recursion(inputs)

    inputs = _inputs(rec)
    inputs["var_bounds"] = np.ascontiguousarray(inputs["var_bounds"][:, :1])
    with pytest.raises(ValueError, match=_message("var_bounds", rule, "shape (10, 1)")):
        _recursion(inputs)


@pytest.mark.parametrize("updater", [None, 1, object()], ids=["None", "int", "object"])
def test_arch_in_mean_updater(rec, updater):
    with pytest.raises(TypeError, match=r"updater must be a VolatilityUpdater; got "):
        rec.ARCHInMeanRecursion(updater)


def test_ewma_initialize_update(rec):
    updater = rec.EWMAUpdater(None)
    rule = f"{VECTOR} with at least 1 element"
    with pytest.raises(TypeError, match=_message("parameters", rule, "None")):
        updater.initialize_update(None, 0.5, NOBS)
    with pytest.raises(ValueError, match=_message("parameters", rule, "shape (0,)")):
        updater.initialize_update(np.empty(0), 0.5, NOBS)

    # The parameter is estimated, so it is used
    params = np.array([0.8])
    updater.initialize_update(params, 0.5, NOBS)
    sigma2 = np.zeros(NOBS)
    resids = np.ones(NOBS)
    bounds = np.ones((NOBS, 2)) * np.array([1e-8, 1e8])
    updater._update_tester(0, params, resids, sigma2, bounds)
    updater._update_tester(1, params, resids, sigma2, bounds)
    assert_allclose(sigma2[:2], [0.5, 0.2 * 1.0 + 0.8 * 0.5], rtol=1e-14)


def test_ewma_fixed_lambda_ignores_parameters(rec):
    # A fixed lambda does not read the parameters
    updater = rec.EWMAUpdater(0.94)
    updater.initialize_update(np.empty(0), 0.5, NOBS)


@pytest.mark.parametrize(("p", "q"), [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_figarch_initialize_update(rec, p, q):
    updater = rec.FIGARCHUpdater(p, q, 2.0, 50)
    size = 2 + p + q
    rule = f"{VECTOR} with at least {size} elements"
    with pytest.raises(TypeError, match=_message("parameters", rule, "None")):
        updater.initialize_update(None, 0.5, NOBS)
    short = np.full(size - 1, 0.1)
    received = f"shape ({size - 1},) (ndarray)"
    with pytest.raises(ValueError, match=_message("parameters", rule, received)):
        updater.initialize_update(short, 0.5, NOBS)


def test_figarch_initialize_update_valid(rec):
    # FIGARCH(0, d, 0) with power 2. At t = 0 every lag is before the sample, so the
    # variance is omega + sum(lambda) * backcast, where lambda are the coefficients
    # of the fractional difference, lambda_i = delta_i for i < truncation
    omega, d, backcast, truncation = 0.1, 0.4, 0.5, 5
    params = np.array([omega, d])
    updater = rec.FIGARCHUpdater(0, 0, 2.0, truncation)
    updater.initialize_update(params, backcast, NOBS)
    delta = [d]
    for i in range(1, truncation):
        delta.append((i - d) / (i + 1) * delta[-1])
    sigma2 = np.zeros(NOBS)
    bounds = np.ones((NOBS, 2)) * np.array([1e-8, 1e8])
    updater._update_tester(0, params, np.zeros(NOBS), sigma2, bounds)
    assert_allclose(sigma2[0], omega + sum(delta) * backcast, rtol=1e-12)


def _garch(rec):
    return rec.GARCHUpdater(1, 0, 1, 2.0)


def _harch(rec):
    return rec.HARCHUpdater(np.array([1, 5], dtype=np.int32))


def _midas(rec):
    return rec.MIDASUpdater(5, False)


def _midas_asym(rec):
    return rec.MIDASUpdater(5, True)


def _egarch(rec):
    return rec.EGARCHUpdater(1, 1, 1)


def _fiaparch(rec):
    return rec.FIAPARCHUpdater(1, 1, 1, 10, True, 1.5)


def _fiaparch_fixed_delta(rec):
    return rec.FIAPARCHUpdater(1, 0, 0, 10, False, 1.5)


PARAMETER_UPDATERS = [
    pytest.param(_garch, 3, id="GARCH"),
    pytest.param(_harch, 3, id="HARCH"),
    pytest.param(_midas, 3, id="MIDAS"),
    pytest.param(_midas_asym, 4, id="MIDAS-asym"),
    pytest.param(_egarch, 4, id="EGARCH"),
    pytest.param(_fiaparch, 6, id="FIAPARCH"),
    pytest.param(_fiaparch_fixed_delta, 3, id="FIAPARCH-fixed-delta"),
]


@pytest.mark.parametrize(("make", "size"), PARAMETER_UPDATERS)
def test_initialize_update_parameters(rec, make, size):
    # The parameters are the view of a longer buffer, so that an unchecked read past
    # the end would not fail
    updater = make(rec)
    rule = f"{VECTOR} with at least {size} elements"
    with pytest.raises(TypeError, match=_message("parameters", rule, "None")):
        updater.initialize_update(None, 0.5, NOBS)
    buffer = np.full(size + 3, 0.1)
    for length in (0, size - 1):
        received = f"shape ({length},) (ndarray)"
        with pytest.raises(ValueError, match=_message("parameters", rule, received)):
            updater.initialize_update(buffer[:length], 0.5, NOBS)
    # The exact length and a longer vector are accepted
    updater.initialize_update(buffer[:size], 0.5, NOBS)
    updater.initialize_update(buffer, 0.5, NOBS)


def _t0(updater, params, backcast):
    updater.initialize_update(params, backcast, NOBS)
    sigma2 = np.zeros(NOBS)
    bounds = np.ones((NOBS, 2)) * np.array([1e-8, 1e8])
    updater._update_tester(0, params, np.zeros(NOBS), sigma2, bounds)
    return sigma2[0]


def test_valid_parameters_first_variance(rec):
    # At t = 0 every lag is before the sample and takes the value of the backcast
    bc = 0.7
    # GARCH(1, 0, 1): omega + (alpha + beta) * backcast
    value = _t0(_garch(rec), np.array([0.1, 0.2, 0.3]), bc)
    assert_allclose(value, 0.1 + 0.5 * bc, rtol=1e-12)
    # HARCH([1, 5]): the weights divided by the lags are applied to lags times
    value = _t0(_harch(rec), np.array([0.1, 0.2, 0.3]), bc)
    assert_allclose(value, 0.1 + 0.5 * bc, rtol=1e-12)
    # MIDAS: the weights sum to 1
    value = _t0(_midas(rec), np.array([0.1, 0.4, 0.5]), bc)
    assert_allclose(value, 0.1 + 0.4 * bc, rtol=1e-12)
    # EGARCH(1, 1, 1): only the beta term is available, it multiplies the backcast
    value = _t0(_egarch(rec), np.array([0.1, 0.2, 0.3, 0.4]), np.log(bc))
    assert_allclose(value, np.exp(0.1 + 0.4 * np.log(bc)), rtol=1e-12)
