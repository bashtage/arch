"""
Tests of fit_smooth_pval_model, which fits the p-value models of the unit root and
cointegration tests from simulated quantiles.

The model is Phi(small(stat)) for stat <= tau_star and Phi(large(stat)) above it,
where small is a polynomial in log|stat| or in stat and large is a cubic in stat. The
p-value is 0 below tau_min and 1 above tau_max.
"""

import importlib.util
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats

PROBS = np.concatenate(
    [np.arange(1, 10) / 1000.0, np.arange(1, 100) / 100.0, [0.991, 0.995, 0.999]]
)
NORMAL_QUANTILES = stats.norm.ppf(PROBS)
# Skewed distributions with no exact representation as the model, and the form of
# the small p polynomial: number of coefficients and whether it is in log|stat|
CASES = {
    # Similar to DFGLS: the small p quadratic turns far in the left tail
    "left-skew-quadratic": (
        -1.5 + NORMAL_QUANTILES - 0.03 * NORMAL_QUANTILES**2,
        3,
        False,
    ),
    "right-skew-quadratic": (
        -1.0 + NORMAL_QUANTILES + 0.1 * NORMAL_QUANTILES**2,
        3,
        False,
    ),
    "student-t-cubic": (-1.5 + stats.t(5).ppf(PROBS), 4, False),
    "logistic-cubic": (-2.0 + stats.logistic.ppf(PROBS), 4, False),
    # Similar to ADF-Z
    "exponential-log": (-4.0 * np.exp(-0.7 * NORMAL_QUANTILES), 4, True),
}
MAX_TAIL_STEP = 1e-12


@pytest.fixture(scope="module")
def simulation_shared():
    path = (
        Path(__file__).parents[2]
        / "unitroot"
        / "critical_values"
        / "simulation"
        / "shared.py"
    )
    if not path.exists():
        pytest.skip("The simulation scripts are not available")
    spec = importlib.util.spec_from_file_location("simulation_shared", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulated(base: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(20261030)
    noise = 1 + 1e-4 * rng.standard_normal((PROBS.shape[0], 50))
    return pd.DataFrame(base[:, None] * noise, index=pd.Index(PROBS))


def ppf_functions(res, use_log):
    small, large = np.array(res.small_p), np.array(res.large_p)

    def ppf_small(x):
        x = np.asarray(x, dtype=float)
        return np.polyval(small[::-1], np.log(np.abs(x)) if use_log else x)

    def ppf_large(x):
        return np.polyval(large[::-1], np.asarray(x, dtype=float))

    return ppf_small, ppf_large


def pvalue(res, use_log, stat):
    """The p-value as the tests compute it from the fitted model"""
    ppf_small, ppf_large = ppf_functions(res, use_log)
    stat = np.asarray(stat, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        q = np.where(stat <= res.tau_star, ppf_small(stat), ppf_large(stat))
    p = stats.norm.cdf(q)
    p = np.where(stat < res.tau_min, 0.0, p)
    return np.where(stat > res.tau_max, 1.0, p)


@pytest.mark.parametrize("case", list(CASES))
def test_fit_is_smooth_and_monotone(simulation_shared, case):
    base, small_order, use_log = CASES[case]
    quantiles = simulated(base)
    res = simulation_shared.fit_smooth_pval_model(
        quantiles,
        small_order=small_order,
        use_log=use_log,
        max_tail_step=MAX_TAIL_STEP,
    )
    assert len(res.small_p) == small_order
    assert len(res.large_p) == 4
    star = res.tau_star
    ppf_small, ppf_large = ppf_functions(res, use_log)
    # Same value and slope at the junction
    assert_allclose(ppf_small(star), ppf_large(star), rtol=0, atol=1e-8)
    assert ppf_large(star) >= ppf_small(star)
    step = 1e-5
    slope_small = (ppf_small(star) - ppf_small(star - step)) / step
    slope_large = (ppf_large(star + step) - ppf_large(star)) / step
    assert slope_small > 0
    assert_allclose(slope_small, slope_large, rtol=1e-3)
    # The p-value is set to 0 and 1 only outside of the simulated quantiles, where
    # it steps by at most MAX_TAIL_STEP
    mean = quantiles.mean(axis=1).to_numpy()
    assert res.tau_min < mean.min()
    assert res.tau_max > mean.max()
    if np.isfinite(res.tau_min):
        p_min = pvalue(res, use_log, np.nextafter(res.tau_min, np.inf))
        assert 0 <= p_min <= MAX_TAIL_STEP
    if np.isfinite(res.tau_max):
        p_max = pvalue(res, use_log, np.nextafter(res.tau_max, -np.inf))
        assert 1 - MAX_TAIL_STEP <= p_max <= 1
    # Monotone on a wide grid that includes both bounds
    span = mean.max() - mean.min()
    grid = np.linspace(mean.min() - 20 * span, mean.max() + 20 * span, 40001)
    grid = np.concatenate([grid, [star, res.tau_min, res.tau_max]])
    grid = np.sort(grid[np.isfinite(grid)])
    if use_log:
        grid = grid[grid != 0]
    pvals = pvalue(res, use_log, grid)
    assert np.all(np.isfinite(pvals))
    assert np.all(np.diff(pvals) >= 0)
    assert pvals[0] < 1e-6
    assert pvals[-1] > 1 - 1e-6
    # The model reproduces the simulated quantiles. The largest errors seen are 8.7e-4
    # in the tail, p <= 0.15, and 1.3e-2 in the body, since polynomials cannot track
    # these distributions exactly
    fitted = pvalue(res, use_log, mean)
    tail = PROBS <= 0.15
    assert_allclose(fitted[tail], PROBS[tail], rtol=0, atol=1.5e-3)
    assert_allclose(fitted, PROBS, rtol=0, atol=2e-2)


def test_fit_turning_point_needs_max_tail_step(simulation_shared):
    base, small_order, use_log = CASES["left-skew-quadratic"]
    quantiles = simulated(base)
    res = simulation_shared.fit_smooth_pval_model(
        quantiles,
        small_order=small_order,
        use_log=use_log,
        max_tail_step=MAX_TAIL_STEP,
    )
    # The quadratic is convex, and so the p-value is 0 below its vertex
    a0, a1, a2 = res.small_p
    assert a2 > 0
    vertex = -a1 / (2 * a2)
    # tau_min is rounded toward tau_star, so that the quadratic is increasing on
    # [tau_min, tau_star]
    assert vertex <= res.tau_min < vertex + 1e-6 * abs(vertex)
    assert stats.norm.cdf(a0 + a1 * vertex + a2 * vertex**2) <= MAX_TAIL_STEP
    with pytest.raises(RuntimeError, match="No junction gives a monotone"):
        simulation_shared.fit_smooth_pval_model(
            quantiles, small_order=small_order, use_log=use_log
        )


def test_fit_default_is_log_cubic(simulation_shared):
    quantiles = simulated(CASES["exponential-log"][0])
    default = simulation_shared.fit_smooth_pval_model(quantiles)
    explicit = simulation_shared.fit_smooth_pval_model(
        quantiles, small_order=4, use_log=True, max_tail_step=None
    )
    assert default == explicit
    assert default.tau_min == -np.inf
    assert default.tau_max == np.inf


@pytest.mark.parametrize("digits", [10, 15])
def test_fit_significant_digits(simulation_shared, digits):
    base, small_order, use_log = CASES["student-t-cubic"]
    res = simulation_shared.fit_smooth_pval_model(
        simulated(base),
        small_order=small_order,
        use_log=use_log,
        max_tail_step=MAX_TAIL_STEP,
        significant_digits=digits,
    )
    for value in res.small_p + res.large_p:
        assert float(f"{value:.{digits - 1}e}") == value


def test_fit_small_order_invalid(simulation_shared):
    quantiles = simulated(CASES["exponential-log"][0])
    with pytest.raises(ValueError, match="small_order must be 3 or 4"):
        simulation_shared.fit_smooth_pval_model(quantiles, small_order=5)


@pytest.mark.parametrize(
    ("small_order", "use_log"),
    [(3, False), (4, False), (3, True), (4, True)],
    ids=["quadratic", "cubic", "log-quadratic", "log-cubic"],
)
def test_junction_constraints(simulation_shared, small_order, use_log):
    rng = np.random.default_rng(1234)
    tau = -2.7
    params = rng.standard_normal(small_order + 4)
    small, large = params[:small_order], params[small_order:]

    def ppf_small(x):
        return np.polyval(small[::-1], np.log(abs(x)) if use_log else x)

    def ppf_large(x):
        return np.polyval(large[::-1], x)

    constraints = simulation_shared._junction_constraints(tau, small_order, use_log)
    assert constraints.shape == (2, small_order + 4)
    value, slope = constraints @ params
    assert_allclose(value, ppf_small(tau) - ppf_large(tau), rtol=1e-12)
    step = 1e-6
    slope_small = (ppf_small(tau + step) - ppf_small(tau - step)) / (2 * step)
    slope_large = (ppf_large(tau + step) - ppf_large(tau - step)) / (2 * step)
    assert_allclose(slope, slope_small - slope_large, rtol=1e-6)


def test_tail_end(simulation_shared):
    tail_end = simulation_shared._tail_end
    # -x**2 falls on [1, inf) without turning
    assert tail_end(np.array([0.0, 0.0, -1.0]), 1.0, 1.0) == np.inf
    # x**2 rises going left from -1
    assert tail_end(np.array([0.0, 0.0, 1.0]), -1.0, -1.0) is None
    # x**2 - 4x falls on [0, 2] and turns at 2
    assert_allclose(tail_end(np.array([0.0, -4.0, 1.0]), 0.0, 1.0), 2.0)
    # 2x + x**3 / 3 rises everywhere, so -(2x + x**3 / 3) falls to the right
    assert tail_end(-np.array([0.0, 2.0, 0.0, 1 / 3]), -5.0, 1.0) == np.inf
    # x - x**3 / 3 turns at -1 and 1, going left from 0.5 it falls until -1
    coefs = np.array([0.0, 1.0, 0.0, -1 / 3])
    assert_allclose(tail_end(coefs, 0.5, -1.0), -1.0)
    # A linear polynomial has no turning point
    assert tail_end(np.array([1.0, -2.0]), 3.0, 1.0) == np.inf


def test_round_toward(simulation_shared):
    round_toward = simulation_shared._round_toward
    assert round_toward(-17.561302895074206, 0.0, 10) == -17.56130289
    assert round_toward(-17.561302895074206, -100.0, 10) == -17.5613029
    assert round_toward(4.488823877123, 0.0, 10) == 4.488823877
    assert round_toward(4.488823877123, 10.0, 10) == 4.488823878
    assert round_toward(-np.inf, 0.0, 10) == -np.inf
    assert round_toward(np.inf, 0.0, 10) == np.inf
