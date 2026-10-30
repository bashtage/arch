"""
Tests that the ADF-Z p-value is a smooth function of the test statistic.

The p-value is a cubic in log|stat| below adf_z_star and a cubic in stat above it.
The two pieces are estimated jointly so that they have the same value and the same
slope at adf_z_star, and the p-value is monotone in the statistic.
"""

import importlib.util
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats

from arch.unitroot.critical_values.dickey_fuller import adf_z_star
from arch.unitroot.unitroot import mackinnonp

TRENDS = ["n", "c", "ct", "ctt"]

# Mean over the 2,000 simulation files adf_z_<trend>-NNNN.npz (written by the
# adf_z_critical_values_simulation scripts in critical_values/simulation) of the 1, 5,
# 10, 50 and 90 percentiles of the ADF-Z statistic with T=2000 observations, rounded
# to 4 decimals. These are the data that the p-value models are fit to, and are the
# reference for the accuracy of the fitted p-values.
PERCENTILES = np.array([0.01, 0.05, 0.10, 0.50, 0.90])
SIMULATED_QUANTILES = {
    "n": [-13.6630, -8.0260, -5.7053, -0.8518, 0.9285],
    "c": [-20.5667, -14.0644, -11.2311, -4.3534, -0.8439],
    "ct": [-29.2396, -21.6459, -18.1980, -9.0901, -3.7642],
    "ctt": [-36.4242, -28.0074, -24.1075, -13.3801, -6.6372],
}
# The largest error seen is 7.8e-5 for the 1, 5 and 10 percentiles and 3.1e-3 for the
# 50 and 90 percentiles (cubic polynomials cannot track the whole distribution)
ATOL = np.array([1.5e-4, 1.5e-4, 1.5e-4, 5e-3, 5e-3])

# Wide enough that the p-value is 0 or 1 to machine precision at the ends
GRID_LIMITS = {"n": -200.0, "c": -200.0, "ct": -400.0, "ctt": -600.0}


def adf_z_pvalue(stat: float, trend: str) -> float:
    return mackinnonp(stat, regression=trend, num_unit_roots=1, dist_type="adf-z")


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_continuous_at_star(trend):
    star = adf_z_star[trend]
    # stat <= star uses the small p model and the next float is in the large p model
    p_small = adf_z_pvalue(star, trend)
    p_large = adf_z_pvalue(float(np.nextafter(star, 0.0)), trend)
    assert_allclose(p_small, p_large, rtol=0, atol=1e-7)
    for step in (1e-3, 1e-6):
        p_small = adf_z_pvalue(star - step, trend)
        p_large = adf_z_pvalue(star + step, trend)
        # p changes by at most step times a slope no larger than 0.2
        assert_allclose(p_small, p_large, rtol=0, atol=0.5 * step)


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_derivative_continuous_at_star(trend):
    star = adf_z_star[trend]
    step = 1e-6
    # One-sided finite differences, using the small p model on the left and the
    # large p model on the right
    left = (adf_z_pvalue(star, trend) - adf_z_pvalue(star - step, trend)) / step
    right = (
        adf_z_pvalue(star + step, trend)
        - adf_z_pvalue(float(np.nextafter(star, 0.0)), trend)
    ) / step
    assert left > 0
    assert_allclose(left, right, rtol=1e-4)


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_monotone_and_bounded(trend):
    star = adf_z_star[trend]
    lower = GRID_LIMITS[trend]
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(lower, 10.0, 8001),
                star + np.linspace(-0.5, 0.5, 4001),
                -np.logspace(-6, 12, 50)[::-1],
                np.logspace(-6, 12, 50),
            ]
        )
    )
    pvals = np.array([adf_z_pvalue(float(stat), trend) for stat in grid])
    assert np.all(np.isfinite(pvals))
    assert np.all(pvals >= 0.0)
    assert np.all(pvals <= 1.0)
    # Allow for rounding when two points of the grid are within 1e-15 of each other
    assert np.all(np.diff(pvals) >= -1e-14)
    # The p-value is not constant in the part of the distribution that matters
    middle = [
        adf_z_pvalue(float(stat), trend) for stat in np.linspace(2 * star, 2, 2001)
    ]
    assert np.all(np.diff(middle) > 0.0)
    assert adf_z_pvalue(-np.inf, trend) == 0.0
    assert adf_z_pvalue(np.inf, trend) == 1.0
    assert pvals[0] < 1e-12
    assert pvals[-1] > 1 - 1e-12


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_matches_simulated_distribution(trend):
    quantiles = SIMULATED_QUANTILES[trend]
    pvals = np.array([adf_z_pvalue(stat, trend) for stat in quantiles])
    # assert_allclose does not take an array for atol
    assert np.all(np.abs(pvals - PERCENTILES) <= ATOL)


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


def test_fit_smooth_pval_model(simulation_shared):
    # A skewed distribution, which has no exact representation as the model
    probs = np.concatenate(
        [np.arange(1, 10) / 1000.0, np.arange(1, 100) / 100.0, [0.991, 0.995, 0.999]]
    )
    base = -4.0 * np.exp(-0.7 * stats.norm.ppf(probs))
    rng = np.random.default_rng(20251030)
    noise = 1 + 1e-4 * rng.standard_normal((probs.shape[0], 50))

    quantiles = pd.DataFrame(base[:, None] * noise, index=pd.Index(probs))
    res = simulation_shared.fit_smooth_pval_model(quantiles)
    small, large = np.array(res.small_p), np.array(res.large_p)
    star = res.tau_star
    assert star < 0
    assert res.tau_min == -np.inf
    assert res.tau_max == np.inf

    def ppf_small(x):
        return np.polyval(small[::-1], np.log(np.abs(x)))

    def ppf_large(x):
        return np.polyval(large[::-1], x)

    assert_allclose(ppf_small(star), ppf_large(star), rtol=0, atol=1e-8)
    step = 1e-5
    slope_small = (ppf_small(star) - ppf_small(star - step)) / step
    slope_large = (ppf_large(star + step) - ppf_large(star)) / step
    assert_allclose(slope_small, slope_large, rtol=1e-3)
    grid = np.linspace(2.0 * base[0], 20.0, 20001)
    below = grid[(grid <= star) & (grid < 0)]
    above = grid[grid > star]
    values = np.concatenate([ppf_small(below), ppf_large(above)])
    assert np.all(np.diff(values) >= 0)
    # The model reproduces the simulated quantiles
    mean = quantiles.mean(axis=1).to_numpy()
    fitted = stats.norm.cdf(np.where(mean <= star, ppf_small(mean), ppf_large(mean)))
    assert_allclose(fitted, probs, rtol=0, atol=2e-3)
