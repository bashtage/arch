"""
Tests of the ADF-Z p-value and critical values.

The p-value is a quartic in log|stat| below adf_z_star and a cubic in stat above it.
The two pieces are estimated jointly so that they have the same value and the same
slope at adf_z_star, and the p-value is monotone in the statistic. The critical
values are fit to the same simulations, and the p-value is fit to the asymptotic
quantiles that are the constants of the critical value regressions, so the p-values
of the asymptotic critical values are 1%, 5% and 10%.
"""

import importlib.util
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats

from arch.unitroot.critical_values.dickey_fuller import adf_z_cv_approx, adf_z_star
from arch.unitroot.unitroot import mackinnoncrit, mackinnonp

TRENDS = ["n", "c", "ct", "ctt"]

# The asymptotic 1, 5, 10, 50 and 90 percentiles of the ADF-Z statistic, rounded to 4
# decimals. These are the constants of the regressions of the percentiles in the 2,000
# simulation files adf_z_<trend>-NNNN.npz (written by
# adf_z_critical_values_simulation_joblib.py in critical_values/simulation) on powers
# of 1/nobs, which adf_z_simlation_process.py fits the p-value models to. They are the
# reference for the accuracy of the fitted p-values.
PERCENTILES = np.array([0.01, 0.05, 0.10, 0.50, 0.90])
SIMULATED_QUANTILES = {
    "n": [-13.6946, -8.0394, -5.7137, -0.8528, 0.9280],
    "c": [-20.6250, -14.0929, -11.2503, -4.3570, -0.8455],
    "ct": [-29.3556, -21.7105, -18.2449, -9.1027, -3.7673],
    "ctt": [-36.5964, -28.1110, -24.1853, -13.4049, -6.6436],
}
# The largest error seen is 4.1e-5 for the 1, 5 and 10 percentiles and 6.7e-3 for the
# 50 and 90 percentiles (the polynomials cannot track the whole distribution)
ATOL = np.array([1e-4, 1e-4, 1e-4, 7.5e-3, 7.5e-3])
# Mean over the simulation files of the 1, 5 and 10 percentiles with nobs = 24 and
# nobs = 99 observations in the regression (series of 25 and 100 observations),
# rounded to 4 decimals. Their standard errors are at most 0.0019.
SIMULATED_CRITICAL_VALUES = {
    "n": {24: [-11.4410, -7.0767, -5.1360], 99: [-13.0796, -7.7830, -5.5619]},
    "c": {24: [-16.4941, -11.9877, -9.8359], 99: [-19.4769, -13.5272, -10.8754]},
    "ct": {24: [-21.7334, -17.2364, -14.9653], 99: [-27.1485, -20.4596, -17.3423]},
    "ctt": {24: [-25.4915, -21.1170, -18.8359], 99: [-33.2677, -26.0924, -22.6684]},
}
# The largest error of the critical value regressions is 0.0075 with "ctt" at 1% and
# nobs = 24 (7 standard errors of the simulated percentile), and 0.0014 (2.5 standard
# errors) otherwise. The asymptotic critical values are the reference quantiles.
CV_ATOL = {24: 1e-2, 99: 2.5e-3, np.inf: 1e-4}

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


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_at_asymptotic_critical_values(trend):
    # The p-value is fit to the constants of the critical value regressions
    critical_values = mackinnoncrit(1, trend, np.inf, "adf-z")
    pvals = [adf_z_pvalue(float(cv), trend) for cv in critical_values]
    assert_allclose(pvals, [0.01, 0.05, 0.10], rtol=1e-3)


@pytest.mark.parametrize("trend", TRENDS)
@pytest.mark.parametrize("nobs", [24, 99, np.inf])
def test_critical_values_match_simulated_distribution(trend, nobs):
    coef = np.array(adf_z_cv_approx[trend])
    critical_values = mackinnoncrit(1, trend, nobs, "adf-z")
    if nobs == np.inf:
        assert_allclose(critical_values, coef[:, 0])
        expected = SIMULATED_QUANTILES[trend][:3]
    else:
        assert_allclose(critical_values, np.polyval(coef[:, ::-1].T, 1.0 / nobs))
        expected = SIMULATED_CRITICAL_VALUES[trend][nobs]
    assert_allclose(critical_values, expected, rtol=0, atol=CV_ATOL[nobs])


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
