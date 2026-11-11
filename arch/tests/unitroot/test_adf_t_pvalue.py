"""
Tests of the ADF t-test p-value and critical values.

With one unit root, the p-value is a cubic in stat below adf_t_star and another cubic
in stat above it. The two pieces are estimated jointly so that they have the same
value and the same slope at adf_t_star, and the p-value is monotone in the statistic.
The critical values are fit to the same simulations, and the p-value is fit to the
asymptotic quantiles that are the constants of the critical value regressions, so
the p-values of the asymptotic critical values are 1%, 5% and 10%. With more than
one unit root, the p-values are those of MacKinnon (1994) and the critical values
are those of MacKinnon (2010).
"""

from typing import Literal

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import stats

from arch.unitroot.critical_values.dickey_fuller import (
    adf_t_cv_approx,
    adf_t_star,
    tau_2010,
    tau_large_p,
    tau_max,
    tau_min,
    tau_small_p,
    tau_star,
)
from arch.unitroot.unitroot import mackinnoncrit, mackinnonp

TRENDS = ["n", "c", "ct", "ctt"]

# The asymptotic 1, 5, 10, 50 and 90 percentiles of the ADF t-statistic, rounded to
# 4 decimals. These are the constants of the regressions of the percentiles in the
# simulation files adf_t_<trend>-NNNN.npz (written by
# adf_z_critical_values_simulation_joblib.py in critical_values/simulation) on powers
# of 1/nobs, which adf_t_simulation_process.py fits the p-value models to. They are
# the reference for the accuracy of the fitted p-values.
PERCENTILES = np.array([0.01, 0.05, 0.10, 0.50, 0.90])
SIMULATED_QUANTILES = {
    "n": [-2.5658, -1.9409, -1.6167, -0.5001, 0.8875],
    "c": [-3.4303, -2.8615, -2.5668, -1.5664, -0.4403],
    "ct": [-3.9585, -3.4104, -3.1270, -2.1806, -1.2464],
    "ctt": [-4.3712, -3.8323, -3.5532, -2.6201, -1.7365],
}
# The largest error seen is 3.3e-5 for the 1, 5 and 10 percentiles and 6.2e-3 for the
# 50 and 90 percentiles (cubic polynomials cannot track the whole distribution)
ATOL = np.array([1e-4, 1e-4, 1e-4, 7.5e-3, 7.5e-3])
# Mean over the simulation files of the 1, 5 and 10 percentiles with nobs = 24 and
# nobs = 99 observations in the regression (series of 25 and 100 observations),
# rounded to 4 decimals. Their standard errors are at most 0.00022.
SIMULATED_CRITICAL_VALUES = {
    "n": {24: [-2.6649, -1.9558, -1.6086], 99: [-2.5887, -1.9440, -1.6144]},
    "c": {24: [-3.7379, -2.9922, -2.6356], 99: [-3.4983, -2.8913, -2.5826]},
    "ct": {24: [-4.3940, -3.6122, -3.2431], 99: [-4.0535, -3.4558, -3.1536]},
    "ctt": {24: [-4.9399, -4.1087, -3.7194], 99: [-4.4923, -3.8935, -3.5909]},
}


def adf_t_pvalue(
    stat: float, trend: Literal["n", "c", "ct", "ctt"], num_unit_roots: int = 1
) -> float:
    return mackinnonp(
        stat, regression=trend, num_unit_roots=num_unit_roots, dist_type="adf-t"
    )


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_continuous_at_star(trend):
    star = adf_t_star[trend]
    # stat <= star uses the small p model and the next float is in the large p model
    p_small = adf_t_pvalue(star, trend)
    p_large = adf_t_pvalue(float(np.nextafter(star, np.inf)), trend)
    assert_allclose(p_small, p_large, rtol=0, atol=1e-7)
    for step in (1e-3, 1e-6):
        p_small = adf_t_pvalue(star - step, trend)
        p_large = adf_t_pvalue(star + step, trend)
        # p changes by 2 * step times the slope, which is at most 0.55 at adf_t_star
        assert_allclose(p_small, p_large, rtol=0, atol=1.2 * step)


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_derivative_continuous_at_star(trend):
    star = adf_t_star[trend]
    step = 1e-6
    # One-sided finite differences, using the small p model on the left and the
    # large p model on the right
    left = (adf_t_pvalue(star, trend) - adf_t_pvalue(star - step, trend)) / step
    right = (
        adf_t_pvalue(star + step, trend)
        - adf_t_pvalue(float(np.nextafter(star, np.inf)), trend)
    ) / step
    assert left > 0
    assert_allclose(left, right, rtol=1e-4)


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_monotone_and_bounded(trend):
    star = adf_t_star[trend]
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(-60.0, 10.0, 8001),
                star + np.linspace(-0.5, 0.5, 4001),
                -np.logspace(-6, 12, 50)[::-1],
                np.logspace(-6, 12, 50),
            ]
        )
    )
    pvals = np.array([adf_t_pvalue(float(stat), trend) for stat in grid])
    assert np.all(np.isfinite(pvals))
    assert np.all(pvals >= 0.0)
    assert np.all(pvals <= 1.0)
    # Allow for rounding when two points of the grid are within 1e-15 of each other
    assert np.all(np.diff(pvals) >= -1e-14)
    # The p-value is not constant in the part of the distribution that matters
    middle = [
        adf_t_pvalue(float(stat), trend) for stat in np.linspace(star - 4, 2, 2001)
    ]
    assert np.all(np.diff(middle) > 0.0)
    assert adf_t_pvalue(-np.inf, trend) == 0.0
    assert adf_t_pvalue(np.inf, trend) == 1.0
    assert pvals[0] < 1e-12
    assert pvals[-1] > 1 - 1e-12


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_matches_simulated_distribution(trend):
    quantiles = SIMULATED_QUANTILES[trend]
    pvals = np.array([adf_t_pvalue(stat, trend) for stat in quantiles])
    # assert_allclose does not take an array for atol
    assert np.all(np.abs(pvals - PERCENTILES) <= ATOL)


@pytest.mark.parametrize("trend", TRENDS)
@pytest.mark.parametrize("num_unit_roots", [2, 3, 6])
def test_more_unit_roots_use_mackinnon_1994(trend, num_unit_roots):
    idx = num_unit_roots - 1
    star = tau_star[trend][idx]
    for point in (star - 1.0, star, star + 0.5):
        stat = float(np.clip(point, tau_min[trend][idx], tau_max[trend][idx]))
        coef = tau_small_p[trend][idx] if stat <= star else tau_large_p[trend][idx]
        expected = stats.norm.cdf(np.polyval(np.asarray(coef)[::-1], stat))
        assert_allclose(adf_t_pvalue(stat, trend, num_unit_roots), expected)


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_at_asymptotic_critical_values(trend):
    # The p-value is fit to the constants of the critical value regressions
    critical_values = mackinnoncrit(1, trend, np.inf, "adf-t")
    pvals = [adf_t_pvalue(float(cv), trend) for cv in critical_values]
    assert_allclose(pvals, [0.01, 0.05, 0.10], rtol=1e-3)


@pytest.mark.parametrize("trend", TRENDS)
@pytest.mark.parametrize("nobs", [24, 99, np.inf])
def test_critical_values_match_simulated_distribution(trend, nobs):
    coef = np.array(adf_t_cv_approx[trend])
    critical_values = mackinnoncrit(1, trend, nobs, "adf-t")
    if nobs == np.inf:
        assert_allclose(critical_values, coef[:, 0])
        expected = SIMULATED_QUANTILES[trend][:3]
    else:
        assert_allclose(critical_values, np.polyval(coef[:, ::-1].T, 1.0 / nobs))
        expected = SIMULATED_CRITICAL_VALUES[trend][nobs]
    # The largest error of the critical value regressions is 0.0003, which is 2.5
    # standard errors of the simulated percentiles
    assert_allclose(critical_values, expected, rtol=0, atol=6e-4)


@pytest.mark.parametrize("trend", ["c", "ct", "ctt"])
@pytest.mark.parametrize("num_unit_roots", [2, 3, 12])
@pytest.mark.parametrize("nobs", [50, np.inf])
def test_more_unit_roots_critical_values_use_mackinnon_2010(
    trend, num_unit_roots, nobs
):
    coef = tau_2010[trend][num_unit_roots - 1]
    if nobs == np.inf:
        expected = coef[:, 0]
    else:
        expected = np.polyval(coef[:, ::-1].T, 1.0 / nobs)
    assert_allclose(mackinnoncrit(num_unit_roots, trend, nobs, "adf-t"), expected)
