"""
Tests that the ADF t-test p-value is a smooth function of the test statistic.

With one unit root, the p-value is a cubic in stat below adf_t_star and another cubic
in stat above it. The two pieces are estimated jointly so that they have the same
value and the same slope at adf_t_star, and the p-value is monotone in the statistic.
With more than one unit root, the p-values are those of MacKinnon (1994).
"""

from typing import Literal

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import stats

from arch.unitroot.critical_values.dickey_fuller import (
    adf_t_star,
    tau_large_p,
    tau_max,
    tau_min,
    tau_small_p,
    tau_star,
)
from arch.unitroot.unitroot import mackinnonp

TRENDS = ["n", "c", "ct", "ctt"]

# Mean over the simulation files adf_t_<trend>-NNNN.npz (written by
# adf_z_critical_values_simulation_joblib.py in critical_values/simulation) of the 1,
# 5, 10, 50 and 90 percentiles of the ADF t-statistic with T=2000 observations,
# rounded to 4 decimals. These are the data that the p-value models are fit to, and
# are the reference for the accuracy of the fitted p-values.
PERCENTILES = np.array([0.01, 0.05, 0.10, 0.50, 0.90])
SIMULATED_QUANTILES = {
    "n": [-2.5671, -1.9410, -1.6164, -0.4997, 0.8881],
    "c": [-3.4332, -2.8631, -2.5676, -1.5660, -0.4394],
    "ct": [-3.9634, -3.4127, -3.1284, -2.1801, -1.2449],
    "ctt": [-4.3775, -3.8353, -3.5551, -2.6196, -1.7348],
}
# The largest error seen is 1.8e-5 for the 1, 5 and 10 percentiles and 5.7e-3 for the
# 50 and 90 percentiles (cubic polynomials cannot track the whole distribution)
ATOL = np.array([1e-4, 1e-4, 1e-4, 7.5e-3, 7.5e-3])


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
