"""
Tests that the DFGLS p-value is a smooth function of the test statistic.

The p-value is a quadratic in stat below dfgls_tau_star and a cubic in stat above it.
The two pieces are estimated jointly so that they have the same value and the same
slope at dfgls_tau_star, and the p-value is monotone in the statistic.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.unitroot.critical_values.dfgls import dfgls_tau_star
from arch.unitroot.unitroot import mackinnonp

TRENDS = ["c", "ct"]

# Mean over the 2,000 simulation files dfgls_<trend>_<i>.npy (written by
# dfgls_critical_values_simulation.py in critical_values/simulation) of the 1, 5, 10,
# 50 and 90 percentiles of the DFGLS statistic with T=2000 observations, rounded to 4
# decimals. These are the data that the p-value models are fit to, and are the
# reference for the accuracy of the fitted p-values.
PERCENTILES = np.array([0.01, 0.05, 0.10, 0.50, 0.90])
SIMULATED_QUANTILES = {
    "c": [-2.5767, -1.9525, -1.6294, -0.5198, 0.8639],
    "ct": [-3.4171, -2.8558, -2.5676, -1.6297, -0.8062],
}
# The largest error seen is 9.4e-5 for the 1, 5 and 10 percentiles and 6.5e-3 for the
# 50 and 90 percentiles
ATOL = np.array([1.5e-4, 1.5e-4, 1.5e-4, 1e-2, 1e-2])


def dfgls_pvalue(stat: float, trend: str) -> float:
    return mackinnonp(stat, regression=trend, num_unit_roots=1, dist_type="dfgls")


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_continuous_at_star(trend):
    star = dfgls_tau_star[trend]
    # stat <= star uses the small p model and the next float is in the large p model
    p_small = dfgls_pvalue(star, trend)
    p_large = dfgls_pvalue(float(np.nextafter(star, 0.0)), trend)
    assert_allclose(p_small, p_large, rtol=0, atol=1e-7)
    for step in (1e-3, 1e-6):
        p_small = dfgls_pvalue(star - step, trend)
        p_large = dfgls_pvalue(star + step, trend)
        # p changes by at most 2 * step times a slope no larger than 0.45
        assert_allclose(p_small, p_large, rtol=0, atol=step)


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_derivative_continuous_at_star(trend):
    star = dfgls_tau_star[trend]
    step = 1e-6
    # One-sided finite differences, using the small p model on the left and the
    # large p model on the right
    left = (dfgls_pvalue(star, trend) - dfgls_pvalue(star - step, trend)) / step
    right = (
        dfgls_pvalue(star + step, trend)
        - dfgls_pvalue(float(np.nextafter(star, 0.0)), trend)
    ) / step
    assert left > 0
    assert_allclose(left, right, rtol=1e-4)


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_monotone_and_bounded(trend):
    star = dfgls_tau_star[trend]
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(-60.0, 15.0, 7501),
                star + np.linspace(-0.5, 0.5, 4001),
                -np.logspace(-6, 12, 50)[::-1],
                np.logspace(-6, 12, 50),
            ]
        )
    )
    pvals = np.array([dfgls_pvalue(float(stat), trend) for stat in grid])
    assert np.all(np.isfinite(pvals))
    assert np.all(pvals >= 0.0)
    assert np.all(pvals <= 1.0)
    # Allow for rounding when two points of the grid are within 1e-15 of each other
    assert np.all(np.diff(pvals) >= -1e-14)
    # The p-value is not constant in the part of the distribution that matters
    middle = [
        dfgls_pvalue(float(stat), trend) for stat in np.linspace(star - 3, 1, 2001)
    ]
    assert np.all(np.diff(middle) > 0.0)
    assert dfgls_pvalue(-np.inf, trend) == 0.0
    assert dfgls_pvalue(np.inf, trend) == 1.0
    assert pvals[0] < 1e-12
    assert pvals[-1] > 1 - 1e-12


@pytest.mark.parametrize("trend", TRENDS)
def test_pvalue_matches_simulated_distribution(trend):
    quantiles = SIMULATED_QUANTILES[trend]
    pvals = np.array([dfgls_pvalue(stat, trend) for stat in quantiles])
    # assert_allclose does not take an array for atol
    assert np.all(np.abs(pvals - PERCENTILES) <= ATOL)
