"""
Tests that the Engle-Granger p-value is a smooth function of the test statistic.

The p-value is a cubic in stat below TAU_STAR and another cubic in stat above it.
The two pieces are estimated jointly so that they have the same value and the same
slope at TAU_STAR, and the p-value is monotone in the statistic.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.unitroot._engle_granger import engle_granger_pval
from arch.unitroot.critical_values.engle_granger import TAU_STAR

KEYS = list(TAU_STAR)

# Mean over the 500 simulation files engle-granger-results-trend-<trend>-NNNN.npz
# (written by engle_granger_simulation.py in critical_values/simulation) of the 1, 5,
# 10, 50 and 90 percentiles of the statistic with T=2000 observations, rounded to 4
# decimals. These are the data that the p-value models are fit to, and are the
# reference for the accuracy of the fitted p-values.
PERCENTILES = np.array([0.01, 0.05, 0.10, 0.50, 0.90])
SIMULATED_QUANTILES = {
    ("n", 1): [-3.3479, -2.7640, -2.4606, -1.4320, -0.2745],
    ("n", 6): [-5.2544, -4.7109, -4.4272, -3.4588, -2.5290],
    ("n", 11): [-6.4565, -5.9232, -5.6439, -4.6874, -3.7710],
    ("c", 1): [-3.9027, -3.3399, -3.0473, -2.0538, -1.0087],
    ("c", 6): [-5.5285, -4.9880, -4.7053, -3.7400, -2.8106],
    ("c", 11): [-6.6643, -6.1324, -5.8537, -4.8982, -3.9814],
    ("ct", 1): [-4.3365, -3.7862, -3.5006, -2.5374, -1.6030],
    ("ct", 6): [-5.7836, -5.2460, -4.9648, -4.0032, -3.0798],
    ("ct", 11): [-6.8639, -6.3329, -6.0544, -5.1003, -4.1849],
    ("ctt", 1): [-4.7041, -4.1615, -3.8798, -2.9298, -2.0296],
    ("ctt", 6): [-6.0270, -5.4916, -5.2116, -4.2543, -3.3372],
    ("ctt", 11): [-7.0575, -6.5272, -6.2494, -5.2967, -4.3827],
}
# The largest error seen is 2.7e-5 for the 1, 5 and 10 percentiles and 2.0e-3 for the
# 50 and 90 percentiles
ATOL = np.array([1e-4, 1e-4, 1e-4, 4e-3, 4e-3])


@pytest.mark.parametrize("key", KEYS)
def test_pvalue_continuous_at_star(key):
    trend, num_x = key
    star = TAU_STAR[key]
    # stat <= star uses the small p model and the next float is in the large p model
    p_small = engle_granger_pval(star, trend, num_x)
    p_large = engle_granger_pval(float(np.nextafter(star, 0.0)), trend, num_x)
    assert_allclose(p_small, p_large, rtol=0, atol=1e-7)
    for step in (1e-3, 1e-6):
        p_small = engle_granger_pval(star - step, trend, num_x)
        p_large = engle_granger_pval(star + step, trend, num_x)
        # p changes by at most 2 * step times a slope no larger than 0.75
        assert_allclose(p_small, p_large, rtol=0, atol=1.5 * step)


@pytest.mark.parametrize("key", KEYS)
def test_pvalue_derivative_continuous_at_star(key):
    trend, num_x = key
    star = TAU_STAR[key]
    step = 1e-6
    # One-sided finite differences, using the small p model on the left and the
    # large p model on the right
    left = (
        engle_granger_pval(star, trend, num_x)
        - engle_granger_pval(star - step, trend, num_x)
    ) / step
    right = (
        engle_granger_pval(star + step, trend, num_x)
        - engle_granger_pval(float(np.nextafter(star, 0.0)), trend, num_x)
    ) / step
    assert left > 0
    assert_allclose(left, right, rtol=1e-4)


@pytest.mark.parametrize("key", KEYS)
def test_pvalue_monotone_and_bounded(key):
    trend, num_x = key
    star = TAU_STAR[key]
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(-60.0, 10.0, 701),
                star + np.linspace(-0.5, 0.5, 201),
                -np.logspace(-6, 12, 30)[::-1],
                np.logspace(-6, 12, 30),
            ]
        )
    )
    pvals = np.array([engle_granger_pval(float(stat), trend, num_x) for stat in grid])
    assert np.all(np.isfinite(pvals))
    assert np.all(pvals >= 0.0)
    assert np.all(pvals <= 1.0)
    # Allow for rounding when two points of the grid are within 1e-15 of each other
    assert np.all(np.diff(pvals) >= -1e-14)
    # The p-value is not constant in the part of the distribution that matters
    middle = [
        engle_granger_pval(float(stat), trend, num_x)
        for stat in np.linspace(star - 3, 0, 301)
    ]
    assert np.all(np.diff(middle) > 0.0)
    assert engle_granger_pval(-np.inf, trend, num_x) == 0.0
    assert engle_granger_pval(np.inf, trend, num_x) == 1.0
    assert pvals[0] < 1e-12
    assert pvals[-1] > 1 - 1e-12


@pytest.mark.parametrize("key", list(SIMULATED_QUANTILES))
def test_pvalue_matches_simulated_distribution(key):
    trend, num_x = key
    quantiles = SIMULATED_QUANTILES[key]
    pvals = np.array([engle_granger_pval(stat, trend, num_x) for stat in quantiles])
    # assert_allclose does not take an array for atol
    assert np.all(np.abs(pvals - PERCENTILES) <= ATOL)
