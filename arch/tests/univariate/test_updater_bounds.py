"""
Tests that the variance bounds are applied once, after all of the terms of the
variance have been added, by the updaters used with ARCH-in-mean
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import gammaln

import arch.univariate.recursions_python as recpy

try:
    import arch.univariate.recursions as rec_cython

    MISSING_EXTENSION = False
except ImportError:
    MISSING_EXTENSION = True


pytestmark = pytest.mark.filterwarnings("ignore::arch.compat.numba.PerformanceWarning")


def midas_weights(m, theta):
    j = np.arange(1.0, m + 1)
    w = np.exp(gammaln(theta + j) - gammaln(j + 1) - gammaln(theta))
    return w / w.sum()


@pytest.mark.parametrize("module", ["python", "cython"])
@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
def test_midas_updater_bounds(module, asym):
    if module == "cython" and MISSING_EXTENSION:
        pytest.skip("Cython extension not available")
    rec = recpy if module == "python" else rec_cython
    m, nobs = 12, 60
    resids = np.random.default_rng(10).standard_normal(nobs) * 0.1
    parameters = np.array([0.0, 0.5, 0.2, 0.6] if asym else [0.0, 0.5, 0.6])
    weights = midas_weights(m, 0.6)
    gamma = 0.2 if asym else 0.0
    backcast = 0.01
    weights_args = (np.array([parameters[0], parameters[1], gamma]), weights, resids)
    unbounded = np.empty(nobs)
    wide = np.empty((nobs, 2))
    wide[:, 0], wide[:, 1] = 1e-300, 1e300
    recpy.midas_recursion_python(*weights_args, unbounded, nobs, backcast, wide)
    # A lower bound that the partial sums of the terms are below, and that many of
    # the final sums are above
    lower = np.quantile(unbounded, 0.25)
    var_bounds = np.empty((nobs, 2))
    var_bounds[:, 0] = lower
    var_bounds[:, 1] = 10.0
    expected = np.empty(nobs)
    recpy.midas_recursion_python(*weights_args, expected, nobs, backcast, var_bounds)
    assert np.sum(unbounded < lower) > 5
    assert np.sum(unbounded > lower) > 5
    sigma2 = np.zeros(nobs)

    updater = rec.MIDASUpdater(m, asym)
    updater.initialize_update(parameters, backcast, nobs)
    for t in range(nobs):
        updater._update_tester(t, parameters, resids, sigma2, var_bounds)
    assert_allclose(sigma2, expected, rtol=1e-12)
