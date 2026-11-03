"""
Tests that the ADF simulation computes the statistics of ADF.

adf_statistics in critical_values/simulation/adf_simulation.py simulates the ADF
z-test and t-test statistics with no lagged differences, from which the critical
values and p-values are estimated.
"""

import importlib.util
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from arch.unitroot import ADF

TRENDS = ["n", "c", "ct", "ctt"]


@pytest.fixture(scope="module")
def adf_simulation():
    path = (
        Path(__file__).parents[2]
        / "unitroot"
        / "critical_values"
        / "simulation"
        / "adf_simulation.py"
    )
    if not path.exists():
        pytest.skip("The simulation scripts are not available")
    spec = importlib.util.spec_from_file_location("adf_simulation", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("trend", TRENDS)
@pytest.mark.parametrize("nobs", [20, 250])
def test_adf_statistics(adf_simulation, trend, nobs):
    num = 5
    z_stat, t_stat = adf_simulation.adf_statistics(
        nobs, trend, num, np.random.default_rng(nobs)
    )
    # The simulation uses the columns of the same draws as random walks
    y = np.cumsum(np.random.default_rng(nobs).standard_normal((nobs, num)), axis=0)
    for i in range(num):
        adf = ADF(y[:, i], lags=0, trend=trend)
        assert_allclose(t_stat[i], adf.stat, rtol=1e-8)
        # nobs is the number of observations in the regression once it is estimated
        gamma = np.asarray(adf.regression.params)[0]
        assert_allclose(z_stat[i], adf.nobs * gamma, rtol=1e-8)
    z_only = adf_simulation.adf_simulation(
        nobs, trend, num, np.random.default_rng(nobs)
    )
    assert_allclose(z_only, z_stat, rtol=0, atol=0)
