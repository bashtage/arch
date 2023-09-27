import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.univariate import GARCH, Normal, ZeroMean


@pytest.fixture(scope="module")
def small_data():
    rs = np.random.RandomState([2389280, 238901, 382908031])
    mod = ZeroMean(None, volatility=GARCH(), distribution=Normal(seed=rs))
    sim = mod.simulate([1e-3, 0.05, 0.90], initial_value_vol=1e-3 / 0.05, nobs=1000)
    return sim.data


@pytest.fixture(scope="module")
def small_data2():
    rs = np.random.RandomState([2389280, 238901, 382908031])
    mod = ZeroMean(None, volatility=GARCH(), distribution=Normal(seed=rs))
    sim = mod.simulate([1e-3, 0.05, 0.90], nobs=1000)
    return sim.data


@pytest.fixture(scope="module")
def std_data():
    rs = np.random.RandomState([2389280, 238901, 382908031])
    mod = ZeroMean(None, volatility=GARCH(), distribution=Normal(seed=rs))
    sim = mod.simulate([1.0, 0.05, 0.90], nobs=1000)
    return sim.data


def test_reproducibility(small_data, small_data2):
    pd.testing.assert_series_equal(small_data, small_data2)


def test_blank(small_data, std_data):
    small_mod = ZeroMean(small_data, volatility=GARCH(), rescale=False)
    small_res = small_mod.fit(starting_values=np.array([1e-3, 0.05, 0.90]), disp="off")
    mod = ZeroMean(std_data, volatility=GARCH(), rescale=False)
    res = mod.fit(starting_values=np.array([1, 0.05, 0.90]), disp="off")

    small_param0, *_ = small_res.params
    param0, *_ = res.params
    assert_allclose(1e3 * small_param0, param0, rtol=5e-3, atol=1e9)


def test_rescale_fit(small_data, std_data):
    small_mod = ZeroMean(small_data, volatility=GARCH(), rescale=True)
    small_res = small_mod.fit(disp="off")
    direct_mod = ZeroMean(10 * small_data, volatility=GARCH())
    direct_res = direct_mod.fit(disp="off")
    assert_allclose(small_res.loglikelihood, direct_res.loglikelihood)
    small_fcast = small_res.forecast(start=0)
    direct_fcast = direct_res.forecast(start=0)
    assert_allclose(small_fcast.variance, direct_fcast.variance)
