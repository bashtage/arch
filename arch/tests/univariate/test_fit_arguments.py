import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.univariate import arch_model
from arch.utility.exceptions import StartingValueWarning


@pytest.fixture(scope="module")
def returns():
    rs = np.random.RandomState(20241016)
    return pd.Series(
        rs.standard_t(5, 600), index=pd.date_range("2000-01-01", periods=600)
    )


@pytest.fixture(scope="module")
def default_start_params(returns):
    # With maxiter=0 the optimizer returns the starting values it was given
    res = arch_model(returns).fit(
        disp="off", options={"maxiter": 0}, show_warning=False
    )
    return res.params


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param([0.0, -5.0, 0.5, 0.5], id="outside-bounds"),
        pytest.param([0.0, 0.1, 0.7, 0.6], id="violates-constraint"),
        pytest.param([0.0, 0.1, 0.1], id="too-short"),
        pytest.param([0.0, 0.1, 0.1, 0.1, 0.1, 0.1], id="too-long"),
        pytest.param([0.0, np.nan, 0.1, 0.1], id="nan"),
    ],
)
def test_invalid_starting_values_ignored(returns, default_start_params, bad):
    mod = arch_model(returns)
    with pytest.warns(StartingValueWarning, match="will be ignored"):
        res = mod.fit(
            disp="off", starting_values=bad, options={"maxiter": 0}, show_warning=False
        )
    assert_allclose(res.params, default_start_params)


def test_valid_starting_values_used(returns):
    mod = arch_model(returns)
    sv = np.array([0.05, 0.02, 0.1, 0.8])
    res = mod.fit(
        disp="off", starting_values=sv, options={"maxiter": 0}, show_warning=False
    )
    assert_allclose(res.params, sv)


def test_invalid_starting_values_fit_matches_default_fit(returns):
    mod = arch_model(returns)
    ref = mod.fit(disp="off")
    with pytest.warns(StartingValueWarning):
        res = mod.fit(disp="off", starting_values=[0.0, -5.0, 0.5, 0.5])
    assert_allclose(res.params, ref.params, rtol=1e-6)
    assert_allclose(res.loglikelihood, ref.loglikelihood, rtol=1e-8)
