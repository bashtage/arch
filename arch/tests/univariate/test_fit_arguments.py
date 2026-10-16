import datetime as dt

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.univariate import GARCH, ConstantMean, MAMean, arch_model
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


@pytest.fixture(scope="module", params=["daily", "monthly-period"])
def indexed(request):
    rs = np.random.RandomState(7)
    if request.param == "daily":
        index = pd.date_range("2000-01-01", periods=200, freq="D")
    else:
        index = pd.period_range("2000-01", periods=200, freq="M")
    return pd.Series(rs.standard_normal(200), index=index)


def cutoff_values(index, loc):
    """Equivalent ways to give location loc as a cutoff, with the expected type"""
    if isinstance(index, pd.PeriodIndex):
        period = index[loc]
        stamp = period.start_time
    else:
        stamp = index[loc]
        period = pd.Period(stamp, freq="D")
    return {
        "int": loc,
        "np-int": np.int64(loc),
        "timestamp": stamp,
        "datetime": stamp.to_pydatetime(),
        "datetime64": stamp.to_datetime64(),
        "date": stamp.date(),
        "period": period,
        "str": stamp.strftime("%Y-%m-%d"),
    }


CUTOFF_KINDS = [
    "int",
    "np-int",
    "timestamp",
    "datetime",
    "datetime64",
    "date",
    "period",
    "str",
]


@pytest.mark.parametrize("kind", CUTOFF_KINDS)
def test_fit_first_last_obs_types(indexed, kind):
    mod = ConstantMean(indexed)
    first, last = 20, 150
    cutoffs_first = cutoff_values(indexed.index, first)
    cutoffs_last = cutoff_values(indexed.index, last)
    res = mod.fit(
        disp="off", first_obs=cutoffs_first[kind], last_obs=cutoffs_last[kind]
    )
    assert res.fit_start == first
    assert res.fit_stop == last
    assert res.nobs == last - first
    assert_allclose(res.params["mu"], indexed.iloc[first:last].mean())


@pytest.mark.parametrize("kind", CUTOFF_KINDS)
def test_fit_garch_first_obs_types(indexed, kind):
    mod = ConstantMean(indexed, volatility=GARCH())
    first = 20
    res = mod.fit(disp="off", first_obs=cutoff_values(indexed.index, first)[kind])
    ref = mod.fit(disp="off", first_obs=first)
    assert res.fit_start == first
    assert_allclose(res.params, ref.params)


@pytest.mark.parametrize("kind", CUTOFF_KINDS)
def test_fix_first_last_obs_types(indexed, kind):
    mod = ConstantMean(indexed, volatility=GARCH())
    params = [0.0, 0.1, 0.1, 0.8]
    res = mod.fix(
        params,
        first_obs=cutoff_values(indexed.index, 30)[kind],
        last_obs=cutoff_values(indexed.index, 120)[kind],
    )
    # Residuals are only available in the sample, and last_obs is exclusive
    used = np.flatnonzero(res.resid.notna().to_numpy())
    assert (used[0], used[-1] + 1) == (30, 120)


@pytest.mark.parametrize("kind", CUTOFF_KINDS)
def test_forecast_start_types(indexed, kind):
    res = ConstantMean(indexed, volatility=GARCH()).fit(disp="off")
    ref = res.forecast(start=150, reindex=False)
    fcast = res.forecast(start=cutoff_values(indexed.index, 150)[kind], reindex=False)
    assert fcast.mean.shape[0] == indexed.shape[0] - 150
    assert_allclose(fcast.mean.to_numpy(), ref.mean.to_numpy())
    assert_allclose(fcast.variance.to_numpy(), ref.variance.to_numpy())


@pytest.mark.parametrize("kind", ["date", "period", "datetime"])
def test_ma_mean_cutoff_types(indexed, kind):
    mod = MAMean(indexed, lags=1)
    cutoffs = cutoff_values(indexed.index, 20)
    res = mod.fit(disp="off", first_obs=cutoffs[kind])
    assert res.fit_start == 20
    fcast = res.forecast(start=cutoff_values(indexed.index, 150)[kind], reindex=False)
    assert fcast.mean.shape[0] == indexed.shape[0] - 150


@pytest.mark.parametrize(
    "bad",
    [
        10.0,
        np.float64(10),
        np.float32(10),
        [10],
        (10,),
        {"a": 1},
        1 + 2j,
        b"2000-01-05",
        np.array([10]),
    ],
    ids=[
        "float",
        "np-float64",
        "np-float32",
        "list",
        "tuple",
        "dict",
        "complex",
        "bytes",
        "array",
    ],
)
def test_unsupported_cutoff_types(returns, bad):
    mod = ConstantMean(returns, volatility=GARCH())
    with pytest.raises(TypeError, match="cutoffs must be"):
        mod.fit(disp="off", first_obs=bad)
    with pytest.raises(TypeError, match="cutoffs must be"):
        mod.fit(disp="off", last_obs=bad)
    with pytest.raises(TypeError, match="cutoffs must be"):
        mod.fix([0.0, 0.1, 0.1, 0.8], first_obs=bad)
    res = mod.fit(disp="off")
    with pytest.raises(TypeError, match="cutoffs must be"):
        res.forecast(start=bad)
    with pytest.raises(TypeError, match="cutoffs must be"):
        MAMean(returns, lags=1).fit(disp="off", first_obs=bad)


@pytest.mark.parametrize(
    "value",
    ["2000-01-05", dt.date(2000, 1, 5), pd.Period("2000-01-05")],
    ids=["str", "date", "period"],
)
def test_date_cutoff_requires_datetime_index(value):
    y = np.random.RandomState(3).standard_normal(100)
    with pytest.raises(ValueError, match="DatetimeIndex or a PeriodIndex"):
        ConstantMean(y).fit(disp="off", first_obs=value)
    with pytest.raises(ValueError, match="DatetimeIndex or a PeriodIndex"):
        ConstantMean(pd.Series(y)).fit(disp="off", first_obs=value)
    res = ConstantMean(y).fit(disp="off")
    with pytest.raises(ValueError, match="DatetimeIndex or a PeriodIndex"):
        res.forecast(start=value)
