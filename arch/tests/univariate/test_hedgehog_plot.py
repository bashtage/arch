from arch.compat.matplotlib import HAS_MATPLOTLIB

import numpy as np
import pandas as pd
import pytest

from arch.univariate import arch_model

pytestmark = pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")

NOBS = 200
# One call to hedgehog_plot requests a forecast once, or, for the bootstrap,
# once per origin until there are enough observations before the origin (100).
# The guard turns a search for a start that never ends, which is what happened
# for models without analytic multi-step forecasts, into a test failure.
MAX_FORECAST_CALLS = 150


@pytest.fixture(scope="module")
def data():
    rs = np.random.RandomState(1234)
    y = 0.1 + rs.standard_normal(NOBS)
    return pd.Series(y, index=pd.date_range("2000-01-01", periods=NOBS, freq="D"))


def guard_forecast_calls(res, monkeypatch):
    """Fail instead of looping when hedgehog_plot keeps calling forecast"""
    calls = []
    forecast = res.forecast

    def counting_forecast(*args, **kwargs):
        calls.append(args)
        if len(calls) > MAX_FORECAST_CALLS:
            pytest.fail("hedgehog_plot called forecast too many times")
        return forecast(*args, **kwargs)

    monkeypatch.setattr(res, "forecast", counting_forecast)
    return calls


NO_ANALYTIC_MULTISTEP = [
    pytest.param({"vol": "EGARCH", "o": 1}, id="egarch"),
    pytest.param({"vol": "APARCH", "o": 1}, id="aparch"),
    pytest.param({"vol": "FIAPARCH", "p": 1, "q": 1}, id="fiaparch"),
    pytest.param({"vol": "GARCH", "power": 1.0}, id="garch-power-1"),
    pytest.param({"vol": "GARCH", "o": 1, "power": 1.0}, id="tarch-power-1"),
]


@pytest.mark.parametrize("spec", NO_ANALYTIC_MULTISTEP)
@pytest.mark.parametrize("start", [None, 100], ids=["default-start", "start-100"])
def test_hedgehog_no_analytic_multistep(data, spec, start, monkeypatch, agg_backend):
    res = arch_model(data, **spec).fit(disp="off", show_warning=False)
    calls = guard_forecast_calls(res, monkeypatch)
    with pytest.raises(ValueError, match="Analytic forecasts not available"):
        res.hedgehog_plot(start=start)
    assert len(calls) == 1


@pytest.mark.parametrize("spec", NO_ANALYTIC_MULTISTEP)
@pytest.mark.parametrize("method", ["simulation", "bootstrap"])
def test_hedgehog_simulation_methods(data, spec, method, monkeypatch, agg_backend):
    res = arch_model(data, **spec).fit(disp="off", show_warning=False)
    calls = guard_forecast_calls(res, monkeypatch)
    fig = res.hedgehog_plot(horizon=3, step=50, method=method, simulations=20)
    if method == "simulation":
        assert len(calls) == 1
    # the data and at least one spine
    assert len(fig.axes[0].get_lines()) > 1


@pytest.mark.parametrize("spec", NO_ANALYTIC_MULTISTEP)
def test_hedgehog_horizon_1_analytic(data, spec, monkeypatch, agg_backend):
    # A one-step analytic forecast exists for every process
    res = arch_model(data, **spec).fit(disp="off", show_warning=False)
    calls = guard_forecast_calls(res, monkeypatch)
    fig = res.hedgehog_plot(horizon=1, step=50)
    assert len(calls) == 1
    assert len(fig.axes[0].get_lines()) > 1


def test_hedgehog_figarch_analytic(data, monkeypatch, agg_backend):
    # FIGARCH has analytic multi-step forecasts, so the defaults work
    res = arch_model(data, vol="FIGARCH").fit(disp="off", show_warning=False)
    guard_forecast_calls(res, monkeypatch)
    fig = res.hedgehog_plot(step=50)
    assert len(fig.axes[0].get_lines()) > 1
