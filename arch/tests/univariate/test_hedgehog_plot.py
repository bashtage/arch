from arch.compat.matplotlib import HAS_MATPLOTLIB

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.univariate import ARX, EGARCH, Normal, arch_model

if HAS_MATPLOTLIB:
    import matplotlib.dates as mdates

pytestmark = pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")

NOBS = 200
# One call to hedgehog_plot requests a forecast once, or, for the bootstrap,
# once per origin until there are enough observations before the origin (100).
# The guard turns a search for a start that never ends, which is what happened
# for models without analytic multi-step forecasts, into a test failure.
MAX_FORECAST_CALLS = 150


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.close("all")


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


def hedgehog_lines(fig):
    """Return the x and y data of the data line and of each spine"""
    lines = fig.axes[0].get_lines()
    return [line.get_xydata() for line in lines]


def expected_spines(res, y_values, plot_type, horizon, step, first_origin, **kwargs):
    """
    Compute the spines using only forecast(), one origin at a time, and the
    index of the data.
    """
    index = res.model._y_series.index
    spines = []
    for origin in range(first_origin, len(index), step):
        if origin + horizon + 1 > len(index):
            continue
        fcast = res.forecast(horizon=horizon, start=origin, **kwargs)
        if plot_type == "mean":
            values = fcast.mean.iloc[0].to_numpy()
        else:
            values = np.sqrt(fcast.variance.iloc[0].to_numpy())
        x = mdates.date2num(index[origin : origin + horizon + 1])
        y = np.hstack([y_values[origin], values])
        spines.append((x, y))
    return spines


MEAN_SPECS = [
    pytest.param({"mean": "AR", "lags": 1}, 0, id="ar1"),
    pytest.param({"mean": "AR", "lags": 3}, 2, id="ar3"),
    pytest.param({"mean": "HAR", "lags": [1, 5, 22]}, 21, id="har"),
]


@pytest.mark.parametrize("plot_type", ["mean", "volatility"])
@pytest.mark.parametrize("start", [None, 60], ids=["default-start", "start-60"])
@pytest.mark.parametrize(("spec", "first_origin"), MEAN_SPECS)
def test_hedgehog_data_analytic(
    data, spec, first_origin, start, plot_type, monkeypatch, agg_backend
):
    res = arch_model(data, vol="GARCH", **spec).fit(disp="off", show_warning=False)
    guard_forecast_calls(res, monkeypatch)
    horizon, step = 4, 37
    fig = res.hedgehog_plot(
        horizon=horizon, step=step, start=start, plot_type=plot_type
    )
    if plot_type == "mean":
        y_values = np.asarray(data)
    else:
        y_values = np.asarray(res.conditional_volatility)
    origin = first_origin if start is None else start
    expected = expected_spines(res, y_values, plot_type, horizon, step, origin)
    lines = hedgehog_lines(fig)
    # The data and then the spines
    assert len(lines) == 1 + len(expected)
    assert_allclose(lines[0][:, 1], y_values)
    for line, (x, y) in zip(lines[1:], expected, strict=True):
        assert_allclose(line[:, 0], x)
        assert_allclose(line[:, 1], y, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("start", [None, 50], ids=["default-start", "start-50"])
def test_hedgehog_data_egarch_simulation(data, start, monkeypatch, agg_backend):
    # Same random numbers in hedgehog_plot and in the direct call to forecast
    rs = np.random.RandomState(98765)
    mod = ARX(
        data,
        lags=2,
        volatility=EGARCH(1, 1, 1),
        distribution=Normal(seed=rs),
    )
    res = mod.fit(disp="off", show_warning=False)
    simulations, horizon, step = 25, 3, 41
    # fit works on a copy of the model, so reset the generator the result uses
    generator = res.model.distribution.generator
    state = generator.get_state()
    guard_forecast_calls(res, monkeypatch)
    fig = res.hedgehog_plot(
        horizon=horizon,
        step=step,
        start=start,
        method="simulation",
        simulations=simulations,
    )
    generator.set_state(state)
    first_origin = 1 if start is None else start
    fcast = res.model.forecast(
        res.params,
        horizon=horizon,
        start=first_origin,
        method="simulation",
        simulations=simulations,
    )
    index = data.index
    y_values = np.asarray(res.conditional_volatility)
    lines = hedgehog_lines(fig)
    origins = [
        o
        for o in range(first_origin, len(index), step)
        if o + horizon + 1 <= len(index)
    ]
    assert len(lines) == 1 + len(origins)
    for line, origin in zip(lines[1:], origins, strict=True):
        row = fcast.variance.iloc[origin - first_origin].to_numpy()
        expected = np.hstack([y_values[origin], np.sqrt(row)])
        assert_allclose(line[:, 1], expected, rtol=1e-10)
        assert_allclose(
            line[:, 0], mdates.date2num(index[origin : origin + horizon + 1])
        )


@pytest.fixture(scope="module", params=["M", "D"], ids=["monthly", "daily"])
def period_data(request):
    rs = np.random.RandomState(4321)
    index = pd.period_range("2000-01", periods=NOBS, freq=request.param)
    return pd.Series(rs.standard_normal(NOBS), index=index)


def test_plot_period_index(period_data, agg_backend):
    res = arch_model(period_data).fit(disp="off", show_warning=False)
    fig = res.plot()
    expected_x = mdates.date2num(period_data.index.to_timestamp())
    lines = [ax.get_lines()[0].get_xydata() for ax in fig.axes]
    assert len(lines) == 2
    for line in lines:
        assert_allclose(line[:, 0], expected_x)
    assert_allclose(lines[0][:, 1], res.resid / res.conditional_volatility)
    assert_allclose(lines[1][:, 1], res.conditional_volatility)


@pytest.mark.parametrize("plot_type", ["mean", "volatility"])
def test_hedgehog_period_index(period_data, plot_type, agg_backend):
    res = arch_model(period_data, mean="AR", lags=2).fit(disp="off", show_warning=False)
    horizon, step = 3, 50
    fig = res.hedgehog_plot(horizon=horizon, step=step, plot_type=plot_type)
    lines = hedgehog_lines(fig)
    timestamps = period_data.index.to_timestamp()
    assert_allclose(lines[0][:, 0], mdates.date2num(timestamps))
    # AR(2): the first origin is 1
    origins = list(range(1, NOBS - horizon, step))
    assert len(lines) == 1 + len(origins)
    for line, origin in zip(lines[1:], origins, strict=True):
        assert_allclose(
            line[:, 0], mdates.date2num(timestamps[origin : origin + horizon + 1])
        )
        fcast = res.forecast(horizon=horizon, start=origin)
        if plot_type == "mean":
            values = fcast.mean.iloc[0].to_numpy()
            first = period_data.iloc[origin]
        else:
            values = np.sqrt(fcast.variance.iloc[0].to_numpy())
            first = res.conditional_volatility.iloc[origin]
        assert_allclose(line[:, 1], np.hstack([first, values]), rtol=1e-10)
