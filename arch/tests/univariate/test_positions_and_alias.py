"""
Negative positions, 0 and 1 as flags and the ``horizon`` alias of ``align``.

A negative position is compared with the equivalent positive position, which is the
code path that was always there, and with the same selection made by slicing the data
in Python.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import arch
from arch.univariate import (
    ARCH,
    ARX,
    GARCH,
    ConstantMean,
    Normal,
    arch_model,
)
from arch.univariate.base import (
    ARCHModel,
    ARCHModelForecast,
    ARCHModelResult,
    _align_forecast,
)
from arch.utility.array import cutoff_to_index, find_index

NOBS = 300
DATES = pd.date_range("2000-01-03", periods=NOBS, freq="B")
PARAMS = np.array([0.05, 0.05, 0.1, 0.8])
Y = pd.Series(
    np.asarray(
        ConstantMean(
            volatility=GARCH(1, 0, 1),
            distribution=Normal(seed=np.random.RandomState(7)),
        ).simulate(PARAMS, NOBS)["data"]
    ),
    index=DATES,
    name="y",
)
OPTIONS = {"options": {"maxiter": 5}}
QUIET = {"disp": "off", "show_warning": False}


@pytest.fixture(scope="module")
def model():
    return arch_model(Y, mean="constant", vol="GARCH", p=1, q=1)


@pytest.fixture(scope="module")
def result(model):
    return model.fit(**QUIET)


# first_obs and last_obs are slice bounds


@pytest.mark.parametrize("first_obs", [-100, -NOBS, np.int64(-37), np.int8(-20)])
def test_fit_negative_first_obs(model, first_obs):
    positive = NOBS + int(first_obs)
    neg = model.fit(first_obs=first_obs, **QUIET, **OPTIONS)
    pos = model.fit(first_obs=positive, **QUIET, **OPTIONS)
    assert neg.fit_start == pos.fit_start == positive
    assert neg.fit_stop == pos.fit_stop == NOBS
    assert neg.nobs == pos.nobs == -int(first_obs)
    assert_allclose(neg.params, pos.params, rtol=1e-10)
    # Python slicing selects the same observations
    expected = Y.to_numpy()[int(first_obs) :]
    assert_allclose(neg.model._fit_y, expected)


@pytest.mark.parametrize("last_obs", [-1, -10, -NOBS + 50, np.int64(-3)])
def test_fit_negative_last_obs(model, last_obs):
    positive = NOBS + int(last_obs)
    neg = model.fit(last_obs=last_obs, **QUIET, **OPTIONS)
    pos = model.fit(last_obs=positive, **QUIET, **OPTIONS)
    assert neg.fit_start == pos.fit_start == 0
    assert neg.fit_stop == pos.fit_stop == positive
    assert neg.nobs == pos.nobs == positive
    assert_allclose(neg.params, pos.params, rtol=1e-10)
    # last_obs=-1 drops the last observation, as y[:-1] does
    expected = Y.to_numpy()[: int(last_obs)]
    assert_allclose(neg.model._fit_y, expected)


def test_fit_negative_first_and_last_obs(model):
    neg = model.fit(first_obs=-200, last_obs=-20, **QUIET, **OPTIONS)
    pos = model.fit(first_obs=100, last_obs=280, **QUIET, **OPTIONS)
    assert (neg.fit_start, neg.fit_stop) == (100, 280)
    assert_allclose(neg.params, pos.params, rtol=1e-10)
    assert_allclose(neg.model._fit_y, Y.to_numpy()[-200:-20])
    mixed = model.fit(first_obs=100, last_obs=-20, **QUIET, **OPTIONS)
    assert (mixed.fit_start, mixed.fit_stop) == (100, 280)


@pytest.mark.parametrize("name", ["first_obs", "last_obs"])
@pytest.mark.parametrize("value", [-NOBS - 1, -10 * NOBS, np.int64(-NOBS - 5)])
def test_fit_position_below_minus_nobs(model, name, value):
    with pytest.raises(ValueError, match=rf"^{name} must be an integer position in "):
        model.fit(disp="off", **{name: value})


def test_fit_negative_last_obs_before_first_obs(model):
    with pytest.raises(ValueError, match=r"^last_obs must be larger than first_obs"):
        model.fit(disp="off", first_obs=-10, last_obs=-20)


def test_fit_negative_first_obs_with_hold_back():
    # The position is relative to the data, and the lags are held back after it
    mod = arch_model(Y, mean="AR", lags=2, vol="ARCH", p=1)
    neg = mod.fit(first_obs=-100, **QUIET, **OPTIONS)
    pos = mod.fit(first_obs=NOBS - 100, **QUIET, **OPTIONS)
    assert neg.fit_start == pos.fit_start == NOBS - 100 + 2
    assert_allclose(neg.params, pos.params, rtol=1e-10)


def test_fix_negative_positions(model):
    neg = model.fix(PARAMS, first_obs=-100, last_obs=-1)
    pos = model.fix(PARAMS, first_obs=NOBS - 100, last_obs=NOBS - 1)
    assert neg.model._fit_indices == [NOBS - 100, NOBS - 1]
    assert neg.nobs == pos.nobs == 99
    assert_allclose(neg.loglikelihood, pos.loglikelihood, rtol=1e-12)
    assert_allclose(neg.conditional_volatility, pos.conditional_volatility)
    with pytest.raises(ValueError, match=r"^first_obs must be an integer position"):
        model.fix(PARAMS, first_obs=-NOBS - 1)


def test_negative_positions_with_dates_are_positions(model):
    by_date = model.fit(first_obs=DATES[200], **QUIET, **OPTIONS)
    by_neg = model.fit(first_obs=-100, **QUIET, **OPTIONS)
    assert by_date.fit_start == by_neg.fit_start == 200
    with pytest.raises(ValueError, match="a date within the index"):
        model.fit(disp="off", first_obs=pd.Timestamp("1990-01-01"))


# start is the index of an observation


@pytest.mark.parametrize("start", [-1, -2, -50, -NOBS + 10, np.int64(-7)])
def test_forecast_negative_start(result, start):
    positive = NOBS + int(start)
    neg = result.forecast(horizon=3, start=start, reindex=False)
    pos = result.forecast(horizon=3, start=positive, reindex=False)
    assert neg.mean.index[0] == DATES[positive]
    assert neg.mean.shape == (NOBS - positive, 3)
    assert_allclose(neg.mean, pos.mean)
    assert_allclose(neg.variance, pos.variance)
    assert_allclose(neg.residual_variance, pos.residual_variance)


def test_forecast_start_minus_one_is_the_last_observation(result):
    fcast = result.forecast(horizon=2, start=-1, reindex=False)
    # One row: the forecast made with the data up to the last observation
    assert fcast.variance.shape == (1, 2)
    assert fcast.variance.index[0] == Y.index[-1]
    expected = result.forecast(horizon=2, reindex=False)
    assert_allclose(fcast.variance, expected.variance)


def test_model_forecast_negative_start(model, result):
    neg = model.forecast(result.params, horizon=2, start=-25, reindex=False)
    pos = model.forecast(result.params, horizon=2, start=NOBS - 25, reindex=False)
    assert_allclose(neg.variance, pos.variance)
    assert neg.variance.shape == (25, 2)


def test_forecast_start_below_minus_nobs(result, model):
    for start in (-NOBS - 1, -(10**6), np.int64(-NOBS - 3)):
        with pytest.raises(ValueError, match=r"^start must be an integer position in "):
            result.forecast(horizon=2, start=start)
        with pytest.raises(ValueError, match=r"^start must be an integer position in "):
            model.forecast(result.params, horizon=2, start=start)


def test_hedgehog_plot_negative_start(result):
    kwargs = {"horizon": 3, "step": 50}
    fig_neg = result.hedgehog_plot(start=-100, **kwargs)
    fig_pos = result.hedgehog_plot(start=NOBS - 100, **kwargs)
    try:
        lines_neg = fig_neg.axes[0].lines
        lines_pos = fig_pos.axes[0].lines
        assert len(lines_neg) == len(lines_pos) > 1
        for line_neg, line_pos in zip(lines_neg, lines_pos, strict=True):
            assert_allclose(line_neg.get_ydata(), line_pos.get_ydata())
    finally:
        plt.close("all")
    with pytest.raises(ValueError, match=r"^start must be an integer position in "):
        result.hedgehog_plot(start=-NOBS - 1)


def test_volatility_process_forecast_negative_start():
    vol = GARCH(1, 0, 1)
    resids = np.random.default_rng(12).standard_normal(100)
    params = np.array([0.1, 0.1, 0.8])
    backcast = vol.backcast(resids)
    bounds = vol.variance_bounds(resids)
    kwargs = {"horizon": 3}
    expected = vol.forecast(params, resids, backcast, bounds, start=95, **kwargs)
    for start in (-5, np.int64(-5)):
        fcast = vol.forecast(params, resids, backcast, bounds, start=start, **kwargs)
        assert_allclose(fcast.forecasts, expected.forecasts)
    last = vol.forecast(params, resids, backcast, bounds, start=-1, **kwargs)
    default = vol.forecast(params, resids, backcast, bounds, **kwargs)
    assert_allclose(last.forecasts, default.forecasts)
    assert last.forecasts.shape[0] == 1
    for start in (-101, -(10**6)):
        with pytest.raises(ValueError, match=r"^start must be an integer in \[-100"):
            vol.forecast(params, resids, backcast, bounds, start=start)


def test_arch_forecast_negative_start_hand_check():
    # ARCH(1): the 1-step forecast from the last observation is
    # omega + alpha * resid[-1] ** 2
    resids = np.random.default_rng(3).standard_normal(50)
    vol = ARCH(1)
    params = np.array([0.4, 0.3])
    backcast = vol.backcast(resids)
    bounds = vol.variance_bounds(resids)
    fcast = vol.forecast(params, resids, backcast, bounds, start=-1, horizon=1)
    assert_allclose(fcast.forecasts[0, 0], 0.4 + 0.3 * resids[-1] ** 2)


# The utilities


def test_cutoff_to_index_negative():
    index = pd.RangeIndex(20)
    assert cutoff_to_index(-1, index, 0) == 19
    assert cutoff_to_index(-20, index, 0) == 0
    assert cutoff_to_index(np.int16(-5), index, 0) == 15
    assert cutoff_to_index(20, index, 0) == 20
    for cutoff in (-21, np.int64(-100)):
        with pytest.raises(ValueError, match=r"^cutoff must be an integer in"):
            cutoff_to_index(cutoff, index, 0)
    dates = pd.date_range("2020-01-01", periods=20)
    assert cutoff_to_index(-1, dates, 0) == 19
    assert cutoff_to_index(dates[3], dates, 0) == 3


def test_find_index_negative():
    series = pd.Series(np.arange(20.0), index=pd.date_range("2020-01-01", periods=20))
    assert find_index(series, -1) == 19
    assert find_index(series, np.int8(-20)) == 0
    assert find_index(series, 7) == 7
    with pytest.raises(ValueError, match=r"^index must be an integer in"):
        find_index(series, -21)
    assert find_index(series, "2020-01-05") == 4


# Flags accept 0 and 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, False), (1, True), (np.int64(0), False), (np.uint8(1), True)],
    ids=["0", "1", "int64-0", "uint8-1"],
)
def test_flag_zero_and_one(value, expected):
    model = arch_model(Y, mean="constant", rescale=value)
    assert model.rescale is expected
    assert ARX(Y, constant=value).constant is expected
    assert ConstantMean(Y, rescale=value).rescale is expected


def test_flags_integers_other_than_zero_and_one():
    with pytest.raises(ValueError, match=r"^rescale must be True or False or None"):
        arch_model(Y, rescale=2)
    with pytest.raises(ValueError, match=r"^constant must be True or False; got -1"):
        ARX(Y, constant=-1)
    for bad in (1.0, 0.0, "1", np.array([1])):
        with pytest.raises(TypeError, match=r"^rescale must be True or False or None"):
            arch_model(Y, rescale=bad)


def test_fit_disp_integers(model):
    zero = model.fit(disp=0, show_warning=False, **OPTIONS)
    off = model.fit(disp="off", show_warning=False, **OPTIONS)
    assert_allclose(zero.params, off.params)
    with pytest.raises(ValueError, match=r"^disp must be True or False; got 2"):
        model.fit(disp=2)
    with pytest.raises(TypeError, match=r"^disp must be True or False; got 1.0"):
        model.fit(disp=1.0)
    for text in ("off", "final", "OFF"):
        model.fit(disp=text, show_warning=False, **OPTIONS)


def test_fit_disp_one_prints(model, capsys):
    model.fit(disp=1, show_warning=False, **OPTIONS)
    assert capsys.readouterr().out != ""
    model.fit(disp=0, show_warning=False, **OPTIONS)
    assert capsys.readouterr().out == ""


def test_arch_test_flags():
    with pytest.raises(ValueError, match=r"^exit must be True or False; got 2"):
        arch.test(exit=2)
    with pytest.raises(TypeError, match=r"^append must be True or False; got 'no'"):
        arch.test(append="no")
    with pytest.raises(TypeError, match=r"^exit must be True or False; got None"):
        arch.test(exit=None)


# The horizon alias of align


@pytest.mark.parametrize("spelling", ["horizon", "HORIZON", "Horizon"])
def test_align_horizon_is_target(model, result, spelling):
    target = result.forecast(horizon=3, start=250, align="target")
    alias = result.forecast(horizon=3, start=250, align=spelling)
    assert_allclose(alias.variance, target.variance)
    assert_allclose(alias.mean, target.mean)
    origin = result.forecast(horizon=3, start=250, align="origin")
    # Hand computation: the h-step forecast in column h moves down h rows
    for i, col in enumerate(origin.variance):
        assert_allclose(
            alias.variance[col].to_numpy()[i + 1 :],
            origin.variance[col].to_numpy()[: -(i + 1)],
        )
    # The model accepts it too
    model_alias = model.forecast(result.params, horizon=3, start=250, align=spelling)
    assert_allclose(model_alias.variance, target.variance)


def test_align_horizon_in_the_container_and_helper():
    frame = pd.DataFrame(np.arange(12.0).reshape(4, 3))
    expected = _align_forecast(frame.copy(), "target")
    assert_allclose(_align_forecast(frame.copy(), "horizon"), expected)
    values = frame.to_numpy()
    index = frame.index
    target = ARCHModelForecast(index, 0, values, values, values, align="target")
    alias = ARCHModelForecast(index, 0, values, values, values, align="Horizon")
    assert_allclose(alias.variance, target.variance)


def test_align_message_lists_horizon(model, result):
    match = r"^align must be one of 'origin', 'target' or 'horizon'; got 'x'"
    with pytest.raises(ValueError, match=match):
        result.forecast(horizon=2, start=250, align="x")
    with pytest.raises(ValueError, match=match):
        model.forecast(result.params, horizon=2, start=250, align="x")


def test_align_horizon_is_documented():
    text = "'horizon' is an alias"
    assert text in (ARCHModel.forecast.__doc__ or "")
    assert text in (ARCHModelResult.forecast.__doc__ or "")
    assert text in (ARCHModelForecast.__doc__ or "")
    # The subclasses inherit the documentation
    assert text in (ConstantMean.forecast.__doc__ or "")
    assert text in (ARX.forecast.__doc__ or "")
