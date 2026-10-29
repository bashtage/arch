"""String options of the univariate models are not case sensitive"""

from arch.compat.matplotlib import HAS_MATPLOTLIB

from functools import partial

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.univariate import (
    APARCH,
    ARCH,
    EGARCH,
    FIAPARCH,
    FIGARCH,
    GARCH,
    HARCH,
    HARX,
    LS,
    NGARCH,
    ARCHInMean,
    ConstantMean,
    ConstantVariance,
    GeneralizedError,
    Normal,
    SkewStudent,
    StudentsT,
    ZeroMean,
    arch_model,
)
from arch.univariate.base import ARCHModelForecast, _align_forecast

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt

# A string that is not an option is a ValueError and anything else a TypeError
BAD_VALUES = [
    pytest.param("bogus", ValueError, id="bogus"),
    pytest.param("", ValueError, id="empty"),
    pytest.param(None, TypeError, id="none"),
    pytest.param(1, TypeError, id="int"),
]


@pytest.fixture(scope="module")
def returns():
    rs = np.random.default_rng(20261017)
    index = pd.date_range("2000-01-03", periods=400, freq="B")
    return pd.Series(rs.standard_normal(400), index=index, name="ret")


@pytest.fixture(scope="module")
def result(returns):
    return arch_model(returns, mean="ar", lags=1).fit(disp="off")


def variants(value):
    return [value.upper(), value.capitalize(), value.title(), value.swapcase()]


MEAN_CLASSES = {
    "constant": ConstantMean,
    "zero": ZeroMean,
    "ls": LS,
    "ar": HARX,
    "arx": HARX,
    "har": HARX,
    "harx": HARX,
}


@pytest.mark.parametrize("mean", list(MEAN_CLASSES))
def test_arch_model_mean_case(returns, mean):
    x = pd.DataFrame({"x": np.arange(400.0) % 7}, index=returns.index)
    kwargs = {"lags": [1, 5] if "har" in mean else 2, "x": x}
    expected = arch_model(returns, mean=mean, **kwargs)
    for spelling in variants(mean):
        model = arch_model(returns, mean=spelling, **kwargs)
        assert type(model) is type(expected)
        assert model.name == expected.name
        assert model.num_params == expected.num_params


# The mean models are ARX or HARX for the names that include lags
@pytest.mark.parametrize("mean", ["constant", "zero", "ls"])
def test_arch_model_mean_class(returns, mean):
    x = pd.DataFrame({"x": np.arange(400.0) % 7}, index=returns.index)
    for spelling in [mean, *variants(mean)]:
        assert type(arch_model(returns, mean=spelling, x=x)) is MEAN_CLASSES[mean]


VOL_CLASSES = {
    "garch": GARCH,
    "arch": ARCH,
    "egarch": EGARCH,
    "figarch": FIGARCH,
    "fiaparch": FIAPARCH,
    "aparch": APARCH,
    "ngarch": NGARCH,
    "harch": HARCH,
    "constant": ConstantVariance,
}


@pytest.mark.parametrize("vol", list(VOL_CLASSES))
def test_arch_model_vol_case(returns, vol):
    # NGARCH needs a shift, since o=0 gives a GARCH
    kwargs = {"o": 1} if vol == "ngarch" else {}
    for spelling in [vol, *variants(vol)]:
        model = arch_model(returns, vol=spelling, **kwargs)
        assert type(model.volatility) is VOL_CLASSES[vol]


DIST_CLASSES = {
    "normal": Normal,
    "gaussian": Normal,
    "t": StudentsT,
    "studentst": StudentsT,
    "skewstudent": SkewStudent,
    "skewt": SkewStudent,
    "ged": GeneralizedError,
    "generalized error": GeneralizedError,
}


@pytest.mark.parametrize("dist", list(DIST_CLASSES))
def test_arch_model_dist_case(returns, dist):
    for spelling in [dist, *variants(dist)]:
        model = arch_model(returns, dist=spelling)
        assert type(model.distribution) is DIST_CLASSES[dist]


@pytest.mark.parametrize("param", ["mean", "vol", "dist"])
@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_arch_model_invalid(returns, param, bad, exc):
    with pytest.raises(exc, match=rf"{param} must be one of '.*'; got "):
        arch_model(returns, **{param: bad})


@pytest.mark.parametrize("cov_type", ["robust", "classic"])
def test_fit_cov_type_case(returns, cov_type):
    model = arch_model(returns)
    expected = model.fit(disp="off", cov_type=cov_type)
    for spelling in variants(cov_type):
        res = model.fit(disp="off", cov_type=spelling)
        assert res.cov_type == cov_type
        assert_allclose(res.params, expected.params)
        assert_allclose(res.param_cov, expected.param_cov)


@pytest.mark.parametrize("cov_type", ["robust", "classic"])
def test_fit_cov_type_case_closed_form(returns, cov_type):
    model = ConstantMean(returns)
    expected = model.fit(disp="off", cov_type=cov_type)
    for spelling in variants(cov_type):
        res = model.fit(disp="off", cov_type=spelling)
        assert res.cov_type == expected.cov_type
        assert_allclose(res.param_cov, expected.param_cov)


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_fit_cov_type_invalid(returns, bad, exc):
    with pytest.raises(
        exc, match=r"cov_type must be one of 'robust' or 'classic'; got "
    ):
        arch_model(returns).fit(disp="off", cov_type=bad)


@pytest.mark.parametrize("disp", ["off", "OFF", "Off"])
def test_fit_disp_off_case(returns, disp, capsys):
    arch_model(returns).fit(disp=disp, update_freq=1)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("disp", ["final", "FINAL", "Final"])
def test_fit_disp_final_case(returns, disp, capsys):
    arch_model(returns).fit(disp=disp, update_freq=0)
    assert "Optimization terminated successfully" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["bogus", ""])
def test_fit_disp_invalid(returns, bad):
    with pytest.raises(ValueError, match=r"disp must be one of 'off' or 'final'"):
        arch_model(returns).fit(disp=bad)


@pytest.mark.parametrize("method", ["analytic", "simulation", "bootstrap"])
@pytest.mark.parametrize("use_result", [True, False], ids=["result", "model"])
def test_forecast_method_case(returns, result, method, use_result):
    model = arch_model(returns, mean="ar", lags=1)

    def forecast(value):
        kwargs = {"horizon": 2, "start": 300, "simulations": 20}
        if method != "analytic":
            kwargs["random_state"] = np.random.RandomState(1)
            kwargs["rng"] = lambda size: np.random.default_rng(3).standard_normal(size)
        if use_result:
            return result.forecast(method=value, **kwargs)
        return model.forecast(result.params, method=value, **kwargs)

    expected = forecast(method)
    for spelling in variants(method):
        fcast = forecast(spelling)
        assert_allclose(fcast.mean, expected.mean)
        assert_allclose(fcast.variance, expected.variance)


@pytest.mark.parametrize("align", ["origin", "target"])
def test_forecast_align_case(result, align):
    expected = result.forecast(horizon=3, start=300, align=align)
    for spelling in variants(align):
        fcast = result.forecast(horizon=3, start=300, align=spelling)
        assert_allclose(fcast.variance, expected.variance)
        assert_allclose(fcast.mean, expected.mean)


def test_forecast_align_values(result):
    origin = result.forecast(horizon=3, start=300, align="ORIGIN").variance
    target = result.forecast(horizon=3, start=300, align="Target").variance
    # Hand computation: the h-step forecast in column h is shifted down h rows
    for i, col in enumerate(origin):
        assert_allclose(
            target[col].to_numpy()[i + 1 :], origin[col].to_numpy()[: -(i + 1)]
        )


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_forecast_align_invalid(result, bad, exc):
    match = r"align must be one of 'origin', 'target' or 'horizon'"
    with pytest.raises(exc, match=match):
        result.forecast(horizon=2, start=300, align=bad)
    frame = pd.DataFrame(np.ones((4, 2)))
    with pytest.raises(exc, match=match):
        _align_forecast(frame, bad)
    values = frame.to_numpy()
    with pytest.raises(exc, match=match):
        ARCHModelForecast(frame.index, 0, values, values, values, align=bad)


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
@pytest.mark.parametrize("use_result", [True, False], ids=["result", "model"])
def test_forecast_method_invalid(returns, result, bad, exc, use_result):
    match = r"method must be one of 'analytic', 'simulation' or 'bootstrap'; got "
    model = arch_model(returns, mean="ar", lags=1)
    forecast = result.forecast if use_result else partial(model.forecast, result.params)
    with pytest.raises(exc, match=match):
        forecast(method=bad)


@pytest.mark.parametrize("annualize", ["d", "w", "m"])
@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plot_annualize_case(result, annualize):
    scale = {"d": 252, "w": 52, "m": 12}[annualize]
    expected = result.conditional_volatility * np.sqrt(scale)
    try:
        for spelling in (annualize, annualize.upper()):
            fig = result.plot(annualize=spelling)
            vol_line = fig.axes[1].lines[0]
            assert_allclose(vol_line.get_ydata(), expected)
            assert fig.axes[1].get_title() == "Annualized Conditional Volatility"
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    ("bad", "exc"), [("bogus", ValueError), ("Q", ValueError), (1, TypeError)]
)
@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plot_annualize_invalid(result, bad, exc):
    with pytest.raises(exc, match=r"annualize must be one of 'd', 'w' or 'm'; got "):
        result.plot(annualize=bad)
    plt.close("all")


@pytest.mark.parametrize("plot_type", ["volatility", "mean"])
@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_hedgehog_plot_type_case(result, plot_type, returns):
    try:
        for spelling in (plot_type, plot_type.upper(), plot_type.title()):
            fig = result.hedgehog_plot(
                horizon=2, step=100, start=250, plot_type=spelling
            )
            data = np.asarray(fig.axes[0].lines[0].get_ydata())
            if plot_type == "mean":
                assert_allclose(data, returns.to_numpy())
            else:
                assert_allclose(data, result.conditional_volatility)
    finally:
        plt.close("all")


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_hedgehog_plot_invalid(result, bad, exc):
    with pytest.raises(exc, match=r"plot_type must be one of 'volatility' or"):
        result.hedgehog_plot(plot_type=bad)
    with pytest.raises(exc, match=r"method must be one of 'analytic', 'sim"):
        result.hedgehog_plot(method=bad)
    plt.close("all")


@pytest.mark.parametrize("form", ["log", "vol", "var"])
def test_arch_in_mean_form_case(returns, form):
    for spelling in variants(form):
        model = ARCHInMean(returns, volatility=GARCH(), form=spelling)
        assert model.form == form


@pytest.mark.parametrize("bad", ["bogus", ""])
def test_arch_in_mean_form_invalid(returns, bad):
    with pytest.raises(ValueError, match=r"form must be 'log', 'vol', 'var' or a "):
        ARCHInMean(returns, volatility=GARCH(), form=bad)
