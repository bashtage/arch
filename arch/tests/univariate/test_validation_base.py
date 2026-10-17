"""
Table-driven tests of the argument validation of the univariate model base classes.

The rows cover the arguments of ``ARCHModel`` (the constructor, ``fit``, ``fix`` and
``compute_param_cov``), of the result classes (``forecast``, ``hedgehog_plot``, ``plot``,
``arch_lm_test``, ``conf_int``, ``hill_estimator`` and ``var_ratio``), of the
functions in ``arch.univariate.base`` and of ``arch.univariate.diagnostics``.
"""

from collections.abc import Callable
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.tests.validation_table import (
    BOOL_BAD,
    BOOL_GOOD,
    FLOAT_BAD,
    FLOAT_GOOD,
    INT_BAD,
    INT_GOOD,
    Row,
    bad_value_rows,
    good_value_rows,
    run_row,
    to_params,
)
from arch.univariate import (
    ARCH,
    GARCH,
    HARCH,
    ConstantMean,
    Normal,
    StudentsT,
)
from arch.univariate.base import constraint, format_float_fixed, implicit_constant
from arch.univariate.diagnostics import excess_kurtosis, hill_estimator, var_ratio
from arch.utility.exceptions import StartingValueWarning

NOBS = 250
INDEX = pd.date_range("2000-01-03", periods=NOBS, freq="B")
PARAMS = np.array([0.05, 0.05, 0.1, 0.8])


def _simulate() -> np.ndarray:
    mod = ConstantMean(volatility=GARCH(1, 0, 1), distribution=Normal(seed=20241017))
    return np.asarray(mod.simulate(PARAMS, NOBS)["data"])


Y = _simulate()
Y_SERIES = pd.Series(Y, index=INDEX, name="y")


def model(y: Any = Y, **kwargs: Any) -> ConstantMean:
    return ConstantMean(y, volatility=GARCH(1, 0, 1), **kwargs)


_FITTED: dict[str, Any] = {}


def fitted():
    """A result that is fit once and shared by the rows that do not change it."""
    if "result" not in _FITTED:
        _FITTED["result"] = model().fit(disp="off", show_warning=False)
    return _FITTED["result"]


def without_none(bad: list[tuple[Any, type[Exception]]]) -> list:
    return [pair for pair in bad if pair[0] is not None]


def plus(bad: list[tuple[Any, type[Exception]]], *extra: tuple[Any, type[Exception]]):
    return [*bad, *extra]


# Arguments of the constructors
def build(**kwargs: Any) -> ConstantMean:
    return ConstantMean(Y, **kwargs)


BUILD: dict[str, Any] = {}
NON_NEGATIVE_INT_BAD = plus(without_none(INT_BAD), (-1, ValueError))
INSTANCE_BAD = [
    ("GARCH", TypeError),
    (1, TypeError),
    (GARCH, TypeError),
    (np.array([1.0]), TypeError),
]

CONSTRUCTOR_ROWS = [
    *bad_value_rows(
        "init",
        build,
        BUILD,
        "hold_back",
        NON_NEGATIVE_INT_BAD,
        rule="an integer >= 0 or None",
    ),
    *good_value_rows("init", build, BUILD, "hold_back", [None, 0, *INT_GOOD]),
    *bad_value_rows("init", build, BUILD, "rescale", without_none(BOOL_BAD)),
    *good_value_rows("init", build, BUILD, "rescale", [None, *BOOL_GOOD]),
    *bad_value_rows("init", build, BUILD, "volatility", INSTANCE_BAD),
    *bad_value_rows("init", build, BUILD, "distribution", INSTANCE_BAD),
    *good_value_rows("init", build, BUILD, "volatility", [None, GARCH()]),
    *good_value_rows("init", build, BUILD, "distribution", [None, StudentsT()]),
]


def set_volatility(value: Any) -> None:
    model().volatility = value


def set_distribution(value: Any) -> None:
    model().distribution = value


SETTER_ROWS = [
    Row(
        f"set-{name}",
        lambda value, setter=setter: setter(value),
        {"value": value},
        TypeError,
        rf"^{name} must be an instance of",
        id=f"set-{name}-{i}",
        name=name,
    )
    for name, setter in (
        ("volatility", set_volatility),
        ("distribution", set_distribution),
    )
    for i, value in enumerate(["GARCH", 1, None, GARCH])
]


# fit
def fit(**kwargs: Any):
    kwargs = {"disp": "off", "show_warning": False, **kwargs}
    return model().fit(**kwargs)


def fit_dates(**kwargs: Any):
    kwargs = {"disp": "off", "show_warning": False, **kwargs}
    return model(Y_SERIES).fit(**kwargs)


FIT: dict[str, Any] = {}
FIT_FAST: dict[str, Any] = {"options": {"maxiter": 3}}
POSITION_BAD = [
    (-1, ValueError),
    (-1000, ValueError),
    (np.int64(-5), ValueError),
    (NOBS + 1, ValueError),
    (10**6, ValueError),
    (2.5, TypeError),
    (5.0, TypeError),
    (True, TypeError),
    ("2000-01-05", ValueError),
    ([5], TypeError),
]
STARTING_BAD = [
    ("abc", TypeError),
    (["a", "b", "c", "d"], TypeError),
    (1j, TypeError),
    (2.5, ValueError),
    ([0.05, 0.05, 0.1], ValueError),
    ([0.05, 0.05, 0.1, 0.8, 0.1], ValueError),
    ([0.05, np.nan, 0.1, 0.8], ValueError),
    ([0.05, np.inf, 0.1, 0.8], ValueError),
    (np.ones((2, 2)), ValueError),
    ([], ValueError),
]
STARTING_GOOD = [
    None,
    list(PARAMS),
    PARAMS.astype(np.float32),
    pd.Series(PARAMS),
    PARAMS.reshape(-1, 1),
    tuple(PARAMS),
]
BACKCAST_BAD = [
    ("abc", TypeError),
    (True, TypeError),
    (1j, TypeError),
    (np.nan, ValueError),
    (np.inf, ValueError),
    (-1.0, ValueError),
    (0.0, ValueError),
    (-1, ValueError),
    ([np.nan, 1.0], ValueError),
    (np.ones((2, 2)), ValueError),
]
OPTIONS_BAD = [
    (1, TypeError),
    ("maxiter", TypeError),
    ([("maxiter", 5)], TypeError),
    (True, TypeError),
]

FIT_ROWS = [
    *bad_value_rows(
        "fit",
        fit,
        FIT_FAST,
        "update_freq",
        plus(without_none(INT_BAD), (-1, ValueError), (-(10**6), ValueError)),
        rule="an integer >= 0",
    ),
    *good_value_rows("fit", fit, FIT_FAST, "update_freq", [0, 1, *INT_GOOD]),
    *bad_value_rows(
        "fit",
        fit,
        FIT_FAST,
        "disp",
        [
            ("zzz", ValueError),
            ("", ValueError),
            (0, TypeError),
            (1, TypeError),
            (None, TypeError),
            (1.0, TypeError),
            (np.array([True]), TypeError),
        ],
    ),
    *good_value_rows(
        "fit", fit, FIT_FAST, "disp", ["off", "final", "OFF", "Final", *BOOL_GOOD]
    ),
    *bad_value_rows("fit", fit, FIT_FAST, "starting_values", STARTING_BAD),
    *good_value_rows("fit", fit, FIT_FAST, "starting_values", STARTING_GOOD),
    *bad_value_rows(
        "fit",
        fit,
        FIT_FAST,
        "cov_type",
        [("zzz", ValueError), (1, TypeError), (None, TypeError)],
    ),
    *good_value_rows("fit", fit, FIT_FAST, "cov_type", ["robust", "CLASSIC"]),
    *bad_value_rows("fit", fit, FIT_FAST, "show_warning", BOOL_BAD),
    *good_value_rows("fit", fit, {}, "show_warning", BOOL_GOOD),
    *bad_value_rows("fit", fit, FIT_FAST, "first_obs", POSITION_BAD),
    *bad_value_rows("fit", fit, FIT_FAST, "last_obs", POSITION_BAD),
    *good_value_rows(
        "fit", fit, FIT_FAST, "first_obs", [None, 0, 10, np.int64(10), np.uint8(10)]
    ),
    *good_value_rows(
        "fit",
        fit,
        FIT_FAST,
        "last_obs",
        [None, NOBS, 100, np.int32(100), np.int64(NOBS)],
    ),
    *bad_value_rows(
        "fit",
        fit,
        FIT_FAST,
        "tol",
        plus(without_none(FLOAT_BAD), (-1.0, ValueError), (-1, ValueError)),
        rule="a finite float >= 0 or None",
    ),
    *good_value_rows("fit", fit, FIT_FAST, "tol", [None, 0.5, 1e-4, 1, *FLOAT_GOOD]),
    *bad_value_rows(
        "fit",
        fit,
        {},
        "options",
        OPTIONS_BAD,
        rule="an instance of Mapping or None",
    ),
    *good_value_rows(
        "fit",
        fit,
        {},
        "options",
        [None, {}, {"maxiter": 3}, MappingProxyType({"maxiter": 3})],
    ),
    *bad_value_rows("fit", fit, FIT_FAST, "backcast", BACKCAST_BAD),
    *good_value_rows(
        "fit", fit, FIT_FAST, "backcast", [None, 0.5, 1, np.float32(0.5), np.int64(1)]
    ),
]

# Dates select the sample of a model with a DatetimeIndex
DATE_BAD = [
    (pd.Timestamp("1990-01-01"), ValueError),
    (pd.Timestamp("2100-01-01"), ValueError),
    ("1990-01-01", ValueError),
    ("not a date", ValueError),
    (1.5, TypeError),
]
DATE_GOOD = [
    pd.Timestamp("2000-03-01"),
    "2000-03-01",
    pd.Timestamp("2000-03-01").to_pydatetime(),
    pd.Timestamp("2000-03-01").to_datetime64(),
    10,
]
DATE_ROWS = [
    *bad_value_rows("fit-dates", fit_dates, FIT_FAST, "first_obs", DATE_BAD),
    *bad_value_rows("fit-dates", fit_dates, FIT_FAST, "last_obs", DATE_BAD),
    *good_value_rows("fit-dates", fit_dates, FIT_FAST, "first_obs", DATE_GOOD),
    *good_value_rows("fit-dates", fit_dates, FIT_FAST, "last_obs", [*DATE_GOOD, None]),
]


# More parameters than observations
def fit_with(volatility: Any, **kwargs: Any):
    return ConstantMean(Y, volatility=volatility).fit(
        disp="off", show_warning=False, **kwargs
    )


TOO_MANY = r"^the number of parameters must be at most the number of observations"
TOO_MANY_ROWS = [
    Row(
        f"too-many-{label}",
        fit_with,
        {"volatility": volatility, **extra},
        ValueError,
        TOO_MANY + rf".* \({nobs}\); got {total} \(int\)$",
        id=f"too-many-{label}",
        name="the number of parameters",
    )
    for label, volatility, extra, nobs, total in [
        ("garch-p", GARCH(p=300), {}, NOBS, 303),
        ("garch-o", GARCH(o=300), {}, NOBS, 304),
        ("garch-q", GARCH(q=300), {}, NOBS, 303),
        ("arch", ARCH(p=300), {}, NOBS, 302),
        ("harch", HARCH(lags=300), {}, NOBS, 302),
        ("hold-back-first-obs", GARCH(), {"first_obs": 100, "last_obs": 102}, 2, 4),
        ("last-obs", GARCH(p=2, q=2), {"last_obs": 5}, 5, 6),
    ]
]


# fix
def fix(**kwargs: Any):
    kwargs = {"params": PARAMS, **kwargs}
    return model().fix(**kwargs)


def fix_dates(**kwargs: Any):
    kwargs = {"params": PARAMS, **kwargs}
    return model(Y_SERIES).fix(**kwargs)


PARAMS_BAD = [
    ("abc", TypeError),
    (None, TypeError),
    (["a", "b", "c", "d"], TypeError),
    (1.0, ValueError),
    ([0.05, 0.05, 0.1], ValueError),
    ([0.05, 0.05, 0.1, 0.8, 0.1], ValueError),
    ([0.05, np.nan, 0.1, 0.8], ValueError),
    ([0.05, np.inf, 0.1, 0.8], ValueError),
    (np.ones((2, 2)), ValueError),
]
PARAMS_GOOD = [
    list(PARAMS),
    PARAMS.astype(np.float32),
    pd.Series(PARAMS),
    PARAMS[:, None],
]

FIX_ROWS = [
    *bad_value_rows("fix", fix, {}, "params", PARAMS_BAD),
    *good_value_rows("fix", fix, {}, "params", PARAMS_GOOD),
    *bad_value_rows("fix", fix, {}, "first_obs", POSITION_BAD),
    *bad_value_rows("fix", fix, {}, "last_obs", POSITION_BAD),
    *good_value_rows("fix", fix, {}, "first_obs", [None, 0, 10, np.int64(10)]),
    *good_value_rows("fix", fix, {}, "last_obs", [None, NOBS, np.int32(100)]),
    *bad_value_rows("fix-dates", fix_dates, {}, "first_obs", DATE_BAD),
    *bad_value_rows("fix-dates", fix_dates, {}, "last_obs", DATE_BAD),
    *good_value_rows("fix-dates", fix_dates, {}, "first_obs", DATE_GOOD),
]


# compute_param_cov
def param_cov(**kwargs: Any):
    res = model().fit(disp="off", show_warning=False)
    kwargs = {"params": np.asarray(res.params), **kwargs}
    return res.model.compute_param_cov(**kwargs)


COV_BASE: dict[str, Any] = {}
COV_ROWS = [
    *bad_value_rows(
        "param-cov", param_cov, COV_BASE, "params", PARAMS_BAD[:-2] + [PARAMS_BAD[-1]]
    ),
    *bad_value_rows("param-cov", param_cov, COV_BASE, "backcast", BACKCAST_BAD),
    *bad_value_rows("param-cov", param_cov, COV_BASE, "robust", BOOL_BAD),
    *good_value_rows("param-cov", param_cov, COV_BASE, "robust", BOOL_GOOD),
    *good_value_rows(
        "param-cov", param_cov, COV_BASE, "backcast", [None, 0.5, 1, np.float32(0.5)]
    ),
]


# Methods of the results
def result_method(name: str) -> Callable[..., Any]:
    def call(**kwargs: Any) -> Any:
        try:
            return getattr(fitted(), name)(**kwargs)
        finally:
            plt.close("all")

    return call


def result_dates(name: str) -> Callable[..., Any]:
    def call(**kwargs: Any) -> Any:
        res = model(Y_SERIES).fit(
            disp="off", show_warning=False, options={"maxiter": 3}
        )
        return getattr(res, name)(**kwargs)

    return call


forecast = result_method("forecast")
hedgehog = result_method("hedgehog_plot")
FORECAST_BASE = {"horizon": 2}
SIM_BASE = {"horizon": 2, "method": "simulation", "simulations": 5}
HEDGEHOG_BASE = {"horizon": 3, "step": 100}
HORIZON_BAD = plus(
    INT_BAD, (0, ValueError), (-1, ValueError), (np.int64(-3), ValueError)
)
OPTION_BAD = [("zzz", ValueError), ("", ValueError), (1, TypeError), (None, TypeError)]
RNG_BAD = [("abc", TypeError), (1, TypeError), ([0.1], TypeError)]
RANDOM_STATE_BAD = [
    (1, TypeError),
    ("abc", TypeError),
    (np.random.default_rng(1), TypeError),
]

FORECAST_ROWS = [
    *bad_value_rows(
        "forecast", forecast, FORECAST_BASE, "params", without_none(PARAMS_BAD)
    ),
    *good_value_rows(
        "forecast",
        forecast,
        FORECAST_BASE,
        "params",
        [None, *PARAMS_GOOD],
    ),
    *bad_value_rows(
        "forecast",
        forecast,
        FORECAST_BASE,
        "horizon",
        HORIZON_BAD,
        rule="an integer >= 1",
    ),
    *good_value_rows("forecast", forecast, {}, "horizon", [1, *INT_GOOD]),
    *bad_value_rows(
        "forecast",
        forecast,
        FORECAST_BASE,
        "start",
        POSITION_BAD,
    ),
    *good_value_rows(
        "forecast",
        forecast,
        FORECAST_BASE,
        "start",
        [None, 0, 100, NOBS - 1, NOBS, np.int64(100), np.int16(100)],
    ),
    *bad_value_rows("forecast", forecast, FORECAST_BASE, "align", OPTION_BAD),
    *good_value_rows(
        "forecast", forecast, FORECAST_BASE, "align", ["origin", "TARGET", "target"]
    ),
    *bad_value_rows("forecast", forecast, FORECAST_BASE, "method", OPTION_BAD),
    *good_value_rows(
        "forecast",
        forecast,
        {"horizon": 2, "simulations": 5},
        "method",
        ["analytic", "Simulation", "bootstrap"],
    ),
    *bad_value_rows(
        "forecast",
        forecast,
        SIM_BASE,
        "simulations",
        plus(INT_BAD, (0, ValueError), (-1, ValueError)),
        rule="an integer >= 1",
    ),
    *good_value_rows("forecast", forecast, SIM_BASE, "simulations", [1, *INT_GOOD]),
    *bad_value_rows("forecast", forecast, SIM_BASE, "rng", RNG_BAD),
    *good_value_rows(
        "forecast",
        forecast,
        SIM_BASE,
        "rng",
        [None, np.random.default_rng(3).standard_normal],
    ),
    *bad_value_rows(
        "forecast",
        forecast,
        {"horizon": 2, "method": "bootstrap", "simulations": 5},
        "random_state",
        RANDOM_STATE_BAD,
    ),
    *good_value_rows(
        "forecast",
        forecast,
        {"horizon": 2, "method": "bootstrap", "simulations": 5},
        "random_state",
        [None, np.random.RandomState(1)],
    ),
    *bad_value_rows("forecast", forecast, FORECAST_BASE, "reindex", BOOL_BAD),
    *good_value_rows("forecast", forecast, FORECAST_BASE, "reindex", BOOL_GOOD),
    *bad_value_rows(
        "forecast-dates", result_dates("forecast"), FORECAST_BASE, "start", DATE_BAD
    ),
    *good_value_rows(
        "forecast-dates", result_dates("forecast"), FORECAST_BASE, "start", DATE_GOOD
    ),
]

HEDGEHOG_ROWS = [
    *bad_value_rows(
        "hedgehog", hedgehog, HEDGEHOG_BASE, "params", without_none(PARAMS_BAD)
    ),
    *good_value_rows(
        "hedgehog", hedgehog, HEDGEHOG_BASE, "params", [None, *PARAMS_GOOD]
    ),
    *bad_value_rows(
        "hedgehog",
        hedgehog,
        HEDGEHOG_BASE,
        "horizon",
        HORIZON_BAD,
        rule="an integer >= 1",
    ),
    *good_value_rows("hedgehog", hedgehog, HEDGEHOG_BASE, "horizon", [1, *INT_GOOD]),
    *bad_value_rows(
        "hedgehog",
        hedgehog,
        HEDGEHOG_BASE,
        "step",
        plus(INT_BAD, (0, ValueError), (-1, ValueError)),
        rule="an integer >= 1",
    ),
    *good_value_rows("hedgehog", hedgehog, HEDGEHOG_BASE, "step", [1000, *INT_GOOD]),
    *bad_value_rows("hedgehog", hedgehog, HEDGEHOG_BASE, "start", POSITION_BAD),
    *good_value_rows(
        "hedgehog", hedgehog, HEDGEHOG_BASE, "start", [None, 0, 100, np.int64(100)]
    ),
    *bad_value_rows("hedgehog", hedgehog, HEDGEHOG_BASE, "plot_type", OPTION_BAD),
    *good_value_rows(
        "hedgehog", hedgehog, HEDGEHOG_BASE, "plot_type", ["mean", "VOLATILITY"]
    ),
    *bad_value_rows("hedgehog", hedgehog, HEDGEHOG_BASE, "method", OPTION_BAD),
    *bad_value_rows(
        "hedgehog",
        hedgehog,
        {**HEDGEHOG_BASE, "method": "simulation"},
        "simulations",
        plus(INT_BAD, (0, ValueError), (-1, ValueError)),
        rule="an integer >= 1",
    ),
    *good_value_rows(
        "hedgehog",
        hedgehog,
        {**HEDGEHOG_BASE, "method": "simulation"},
        "simulations",
        [3, *INT_GOOD],
    ),
]

conf_int = result_method("conf_int")
arch_lm_test = result_method("arch_lm_test")
hill = result_method("hill_estimator")
ratio = result_method("var_ratio")
plot = result_method("plot")
ALPHA_BAD = plus(
    FLOAT_BAD,
    (0.0, ValueError),
    (1.0, ValueError),
    (-1.0, ValueError),
    (2, ValueError),
    (0, ValueError),
    (-np.inf, ValueError),
)

RESULT_ROWS = [
    *bad_value_rows(
        "conf-int", conf_int, {}, "alpha", ALPHA_BAD, rule=r"a finite float in \(0, 1\)"
    ),
    *good_value_rows(
        "conf-int",
        conf_int,
        {},
        "alpha",
        [0.05, 0.9, 0.999, np.float32(0.1), *FLOAT_GOOD],
    ),
    *bad_value_rows(
        "arch-lm",
        arch_lm_test,
        {},
        "lags",
        plus(without_none(INT_BAD), (0, ValueError), (-1, ValueError)),
        rule="an integer >= 1 or None",
    ),
    *good_value_rows("arch-lm", arch_lm_test, {}, "lags", [None, 1, *INT_GOOD]),
    *bad_value_rows("arch-lm", arch_lm_test, {}, "standardized", BOOL_BAD),
    *good_value_rows("arch-lm", arch_lm_test, {}, "standardized", BOOL_GOOD),
    *bad_value_rows(
        "hill",
        hill,
        {},
        "k",
        plus(
            without_none(INT_BAD),
            (0, ValueError),
            (-1, ValueError),
            (10**6, ValueError),
        ),
    ),
    *good_value_rows("hill", hill, {}, "k", [None, 1, *INT_GOOD]),
    *bad_value_rows(
        "var-ratio",
        ratio,
        {},
        "level",
        plus(
            FLOAT_BAD,
            (0.5, ValueError),
            (1.0, ValueError),
            (0.0, ValueError),
            (1.5, ValueError),
            (0, ValueError),
            (-0.9, ValueError),
        ),
        rule=r"a finite float in \(0.5, 1\)",
    ),
    *good_value_rows(
        "var-ratio", ratio, {}, "level", [0.99, 0.999, np.float32(0.9), Fraction(9, 10)]
    ),
    *bad_value_rows(
        "plot",
        plot,
        {},
        "annualize",
        [("zzz", ValueError), ("", ValueError), (1, TypeError), (252, TypeError)],
    ),
    *good_value_rows("plot", plot, {}, "annualize", [None, "d", "W", "m"]),
    *bad_value_rows(
        "plot",
        plot,
        {},
        "scale",
        plus(
            without_none(FLOAT_BAD),
            (0.0, ValueError),
            (-1, ValueError),
            (0, ValueError),
        ),
        rule="a finite float > 0 or None",
    ),
    *good_value_rows(
        "plot", plot, {}, "scale", [None, 252, 360.0, np.float32(52), np.int64(12)]
    ),
    # annualize is checked also when scale replaces it
    Row(
        "plot",
        plot,
        {"annualize": "zzz", "scale": 252.0},
        ValueError,
        r"^annualize must be one of 'd', 'w' or 'm'; got 'zzz' \(str\)",
        id="plot-annualize-with-scale",
        name="annualize",
    ),
]


# Functions of the module
def constraints(**kwargs: Any):
    return constraint(**kwargs)


CONSTRAINT_BASE = {"a": np.eye(2), "b": np.zeros(2)}
FORMAT_BASE = {"x": 1.5, "max_digits": 10, "decimal": 4}
FLOAT_NAN_OK = [
    pair
    for pair in FLOAT_BAD
    if not (isinstance(pair[0], float) and not np.isfinite(pair[0]))
]
MODULE_ROWS = [
    *bad_value_rows(
        "constraint",
        constraints,
        CONSTRAINT_BASE,
        "a",
        [
            ("abc", TypeError),
            (None, TypeError),
            (np.ones(2), ValueError),
            (np.ones((2, 2, 2)), ValueError),
            (np.full((2, 2), np.nan), ValueError),
        ],
    ),
    Row(
        "constraint",
        constraints,
        {"a": np.ones((3, 2)), "b": np.zeros(2)},
        ValueError,
        r"^b must be a 1-d array of finite numbers with shape \(3,\)",
        id="constraint-rows-differ",
        name="b",
    ),
    *bad_value_rows(
        "constraint",
        constraints,
        CONSTRAINT_BASE,
        "b",
        [
            ("abc", TypeError),
            (None, TypeError),
            (np.ones((2, 2)), ValueError),
            (np.ones(3), ValueError),
            (np.array([0.0, np.nan]), ValueError),
        ],
    ),
    *good_value_rows(
        "constraint",
        constraints,
        CONSTRAINT_BASE,
        "a",
        [np.eye(2), np.eye(2, dtype=int), [[1.0, 0.0], [0.0, 1.0]]],
    ),
    Row(
        "constraint",
        constraints,
        {"a": np.empty((0, 4)), "b": np.empty(0)},
        None,
        id="constraint-empty",
    ),
    *bad_value_rows(
        "format-float",
        format_float_fixed,
        FORMAT_BASE,
        "x",
        [pair for pair in FLOAT_NAN_OK if pair[0] is not None] + [(None, TypeError)],
        rule=r"a float \(NaN and inf allowed\)",
    ),
    *good_value_rows(
        "format-float",
        format_float_fixed,
        FORMAT_BASE,
        "x",
        [0.0, 1, np.float32(0.5), np.nan, np.inf, -np.inf, 1e-12, 1e12, *FLOAT_GOOD],
    ),
    *bad_value_rows(
        "format-float",
        format_float_fixed,
        FORMAT_BASE,
        "max_digits",
        plus(
            without_none(INT_BAD), (None, TypeError), (0, ValueError), (-1, ValueError)
        ),
    ),
    *good_value_rows(
        "format-float", format_float_fixed, FORMAT_BASE, "max_digits", [1, *INT_GOOD]
    ),
    *bad_value_rows(
        "format-float",
        format_float_fixed,
        FORMAT_BASE,
        "decimal",
        plus(INT_BAD, (-1, ValueError)),
    ),
    *good_value_rows(
        "format-float", format_float_fixed, FORMAT_BASE, "decimal", [0, *INT_GOOD]
    ),
    *bad_value_rows(
        "implicit-constant",
        implicit_constant,
        {"x": np.ones((10, 2))},
        "x",
        [
            ("abc", TypeError),
            (None, TypeError),
            (np.full((10, 2), np.nan), ValueError),
            (np.full((10, 2), np.inf), ValueError),
            (np.ones((2, 2, 2)), ValueError),
        ],
    ),
    *good_value_rows(
        "implicit-constant",
        implicit_constant,
        {"x": np.ones((10, 2))},
        "x",
        [
            np.ones((10, 2)),
            np.ones((10, 2), dtype=int),
            np.ones(10),
            pd.DataFrame(np.ones((10, 2))),
        ],
    ),
]


# diagnostics
STD_RESID = np.random.default_rng(7).standard_t(6, 200)
STD_BASE = {"std_resid": STD_RESID}
STD_BAD = [
    ("abc", TypeError),
    (None, TypeError),
    (["a", "b"], TypeError),
    (np.ones((5, 2)), ValueError),
    (np.ones((2, 2, 2)), ValueError),
]
STD_GOOD = [
    STD_RESID,
    list(STD_RESID),
    pd.Series(STD_RESID),
    STD_RESID[:, None],
    np.r_[STD_RESID, np.nan, np.inf],  # NaN and inf are dropped by design
]
DIAGNOSTIC_ROWS = [
    *bad_value_rows("excess-kurtosis", excess_kurtosis, STD_BASE, "std_resid", STD_BAD),
    *good_value_rows(
        "excess-kurtosis", excess_kurtosis, STD_BASE, "std_resid", STD_GOOD
    ),
    *bad_value_rows("hill-function", hill_estimator, STD_BASE, "std_resid", STD_BAD),
    *good_value_rows("hill-function", hill_estimator, STD_BASE, "std_resid", STD_GOOD),
    *bad_value_rows(
        "hill-function",
        hill_estimator,
        STD_BASE,
        "k",
        plus(
            without_none(INT_BAD),
            (0, ValueError),
            (-1, ValueError),
            (10**6, ValueError),
        ),
    ),
    *good_value_rows(
        "hill-function", hill_estimator, STD_BASE, "k", [None, 1, *INT_GOOD]
    ),
    *bad_value_rows(
        "var-ratio-function",
        var_ratio,
        {},
        "level",
        plus(
            FLOAT_BAD,
            (0.5, ValueError),
            (1.0, ValueError),
            (0.0, ValueError),
            (1.5, ValueError),
            (-0.9, ValueError),
        ),
        rule=r"a finite float in \(0.5, 1\)",
    ),
    *good_value_rows(
        "var-ratio-function",
        var_ratio,
        {},
        "level",
        [0.99, np.float32(0.9), Fraction(9, 10)],
    ),
    *bad_value_rows(
        "var-ratio-function",
        var_ratio,
        {},
        "distribution",
        [("normal", TypeError), (1, TypeError), (Normal, TypeError)],
        rule="an instance of Distribution or None",
    ),
    *good_value_rows(
        "var-ratio-function", var_ratio, {}, "distribution", [None, Normal()]
    ),
    *bad_value_rows(
        "var-ratio-function",
        var_ratio,
        {"distribution": StudentsT(), "parameters": [8.0]},
        "parameters",
        [
            (None, TypeError),
            ("abc", TypeError),
            ([np.nan], ValueError),
            ([np.inf], ValueError),
            ([8.0, 9.0], ValueError),
            ([], ValueError),
            (np.ones((1, 2)), ValueError),
        ],
    ),
    *good_value_rows(
        "var-ratio-function",
        var_ratio,
        {"distribution": StudentsT()},
        "parameters",
        [[8.0], np.array([8.0], dtype=np.float32), pd.Series([8.0]), (8.0,)],
    ),
    Row(
        "var-ratio-function",
        var_ratio,
        {"distribution": Normal(), "parameters": [1.0]},
        ValueError,
        r"^parameters must be a 1-d array of finite numbers with shape \(0,\)",
        id="var-ratio-function-normal-parameters",
        name="parameters",
    ),
]


@pytest.mark.parametrize(
    "row",
    to_params(
        [
            *CONSTRUCTOR_ROWS,
            *SETTER_ROWS,
            *FIT_ROWS,
            *DATE_ROWS,
            *TOO_MANY_ROWS,
            *FIX_ROWS,
            *COV_ROWS,
            *FORECAST_ROWS,
            *HEDGEHOG_ROWS,
            *RESULT_ROWS,
            *MODULE_ROWS,
            *DIAGNOSTIC_ROWS,
        ]
    ),
)
def test_validation(row):
    run_row(row)


def test_starting_values_outside_bounds_warn_and_fall_back():
    mod = model()
    ref = mod.fit(disp="off", show_warning=False, options={"maxiter": 0})
    with pytest.warns(StartingValueWarning, match="will be ignored"):
        res = mod.fit(
            disp="off",
            show_warning=False,
            options={"maxiter": 0},
            starting_values=[0.0, -5.0, 0.5, 0.5],
        )
    assert_allclose(res.params, ref.params)


def test_fit_sample_is_a_position_in_the_data():
    res = model().fit(disp="off", show_warning=False, first_obs=10, last_obs=200)
    assert res.fit_start == 10
    assert res.fit_stop == 200
    with pytest.raises(ValueError, match=r"^first_obs must be an integer position"):
        model().fit(disp="off", first_obs=-1000)
    with pytest.raises(ValueError, match=r"^last_obs must be an integer position"):
        model().fit(disp="off", last_obs=10**6)


def test_valid_fit_is_unchanged_by_integer_types():
    ref = model().fit(disp="off", show_warning=False, update_freq=0, first_obs=5)
    res = model().fit(
        disp="off", show_warning=False, update_freq=np.int8(0), first_obs=np.int64(5)
    )
    assert_allclose(res.params, ref.params)
    assert res.fit_start == 5


def test_model_without_data_still_simulates():
    mod = ConstantMean(volatility=GARCH(1, 0, 1), hold_back=None)
    out = mod.simulate(PARAMS, 20, burn=5)
    assert out.shape[0] == 20


def test_forecast_of_fixed_result_checks_arguments():
    fixed = model().fix(PARAMS)
    with pytest.raises(TypeError, match=r"^horizon must be an integer >= 1"):
        fixed.forecast(horizon=2.0)
    with pytest.raises(ValueError, match=r"^start must be an integer position"):
        fixed.forecast(start=10**6)
    with pytest.raises(ValueError, match=r"^params must be a 1-d array"):
        fixed.forecast(params=[1.0, 2.0])
    assert fixed.forecast(horizon=np.int64(2)).mean.shape[1] == 2


def test_plot_error_does_not_leave_a_figure():
    plt.close("all")
    with pytest.raises(ValueError, match=r"^annualize must be"):
        fitted().plot(annualize="zzz")
    assert plt.get_fignums() == []
