"""
Input validation of the cointegration tests and estimators.

The table-driven rows are built on arch.tests.validation_table.
"""

from arch.compat.matplotlib import HAS_MATPLOTLIB
from arch.compat.pandas import MONTH_END

import numpy as np
import pandas as pd
import pytest

from arch.tests.validation_table import (
    BOOL_BAD,
    BOOL_GOOD,
    FLOAT_BAD,
    INT_BAD,
    INT_GOOD,
    Row,
    bad_value_rows,
    good_value_rows,
    run_row,
    to_params,
)
from arch.unitroot.cointegration import (
    CanonicalCointegratingReg,
    DynamicOLS,
    FullyModifiedOLS,
    engle_granger,
    phillips_ouliaris,
)

NOBS = 120
RNG = np.random.default_rng(20261017)
X = np.cumsum(RNG.standard_normal((NOBS, 2)), 0)
Y = X @ np.array([1.0, 0.5]) + RNG.standard_normal(NOBS)
X3 = np.column_stack([X, np.cumsum(RNG.standard_normal(NOBS))])

TREND_TEXT = {
    "n": "No Trend",
    "c": "Constant",
    "ct": "Constant and Linear Time Trend",
    "ctt": "Constant, Linear and Quadratic Time Trends",
}


def _eg(**kw):
    return engle_granger(**kw)


def _po(**kw):
    return phillips_ouliaris(**kw)


def _dols(**kw):
    return DynamicOLS(**kw)


def _fm(**kw):
    return FullyModifiedOLS(**kw)


def _ccr(**kw):
    return CanonicalCointegratingReg(**kw)


# The callables that take y, x and trend
DATA_TARGETS = {
    "engle_granger": _eg,
    "phillips_ouliaris": _po,
    "DynamicOLS": _dols,
    "FullyModifiedOLS": _fm,
    "CanonicalCointegratingReg": _ccr,
}
BASE = {"y": Y, "x": X}


def _with(array, position, value):
    out = np.array(array, dtype=float)
    out[position] = value
    return out


# (label, value, exception) of data that is not accepted
Y_BAD = [
    ("str", "abc", TypeError),
    ("list-of-str", ["a"] * NOBS, TypeError),
    ("none", None, TypeError),
    ("complex", np.ones(NOBS, dtype=complex), TypeError),
    ("2d", np.ones((NOBS, 2)), ValueError),
    ("3d", np.ones((NOBS, 1, 2)), ValueError),
    ("nan", _with(Y, 3, np.nan), ValueError),
    ("inf", _with(Y, 3, np.inf), ValueError),
    ("-inf", _with(Y, 3, -np.inf), ValueError),
    ("series-nan", pd.Series(_with(Y, 0, np.nan)), ValueError),
]
X_BAD = [
    ("str", "abc", TypeError),
    ("none", None, TypeError),
    ("complex", np.ones((NOBS, 2), dtype=complex), TypeError),
    ("3d", np.ones((NOBS, 2, 2)), ValueError),
    ("nan", _with(X, (3, 1), np.nan), ValueError),
    ("inf", _with(X, (3, 0), np.inf), ValueError),
    ("-inf", _with(X, (3, 0), -np.inf), ValueError),
    ("frame-nan", pd.DataFrame(_with(X, (0, 0), np.nan)), ValueError),
    ("no-columns", np.empty((NOBS, 0)), ValueError),
]


def _data_rows():
    rows = []
    for target, func in DATA_TARGETS.items():
        for name, bad in (("y", Y_BAD), ("x", X_BAD)):
            for label, value, exc in bad:
                rows.append(
                    Row(
                        target,
                        func,
                        {**BASE, name: value},
                        exc,
                        rf"^{name} must be ",
                        id=f"{target}-{name}-{label}",
                        name=name,
                    )
                )
    return rows


DATA_ROWS = _data_rows()


@pytest.mark.parametrize("row", to_params(DATA_ROWS))
def test_data(row):
    run_row(row)


MISMATCH_ROWS = [
    Row(
        target,
        func,
        {**BASE, name: value},
        ValueError,
        r"^The number of observations in y and x differ\. y has \d+ observations",
        id=f"{target}-mismatch-{label}",
        form=False,
    )
    for target, func in DATA_TARGETS.items()
    for label, name, value in (
        ("short-y", "y", Y[:-1]),
        ("short-x", "x", X[:-1]),
        ("empty-y", "y", np.empty(0)),
    )
]


@pytest.mark.parametrize("row", to_params(MISMATCH_ROWS))
def test_mismatched_rows(row):
    run_row(row)


INDEX_ROWS = [
    Row(
        target,
        func,
        {
            "y": pd.Series(
                Y, index=pd.date_range("2000-01-31", periods=NOBS, freq=MONTH_END)
            ),
            "x": pd.DataFrame(X),
        },
        ValueError,
        r"^x must be array_like with the same index as y; got ",
        id=f"{target}-index",
        name="x",
    )
    for target, func in DATA_TARGETS.items()
]


@pytest.mark.parametrize("row", to_params(INDEX_ROWS))
def test_mismatched_index(row):
    run_row(row)


# The regression has num_x + the number of terms of the trend parameters and the
# estimators need two more rows
@pytest.mark.parametrize(
    ("trend", "num_x", "min_rows"),
    [("n", 2, 4), ("c", 2, 5), ("ct", 2, 6), ("ctt", 2, 7), ("c", 1, 4), ("c", 3, 6)],
)
@pytest.mark.parametrize("target", list(DATA_TARGETS))
def test_too_few_observations(target, trend, num_x, min_rows):
    nobs = min_rows - 1
    x = X3[:nobs, :num_x]
    y = Y[:nobs]
    match = rf"^x must be a 2-d array of finite numbers with at least {min_rows} rows"
    with pytest.raises(ValueError, match=match):
        DATA_TARGETS[target](y=y, x=x, trend=trend)


@pytest.mark.parametrize(("trend", "num_x", "min_rows"), [("n", 2, 4), ("ctt", 3, 8)])
def test_minimum_observations(trend, num_x, min_rows):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((min_rows, num_x))
    y = rng.standard_normal(min_rows)
    FullyModifiedOLS(y, x, trend)
    CanonicalCointegratingReg(y, x, trend)


DATES = pd.date_range("2000-01-31", periods=NOBS, freq=MONTH_END)
Y_GOOD = [
    pd.Series(Y, index=DATES, name="target"),
    Y.tolist(),
    Y[:, None],
    pd.DataFrame(Y, index=DATES),
    np.round(Y).astype(np.int64),
    Y.astype(np.float32),
    pd.Series(Y).astype("Float64"),
]
X_GOOD = [
    pd.DataFrame(X, columns=["a", "b"]),
    X.tolist(),
    X[:, 0],
    pd.Series(X[:, 0], name="a"),
    np.round(X).astype(np.int64),
    X.astype(np.float32),
    X[:, :1],
]


def _good_data_rows():
    rows = []
    for target, func in DATA_TARGETS.items():
        rows += good_value_rows(f"{target}-y", func, BASE, "y", Y_GOOD)
        rows += good_value_rows(f"{target}-x", func, BASE, "x", X_GOOD)
    return rows


@pytest.mark.parametrize("row", to_params(_good_data_rows()))
def test_good_data(row):
    run_row(row)


def _no_none(bad):
    return [(value, exc) for value, exc in bad if value is not None]


INT_NONE_BAD = _no_none(INT_BAD)
BOOL_NONE_BAD = _no_none(BOOL_BAD)
FLOAT_NONE_BAD = _no_none(FLOAT_BAD)
LAGS_RULE = r"an integer >= 0 or None"
BOOL_RULE = r"True or False( or None)?"

EG_ROWS = [
    *bad_value_rows("eg", _eg, BASE, "lags", INT_NONE_BAD, rule=LAGS_RULE),
    Row(
        "eg",
        _eg,
        {**BASE, "lags": -1},
        ValueError,
        rf"^lags must be {LAGS_RULE}; got -1",
        id="eg-lags-negative",
        name="lags",
    ),
    *bad_value_rows(
        "eg",
        _eg,
        BASE,
        "max_lags",
        INT_NONE_BAD,
        rule=rf"an integer in \[0, {NOBS - 2}\] or None",
    ),
    Row(
        "eg",
        _eg,
        {**BASE, "max_lags": -1},
        ValueError,
        rf"^max_lags must be an integer in \[0, {NOBS - 2}\] or None; got -1",
        id="eg-max_lags-negative",
        name="max_lags",
    ),
    Row(
        "eg",
        _eg,
        {**BASE, "max_lags": NOBS - 1},
        ValueError,
        rf"^max_lags must be an integer in \[0, {NOBS - 2}\] or None; got {NOBS - 1}",
        id="eg-max_lags-large",
        name="max_lags",
    ),
    *bad_value_rows("eg", _eg, BASE, "low_memory", BOOL_NONE_BAD, rule=BOOL_RULE),
    *good_value_rows("eg", _eg, BASE, "lags", [None, 0, *INT_GOOD]),
    *good_value_rows("eg", _eg, BASE, "max_lags", [None, 0, *INT_GOOD]),
    *good_value_rows("eg", _eg, BASE, "low_memory", [None, *BOOL_GOOD]),
]


@pytest.mark.parametrize("row", to_params(EG_ROWS))
def test_engle_granger(row):
    run_row(row)


BANDWIDTH_RULE = r"a finite float >= 0 or None"
BANDWIDTH_BAD = [
    *FLOAT_NONE_BAD,
    (-1, ValueError),
    (-0.5, ValueError),
    (np.float64(-1.0), ValueError),
    ([3], TypeError),
]
BANDWIDTH_GOOD = [None, 0, 0.0, 3, 3.5, np.int64(3), np.float32(2.5), np.float64(3.0)]
KERNEL_FIT_BASE = {"y": Y, "x": X}


def _po_rows(func, base, prefix):
    return [
        *bad_value_rows(
            prefix, func, base, "bandwidth", BANDWIDTH_BAD, rule=BANDWIDTH_RULE
        ),
        *bad_value_rows(prefix, func, base, "force_int", BOOL_BAD, rule=BOOL_RULE),
        *good_value_rows(prefix, func, base, "bandwidth", BANDWIDTH_GOOD),
        *good_value_rows(prefix, func, base, "force_int", BOOL_GOOD),
    ]


PO_ROWS = [
    *_po_rows(_po, {**BASE, "test_type": "Zt"}, "po"),
    Row(
        "po",
        _po,
        {**BASE, "x": np.ones((NOBS, 13)) * RNG.standard_normal((NOBS, 13))},
        ValueError,
        r"^The number of stochastic trends must be between 2 and 12",
        id="po-too-many-columns",
        form=False,
    ),
]


@pytest.mark.parametrize("row", to_params(PO_ROWS))
def test_phillips_ouliaris_arguments(row):
    run_row(row)


DOLS_BASE = {**BASE, "lags": 1, "leads": 1}
ORDER_RULE = rf"an integer in \[0, {NOBS - 2}\] or None"


def _order_rows(name):
    return [
        *bad_value_rows("dols", _dols, DOLS_BASE, name, INT_NONE_BAD, rule=ORDER_RULE),
        Row(
            "dols",
            _dols,
            {**DOLS_BASE, name: -1},
            ValueError,
            rf"^{name} must be {ORDER_RULE}; got -1",
            id=f"dols-{name}-negative",
            name=name,
        ),
        Row(
            "dols",
            _dols,
            {**DOLS_BASE, name: NOBS - 1},
            ValueError,
            rf"^{name} must be {ORDER_RULE}; got {NOBS - 1}",
            id=f"dols-{name}-large",
            name=name,
        ),
        *good_value_rows("dols", _dols, DOLS_BASE, name, [None, 0, *INT_GOOD]),
    ]


DOLS_ROWS = [
    *_order_rows("lags"),
    *_order_rows("leads"),
    *_order_rows("max_lag"),
    *_order_rows("max_lead"),
    *bad_value_rows(
        "dols", _dols, DOLS_BASE, "common", BOOL_BAD, rule=r"True or False"
    ),
    *good_value_rows(
        "dols", _dols, {**DOLS_BASE, "lags": 2, "leads": 2}, "common", BOOL_GOOD
    ),
    Row(
        "dols",
        _dols,
        {**DOLS_BASE, "lags": 1, "leads": 2, "common": True},
        ValueError,
        r"^common is specified but leads and lags have different values",
        id="dols-common-lags-leads",
        form=False,
    ),
    Row(
        "dols",
        _dols,
        {**BASE, "max_lag": 3, "max_lead": 4, "common": True},
        ValueError,
        r"^common is specified but max_lead and max_lag have different values",
        id="dols-common-max",
        form=False,
    ),
]

# Too many lags and leads for the sample: the message names all four arguments
TOO_LARGE = r"^lags, leads, max_lag and max_lead are too large for the amount of data"
DOLS_SIZE_ROWS = [
    Row(
        "dols",
        _dols,
        {"y": Y[:20], "x": X[:20]},
        ValueError,
        TOO_LARGE,
        id="dols-default-20-observations",
        form=False,
    ),
    Row(
        "dols",
        _dols,
        {**BASE, "lags": 60, "leads": 59},
        ValueError,
        TOO_LARGE,
        id="dols-lags-leads-exceed-sample",
        form=False,
    ),
    Row(
        "dols",
        _dols,
        {"y": Y[:40], "x": X[:40], "lags": 6, "leads": 6},
        ValueError,
        TOO_LARGE,
        id="dols-more-parameters-than-rows",
        form=False,
    ),
    Row(
        "dols",
        _dols,
        {"y": Y[:40], "x": X[:40], "lags": 6, "leads": 5},
        None,
        id="dols-fewest-rows-ok",
    ),
    Row(
        "dols",
        _dols,
        {"y": Y[:40], "x": X[:40], "lags": 3, "leads": 3, "max_lag": 4},
        None,
        id="dols-lags-before-max_lag-ok",
    ),
]


@pytest.mark.parametrize("row", to_params(DOLS_ROWS + DOLS_SIZE_ROWS))
def test_dynamic_ols(row):
    run_row(row)


def _dols_fit(**kw):
    return DynamicOLS(Y, X, lags=1, leads=1).fit(**kw)


FIT_BASE = {"kernel": "bartlett", "bandwidth": 3}
DOLS_FIT_ROWS = [
    *bad_value_rows(
        "dols-fit", _dols_fit, FIT_BASE, "bandwidth", BANDWIDTH_BAD, rule=BANDWIDTH_RULE
    ),
    *bad_value_rows(
        "dols-fit", _dols_fit, FIT_BASE, "force_int", BOOL_BAD, rule=r"True or False"
    ),
    *bad_value_rows(
        "dols-fit", _dols_fit, FIT_BASE, "df_adjust", BOOL_BAD, rule=r"True or False"
    ),
    *good_value_rows("dols-fit", _dols_fit, FIT_BASE, "bandwidth", BANDWIDTH_GOOD),
    *good_value_rows("dols-fit", _dols_fit, FIT_BASE, "force_int", BOOL_GOOD),
    *good_value_rows("dols-fit", _dols_fit, FIT_BASE, "df_adjust", BOOL_GOOD),
]


@pytest.mark.parametrize("row", to_params(DOLS_FIT_ROWS))
def test_dynamic_ols_fit(row):
    run_row(row)


X_TREND_RULE = r"one of 'c', 'ct' or 'ctt'"
X_TREND_BAD = [
    ("zzz", ValueError),
    ("", ValueError),
    (5, TypeError),
    (["c"], TypeError),
    (np.str_("zzz"), ValueError),
]
X_TREND_GOOD = [None, "c", "ct", "ctt", "CT", np.str_("ct"), "n", "t"]


def _fm_fit(cls):
    def func(**kw):
        return cls(Y, X).fit(**kw)

    return func


def _fm_rows(name, ctor, fit):
    return [
        *bad_value_rows(
            f"{name}", ctor, BASE, "x_trend", X_TREND_BAD, rule=X_TREND_RULE
        ),
        *good_value_rows(f"{name}", ctor, BASE, "x_trend", X_TREND_GOOD),
        *bad_value_rows(
            f"{name}-fit",
            fit,
            FIT_BASE,
            "bandwidth",
            BANDWIDTH_BAD,
            rule=BANDWIDTH_RULE,
        ),
        *[
            row
            for flag in ("force_int", "diff", "df_adjust")
            for row in (
                *bad_value_rows(
                    f"{name}-fit", fit, FIT_BASE, flag, BOOL_BAD, rule=r"True or False"
                ),
                *good_value_rows(f"{name}-fit", fit, FIT_BASE, flag, BOOL_GOOD),
            )
        ],
        *good_value_rows(f"{name}-fit", fit, FIT_BASE, "bandwidth", BANDWIDTH_GOOD),
    ]


FM_ROWS = [
    *_fm_rows("fmols", _fm, _fm_fit(FullyModifiedOLS)),
    *_fm_rows("ccr", _ccr, _fm_fit(CanonicalCointegratingReg)),
]


@pytest.mark.parametrize("row", to_params(FM_ROWS))
def test_fmols_ccr(row):
    run_row(row)


RES_EG = engle_granger(Y, X)
RES_PO = phillips_ouliaris(Y, X)
RES_DOLS = DynamicOLS(Y, X, lags=1, leads=1).fit()
STR_BAD = [(5, TypeError), (["t"], TypeError), (True, TypeError), (b"t", TypeError)]


class NewAxes:
    """Marks a row that plots in axes that the test creates"""


def _plot(res):
    def func(**kw):
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if isinstance(kw.get("axes"), NewAxes):
            _, kw["axes"] = plt.subplots(1, 1)
        fig = res.plot(**kw)
        plt.close(fig)

    return func


def _summary(**kw):
    return RES_DOLS.summary(**kw)


def _plot_rows():
    rows = []
    for name, res in (("eg", RES_EG), ("po", RES_PO)):
        func = _plot(res)
        rows += [
            *bad_value_rows(
                f"plot-{name}",
                func,
                {},
                "axes",
                [("ax", TypeError), (5, TypeError), ([], TypeError)],
                rule=r"an instance of Axes or None",
            ),
            *bad_value_rows(
                f"plot-{name}", func, {}, "title", STR_BAD, rule=r"a string or None"
            ),
            *good_value_rows(f"plot-{name}", func, {}, "axes", [None, NewAxes()]),
            *good_value_rows(f"plot-{name}", func, {}, "title", [None, "Title", ""]),
        ]
    return rows


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
@pytest.mark.usefixtures("agg_backend")
@pytest.mark.parametrize("row", to_params(_plot_rows()))
def test_plot(row):
    run_row(row)


SUMMARY_ROWS = [
    *bad_value_rows("summary", _summary, {}, "full", BOOL_BAD, rule=r"True or False"),
    *good_value_rows("summary", _summary, {}, "full", BOOL_GOOD),
]


@pytest.mark.parametrize("row", to_params(SUMMARY_ROWS))
def test_summary(row):
    run_row(row)
