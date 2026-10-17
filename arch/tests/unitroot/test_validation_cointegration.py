"""
Input validation of the cointegration tests and estimators.

The table-driven rows are built on arch.tests.validation_table.
"""

import numpy as np
import pandas as pd
import pytest

from arch.tests.validation_table import (
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
        {"y": pd.Series(Y, index=pd.date_range("2000-01-31", periods=NOBS, freq="ME")),
         "x": pd.DataFrame(X)},
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
    "trend,num_x,min_rows",
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


@pytest.mark.parametrize("trend,num_x,min_rows", [("n", 2, 4), ("ctt", 3, 8)])
def test_minimum_observations(trend, num_x, min_rows):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((min_rows, num_x))
    y = rng.standard_normal(min_rows)
    FullyModifiedOLS(y, x, trend)
    CanonicalCointegratingReg(y, x, trend)


DATES = pd.date_range("2000-01-31", periods=NOBS, freq="ME")
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
