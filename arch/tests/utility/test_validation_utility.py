"""Input validation of arch.utility: array, timeseries, io and testing."""

from __future__ import annotations

import datetime as dt

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest

from arch.tests.validation_table import (
    BOOL_BAD,
    BOOL_GOOD,
    INT_BAD,
    INT_GOOD,
    Row,
    bad_value_rows,
    good_value_rows,
    run_row,
    to_params,
)
from arch.utility.array import (
    as_float64,
    cutoff_to_index,
    date_to_index,
    ensure1d,
    ensure2d,
    find_index,
    parse_dataframe,
    to_array_1d,
)
from arch.utility.io import pval_format, str_format
from arch.utility.testing import WaldTestStatistic
from arch.utility.timeseries import add_trend


@pytest.mark.parametrize(
    ("trend", "expected_columns"),
    [
        ("c", ["a", "b"]),
        ("ct", ["a", "b", "trend"]),
        ("ctt", ["a", "b", "trend", "quadratic_trend"]),
        ("t", ["a", "b", "trend"]),
    ],
)
def test_add_trend_frame_skip_constant(trend, expected_columns):
    # x has a constant column, so "skip" drops the constant of the trend. The
    # DataFrame branch used to name the dropped constant and fail on the shape.
    x = pd.DataFrame({"a": np.ones(4), "b": [1.0, 2.0, 4.0, 3.0]})
    result = add_trend(x, trend, has_constant="skip")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == expected_columns
    # Hand computed: the trend columns are 1, ..., 4 and their squares
    columns = {
        "trend": np.arange(1.0, 5.0),
        "quadratic_trend": np.arange(1.0, 5.0) ** 2,
    }
    for name in expected_columns[2:]:
        assert_array_equal(result[name].to_numpy(), columns[name])
    assert_array_equal(result.to_numpy(), add_trend(x.to_numpy(), trend))
    assert result.index.equals(x.index)


NOBS = 12
IDX = pd.RangeIndex(NOBS)
DATES = pd.date_range("2020-01-01", periods=NOBS, freq="D")
SERIES = pd.Series(np.arange(float(NOBS)), index=DATES)
X2 = np.arange(24.0).reshape((12, 2))


STR_BAD = [
    (1, TypeError),
    (None, TypeError),
    (["a"], TypeError),
    (2.5, TypeError),
    (b"a", TypeError),
]
STR_GOOD = ["", "a", np.str_("a")]


# ---------------------------------------------------------------- add_trend
def _add_trend(**kw):
    return add_trend(**kw)


ADD_TREND_BASE = {"x": X2, "trend": "ct"}
ADD_TREND_NOBS = {"nobs": 5, "trend": "ct"}
ADD_TREND_ROWS = [
    *bad_value_rows("add_trend", _add_trend, ADD_TREND_BASE, "prepend", BOOL_BAD),
    *good_value_rows("add_trend", _add_trend, ADD_TREND_BASE, "prepend", BOOL_GOOD),
    *bad_value_rows(
        "add_trend",
        _add_trend,
        ADD_TREND_NOBS,
        "nobs",
        [*[(v, e) for v, e in INT_BAD if v is not None], (0, ValueError)]
        + [(-1, ValueError), (np.int64(-3), ValueError)],
        rule=r"an integer >= 1 or None",
    ),
    *good_value_rows("add_trend", _add_trend, ADD_TREND_NOBS, "nobs", INT_GOOD),
    Row(
        "add_trend",
        _add_trend,
        {"x": X2, "nobs": 5},
        ValueError,
        r"One and only one of x or nobs",
        id="add_trend-x-and-nobs",
    ),
    Row(
        "add_trend",
        _add_trend,
        {"x": None, "nobs": None},
        ValueError,
        r"One and only one of x or nobs",
        id="add_trend-no-x-no-nobs",
    ),
    Row(
        "add_trend",
        _add_trend,
        {"x": X2, "nobs": 2.5},
        TypeError,
        r"^nobs must be an integer",
        id="add_trend-nobs-float-with-x",
        name="nobs",
    ),
    Row(
        "add_trend",
        _add_trend,
        {"x": [["a", "b"]], "trend": "c"},
        TypeError,
        r"x must",
        id="add_trend-x-str",
        form=False,
    ),
]

# ---------------------------------------------------------------- io
IO_STR = [
    ("0.5", TypeError),
    (True, TypeError),
    (np.True_, TypeError),
    (None, TypeError),
    (1j, TypeError),
    ([0.5], TypeError),
]
IO_ROWS = [
    *bad_value_rows("str_format", str_format, {"v": 1.0}, "v", IO_STR),
    *bad_value_rows(
        "pval_format",
        pval_format,
        {"v": 0.5},
        "v",
        [
            *IO_STR,
            (np.inf, ValueError),
            (-np.inf, ValueError),
            (-0.1, ValueError),
            (1.1, ValueError),
            (np.float64(2.0), ValueError),
        ],
    ),
    *good_value_rows(
        "str_format",
        str_format,
        {},
        "v",
        [0, 1, 0.5, np.float32(0.5), np.int64(3), np.nan, np.inf, -np.inf, -2.5],
    ),
    *good_value_rows(
        "pval_format",
        pval_format,
        {},
        "v",
        [0, 1, 0.0, 1.0, 0.5, np.float32(0.5), np.int8(1), np.nan],
    ),
]

# ---------------------------------------------------------------- Wald test
WALD_BASE = {"stat": 3.0, "df": 2, "null": "H0", "alternative": "H1"}
WALD_STAT_BAD = [
    ("0.5", TypeError),
    (True, TypeError),
    (np.True_, TypeError),
    (None, TypeError),
    (1j, TypeError),
    ([0.5], TypeError),
    (-1.0, ValueError),
    (-np.inf, ValueError),
    (np.float64(-1e-3), ValueError),
]
WALD_ROWS = [
    *bad_value_rows("Wald", WaldTestStatistic, WALD_BASE, "stat", WALD_STAT_BAD),
    *good_value_rows(
        "Wald",
        WaldTestStatistic,
        WALD_BASE,
        "stat",
        [0, 0.0, 3, np.int64(3), np.float32(2.5), np.float64(1.0), np.nan, np.inf],
    ),
    *bad_value_rows(
        "Wald",
        WaldTestStatistic,
        WALD_BASE,
        "df",
        [*INT_BAD, (0, ValueError), (-1, ValueError), (np.int64(-4), ValueError)],
    ),
    *good_value_rows("Wald", WaldTestStatistic, WALD_BASE, "df", [1, *INT_GOOD]),
]
for _name in ("null", "alternative", "name"):
    WALD_ROWS.extend(
        bad_value_rows("Wald", WaldTestStatistic, WALD_BASE, _name, STR_BAD)
    )
    WALD_ROWS.extend(
        good_value_rows("Wald", WaldTestStatistic, WALD_BASE, _name, STR_GOOD)
    )

# ---------------------------------------------------------------- array
ENSURE1D_BASE = {"x": np.arange(3.0), "name": "y"}
ENSURE2D_BASE = {"x": np.arange(3.0), "name": "y"}
ARRAY_ROWS = [
    *bad_value_rows(
        "ensure1d",
        ensure1d,
        ENSURE1D_BASE,
        "series",
        BOOL_BAD,
    ),
    *good_value_rows(
        "ensure1d",
        ensure1d,
        ENSURE1D_BASE,
        "series",
        BOOL_GOOD,
    ),
    *bad_value_rows(
        "ensure1d",
        ensure1d,
        {**ENSURE1D_BASE, "series": True},
        "name",
        [([1], TypeError), ({"a": 1}, TypeError)],
    ),
    *good_value_rows(
        "ensure1d",
        ensure1d,
        {**ENSURE1D_BASE, "series": True},
        "name",
        [None, "a", 3, ("a", "b")],
    ),
    *bad_value_rows(
        "ensure2d",
        ensure2d,
        ENSURE2D_BASE,
        "name",
        STR_BAD,
    ),
    *good_value_rows("ensure2d", ensure2d, ENSURE2D_BASE, "name", STR_GOOD),
    *bad_value_rows(
        "parse_dataframe",
        parse_dataframe,
        {"x": X2, "name": ["a", "b"]},
        "name",
        [(1, TypeError), (None, TypeError), (("a", "b"), TypeError)],
    ),
    *bad_value_rows(
        "parse_dataframe_none",
        parse_dataframe,
        {"x": None, "name": "a"},
        "name",
        [(1, TypeError), (None, TypeError), (["a"], TypeError)],
    ),
    *good_value_rows(
        "parse_dataframe",
        parse_dataframe,
        {"x": X2},
        "name",
        ["a", ["a", "b"]],
    ),
    *bad_value_rows(
        "to_array_1d",
        to_array_1d,
        {"x": np.arange(3.0)},
        "x",
        [([1.0, 2.0], TypeError), (None, TypeError), (1.0, TypeError)],
    ),
    Row(
        "to_array_1d",
        to_array_1d,
        {"x": np.ones((2, 2))},
        ValueError,
        r"^x must be 1D or 1D convertible; got shape \(2, 2\) \(ndarray\)$",
        id="to_array_1d-2d",
        name="x",
    ),
]


@pytest.mark.parametrize("func", [as_float64, ensure1d, ensure2d])
@pytest.mark.parametrize(
    "value",
    [None, [1.0, None], (None,), np.array([1.0, None], dtype=object), [[None]]],
    ids=["None", "list-with-None", "tuple-None", "object-with-None", "nested-None"],
)
def test_none_is_not_nan(func, value):
    # np.asarray(None) is an object array and astype(float) turns None into NaN
    with pytest.raises(TypeError, match=r"^x must .*None"):
        func(value, "x")


@pytest.mark.parametrize("func", [as_float64, ensure1d, ensure2d])
def test_nan_is_still_data(func):
    # NaN is a number and missing values in float data are NaN
    out = func(np.array([1.0, np.nan]), "x")
    assert np.isnan(out).sum() == 1
    out = func(np.array([1.0, None], dtype=float), "x")
    assert np.isnan(out).sum() == 1


# ------------------------------------------------------ cutoff_to_index etc.
def _cutoff(**kw):
    return cutoff_to_index(**kw)


CUT_BASE = {"cutoff": 3, "index": IDX, "default": 0}
CUT_INT_BAD = [
    *[(v, ValueError if v == "2" else e) for v, e in INT_BAD if v is not None],
    (-NOBS - 1, ValueError),
    (np.int64(-1000), ValueError),
    (NOBS + 1, ValueError),
    (np.int32(10**6), ValueError),
]
CUT_ROWS = [
    *bad_value_rows(
        "cutoff_to_index",
        _cutoff,
        CUT_BASE,
        "cutoff",
        [(v, e) for v, e in CUT_INT_BAD if e is ValueError],
    ),
    # The type error keeps the word "cutoffs", which fit and forecast report
    *[
        Row(
            "cutoff_to_index",
            _cutoff,
            {**CUT_BASE, "cutoff": value},
            TypeError,
            r"^cutoffs must be None, an integer, a string, a date",
            id=f"cutoff_to_index-cutoff-type-{i}",
            name="cutoffs",
        )
        for i, (value, e) in enumerate(CUT_INT_BAD)
        if e is TypeError
    ],
    *good_value_rows(
        "cutoff_to_index",
        _cutoff,
        CUT_BASE,
        "cutoff",
        [0, NOBS, None, np.int64(NOBS), np.uint8(5), -1, -NOBS, np.int8(-3), *INT_GOOD],
    ),
    *bad_value_rows(
        "cutoff_to_index",
        _cutoff,
        CUT_BASE,
        "default",
        [
            *INT_BAD,
            (-1, ValueError),
            (NOBS + 1, ValueError),
        ],
    ),
    *good_value_rows(
        "cutoff_to_index", _cutoff, CUT_BASE, "default", [0, NOBS, *INT_GOOD]
    ),
    *bad_value_rows(
        "cutoff_to_index",
        _cutoff,
        CUT_BASE,
        "index",
        [(None, TypeError), ([1, 2], TypeError), (np.arange(3), TypeError)],
    ),
]
for _i, _date in enumerate(
    [
        "2019-12-31",
        "2020-01-13",
        pd.Timestamp("2019-01-01"),
        np.datetime64("2021-01-01"),
        dt.datetime(2019, 6, 1),
        dt.date(2025, 1, 1),
        pd.Period("2020-02", freq="M"),
        "not a date",
    ]
):
    CUT_ROWS.append(
        Row(
            "cutoff_to_index",
            _cutoff,
            {"cutoff": _date, "index": DATES, "default": 0},
            ValueError,
            r"^cutoff must be .*; got ",
            id=f"cutoff_to_index-outside-{_i}",
            name="cutoff",
        )
    )
CUT_ROWS += [
    Row(
        "cutoff_to_index",
        _cutoff,
        {"cutoff": "2020-01-05", "index": IDX, "default": 0},
        ValueError,
        r"^cutoff must be an integer when the data does not have a DatetimeIndex",
        id="cutoff_to_index-date-no-datetime-index",
        name="cutoff",
    ),
    Row(
        "cutoff_to_index",
        _cutoff,
        {"cutoff": "2020-01-05", "index": pd.DatetimeIndex([]), "default": 0},
        ValueError,
        r"^cutoff must be a date within the index",
        id="cutoff_to_index-empty-index",
        name="cutoff",
    ),
]
for _i, _date in enumerate(
    [
        "2020-01-01",
        "2020-01-12",
        "2020-01-05 12:00",
        pd.Timestamp("2020-01-03"),
        np.datetime64("2020-01-04"),
        dt.datetime(2020, 1, 6),
        dt.date(2020, 1, 7),
        pd.Period("2020-01", freq="M"),
    ]
):
    CUT_ROWS.append(
        Row(
            "cutoff_to_index",
            _cutoff,
            {"cutoff": _date, "index": DATES, "default": 0},
            None,
            id=f"cutoff_to_index-inside-{_i}",
            name="cutoff",
        )
    )
CUT_ROWS.append(
    Row(
        "cutoff_to_index",
        _cutoff,
        {
            "cutoff": pd.Period("2020-01-03", freq="D"),
            "index": DATES.to_period("D"),
            "default": 0,
        },
        None,
        id="cutoff_to_index-period-index",
        name="cutoff",
    )
)


def _date_to_index(**kw):
    return date_to_index(**kw)


DATE_BASE = {"date": "2020-01-05", "date_index": DATES}
DATE_ROWS = [
    *bad_value_rows(
        "date_to_index",
        _date_to_index,
        DATE_BASE,
        "date",
        [
            (3, TypeError),
            (None, TypeError),
            ([1], TypeError),
            (dt.date(2020, 1, 5), TypeError),
            (2.5, TypeError),
            ("not a date", ValueError),
            ("NaT", ValueError),
            (np.datetime64("NaT", "D"), ValueError),
        ],
    ),
    *bad_value_rows(
        "date_to_index",
        _date_to_index,
        DATE_BASE,
        "date_index",
        [
            (None, TypeError),
            (np.arange(5), TypeError),
            (IDX, TypeError),
            (["2020-01-01"], TypeError),
            (DATES[::-1], ValueError),
            (DATES.repeat(2), ValueError),
            (DATES[:0], ValueError),
        ],
    ),
    *good_value_rows(
        "date_to_index",
        _date_to_index,
        DATE_BASE,
        "date",
        [
            "2020-01-05",
            pd.Timestamp("2020-01-05"),
            dt.datetime(2020, 1, 5),
            np.datetime64("2020-01-05"),
        ],
    ),
    *good_value_rows(
        "date_to_index",
        _date_to_index,
        DATE_BASE,
        "date_index",
        [DATES, DATES.to_numpy(), pd.Series(DATES)],
    ),
]


def _find_index(**kw):
    return find_index(**kw)


FIND_BASE = {"s": SERIES, "index": 3}
FIND_ROWS = [
    *bad_value_rows(
        "find_index",
        _find_index,
        FIND_BASE,
        "index",
        [
            (2.5, TypeError),
            (True, TypeError),
            (np.True_, TypeError),
            (None, TypeError),
            ([3], TypeError),
            (-NOBS - 1, ValueError),
            (NOBS + 1, ValueError),
            ("not a date", ValueError),
            ("2021-01-01", ValueError),
        ],
    ),
    *good_value_rows(
        "find_index",
        _find_index,
        FIND_BASE,
        "index",
        [
            0,
            NOBS,
            np.int64(4),
            -1,
            -NOBS,
            np.int8(-3),
            "2020-01-03",
            pd.Timestamp("2020-01-04"),
        ],
    ),
    *bad_value_rows(
        "find_index",
        _find_index,
        FIND_BASE,
        "s",
        [(None, TypeError), (np.arange(3), TypeError), ([1, 2], TypeError)],
    ),
]

ROWS = [
    *ADD_TREND_ROWS,
    *IO_ROWS,
    *WALD_ROWS,
    *ARRAY_ROWS,
    *CUT_ROWS,
    *DATE_ROWS,
    *FIND_ROWS,
]


@pytest.mark.parametrize("row", to_params(ROWS))
def test_validation(row):
    run_row(row)


def test_cutoff_to_index_values():
    # Positions and dates are found by hand: 2020-01-01 is position 0
    assert cutoff_to_index(None, DATES, 7) == 7
    assert cutoff_to_index(NOBS, DATES, 0) == NOBS
    assert cutoff_to_index(np.int32(4), DATES, 0) == 4
    assert cutoff_to_index("2020-01-05", DATES, 0) == 4
    assert cutoff_to_index(pd.Timestamp("2020-01-12"), DATES, 0) == NOBS - 1
    assert cutoff_to_index(pd.Period("2020-01-03", freq="D"), DATES, 0) == 2
    # Between two dates: the location where it would be inserted
    assert cutoff_to_index("2020-01-05 12:00", DATES, 0) == 5


def test_wald_values():
    wald = WaldTestStatistic(np.int64(3), np.int32(2), "a", "b", name="t")
    assert wald.df == 2
    assert isinstance(wald.df, int)
    assert wald.dist_name == "chi2(2)"
    # The chi2(2) survival function is exp(-x / 2)
    assert_allclose(wald.pval, np.exp(-1.5), rtol=1e-12)
    assert np.isnan(WaldTestStatistic(np.nan, 1, "a", "b").pval)
    assert WaldTestStatistic(np.inf, 1, "a", "b").pval == 0.0


def test_formats():
    assert str_format(np.nan) == "        "
    assert str_format(np.inf) == "     inf"
    assert str_format(-np.inf) == "    -inf"
    assert str_format(1.5) == "1.5000"
    assert pval_format(np.nan) == "        "
    assert pval_format(0.05) == "0.0500"
    assert pval_format(1) == "1.0000"


def test_add_trend_values():
    # nobs given: the trend is 1, ..., nobs
    out = add_trend(nobs=np.int64(4), trend="ct")
    assert_array_equal(out, np.column_stack([np.ones(4), np.arange(1.0, 5.0)]))
    assert add_trend(nobs=3, trend="n").shape == (3, 0)
    out = add_trend(X2, "t", prepend=np.True_)
    assert_array_equal(out[:, 0], np.arange(1.0, 13.0))
