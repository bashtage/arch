"""
Table-driven tests of the input validation of the covariance estimators and cov_nw.

Each row is a call and the error that it must raise, or None for a valid call.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from arch.covariance.kernel import (
    Andrews,
    Bartlett,
    CovarianceEstimator,
    Gallant,
    NeweyWest,
    Parzen,
    ParzenCauchy,
    ParzenGeometric,
    ParzenRiesz,
    QuadraticSpectral,
    TukeyHamming,
    TukeyHanning,
    TukeyParzen,
    ZeroLag,
)
from arch.covariance.var import PreWhitenedRecolored
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
from arch.utility.cov import cov_nw

NOBS = 100
RNG = np.random.default_rng(20241017)
X = RNG.standard_normal((NOBS, 2))
X_LONG = np.random.default_rng(1).standard_normal((250, 2))

KERNELS: list[type[CovarianceEstimator]] = [
    Andrews,
    Bartlett,
    Gallant,
    NeweyWest,
    Parzen,
    ParzenCauchy,
    ParzenGeometric,
    ParzenRiesz,
    QuadraticSpectral,
    TukeyHamming,
    TukeyHanning,
    TukeyParzen,
    ZeroLag,
]


def _kernel(cls: type[CovarianceEstimator]):
    def func(**kwargs: Any) -> Any:
        # The covariance is computed to check that the arguments are usable
        return cls(**kwargs).cov.long_run

    return func


def _pwrc(**kwargs: Any) -> Any:
    return PreWhitenedRecolored(**kwargs).cov.long_run


def _pwrc_resid(**kwargs: Any) -> Any:
    # The VAR with the largest order is not covariance stationary
    return PreWhitenedRecolored(**kwargs).resid


def _without_none(bad: list[tuple[Any, type[Exception]]]):
    return [item for item in bad if item[0] is not None]


def _kernel_rows(cls: type[CovarianceEstimator]) -> list[Row]:
    target = cls.__name__
    func = _kernel(cls)
    base = {"x": X}
    bandwidth_bad = [*_without_none(FLOAT_BAD), (-1.0, ValueError), (-1, ValueError)]
    df_bad = [*INT_BAD, (-1, ValueError), (NOBS, ValueError), (NOBS + 5, ValueError)]
    rows = [
        *bad_value_rows(target, func, base, "bandwidth", bandwidth_bad),
        *bad_value_rows(target, func, base, "df_adjust", df_bad),
        *bad_value_rows(target, func, base, "center", BOOL_BAD),
        *bad_value_rows(target, func, base, "force_int", BOOL_BAD),
        *good_value_rows(
            target, func, base, "bandwidth", [None, 0, 0.0, 4, *FLOAT_GOOD]
        ),
        *good_value_rows(target, func, base, "df_adjust", [0, NOBS - 1, *INT_GOOD]),
        *good_value_rows(target, func, base, "center", BOOL_GOOD),
        *good_value_rows(target, func, base, "force_int", BOOL_GOOD),
        *good_value_rows(target, func, base, "weights", [None]),
    ]
    return rows


def _array_rows(
    target: str, func: Any, base: dict[str, Any], name: str, nvar: int = 2
) -> list[Row]:
    """Rows for the data argument and the weights of a kernel estimator"""
    short = X[:1]
    rows: list[Row] = []
    bad_x: list[tuple[Any, type[Exception], str]] = [
        (np.where(np.arange(NOBS)[:, None] == 3, np.nan, X), ValueError, "nan"),
        (np.where(np.arange(NOBS)[:, None] == 3, np.inf, X), ValueError, "inf"),
        (np.where(np.arange(NOBS)[:, None] == 3, -np.inf, X), ValueError, "-inf"),
        (
            pd.DataFrame(np.where(np.arange(NOBS)[:, None] == 3, np.nan, X)),
            ValueError,
            "nan-frame",
        ),
        (np.ones((NOBS, 2, 2)), ValueError, "3d"),
        (short, ValueError, "one-row"),
        (np.empty((0, 2)), ValueError, "empty"),
        (np.empty((NOBS, 0)), ValueError, "no-columns"),
        (np.array([["a", "b"]] * NOBS), TypeError, "strings"),
        (None, TypeError, "none"),
        ("abc", TypeError, "str"),
        (np.ones((NOBS, 2), dtype=complex), TypeError, "complex"),
    ]
    for value, exc, label in bad_x:
        rows.append(
            Row(
                target,
                func,
                {**base, name: value},
                exc,
                rf"^{name} must be ",
                id=f"{target}-{name}-{label}",
                name=name,
            )
        )
    good_x = [
        (X, "ndarray"),
        (X.tolist(), "list"),
        (pd.DataFrame(X, columns=["a", "b"]), "frame"),
        (pd.Series(X[:, 0]), "series"),
        (X[:, 0], "1d"),
        (np.round(X * 3).astype(int), "int"),
        (X > 0, "bool"),
        (X.astype(np.float32), "float32"),
    ]
    for value, label in good_x:
        rows.append(
            Row(
                target,
                func,
                {**base, name: value},
                None,
                id=f"{target}-{name}-ok-{label}",
            )
        )
    return rows


def _weights_rows(cls: type[CovarianceEstimator]) -> list[Row]:
    target = cls.__name__
    func = _kernel(cls)
    base = {"x": X}
    nan_w = np.array([1.0, np.nan])
    inf_w = np.array([1.0, np.inf])
    bad: list[tuple[Any, type[Exception], str]] = [
        (nan_w, ValueError, "nan"),
        (inf_w, ValueError, "inf"),
        (np.ones(3), ValueError, "length-3"),
        (np.ones(1), ValueError, "length-1"),
        (np.ones((2, 2)), ValueError, "2d"),
        (np.array([1.0, -1.0]), ValueError, "negative"),
        (np.zeros(2), ValueError, "zeros"),
        ("ab", TypeError, "str"),
        (np.array(["a", "b"]), TypeError, "strings"),
        (1j * np.ones(2), TypeError, "complex"),
    ]
    rows = [
        Row(
            target,
            func,
            {**base, "weights": value},
            exc,
            r"^weights must be ",
            id=f"{target}-weights-{label}",
            name="weights",
        )
        for value, exc, label in bad
    ]
    good: list[tuple[Any, str]] = [
        (np.ones(2), "ones"),
        ([1, 2], "list"),
        (np.array([[1.0], [2.0]]), "column"),
        (pd.Series([1.0, 0.0]), "series"),
        (np.array([1, 0]), "int"),
    ]
    rows += [
        Row(
            target,
            func,
            {**base, "weights": value},
            None,
            id=f"{target}-weights-ok-{label}",
        )
        for value, label in good
    ]
    return rows


KERNEL_ROWS: list[Row] = []
for _cls in KERNELS:
    KERNEL_ROWS += _kernel_rows(_cls)
# The data and the weights are checked once by the base class, so the table uses
# a few classes
for _cls in (Bartlett, QuadraticSpectral, ZeroLag):
    KERNEL_ROWS += _array_rows(_cls.__name__, _kernel(_cls), {}, "x")
    KERNEL_ROWS += _weights_rows(_cls)


# PreWhitenedRecolored
PW_BASE: dict[str, Any] = {"x": X_LONG}
PW_ROWS: list[Row] = [
    *bad_value_rows(
        "PWRC", _pwrc, PW_BASE, "lags", [*_without_none(INT_BAD), (-1, ValueError)]
    ),
    *bad_value_rows(
        "PWRC", _pwrc, PW_BASE, "max_lag", [*_without_none(INT_BAD), (-1, ValueError)]
    ),
    *bad_value_rows("PWRC", _pwrc, PW_BASE, "diagonal", BOOL_BAD),
    *bad_value_rows("PWRC", _pwrc, PW_BASE, "sample_autocov", BOOL_BAD),
    *bad_value_rows("PWRC", _pwrc, PW_BASE, "center", BOOL_BAD),
    *bad_value_rows("PWRC", _pwrc, PW_BASE, "force_int", BOOL_BAD),
    *bad_value_rows(
        "PWRC",
        _pwrc,
        PW_BASE,
        "bandwidth",
        [*_without_none(FLOAT_BAD), (-1.0, ValueError)],
    ),
    *bad_value_rows(
        "PWRC",
        _pwrc,
        PW_BASE,
        "df_adjust",
        [*INT_BAD, (-1, ValueError), (250, ValueError)],
    ),
    *good_value_rows("PWRC", _pwrc, PW_BASE, "lags", [None, 0, 1, *INT_GOOD]),
    *good_value_rows("PWRC", _pwrc, PW_BASE, "max_lag", [None, 0, 2, *INT_GOOD]),
    *good_value_rows("PWRC", _pwrc, PW_BASE, "diagonal", BOOL_GOOD),
    *good_value_rows(
        "PWRC", _pwrc, {**PW_BASE, "lags": 1}, "sample_autocov", BOOL_GOOD
    ),
    *good_value_rows("PWRC", _pwrc, {**PW_BASE, "lags": 1}, "center", BOOL_GOOD),
    *good_value_rows("PWRC", _pwrc, {**PW_BASE, "lags": 1}, "force_int", BOOL_GOOD),
    *good_value_rows(
        "PWRC", _pwrc, PW_BASE, "bandwidth", [None, 0, 0.0, 3, *FLOAT_GOOD]
    ),
    *good_value_rows("PWRC", _pwrc, PW_BASE, "df_adjust", [0, 249, *INT_GOOD]),
    *good_value_rows("PWRC", _pwrc, PW_BASE, "weights", [None, [1, 2], np.ones(2)]),
    Row(
        "PWRC",
        _pwrc,
        {**PW_BASE, "kernel": None, "bandwidth": 2.5},
        ValueError,
        r"^bandwidth must be None or 0 when kernel is None; got 2.5 \(float\)$",
        id="PWRC-bandwidth-kernel-none",
        name="bandwidth",
    ),
    Row(
        "PWRC",
        _pwrc,
        {**PW_BASE, "kernel": None, "bandwidth": 0},
        None,
        id="PWRC-bandwidth-kernel-none-zero",
    ),
    Row(
        "PWRC",
        _pwrc,
        {**PW_BASE, "lags": 83},
        ValueError,
        r"^lags must be at most 82 when x has 250 observations and 2 series so ",
        id="PWRC-lags-too-large",
        name="lags",
    ),
    Row(
        "PWRC",
        _pwrc_resid,
        {**PW_BASE, "lags": 82},
        None,
        id="PWRC-lags-largest",
    ),
    Row(
        "PWRC",
        _pwrc_resid,
        {**PW_BASE, "lags": 83, "center": False},
        None,
        id="PWRC-lags-largest-no-center",
    ),
    Row(
        "PWRC",
        _pwrc,
        {**PW_BASE, "lags": 84, "center": False},
        ValueError,
        r"^lags must be at most 83 when x has 250 observations",
        id="PWRC-lags-too-large-no-center",
        name="lags",
    ),
    Row(
        "PWRC",
        _pwrc,
        {"x": np.where(np.arange(250)[:, None] == 9, np.nan, X_LONG)},
        ValueError,
        r"^x must be a 2-d array of finite numbers with at least 2 rows; got 2 NaN ",
        id="PWRC-x-nan-long",
        name="x",
    ),
    Row(
        "PWRC",
        _pwrc,
        {"x": X_LONG, "kernel": None, "lags": 1},
        None,
        id="PWRC-var-hac",
    ),
    Row(
        "PWRC",
        _pwrc_resid,
        {"x": X_LONG, "max_lag": 100},
        None,
        id="PWRC-max-lag-limited",
    ),
    Row(
        "PWRC",
        _pwrc,
        {"x": X_LONG, "lags": 1, "center": np.bool_},
        TypeError,
        r"^center must be True or False; got <class 'numpy.bool'> \(type\)$",
        id="PWRC-center-bool-class",
        name="center",
    ),
]
for _name in ("x",):
    PW_ROWS += _array_rows("PWRC", _pwrc, {}, _name)
PW_ROWS += _weights_rows(PreWhitenedRecolored)  # type: ignore[arg-type]


# cov_nw
def _cov_nw(**kwargs: Any) -> Any:
    return cov_nw(**kwargs)


CN_BASE: dict[str, Any] = {"y": X}
CN_ROWS: list[Row] = [
    *bad_value_rows(
        "cov_nw",
        _cov_nw,
        CN_BASE,
        "lags",
        [*INT_BAD, (-1, ValueError), (NOBS + 1, ValueError)],
    ),
    *bad_value_rows("cov_nw", _cov_nw, CN_BASE, "demean", BOOL_BAD),
    *bad_value_rows(
        "cov_nw",
        _cov_nw,
        CN_BASE,
        "axis",
        [*INT_BAD, (2, ValueError), (-3, ValueError), (10, ValueError)],
    ),
    *bad_value_rows(
        "cov_nw",
        _cov_nw,
        CN_BASE,
        "ddof",
        [*INT_BAD, (-1, ValueError), (NOBS, ValueError), (NOBS + 1, ValueError)],
    ),
    *good_value_rows("cov_nw", _cov_nw, CN_BASE, "lags", [0, NOBS, *INT_GOOD]),
    *good_value_rows("cov_nw", _cov_nw, CN_BASE, "demean", BOOL_GOOD),
    *good_value_rows("cov_nw", _cov_nw, CN_BASE, "axis", [0, 1, -1, -2]),
    *good_value_rows("cov_nw", _cov_nw, CN_BASE, "axis", [np.int8(0), np.int64(-1)]),
    *good_value_rows("cov_nw", _cov_nw, CN_BASE, "ddof", [0, NOBS - 1, *INT_GOOD]),
    Row(
        "cov_nw",
        _cov_nw,
        {"y": X.T, "axis": 1, "lags": 2, "ddof": 1},
        None,
        id="cov_nw-axis-1-ok",
    ),
    Row(
        "cov_nw",
        _cov_nw,
        {"y": X.T, "axis": 1, "lags": 3, "ddof": 1},
        None,
        id="cov_nw-axis-1-lags-ok",
    ),
    Row(
        "cov_nw",
        _cov_nw,
        {"y": X.T, "axis": 0, "lags": 3},
        ValueError,
        r"^lags must be an integer in \[0, 2\]; got 3 \(int\)$",
        id="cov_nw-lags-uses-axis",
        name="lags",
    ),
    Row(
        "cov_nw",
        _cov_nw,
        {"y": X[:, 0], "axis": 1},
        ValueError,
        r"^axis must be an integer in \[-1, 0\]; got 1 \(int\)$",
        id="cov_nw-axis-1d",
        name="axis",
    ),
    Row(
        "cov_nw",
        _cov_nw,
        {"y": X[:, 0], "axis": -1},
        None,
        id="cov_nw-axis-1d-ok",
    ),
]
_y_bad: list[tuple[Any, type[Exception], str]] = [
    (np.where(np.arange(NOBS)[:, None] == 3, np.nan, X), ValueError, "nan"),
    (np.where(np.arange(NOBS)[:, None] == 3, np.inf, X), ValueError, "inf"),
    (np.where(np.arange(NOBS) == 3, np.nan, X[:, 0]), ValueError, "nan-1d"),
    (np.ones((NOBS, 2, 2)), ValueError, "3d"),
    (np.empty((0, 2)), ValueError, "empty"),
    (np.empty(0), ValueError, "empty-1d"),
    (np.float64(1.0), ValueError, "scalar"),
    (np.array([["a", "b"]] * NOBS), TypeError, "strings"),
    (None, TypeError, "none"),
    ("abc", TypeError, "str"),
]
CN_ROWS += [
    Row(
        "cov_nw",
        _cov_nw,
        {"y": value},
        exc,
        r"^y must be ",
        id=f"cov_nw-y-{label}",
        name="y",
    )
    for value, exc, label in _y_bad
]
CN_ROWS += [
    Row("cov_nw", _cov_nw, {"y": value}, None, id=f"cov_nw-y-ok-{label}")
    for value, label in [
        (X, "ndarray"),
        (X[:, 0], "1d"),
        (X.tolist(), "list"),
        (pd.DataFrame(X), "frame"),
        (pd.Series(X[:, 0]), "series"),
        (np.round(X * 3).astype(int), "int"),
        (np.ones((NOBS, 1)), "column"),
        (np.ones(1), "one-obs"),
    ]
]

ALL_ROWS = [*KERNEL_ROWS, *PW_ROWS, *CN_ROWS]


@pytest.mark.parametrize("row", to_params(ALL_ROWS))
def test_validation(row: Row) -> None:
    run_row(row)


def test_bandwidth_is_float() -> None:
    est = Bartlett(X, bandwidth=np.float32(4.0))
    assert isinstance(est.bandwidth, float)
    assert est.bandwidth == 4.0


def test_df_adjust_is_python_int() -> None:
    est = Bartlett(X, df_adjust=np.int8(3), bandwidth=2)
    assert "Degree of Freedom Adjustment: 3" in str(est)


def test_cov_nw_values() -> None:
    # The checked arguments give the same values as the plain arguments
    expected = cov_nw(X, lags=2, ddof=1)
    np.testing.assert_allclose(
        cov_nw(X, lags=np.int64(2), demean=np.True_, axis=np.int8(0), ddof=np.int16(1)),
        expected,
    )
    np.testing.assert_allclose(cov_nw(X.tolist(), lags=2, ddof=1), expected)


def test_ddof_equal_to_nobs_no_longer_returns_inf() -> None:
    # ddof == nobs divided by zero and returned inf or NaN
    with pytest.raises(ValueError, match=r"ddof must be an integer in \[0, 99\]"):
        cov_nw(X, ddof=NOBS)
