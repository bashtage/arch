"""
Every unit root and cointegration test with a constant series as y.

The statistics are ratios of quantities that are 0 for a constant series, so a
constant y must raise InfeasibleTestException (a ValueError). Previously the tests
returned nan, raised ZeroDivisionError or LinAlgError, or computed arbitrary
statistics and p-values from the rounding errors of removing the mean of a constant
that is not 0 (KPSS rejected stationarity with a p-value of 0.0001).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from arch.unitroot import (
    ADF,
    DFGLS,
    KPSS,
    PhillipsPerron,
    VarianceRatio,
    ZivotAndrews,
)
from arch.unitroot.cointegration import (
    CanonicalCointegratingReg,
    DynamicOLS,
    FullyModifiedOLS,
    engle_granger,
    phillips_ouliaris,
)
from arch.utility.exceptions import InfeasibleTestException

NOBS = 250
X = pd.DataFrame(
    np.random.default_rng(0).standard_normal((NOBS, 2)).cumsum(0),
    columns=["x0", "x1"],
)


def _unit_root_cases() -> list[Any]:
    lags = (None, 2)
    cases: list[Any] = [
        pytest.param(ADF, {"trend": trend, "lags": lag})
        for trend in ("n", "c", "ct", "ctt")
        for lag in lags
    ]
    cases.extend(
        pytest.param(test, {"trend": trend, "lags": lag})
        for test in (DFGLS, KPSS)
        for trend in ("c", "ct")
        for lag in lags
    )
    cases.extend(
        pytest.param(
            PhillipsPerron, {"trend": trend, "test_type": test_type, "lags": lag}
        )
        for trend in ("n", "c", "ct")
        for test_type in ("tau", "rho")
        for lag in lags
    )
    cases.extend(
        pytest.param(ZivotAndrews, {"trend": trend, "lags": lag})
        for trend in ("c", "t", "ct")
        for lag in lags
    )
    cases.extend(
        pytest.param(
            VarianceRatio,
            {"lags": 2, "trend": trend, "robust": robust, "overlap": overlap},
        )
        for trend in ("n", "c")
        for robust in (True, False)
        for overlap in (True, False)
    )
    return cases


UNIT_ROOT_CASES = _unit_root_cases()


def _id(case: Any) -> str:
    test, kwargs = case.values
    return test.__name__ + "-" + "-".join(f"{k}={v}" for k, v in kwargs.items())


UNIT_ROOT_IDS = [_id(case) for case in UNIT_ROOT_CASES]


def _cointegration_cases() -> list[Any]:
    trends = ("n", "c", "ct", "ctt")
    cases: list[Any] = [
        pytest.param(
            lambda y, trend=trend: engle_granger(y, X, trend=trend),
            id=f"engle_granger-{trend}",
        )
        for trend in trends
    ]
    cases.extend(
        pytest.param(
            lambda y, trend=trend, test_type=test_type: phillips_ouliaris(
                y, X, trend=trend, test_type=test_type
            ),
            id=f"phillips_ouliaris-{trend}-{test_type}",
        )
        for trend in trends
        for test_type in ("za", "zt", "pu", "pz")
    )
    cases.extend(
        pytest.param(
            lambda y, trend=trend: DynamicOLS(y, X, trend=trend).fit(),
            id=f"DynamicOLS-{trend}",
        )
        for trend in trends
    )
    cases.extend(
        pytest.param(
            lambda y, cls=cls, trend=trend: cls(y, X, trend=trend).fit(),
            id=f"{cls.__name__}-{trend}",
        )
        for trend in ("n", "c", "ct")
        for cls in (FullyModifiedOLS, CanonicalCointegratingReg)
    )
    return cases


COINTEGRATION_CASES = _cointegration_cases()

CONSTANTS = [0.0, 1.0, 5.3, -2.5, 1e8]
CONTAINERS: list[tuple[str, Callable[[float], Any]]] = [
    ("ndarray", lambda value: np.full(NOBS, value)),
    ("series", lambda value: pd.Series(np.full(NOBS, value), index=X.index)),
    ("list", lambda value: [value] * NOBS),
]
CONTAINER_IDS = [name for name, _ in CONTAINERS]


def _check_raises(func: Callable[[], Any], value: float) -> None:
    match = r"^y is constant \(every value is " + str(float(value)).replace("+", r"\+")
    with pytest.raises(InfeasibleTestException, match=match) as info:
        func()
    # InfeasibleTestException is also a ValueError
    assert isinstance(info.value, ValueError)


@pytest.mark.parametrize("value", CONSTANTS)
@pytest.mark.parametrize("container", [c for _, c in CONTAINERS], ids=CONTAINER_IDS)
@pytest.mark.parametrize(("test", "kwargs"), UNIT_ROOT_CASES, ids=UNIT_ROOT_IDS)
def test_unit_root_constant(test, kwargs, container, value):
    # Raised when the test is created, and not when the statistic is computed
    _check_raises(lambda: test(container(value), **kwargs), value)


@pytest.mark.parametrize("value", CONSTANTS)
@pytest.mark.parametrize("container", [c for _, c in CONTAINERS], ids=CONTAINER_IDS)
@pytest.mark.parametrize("func", COINTEGRATION_CASES)
def test_cointegration_constant(func, container, value):
    _check_raises(lambda: func(container(value)), value)


def test_integer_constant():
    _check_raises(lambda: KPSS(np.full(NOBS, 3, dtype=np.int64)), 3.0)
    _check_raises(lambda: engle_granger(np.full(NOBS, 3, dtype=np.int64), X), 3.0)


@pytest.mark.parametrize(("test", "kwargs"), UNIT_ROOT_CASES, ids=UNIT_ROOT_IDS)
def test_unit_root_varying(test, kwargs):
    # A series that varies is not affected. 251 observations have 250 differences,
    # a multiple of the lags of the variance ratio test without overlap
    y = np.random.default_rng(1).standard_normal(NOBS + 1).cumsum()
    res = test(y, **kwargs)
    assert np.isfinite(res.stat)
    assert np.isfinite(res.pvalue)


@pytest.mark.parametrize("func", COINTEGRATION_CASES)
def test_cointegration_varying(func):
    y = X.sum(axis=1) + np.random.default_rng(2).standard_normal(NOBS)
    res = func(y)
    if hasattr(res, "stat"):
        assert np.isfinite(res.stat)
        assert np.isfinite(res.pvalue)
    else:
        assert np.all(np.isfinite(np.asarray(res.params)))


def test_constant_except_one_value():
    # Only a series in which every value is the same is constant
    y = np.zeros(NOBS)
    y[NOBS // 2] = 1.0
    KPSS(y)
    ADF(y, lags=2)


def test_too_few_observations_is_reported_first():
    # A specification that is not possible is reported before a constant series
    with pytest.raises(InfeasibleTestException, match="at least 4 observations"):
        ADF(np.zeros(2))
    with pytest.raises(ValueError, match=r"x must be .* with at least 5 rows"):
        engle_granger(np.zeros(2), np.ones((2, 2)))
