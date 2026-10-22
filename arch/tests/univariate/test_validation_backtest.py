"""
Table-driven tests of the argument validation of the VaR backtests.

The rows cover ``returns``, ``var``, ``p`` and ``window`` of ``var_exceptions``,
``kupiec_pof``, ``christoffersen_independence``,
``christoffersen_conditional_coverage`` and ``traffic_light``.
"""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from arch.tests.validation_table import (
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
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_pof,
    traffic_light,
    var_exceptions,
)

NOBS = 60
RNG = np.random.default_rng(904)
RETURNS = RNG.standard_normal(NOBS)
VAR = np.full(NOBS, 1.5)

# (target, function, the other required arguments)
FUNCTIONS = {
    "var_exceptions": (var_exceptions, {}, 1),
    "kupiec_pof": (kupiec_pof, {"p": 0.05}, 1),
    "christoffersen_independence": (christoffersen_independence, {}, 2),
    "christoffersen_conditional_coverage": (
        christoffersen_conditional_coverage,
        {"p": 0.05},
        2,
    ),
    "traffic_light": (traffic_light, {"p": 0.05, "window": 20}, 1),
}

# Values that are not data: (value, exception)
DATA_BAD = [
    ("abc", TypeError),
    (["a", "b"], TypeError),
    (None, TypeError),
    (np.array(["a"] * NOBS), TypeError),
    (np.zeros(NOBS, dtype=np.float16), TypeError),
    (np.zeros(NOBS, dtype=np.complex128), TypeError),
    (pd.Series(pd.date_range("2000-01-01", periods=NOBS)), TypeError),
    (np.zeros((NOBS, 2)), ValueError),
    (np.zeros((2, NOBS, 1)), ValueError),
    (np.array([]), ValueError),
    ([], ValueError),
    (np.array([1.0, np.nan] + [0.0] * (NOBS - 2)), ValueError),
    (np.array([1.0, np.inf] + [0.0] * (NOBS - 2)), ValueError),
    (np.array([1.0, -np.inf] + [0.0] * (NOBS - 2)), ValueError),
    (pd.Series([np.nan] * NOBS), ValueError),
    ([float("nan")] * NOBS, ValueError),
]
DATA_RULE = r"a 1-d array of finite .*numbers.*with at least 1 element"


def _rows() -> list[Row]:
    rows: list[Row] = []
    for target, (func, extra, _) in FUNCTIONS.items():
        base = {"returns": RETURNS, "var": VAR, **extra}
        for name in ("returns", "var"):
            # one row per bad value, with ids that are unique over the targets
            rows += bad_value_rows(target, func, base, name, DATA_BAD, rule=DATA_RULE)
        if "p" in extra:
            p_bad = [
                *FLOAT_BAD,
                (0.0, ValueError),
                (1.0, ValueError),
                (-0.1, ValueError),
                (1.5, ValueError),
                (np.float64(1.0), ValueError),
                (0, ValueError),
                (1, ValueError),
            ]
            rows += bad_value_rows(
                target,
                func,
                base,
                "p",
                p_bad,
                rule=r"a finite float in \(0, 1\)",
            )
            rows += good_value_rows(target, func, base, "p", [*FLOAT_GOOD, 1e-9])
        if "window" in extra:
            rows += bad_value_rows(
                target,
                func,
                base,
                "window",
                [*INT_BAD, (0, ValueError), (-5, ValueError)],
                rule=r"an integer >= 1",
            )
            rows += good_value_rows(target, func, base, "window", INT_GOOD)
            rows.append(
                Row(
                    target,
                    func,
                    {**base, "window": NOBS + 1},
                    ValueError,
                    r"^window must be an integer no larger than the number of aligned "
                    r"observations \(60\); got 61 \(int\)",
                    id=f"{target}-window-too-long",
                    name="window",
                )
            )
    return rows


def _data_rows() -> list[Row]:
    rows: list[Row] = []
    for target, (func, extra, min_obs) in FUNCTIONS.items():
        base = {"returns": RETURNS, "var": VAR, **extra}
        rows.append(
            Row(
                target,
                func,
                {**base, "var": VAR[:-1]},
                ValueError,
                r"^var must be an array with the same length as returns \(60\) or "
                r"aligned with returns on the index; got length 59 \(ndarray\)",
                id=f"{target}-var-length",
                name="var",
            )
        )
        rows.append(
            Row(
                target,
                func,
                {**base, "returns": list(RETURNS[:-5])},
                ValueError,
                r"^var must be an array with the same length as returns \(55\)",
                id=f"{target}-returns-length",
                name="var",
            )
        )
        # Series that share no index value
        left = pd.Series(RETURNS, index=pd.RangeIndex(NOBS))
        right = pd.Series(VAR, index=pd.RangeIndex(NOBS, 2 * NOBS))
        rows.append(
            Row(
                target,
                func,
                {**base, "returns": left, "var": right},
                ValueError,
                rf"^returns must be aligned with var on at least {min_obs} "
                r"observations?; got 0 \(int\)$",
                id=f"{target}-disjoint",
                name="returns",
            )
        )
        dup = pd.Series(VAR, index=[0, 0, *range(2, NOBS)])
        rows.append(
            Row(
                target,
                func,
                {**base, "returns": left, "var": dup},
                ValueError,
                r"^returns and var must have an index without duplicates",
                id=f"{target}-duplicated-index",
                form=False,
            )
        )
        # Accepted: a list, a Series and a one-column DataFrame
        for label, ret in (
            ("list", list(RETURNS)),
            ("series", pd.Series(RETURNS)),
            ("frame", pd.DataFrame(RETURNS)),
            ("int", np.zeros(NOBS, dtype=np.int64)),
            ("float32", RETURNS.astype(np.float32)),
        ):
            rows.append(
                Row(
                    target,
                    func,
                    {**base, "returns": ret},
                    None,
                    id=f"{target}-returns-{label}-ok",
                    name="returns",
                )
            )
        # The shortest series that each function accepts
        one = {**base, "returns": RETURNS[:1], "var": VAR[:1]}
        if min_obs == 2:
            rows.append(
                Row(
                    target,
                    func,
                    one,
                    ValueError,
                    r"^returns must be aligned with var on at least 2 observations; "
                    r"got 1 \(int\)$",
                    id=f"{target}-one-observation",
                    name="returns",
                )
            )
            two = {**base, "returns": RETURNS[:2], "var": VAR[:2]}
            rows.append(Row(target, func, two, None, id=f"{target}-two-obs-ok"))
        elif "window" in extra:
            rows.append(
                Row(
                    target,
                    func,
                    {**one, "window": 1},
                    None,
                    id=f"{target}-one-observation-ok",
                )
            )
        else:
            rows.append(Row(target, func, one, None, id=f"{target}-one-obs-ok"))
    return rows


ROWS = _rows() + _data_rows()


@pytest.mark.parametrize("row", to_params(ROWS))
def test_validation(row):
    run_row(row)


def test_p_must_be_given():
    with pytest.raises(TypeError, match="p"):
        kupiec_pof(RETURNS, VAR)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="p"):
        christoffersen_conditional_coverage(RETURNS, VAR)  # type: ignore[call-arg]


def test_fraction_p_accepted():
    assert (
        kupiec_pof(RETURNS, VAR, Fraction(1, 20)).stat
        == kupiec_pof(RETURNS, VAR, 0.05).stat
    )


def test_defaults_of_traffic_light():
    returns = np.zeros(250)
    result = traffic_light(returns, np.ones(250))
    assert result.p == 0.01
    assert result.nobs == 250
