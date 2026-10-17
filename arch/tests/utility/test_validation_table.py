"""Tests of the shared helper for table-driven validation tests"""

import numpy as np
import pytest

from arch.tests.validation_table import (
    BOOL_BAD,
    BOOL_GOOD,
    FLOAT_BAD,
    FLOAT_GOOD,
    INT_BAD,
    INT_GOOD,
    SEED_BAD,
    SEED_GOOD,
    Row,
    bad_value_rows,
    good_value_rows,
    ids_for,
    message_form,
    run_row,
    to_params,
)


def make_func(*, lags, flag=True):
    """A function with a validation in the common form, written by hand."""
    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)):
        raise TypeError(
            f"lags must be an integer; got {lags!r} ({type(lags).__name__})"
        )
    if lags < 0:
        raise ValueError(f"lags must be an integer >= 0; got {lags!r} (int)")
    if not isinstance(flag, bool):
        raise TypeError(
            f"flag must be True, False, 0 or 1; got {flag!r} ({type(flag).__name__})"
        )


BASE = {"lags": 2, "flag": True}
ROWS = [
    *bad_value_rows(
        "make", make_func, BASE, "lags", [(2.5, TypeError), (True, TypeError)]
    ),
    *good_value_rows("make", make_func, BASE, "lags", [3, np.int8(3)]),
    *bad_value_rows(
        "make", make_func, BASE, "flag", [(0, TypeError), ("no", TypeError)]
    ),
    Row(
        "make",
        make_func,
        {**BASE, "lags": -1},
        ValueError,
        r"lags must be an integer >= 0; got -1 \(int\)",
        id="make-lags-range",
        name="lags",
    ),
]


@pytest.mark.parametrize("row", to_params(ROWS))
def test_rows(row):
    run_row(row)


def test_ids_are_unique_and_deterministic():
    for values in (INT_BAD, FLOAT_BAD, BOOL_BAD, SEED_BAD):
        ids = ids_for(values)
        assert len(set(ids)) == len(ids)
    for values in (INT_GOOD, FLOAT_GOOD, BOOL_GOOD, SEED_GOOD):
        ids = ids_for(values)
        assert len(set(ids)) == len(ids)
    # 2, np.int64(2) and 2.0 would all render as "2"
    ids = ids_for([2, np.int64(2), 2.0])
    assert len(set(ids)) == 3
    assert ids_for([(2, TypeError)]) == ids_for([2])


def test_table_helpers_detect_problems():
    ok = Row("t", make_func, BASE, None)
    run_row(ok)
    # a call that should raise but does not
    with pytest.raises(pytest.fail.Exception):
        run_row(Row("t", make_func, BASE, ValueError, id="x"))
    # a call that raises with another message
    with pytest.raises(AssertionError):
        run_row(Row("t", make_func, {**BASE, "lags": -1}, ValueError, "flag"))

    # a message that is not in the common form
    def odd(**kwargs):
        raise ValueError("lags is bad")

    with pytest.raises(AssertionError):
        run_row(Row("t", odd, {}, ValueError, "lags", name="lags"))
    run_row(Row("t", odd, {}, ValueError, "lags", name="lags", form=False))


def test_to_params_rejects_duplicate_ids():
    rows = [
        Row("t", make_func, BASE, None, id="same"),
        Row("t", make_func, BASE, None, id="same"),
    ]
    with pytest.raises(ValueError, match="duplicate row id same"):
        to_params(rows)
    params = to_params(
        [Row("t", make_func, BASE, None), Row("t", make_func, BASE, None)]
    )
    assert [p.id for p in params] == ["t-0", "t-1"]


def test_message_form_pattern():
    pattern = message_form("size")
    assert pattern.match("size must be a float in (0, 1); got 2 (int)")
    assert not pattern.match("size is wrong")
    assert not pattern.match("sizes must be a float; got 2 (int)")
    assert message_form("a.b").match("a.b must be x; got 1 (int)")
    assert not message_form("a.b").match("aXb must be x; got 1 (int)")
