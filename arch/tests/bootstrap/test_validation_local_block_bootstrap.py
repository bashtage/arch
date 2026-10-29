"""
Table-driven tests of the input validation of LocalBlockBootstrap.

The rows are built on the helpers of arch/tests/validation_table.py.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from arch.bootstrap import LocalBlockBootstrap
from arch.tests.validation_table import (
    BOOL_BAD,
    BOOL_GOOD,
    INT_BAD,
    INT_GOOD,
    SEED_BAD,
    SEED_GOOD,
    Row,
    bad_value_rows,
    good_value_rows,
    run_row,
    to_params,
)

NOBS = 100
Y = np.random.default_rng(20261018).standard_normal(NOBS)
SEED_RULE = r"None, a non-negative integer, a NumPy Generator or a NumPy RandomState"


def mean_fn(y: Any) -> float:
    return float(np.mean(y))


def ctor(block_size: Any = 5, locality: Any = 8, **kwargs: Any) -> LocalBlockBootstrap:
    return LocalBlockBootstrap(block_size, locality, Y, **kwargs)


ROWS: list[Row] = []

# ---- locality: an integer >= 0
LOCALITY_BAD = [*INT_BAD, (-1, ValueError), (np.int64(-3), ValueError)]
ROWS += bad_value_rows(
    "LBB-locality",
    lambda locality: ctor(locality=locality, seed=0),
    {},
    "locality",
    LOCALITY_BAD,
    rule=r"an integer >= 0",
)
ROWS += good_value_rows(
    "LBB-locality",
    lambda locality: ctor(locality=locality, seed=0),
    {},
    "locality",
    [*INT_GOOD, 0, 1, NOBS, 10**6, 10**30],
)
ROWS.append(
    Row(
        "LBB-locality-none",
        lambda: LocalBlockBootstrap(5, None, Y),
        {},
        TypeError,
        r"^locality must be an integer >= 0; got None \(NoneType\)",
        id="LBB-locality-None-positional",
        name="locality",
    )
)

# ---- block_size: an integer >= 1, and at most the number of observations
BLOCK_BAD = [*INT_BAD, (0, ValueError), (-1, ValueError), (np.int64(-3), ValueError)]
ROWS += bad_value_rows(
    "LBB-block_size",
    lambda block_size: ctor(block_size=block_size, seed=0),
    {},
    "block_size",
    BLOCK_BAD,
    rule=r"an integer >= 1",
)
ROWS += good_value_rows(
    "LBB-block_size",
    lambda block_size: ctor(block_size=block_size, seed=0),
    {},
    "block_size",
    [*INT_GOOD, 1, NOBS],
)
for _size in (NOBS + 1, 10**6):
    ROWS.append(
        Row(
            "LBB-block_size-too-large",
            lambda block_size: ctor(block_size=block_size, seed=0).bootstrap(1),
            {"block_size": _size},
            ValueError,
            rf"^block_size must be an integer in \[1, {NOBS}\]; got {_size} \(int\)",
            id=f"LBB-block_size-gt-n-{_size}",
            name="block_size",
        )
    )
    ROWS.append(
        Row(
            "LBB-block_size-too-large-cov",
            lambda block_size: ctor(block_size=block_size, seed=0).cov(mean_fn, 5),
            {"block_size": _size},
            ValueError,
            rf"^block_size must be an integer in \[1, {NOBS}\]; got {_size} \(int\)",
            id=f"LBB-block_size-gt-n-cov-{_size}",
            name="block_size",
        )
    )
    ROWS.append(
        Row(
            "LBB-block_size-too-large-update_indices",
            lambda block_size: ctor(block_size=block_size, seed=0).update_indices(),
            {"block_size": _size},
            ValueError,
            rf"^block_size must be an integer in \[1, {NOBS}\]; got {_size} \(int\)",
            id=f"LBB-block_size-gt-n-update_indices-{_size}",
            name="block_size",
        )
    )

# ---- seed: the constructor and clone
ROWS += bad_value_rows(
    "LBB-ctor",
    lambda seed: ctor(seed=seed),
    {},
    "seed",
    SEED_BAD,
    rule=SEED_RULE,
)
ROWS += good_value_rows(
    "LBB-ctor", lambda seed: ctor(seed=seed), {}, "seed", [*SEED_GOOD, None]
)
ROWS += bad_value_rows(
    "LBB-clone",
    lambda seed: ctor(seed=0).clone(Y, seed=seed),
    {},
    "seed",
    SEED_BAD,
    rule=SEED_RULE,
)
ROWS += good_value_rows(
    "LBB-clone", lambda seed: ctor(seed=0).clone(Y, seed=seed), {}, "seed", SEED_GOOD
)

# ---- the data
DATA_BAD: list[tuple[str, Any, type[Exception], str]] = [
    ("list", [1.0, 2.0, 3.0], TypeError, r"must be an instance of ndarray"),
    ("None", None, TypeError, r"must be an instance of ndarray"),
    ("scalar", 5.0, TypeError, r"must be an instance of ndarray"),
    ("empty-array", np.array([]), ValueError, r"at least 1 row"),
    ("0-d-array", np.array(3.0), ValueError, r"at least 1 row"),
    ("empty-frame", pd.DataFrame({"a": []}), ValueError, r"at least 1 row"),
]
for _label, _value, _exc, _rule in DATA_BAD:
    ROWS.append(
        Row(
            "LBB-data",
            lambda data: LocalBlockBootstrap(5, 8, data=data),
            {"data": _value},
            _exc,
            rf"^data {_rule}" if _exc is TypeError else rf"^data must be .*{_rule}",
            id=f"LBB-data-kw-{_label}",
            name="data",
        )
    )
ROWS += [
    Row(
        "LBB-nodata",
        lambda: LocalBlockBootstrap(5, 8),
        {},
        ValueError,
        r"^args must be at least one NumPy array, Series or DataFrame",
        id="LBB-no-data",
        name="args",
    ),
    Row(
        "LBB-unequal",
        lambda: LocalBlockBootstrap(5, 8, Y, Y[:50]),
        {},
        ValueError,
        r"^args\[1\] must be an array with 100 rows",
        id="LBB-unequal-pos",
        name="args[1]",
    ),
    Row(
        "LBB-unequal-kw",
        lambda: LocalBlockBootstrap(5, 8, Y, z=Y[:50]),
        {},
        ValueError,
        r"^z must be an array with 100 rows",
        id="LBB-unequal-kw",
        name="z",
    ),
    Row(
        "LBB-reserved",
        lambda: LocalBlockBootstrap(5, 8, locality=Y),
        {},
        TypeError,
        r"multiple values for argument 'locality'",
        id="LBB-locality-keyword-twice",
        form=False,
    ),
    Row(
        "LBB-reserved",
        lambda: LocalBlockBootstrap(5, 8, index=Y),
        {},
        ValueError,
        r"^kwargs must be keywords that are not attributes or methods of the "
        r"bootstrap; got 'index' \(str\)",
        id="LBB-reserved-index",
        name="kwargs",
    ),
]
GOOD_DATA = {
    "series": pd.Series(Y),
    "frame": pd.DataFrame({"a": Y, "b": Y}),
    "2d": np.ones((NOBS, 2)),
    "int": np.arange(NOBS),
    "bool": np.arange(NOBS) > 50,
    "nan": np.r_[np.nan, Y[1:]],
    "object": np.array(["a", "b"] * 50, dtype=object),
}
for _label, _value in GOOD_DATA.items():
    ROWS.append(
        Row(
            "LBB-data-good",
            lambda v: LocalBlockBootstrap(5, 8, v).bootstrap(1),
            {"v": _value},
            None,
            id=f"LBB-good-{_label}",
        )
    )
ROWS.append(
    Row(
        "LBB-one-row",
        lambda: LocalBlockBootstrap(1, 0, np.zeros(1)).bootstrap(1),
        {},
        None,
        id="LBB-one-row",
    )
)

# ---- bootstrap, apply, cov, var and conf_int: reps, and the conf_int arguments
REPS_BAD = [*INT_BAD, (0, ValueError), (-1, ValueError), (np.int64(-2), ValueError)]
METHODS: dict[str, Callable[..., Any]] = {
    "bootstrap": lambda **kw: list(ctor(seed=0).bootstrap(**kw)),
    "apply": lambda **kw: ctor(seed=0).apply(**kw),
    "cov": lambda **kw: ctor(seed=0).cov(**kw),
    "var": lambda **kw: ctor(seed=0).var(**kw),
    "conf_int": lambda **kw: ctor(seed=0).conf_int(**kw),
}
BASE = {"func": mean_fn, "reps": 3}
for _name, _call in METHODS.items():
    _base = {"reps": 3} if _name == "bootstrap" else BASE
    ROWS += bad_value_rows(
        f"LBB-{_name}-reps", _call, _base, "reps", REPS_BAD, rule=r"an integer >= 1"
    )
    ROWS += good_value_rows(f"LBB-{_name}-reps", _call, _base, "reps", [*INT_GOOD, 1])
CI = METHODS["conf_int"]
CI_BASE = {"func": mean_fn, "reps": 20}
ROWS += bad_value_rows(
    "LBB-conf_int-reuse", CI, CI_BASE, "reuse", BOOL_BAD, rule=r"True, False, 0 or 1"
)
ROWS += good_value_rows("LBB-conf_int-reuse", CI, CI_BASE, "reuse", BOOL_GOOD)
ROWS += bad_value_rows(
    "LBB-conf_int-method",
    CI,
    CI_BASE,
    "method",
    [(5, TypeError), (None, TypeError), ("nothing", ValueError), ("", ValueError)],
    rule=r"one of .+",
)
ROWS += good_value_rows(
    "LBB-conf_int-method",
    CI,
    CI_BASE,
    "method",
    ["percentile", "BASIC", "Norm", "bc", "BCa", "studentized", "Studentized"],
)
ROWS += bad_value_rows(
    "LBB-conf_int-studentize_reps",
    CI,
    {**CI_BASE, "method": "studentized"},
    "studentize_reps",
    [*INT_BAD, (0, ValueError), (-1, ValueError)],
    rule=r"an integer >= 1",
)
ROWS += bad_value_rows(
    "LBB-conf_int-std_err_func",
    CI,
    {**CI_BASE, "method": "studentized"},
    "std_err_func",
    [(5, TypeError), ("abc", TypeError)],
    rule=r"callable or None",
)


@pytest.mark.filterwarnings("ignore:The conditions to reuse:RuntimeWarning")
@pytest.mark.parametrize("row", to_params(ROWS))
def test_validation(row: Row) -> None:
    run_row(row)
