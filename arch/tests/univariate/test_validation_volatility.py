"""
Table-driven tests of the validation of the arguments of the volatility processes.

The constructors, the arguments of ``simulate`` and the user-facing arguments of
``forecast`` are checked. The arrays that ``compute_variance`` passes to the compiled
recursions are checked in ``test_validation_crash_guards.py``.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
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
    APARCH,
    ARCH,
    EGARCH,
    FIAPARCH,
    FIGARCH,
    GARCH,
    HARCH,
    ConstantVariance,
    EWMAVariance,
    FixedVariance,
    MIDASHyperbolic,
    RiskMetrics2006,
)
from arch.univariate.volatility import BootstrapRng

INT_MAX = 2**31 - 1
RESIDS = np.random.default_rng(20241017).standard_normal(300)


def not_none(bad):
    """Remove None from a list of bad values, for an argument that accepts it."""
    return [item for item in bad if item[0] is not None]


# Values that are of the right type but not allowed, for the orders and lengths
COUNT_BAD_0 = [(-1, ValueError), (np.int64(-3), ValueError), (INT_MAX + 1, ValueError)]
COUNT_BAD_1 = [(0, ValueError), *COUNT_BAD_0]
POWER_BAD = [(0.0, ValueError), (-0.5, ValueError), (np.float32(-1.0), ValueError)]
ORDER01_BAD = [(2, ValueError), (-1, ValueError), (np.int64(5), ValueError)]
DELTA_BAD = [(0.0, ValueError), (0.05, ValueError), (4.0, ValueError), (5, ValueError)]
DELTA_GOOD = [None, 0.06, 2, 3.99, np.float32(1.5), np.int64(1), Fraction(3, 2)]


def make(cls):
    """A function that builds ``cls(**kwargs)``."""

    def func(**kwargs):
        return cls(**kwargs)

    func.__name__ = cls.__name__
    return func


CONSTRUCTOR_ROWS: list[Row] = []


def add_bad(cls, name, bad, base=None, tag=""):
    target = cls.__name__ + tag
    CONSTRUCTOR_ROWS.extend(bad_value_rows(target, make(cls), base or {}, name, bad))


def add_good(cls, name, good, base=None, tag=""):
    target = cls.__name__ + tag
    CONSTRUCTOR_ROWS.extend(good_value_rows(target, make(cls), base or {}, name, good))


# GARCH
for order in ("p", "o", "q"):
    add_bad(GARCH, order, [*INT_BAD, *COUNT_BAD_0])
    # p = 0 is only valid if o is positive
    add_good(GARCH, order, INT_GOOD if order == "p" else [*INT_GOOD, 0], {"p": 1})
add_bad(GARCH, "power", [*FLOAT_BAD, *POWER_BAD])
add_good(GARCH, "power", [*FLOAT_GOOD, 1, 2, 1.0, 2.0, 0.25, np.int64(3)])
CONSTRUCTOR_ROWS.append(
    Row(
        "GARCH",
        make(GARCH),
        {"p": 0, "o": 0, "q": 1},
        ValueError,
        r"^p must be an integer >= 1 when o is 0; got 0 \(int\)$",
        id="GARCH-p-o-both-zero",
        name="p",
    )
)
CONSTRUCTOR_ROWS.append(Row("GARCH", make(GARCH), {"p": 0, "o": 1, "q": 0}, None))

# ARCH
add_bad(ARCH, "p", [*INT_BAD, *COUNT_BAD_1])
add_good(ARCH, "p", INT_GOOD)

# HARCH
LAGS_BAD = [
    *[(value, exc) for value, exc in INT_BAD if not isinstance(value, list)],
    *COUNT_BAD_1,
    ([], ValueError),
    ((), ValueError),
    ([0, 1], ValueError),
    ([1, -5], ValueError),
    (np.array([0, 1]), ValueError),
    (np.ones((2, 2), dtype=int), ValueError),
    ([[1, 2], [3]], ValueError),
    ([1, INT_MAX + 1], ValueError),
    ([1.5, 2], TypeError),
    ([1.0, 2.0], TypeError),
    ([True, False], TypeError),
    (["a", "b"], TypeError),
    (np.array([1.0, 5.0]), TypeError),
]
LAGS_GOOD = [
    *INT_GOOD,
    [1, 5, 22],
    (2, 3),
    np.array([1, 5]),
    np.array([1, 5], dtype=np.int8),
    np.array([1, 5], dtype=np.uint16),
    [np.int64(1), 4],
    range(1, 4),
    pd.Series([1, 5]),
]
add_bad(HARCH, "lags", LAGS_BAD)
add_good(HARCH, "lags", LAGS_GOOD)

# MIDASHyperbolic
add_bad(MIDASHyperbolic, "m", [*INT_BAD, *COUNT_BAD_1])
add_good(MIDASHyperbolic, "m", [1, *INT_GOOD])
add_bad(MIDASHyperbolic, "asym", BOOL_BAD)
add_good(MIDASHyperbolic, "asym", BOOL_GOOD)

# EWMAVariance
LAM_BAD = [(0.0, ValueError), (1.0, ValueError), (-0.5, ValueError), (1.5, ValueError)]
add_bad(EWMAVariance, "lam", [*not_none(FLOAT_BAD), *LAM_BAD])
add_good(EWMAVariance, "lam", [*FLOAT_GOOD, None, 0.94, np.float32(0.94), 0.001])

# RiskMetrics2006
TAU_GOOD = [2000, 2000.0, np.float32(2000.0), np.float64(2000.0), np.int64(2000)]
add_bad(RiskMetrics2006, "tau0", [*FLOAT_BAD, (0.0, ValueError), (4.0, ValueError)])
add_good(RiskMetrics2006, "tau0", TAU_GOOD)
add_bad(RiskMetrics2006, "tau1", [*FLOAT_BAD, (0.0, ValueError), (-1.0, ValueError)])
CONSTRUCTOR_ROWS.append(
    Row(
        "RiskMetrics2006",
        make(RiskMetrics2006),
        {"tau0": 1560, "tau1": 2000},
        ValueError,
        r"^tau0 must be a finite float greater than tau1 \(2000\); got 1560 \(int\)$",
        id="RiskMetrics2006-tau0-not-above-tau1",
        name="tau0",
    )
)
add_good(RiskMetrics2006, "tau1", [*FLOAT_GOOD, 1, 4.0])
add_bad(RiskMetrics2006, "kmax", [*INT_BAD, *COUNT_BAD_1, (20, ValueError)])
add_good(RiskMetrics2006, "kmax", [1, *INT_GOOD])
add_bad(RiskMetrics2006, "rho", [*FLOAT_BAD, (1.0, ValueError), (0.5, ValueError)])
add_good(
    RiskMetrics2006, "rho", [1.5, np.float32(1.5), np.float64(1.5), Fraction(3, 2)]
)

# EGARCH
for order in ("p", "o", "q"):
    add_bad(EGARCH, order, [*INT_BAD, *COUNT_BAD_0])
    add_good(EGARCH, order, INT_GOOD if order == "p" else [*INT_GOOD, 0], {"p": 1})
CONSTRUCTOR_ROWS.append(
    Row(
        "EGARCH",
        make(EGARCH),
        {"p": 0, "o": 0, "q": 1},
        ValueError,
        r"^p must be an integer >= 1 when o is 0; got 0 \(int\)$",
        id="EGARCH-p-o-both-zero",
        name="p",
    )
)

# FixedVariance
VARIANCE = np.linspace(0.5, 1.5, 20)
VARIANCE_BAD = [
    ("abc", TypeError),
    (None, TypeError),
    (["a", "b"], TypeError),
    (1j, TypeError),
    (np.ones((2, 2)), ValueError),
    (np.r_[VARIANCE, np.nan], ValueError),
    (np.r_[VARIANCE, np.inf], ValueError),
    (np.r_[VARIANCE, -np.inf], ValueError),
    (np.r_[VARIANCE, 0.0], ValueError),
    (np.r_[VARIANCE, -1.0], ValueError),
    (pd.Series(np.r_[VARIANCE, np.nan]), ValueError),
]
VARIANCE_GOOD = [
    VARIANCE,
    list(VARIANCE),
    VARIANCE.astype(np.float32),
    np.arange(1, 21),
    pd.Series(VARIANCE),
    pd.DataFrame({"v": VARIANCE}),
    VARIANCE[:, None],
]
add_bad(FixedVariance, "variance", VARIANCE_BAD, {})
add_good(FixedVariance, "variance", VARIANCE_GOOD, {})
add_bad(FixedVariance, "unit_scale", BOOL_BAD, {"variance": VARIANCE})
add_good(FixedVariance, "unit_scale", BOOL_GOOD, {"variance": VARIANCE})

# FIGARCH
add_bad(FIGARCH, "p", [*INT_BAD, *ORDER01_BAD])
add_good(FIGARCH, "p", [0, 1, np.int8(1), np.int64(0)])
add_bad(FIGARCH, "q", [*INT_BAD, *ORDER01_BAD])
add_good(FIGARCH, "q", [0, 1, np.int8(1), np.int64(0)])
add_bad(FIGARCH, "power", [*FLOAT_BAD, *POWER_BAD])
add_good(FIGARCH, "power", [*FLOAT_GOOD, 1, 2.0])
add_bad(FIGARCH, "truncation", [*INT_BAD, *COUNT_BAD_1])
add_good(FIGARCH, "truncation", [1, *INT_GOOD])

# FIAPARCH
for order in ("p", "o", "q"):
    add_bad(FIAPARCH, order, [*INT_BAD, *ORDER01_BAD])
    add_good(FIAPARCH, order, [0, 1, np.int8(1), np.int64(0)])
add_bad(FIAPARCH, "delta", [*not_none(FLOAT_BAD), *DELTA_BAD])
add_good(FIAPARCH, "delta", DELTA_GOOD)
add_bad(FIAPARCH, "truncation", [*INT_BAD, *COUNT_BAD_1])
add_good(FIAPARCH, "truncation", [1, *INT_GOOD])

# APARCH
add_bad(APARCH, "p", [*INT_BAD, *COUNT_BAD_1])
add_good(APARCH, "p", INT_GOOD)
add_bad(APARCH, "o", [*INT_BAD, *COUNT_BAD_0, (2, ValueError)], {"p": 1})
add_good(APARCH, "o", [0, 1, np.int8(1), np.int64(0)], {"p": 1})
add_bad(APARCH, "q", [*INT_BAD, *COUNT_BAD_0])
add_good(APARCH, "q", [*INT_GOOD, 0])
add_bad(APARCH, "delta", [*not_none(FLOAT_BAD), *DELTA_BAD])
add_good(APARCH, "delta", DELTA_GOOD)
add_bad(APARCH, "common_asym", BOOL_BAD)
add_good(APARCH, "common_asym", BOOL_GOOD)

# BootstrapRng
STD_RESID = RESIDS[:200]
BOOTSTRAP_BASE = {"std_resid": STD_RESID, "start": 100}
add_bad(
    BootstrapRng,
    "start",
    [*INT_BAD, (0, ValueError), (-1, ValueError), (201, ValueError)],
    BOOTSTRAP_BASE,
)
add_good(BootstrapRng, "start", [1, 200, *INT_GOOD], BOOTSTRAP_BASE)
add_bad(
    BootstrapRng,
    "random_state",
    [
        (5, TypeError),
        ("a", TypeError),
        (np.random.default_rng(0), TypeError),
        (np.random.SeedSequence(0), TypeError),
    ],
    BOOTSTRAP_BASE,
)
add_good(BootstrapRng, "random_state", [None, np.random.RandomState(0)], BOOTSTRAP_BASE)
add_bad(
    BootstrapRng,
    "std_resid",
    [
        ("abc", TypeError),
        (None, TypeError),
        (np.ones((2, 2)), ValueError),
    ],
    BOOTSTRAP_BASE,
)
add_good(
    BootstrapRng,
    "std_resid",
    [list(STD_RESID), pd.Series(STD_RESID), np.r_[STD_RESID, np.nan]],
    BOOTSTRAP_BASE,
)

SIZE_ROWS = [
    # The largest integers that fit in a C int
    Row("GARCH", make(GARCH), {"p": INT_MAX}, None, id="GARCH-p-int32-max"),
    Row("HARCH", make(HARCH), {"lags": [1, INT_MAX]}, None, id="HARCH-lags-int32-max"),
    Row(
        "RiskMetrics2006",
        make(RiskMetrics2006),
        {"kmax": INT_MAX},
        ValueError,
        r"^kmax must be such that tau1 \* rho \*\* \(kmax - 1\) <= tau0 \(1560\); got",
        id="RiskMetrics2006-kmax-overflow",
        name="kmax",
    ),
]


@pytest.mark.parametrize("row", to_params(CONSTRUCTOR_ROWS))
def test_constructor(row):
    run_row(row)


@pytest.mark.parametrize("row", to_params(SIZE_ROWS))
def test_constructor_size(row):
    run_row(row)


def test_large_order_is_not_rejected():
    # Orders larger than the sample are rejected by fit (decision 73), not here
    assert GARCH(p=300, q=300).num_params == 601


# The simulation of every process
def normal_rng(size):
    return np.random.default_rng(0).standard_normal(size)


# The number of initial values of each process: None if it is ignored and 0 if only a
# scalar is accepted
PROCESSES = {
    "constant": (ConstantVariance, None),
    "garch": (lambda: GARCH(1, 1, 1), 1),
    "arch": (lambda: ARCH(2), 2),
    "harch": (lambda: HARCH([1, 5]), 5),
    "midas": (lambda: MIDASHyperbolic(m=6), 6),
    "midas-asym": (lambda: MIDASHyperbolic(m=6, asym=True), 6),
    "ewma-estimated": (lambda: EWMAVariance(None), 0),
    "ewma-fixed": (lambda: EWMAVariance(0.94), 0),
    "rm2006": (RiskMetrics2006, 0),
    "egarch": (lambda: EGARCH(1, 1, 1), 1),
    "figarch": (lambda: FIGARCH(truncation=30), 30),
    "fiaparch": (lambda: FIAPARCH(truncation=30), 30),
    "aparch": (lambda: APARCH(1, 1, 1), 1),
}


def simulator(key):
    factory = PROCESSES[key][0]

    def func(**kwargs):
        process = factory()
        parameters = process.starting_values(RESIDS)
        return process.simulate(parameters, **kwargs)

    return func


SIM_BASE = {"nobs": 20, "rng": normal_rng, "burn": 5}
SIM_NOBS_BAD = [*INT_BAD, (0, ValueError), (-1, ValueError), (np.int64(-7), ValueError)]
SIM_BURN_BAD = [*INT_BAD, (-1, ValueError), (np.int64(-7), ValueError)]
SIM_RNG_BAD = [(None, TypeError), (5, TypeError), ("abc", TypeError)]
SIM_ROWS: list[Row] = []
for key, (_, size) in PROCESSES.items():
    func = simulator(key)
    SIM_ROWS.extend(bad_value_rows(key, func, SIM_BASE, "nobs", SIM_NOBS_BAD))
    SIM_ROWS.extend(bad_value_rows(key, func, SIM_BASE, "burn", SIM_BURN_BAD))
    SIM_ROWS.extend(bad_value_rows(key, func, SIM_BASE, "rng", SIM_RNG_BAD))
    SIM_ROWS.extend(good_value_rows(key, func, SIM_BASE, "nobs", [1, *INT_GOOD]))
    SIM_ROWS.extend(good_value_rows(key, func, SIM_BASE, "burn", [0, *INT_GOOD]))
    if size is None:
        # ConstantVariance ignores the initial value
        continue
    init_bad = [
        ("a", TypeError),
        (True, TypeError),
        (np.True_, TypeError),
        (1j, TypeError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (-np.inf, ValueError),
    ]
    init_good = [None, 0.0, 1.0, 2, np.float32(1.5), np.int64(3), np.float64(0.5)]
    if size == 0:
        # Only a scalar
        init_bad.extend([(np.ones(1), TypeError), ([1.0], TypeError)])
    else:
        init_bad.extend(
            [
                (np.full(size, np.nan), ValueError),
                (np.full(size, np.inf), ValueError),
                (np.ones(size + 1), ValueError),
                (np.ones(0), ValueError),
                (np.ones((2, 2)), ValueError),
            ]
        )
        init_good.extend(
            [
                np.ones(size),
                [1.0] * size,
                np.array([1.0]),
                np.float32(2.0) * np.ones(size, dtype=np.float32),
            ]
        )
    if key == "egarch":
        # The log of the variance can be negative
        init_good.extend([-1.0, np.float32(-2.0), -np.ones(size)])
    else:
        init_bad.append((-1.0, ValueError))
        if size:
            init_bad.append((-np.ones(size), ValueError))
    SIM_ROWS.extend(
        bad_value_rows(key, func, SIM_BASE, "initial_value", init_bad)
        + good_value_rows(key, func, SIM_BASE, "initial_value", init_good)
    )


@pytest.mark.parametrize("row", to_params(SIM_ROWS))
def test_simulate(row):
    run_row(row)


def test_constant_variance_ignores_initial_value():
    process = ConstantVariance()
    parameters = np.array([2.0])
    for value in ("a", np.nan, -1.0, np.ones(7)):
        data, variance = process.simulate(parameters, 10, normal_rng, 3, value)
        assert data.shape == variance.shape == (10,)
        assert np.all(variance == 2.0)


def test_fixed_variance_does_not_simulate():
    process = FixedVariance(VARIANCE)
    with pytest.raises(NotImplementedError, match="do not support simulation"):
        process.simulate([], 10, normal_rng)


# The forecast
PROCESS = GARCH(1, 0, 1)
PARAMETERS = np.array([0.05, 0.1, 0.85])
BACKCAST = PROCESS.backcast(RESIDS)
VAR_BOUNDS = PROCESS.variance_bounds(RESIDS)
FORECAST_BASE = {
    "parameters": PARAMETERS,
    "resids": RESIDS,
    "backcast": BACKCAST,
    "var_bounds": VAR_BOUNDS,
}
SIMULATION = {
    **FORECAST_BASE,
    "method": "simulation",
    "simulations": 5,
    "rng": normal_rng,
}


def forecast(**kwargs):
    return PROCESS.forecast(**kwargs)


FORECAST_ROWS: list[Row] = []


def add_forecast_bad(name, bad, base=FORECAST_BASE):
    target = "forecast-simulation" if base is SIMULATION else "forecast"
    FORECAST_ROWS.extend(bad_value_rows(target, forecast, base, name, bad))


def add_forecast_good(name, good, base=FORECAST_BASE):
    target = "forecast-simulation" if base is SIMULATION else "forecast"
    FORECAST_ROWS.extend(good_value_rows(target, forecast, base, name, good))


add_forecast_bad("horizon", [*INT_BAD, (0, ValueError), (-1, ValueError)])
add_forecast_good("horizon", [1, *INT_GOOD])
add_forecast_bad(
    "start",
    [
        *not_none(INT_BAD),
        (-301, ValueError),
        (np.int64(-310), ValueError),
        (301, ValueError),
    ],
)
add_forecast_good("start", [None, 0, 1, 299, 300, -1, -300, np.int64(-10), *INT_GOOD])
add_forecast_bad("simulations", [*INT_BAD, (0, ValueError), (-1, ValueError)])
add_forecast_bad(
    "simulations", [*INT_BAD, (0, ValueError), (-1, ValueError)], SIMULATION
)
add_forecast_good("simulations", [1, *INT_GOOD], SIMULATION)
add_forecast_bad("rng", [(5, TypeError), ("abc", TypeError)])
add_forecast_bad(
    "rng", [(None, TypeError), (5, TypeError), ("abc", TypeError)], SIMULATION
)
add_forecast_good("rng", [None, normal_rng])
add_forecast_bad(
    "random_state",
    [
        (5, TypeError),
        ("a", TypeError),
        (np.random.default_rng(0), TypeError),
    ],
)
add_forecast_good("random_state", [None, np.random.RandomState(0)])
add_forecast_bad(
    "asym_weight", [*FLOAT_BAD, (-0.1, ValueError), (1.1, ValueError), (2, ValueError)]
)
add_forecast_good(
    "asym_weight", [*FLOAT_GOOD, 0.0, 1.0, 0, 1, np.int64(1), np.float64(0.25)]
)


@pytest.mark.parametrize("row", to_params(FORECAST_ROWS))
def test_forecast(row):
    run_row(row)


def test_forecast_default_start_is_the_last_observation():
    default = PROCESS.forecast(**FORECAST_BASE)
    explicit = PROCESS.forecast(**FORECAST_BASE, start=RESIDS.shape[0] - 1)
    assert default.forecasts is not None
    assert default.forecasts.shape == (1, 1)
    np.testing.assert_array_equal(default.forecasts, explicit.forecasts)


def test_forecast_start_at_the_end_has_no_rows():
    result = PROCESS.forecast(**FORECAST_BASE, start=RESIDS.shape[0], horizon=2)
    assert result.forecasts is not None
    assert result.forecasts.shape == (0, 2)


def test_ewma_float32_lam():
    # np.float32 failed an assert in compute_variance
    lam = np.float32(0.9)
    process = EWMAVariance(lam)
    assert type(process.lam) is float
    assert process.lam == float(lam)
    process.start, process.stop = 0, RESIDS.shape[0]
    sigma2 = np.empty(RESIDS.shape[0])
    backcast = process.backcast(RESIDS)
    bounds = process.variance_bounds(RESIDS)
    process.compute_variance(np.empty(0), RESIDS, sigma2, backcast, bounds)
    expected = np.empty(RESIDS.shape[0])
    expected[0] = backcast
    for t in range(1, RESIDS.shape[0]):
        expected[t] = (
            float(lam) * expected[t - 1] + (1 - float(lam)) * RESIDS[t - 1] ** 2
        )
    np.testing.assert_allclose(sigma2, expected, rtol=1e-12)


def test_numpy_scalars_become_python_numbers():
    garch = GARCH(np.int64(2), np.int8(1), np.uint16(3), np.float32(1.5))
    assert [type(v) for v in (garch.p, garch.o, garch.q, garch.power)] == [
        int,
        int,
        int,
        float,
    ]
    assert garch.num_params == 1 + 2 + 1 + 3
    assert type(MIDASHyperbolic(np.int32(5), np.True_)._asym) is bool
    rm = RiskMetrics2006(np.float32(2000), np.int16(4), np.int8(5), np.float32(1.5))
    assert [type(v) for v in (rm.tau0, rm.tau1, rm.kmax, rm.rho)] == [
        float,
        float,
        int,
        float,
    ]
    aparch = APARCH(np.int64(2), np.int64(1), np.int64(1), np.float32(1.5), np.True_)
    assert type(aparch.delta) is float
    assert aparch.common_asym is True


def test_harch_lags_are_int32():
    for lags in (3, [1, 5], np.array([2, 4], dtype=np.int64), range(1, 3)):
        harch = HARCH(lags)
        assert harch.lags.dtype == np.int32
    np.testing.assert_array_equal(HARCH(np.int64(3)).lags, [1, 2, 3])
    np.testing.assert_array_equal(HARCH([5, 1]).lags, [5, 1])
    assert HARCH(np.array([1, 5])).num_params == 3
