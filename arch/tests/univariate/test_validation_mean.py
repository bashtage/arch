"""
Table-driven tests of the input validation of the mean models.

Covers ``arch_model`` and the constructors, ``simulate``, ``forecast``, ``resids``,
``append`` and the sample arguments of ``fit`` and ``fix`` of ``HARX``, ``ARX``, ``LS``,
``ConstantMean``, ``ZeroMean``, ``MAMean`` and ``ARCHInMean``.
"""

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
    ARX,
    GARCH,
    HARX,
    LS,
    ARCHInMean,
    ConstantMean,
    MAMean,
    Normal,
    ZeroMean,
    arch_model,
)

NOBS = 120
RNG = np.random.default_rng(20241017)
Y = RNG.standard_normal(NOBS)
X = RNG.standard_normal((NOBS, 2))
X_NAN = X.copy()
X_NAN[3, 0] = np.nan
X_INF = X.copy()
X_INF[3, 0] = np.inf
NAN_Y = Y.copy()
NAN_Y[4] = np.nan

NOT_NONE = [item for item in BOOL_BAD if item[0] is not None]
INT_NOT_NONE = [item for item in INT_BAD if item[0] is not None]
# Values for an integer that must be at least 0 or 1
NEGATIVE = [(-1, ValueError), (np.int64(-5), ValueError)]
WRONG_CLASS = [
    ("garch", TypeError),
    (5, TypeError),
    (GARCH, TypeError),
    (Normal(), TypeError),
]
WRONG_DIST = [("normal", TypeError), (5, TypeError), (Normal, TypeError)]
FINITE_BAD = [(np.nan, ValueError), (np.inf, ValueError), (-np.inf, ValueError)]


def harx(**kw):
    return HARX(**kw)


def arx(**kw):
    return ARX(**kw)


def ls(**kw):
    return LS(**kw)


def arch_in_mean(**kw):
    return ARCHInMean(**kw)


def constant_mean(**kw):
    return ConstantMean(**kw)


def zero_mean(**kw):
    return ZeroMean(**kw)


def ma_mean(**kw):
    return MAMean(**kw)


def model(**kw):
    return arch_model(**kw)


def with_hold_back(kwargs):
    return {**kwargs, "hold_back": None}


# The constructors
HARX_BASE = {"y": Y, "x": X, "lags": [1, 2]}
ARX_BASE = {"y": Y, "x": X, "lags": 2}
GIM_BASE = {"y": Y, "x": X, "lags": 2, "volatility": GARCH()}
LS_BASE = {"y": Y, "x": X}
SIMPLE_BASE = {"y": Y}
MA_BASE = {"y": Y, "lags": 2}
CONSTRUCTORS = {
    "HARX": (harx, HARX_BASE),
    "ARX": (arx, ARX_BASE),
    "ARCHInMean": (arch_in_mean, GIM_BASE),
    "LS": (ls, LS_BASE),
    "ConstantMean": (constant_mean, SIMPLE_BASE),
    "ZeroMean": (zero_mean, SIMPLE_BASE),
    "MAMean": (ma_mean, MA_BASE),
}
WITH_CONSTANT = ("HARX", "ARX", "ARCHInMean", "LS", "MAMean")
WITH_X = ("HARX", "ARX", "ARCHInMean", "LS")

ROWS: list[Row] = []

for _name, (_func, _base) in CONSTRUCTORS.items():
    ROWS += bad_value_rows(_name, _func, _base, "hold_back", INT_NOT_NONE + NEGATIVE)
    ROWS += good_value_rows(_name, _func, _base, "hold_back", [*INT_GOOD, None])
    ROWS += bad_value_rows(_name, _func, _base, "rescale", NOT_NONE)
    ROWS += good_value_rows(_name, _func, _base, "rescale", [*BOOL_GOOD, None])
    ROWS += bad_value_rows(_name, _func, _base, "volatility", WRONG_CLASS)
    ROWS += good_value_rows(
        _name,
        _func,
        _base,
        "volatility",
        [GARCH()] if _name == "ARCHInMean" else [None, GARCH()],
    )
    ROWS += bad_value_rows(_name, _func, _base, "distribution", WRONG_DIST)
    ROWS += good_value_rows(_name, _func, _base, "distribution", [None, Normal()])
    if _name in WITH_CONSTANT:
        ROWS += bad_value_rows(_name, _func, _base, "constant", BOOL_BAD)
        ROWS += good_value_rows(_name, _func, _base, "constant", BOOL_GOOD)
    if _name in WITH_X:
        _bad_x = [
            (X_NAN, ValueError),
            (X_INF, ValueError),
            (X[:-1], ValueError),
            (X[:, :, None], ValueError),
        ]
        ROWS += bad_value_rows(_name, _func, _base, "x", _bad_x)
        ROWS += good_value_rows(_name, _func, _base, "x", [X, X[:, 0], None])
        # The message of the conversion of the data to float64
        ROWS.append(
            Row(
                _name,
                _func,
                {**_base, "x": np.full((NOBS, 2), "a")},
                TypeError,
                r"^x must be array_like with integer, Boolean or floating point",
                id=f"{_name}-x-strings",
            )
        )

ROWS += bad_value_rows("HARX", harx, HARX_BASE, "use_rotated", BOOL_BAD)
ROWS += good_value_rows("HARX", harx, HARX_BASE, "use_rotated", BOOL_GOOD)

# lags
INT_LAGS = [item for item in INT_NOT_NONE if not isinstance(item[0], list)]
ARRAY_LAGS_BAD = [
    ([1.5, 2], TypeError),
    ([1.0, 2.0], TypeError),
    (np.array([1.0, 2.0]), TypeError),
    (["a", "b"], TypeError),
    ([True, False], TypeError),
    ([1, None], TypeError),
    ([0, 1], ValueError),
    ([1, -2], ValueError),
    ([], ValueError),
    (np.empty(0, dtype=int), ValueError),
    ([[1, 2], [3]], ValueError),
    (np.ones((2, 2, 2), dtype=int), ValueError),
]
TWO_D_LAGS_BAD = [
    ([[0], [1]], ValueError),
    ([[2], [1]], ValueError),
    ([[1, 2, 3]], ValueError),
    (np.eye(3), ValueError),
    ([[1.0], [2.0]], TypeError),
    ([[1], [2], [3]], ValueError),
]
ROWS += bad_value_rows("HARX-lags", harx, HARX_BASE, "lags", INT_LAGS + NEGATIVE)
ROWS += bad_value_rows("HARX-lags-array", harx, HARX_BASE, "lags", ARRAY_LAGS_BAD)
ROWS += bad_value_rows("HARX-lags-2d", harx, HARX_BASE, "lags", TWO_D_LAGS_BAD)
ROWS += [
    Row(
        "HARX-lags-redundant",
        harx,
        {**HARX_BASE, "lags": [[1, 1, 3], [2, 3, 3]]},
        ValueError,
        r"lags contains redundant entries",
        id="HARX-lags-redundant",
        form=False,
    ),
    Row(
        "HARX-lags-large",
        harx,
        {**HARX_BASE, "lags": NOBS},
        ValueError,
        r"^lags must be smaller than the number of observations \(120\); got a largest",
        id="HARX-lags-large",
        name="lags",
    ),
]
GOOD_LAGS = [
    0,
    2,
    np.int8(2),
    np.int64(2),
    np.array(2),
    [1, 5],
    (1, 5),
    np.array([1, 5]),
    range(1, 4),
    [[1, 2], [1, 5]],
    np.array([[1, 3], [2, 5]]),
    None,
]
ROWS += good_value_rows("HARX-lags", harx, HARX_BASE, "lags", GOOD_LAGS)
for _name in ("ARX", "ARCHInMean"):
    _func, _base = CONSTRUCTORS[_name]
    ROWS += bad_value_rows(_name + "-lags", _func, _base, "lags", INT_LAGS + NEGATIVE)
    ROWS += bad_value_rows(
        _name + "-lags-array",
        _func,
        _base,
        "lags",
        ARRAY_LAGS_BAD
        + TWO_D_LAGS_BAD[:2]
        + [(np.array([[1, 2], [3, 4]]), ValueError)],
    )
    ROWS += good_value_rows(
        _name + "-lags",
        _func,
        _base,
        "lags",
        [0, 2, np.int16(2), np.int64(2), [1, 5], (1, 5), np.array([1, 5]), None],
    )
ROWS += bad_value_rows("MAMean-lags", ma_mean, MA_BASE, "lags", INT_NOT_NONE)
ROWS += bad_value_rows("MAMean-lags-value", ma_mean, MA_BASE, "lags", [(0, ValueError)])
ROWS += bad_value_rows("MAMean-lags-neg", ma_mean, MA_BASE, "lags", NEGATIVE)
ROWS += good_value_rows("MAMean-lags", ma_mean, MA_BASE, "lags", [1, *INT_GOOD])

# form of ARCHInMean
FORM_BAD = [
    (0, ValueError),
    (0.0, ValueError),
    (np.float32(0.0), ValueError),
    ("unknown", ValueError),
    ("", ValueError),
    (np.nan, ValueError),
    (np.inf, ValueError),
    (None, TypeError),
    (True, TypeError),
    (np.True_, TypeError),
    (1j, TypeError),
    ([1.0], TypeError),
]
ROWS += bad_value_rows("ARCHInMean-form", arch_in_mean, GIM_BASE, "form", FORM_BAD)
ROWS += good_value_rows(
    "ARCHInMean-form",
    arch_in_mean,
    GIM_BASE,
    "form",
    ["vol", "VAR", "Log", 2, 1.5, -1, np.int64(2), np.float32(1.5), np.float16(0.5)],
)

# The sample used by fit and fix
FIT_FUNCS = {
    "ConstantMean": lambda **kw: ConstantMean(Y).fit(disp="off", **kw),
    "HARX": lambda **kw: HARX(Y, lags=[1, 2]).fit(disp="off", **kw),
    "ARX": lambda **kw: ARX(Y, X, lags=1).fit(disp="off", **kw),
    "MAMean": lambda **kw: MAMean(Y, lags=1).fit(disp="off", **kw),
}
FIX_FUNCS = {
    "ConstantMean": lambda **kw: ConstantMean(Y).fix([0.1, 1.0], **kw),
    "MAMean": lambda **kw: MAMean(Y, lags=1).fix([0.1, 0.2, 1.0], **kw),
}
SAMPLE_BAD = [
    (-NOBS - 1, ValueError),
    (np.int64(-1000), ValueError),
    (NOBS + 1, ValueError),
    (10**6, ValueError),
    (2.5, TypeError),
    (5.0, TypeError),
    (True, TypeError),
    ([3], TypeError),
    ("abc", ValueError),
    ("2000-01-05", ValueError),
]
for _name, _func in FIT_FUNCS.items():
    for _arg in ("first_obs", "last_obs"):
        ROWS += bad_value_rows(f"fit-{_name}-{_arg}", _func, {}, _arg, SAMPLE_BAD)
        ROWS += good_value_rows(
            f"fit-{_name}-{_arg}", _func, {}, _arg, [None, np.int64(10), -10]
        )
for _name, _func in FIX_FUNCS.items():
    ROWS += bad_value_rows(f"fix-{_name}", _func, {}, "first_obs", SAMPLE_BAD)
    ROWS += bad_value_rows(f"fix-{_name}", _func, {}, "last_obs", SAMPLE_BAD)
ROWS += [
    Row(
        "fit-last-first",
        FIT_FUNCS["ConstantMean"],
        {"first_obs": 50, "last_obs": 40},
        ValueError,
        r"^last_obs must be larger than first_obs \(50\); got 40 \(int\)$",
        id="fit-last-first",
        name="last_obs",
    ),
    Row(
        "fit-first-end",
        FIT_FUNCS["ConstantMean"],
        {"first_obs": NOBS},
        ValueError,
        r"^first_obs must be smaller than the number of observations \(120\)",
        id="fit-first-end",
        name="first_obs",
    ),
    Row(
        "fit-bounds-ok",
        FIT_FUNCS["ConstantMean"],
        {"first_obs": 0, "last_obs": NOBS},
        None,
        id="fit-bounds-ok",
    ),
    Row(
        "fit-hold-back",
        lambda **kw: ConstantMean(Y, hold_back=kw["hold_back"]).fit(disp="off"),
        {"hold_back": NOBS},
        ValueError,
        r"^hold_back must be smaller than the number of observations in the estimation"
        r" sample \(120\)",
        id="fit-hold-back",
        name="hold_back",
    ),
    Row(
        "fit-hold-back-window",
        lambda **kw: ConstantMean(Y, hold_back=30).fit(disp="off", **kw),
        {"first_obs": 10, "last_obs": 40},
        ValueError,
        r"^hold_back must be smaller than the number of observations in the estimation"
        r" sample \(30\), where hold_back is at least the largest lag; got 30 \(int\)",
        id="fit-hold-back-window",
        name="hold_back",
    ),
    Row(
        "fit-lags-sample",
        lambda **kw: ARX(Y, lags=10).fit(disp="off", **kw),
        {"last_obs": 10},
        ValueError,
        r"^hold_back must be smaller than the number of observations in the estimation"
        r" sample \(10\), where hold_back is at least the largest lag; got 10 \(int\)",
        id="fit-lags-sample",
        name="hold_back",
    ),
    Row(
        "fit-empty-y",
        lambda **kw: ConstantMean(np.empty(0)).fit(disp="off"),
        {},
        ValueError,
        r"^y must be a 1-d array of finite numbers with at least 1 element; got",
        id="fit-empty-y",
        name="y",
    ),
    Row(
        "fit-empty-y-ma",
        lambda **kw: MAMean(np.empty(0)).fit(disp="off"),
        {},
        ValueError,
        r"^y must be a 1-d array of finite numbers with at least 1 element; got",
        id="fit-empty-y-ma",
        name="y",
    ),
]

# simulate
NSIM = {"nobs": 10, "burn": 5}
SIM_MODELS = {
    "HARX": (
        lambda: HARX(lags=[1, 2], volatility=GARCH()),
        np.r_[0.1, 0.2, 0.1, 0.05, 0.1, 0.8],
    ),
    "ARX": (lambda: ARX(lags=1, volatility=GARCH()), np.r_[0.1, 0.2, 0.05, 0.1, 0.8]),
    "ARCHInMean": (
        lambda: ARCHInMean(lags=1, volatility=GARCH()),
        np.r_[0.1, 0.2, 0.05, 0.05, 0.1, 0.8],
    ),
    "LS": (lambda: LS(volatility=GARCH()), np.r_[0.1, 0.05, 0.1, 0.8]),
    "ConstantMean": (
        lambda: ConstantMean(volatility=GARCH()),
        np.r_[0.1, 0.05, 0.1, 0.8],
    ),
    "ZeroMean": (lambda: ZeroMean(volatility=GARCH()), np.r_[0.05, 0.1, 0.8]),
    "MAMean": (
        lambda: MAMean(lags=2, volatility=GARCH()),
        np.r_[0.1, 0.3, 0.1, 0.05, 0.1, 0.8],
    ),
}
BAD_PARAMS = [("abc", TypeError), (None, TypeError), (1j, TypeError)]
for _name, (_make, _params) in SIM_MODELS.items():

    def _sim(_make=_make, **kw):
        return _make().simulate(**kw)

    _base = {"params": _params, **NSIM}
    ROWS += bad_value_rows(f"simulate-{_name}", _sim, _base, "nobs", INT_BAD)
    ROWS += bad_value_rows(
        f"simulate-{_name}-value", _sim, _base, "nobs", [(0, ValueError), *NEGATIVE]
    )
    ROWS += good_value_rows(f"simulate-{_name}", _sim, _base, "nobs", INT_GOOD)
    ROWS += bad_value_rows(f"simulate-{_name}", _sim, _base, "burn", INT_BAD)
    ROWS += bad_value_rows(f"simulate-{_name}-value", _sim, _base, "burn", NEGATIVE)
    ROWS += good_value_rows(f"simulate-{_name}", _sim, _base, "burn", [0, *INT_GOOD])
    _bad_params = [
        (_params[:-1], ValueError),
        (np.r_[_params, 0.5], ValueError),
        (np.empty(0), ValueError),
        (np.tile(_params, (2, 2)), ValueError),
        (np.where(np.arange(_params.shape[0]) == 1, np.nan, _params), ValueError),
        (np.where(np.arange(_params.shape[0]) == 1, np.inf, _params), ValueError),
        *BAD_PARAMS,
    ]
    ROWS += bad_value_rows(f"simulate-{_name}", _sim, _base, "params", _bad_params)
    ROWS += good_value_rows(
        f"simulate-{_name}", _sim, _base, "params", [_params, list(_params)]
    )
    ROWS += bad_value_rows(
        f"simulate-{_name}",
        _sim,
        _base,
        "initial_value_vol",
        [
            ("abc", TypeError),
            (True, TypeError),
            (np.nan, ValueError),
            (np.inf, ValueError),
            (np.array([1.0, np.nan]), ValueError),
        ],
    )
    ROWS += good_value_rows(
        f"simulate-{_name}", _sim, _base, "initial_value_vol", [None, 1.0, *FLOAT_GOOD]
    )

# initial_value and x of the models with mean dynamics and regressors
HARX_SIM = {"params": SIM_MODELS["HARX"][1], **NSIM}
ARX_SIM = {"params": SIM_MODELS["ARX"][1], **NSIM}
GIM_SIM = {"params": SIM_MODELS["ARCHInMean"][1], **NSIM}
BAD_INITIAL = [
    ("0.5", TypeError),
    (True, TypeError),
    (1j, TypeError),
    (np.nan, ValueError),
    (np.inf, ValueError),
    ([0.5] * 4, ValueError),
    ([0.5, 0.1, 0.3], ValueError),
    ([0.5, np.nan], ValueError),
    (np.ones((2, 2)), ValueError),
]
for _name, _base in (("HARX", HARX_SIM), ("ARX", ARX_SIM), ("ARCHInMean", GIM_SIM)):
    _make = SIM_MODELS[_name][0]
    _initial = [0.5, 0.25] if _name == "HARX" else [0.5]

    def _sim(_make=_make, **kw):
        return _make().simulate(**kw)

    ROWS += bad_value_rows(
        f"simulate-{_name}", _sim, _base, "initial_value", BAD_INITIAL
    )
    ROWS += good_value_rows(
        f"simulate-{_name}",
        _sim,
        _base,
        "initial_value",
        [None, 0.5, 1, np.float32(0.5), np.array(0.5), _initial, np.array(_initial)],
    )

X_SIM_ROWS = NSIM["nobs"] + NSIM["burn"]
X_SIM = RNG.standard_normal((X_SIM_ROWS, 2))
X_SIM_NAN = X_SIM.copy()
X_SIM_NAN[0, 0] = np.nan
HARX_X_SIM = {"params": np.r_[0.1, 0.2, 0.1, 0.3, 0.4, 0.05, 0.1, 0.8], **NSIM}


def _sim_x(**kw):
    return HARX(lags=[1, 2], volatility=GARCH()).simulate(**kw)


ROWS += bad_value_rows(
    "simulate-HARX-x",
    _sim_x,
    HARX_X_SIM,
    "x",
    [
        (X_SIM[:-1], ValueError),
        (X_SIM[1:], ValueError),
        (X_SIM_NAN, ValueError),
        (np.full((X_SIM_ROWS, 2), "a"), TypeError),
        (X_SIM[:, :, None], ValueError),
        (5.0, ValueError),
    ],
)
ROWS += good_value_rows("simulate-HARX-x", _sim_x, HARX_X_SIM, "x", [X_SIM])
for _name in ("ConstantMean", "ZeroMean", "MAMean"):
    _make, _params = SIM_MODELS[_name]
    _base = {"params": _params, **NSIM}

    def _sim_none(_make=_make, **kw):
        return _make().simulate(**kw)

    ROWS += bad_value_rows(
        f"simulate-{_name}-none",
        _sim_none,
        _base,
        "initial_value",
        [(0.5, ValueError), ([0.5], ValueError), (np.nan, ValueError)],
    )
    ROWS += bad_value_rows(
        f"simulate-{_name}-none",
        _sim_none,
        _base,
        "x",
        [(X_SIM, ValueError), (np.array(10), ValueError), (0.5, ValueError)],
    )

# forecast
FIT_PARAMS = {
    "ConstantMean": ConstantMean(Y, volatility=GARCH()),
    "ZeroMean": ZeroMean(Y, volatility=GARCH()),
    "ARX": ARX(Y, lags=2, volatility=GARCH()),
    "HARX": HARX(Y, lags=[1, 3], volatility=GARCH()),
    "MAMean": MAMean(Y, lags=1, volatility=GARCH()),
}
FORECAST_PARAMS = {
    "ConstantMean": np.r_[0.1, 0.05, 0.1, 0.8],
    "ZeroMean": np.r_[0.05, 0.1, 0.8],
    "ARX": np.r_[0.1, 0.1, 0.1, 0.05, 0.1, 0.8],
    "HARX": np.r_[0.1, 0.1, 0.1, 0.05, 0.1, 0.8],
    "MAMean": np.r_[0.1, 0.1, 0.05, 0.1, 0.8],
}


def _forecast(name):
    def func(**kw):
        return FIT_PARAMS[name].forecast(**kw)

    return func


for _name, _params in FORECAST_PARAMS.items():
    _func = _forecast(_name)
    _base = {"params": _params, "horizon": 2}
    _bad_params = [
        (_params[:-1], ValueError),
        (np.r_[_params, 1.0], ValueError),
        (np.empty(0), ValueError),
        (np.tile(_params, (2, 2)), ValueError),
        (np.where(np.arange(_params.shape[0]) == 1, np.nan, _params), ValueError),
        (np.where(np.arange(_params.shape[0]) == 1, np.inf, _params), ValueError),
        ("abc", TypeError),
        (None, TypeError),
    ]
    ROWS += bad_value_rows(f"forecast-{_name}", _func, _base, "params", _bad_params)
    ROWS += good_value_rows(
        f"forecast-{_name}",
        _func,
        _base,
        "params",
        [_params, list(_params), pd.Series(_params)],
    )
    ROWS += bad_value_rows(f"forecast-{_name}", _func, _base, "horizon", INT_BAD)
    ROWS += bad_value_rows(
        f"forecast-{_name}-value",
        _func,
        _base,
        "horizon",
        [(0, ValueError), *NEGATIVE],
    )
    ROWS += good_value_rows(f"forecast-{_name}", _func, _base, "horizon", INT_GOOD)
    ROWS += bad_value_rows(
        f"forecast-{_name}",
        _func,
        _base,
        "start",
        [
            (-NOBS - 1, ValueError),
            (np.int64(-NOBS - 7), ValueError),
            (NOBS + 1, ValueError),
            (10**6, ValueError),
            (2.5, TypeError),
            (5.0, TypeError),
            (True, TypeError),
            ([3], TypeError),
            ("abc", ValueError),
            ("2000-01-05", ValueError),
        ],
    )
    ROWS += good_value_rows(
        f"forecast-{_name}",
        _func,
        _base,
        "start",
        [None, NOBS - 1, np.int64(NOBS - 1), -1, np.int64(-100)],
    )
    ROWS += bad_value_rows(
        f"forecast-{_name}",
        _func,
        _base,
        "align",
        [("zzz", ValueError), (5, TypeError), (None, TypeError), ([], TypeError)],
    )
    ROWS += good_value_rows(
        f"forecast-{_name}", _func, _base, "align", ["origin", "TARGET", "Target"]
    )
    ROWS += bad_value_rows(
        f"forecast-{_name}",
        _func,
        _base,
        "method",
        [("zzz", ValueError), (5, TypeError), (None, TypeError), ([], TypeError)],
    )
    ROWS += good_value_rows(
        f"forecast-{_name}", _func, _base, "method", ["analytic", "ANALYTIC"]
    )
    _sim = {**_base, "method": "simulation"}
    ROWS += bad_value_rows(f"forecast-{_name}", _func, _sim, "simulations", INT_BAD)
    ROWS += bad_value_rows(
        f"forecast-{_name}-value",
        _func,
        _sim,
        "simulations",
        [(0, ValueError), *NEGATIVE],
    )
    ROWS += good_value_rows(
        f"forecast-{_name}", _func, _sim, "simulations", [5, np.int16(5)]
    )
    ROWS += bad_value_rows(
        f"forecast-{_name}",
        _func,
        _sim,
        "rng",
        [("abc", TypeError), (5, TypeError), ([0.1], TypeError)],
    )
    ROWS += good_value_rows(
        f"forecast-{_name}",
        _func,
        {**_sim, "simulations": 5},
        "rng",
        [None, lambda size: np.random.default_rng(0).standard_normal(size)],
    )
    _boot = {**_base, "method": "bootstrap", "simulations": 5}
    ROWS += bad_value_rows(
        f"forecast-{_name}",
        _func,
        _boot,
        "random_state",
        [
            ("abc", TypeError),
            (5, TypeError),
            (np.random.default_rng(0), TypeError),
            (np.random.RandomState, TypeError),
        ],
    )
    ROWS += good_value_rows(
        f"forecast-{_name}",
        _func,
        _boot,
        "random_state",
        [None, np.random.RandomState(0)],
    )
    ROWS += bad_value_rows(f"forecast-{_name}", _func, _base, "reindex", BOOL_BAD)
    ROWS += good_value_rows(f"forecast-{_name}", _func, _base, "reindex", BOOL_GOOD)

# x in forecast
ARX_X = ARX(Y, x=X, lags=1, volatility=GARCH())
ARX_X_PARAMS = np.r_[0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.8]
X_FCAST = RNG.standard_normal((1, 2))
X_FCAST_3D = RNG.standard_normal((2, 1, 2))
X_FCAST_NAN = X_FCAST_3D.copy()
X_FCAST_NAN[0, 0, 0] = np.nan


def _forecast_x(**kw):
    return ARX_X.forecast(**kw)


X_BASE = {"params": ARX_X_PARAMS, "horizon": 2}
ROWS += bad_value_rows(
    "forecast-ARX-x",
    _forecast_x,
    X_BASE,
    "x",
    [
        (np.full((2, 1, 2), "a"), TypeError),
        (np.where(X_FCAST_3D > 0, np.inf, X_FCAST_3D), ValueError),
        (5.0, ValueError),
        (RNG.standard_normal((2, 1, 2, 1)), ValueError),
        ({"x0": np.full((1, 2), "a"), "x1": X_FCAST}, TypeError),
        ({"x0": 5.0, "x1": 5.0}, ValueError),
    ],
)
ROWS += good_value_rows(
    "forecast-ARX-x",
    _forecast_x,
    X_BASE,
    "x",
    [X_FCAST_3D, X_FCAST_NAN, {"x0": X_FCAST, "x1": X_FCAST}],
)
ROWS += [
    Row(
        "forecast-ARX-x-shapes",
        _forecast_x,
        {**X_BASE, "x": {"x0": X_FCAST, "x1": np.ones((1, 3))}},
        ValueError,
        r"^The shapes of the arrays contained in the dictionary differ",
        id="forecast-ARX-x-shapes",
        form=False,
    ),
    Row(
        "forecast-MAMean-x",
        _forecast("MAMean"),
        {"params": FORECAST_PARAMS["MAMean"], "x": X_FCAST},
        TypeError,
        r"^x is not None but the model does not contain any exogenous variables",
        id="forecast-MAMean-x",
        form=False,
    ),
    Row(
        "forecast-ARX-x-missing",
        _forecast_x,
        {"params": ARX_X_PARAMS},
        TypeError,
        r"^x is None but the model contains exogenous variables",
        id="forecast-ARX-x-missing",
        form=False,
    ),
]

# resids
RESID_MODELS = {
    "HARX": (HARX(Y, X, lags=[1, 2]), np.r_[0.1, 0.1, 0.1, 0.2, 0.3], 5),
    "ARX": (ARX(Y, X, lags=1), np.r_[0.1, 0.2, 0.3, 0.4], 4),
    "ARCHInMean": (
        ARCHInMean(Y, lags=1, volatility=GARCH()),
        np.r_[0.1, 0.2, 0.3],
        3,
    ),
    "LS": (LS(Y, X), np.r_[0.1, 0.2, 0.3], 3),
    "ConstantMean": (ConstantMean(Y), np.r_[0.1], 1),
    "ZeroMean": (ZeroMean(Y), np.empty(0), 0),
    "MAMean": (MAMean(Y, lags=2), np.r_[0.1, 0.3, 0.2], 3),
}
for _name, (_model, _params, _k) in RESID_MODELS.items():

    def _resids(_model=_model, **kw):
        return _model.resids(**kw)

    _bad = [
        (np.r_[_params, 1.0], ValueError),
        (np.tile(_params[:, None], (1, 2)) if _k else np.ones((2, 2)), ValueError),
        ("abc", TypeError),
        (None, TypeError),
    ]
    if _k:
        _bad.append((_params[:-1], ValueError))
    _base = {"params": _params}
    ROWS += bad_value_rows(f"resids-{_name}", _resids, _base, "params", _bad)
    # NaN and inf are not rejected, since the likelihood evaluates resids for any
    # parameters that the optimizer tries
    _nan_params = np.full(_k, np.nan)
    ROWS += good_value_rows(
        f"resids-{_name}",
        _resids,
        _base,
        "params",
        [_params, list(_params), _nan_params],
    )
    _y_base = {"params": _params, "y": Y[:50], "regressors": None}
    if _name in ("HARX", "ARX", "LS", "ARCHInMean"):
        # ARCHInMean has one regressor less than the parameters
        _y_base["regressors"] = np.ones((50, _k - int(_name == "ARCHInMean")))
    _bad_y = [
        (NAN_Y[:50], ValueError),
        (np.empty(0), ValueError),
        (np.ones((2, 2)), ValueError),
        ("abc", TypeError),
        (np.array(["a", "b"]), TypeError),
    ]
    ROWS += bad_value_rows(f"resids-{_name}", _resids, _y_base, "y", _bad_y)
    ROWS += good_value_rows(
        f"resids-{_name}", _resids, _y_base, "y", [Y[:50], list(Y[:50])]
    )
    _kr = _k - int(_name == "ARCHInMean")
    _bad_reg = [
        (np.full((50, max(_kr, 1)), np.nan), ValueError),
        (np.full((50, max(_kr, 1)), "a"), TypeError),
        ("abc", TypeError),
        (np.ones((50, max(_kr, 1), 1)), ValueError),
    ]
    if _name in ("HARX", "ARX", "LS", "ARCHInMean"):
        _bad_reg += [
            (np.ones((49, _kr)), ValueError),
            (np.ones((50, _kr + 1)), ValueError),
            (None, TypeError),
        ]
    ROWS += bad_value_rows(f"resids-{_name}", _resids, _y_base, "regressors", _bad_reg)

# append
APPEND_FUNCS = {
    "HARX": lambda **kw: HARX(Y[:100], X[:100], lags=[1, 2]).append(**kw),
    "ARX": lambda **kw: ARX(Y[:100], X[:100], lags=1).append(**kw),
    "ARCHInMean": lambda **kw: ARCHInMean(
        Y[:100], X[:100], lags=1, volatility=GARCH()
    ).append(**kw),
    "LS": lambda **kw: LS(Y[:100], X[:100]).append(**kw),
}
for _name, _func in APPEND_FUNCS.items():
    _base = {"y": Y[100:], "x": X[100:]}
    ROWS += bad_value_rows(
        f"append-{_name}",
        _func,
        _base,
        "x",
        [
            (np.where(X[100:] > 1, np.nan, X[100:]), ValueError),
            (np.where(X[100:] > 1, np.inf, X[100:]), ValueError),
            (np.full((20, 2), "a"), TypeError),
        ],
    )
    ROWS += good_value_rows(f"append-{_name}", _func, _base, "x", [X[100:]])

# arch_model
AM_BASE = {"y": Y}
ORDER_BAD = [*INT_BAD, (-1, ValueError), ([1, 2], TypeError)]
ORDER_OK = [0, 1, 2, np.int64(1), np.uint8(1)]
ROWS += bad_value_rows("arch_model-p", model, AM_BASE, "p", ORDER_BAD)
ROWS += good_value_rows("arch_model-p", model, AM_BASE, "p", [1, 2, np.int64(1)])
ROWS += bad_value_rows("arch_model-o", model, AM_BASE, "o", ORDER_BAD)
ROWS += good_value_rows("arch_model-o", model, AM_BASE, "o", ORDER_OK)
ROWS += bad_value_rows("arch_model-q", model, AM_BASE, "q", ORDER_BAD)
ROWS += good_value_rows("arch_model-q", model, AM_BASE, "q", ORDER_OK)
ROWS += bad_value_rows(
    "arch_model-power",
    model,
    AM_BASE,
    "power",
    [*FLOAT_BAD, (0.0, ValueError), (-1.0, ValueError)],
)
ROWS += good_value_rows(
    "arch_model-power", model, AM_BASE, "power", [1, 2, *FLOAT_GOOD]
)
ROWS += bad_value_rows(
    "arch_model-power-figarch",
    model,
    {**AM_BASE, "vol": "figarch"},
    "power",
    [*FLOAT_BAD, (0.0, ValueError)],
)
for _vol in ("arch", "egarch", "aparch", "figarch", "fiaparch"):
    ROWS += bad_value_rows(
        f"arch_model-p-{_vol}", model, {**AM_BASE, "vol": _vol}, "p", ORDER_BAD
    )
for _vol in ("egarch", "aparch", "fiaparch"):
    ROWS += bad_value_rows(
        f"arch_model-o-{_vol}", model, {**AM_BASE, "vol": _vol}, "o", ORDER_BAD
    )
for _vol in ("egarch", "aparch", "fiaparch", "figarch"):
    ROWS += bad_value_rows(
        f"arch_model-q-{_vol}", model, {**AM_BASE, "vol": _vol}, "q", ORDER_BAD
    )
HARCH_BASE = {**AM_BASE, "vol": "harch"}
ROWS += bad_value_rows(
    "arch_model-p-harch",
    model,
    HARCH_BASE,
    "p",
    [
        (2.5, TypeError),
        (True, TypeError),
        ("2", TypeError),
        (None, TypeError),
        (-1, ValueError),
        ([1.5, 2], TypeError),
        ([0, 1], ValueError),
        ([], ValueError),
        ([[1, 2], [3, 4]], ValueError),
    ],
)
ROWS += good_value_rows(
    "arch_model-p-harch",
    model,
    HARCH_BASE,
    "p",
    [1, np.int64(3), [1, 5], np.array([1, 5]), (1, 5)],
)
# Arguments that do not apply to the model are ignored
IGNORED = [
    Row(
        "arch_model-ignored",
        model,
        {**AM_BASE, **kw},
        None,
        id=f"arch_model-ignored-{i}",
    )
    for i, kw in enumerate(
        [
            {"mean": "zero", "lags": "bad", "x": "bad"},
            {"mean": "constant", "lags": 2.5},
            {"mean": "ls", "lags": [1.5]},
            {"vol": "arch", "o": "bad", "q": "bad", "power": "bad"},
            {"vol": "constant", "p": "bad", "o": "bad", "q": "bad", "power": "bad"},
            {"vol": "harch", "o": "bad", "q": "bad", "power": "bad"},
            {"vol": "egarch", "power": np.nan},
            {"vol": "aparch", "power": "bad"},
            {"vol": "fiaparch", "power": "bad"},
            {"vol": "arch", "p": 2, "q": -1},
            {"vol": "garch", "p": 1, "o": 1, "q": 1, "power": 1},
        ]
    )
]
ROWS += IGNORED
for _mean in ("ar", "arx", "har", "harx"):
    ROWS += bad_value_rows(
        f"arch_model-lags-{_mean}",
        model,
        {**AM_BASE, "mean": _mean, "x": X},
        "lags",
        [(2.5, TypeError), (5.0, TypeError), (True, TypeError), (-1, ValueError)],
    )
for _mean in ("zero", "constant", "ls", "ar", "arx", "har", "harx"):
    _base = {**AM_BASE, "mean": _mean, "x": X}
    ROWS += bad_value_rows(
        f"arch_model-hold_back-{_mean}",
        model,
        _base,
        "hold_back",
        INT_NOT_NONE + NEGATIVE,
    )
    ROWS += bad_value_rows(
        f"arch_model-rescale-{_mean}", model, _base, "rescale", NOT_NONE
    )
    ROWS += good_value_rows(
        f"arch_model-rescale-{_mean}", model, _base, "rescale", [*BOOL_GOOD, None]
    )
    ROWS += good_value_rows(
        f"arch_model-hold_back-{_mean}",
        model,
        _base,
        "hold_back",
        [None, 0, np.int64(3)],
    )


@pytest.mark.parametrize("row", to_params(ROWS))
def test_validation(row):
    run_row(row)


def test_hold_back_is_stored_as_int():
    mod = ConstantMean(Y, hold_back=np.int64(3))
    assert type(mod.hold_back) is int
    assert mod.hold_back == 3
    assert mod._hold_back == 3


def test_flags_are_stored_as_bool():
    mod = HARX(Y, lags=2, constant=np.False_, use_rotated=np.True_, rescale=np.False_)
    assert type(mod.constant) is bool
    assert type(mod.use_rotated) is bool
    assert type(mod.rescale) is bool
    assert mod.constant is False
    assert mod.use_rotated is True
    assert type(MAMean(Y, constant=np.False_).constant) is bool


def test_negative_hold_back_is_rejected():
    # hold_back=-5 used to be ignored, with a warning only for models with lags
    with pytest.raises(
        ValueError, match=r"^hold_back must be an integer >= 0 or None; got -5"
    ):
        ConstantMean(Y, hold_back=-5)
    with pytest.raises(
        ValueError, match=r"^hold_back must be an integer >= 0 or None; got -5"
    ):
        arch_model(Y, mean="ar", lags=2, hold_back=-5)


def test_lags_are_not_truncated():
    # lags=2.5 used to be truncated to 2
    for cls in (ARX, HARX, ARCHInMean):
        with pytest.raises(TypeError, match=r"^lags must be an integer >= 0; got 2.5"):
            cls(Y, lags=2.5)
    with pytest.raises(TypeError, match=r"^lags must be an integer >= 1; got 2.5"):
        MAMean(Y, lags=2.5)


def test_out_of_range_sample_is_not_recorded():
    # first_obs=-1000 used to be recorded as the start of the fit
    mod = ConstantMean(Y)
    with pytest.raises(ValueError, match=r"^first_obs must be an integer position in"):
        mod.fit(disp="off", first_obs=-1000)
    with pytest.raises(ValueError, match=r"^last_obs must be an integer position in"):
        mod.fit(disp="off", last_obs=10**6)
    assert mod._fit_indices == [0, NOBS]


def test_simulate_sizes_are_checked():
    # nobs=-1 and burn=-1 used to return empty or short frames
    mod = ZeroMean(volatility=GARCH())
    params = np.r_[0.05, 0.1, 0.8]
    with pytest.raises(ValueError, match=r"^nobs must be an integer >= 1; got -1"):
        mod.simulate(params, -1, burn=-1)
    with pytest.raises(ValueError, match=r"^burn must be an integer >= 0; got -1"):
        mod.simulate(params, 10, burn=-1)
    assert mod.simulate(params, 10, burn=0).shape == (10, 3)


def test_message_shows_the_value():
    with pytest.raises(TypeError) as info:
        ARX(Y, lags=2.5)
    assert str(info.value) == "lags must be an integer >= 0; got 2.5 (float)"
    with pytest.raises(TypeError) as info:
        HARX(Y, constant="no")
    assert str(info.value) == "constant must be True, False, 0 or 1; got 'no' (str)"
    with pytest.raises(ValueError, match=r"^p must be") as info:
        arch_model(Y, p=-1)
    assert str(info.value) == "p must be an integer >= 0; got -1 (int)"


def test_forecast_values_unchanged_by_checks():
    mod = ARX(Y, lags=2, volatility=GARCH())
    params = np.r_[0.1, 0.1, 0.1, 0.05, 0.1, 0.8]
    base = mod.forecast(params, horizon=3, reindex=False)
    other = mod.forecast(
        pd.Series(params),
        horizon=np.int64(3),
        align="ORIGIN",
        method="Analytic",
        reindex=np.False_,
    )
    pd.testing.assert_frame_equal(base.mean, other.mean)
    pd.testing.assert_frame_equal(base.variance, other.variance)


def test_resids_nan_params_follow_the_input():
    # resids is evaluated for any parameters by the likelihood, and so NaN is not
    # rejected and propagates
    mod = ConstantMean(Y)
    assert np.all(np.isnan(mod.resids(np.array([np.nan]))))
    assert np.all(np.isnan(ARX(Y, lags=1).resids(np.array([np.nan, 0.1]))))
