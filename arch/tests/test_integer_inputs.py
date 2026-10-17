"""
Integer, unsigned integer, Boolean and list inputs to the public API.

No estimator may work on integer data: products and sums of integers overflow
or wrap, and an integer result is silently truncated. Every case below runs a
public entry point on data that are stored as one of several non-float64 types
and requires the result to be the one computed from the same data stored as
float64, without any warning and without modifying the data of the caller.
"""

from __future__ import annotations

from collections.abc import Callable
import copy
from typing import Any, NamedTuple
import warnings

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest

from arch import arch_model
from arch.bootstrap import (
    MCS,
    SPA,
    IIDBootstrap,
    StepM,
    optimal_block_length,
)
from arch.covariance.kernel import Bartlett, Parzen, QuadraticSpectral
from arch.covariance.var import PreWhitenedRecolored
from arch.unitroot import (
    ADF,
    DFGLS,
    KPSS,
    PhillipsPerron,
    VarianceRatio,
    ZivotAndrews,
    engle_granger,
)
from arch.unitroot.cointegration import (
    CanonicalCointegratingReg,
    DynamicOLS,
    FullyModifiedOLS,
    phillips_ouliaris,
)
from arch.unitroot.unitroot import auto_bandwidth
from arch.univariate import (
    APARCH,
    ARX,
    EGARCH,
    FIGARCH,
    GARCH,
    HARCH,
    LS,
    ARCHInMean,
    ConstantMean,
    ConstantVariance,
    EWMAVariance,
    FixedVariance,
    GeneralizedError,
    MAMean,
    Normal,
    RiskMetrics2006,
    SkewStudent,
    StudentsT,
    ZeroMean,
)
from arch.utility.cov import cov_nw
from arch.utility.timeseries import add_trend

NOBS = 240
RTOL = 1e-10
ATOL = 1e-12

# How the data of a case are stored. The type of the elements is the one that is
# tested; the container is a list, an ndarray, an unnamed Series with a DatetimeIndex
# or a DataFrame with column names and a non-default index.
KINDS = ("int64", "int32", "uint8", "bool", "list", "series", "frame")


def _pandas_container(values: np.ndarray, kind: str) -> Any:
    """Store the values as a Series or DataFrame, or as dicts of DataFrames."""
    nobs = values.shape[0]
    if kind == "series":
        prefix = "c"
        index = pd.date_range("2000-01-03", periods=nobs, freq="B")
    else:
        prefix = "v"
        index = pd.Index(np.arange(nobs) + 10, name="obs")
    if values.ndim == 3:
        return {f"{prefix}{i}": pd.DataFrame(values[i]) for i in range(len(values))}
    if values.ndim == 1:
        if kind == "series":
            return pd.Series(values, index=index)
        return pd.DataFrame({"data": values}, index=index)
    cols = [f"{prefix}{i}" for i in range(values.shape[1])]
    return pd.DataFrame(values, index=index, columns=cols)


def _container(arr: np.ndarray, kind: str, dtype: type, floating: bool) -> Any:
    """Store a float64 array holding integers as the container of the kind."""
    if kind == "list":
        return arr.tolist() if floating else arr.astype(np.int64).tolist()
    values = arr.astype(np.float64 if floating else dtype)
    if kind in ("int64", "int32", "uint8", "bool"):
        return values
    return _pandas_container(values, kind)


class Case(NamedTuple):
    name: str
    arrays: Callable[[], dict[str, np.ndarray]]
    run: Callable[[dict[str, Any]], dict[str, Any]]
    kinds: tuple[str, ...]
    boolean: tuple[str, ...]


CASES: list[Case] = []


def case(
    arrays: Callable[[], dict[str, np.ndarray]],
    boolean: tuple[str, ...] = (),
    kinds: tuple[str, ...] | None = None,
) -> Callable[[Callable[[dict[str, Any]], dict[str, Any]]], Callable[..., Any]]:
    """
    Register a case.

    arrays are float64 arrays that hold non-negative integers of at most 255, so
    that they can be stored as any of the types. boolean lists the arrays that hold
    only 0 and 1. The Boolean kind is run only for cases that have a boolean array,
    which is stored as Boolean while the other arrays are int64.
    """

    def decorator(
        func: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> Callable[..., Any]:
        if kinds is None:
            ks = tuple(k for k in KINDS if k != "bool" or boolean)
        else:
            ks = kinds
        CASES.append(Case(func.__name__, arrays, func, ks, boolean))
        return func

    return decorator


def _dtype_for(kind: str, name: str, boolean: tuple[str, ...]) -> type:
    if kind == "bool":
        return np.bool_ if name in boolean else np.int64
    return {"int32": np.int32, "uint8": np.uint8}.get(kind, np.int64)


def build(case_: Case, kind: str, floating: bool) -> dict[str, Any]:
    out = {}
    for key, arr in case_.arrays().items():
        dtype = _dtype_for(kind, key, case_.boolean)
        out[key] = _container(arr, kind, dtype, floating)
    return out


def assert_same(actual: Any, desired: Any, label: str) -> None:
    """Compare nested results with a tolerance."""
    if isinstance(desired, dict):
        actual_keys, desired_keys = set(actual), set(desired)
        assert actual_keys == desired_keys, label
        for key in desired:
            assert_same(actual[key], desired[key], f"{label}.{key}")
    elif isinstance(desired, (str, bool, int, type(None))):
        assert actual == desired, label
    elif np.asarray(desired).dtype.kind in "OUS":
        assert_array_equal(np.asarray(actual), np.asarray(desired), err_msg=label)
    else:
        assert_allclose(
            np.asarray(actual, dtype=float),
            np.asarray(desired, dtype=float),
            rtol=RTOL,
            atol=ATOL,
            err_msg=label,
        )


def assert_unchanged(before: Any, after: Any) -> None:
    """The data of the caller must not be modified, including names and dtypes."""
    if isinstance(before, dict):
        before_keys, after_keys = set(before), set(after)
        assert before_keys == after_keys
        for key, value in before.items():
            assert_unchanged(value, after[key])
    elif isinstance(before, pd.DataFrame):
        assert_frame_equal(after, before)
    elif isinstance(before, pd.Series):
        assert_series_equal(after, before)
    elif isinstance(before, np.ndarray):
        assert after.dtype == before.dtype
        assert_array_equal(after, before)
    else:
        assert after == before
        after_type, before_type = type(after), type(before)
        assert after_type is before_type


def _walk(seed: int, nvar: int = 1) -> np.ndarray:
    """Random walks that start at 100 with steps of -2 to 2, so they stay positive."""
    rng = np.random.default_rng(seed)
    steps = rng.integers(-2, 3, size=(NOBS, nvar))
    out = 100.0 + np.cumsum(steps, axis=0)
    lower, upper = out.min(), out.max()
    assert 0 <= lower
    assert upper <= 255
    return out[:, 0] if nvar == 1 else out


def _garch_counts(seed: int = 0) -> np.ndarray:
    """GARCH data that are rounded to integers and shifted to be positive."""
    rng = np.random.default_rng(seed)
    nobs = NOBS + 50
    e = rng.standard_normal(nobs)
    sigma2 = np.empty(nobs)
    y = np.empty(nobs)
    sigma2[0] = 1.0
    for t in range(nobs):
        if t:
            sigma2[t] = 0.2 + 0.2 * y[t - 1] ** 2 + 0.6 * sigma2[t - 1]
        y[t] = np.sqrt(sigma2[t]) * e[t]
    return np.clip(np.round(50 + 4 * y[50:]), 0, 255)


def _regressors(seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 10, size=(NOBS, 2)).astype(float)


def _dummies(seed: int = 2) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, size=(NOBS, 2)).astype(float)


def _forecast_x() -> np.ndarray:
    return np.random.default_rng(3).integers(0, 10, size=(2, 1, 3)).astype(float)


def _result(res: Any) -> dict[str, Any]:
    return {
        "params": res.params.to_numpy(),
        "loglikelihood": res.loglikelihood,
        "volatility": res.conditional_volatility,
        "resid": res.resid,
    }


# ---------------------------------------------------------------------------
# arch.univariate: the data and the exogenous regressors of the models
# ---------------------------------------------------------------------------
def _data_x() -> dict[str, np.ndarray]:
    return {"y": _garch_counts(), "x": _regressors()}


@case(lambda: {"y": _garch_counts()})
def garch_fit(d: dict[str, Any]) -> dict[str, Any]:
    return _result(arch_model(d["y"]).fit(disp="off"))


@case(_data_x)
def arx_egarch_t_fit(d: dict[str, Any]) -> dict[str, Any]:
    mod = arch_model(d["y"], d["x"], mean="ARX", lags=1, vol="EGARCH", o=1, dist="t")
    return _result(mod.fit(disp="off"))


@case(_data_x)
def harx_arch_fit(d: dict[str, Any]) -> dict[str, Any]:
    mod = arch_model(d["y"], d["x"], mean="HARX", lags=[1, 5], vol="ARCH", p=2)
    return _result(mod.fit(disp="off"))


@case(_data_x, boolean=())
def ls_aparch_ged_fit(d: dict[str, Any]) -> dict[str, Any]:
    mod = arch_model(d["y"], d["x"], mean="LS", vol="APARCH", dist="ged")
    return _result(mod.fit(disp="off"))


@case(lambda: {"y": _garch_counts(), "x": _dummies()}, boolean=("x",))
def dummy_regressors_fit(d: dict[str, Any]) -> dict[str, Any]:
    mod = arch_model(d["y"], d["x"], mean="ARX", lags=1, vol="GARCH", dist="skewt")
    return _result(mod.fit(disp="off"))


@case(lambda: {"y": _garch_counts()})
def zero_mean_fit(d: dict[str, Any]) -> dict[str, Any]:
    mod = ZeroMean(d["y"], volatility=HARCH([1, 5]))
    return _result(mod.fit(disp="off"))


@case(lambda: {"y": _garch_counts(), "x": _regressors(), "xf": _forecast_x()})
def arx_forecast_x(d: dict[str, Any]) -> dict[str, Any]:
    mod = arch_model(d["y"], d["x"], mean="ARX", lags=2, vol="GARCH")
    res = mod.fit(disp="off")
    out = {}
    for method in ("analytic", "simulation", "bootstrap"):
        fcast = res.forecast(
            horizon=3,
            x=d["xf"],
            method=method,
            simulations=20,
            rng=RandomState(0).standard_normal,
            random_state=RandomState(1),
        )
        out[method] = {"mean": fcast.mean, "variance": fcast.variance}
    return out


@case(lambda: {"y": _garch_counts(), "x": _regressors()})
def forecast_params(d: dict[str, Any]) -> dict[str, Any]:
    mod = arch_model(d["y"], mean="AR", lags=1, vol="GARCH")
    res = mod.fit(disp="off")
    params = np.round(res.params.to_numpy())
    fcast = mod.forecast(params, horizon=2)
    return {"mean": fcast.mean, "variance": fcast.variance}


# ---------------------------------------------------------------------------
# arch.univariate: fix, simulate, fit arguments and the residuals
# ---------------------------------------------------------------------------
def _fix_arrays() -> dict[str, np.ndarray]:
    # constant, lag, 2 regressors, omega, alpha, beta
    params = np.array([10.0, 0.0, 1.0, 2.0, 2.0, 0.0, 0.0, 8.0])
    return {"y": _garch_counts(), "x": _regressors(), "params": params}


@case(_fix_arrays)
def fix_integer_params(d: dict[str, Any]) -> dict[str, Any]:
    mod = arch_model(d["y"], d["x"], mean="ARX", lags=1, vol="GARCH", dist="t")
    res = mod.fix(d["params"])
    fcast = res.forecast(
        horizon=2,
        x=np.ones((2, 1, 2)),
        method="simulation",
        simulations=10,
        rng=RandomState(0).standard_normal,
    )
    return {
        "params": res.params,
        "loglikelihood": res.loglikelihood,
        "volatility": res.conditional_volatility,
        "mean": fcast.mean,
        "variance": fcast.variance,
    }


@case(
    lambda: {
        "y": _garch_counts(),
        "x": _regressors(),
        "xr": np.column_stack([np.ones(NOBS), _regressors()]),
        "params": np.array([10.0, 1.0, 2.0]),
    }
)
def resids_arguments(d: dict[str, Any]) -> dict[str, Any]:
    mod = LS(d["y"], d["x"])
    return {
        "default": mod.resids(d["params"]),
        "alternative": mod.resids(d["params"], y=d["y"], regressors=d["xr"]),
    }


def _simulate_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "params": np.array([10.0, 0.0, 1.0, 2.0, 2.0, 0.0, 0.0, 8.0]),
        "x": rng.integers(0, 10, size=(60, 2)).astype(float),
        "initial_value": np.array([5.0]),
        "initial_value_vol": np.array([4.0]),
    }


@case(_simulate_arrays)
def simulate_arguments(d: dict[str, Any]) -> dict[str, Any]:
    mod = ARX(
        None,
        lags=1,
        volatility=GARCH(),
        distribution=StudentsT(seed=RandomState(11)),
    )
    sim = mod.simulate(
        d["params"],
        50,
        burn=10,
        initial_value=d["initial_value"],
        x=d["x"],
        initial_value_vol=d["initial_value_vol"],
    )
    return {"sim": sim}


@case(lambda: {"y": _garch_counts(), "sv": np.array([50.0, 2.0, 0.0, 0.0])})
def fit_starting_values(d: dict[str, Any]) -> dict[str, Any]:
    mod = ConstantMean(d["y"])
    return _result(mod.fit(starting_values=d["sv"], disp="off"))


@case(lambda: {"y": _garch_counts(), "backcast": np.array([16.0])})
def fit_backcast(d: dict[str, Any]) -> dict[str, Any]:
    mod = ConstantMean(d["y"])
    return _result(mod.fit(backcast=d["backcast"], disp="off"))


@case(_data_x)
def append_observations(d: dict[str, Any]) -> dict[str, Any]:
    y, x = d["y"], d["x"]
    mod = arch_model(y[:200], x[:200], mean="ARX", lags=1)
    mod.append(y[200:], x[200:])
    return _result(mod.fit(disp="off"))


# ---------------------------------------------------------------------------
# arch.univariate: volatility processes and distributions used directly
# ---------------------------------------------------------------------------
def _processes() -> list[Any]:
    return [
        ConstantVariance(),
        GARCH(1, 1, 1),
        GARCH(2, 0, 1),
        EGARCH(1, 1, 1),
        APARCH(1, 1, 1),
        FIGARCH(1, 1, 1),
        HARCH([1, 5]),
        EWMAVariance(0.94),
        RiskMetrics2006(),
    ]


@case(lambda: {"resids": _garch_counts()})
def volatility_process_methods(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    resids = d["resids"]
    for i, vol in enumerate(_processes()):
        sigma2 = np.empty(NOBS)
        start = vol.starting_values(resids)
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        vol.compute_variance(start, resids, sigma2, backcast, var_bounds)
        out[f"{i}"] = {
            "start": start,
            "backcast": backcast,
            "bounds": vol.bounds(resids),
            "var_bounds": var_bounds,
            "sigma2": sigma2,
        }
    return out


@case(lambda: {"resids": _garch_counts(), "params": np.array([2.0, 0.0, 0.0])})
def volatility_integer_parameters(d: dict[str, Any]) -> dict[str, Any]:
    resids = d["resids"]
    out = {}
    for name, vol in {"garch": GARCH(1, 0, 1), "arch": GARCH(1, 0, 0)}.items():
        params = d["params"][: vol.num_params]
        sigma2 = np.empty(NOBS)
        var_bounds = vol.variance_bounds(resids)
        backcast = vol.backcast(resids)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        out[name] = sigma2
    const = ConstantVariance()
    sigma2 = np.empty(NOBS)
    const.compute_variance(d["params"][:1], resids, sigma2, 1.0, np.ones((NOBS, 2)))
    out["constant"] = sigma2
    return out


def _distribution_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(9)
    return {
        "resids": rng.integers(0, 12, size=NOBS).astype(float),
        "sigma2": rng.integers(1, 10, size=NOBS).astype(float),
        "std_resid": rng.integers(0, 4, size=NOBS).astype(float),
        "p_normal": np.empty(0),
        "p_t": np.array([8.0]),
        "p_skew": np.array([8.0, 0.0]),
        "p_ged": np.array([2.0]),
    }


@case(_distribution_arrays)
def distribution_methods(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    dists = {
        "normal": (Normal(), "p_normal"),
        "t": (StudentsT(), "p_t"),
        "skew": (SkewStudent(), "p_skew"),
        "ged": (GeneralizedError(), "p_ged"),
    }
    for name, (dist, key) in dists.items():
        params = d[key]
        out[name] = {
            "ll": dist.loglikelihood(params, d["resids"], d["sigma2"]),
            "individual": dist.loglikelihood(
                params, d["resids"], d["sigma2"], individual=True
            ),
            "start": dist.starting_values(d["std_resid"]),
            "bounds": dist.bounds(d["resids"]),
            "cdf": dist.cdf(d["resids"], params),
            "moment": dist.moment(2, params),
            "partial": dist.partial_moment(2, 1, params),
        }
    return out


@case(
    lambda: {
        "y": _garch_counts(),
        "variance": np.random.default_rng(5).integers(1, 9, size=NOBS).astype(float),
    }
)
def fixed_variance(d: dict[str, Any]) -> dict[str, Any]:
    mod = ConstantMean(d["y"], volatility=FixedVariance(d["variance"]))
    return _result(mod.fit(disp="off"))


@case(lambda: {"y": _garch_counts(), "params": np.array([10.0, 1.0])})
def resids_mean_models(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    y, params = d["y"], d["params"]
    out["constant"] = ConstantMean(y).resids(params[:1], y=y)
    out["zero"] = ZeroMean(y).resids(params[:0], y=y)
    out["ma"] = MAMean(y, lags=1).resids(params, y=y)
    return out


@case(lambda: {"y": _garch_counts(), "x": _regressors()})
def mean_model_variants(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    out["ma"] = _result(MAMean(d["y"], lags=1).fit(disp="off"))
    out["in_mean"] = _result(
        ARCHInMean(d["y"], lags=1, volatility=GARCH(), form="vol").fit(disp="off")
    )
    return out


# ---------------------------------------------------------------------------
# arch.unitroot: unit root tests
# ---------------------------------------------------------------------------
def _test_result(test: Any) -> dict[str, Any]:
    return {
        "stat": test.stat,
        "pvalue": test.pvalue,
        "lags": test.lags,
        "critical_values": test.critical_values,
        "nobs": test.nobs,
    }


@case(lambda: {"y": _walk(10)})
def adf(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for trend in ("n", "c", "ct", "ctt"):
        out[trend] = _test_result(ADF(d["y"], trend=trend, lags=2))
    out["autolag"] = _test_result(ADF(d["y"], max_lags=6, method="bic"))
    out["low_memory"] = _test_result(ADF(d["y"], max_lags=6, low_memory=True))
    return out


@case(lambda: {"y": _walk(11)})
def dfgls(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for trend in ("c", "ct"):
        out[trend] = _test_result(DFGLS(d["y"], trend=trend, lags=2))
    out["autolag"] = _test_result(DFGLS(d["y"], max_lags=6, method="aic"))
    return out


@case(lambda: {"y": _walk(12)})
def phillips_perron(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for trend in ("n", "c", "ct"):
        for test_type in ("tau", "rho"):
            pp = PhillipsPerron(d["y"], trend=trend, lags=4, test_type=test_type)
            out[f"{trend}-{test_type}"] = _test_result(pp)
    return out


@case(lambda: {"y": _walk(13)})
def kpss(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for trend in ("c", "ct"):
        out[trend] = _test_result(KPSS(d["y"], lags=4, trend=trend))
    out["legacy"] = _test_result(KPSS(d["y"], lags=-1))
    return out


@case(lambda: {"y": _walk(14)})
def zivot_andrews(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for trend in ("c", "t", "ct"):
        out[trend] = _test_result(ZivotAndrews(d["y"], trend=trend, lags=2))
    return out


@case(lambda: {"y": _walk(15)})
def variance_ratio(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for debiased, robust in ((True, True), (False, False)):
        vr = VarianceRatio(d["y"], lags=4, debiased=debiased, robust=robust)
        out[f"{debiased}-{robust}"] = _test_result(vr)
    return out


@case(lambda: {"y": _walk(16)})
def auto_bandwidth_function(d: dict[str, Any]) -> dict[str, Any]:
    return {k: auto_bandwidth(d["y"], k) for k in ("ba", "pa", "qs")}


# ---------------------------------------------------------------------------
# arch.unitroot: cointegration
# ---------------------------------------------------------------------------
def _cointegrated() -> dict[str, np.ndarray]:
    x = _walk(20, 2)
    rng = np.random.default_rng(21)
    y = np.round(x[:, 0] + x[:, 1] / 2 + rng.integers(-1, 2, size=NOBS) + 10)
    lower, upper = y.min(), y.max()
    assert 0 <= lower
    assert upper <= 255
    return {"y": y, "x": x}


def _coint_result(res: Any) -> dict[str, Any]:
    return {
        "params": res.params,
        "bse": res.std_errors,
        "cov": res.cov,
        "resid": res.resid,
        "rsquared": res.rsquared,
    }


@case(_cointegrated)
def engle_granger_test(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for trend in ("n", "c", "ct"):
        res = engle_granger(d["y"], d["x"], trend=trend, lags=2)
        out[trend] = {
            "stat": res.stat,
            "pvalue": res.pvalue,
            "critical_values": res.critical_values,
            "cointegrating_vector": res.cointegrating_vector,
        }
    res = engle_granger(d["y"], d["x"], max_lags=4, method="bic")
    out["autolag"] = {"stat": res.stat, "pvalue": res.pvalue}
    return out


@case(_cointegrated)
def phillips_ouliaris_test(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for test_type in ("Za", "Zt", "Pu", "Pz"):
        res = phillips_ouliaris(d["y"], d["x"], test_type=test_type, bandwidth=4)
        out[test_type] = {
            "stat": res.stat,
            "pvalue": res.pvalue,
            "critical_values": res.critical_values,
        }
    return out


@case(_cointegrated)
def dynamic_ols(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    model = DynamicOLS(d["y"], d["x"], trend="ct", lags=2, leads=2)
    out["unadjusted"] = _coint_result(model.fit())
    out["robust"] = _coint_result(model.fit(cov_type="robust", bandwidth=4))
    out["auto"] = _coint_result(DynamicOLS(d["y"], d["x"], max_lag=3, max_lead=3).fit())
    return out


@case(_cointegrated)
def fmols_ccr(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for cls in (FullyModifiedOLS, CanonicalCointegratingReg):
        for trend in ("n", "c", "ct"):
            res = cls(d["y"], d["x"], trend=trend).fit(bandwidth=4)
            out[f"{cls.__name__}-{trend}"] = _coint_result(res)
    return out


# ---------------------------------------------------------------------------
# arch.covariance and arch.utility.cov
# ---------------------------------------------------------------------------
def _covariance_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(30)
    data = rng.integers(0, 2, size=(NOBS, 3)) * rng.integers(1, 200, size=(NOBS, 3))
    return {
        "x": data.astype(float),
        "dummies": rng.integers(0, 2, size=(NOBS, 2)).astype(float),
        "weights": np.array([1.0, 2.0, 1.0]),
    }


def _estimate(cov: Any) -> dict[str, Any]:
    return {
        "long_run": cov.long_run,
        "short_run": cov.short_run,
        "one_sided": cov.one_sided,
        "one_sided_strict": cov.one_sided_strict,
    }


@case(_covariance_arrays, boolean=("dummies",))
def kernel_estimators(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for kernel in (Bartlett, Parzen, QuadraticSpectral):
        for center in (True, False):
            est = kernel(d["x"], bandwidth=3, center=center, weights=d["weights"])
            out[f"{kernel.__name__}-{center}"] = _estimate(est.cov)
        out[f"{kernel.__name__}-auto"] = {
            "bandwidth": kernel(d["x"]).bandwidth,
            "dummies": _estimate(kernel(d["dummies"], bandwidth=2, center=False).cov),
        }
    return out


@case(_covariance_arrays, boolean=("dummies",))
def pre_whitened_recolored(d: dict[str, Any]) -> dict[str, Any]:
    est = PreWhitenedRecolored(d["x"], lags=2, bandwidth=3)
    est_auto = PreWhitenedRecolored(d["dummies"], method="bic", max_lag=3)
    return {"fixed": _estimate(est.cov), "auto": _estimate(est_auto.cov)}


@case(_covariance_arrays, boolean=("dummies",))
def newey_west_function(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for lags in (0, 3):
        for demean in (True, False):
            out[f"{lags}-{demean}"] = {
                "2d": cov_nw(d["x"], lags=lags, demean=demean),
                "axis": cov_nw(np.asarray(d["x"]).T, lags=lags, axis=1),
                "1d": cov_nw(np.asarray(d["x"])[:, 0], lags=lags, ddof=1),
                "dummies": cov_nw(d["dummies"], lags=lags, demean=demean),
            }
    return out


def _large_values() -> dict[str, np.ndarray]:
    """Values near 2**32 so that squares and products overflow int64."""
    rng = np.random.default_rng(31)
    values = 2.0**32 + rng.integers(0, 1000, size=(NOBS, 2))
    return {"x": values, "y": values[:, 0]}


@case(_large_values, kinds=("int64", "list", "series", "frame"))
def large_values_covariance(d: dict[str, Any]) -> dict[str, Any]:
    out = {"newey_west": cov_nw(d["x"], lags=2, demean=False)}
    out["newey_west_1d"] = cov_nw(d["y"], lags=2, demean=False)
    for kernel in (Bartlett, Parzen):
        est = kernel(d["x"], bandwidth=3, center=False)
        out[kernel.__name__] = _estimate(est.cov)
    return out


@case(_large_values, kinds=("int64", "list", "series", "frame"))
def large_values_unit_root(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "adf": _test_result(ADF(d["y"], trend="n", lags=1)),
        "pp": _test_result(PhillipsPerron(d["y"], trend="n", lags=3)),
        "vr": _test_result(VarianceRatio(d["y"], lags=3)),
        "kpss": _test_result(KPSS(d["y"], lags=3)),
    }


# ---------------------------------------------------------------------------
# arch.bootstrap
# ---------------------------------------------------------------------------
def _losses() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(40)
    losses = rng.integers(0, 40, size=(120, 4)) + np.array([5, 0, 2, 4])
    return {"losses": losses.astype(float), "benchmark": losses[:, :1].astype(float)}


@case(_losses)
def model_confidence_set(d: dict[str, Any]) -> dict[str, Any]:
    mcs = MCS(d["losses"], size=0.2, reps=50, block_size=4, seed=0)
    mcs.compute()
    return {"pvalues": mcs.pvalues, "included": mcs.included, "excluded": mcs.excluded}


@case(_losses)
def superior_predictive_ability(d: dict[str, Any]) -> dict[str, Any]:
    spa = SPA(d["benchmark"], d["losses"], reps=50, block_size=4, seed=0)
    spa.compute()
    return {"pvalues": spa.pvalues, "critical": spa.critical_values()}


@case(_losses)
def stepwise_multiple_testing(d: dict[str, Any]) -> dict[str, Any]:
    stepm = StepM(d["benchmark"], d["losses"], reps=50, block_size=4, seed=0)
    stepm.compute()
    return {"superior": stepm.superior_models}


@case(_losses)
def optimal_block_length_function(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "2d": optimal_block_length(d["losses"]),
        "1d": optimal_block_length(d["benchmark"]),
    }


# ---------------------------------------------------------------------------
# arch.utility
# ---------------------------------------------------------------------------
@case(lambda: {"x": _regressors(), "dummies": _dummies()}, boolean=("dummies",))
def add_trend_function(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name in ("x", "dummies"):
        for trend in ("n", "c", "ct", "ctt", "t"):
            out[f"{name}-{trend}"] = add_trend(d[name], trend=trend)
        out[f"{name}-prepend"] = add_trend(d[name], "ct", prepend=True)
    return out


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------
def _params() -> list[Any]:
    return [pytest.param(c, k, id=f"{c.name}-{k}") for c in CASES for k in c.kinds]


@pytest.mark.parametrize(("case_", "kind"), _params())
def test_matches_float64(case_: Case, kind: str) -> None:
    reference_input = build(case_, kind, floating=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reference = case_.run(reference_input)
    data = build(case_, kind, floating=False)
    before = copy.deepcopy(data)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = case_.run(data)
    assert_same(result, reference, case_.name)
    assert_unchanged(before, data)


# ---------------------------------------------------------------------------
# Data that are not numbers, and data that are passed through unchanged
# ---------------------------------------------------------------------------
STRINGS_1D = ["a", "b", "c"] * 40
STRINGS_2D = [["a", "b"], ["c", "d"]] * 30


@pytest.mark.parametrize(
    ("func", "name"),
    [
        pytest.param(arch_model, "y", id="arch_model-y"),
        pytest.param(ConstantMean, "y", id="ConstantMean-y"),
        pytest.param(ADF, "y", id="ADF-y"),
        pytest.param(lambda v: KPSS(v, lags=2), "y", id="KPSS-y"),
        pytest.param(auto_bandwidth, "y", id="auto_bandwidth-y"),
        pytest.param(lambda v: engle_granger(v, np.ones((90, 1))), "y", id="eg-y"),
        pytest.param(lambda v: DynamicOLS(v, np.ones((90, 1))), "y", id="dols-y"),
    ],
)
def test_non_numeric_1d_raises_type_error(
    func: Callable[[Any], Any], name: str
) -> None:
    with pytest.raises(TypeError, match=rf"^{name} must .*float64"):
        func(STRINGS_1D)


@pytest.mark.parametrize(
    ("func", "name"),
    [
        pytest.param(
            lambda v: arch_model(np.ones(60), v, mean="LS"), "x", id="arch_model-x"
        ),
        pytest.param(Bartlett, "x", id="Bartlett-x"),
        pytest.param(PreWhitenedRecolored, "x", id="PWRC-x"),
        pytest.param(lambda v: engle_granger(np.ones(60), v), "x", id="eg-x"),
        pytest.param(lambda v: SPA(np.ones((60, 1)), v), "models", id="SPA-models"),
        pytest.param(lambda v: MCS(v, size=0.1), "losses", id="MCS-losses"),
        pytest.param(optimal_block_length, "x", id="block_length-x"),
        pytest.param(cov_nw, "y", id="cov_nw-y"),
    ],
)
def test_non_numeric_2d_raises_type_error(
    func: Callable[[Any], Any], name: str
) -> None:
    with pytest.raises(TypeError, match=rf"^{name} must .*float64"):
        func(STRINGS_2D)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.uint8, np.bool_])
def test_bootstrap_data_are_not_converted(dtype: type) -> None:
    # The data of a bootstrap are passed to the function of the user as given
    data = (np.arange(40) % 2).astype(dtype)
    seen = []

    def func(x: np.ndarray) -> float:
        seen.append(x.dtype)
        return float(x.sum())

    bs = IIDBootstrap(data, seed=0)
    bs.apply(func, 3)
    for resampled, _ in bs.bootstrap(2):
        assert resampled[0][0].dtype == dtype
    assert seen
    assert set(seen) == {np.dtype(dtype)}
