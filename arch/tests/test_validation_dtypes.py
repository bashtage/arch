"""
Table-driven tests of the data types that the public API accepts.

Every entry point that takes numeric data follows one rule (``as_float64`` in
``arch/utility/array.py`` and ``check_array`` in ``arch/utility/_validation.py``):

* bool, integer, unsigned integer, float32, float64 and longdouble data are converted
  to float64, and the calculation is done in float64;
* float16 data are rejected with a ``TypeError``;
* complex data are rejected with a ``TypeError``, except by the entry points that are
  Hermitian: the kernel covariance estimators, ``cov_nw`` and ``add_trend``, which
  convert complex data to complex128.

A row is one entry point with one argument replaced by data of one data type.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from arch import arch_model
from arch.bootstrap import MCS, SPA, StepM, optimal_block_length
from arch.covariance.kernel import (
    Bartlett,
    Parzen,
    QuadraticSpectral,
    ZeroLag,
)
from arch.covariance.var import PreWhitenedRecolored
from arch.tests.validation_table import Row, ids_for, run_row, to_params
from arch.unitroot import (
    ADF,
    DFGLS,
    KPSS,
    PhillipsPerron,
    VarianceRatio,
    ZivotAndrews,
    auto_bandwidth,
    engle_granger,
)
from arch.unitroot.cointegration import (
    CanonicalCointegratingReg,
    DynamicOLS,
    FullyModifiedOLS,
    phillips_ouliaris,
)
from arch.univariate import (
    ARX,
    GARCH,
    HARX,
    LS,
    ConstantMean,
    Normal,
    StudentsT,
    ZeroMean,
)
from arch.utility.cov import cov_nw
from arch.utility.timeseries import add_trend

NOBS = 120


def _walk(seed: int, nvar: int = 1) -> np.ndarray:
    """Integer valued random walks in [0, 255], so any data type can hold them."""
    rng = np.random.default_rng(seed)
    out = 100.0 + np.cumsum(rng.integers(-2, 3, size=(NOBS, nvar)), axis=0)
    assert out.min() >= 0
    assert out.max() <= 255
    return out[:, 0] if nvar == 1 else out


def _noise(seed: int, nvar: int = 1) -> np.ndarray:
    """Integer valued noise in [0, 20]."""
    rng = np.random.default_rng(seed)
    out = rng.integers(0, 21, size=(NOBS, nvar)).astype(float)
    return out[:, 0] if nvar == 1 else out


Y = _walk(1)
X = _walk(2, 2)
E = _noise(3)
E2 = _noise(4, 2)
SIGMA2 = np.full(NOBS, 10.0)

# The data types that are rejected, and the text of the data type in the message
REJECTED: list[tuple[type, str]] = [
    (np.float16, "float16"),
    (np.complex64, "complex64"),
    (np.complex128, "complex128"),
]
# The data types that are converted to float64
CONVERTED: list[type] = [
    np.int8,
    np.int32,
    np.int64,
    np.uint8,
    np.uint64,
    np.float32,
    np.float64,
    np.longdouble,
]
COMPLEX: list[type] = [np.complex64, np.complex128]


@dataclass(frozen=True)
class Target:
    """
    An entry point and its data.

    arrays are valid arguments, integer valued so that every data type can hold them.
    """

    name: str
    func: Callable[..., Any]
    arrays: Mapping[str, np.ndarray]
    other: Mapping[str, Any] = field(default_factory=dict)
    complex_ok: tuple[str, ...] = ()


def _adf(**kw: Any) -> Any:
    return ADF(**kw).stat


def _dfgls(**kw: Any) -> Any:
    return DFGLS(**kw).stat


def _pp(**kw: Any) -> Any:
    return PhillipsPerron(**kw).stat


def _kpss(**kw: Any) -> Any:
    return KPSS(**kw).stat


def _vr(**kw: Any) -> Any:
    return VarianceRatio(**kw).stat


def _za(**kw: Any) -> Any:
    return ZivotAndrews(**kw).stat


def _bandwidth(**kw: Any) -> Any:
    return auto_bandwidth(**kw)


def _eg(**kw: Any) -> Any:
    return engle_granger(**kw).stat


def _po(**kw: Any) -> Any:
    return phillips_ouliaris(**kw).stat


def _dols(**kw: Any) -> Any:
    return DynamicOLS(**kw).fit().params


def _fmols(**kw: Any) -> Any:
    return FullyModifiedOLS(**kw).fit().params


def _ccr(**kw: Any) -> Any:
    return CanonicalCointegratingReg(**kw).fit().params


def _fixed_garch(mod: Any) -> Any:
    # Evaluate the model at fixed parameters instead of fitting it. Results are not
    # bit-identical between calls on every platform (macOS on x86-64), and the
    # optimizer amplifies a difference in the last bits, especially at an estimate on
    # the boundary such as the GARCH(1,1) of these i.i.d. data
    params = np.r_[np.full(mod.num_params, 0.01), 0.1, 0.1, 0.8]
    res = mod.fix(params)
    return np.r_[res.loglikelihood, np.asarray(res.conditional_volatility)]


def _garch_model(**kw: Any) -> Any:
    return _fixed_garch(arch_model(**kw))


def _arx(**kw: Any) -> Any:
    return _fixed_garch(arch_model(mean="ARX", lags=1, **kw))


def _constant_mean(**kw: Any) -> Any:
    return ConstantMean(**kw).fit(disp="off").params


def _zero_mean(**kw: Any) -> Any:
    return ZeroMean(**kw).fit(disp="off").params


def _ls(**kw: Any) -> Any:
    return LS(**kw).fit(disp="off").params


def _harx(**kw: Any) -> Any:
    return HARX(lags=[1, 3], **kw).fit(disp="off").params


def _arx_class(**kw: Any) -> Any:
    return ARX(lags=1, **kw).fit(disp="off").params


def _garch_start(**kw: Any) -> Any:
    return GARCH().starting_values(**kw)


def _garch_backcast(**kw: Any) -> Any:
    return GARCH().backcast(**kw)


def _normal_ll(**kw: Any) -> Any:
    return Normal().loglikelihood(parameters=np.empty(0), **kw)


def _t_ll(**kw: Any) -> Any:
    return StudentsT().loglikelihood(parameters=np.array([8.0]), **kw)


def _normal_cdf(**kw: Any) -> Any:
    return Normal().cdf(**kw)


def _bartlett(**kw: Any) -> Any:
    return Bartlett(**kw).cov.long_run


def _parzen(**kw: Any) -> Any:
    return Parzen(**kw).cov.long_run


def _qs(**kw: Any) -> Any:
    return QuadraticSpectral(**kw).cov.long_run


def _zero_lag(**kw: Any) -> Any:
    return ZeroLag(**kw).cov.long_run


def _pwrc(**kw: Any) -> Any:
    return PreWhitenedRecolored(**kw).cov.long_run


def _pwrc_var_hac(**kw: Any) -> Any:
    return PreWhitenedRecolored(kernel=None, lags=1, **kw).cov.long_run


def _cov_nw(**kw: Any) -> Any:
    return cov_nw(lags=2, **kw)


def _block(**kw: Any) -> Any:
    return optimal_block_length(**kw)


def _mcs(**kw: Any) -> Any:
    return MCS(size=0.2, reps=50, seed=0, **kw).compute()


def _spa(**kw: Any) -> Any:
    return SPA(reps=50, seed=0, **kw).compute()


def _stepm(**kw: Any) -> Any:
    return StepM(reps=50, seed=0, **kw).compute()


def _add_trend(**kw: Any) -> Any:
    return add_trend(trend="ct", **kw)


TARGETS: list[Target] = [
    # Unit root tests
    Target("ADF", _adf, {"y": Y}),
    Target("DFGLS", _dfgls, {"y": Y}),
    Target("PhillipsPerron", _pp, {"y": Y}),
    Target("KPSS", _kpss, {"y": Y}),
    Target("VarianceRatio", _vr, {"y": Y}, {"lags": 4}),
    Target("ZivotAndrews", _za, {"y": Y}, {"max_lags": 2}),
    Target("auto_bandwidth", _bandwidth, {"y": Y}),
    # Cointegration
    Target("engle_granger", _eg, {"y": Y, "x": X}),
    Target("phillips_ouliaris", _po, {"y": Y, "x": X}),
    Target("DynamicOLS", _dols, {"y": Y, "x": X}, {"lags": 1, "leads": 1}),
    Target("FullyModifiedOLS", _fmols, {"y": Y, "x": X}),
    Target("CanonicalCointegratingReg", _ccr, {"y": Y, "x": X}),
    # ARCH models
    Target("arch_model", _garch_model, {"y": Y}),
    Target("arch_model-ARX", _arx, {"y": Y, "x": X}),
    Target("ConstantMean", _constant_mean, {"y": Y}),
    Target("ZeroMean", _zero_mean, {"y": Y}),
    Target("LS", _ls, {"y": Y, "x": X}),
    Target("HARX", _harx, {"y": Y, "x": X}),
    Target("ARX", _arx_class, {"y": Y}),
    # Volatility processes and distributions
    Target("GARCH.starting_values", _garch_start, {"resids": E}),
    Target("GARCH.backcast", _garch_backcast, {"resids": E}),
    Target("Normal.loglikelihood", _normal_ll, {"resids": E, "sigma2": SIGMA2}),
    Target("StudentsT.loglikelihood", _t_ll, {"resids": E, "sigma2": SIGMA2}),
    Target("Normal.cdf", _normal_cdf, {"resids": E}),
    # Covariance estimators: Hermitian for complex data, except the VAR
    Target("Bartlett", _bartlett, {"x": E2}, complex_ok=("x",)),
    Target("Parzen", _parzen, {"x": E2}, complex_ok=("x",)),
    Target("QuadraticSpectral", _qs, {"x": E2}, complex_ok=("x",)),
    Target("ZeroLag", _zero_lag, {"x": E2}, complex_ok=("x",)),
    Target("PreWhitenedRecolored", _pwrc, {"x": E2}, {"lags": 1}),
    Target("PreWhitenedRecolored-VAR-HAC", _pwrc_var_hac, {"x": E2}),
    Target("cov_nw", _cov_nw, {"y": E2}, complex_ok=("y",)),
    # Bootstrap utilities
    Target("optimal_block_length", _block, {"x": E2}),
    Target("MCS", _mcs, {"losses": E2}),
    Target("SPA", _spa, {"benchmark": E, "models": E2}),
    Target("StepM", _stepm, {"benchmark": E, "models": E2}),
    # Utilities, where complex data are accepted
    Target("add_trend", _add_trend, {"x": E2}, complex_ok=("x",)),
]


def _row(
    target: Target,
    name: str,
    value: Any,
    label: str,
    exc: type[Exception] | None,
    match: str | None,
) -> Row:
    kwargs = {**target.arrays, **target.other, name: value}
    return Row(
        target.name,
        target.func,
        kwargs,
        exc,
        match,
        id=f"{target.name}-{name}-{label}",
        name=name,
    )


def _rows() -> list[Row]:
    rows: list[Row] = []
    for target in TARGETS:
        for name, arr in target.arrays.items():
            complex_ok = name in target.complex_ok
            for dtype, text in REJECTED:
                if complex_ok and dtype in COMPLEX:
                    continue
                match = rf"^{name} must be .*; got {text} data with shape "
                rows.append(
                    _row(target, name, arr.astype(dtype), text, TypeError, match)
                )
            for dtype in CONVERTED:
                label = dtype.__name__
                rows.append(_row(target, name, arr.astype(dtype), label, None, None))
            if complex_ok:
                for dtype in COMPLEX:
                    label = np.dtype(dtype).name
                    value = arr.astype(dtype) + 1j * arr.astype(dtype)[::-1]
                    rows.append(_row(target, name, value, label, None, None))
    return rows


ROWS = _rows()


@pytest.mark.parametrize("row", to_params(ROWS))
def test_dtype(row: Row) -> None:
    run_row(row)


# pandas objects: the data type of each column is checked
PANDAS_TARGETS = [
    t
    for t in TARGETS
    if "y" in t.arrays and t.arrays["y"].ndim == 1 and t.name != "auto_bandwidth"
]


@pytest.mark.parametrize(
    "target", PANDAS_TARGETS, ids=ids_for([t.name for t in PANDAS_TARGETS])
)
@pytest.mark.parametrize(
    "dtype", [np.float16, np.complex128], ids=["float16", "complex"]
)
def test_pandas_series_dtype(target: Target, dtype: type) -> None:
    index = pd.date_range("2000-01-31", periods=NOBS, freq="D")
    series = pd.Series(target.arrays["y"].astype(dtype), index=index)
    kwargs = {**target.arrays, **target.other, "y": series}
    text = np.dtype(dtype).name
    if "x" in kwargs and kwargs["x"].ndim == 2:
        kwargs["x"] = pd.DataFrame(kwargs["x"], index=index)
    with pytest.raises(TypeError, match=rf"^y must be .*; got {text} data with shape"):
        target.func(**kwargs)


BAD_COLUMN_CALLS: list[tuple[type, str, Callable[[pd.DataFrame], Any]]] = [
    (np.float16, "Bartlett", Bartlett),
    (np.float16, "engle_granger", lambda frame: engle_granger(Y, frame)),
    (np.complex128, "PreWhitenedRecolored", PreWhitenedRecolored),
    (np.complex128, "engle_granger", lambda frame: engle_granger(Y, frame)),
    (np.complex128, "optimal_block_length", optimal_block_length),
]


@pytest.mark.parametrize(
    ("dtype", "name", "call"),
    BAD_COLUMN_CALLS,
    ids=[f"{name}-{np.dtype(dtype).name}" for dtype, name, _ in BAD_COLUMN_CALLS],
)
def test_dataframe_with_one_bad_column(
    dtype: type, name: str, call: Callable[[pd.DataFrame], Any]
) -> None:
    # The merged values of the columns would be float64 (float16 with float64) or
    # complex (complex with float64), and the data type of each column is checked
    frame = pd.DataFrame(np.tile(X[:, [0]], (1, 3)) + _noise(5, 3))
    frame[1] = frame[1].astype(dtype)
    text = np.dtype(dtype).name
    with pytest.raises(TypeError, match=rf"^x must be .*; got {text} data with shape"):
        call(frame)


def test_float16_message_names_the_accepted_types() -> None:
    with pytest.raises(TypeError) as info:
        ADF(Y.astype(np.float16))
    assert str(info.value) == (
        "y must be a 1-d array of finite real numbers (integer, Boolean, float32 or "
        f"float64 data); got float16 data with shape ({NOBS},) (ndarray)"
    )


def test_complex_message_when_complex_is_accepted() -> None:
    with pytest.raises(TypeError) as info:
        Bartlett(E2.astype(np.float16))
    assert str(info.value) == (
        "x must be a 2-d array of finite real or complex numbers (integer, Boolean, "
        "float32, float64, complex64 or complex128 data) with at least 2 rows; got "
        f"float16 data with shape ({NOBS}, 2) (ndarray)"
    )
