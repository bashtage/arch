from itertools import product

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest

from arch.data import sp500
from arch.univariate import (
    APARCH,
    ARX,
    EGARCH,
    FIGARCH,
    GARCH,
    HARCH,
    HARX,
    LS,
    ARCHInMean,
    ConstantMean,
    ConstantVariance,
    EWMAVariance,
    MIDASHyperbolic,
    RiskMetrics2006,
    ZeroMean,
    arch_model,
)
from arch.univariate.base import ARCHModel
from arch.utility.array import append_same_type
from arch.utility.exceptions import DataScaleWarning

SP500 = 100 * sp500.load()["Adj Close"].pct_change().dropna()
N = SP500.shape[0]
SP500_initial = SP500.iloc[: N // 2]
SP500_append = SP500.iloc[N // 2 :]

# A smaller sample is used when the size of the sample is not important
SMALL = SP500.iloc[:700]
SMALL_initial = SMALL.iloc[:500]
SMALL_append = SMALL.iloc[500:]


class HARXWrapper(HARX):
    def __init__(self, y, x=None, volatility=None, **kwargs):
        super().__init__(y, lags=[1, 5], x=x, volatility=volatility, **kwargs)


class ARXWrapper(ARX):
    def __init__(self, y, x=None, volatility=None, **kwargs):
        super().__init__(y, lags=2, x=x, volatility=volatility, **kwargs)


class ARCHInMeanWrapper(ARCHInMean):
    def __init__(self, y, x=None, volatility=None, **kwargs):
        super().__init__(y, lags=1, x=x, volatility=volatility, **kwargs)


class LSWrapper(LS):
    def __init__(self, y, x=None, volatility=None, **kwargs):
        super().__init__(y, x=x, volatility=volatility, **kwargs)


class ConstantMeanWrapper(ConstantMean):
    def __init__(self, y, x=None, volatility=None, **kwargs):
        super().__init__(y, volatility=volatility, **kwargs)


class ZeroMeanWrapper(ZeroMean):
    def __init__(self, y, x=None, volatility=None, **kwargs):
        super().__init__(y, volatility=volatility, **kwargs)


MEAN_MODELS = [
    HARXWrapper,
    ARXWrapper,
    ConstantMean,
    ZeroMean,
]

VOLATILITIES = [
    ConstantVariance(),
    GARCH(),
    FIGARCH(),
    EWMAVariance(lam=0.94),
    MIDASHyperbolic(),
    HARCH(lags=[1, 5, 22]),
    RiskMetrics2006(),
    APARCH(),
    EGARCH(),
]

X_MEAN_MODELS = [HARXWrapper, ARXWrapper, LS]
# Mean models that can be used with and without x
EXOG_MODELS = [HARXWrapper, ARXWrapper, LSWrapper, ARCHInMeanWrapper]
ALL_MODELS = [
    HARXWrapper,
    ARXWrapper,
    LSWrapper,
    ARCHInMeanWrapper,
    ConstantMeanWrapper,
    ZeroMeanWrapper,
]

MODEL_SPECS = list(product(MEAN_MODELS, VOLATILITIES))

IDS = [f"{mean.__name__}-{str(vol).split('(')[0]}" for mean, vol in MODEL_SPECS]

DATA_KINDS = ["series", "frame", "array", "list", "tuple"]
NO_RESCALE = {"rescale": False}


@pytest.fixture(params=MODEL_SPECS, ids=IDS)
def mean_volatility(request):
    mean, vol = request.param
    return mean, vol


def as_type(s, kind):
    if kind == "series":
        return s
    elif kind == "frame":
        return s.to_frame()
    elif kind == "array":
        return s.to_numpy()
    elif kind == "list":
        return s.tolist()
    elif kind == "tuple":
        return tuple(s.tolist())
    raise NotImplementedError(kind)  # pragma: no cover


def make_x(index, ncol=2, seed=1234):
    rs = np.random.RandomState(seed)
    return pd.DataFrame(
        rs.standard_normal((len(index), ncol)),
        columns=["a", "b", "c"][:ncol],
        index=index,
    )


def fit(mod, **kwargs):
    return mod.fit(disp="off", show_warning=False, **kwargs)


def assert_results_equal(res, direct):
    assert_allclose(res.params, direct.params, rtol=1e-7, atol=1e-9)
    assert_allclose(res.loglikelihood, direct.loglikelihood, rtol=1e-8)
    assert_allclose(res.resid, direct.resid, equal_nan=True, rtol=1e-7, atol=1e-9)
    assert_allclose(
        res.conditional_volatility,
        direct.conditional_volatility,
        equal_nan=True,
        rtol=1e-7,
        atol=1e-9,
    )
    assert res.nobs == direct.nobs
    if isinstance(res.resid, pd.Series):
        assert res.resid.index.equals(direct.resid.index)


def assert_models_equal(mod, direct):
    """Check that appending produces the same model as constructing directly"""
    assert type(mod.y) is type(direct.y)
    assert_allclose(np.asarray(mod.y, dtype=float), np.asarray(direct.y, dtype=float))
    assert_allclose(mod._y, direct._y)
    assert mod._y_series.index.equals(direct._y_series.index)
    assert mod._y_series.name == direct._y_series.name
    assert_allclose(mod.regressors, direct.regressors)
    assert mod.parameter_names() == direct.parameter_names()
    assert mod._fit_indices == direct._fit_indices or direct._fit_indices == [
        0,
        direct._y.shape[0],
    ]
    if direct.x is None:
        assert mod.x is None
    else:
        assert type(mod.x) is type(direct.x)
        assert_allclose(mod.x, direct.x)
        assert mod._x_names == direct._x_names


def snapshot(mod):
    x = None if mod.x is None else np.array(mod.x, dtype=float)
    return {
        "y": mod._y.copy(),
        "y_original": np.array(mod.y, dtype=float),
        "y_type": type(mod.y),
        "index": mod._y_series.index.copy(),
        "name": mod._y_series.name,
        "regressors": mod.regressors.copy(),
        "fit_indices": list(mod._fit_indices),
        "fit_y": mod._fit_y.copy(),
        "x": x,
        "backcast": mod._backcast,
        "scale": mod.scale,
    }


def assert_snapshot_unchanged(mod, snap):
    assert_array_equal(mod._y, snap["y"])
    assert_array_equal(np.array(mod.y, dtype=float), snap["y_original"])
    assert isinstance(mod.y, snap["y_type"])
    assert mod._y_series.index.equals(snap["index"])
    assert mod._y_series.name == snap["name"]
    assert_array_equal(mod.regressors, snap["regressors"])
    assert mod._fit_indices == snap["fit_indices"]
    assert_array_equal(mod._fit_y, snap["fit_y"])
    if snap["x"] is None:
        assert mod.x is None
    else:
        assert_array_equal(np.array(mod.x, dtype=float), snap["x"])
    assert mod._backcast is snap["backcast"]
    assert mod.scale == snap["scale"]


def test_append():
    mod = arch_model(SP500_initial)
    mod.append(SP500_append)
    res = mod.fit(disp="off")

    direct = arch_model(SP500)
    res_direct = direct.fit(disp="off")
    assert_allclose(res.params, res_direct.params, rtol=1e-5)
    assert_allclose(res.conditional_volatility, res_direct.conditional_volatility)
    assert_allclose(res.resid, res_direct.resid)
    assert_allclose(mod._backcast, direct._backcast)


def test_alt_means(mean_volatility):
    mean, vol = mean_volatility
    mod = mean(SP500_initial, volatility=vol)
    mod.append(SP500_append)
    res = mod.fit(disp="off")

    direct = mean(SP500, volatility=vol)
    res_direct = direct.fit(disp="off")
    assert_allclose(res.conditional_volatility, res_direct.conditional_volatility)
    assert_allclose(res.resid, res_direct.resid)
    if mod._backcast is not None:
        assert_allclose(mod._backcast, direct._backcast)
    else:
        assert direct._backcast is None


def test_alt_means_params(mean_volatility):
    mean, vol = mean_volatility
    mod = mean(SMALL_initial, volatility=vol)
    mod.append(SMALL_append)
    direct = mean(SMALL, volatility=vol)
    assert_models_equal(mod, direct)
    assert_results_equal(fit(mod), fit(direct))


@pytest.mark.parametrize("kind", DATA_KINDS)
@pytest.mark.parametrize("mean", [ARXWrapper, HARXWrapper, ConstantMean, ZeroMean])
def test_container_types(mean, kind):
    mod = mean(as_type(SMALL_initial, kind), volatility=GARCH())
    mod.append(as_type(SMALL_append, kind))
    direct = mean(as_type(SMALL, kind), volatility=GARCH())
    assert_models_equal(mod, direct)
    assert_results_equal(fit(mod), fit(direct))
    # The index of the results must be aligned with the extended data
    if kind in ("series", "frame"):
        assert fit(mod).resid.index.equals(SMALL.index)


@pytest.mark.parametrize("kind", ["array", "list"])
def test_append_scalar(mean_volatility, kind):
    mean, vol = mean_volatility
    mod = mean(as_type(SMALL_initial, kind), volatility=vol)
    for val in np.asarray(SMALL_append.iloc[:25]):
        mod.append(val)
    direct = mean(as_type(SMALL.iloc[:525], kind), volatility=vol)
    assert_models_equal(mod, direct)
    assert_results_equal(fit(mod), fit(direct))


@pytest.mark.parametrize(
    "scalar",
    [1, 0.5, np.float32(0.25), np.int64(2), np.float64(1)],
    ids=["int", "float", "float32", "int64", "float64"],
)
def test_append_scalar_types(scalar):
    mod = ConstantMean(np.asarray(SMALL_initial))
    mod.append(scalar)
    assert mod.y.shape[0] == SMALL_initial.shape[0] + 1
    assert mod._y[-1] == float(scalar)
    mod = ConstantMean(SMALL_initial.tolist())
    mod.append(scalar)
    assert isinstance(mod.y, list)
    assert len(mod.y) == SMALL_initial.shape[0] + 1
    assert mod._y[-1] == float(scalar)


@pytest.mark.parametrize(
    "scalar",
    [True, np.bool_(False), "1.0", None],
    ids=["bool", "np-bool", "str", "none"],
)
def test_append_scalar_bad_type(scalar):
    mod = ConstantMean(np.asarray(SMALL_initial))
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(scalar)
    assert mod._y.shape[0] == SMALL_initial.shape[0]


def test_append_scalar_bad_value():
    mod = HARX(SP500_initial, lags=[1, 5], volatility=GARCH())
    with pytest.raises(TypeError):
        mod.append(SP500_append.iloc[0])


def test_append_type_mismatch(mean_volatility):
    mean, vol = mean_volatility
    mod = mean(SP500_initial, volatility=vol)
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(np.asarray(SP500_append))
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(SP500_append.tolist())

    mod_arr = mean(np.asarray(SP500_initial), volatility=vol)
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod_arr.append(SP500_append)
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod_arr.append(SP500_append.tolist())

    mod_list = mean(SP500_initial.tolist(), volatility=vol)
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod_list.append(SP500_append)
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod_list.append(np.asarray(SP500_append))


def test_append_series_frame_mismatch():
    mod = ConstantMean(SMALL_initial)
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(SMALL_append.to_frame())
    mod = ConstantMean(SMALL_initial.to_frame())
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(SMALL_append)


def test_append_frame_column_mismatch():
    mod = ConstantMean(SMALL_initial.to_frame())
    snap = snapshot(mod)
    with pytest.raises(ValueError, match="The columns of the appended data"):
        mod.append(SMALL_append.rename("other").to_frame())
    assert_snapshot_unchanged(mod, snap)
    two_cols = pd.concat([SMALL_append, SMALL_append], axis=1)
    with pytest.raises(ValueError, match="The columns of the appended data"):
        mod.append(two_cols)
    assert_snapshot_unchanged(mod, snap)


def test_base_class_append():
    # Models that do not derive from HARX use the base class implementation
    mod = ConstantMean(SMALL_initial, volatility=GARCH())
    fit(mod)
    ARCHModel.append(mod, SMALL_append)
    assert mod._y.shape[0] == SMALL.shape[0]
    assert mod._fit_indices == [0, SMALL.shape[0]]
    assert mod._fit_y is mod._y
    assert mod._backcast is None
    assert mod._var_bounds is None
    pd.testing.assert_series_equal(mod.y, SMALL)
    assert mod._y_series.index.equals(SMALL.index)
    with pytest.raises(ValueError, match="overlaps the index"):
        ARCHModel.append(mod, SMALL_append)
    # The base class has no exogenous regressors, and nothing is changed
    snap = snapshot(mod)
    new_y = pd.Series([0.1], index=[SMALL.index[-1] + pd.Timedelta(days=1)])
    with pytest.raises(ValueError, match="does not include exogenous regressors"):
        ARCHModel.append(mod, new_y, x=np.ones((1, 1)))
    assert_snapshot_unchanged(mod, snap)
    with pytest.raises(RuntimeError, match="created without data"):
        ARCHModel.append(ConstantMean(None), SMALL_append)


def test_append_without_data():
    mod = ConstantMean(None)
    with pytest.raises(RuntimeError, match="created without data"):
        mod.append(SMALL_append)
    mod = arch_model(None)
    with pytest.raises(RuntimeError, match="created without data"):
        mod.append(np.zeros(10))


def test_append_empty():
    mod = ConstantMean(SMALL_initial)
    with pytest.raises(ValueError, match="at least one observation"):
        mod.append(SMALL_append.iloc[:0])
    mod = ConstantMean(np.asarray(SMALL_initial))
    with pytest.raises(ValueError, match="at least one observation"):
        mod.append(np.empty(0))
    mod = ConstantMean(SMALL_initial.tolist())
    with pytest.raises(ValueError, match="at least one observation"):
        mod.append([])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("kind", ["series", "array", "list", "scalar"])
def test_append_non_finite(kind, bad):
    new = SMALL_append.iloc[:5].copy()
    new.iloc[2] = bad
    mod_kind = "series" if kind == "series" else kind
    if kind == "scalar":
        mod_kind = "array"
        new = bad
    elif kind != "series":
        new = as_type(new, kind)
    mod = ARXWrapper(as_type(SMALL_initial, mod_kind), volatility=GARCH())
    snap = snapshot(mod)
    with pytest.raises(ValueError, match="NaN or inf values found in y"):
        mod.append(new)
    assert_snapshot_unchanged(mod, snap)


def test_append_preserves_dtypes_of_inputs():
    values = SMALL_initial.to_numpy()
    new_values = SMALL_append.to_numpy()
    values_copy, new_copy = values.copy(), new_values.copy()
    mod = ConstantMean(values)
    mod.append(new_values)
    assert_array_equal(values, values_copy)
    assert_array_equal(new_values, new_copy)
    assert values.shape[0] == SMALL_initial.shape[0]

    as_list = SMALL_initial.tolist()
    new_list = SMALL_append.tolist()
    mod = ConstantMean(as_list)
    mod.append(new_list)
    mod.append(1.0)
    assert len(as_list) == SMALL_initial.shape[0]
    assert len(new_list) == SMALL_append.shape[0]
    assert len(mod.y) == SMALL.shape[0] + 1


def test_append_does_not_modify_pandas_inputs():
    orig = SMALL_initial.copy()
    new = SMALL_append.copy()
    mod = ARXWrapper(orig, volatility=GARCH())
    mod.append(new)
    pd.testing.assert_series_equal(orig, SMALL_initial)
    pd.testing.assert_series_equal(new, SMALL_append)
    assert mod.y is not orig
    assert orig.shape[0] == SMALL_initial.shape[0]
    assert mod.y.shape[0] == SMALL.shape[0]


@pytest.mark.parametrize("new_name", [None, "other", "Adj Close"])
def test_append_series_name(new_name):
    orig = SMALL_initial.rename("returns")
    new = SMALL_append.rename(new_name)
    mod = ARXWrapper(orig, volatility=GARCH())
    names = mod.parameter_names()
    mod.append(new)
    assert mod.parameter_names() == names
    assert mod._y_series.name == "returns"
    assert mod.y.name == "returns"
    assert new.name == new_name
    res = fit(mod)
    assert res.resid.name == "resid"
    assert "returns[1]" in res.params.index


def test_append_unnamed_series():
    orig = SMALL_initial.rename(None)
    new = SMALL_append.rename(None)
    mod = ARXWrapper(orig, volatility=GARCH())
    names = mod.parameter_names()
    mod.append(new)
    assert mod.parameter_names() == names
    direct = ARXWrapper(SMALL.rename(None), volatility=GARCH())
    assert mod.parameter_names() == direct.parameter_names()
    assert_models_equal(mod, direct)


def test_append_array_2d_column():
    orig = SMALL_initial.to_numpy()[:, None]
    mod = ConstantMean(orig, volatility=GARCH())
    mod.append(SMALL_append.to_numpy())
    assert mod.y.shape == (SMALL.shape[0], 1)
    mod.append(SMALL_append.to_numpy()[:, None])
    assert mod.y.shape == (SMALL.shape[0] + SMALL_append.shape[0], 1)
    mod.append(1.0)
    assert mod.y.shape[0] == SMALL.shape[0] + SMALL_append.shape[0] + 1
    direct = ConstantMean(
        np.concatenate((SMALL, SMALL_append, [1.0]))[:, None], volatility=GARCH()
    )
    assert_models_equal(mod, direct)
    assert_results_equal(fit(mod), fit(direct))

    mod = ConstantMean(orig)
    with pytest.raises(ValueError, match="same number of columns"):
        mod.append(np.ones((5, 2)))


class TestIndex:
    def test_overlap(self):
        mod = ARXWrapper(SMALL_initial, volatility=GARCH())
        snap = snapshot(mod)
        with pytest.raises(ValueError, match="overlaps the index"):
            mod.append(SMALL.iloc[490:510])
        assert_snapshot_unchanged(mod, snap)
        # Complete overlap
        with pytest.raises(ValueError, match="overlaps the index"):
            mod.append(SMALL_initial)
        assert_snapshot_unchanged(mod, snap)

    def test_before(self):
        mod = ARXWrapper(SMALL_append, volatility=GARCH())
        snap = snapshot(mod)
        with pytest.raises(ValueError, match="must be increasing"):
            mod.append(SMALL_initial)
        assert_snapshot_unchanged(mod, snap)

    def test_not_increasing(self):
        mod = ARXWrapper(SMALL_initial, volatility=GARCH())
        snap = snapshot(mod)
        with pytest.raises(ValueError, match="must be increasing"):
            mod.append(SMALL_append.iloc[::-1])
        assert_snapshot_unchanged(mod, snap)

    def test_gap_allowed(self):
        mod = ARXWrapper(SMALL.iloc[:300], volatility=GARCH())
        mod.append(SMALL.iloc[400:])
        assert mod._y.shape[0] == 300 + 300
        assert mod._y_series.index.is_monotonic_increasing

    def test_reset_index(self):
        orig = SMALL_initial.reset_index(drop=True)
        new = SMALL_append.reset_index(drop=True)
        mod = ARXWrapper(orig, volatility=GARCH())
        with pytest.raises(ValueError, match="overlaps the index"):
            mod.append(new)
        new.index = pd.RangeIndex(orig.shape[0], orig.shape[0] + new.shape[0])
        mod.append(new)
        direct = ARXWrapper(SMALL.reset_index(drop=True), volatility=GARCH())
        assert_models_equal(mod, direct)

    def test_incompatible_index(self):
        mod = ARXWrapper(SMALL_initial, volatility=GARCH())
        snap = snapshot(mod)
        with pytest.raises(ValueError, match="index"):
            mod.append(SMALL_append.reset_index(drop=True))
        assert_snapshot_unchanged(mod, snap)

    def test_unsorted_original(self):
        rs = np.random.RandomState(8942)
        order = rs.permutation(SMALL_initial.shape[0])
        orig = SMALL_initial.iloc[order]
        mod = ARXWrapper(orig, volatility=GARCH())
        # No ordering restriction when the existing index is not increasing
        mod.append(SMALL_append.iloc[::-1])
        assert mod._y.shape[0] == SMALL.shape[0]
        with pytest.raises(ValueError, match="overlaps"):
            mod.append(SMALL_append.iloc[:3])

    def test_array_models_have_default_index(self):
        mod = ARXWrapper(np.asarray(SMALL_initial), volatility=GARCH())
        mod.append(np.asarray(SMALL_append))
        assert_array_equal(
            mod._y_series.index.to_numpy(), np.arange(SMALL.shape[0], dtype=int)
        )
        assert mod._y_series.index.is_unique


@pytest.mark.parametrize("mean", EXOG_MODELS)
@pytest.mark.parametrize("vol", [GARCH(), EGARCH()])
@pytest.mark.parametrize("x_kind", ["frame", "array"])
def test_append_exog(mean, vol, x_kind):
    x = make_x(SMALL.index)
    if x_kind == "array":
        x = x.to_numpy()
    x_initial, x_append = x[:500], x[500:]
    mod = mean(SMALL_initial, x=x_initial, volatility=vol)
    mod.append(SMALL_append, x=x_append)
    direct = mean(SMALL, x=x, volatility=vol)
    assert_models_equal(mod, direct)
    assert_results_equal(fit(mod), fit(direct))
    assert mod.parameter_names() == direct.parameter_names()


@pytest.mark.parametrize("mean", [HARXWrapper, ARXWrapper, LSWrapper])
def test_append_exog_constant_variance(mean):
    x = make_x(SMALL.index)
    mod = mean(SMALL_initial, x=x.iloc[:500], volatility=ConstantVariance())
    mod.append(SMALL_append, x=x.iloc[500:])
    direct = mean(SMALL, x=x, volatility=ConstantVariance())
    assert_models_equal(mod, direct)
    assert_results_equal(fit(mod), fit(direct))


@pytest.mark.parametrize("mean", EXOG_MODELS)
def test_append_exog_fit_before(mean):
    x = make_x(SMALL.index)
    mod = mean(SMALL_initial, x=x.iloc[:500], volatility=GARCH())
    fit(mod)
    mod.append(SMALL_append, x=x.iloc[500:])
    direct = mean(SMALL, x=x, volatility=GARCH())
    assert_results_equal(fit(mod), fit(direct))


@pytest.mark.parametrize("mean", EXOG_MODELS)
def test_append_exog_sequential(mean):
    x = make_x(SMALL.index)
    mod = mean(SMALL.iloc[:300], x=x.iloc[:300], volatility=GARCH())
    for start in (300, 450, 600):
        stop = start + 150 if start < 600 else 700
        mod.append(SMALL.iloc[start:stop], x=x.iloc[start:stop])
    direct = mean(SMALL, x=x, volatility=GARCH())
    assert_models_equal(mod, direct)
    assert_results_equal(fit(mod), fit(direct))


def test_append_exog_series():
    x = make_x(SMALL.index, ncol=1)["a"]
    for mean in EXOG_MODELS:
        mod = mean(SMALL_initial, x=x.iloc[:500], volatility=GARCH())
        mod.append(SMALL_append, x=x.iloc[500:])
        direct = mean(SMALL, x=x, volatility=GARCH())
        assert_models_equal(mod, direct)
        assert_results_equal(fit(mod), fit(direct))


def test_append_exog_single_column_array_forms():
    x = make_x(SMALL.index, ncol=1).to_numpy()
    direct = ARXWrapper(SMALL, x=x, volatility=GARCH())
    res_direct = fit(direct)
    # 1-d and 2-d inputs for a single regressor
    for x_new in (x[500:, 0], x[500:]):
        mod = ARXWrapper(SMALL_initial, x=x[:500], volatility=GARCH())
        mod.append(SMALL_append, x=x_new)
        assert_allclose(mod.x, direct.x)
        assert_results_equal(fit(mod), res_direct)
    mod = ARXWrapper(SMALL_initial, x=x[:500], volatility=GARCH())
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(SMALL_append, x=x[500:, 0].tolist())


def test_append_exog_1d_original():
    x = make_x(SMALL.index, ncol=1).to_numpy()[:, 0]
    mod = ARXWrapper(SMALL_initial, x=x[:500], volatility=GARCH())
    mod.append(SMALL_append, x=x[500:])
    direct = ARXWrapper(SMALL, x=x, volatility=GARCH())
    assert mod._x_original.ndim == 1
    assert_allclose(mod.x, direct.x)
    assert_results_equal(fit(mod), fit(direct))


@pytest.mark.parametrize("ncol", [1, 3])
def test_append_exog_scalar_y(ncol):
    x = make_x(SMALL.index, ncol=ncol).to_numpy()
    mod = ARXWrapper(
        SMALL_initial.to_numpy(), x=x[:500], volatility=GARCH(), **NO_RESCALE
    )
    new_y = SMALL_append.to_numpy()[:10]
    for i, val in enumerate(new_y):
        # 1-d for k > 1 is one row, and anything for a single regressor
        row = x[500 + i] if ncol > 1 else x[500 + i, 0]
        mod.append(val, x=row)
    direct = ARXWrapper(SMALL.to_numpy()[:510], x=x[:510], volatility=GARCH())
    assert_models_equal(mod, direct)
    assert_results_equal(fit(mod), fit(direct))
    assert mod.x.shape == (510, ncol)


def test_append_exog_list_scalar_y():
    x = make_x(SMALL.index, ncol=2).to_numpy()
    mod = ARXWrapper(SMALL_initial.tolist(), x=x[:500].tolist(), volatility=GARCH())
    mod.append(0.2, x=[0.1, -0.2])
    mod.append([0.1, 0.3], x=[[0.1, -0.2], [0.5, 0.2]])
    assert isinstance(mod._x_original, list)
    assert len(mod._x_original) == 503
    assert mod.x.shape == (503, 2)
    assert_allclose(mod.x[500], [0.1, -0.2])
    assert_allclose(mod.x[-1], [0.5, 0.2])
    assert mod._y.shape[0] == 503


def test_append_exog_x_type_mismatch():
    x = make_x(SMALL.index)
    mod = ARXWrapper(SMALL_initial, x=x.iloc[:500], volatility=GARCH())
    snap = snapshot(mod)
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(SMALL_append, x=x.iloc[500:].to_numpy())
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(SMALL_append, x=x["a"].iloc[500:])
    assert_snapshot_unchanged(mod, snap)

    mod = ARXWrapper(SMALL_initial, x=x.iloc[:500].to_numpy(), volatility=GARCH())
    snap = snapshot(mod)
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(SMALL_append, x=x.iloc[500:])
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(SMALL_append, x=x.iloc[500:].to_numpy().tolist())
    assert_snapshot_unchanged(mod, snap)

    xs = x["a"]
    mod = ARXWrapper(SMALL_initial, x=xs.iloc[:500], volatility=GARCH())
    snap = snapshot(mod)
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(SMALL_append, x=xs.iloc[500:].to_frame())
    assert_snapshot_unchanged(mod, snap)


@pytest.mark.parametrize("mean", X_MEAN_MODELS)
def test_bad_append_model_with_exog(mean):
    mod = mean(SP500_initial, volatility=GARCH())
    rs = np.random.RandomState(438924)
    x = pd.DataFrame(
        rs.randn(SP500_append.shape[0], 2),
        columns=["a", "b"],
        index=SP500_append.index,
    )
    snap = snapshot(mod)
    with pytest.raises(ValueError, match="x was not provided in the original model"):
        mod.append(SP500_append, x=x)
    assert_snapshot_unchanged(mod, snap)

    x_initial = pd.DataFrame(
        rs.randn(SP500_initial.shape[0], 2),
        columns=["a", "b"],
        index=SP500_initial.index,
    )
    mod = mean(SP500_initial, x=x_initial, volatility=GARCH())
    snap = snapshot(mod)
    with pytest.raises(ValueError, match="x must be provided when appending"):
        mod.append(SP500_append)
    assert_snapshot_unchanged(mod, snap)


@pytest.mark.parametrize("mean", EXOG_MODELS)
def test_bad_append_exog_shape(mean):
    x = make_x(SMALL.index)
    mod = mean(SMALL_initial, x=x.iloc[:500], volatility=GARCH())
    snap = snapshot(mod)
    # Too few rows
    with pytest.raises(ValueError, match="same number of observations as y"):
        mod.append(SMALL_append, x=x.iloc[500:600])
    # Too many rows
    with pytest.raises(ValueError, match="same number of observations as y"):
        mod.append(SMALL_append.iloc[:100], x=x.iloc[500:])
    # Wrong number of columns
    with pytest.raises(ValueError, match="columns"):
        mod.append(SMALL_append, x=x[["a"]].iloc[500:])
    with pytest.raises(ValueError, match="columns"):
        mod.append(SMALL_append, x=x.assign(c=1.0).iloc[500:])
    # Right number of columns, wrong labels
    with pytest.raises(ValueError, match="columns"):
        mod.append(SMALL_append, x=x.iloc[500:].rename(columns={"b": "z"}))
    # Empty
    with pytest.raises(ValueError, match="at least one observation"):
        mod.append(SMALL_append, x=x.iloc[500:500])
    assert_snapshot_unchanged(mod, snap)

    x_arr = x.to_numpy()
    mod = mean(SMALL_initial, x=x_arr[:500], volatility=GARCH())
    snap = snapshot(mod)
    with pytest.raises(ValueError, match="same number of columns"):
        mod.append(SMALL_append, x=x_arr[500:, :1])
    with pytest.raises(ValueError, match="same number of columns"):
        mod.append(SMALL_append, x=np.hstack([x_arr[500:], x_arr[500:]]))
    with pytest.raises(ValueError, match="same number of observations as y"):
        mod.append(SMALL_append, x=x_arr[500:600])
    # Wrong number of dimensions
    with pytest.raises(ValueError, match="same number of columns"):
        mod.append(SMALL_append, x=x_arr[500:].reshape((-1, 2, 1)))
    assert_snapshot_unchanged(mod, snap)
    # The model is still valid
    mod.append(SMALL_append, x=x_arr[500:])
    direct = mean(SMALL, x=x_arr, volatility=GARCH())
    assert_results_equal(fit(mod), fit(direct))


def test_bad_append_ls():
    x = make_x(SMALL.index)
    mod = LS(SMALL_initial, x=x.iloc[:500], volatility=GARCH())
    snap = snapshot(mod)
    with pytest.raises(ValueError, match="x must be provided when appending"):
        mod.append(SMALL_append)
    with pytest.raises(ValueError, match="same number of observations as y"):
        mod.append(SMALL_append, x=x.iloc[500:510])
    assert_snapshot_unchanged(mod, snap)
    mod = LS(SMALL_initial, volatility=GARCH())
    with pytest.raises(ValueError, match="x was not provided in the original"):
        mod.append(SMALL_append, x=x.iloc[500:])
    # Succeeds without x when none are in the model
    mod.append(SMALL_append)
    assert_results_equal(fit(mod), fit(LS(SMALL, volatility=GARCH())))


def test_append_x_type_mismatch():
    x = make_x(SMALL.index).to_numpy()
    mod = HARXWrapper(SMALL_initial, x=x[:500])
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(SMALL_append, x=pd.DataFrame(x[500:]))
    mod = HARXWrapper(SMALL_initial, x=pd.DataFrame(x[:500]))
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(SMALL_append, x=x[500:])


@pytest.mark.parametrize("failure", ["y-type", "y-nan", "y-index", "x-rows", "x-type"])
def test_failed_append_is_atomic_with_exog(failure):
    x = make_x(SMALL.index)
    mod = ARXWrapper(SMALL_initial, x=x.iloc[:500], volatility=GARCH())
    fit(mod)
    snap = snapshot(mod)
    new_x = x.iloc[500:]
    if failure == "y-type":
        args, kwargs, exc = (SMALL_append.to_numpy(),), {"x": new_x}, TypeError
    elif failure == "y-nan":
        bad = SMALL_append.copy()
        bad.iloc[-1] = np.nan
        args, kwargs, exc = (bad,), {"x": new_x}, ValueError
    elif failure == "y-index":
        args, kwargs, exc = (SMALL.iloc[490:]), {"x": new_x}, ValueError
        args = (args,)
    elif failure == "x-rows":
        args, kwargs, exc = (SMALL_append,), {"x": new_x.iloc[:-1]}, ValueError
    else:
        args, kwargs, exc = (SMALL_append,), {"x": new_x.to_numpy()}, TypeError
    with pytest.raises(exc):
        mod.append(*args, **kwargs)
    assert_snapshot_unchanged(mod, snap)
    # The model is still functioning and can be appended to
    mod.append(SMALL_append, x=new_x)
    direct = ARXWrapper(SMALL, x=x, volatility=GARCH())
    assert_results_equal(fit(mod), fit(direct))


@pytest.mark.parametrize("kind", ["series", "array", "list"])
def test_failed_append_is_atomic(kind):
    mod = ARXWrapper(as_type(SMALL_initial, kind), volatility=GARCH())
    fit(mod)
    snap = snapshot(mod)
    bad = SMALL_append.copy()
    bad.iloc[3] = np.inf
    with pytest.raises(ValueError, match="NaN or inf"):
        mod.append(as_type(bad, kind))
    assert_snapshot_unchanged(mod, snap)
    other = "array" if kind == "series" else "series"
    with pytest.raises(TypeError, match="Input data must be the same"):
        mod.append(as_type(SMALL_append, other))
    assert_snapshot_unchanged(mod, snap)
    mod.append(as_type(SMALL_append, kind))
    direct = ARXWrapper(as_type(SMALL, kind), volatility=GARCH())
    assert_results_equal(fit(mod), fit(direct))


def test_sequential_appends():
    mod = ARXWrapper(SMALL.iloc[:200], volatility=GARCH())
    for start in range(200, 700, 100):
        mod.append(SMALL.iloc[start : start + 100])
    direct = ARXWrapper(SMALL, volatility=GARCH())
    assert_models_equal(mod, direct)
    assert_results_equal(fit(mod), fit(direct))

    mod = ARXWrapper(SMALL.iloc[:200], volatility=GARCH())
    mod.append(SMALL.iloc[200:])
    assert_results_equal(fit(mod), fit(direct))


@pytest.mark.parametrize("fit_first", [True, False])
@pytest.mark.parametrize("mean", MEAN_MODELS)
def test_fit_append_fit(mean, fit_first):
    mod = mean(SMALL_initial, volatility=GARCH())
    if fit_first:
        res_initial = fit(mod)
        params_initial = res_initial.params.copy()
    mod.append(SMALL_append)
    res = fit(mod)
    direct = mean(SMALL, volatility=GARCH())
    res_direct = fit(direct)
    assert_results_equal(res, res_direct)
    if fit_first:
        # Fitting using more data should differ
        assert not np.allclose(params_initial, res.params)
    assert res.nobs == res_direct.nobs
    assert mod._fit_indices == direct._fit_indices


@pytest.mark.filterwarnings("ignore:overflow encountered in power:RuntimeWarning")
@pytest.mark.parametrize("dist", ["normal", "t", "skewt", "ged"])
@pytest.mark.parametrize("mean", ["Constant", "Zero", "AR", "HAR", "LS"])
@pytest.mark.parametrize(
    "vol_args",
    [
        {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
        {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
        {"vol": "ARCH", "p": 2},
    ],
    ids=["garch", "egarch", "arch"],
)
def test_arch_model_factory(mean, dist, vol_args):
    kwargs = {"mean": mean, "dist": dist, **vol_args}
    if mean in ("AR", "HAR"):
        kwargs["lags"] = [1, 3] if mean == "HAR" else 2
    if mean == "LS":
        kwargs["x"] = make_x(SMALL.index)
        x_initial, x_append = kwargs["x"].iloc[:500], kwargs["x"].iloc[500:]
        mod = arch_model(SMALL_initial, **{**kwargs, "x": x_initial})
        mod.append(SMALL_append, x=x_append)
    else:
        mod = arch_model(SMALL_initial, **kwargs)
        mod.append(SMALL_append)
    direct = arch_model(SMALL, **kwargs)
    assert_models_equal(mod, direct)
    assert_results_equal(fit(mod), fit(direct))


@pytest.mark.parametrize("form", ["vol", "var", "log", 1.5])
@pytest.mark.parametrize("vol", [GARCH(), EGARCH(), FIGARCH(), HARCH(lags=[1, 5])])
def test_arch_in_mean(vol, form):
    mod = ARCHInMean(SMALL_initial, lags=1, volatility=vol, form=form)
    mod.append(SMALL_append)
    direct = ARCHInMean(SMALL, lags=1, volatility=vol, form=form)
    assert_models_equal(mod, direct)
    assert_results_equal(fit(mod), fit(direct))


def test_hold_back():
    mod = HARXWrapper(SMALL_initial, volatility=GARCH(), hold_back=20)
    assert mod._fit_indices == [0, 500]
    mod.append(SMALL_append)
    assert mod._fit_indices == [20, 700]
    assert mod._fit_y.shape[0] == 680
    assert mod._fit_regressors.shape[0] == 680
    direct = HARXWrapper(SMALL, volatility=GARCH(), hold_back=20)
    res_direct = fit(direct)
    assert_results_equal(fit(mod), res_direct)
    assert mod._fit_indices == direct._fit_indices


def test_append_short_sample():
    # Fewer observations than needed after the held back observations
    mod = ARX(SMALL.iloc[:10], lags=2, hold_back=20, volatility=GARCH())
    assert mod._fit_indices == [0, 10]
    mod.append(SMALL.iloc[10:15])
    assert mod._fit_indices == [0, 15]
    mod.append(SMALL.iloc[15:20])
    assert mod._fit_indices == [0, 20]
    with pytest.raises(ValueError, match="empty array"):
        mod.fit(disp="off")
    mod.append(SMALL.iloc[20:25])
    assert mod._fit_indices == [20, 25]
    mod.append(SMALL.iloc[25:])
    direct = ARX(SMALL, lags=2, hold_back=20, volatility=GARCH())
    assert_models_equal(mod, direct)
    assert_results_equal(fit(mod), fit(direct))
    assert mod._fit_indices == [20, 700]


def test_append_resets_sample():
    mod = ARXWrapper(SMALL_initial, volatility=GARCH())
    fit(mod, first_obs=50, last_obs=400)
    assert mod._fit_indices == [52, 400]
    mod.append(SMALL_append)
    assert mod._fit_indices == [2, 700]
    assert mod._fit_y.shape[0] == 698
    assert mod._fit_regressors.shape[0] == 698
    assert mod.volatility.start == 2
    assert mod.volatility.stop == 700
    res = fit(mod)
    assert res.nobs == 698
    direct = ARXWrapper(SMALL, volatility=GARCH())
    assert_results_equal(res, fit(direct))


def test_append_then_fit_subsample():
    mod = ARXWrapper(SMALL_initial, volatility=GARCH())
    mod.append(SMALL_append)
    res = fit(mod, first_obs=SMALL.index[100], last_obs=SMALL.index[650])
    direct = ARXWrapper(SMALL, volatility=GARCH())
    res_direct = fit(direct, first_obs=SMALL.index[100], last_obs=SMALL.index[650])
    assert_results_equal(res, res_direct)


def test_cached_values_reset():
    mod = ARXWrapper(SMALL_initial, volatility=GARCH())
    res = fit(mod)
    assert mod._backcast is not None
    assert mod._var_bounds is not None
    mod.append(SMALL_append)
    assert mod._backcast is None
    assert mod._var_bounds is None
    direct = ARXWrapper(SMALL, volatility=GARCH())
    fit(direct)
    assert_allclose(
        mod.compute_param_cov(res.params), direct.compute_param_cov(res.params)
    )
    assert mod._backcast is not None


def test_backcast_short_initial_sample():
    # The backcast uses the first 75 observations, and so changes if initial
    # sample is shorter
    mod = ConstantMean(SMALL.iloc[:40], volatility=GARCH())
    res = fit(mod)
    backcast = mod._backcast
    mod.append(SMALL.iloc[40:])
    res_app = fit(mod)
    direct = ConstantMean(SMALL, volatility=GARCH())
    res_direct = fit(direct)
    assert not np.allclose(backcast, mod._backcast)
    assert_allclose(mod._backcast, direct._backcast)
    assert_results_equal(res_app, res_direct)
    assert res.nobs == 40


@pytest.mark.parametrize("mean", MEAN_MODELS)
@pytest.mark.parametrize(
    "vol", [GARCH(), EGARCH(), FIGARCH(), HARCH(lags=[1, 5]), APARCH()]
)
def test_fix_after_append(mean, vol):
    mod = mean(SMALL_initial, volatility=vol)
    res = fit(mod)
    mod.append(SMALL_append)
    fixed = mod.fix(res.params)
    direct = mean(SMALL, volatility=vol)
    fixed_direct = direct.fix(res.params)
    assert_allclose(fixed.loglikelihood, fixed_direct.loglikelihood)
    assert_allclose(fixed.resid, fixed_direct.resid, equal_nan=True)
    assert_allclose(
        fixed.conditional_volatility,
        fixed_direct.conditional_volatility,
        equal_nan=True,
    )
    assert fixed.resid.shape[0] == SMALL.shape[0]
    assert fixed.resid.index.equals(SMALL.index)


@pytest.mark.parametrize("mean", MEAN_MODELS)
@pytest.mark.parametrize(
    ("vol", "horizon"),
    [(GARCH(), 4), (EGARCH(), 1), (FIGARCH(), 4), (ConstantVariance(), 4)],
    ids=["garch", "egarch", "figarch", "constant"],
)
@pytest.mark.parametrize("fit_first", [True, False])
def test_forecast_after_append(mean, vol, horizon, fit_first):
    mod = mean(SMALL_initial, volatility=vol)
    res = fit(mod)
    mod.append(SMALL_append)
    fcast = mod.forecast(res.params, horizon=horizon, reindex=False)
    direct = mean(SMALL, volatility=vol)
    if fit_first:
        fit(direct)
    fcast_direct = direct.fix(res.params).forecast(horizon=horizon, reindex=False)
    pd.testing.assert_frame_equal(fcast.mean, fcast_direct.mean)
    pd.testing.assert_frame_equal(fcast.variance, fcast_direct.variance)
    pd.testing.assert_frame_equal(
        fcast.residual_variance, fcast_direct.residual_variance
    )
    # The default is to forecast from the last observation
    assert fcast.mean.shape[0] == 1
    assert fcast.mean.index[-1] == SMALL.index[-1]


@pytest.mark.parametrize("mean", MEAN_MODELS)
def test_forecast_after_append_never_fit(mean):
    params = fit(mean(SMALL_initial, volatility=GARCH())).params
    mod = mean(SMALL_initial, volatility=GARCH())
    mod.append(SMALL_append)
    fcast = mod.forecast(params, horizon=2, reindex=False)
    direct = mean(SMALL, volatility=GARCH())
    fcast_direct = direct.fix(params).forecast(horizon=2, reindex=False)
    pd.testing.assert_frame_equal(fcast.mean, fcast_direct.mean)
    pd.testing.assert_frame_equal(fcast.variance, fcast_direct.variance)


@pytest.mark.parametrize("vol", [GARCH(), EGARCH()])
@pytest.mark.parametrize("method", ["simulation", "bootstrap"])
def test_forecast_simulation_after_append(vol, method):
    mod = ARXWrapper(SMALL_initial, volatility=vol)
    res = fit(mod)
    mod.append(SMALL_append)
    kwargs = {"horizon": 3, "method": method, "simulations": 50, "reindex": False}

    def rng_kwargs():
        rs = np.random.RandomState(1)
        return {"random_state": rs, "rng": rs.standard_normal}

    fcast = mod.forecast(res.params, **rng_kwargs(), **kwargs)
    direct = ARXWrapper(SMALL, volatility=vol)
    fcast_direct = direct.fix(res.params).forecast(**rng_kwargs(), **kwargs)
    assert_allclose(fcast.simulations.values, fcast_direct.simulations.values)
    assert_allclose(fcast.variance, fcast_direct.variance)


def test_forecast_after_append_scalar_updates():
    # Online updating: observe one value at a time and forecast
    res = fit(ARXWrapper(SMALL_initial.to_numpy(), volatility=GARCH()))
    mod = ARXWrapper(SMALL_initial.to_numpy(), volatility=GARCH())
    for i in range(500, 510):
        mod.append(SMALL.iloc[i])
        fcast = mod.forecast(res.params, horizon=1, reindex=False)
        direct = ARXWrapper(SMALL.to_numpy()[: i + 1], volatility=GARCH())
        fcast_direct = direct.fix(res.params).forecast(horizon=1, reindex=False)
        assert fcast.mean.shape[0] == 1
        assert_allclose(fcast.mean.iloc[-1], fcast_direct.mean.iloc[-1])
        assert_allclose(fcast.variance.iloc[-1], fcast_direct.variance.iloc[-1])


@pytest.mark.parametrize("mean", [ARXWrapper, HARXWrapper, LSWrapper])
def test_forecast_after_append_exog(mean):
    x = make_x(SMALL.index)
    mod = mean(SMALL_initial, x=x.iloc[:500], volatility=GARCH())
    res = fit(mod)
    mod.append(SMALL_append, x=x.iloc[500:])
    expected = {"a": np.array([0.1, 0.2, 0.3]), "b": np.array([-0.1, 0.0, 0.1])}
    fcast = mod.forecast(res.params, horizon=3, x=expected, reindex=False)
    direct = mean(SMALL, x=x, volatility=GARCH())
    fcast_direct = direct.fix(res.params).forecast(horizon=3, x=expected, reindex=False)
    pd.testing.assert_frame_equal(fcast.mean, fcast_direct.mean)
    pd.testing.assert_frame_equal(fcast.variance, fcast_direct.variance)


def test_forecast_with_start_after_append():
    mod = ARXWrapper(SMALL_initial, volatility=GARCH())
    res = fit(mod)
    mod.append(SMALL_append)
    start = SMALL.index[600]
    fcast = mod.forecast(res.params, horizon=2, start=start, reindex=False)
    assert fcast.mean.index[0] == start
    assert fcast.mean.shape[0] == 700 - 600
    direct = ARXWrapper(SMALL, volatility=GARCH())
    fcast_direct = direct.fix(res.params).forecast(
        horizon=2, start=start, reindex=False
    )
    pd.testing.assert_frame_equal(fcast.mean, fcast_direct.mean)


def test_previous_results_unchanged():
    mod = ARXWrapper(SMALL_initial, volatility=GARCH())
    res = fit(mod)
    fixed = mod.fix(res.params)
    resid = res.resid.copy()
    vol = res.conditional_volatility.copy()
    fcast = res.forecast(horizon=2, reindex=False)
    fixed_fcast = fixed.forecast(horizon=2, reindex=False)
    nobs = res.nobs
    model_y = res.model._y.copy()
    mod.append(SMALL_append)
    assert res.nobs == nobs
    assert res.model._y.shape[0] == SMALL_initial.shape[0]
    assert_array_equal(res.model._y, model_y)
    assert fixed.model._y.shape[0] == SMALL_initial.shape[0]
    pd.testing.assert_series_equal(res.resid, resid)
    pd.testing.assert_series_equal(res.conditional_volatility, vol)
    pd.testing.assert_frame_equal(
        res.forecast(horizon=2, reindex=False).mean, fcast.mean
    )
    pd.testing.assert_frame_equal(
        fixed.forecast(horizon=2, reindex=False).mean, fixed_fcast.mean
    )
    assert "Adj Close" in str(res.summary())
    # And the appended model does not modify previous results when fit
    res_new = fit(mod)
    assert res_new.nobs == 698
    assert res.nobs == nobs
    assert res.model is not mod


@pytest.mark.parametrize("scale", [1e-3, 1e3])
@pytest.mark.parametrize("mean", MEAN_MODELS)
def test_append_rescale(mean, scale):
    y = SMALL * scale
    y_initial, y_append = y.iloc[:500], y.iloc[500:]
    mod = mean(y_initial, volatility=GARCH(), rescale=True)
    res = fit(mod)
    assert mod.scale != 1.0
    model_scale = mod.scale
    mod.append(y_append)
    assert mod.scale == model_scale
    assert_allclose(mod._y, model_scale * y.to_numpy())
    assert_allclose(mod._fit_y, model_scale * y.to_numpy()[mod._fit_indices[0] :])
    # Original data is not rescaled
    assert_allclose(np.asarray(mod.y), y.to_numpy())

    direct = mean(y, volatility=GARCH(), rescale=True)
    res_direct = fit(direct)
    assert direct.scale == model_scale
    assert_allclose(mod._y, direct._y)
    assert_allclose(mod.regressors, direct.regressors)
    assert_results_equal(fit(mod), res_direct)
    assert mod.scale == model_scale

    # Fixed parameters estimated using the smaller sample
    fixed = mod.fix(res.params)
    fixed_direct = direct.fix(res.params)
    assert_allclose(fixed.loglikelihood, fixed_direct.loglikelihood)
    assert_allclose(
        fixed.conditional_volatility,
        fixed_direct.conditional_volatility,
        equal_nan=True,
    )
    fcast = mod.forecast(res.params, horizon=2, reindex=False)
    fcast_direct = direct.fix(res.params).forecast(horizon=2, reindex=False)
    pd.testing.assert_frame_equal(fcast.mean, fcast_direct.mean)
    pd.testing.assert_frame_equal(fcast.variance, fcast_direct.variance)


@pytest.mark.parametrize("kind", ["series", "array", "list"])
def test_append_rescale_scalar(kind):
    y = SMALL * 1e-3
    mod = ARXWrapper(as_type(y.iloc[:500], kind), volatility=GARCH(), rescale=True)
    fit(mod)
    scale = mod.scale
    assert scale != 1.0
    if kind == "series":
        mod.append(y.iloc[500:510])
    else:
        for val in y.iloc[500:510]:
            mod.append(float(val))
    assert_allclose(mod._y, scale * y.to_numpy()[:510])
    assert_allclose(mod._y[-1], scale * y.iloc[509])


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_append_before_rescale(scale):
    # Rescaling only happens when fit is called, and applies to the full sample
    y = SMALL * scale
    mod = ARXWrapper(y.iloc[:500], volatility=GARCH(), rescale=True)
    mod.append(y.iloc[500:])
    assert mod.scale == 1.0
    assert_allclose(mod._y, y.to_numpy())
    direct = ARXWrapper(y, volatility=GARCH(), rescale=True)
    assert_results_equal(fit(mod), fit(direct))
    assert mod.scale == direct.scale != 1.0


def test_append_no_rescale_scale_warning():
    y = SMALL * 1e-3
    mod = ARXWrapper(y.iloc[:500], volatility=GARCH())
    with pytest.warns(DataScaleWarning):
        mod.fit(disp="off", show_warning=False)
    assert mod.scale == 1.0
    mod.append(y.iloc[500:])
    assert mod.scale == 1.0
    assert_allclose(mod._y, y.to_numpy())
    mod = ARXWrapper(y.iloc[:500], volatility=GARCH(), rescale=False)
    fit(mod)
    mod.append(y.iloc[500:])
    assert_allclose(mod._y, y.to_numpy())


def test_summary_and_results_after_append():
    mod = ARXWrapper(SMALL_initial, volatility=GARCH())
    mod.append(SMALL_append)
    res = fit(mod)
    summ = str(res.summary())
    assert "No. Observations:" in summ
    assert str(SMALL.shape[0] - 2) in summ
    assert res.resid.index.equals(SMALL.index)
    assert res.conditional_volatility.index.equals(SMALL.index)
    assert res.resid.isna().sum() == 2
    assert res.nobs == SMALL.shape[0] - 2


def test_append_shares_nothing_with_old_results():
    mod = ARXWrapper(SMALL_initial, volatility=GARCH())
    res = fit(mod)
    old_y = res.model.y.copy()
    mod.append(SMALL_append)
    pd.testing.assert_series_equal(res.model.y, old_y)
    assert res.model.y is not mod.y


class TestAppendSameType:
    def test_series(self):
        s = pd.Series(np.arange(5.0), name="x", index=pd.RangeIndex(5))
        new = pd.Series(np.arange(5.0, 8.0), index=pd.RangeIndex(5, 8))
        out = append_same_type(s, new)
        assert isinstance(out, pd.Series)
        assert out.name == "x"
        assert_array_equal(out.to_numpy(), np.arange(8.0))
        assert s.shape[0] == 5
        assert new.name is None
        # Same name
        out = append_same_type(s, new.rename("x"))
        assert out.name == "x"

    def test_series_name_none(self):
        s = pd.Series(np.arange(5.0), index=pd.RangeIndex(5))
        out = append_same_type(s, pd.Series([9.0], index=[10], name="a"))
        assert out.name is None

    def test_frame(self):
        df = pd.DataFrame(np.ones((4, 2)), columns=["a", "b"])
        new = pd.DataFrame(np.zeros((2, 2)), columns=["a", "b"], index=[4, 5])
        out = append_same_type(df, new)
        assert out.shape == (6, 2)
        assert list(out.columns) == ["a", "b"]
        assert df.shape == (4, 2)
        with pytest.raises(ValueError, match="columns of the appended"):
            append_same_type(df, new.rename(columns={"b": "c"}))
        with pytest.raises(ValueError, match="columns of the appended"):
            append_same_type(df, new[["a"]])
        with pytest.raises(ValueError, match="columns of the appended"):
            append_same_type(df, new[["b", "a"]])

    def test_ndarray_1d(self):
        arr = np.arange(4.0)
        out = append_same_type(arr, np.arange(4.0, 6.0))
        assert isinstance(out, np.ndarray)
        assert_array_equal(out, np.arange(6.0))
        assert_array_equal(arr, np.arange(4.0))
        assert_array_equal(append_same_type(arr, 4.0), np.arange(5.0))
        assert_array_equal(append_same_type(arr, np.float32(4.0)), np.arange(5.0))
        assert_array_equal(append_same_type(arr, 4), np.arange(5.0))
        assert_array_equal(append_same_type(arr, np.array(4.0)), np.arange(5.0))
        # Single column 2-d is treated as 1-d
        assert_array_equal(
            append_same_type(arr, np.ones((3, 1))), [0, 1, 2, 3, 1, 1, 1]
        )
        assert_array_equal(
            append_same_type(arr, np.ones((1, 3))), [0, 1, 2, 3, 1, 1, 1]
        )
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            append_same_type(arr, np.ones((2, 2)))

    def test_ndarray_0d_original(self):
        out = append_same_type(np.array(1.0), 2.0)
        assert_array_equal(out, [1.0, 2.0])

    def test_ndarray_2d(self):
        arr = np.arange(6.0).reshape((3, 2))
        out = append_same_type(arr, np.ones((2, 2)))
        assert out.shape == (5, 2)
        # 1-d is a single observation
        out = append_same_type(arr, np.array([8.0, 9.0]))
        assert out.shape == (4, 2)
        assert_array_equal(out[-1], [8.0, 9.0])
        with pytest.raises(ValueError, match="same number of columns"):
            append_same_type(arr, np.ones((2, 3)))
        with pytest.raises(ValueError, match="same number of columns"):
            append_same_type(arr, np.ones(3))
        with pytest.raises(ValueError, match="same number of columns"):
            append_same_type(arr, 1.0)
        with pytest.raises(ValueError, match="same number of columns"):
            append_same_type(arr, np.ones((2, 2, 1)))
        with pytest.raises(ValueError, match="original data must be 1-d or 2-d"):
            append_same_type(np.ones((2, 2, 2)), np.ones((2, 2, 2)))

    def test_ndarray_single_column(self):
        arr = np.arange(3.0)[:, None]
        assert append_same_type(arr, np.ones(2)).shape == (5, 1)
        assert append_same_type(arr, 2.0).shape == (4, 1)
        assert append_same_type(arr, np.ones((2, 1))).shape == (5, 1)

    def test_list(self):
        orig = [1.0, 2.0]
        out = append_same_type(orig, [3.0, 4.0])
        assert out == [1.0, 2.0, 3.0, 4.0]
        assert orig == [1.0, 2.0]
        assert append_same_type(orig, 3.0) == [1.0, 2.0, 3.0]
        assert append_same_type(orig, np.float64(3.0)) == [1.0, 2.0, 3.0]
        with pytest.raises(TypeError, match="Input data must be the same"):
            append_same_type(orig, np.array([3.0]))
        with pytest.raises(TypeError, match="Input data must be the same"):
            append_same_type(orig, (3.0,))
        nested = [[1.0, 2.0], [3.0, 4.0]]
        out = append_same_type(nested, [[5.0, 6.0]])
        assert out == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        out = append_same_type(nested, [5.0, 6.0])
        assert out == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        assert nested == [[1.0, 2.0], [3.0, 4.0]]
        with pytest.raises(ValueError, match="same number of columns"):
            append_same_type(nested, [[5.0, 6.0, 7.0]])
        with pytest.raises(ValueError, match="original data must be 1-d or 2-d"):
            append_same_type([[[1.0]]], [[[1.0]]])

    def test_tuple(self):
        orig = (1.0, 2.0)
        out = append_same_type(orig, (3.0, 4.0))
        assert out == (1.0, 2.0, 3.0, 4.0)
        assert isinstance(out, tuple)
        assert append_same_type(orig, 3.0) == (1.0, 2.0, 3.0)
        nested = ((1.0, 2.0), (3.0, 4.0))
        out = append_same_type(nested, ((5.0, 6.0),))
        assert out == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        assert isinstance(out[-1], tuple)
        with pytest.raises(TypeError, match="Input data must be the same"):
            append_same_type(orig, [3.0])

    @pytest.mark.parametrize(
        "original",
        [1.0, "abc", None, {"a": 1}, range(3)],
        ids=["float", "str", "none", "dict", "range"],
    )
    def test_bad_original(self, original):
        with pytest.raises(TypeError, match="The original data must be"):
            append_same_type(original, 1.0)

    @pytest.mark.parametrize(
        "new",
        [True, np.bool_(True), "a", None, {"a": 1}],
        ids=["bool", "np-bool", "str", "none", "dict"],
    )
    def test_bad_new(self, new):
        for original in (np.arange(3.0), [1.0, 2.0], (1.0, 2.0), pd.Series([1.0])):
            with pytest.raises(TypeError, match="Input data must be the same"):
                append_same_type(original, new)

    @pytest.mark.parametrize(
        "scalar", [1.0, 1, np.float64(1.0)], ids=["float", "int", "np-float"]
    )
    def test_pandas_scalar(self, scalar):
        with pytest.raises(TypeError, match="Input data must be the same"):
            append_same_type(pd.Series([1.0]), scalar)
        with pytest.raises(TypeError, match="Input data must be the same"):
            append_same_type(pd.DataFrame([1.0]), scalar)

    def test_empty(self):
        for original, new in (
            (np.arange(3.0), np.empty(0)),
            ([1.0], []),
            ((1.0,), ()),
            (pd.Series([1.0]), pd.Series([], dtype=float)),
            (pd.DataFrame({"a": [1.0]}), pd.DataFrame({"a": []}, dtype=float)),
        ):
            with pytest.raises(ValueError, match="at least one observation"):
                append_same_type(original, new)

    def test_non_numeric_list(self):
        with pytest.raises(ValueError, match="could not convert"):
            append_same_type([1.0], ["a"])
