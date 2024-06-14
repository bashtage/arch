import datetime as dt
from functools import partial
from itertools import product

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose
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
    ConstantMean,
    ConstantVariance,
    EWMAVariance,
    MIDASHyperbolic,
    RiskMetrics2006,
    ZeroMean,
    arch_model,
)

SP500 = 100 * sp500.load()["Adj Close"].pct_change().dropna()
N = SP500.shape[0]
SP500_initial = SP500.iloc[: N // 2]
SP500_append = SP500.iloc[N // 2 :]


class HARXWrapper(HARX):
    def __init__(self, y, x=None, volatility=None):
        super().__init__(y, lags=[1, 5], x=x, volatility=volatility)


class ARXWrapper(ARX):
    def __init__(self, y, x=None, volatility=None):
        super().__init__(y, lags=2, x=x, volatility=volatility)


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

MODEL_SPECS = list(product(MEAN_MODELS, VOLATILITIES))

IDS = [f"{mean.__name__}-{str(vol).split('(')[0]}" for mean, vol in MODEL_SPECS]


@pytest.fixture(params=MODEL_SPECS, ids=IDS)
def mean_volatility(request):
    mean, vol = request.param
    return mean, vol


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


def test_append_scalar_no_reestiamtion(mean_volatility):
    mean, vol = mean_volatility
    mod = mean(np.asarray(SP500_initial), volatility=vol)
    for val in np.asarray(SP500_append):
        mod.append(val)


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


def test_append_x_type_mismatch():
    pass


@pytest.mark.parametrize("mean", X_MEAN_MODELS)
def test_bad_append_model_with_exog(mean):
    mod = mean(SP500_initial, volatility=GARCH())
    x = pd.DataFrame(
        np.random.randn(SP500_append.shape[0], 2),
        columns=["a", "b"],
        index=SP500_append.index,
    )
    with pytest.raises(ValueError, match=""):
        mod.append(SP500_append, x=x)

    x_initial = pd.DataFrame(
        np.random.randn(SP500_initial.shape[0], 2),
        columns=["a", "b"],
        index=SP500_initial.index,
    )
    mod = mean(SP500_initial, x=x_initial, volatility=GARCH())
    with pytest.raises(ValueError, match=""):
        mod.append(SP500_append)


def test_bad_append_ls():
    pass
