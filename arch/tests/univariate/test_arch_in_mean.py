import numpy as np
import pandas as pd
import pytest

from arch.data import sp500
from arch.univariate import ARCHInMean, Normal
from arch.univariate.volatility import (
    ARCH,
    EGARCH,
    FIGARCH,
    GARCH,
    HARCH,
    EWMAVariance,
    MIDASHyperbolic,
    RiskMetrics2006,
)

SP500 = 100 * sp500.load()["Adj Close"].pct_change().dropna()
X = pd.concat([SP500, SP500], axis=1, copy=True)
X.columns = [0, 1]
RANDOMSTATE = np.random.RandomState(12349876)
X.loc[:, :] = RANDOMSTATE.standard_normal(X.shape)

SUPPORTED = [HARCH, ARCH, GARCH, EWMAVariance, MIDASHyperbolic]
UNSUPPORTED = [RiskMetrics2006, FIGARCH, EGARCH]


def test_exceptions():
    with pytest.raises(TypeError):
        ARCHInMean(SP500, form=0 + 3j, volatility=GARCH())
    with pytest.raises(ValueError):
        ARCHInMean(SP500, form=0, volatility=GARCH())
    with pytest.raises(ValueError):
        ARCHInMean(SP500, form="unknown", volatility=GARCH())


@pytest.mark.parametrize("form_and_id", [("vol", 1), ("var", 2), ("log", 0), (1.5, 3)])
def test_formid(form_and_id):
    form, form_id = form_and_id
    mod = ARCHInMean(SP500, volatility=GARCH(), form=form)
    assert mod.form == form
    assert mod._form_id == form_id

    mod_str = str(mod)
    if isinstance(form, str):
        assert f"form: {form}" in mod_str
        assert "numeric" not in mod_str
    else:
        assert f"form: {form} (numeric)" in mod_str
    assert mod.num_params == 2


@pytest.mark.parametrize("form", ["vol", "var", "log", 1.5])
def test_smoke(form):
    mod = ARCHInMean(SP500, volatility=GARCH(), form=form)
    res = mod.fit(disp=False)
    assert "kappa" in res.params.index
    assert res.params.shape[0] == 5
    assert res.param_cov.shape == (5, 5)
    assert isinstance(res.param_cov, pd.DataFrame)

    with pytest.raises(NotImplementedError):
        res.forecast(reindex=True)


def test_example_smoke():
    rets = SP500
    gim = ARCHInMean(rets, lags=[1, 2], volatility=GARCH())
    res = gim.fit(disp=False)
    assert res.params.shape[0] == 7


def test_no_constant():
    gim = ARCHInMean(SP500, constant=False, volatility=GARCH())
    res = gim.fit(disp=False)
    assert res.params.shape[0] == 4


@pytest.mark.parametrize("x", [X[0], X])
def test_exog_smoke(x):
    gim = ARCHInMean(SP500, constant=False, volatility=GARCH(), x=x)
    res = gim.fit(disp="off")
    x_shape = 1 if isinstance(x, pd.Series) else x.shape[1]
    assert res.params.shape[0] == 4 + x_shape


def test_simulate():
    normal = Normal(random_state=np.random.RandomState(0))
    gim = ARCHInMean(SP500, volatility=GARCH(), distribution=normal)
    res = gim.fit(disp="off")
    sim = gim.simulate(res.params, 1000)
    assert sim.shape == (1000, 3)
    assert "data" in sim
    assert "volatility" in sim
    assert "errors" in sim
    mean = sim.data - sim.errors
    vol = mean - res.params.iloc[0]
    kappa = res.params.iloc[1]
    rescaled_vol = vol / kappa
    np.testing.assert_allclose(rescaled_vol, sim.volatility)
    with pytest.raises(ValueError, match="initial_value has the wrong shape"):
        gim.simulate(res.params, 1000, initial_value=np.array([0.0, 0.0]))


@pytest.mark.parametrize("bad_vol", UNSUPPORTED)
def test_unsupported(bad_vol):
    with pytest.raises(ValueError, match="The volatility process"):
        ARCHInMean(SP500, volatility=bad_vol())


@pytest.mark.parametrize("good_vol", [MIDASHyperbolic])
def test_supported(good_vol):
    aim = ARCHInMean(SP500, volatility=good_vol())
    assert isinstance(aim, ARCHInMean)
    res = aim.fit(disp=False)
    n = res.params.shape[0]
    assert res.param_cov.shape == (n, n)


@pytest.mark.parametrize("form", ["log", "vol", 1.5])
def test_simulate_arx(form):
    normal = Normal(random_state=np.random.RandomState(0))
    gim = ARCHInMean(
        SP500,
        constant=False,
        lags=2,
        volatility=GARCH(),
        distribution=normal,
        x=X,
        form=form,
    )
    res = gim.fit(disp="off")
    sim = gim.simulate(res.params, 1000, x=X.iloc[:1500], initial_value=0.0)
    assert sim.shape == (1000, 3)
    assert "data" in sim
    assert "volatility" in sim
    assert "errors" in sim
    gim.simulate(res.params, 1000, x=X.iloc[:1500], initial_value=np.zeros(2))
