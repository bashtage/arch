import numpy as np
import pandas as pd
import pytest

from arch.data import sp500
from arch.univariate import ARCHInMean, Normal
from arch.univariate.recursions_python import ARCHInMeanRecursion
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
SP500 = SP500.iloc[SP500.shape[0] // 2 :]
RANDOMSTATE = np.random.RandomState(12349876)
X = pd.DataFrame(
    RANDOMSTATE.standard_normal((SP500.shape[0], 2)), columns=[0, 1], index=SP500.index
)


SUPPORTED = [
    HARCH,
    ARCH,
    GARCH,
    EWMAVariance,
    MIDASHyperbolic,
    FIGARCH,
    RiskMetrics2006,
    EGARCH,
]


def test_exceptions():
    with pytest.raises(TypeError, match=r"form must be a floating point "):
        ARCHInMean(SP500, form=0 + 3j, volatility=GARCH())
    with pytest.raises(
        ValueError, match=r"When using a floating point number for form"
    ):
        ARCHInMean(SP500, form=0, volatility=GARCH())
    with pytest.raises(
        ValueError, match=r"form must be a floating point number of one"
    ):
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

    with pytest.raises(
        NotImplementedError, match=r"forecasts are not implemented for \(G\)ARCH"
    ):
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
    normal = Normal(seed=np.random.RandomState(0))
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
    with pytest.raises(ValueError, match=r"initial_value has the wrong shape"):
        gim.simulate(res.params, 1000, initial_value=np.array([0.0, 0.0]))


@pytest.mark.slow
@pytest.mark.parametrize("good_vol", SUPPORTED)
def test_supported(good_vol):
    aim = ARCHInMean(SP500, volatility=good_vol(), form="log")
    assert isinstance(aim, ARCHInMean)
    res = aim.fit(disp=False)
    n = res.params.shape[0]
    assert res.param_cov.shape == (n, n)
    res2 = aim.fit(disp=False, starting_values=res.params)
    assert res2.params.shape == (n,)


def test_egarch_bad_params():
    aim = ARCHInMean(SP500, volatility=EGARCH(), form="log")
    res = aim.fit(disp=False)
    sv = res.params.copy()
    sv["omega"] = 4
    sv["alpha[1]"] = 0.75
    sv["beta[1]"] = 0.999998
    res2 = aim.fit(disp=False, starting_values=sv)
    n = res2.params.shape[0]
    assert res.param_cov.shape == (n, n)
    res3 = aim.fit(disp=False, starting_values=res.params)
    assert res3.params.shape == (n,)


@pytest.mark.parametrize("form", ["log", "vol", 1.5])
def test_simulate_arx(form):
    normal = Normal(seed=np.random.RandomState(0))
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


@pytest.mark.slow
@pytest.mark.parametrize("m", [22, 33])
@pytest.mark.parametrize("asym", [True, False])
def test_alt_parameterizations(asym, m):
    mod = ARCHInMean(SP500, volatility=MIDASHyperbolic(m=m, asym=asym))
    res = mod.fit(disp=False)
    assert res.params.shape[0] == 5 + asym
    res2 = mod.fit(disp=False)
    np.testing.assert_allclose(res.params, res2.params)


def test_not_updateable():
    class NonUpdateableGARCH(GARCH):
        _updatable = False

        def __init__(self):
            super().__init__()
            self._volatility_updater = None

    nug = NonUpdateableGARCH()
    with pytest.raises(
        NotImplementedError, match=r"Subclasses may optionally implement"
    ):
        _ = nug.volatility_updater
    with pytest.raises(ValueError, match=r"The volatility process"):
        _ = ARCHInMean(SP500, volatility=nug)


def test_wrong_process():

    with pytest.raises(TypeError, match=r"updater must be a VolatilityUpdater"):
        ARCHInMeanRecursion(updater=object())
