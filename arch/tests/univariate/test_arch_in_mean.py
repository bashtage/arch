import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats

from arch.data import sp500
from arch.univariate import ARCHInMean, Normal
from arch.univariate.recursions_python import ARCHInMeanRecursion
from arch.univariate.volatility import (
    ARCH,
    EGARCH,
    FIAPARCH,
    FIGARCH,
    GARCH,
    HARCH,
    NGARCH,
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

# FIAPARCH intentionally left out
SUPPORTED = [
    HARCH,
    ARCH,
    GARCH,
    EWMAVariance,
    MIDASHyperbolic,
    FIGARCH,
    RiskMetrics2006,
    EGARCH,
    NGARCH,
]


def test_exceptions():
    with pytest.raises(TypeError, match=r"form must be 'log', 'vol', 'var' or a "):
        ARCHInMean(SP500, form=0 + 3j, volatility=GARCH())
    with pytest.raises(ValueError, match=r"form must be 'log', 'vol', 'var' or a "):
        ARCHInMean(SP500, form=0, volatility=GARCH())
    with pytest.raises(ValueError, match=r"form must be 'log', 'vol', 'var' or a "):
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


@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("lags", [None, 1])
@pytest.mark.parametrize("x_type", ["frame", "fortran"])
def test_exog_not_c_contiguous(constant, lags, x_type):
    # DataFrames, and some arrays, are F-ordered, and the compiled
    # recursions require C-ordered regressors
    x_f = X if x_type == "frame" else np.asfortranarray(X.to_numpy())
    x_c = np.ascontiguousarray(X.to_numpy())
    mod = ARCHInMean(SP500, x=x_f, lags=lags, constant=constant, volatility=GARCH())
    assert mod.regressors.flags.c_contiguous
    res = mod.fit(disp="off")
    mod_c = ARCHInMean(SP500, x=x_c, lags=lags, constant=constant, volatility=GARCH())
    res_c = mod_c.fit(disp="off")
    assert_allclose(res.params, res_c.params)


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
    with pytest.raises(ValueError, match=r"initial_value must be a 1-d array"):
        gim.simulate(res.params, 1000, initial_value=np.array([0.0, 0.0]))


@pytest.mark.slow
def test_fiaparch():
    # Special case for FIAPARCH which is very slow
    aim = ARCHInMean(SP500, volatility=FIAPARCH(), form="log")
    assert isinstance(aim, ARCHInMean)
    sv = np.array(
        [0.051519, 0.019036, 0.102975, 0.188827, 0.528414, 0.600234, 0.999700, 0.894997]
    )
    res = aim.fit(disp=True, starting_values=sv)
    print(res.params)


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


IN_MEAN_FORMS = ["log", "vol", "var", 1.5]
IN_MEAN_FORM_IDS = ["log", "vol", "var", "power-1.5"]
IN_MEAN_SPECS = [
    {"lags": None, "x": None},
    {"lags": [1], "x": None},
    {"lags": [1, 2], "x": X},
]
IN_MEAN_SPEC_IDS = ["constant", "ar1", "ar12-x"]


def _in_mean_function(form, sigma2):
    """The form of the conditional variance in the mean, from the documentation"""
    if form == "log":
        return np.log(sigma2)
    elif form == "vol":
        return np.sqrt(sigma2)
    elif form == "var":
        return sigma2
    return sigma2 ** (form / 2.0)


def _in_mean_reference(y, lags, x, form, params, backcast):
    """
    Direct implementation of a GARCH(1,1)-in-mean model

    y_t = mu + sum_i phi_i y_{t-i} + gamma'x_t + kappa g(sigma2_t) + e_t
    sigma2_t = omega + alpha e_{t-1}^2 + beta sigma2_{t-1}

    sigma2_0 = omega + (alpha + beta) * backcast
    params are ordered (mean parameters, kappa, omega, alpha, beta)
    """
    y = np.asarray(y, dtype=float)
    max_lag = 0 if lags is None else max(lags)
    nobs = y.shape[0] - max_lag
    columns = [np.ones(nobs)]
    columns.extend(y[max_lag - lag : y.shape[0] - lag] for lag in lags or [])
    if x is not None:
        columns.extend(np.asarray(x, dtype=float)[max_lag:].T)
    regressors = np.column_stack(columns)
    k = regressors.shape[1]
    mean_params, kappa = params[:k], params[k]
    omega, alpha, beta = params[k + 1 :]
    y_fit = y[max_lag:]

    eps = np.empty(nobs)
    sigma2 = np.empty(nobs)
    for t in range(nobs):
        if t == 0:
            sigma2[t] = omega + (alpha + beta) * backcast
        else:
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = (
            y_fit[t]
            - regressors[t] @ mean_params
            - kappa * _in_mean_function(form, sigma2[t])
        )
    return y_fit, eps, sigma2


def _assert_in_mean_result(res, mod, spec, form, params):
    max_lag = 0 if spec["lags"] is None else max(spec["lags"])
    y_fit, eps, sigma2 = _in_mean_reference(
        SP500.to_numpy(), spec["lags"], spec["x"], form, params, mod._backcast
    )
    resid = np.asarray(res.resid)
    vol = np.asarray(res.conditional_volatility)
    assert np.all(np.isnan(resid[:max_lag]))
    assert np.all(np.isnan(vol[:max_lag]))
    assert_allclose(resid[max_lag:], eps, rtol=1e-8, atol=1e-10)
    assert_allclose(vol[max_lag:], np.sqrt(sigma2), rtol=1e-8)
    assert_allclose(
        np.asarray(res.std_resid)[max_lag:], eps / np.sqrt(sigma2), rtol=1e-8
    )

    # Internal consistency: the log-likelihood of the reported residuals and
    # volatility, using an independent normal density, is the log-likelihood
    llf = stats.norm.logpdf(resid[max_lag:], scale=vol[max_lag:]).sum()
    assert_allclose(llf, res.loglikelihood, rtol=1e-8)
    return y_fit, eps


@pytest.mark.parametrize("spec", IN_MEAN_SPECS, ids=IN_MEAN_SPEC_IDS)
@pytest.mark.parametrize("form", IN_MEAN_FORMS, ids=IN_MEAN_FORM_IDS)
def test_in_mean_resid_and_volatility(form, spec):
    # The in-mean term was previously omitted from resid, conditional_volatility
    # and R-squared although it was correctly part of the log-likelihood
    mod = ARCHInMean(SP500, volatility=GARCH(), form=form, **spec)
    res = mod.fit(disp="off")
    assert abs(res.params["kappa"]) > 1e-3
    y_fit, eps = _assert_in_mean_result(res, mod, spec, form, res.params.to_numpy())

    tss = np.sum((y_fit - y_fit.mean()) ** 2)
    assert_allclose(res.rsquared, 1.0 - np.sum(eps**2) / tss, rtol=1e-8, atol=1e-10)
    assert_allclose(res.resid.mean(), eps.mean(), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("spec", IN_MEAN_SPECS, ids=IN_MEAN_SPEC_IDS)
@pytest.mark.parametrize("form", IN_MEAN_FORMS, ids=IN_MEAN_FORM_IDS)
def test_in_mean_fix_resid_and_volatility(form, spec):
    mod = ARCHInMean(SP500, volatility=GARCH(), form=form, **spec)
    k = 1 + (0 if spec["lags"] is None else len(spec["lags"]))
    k += 0 if spec["x"] is None else spec["x"].shape[1]
    params = np.r_[np.full(k, 0.02), 0.15, 0.05, 0.08, 0.9]
    res = mod.fix(params)
    _assert_in_mean_result(res, mod, spec, form, params)
