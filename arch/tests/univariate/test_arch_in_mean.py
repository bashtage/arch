import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.special import binom

from arch.data import sp500
from arch.univariate import ARCHInMean, Normal
import arch.univariate.recursions_python as recpy
from arch.univariate.recursions_python import ARCHInMeanRecursion
from arch.univariate.volatility import (
    ARCH,
    EGARCH,
    FIAPARCH,
    FIGARCH,
    GARCH,
    HARCH,
    EWMAVariance,
    MIDASHyperbolic,
    RiskMetrics2006,
)

try:
    import arch.univariate.recursions as rec_cython
except ImportError:
    rec_cython = None

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
    with pytest.raises(ValueError, match=r"initial_value has the wrong shape"):
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


def _hand_figarch_lambda(phi, d, beta, truncation):
    """
    ARCH(infinity) weights of lambda(L) = 1 - (1 - phi L) (1 - L)**d / (1 - beta L)
    for lags 1, ..., truncation, from the binomial series and polynomial division
    """
    lags = np.arange(truncation + 1)
    num = np.convolve(binom(d, lags) * (-1.0) ** lags, [1.0, -phi])[: truncation + 1]
    quot = np.empty(truncation + 1)
    quot[0] = num[0]
    for k in range(1, truncation + 1):
        quot[k] = num[k] + beta * quot[k - 1]
    return -quot[1:]


def _hand_aim_loglikelihood(y, mu, phi, kappa, form, power, vol, params, backcast):
    """
    Log-likelihood of an AR(1) in-mean model with normal errors and a GARCH(1,1,q)
    or FIGARCH(1,d,1) process whose recursion runs in sigma ** power, with the
    first observation used as a lag
    """
    obs = y[1:]
    lag = y[:-1]
    nobs = obs.shape[0]
    f = np.empty(nobs)
    sigma2 = np.empty(nobs)
    e = np.empty(nobs)
    if vol == "garch":
        omega, alpha, gamma = params[:3]
        betas = params[3:]
    else:
        omega, fi_phi, d, beta, truncation = params
        lam = _hand_figarch_lambda(fi_phi, d, beta, truncation)
    llf = 0.0
    for t in range(nobs):
        if vol == "garch":
            # t - 1 is before the sample only for t = 0
            prev_abs = backcast if t == 0 else abs(e[t - 1]) ** power
            prev_neg = 0.5 * backcast if t == 0 else prev_abs * (e[t - 1] < 0)
            f[t] = omega + alpha * prev_abs + gamma * prev_neg
            for j, b in enumerate(betas):
                f[t] += b * (backcast if t - 1 - j < 0 else f[t - 1 - j])
        else:
            f[t] = omega / (1.0 - beta)
            for i in range(1, truncation + 1):
                if t - i >= 0:
                    f[t] += lam[i - 1] * abs(e[t - i]) ** power
                else:
                    f[t] += lam[i - 1] * backcast
        sigma2[t] = f[t] ** (2.0 / power)
        if form == "var":
            in_mean = sigma2[t]
        elif form == "vol":
            in_mean = sigma2[t] ** 0.5
        else:
            in_mean = np.log(sigma2[t])
        e[t] = obs[t] - mu - phi * lag[t] - kappa * in_mean
        llf += -0.5 * (np.log(2 * np.pi) + np.log(sigma2[t]) + e[t] ** 2 / sigma2[t])
    return llf, sigma2


def _hand_backcast(y, power):
    # Residuals from the least squares estimates are used to start the model
    x = np.column_stack([np.ones(y.shape[0] - 1), y[:-1]])
    b = np.linalg.lstsq(x, y[1:], rcond=None)[0]
    r = y[1:] - x @ b
    tau = min(75, r.shape[0])
    w = 0.94 ** np.arange(tau)
    w = w / w.sum()
    return float(np.sum(w * np.abs(r[:tau]) ** power))


def _compare_with_model(model, params, backcast, llf, sigma2):
    """Compare the log-likelihood and conditional variance used in estimation"""
    res = model.fix(params)
    resids = model.resids(model.starting_values())
    var_bounds = model.volatility.variance_bounds(resids)
    assert_allclose(model.volatility.backcast(resids), backcast, rtol=1e-10)
    model_sigma2 = np.zeros(resids.shape[0])
    model_llf = -model._loglikelihood(params, model_sigma2, backcast, var_bounds)
    assert_allclose(model_llf, llf, rtol=1e-10)
    assert_allclose(model_sigma2, sigma2, rtol=1e-10)
    assert_allclose(res.loglikelihood, llf, rtol=1e-10)


UPDATER_MODULES = [
    pytest.param(recpy, id="python"),
    pytest.param(
        rec_cython,
        id="cython",
        marks=pytest.mark.skipif(rec_cython is None, reason="No Cython extension"),
    ),
]


@pytest.fixture(scope="module")
def power_data():
    rg = np.random.default_rng(4)
    return pd.Series(rg.standard_t(5, 300) * 1.1 + 0.1)


@pytest.mark.filterwarnings("ignore::arch.compat.numba.PerformanceWarning")
@pytest.mark.parametrize("module", UPDATER_MODULES)
@pytest.mark.parametrize("form", ["var", "vol", "log"])
@pytest.mark.parametrize("power", [1.0, 1.5, 2.0], ids=["p1", "p1.5", "p2"])
@pytest.mark.parametrize(
    ("q", "vol_params"),
    [(1, [0.1, 0.05, 0.1, 0.8]), (2, [0.1, 0.05, 0.1, 0.5, 0.3])],
    ids=["q1", "q2"],
)
def test_garch_power_loglikelihood(power_data, module, form, power, q, vol_params):
    # The GARCH recursion runs in sigma ** power, but the in-mean term, the
    # normal log-likelihood and the conditional variance use sigma ** 2
    kappa = 0.05
    vol = GARCH(1, 1, q, power=power)
    model = ARCHInMean(power_data, lags=1, form=form, volatility=vol)
    updater = module.GARCHUpdater(1, 1, q, power)
    model._volatility_updater = updater
    model._recursion = module.ARCHInMeanRecursion(updater)
    params = np.r_[0.1, 0.2, kappa, vol_params]
    y = power_data.to_numpy()
    backcast = _hand_backcast(y, power)
    llf, sigma2 = _hand_aim_loglikelihood(
        y, 0.1, 0.2, kappa, form, power, "garch", vol_params, backcast
    )
    _compare_with_model(model, params, backcast, llf, sigma2)


@pytest.mark.filterwarnings("ignore::arch.compat.numba.PerformanceWarning")
@pytest.mark.parametrize("module", UPDATER_MODULES)
@pytest.mark.parametrize("form", ["var", "vol", "log"])
@pytest.mark.parametrize("power", [1.0, 1.5, 2.0], ids=["p1", "p1.5", "p2"])
def test_figarch_power_loglikelihood(power_data, module, form, power):
    kappa = 0.05
    truncation = 40
    vol_params = [0.1, 0.2, 0.4, 0.3]
    vol = FIGARCH(1, 1, power=power, truncation=truncation)
    model = ARCHInMean(power_data, lags=1, form=form, volatility=vol)
    updater = module.FIGARCHUpdater(1, 1, power, truncation)
    model._volatility_updater = updater
    model._recursion = module.ARCHInMeanRecursion(updater)
    params = np.r_[0.1, 0.2, kappa, vol_params]
    y = power_data.to_numpy()
    backcast = _hand_backcast(y, power)
    llf, sigma2 = _hand_aim_loglikelihood(
        y,
        0.1,
        0.2,
        kappa,
        form,
        power,
        "figarch",
        vol_params + [truncation],
        backcast,
    )
    _compare_with_model(model, params, backcast, llf, sigma2)
