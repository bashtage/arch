import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.special import binom

from arch.univariate import ARCHInMean
import arch.univariate.recursions_python as recpy
from arch.univariate.volatility import FIGARCH, GARCH

try:
    import arch.univariate.recursions as rec_cython
except ImportError:
    rec_cython = None


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
