from arch.compat.matplotlib import HAS_MATPLOTLIB

import warnings

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import optimize
from statsmodels.tsa.arima.model import ARIMA

from arch.univariate import (
    ARCH,
    EGARCH,
    GARCH,
    ConstantMean,
    ConstantVariance,
    EWMAVariance,
    FixedVariance,
    GeneralizedError,
    MAMean,
    Normal,
    SkewStudent,
    StudentsT,
)
from arch.univariate.base import ARCHModelForecast, ARCHModelResult
from arch.univariate.mean import _ma_starting_values, _make_invertible
from arch.utility.exceptions import DataScaleWarning

GARCH_PARAMS = np.array([0.05, 0.1, 0.85])
MA2_PARAMS = np.array([0.1, 0.6, 0.2])


def ma_resids(y, mu, theta):
    """The recursion for the innovations, written as loops"""
    e = np.empty(y.shape[0])
    for t in range(y.shape[0]):
        value = y[t] - mu
        for j in range(1, len(theta) + 1):
            if t - j >= 0:
                value -= theta[j - 1] * e[t - j]
        e[t] = value
    return e


def ma_process(e, mu, theta):
    """y_t = mu + e_t + sum_j theta_j e_{t-j} with e_t = 0 before the sample"""
    y = np.full(e.shape[0], float(mu))
    for t in range(e.shape[0]):
        y[t] += e[t]
        for j in range(1, len(theta) + 1):
            if t - j >= 0:
                y[t] += theta[j - 1] * e[t - j]
    return y


def simulate_ma(theta, nobs, mu=0.0, seed=0, scale=1.0):
    e = RandomState(seed).standard_normal(nobs + 100) * scale
    return ma_process(e, mu, theta)[100:]


def is_invertible(theta, margin=0.0):
    roots = np.roots(np.r_[np.asarray(theta, dtype=float)[::-1], 1.0])
    return bool(roots.shape[0] == 0 or np.all(np.abs(roots) > 1.0 + margin))


@pytest.fixture(scope="module", params=[False, True], ids=["numpy", "pandas"])
def garch_ma2(request):
    model = MAMean(
        lags=2, volatility=GARCH(), distribution=Normal(seed=RandomState(20261010))
    )
    params = np.concatenate((MA2_PARAMS, GARCH_PARAMS))
    y = model.simulate(params, 1500)["data"]
    if request.param:
        index = pd.date_range("2000-01-03", periods=y.shape[0], freq="B")
        return pd.Series(y.to_numpy(), index=index, name="ret")
    return y.to_numpy()


@pytest.fixture(scope="module")
def ma2_constant_variance():
    return simulate_ma([0.6, 0.2], 1500, mu=0.1, seed=1234)


# Construction and attributes
@pytest.mark.parametrize("lags", [0, -1, 1.5, True, False, "2", None, [1, 2], np.nan])
def test_invalid_lags(lags):
    with pytest.raises(ValueError, match="lags must be a positive integer"):
        MAMean(lags=lags)


@pytest.mark.parametrize("lags", [1, 3, np.int64(2), np.int8(4)])
def test_valid_lags(lags):
    model = MAMean(lags=lags)
    assert model.lags == int(lags)
    assert isinstance(model.lags, int)


@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("lags", [1, 2, 4])
def test_names_and_counts(lags, constant):
    model = MAMean(lags=lags, constant=constant)
    thetas = [f"theta[{i}]" for i in range(1, lags + 1)]
    expected = (["mu"] if constant else []) + thetas
    assert model.parameter_names() == expected
    assert model.num_params == len(expected)
    assert model.name == "MA"
    assert model.lags == lags
    assert model.constant is constant
    model.volatility = GARCH()
    assert model._all_parameter_names() == expected + [
        "omega",
        "alpha[1]",
        "beta[1]",
    ]


def test_default_arguments():
    model = MAMean()
    assert model.lags == 1
    assert model.constant
    assert isinstance(model.volatility, ConstantVariance)
    assert isinstance(model.distribution, Normal)
    assert model.y is None


def test_str_repr_html():
    model = MAMean(lags=3, volatility=GARCH(), distribution=StudentsT())
    text = str(model)
    assert text.startswith("MA(")
    assert "constant: yes" in text
    assert "lags: 3" in text
    assert "GARCH" in text
    assert "Standardized Student's t" in text
    assert "id: " in repr(model)
    html = model._repr_html_()
    assert "<strong>MA</strong>" in html
    assert "<strong>lags: </strong>3" in html
    assert "constant: no" in str(MAMean(constant=False))


def test_no_data():
    model = MAMean(lags=2)
    with pytest.raises(RuntimeError, match="Cannot estimate model without data"):
        model.fit()
    sim = model.simulate(np.r_[MA2_PARAMS, 1.0], 20)
    assert sim.shape == (20, 3)


# Innovations
@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("lags", [1, 2, 3])
def test_resids_reference(lags, constant):
    theta = np.array([0.5, -0.3, 0.2])[:lags]
    y = simulate_ma(theta, 200, mu=0.4, seed=lags)
    mu = 0.35
    params = np.r_[mu, theta] if constant else theta
    model = MAMean(y, lags=lags, constant=constant)
    expected = ma_resids(y, mu if constant else 0.0, theta)
    assert_allclose(model.resids(params), expected, atol=1e-12)
    # alternative data
    other = simulate_ma(theta, 50, seed=100)
    expected = ma_resids(other, mu if constant else 0.0, theta)
    assert_allclose(model.resids(params, other), expected, atol=1e-12)
    assert_allclose(model.resids(params, list(other)), expected, atol=1e-12)


def test_resids_zero_presample():
    # The first residual is y - mu, and then the lagged residuals matter
    model = MAMean(np.array([1.0, 2.0, 3.0]), lags=2, constant=False)
    assert_allclose(model.resids(np.array([0.5, 0.25])), [1.0, 1.5, 3.0 - 0.75 - 0.25])


def test_resids_exact_filter_converges():
    # The exact (Kalman filter) innovations from statsmodels match the
    # conditional ones after the effect of the zero pre-sample innovations has
    # decayed. This verifies the sign convention of theta.
    theta = np.array([0.5, 0.2])
    y = simulate_ma(theta, 400, mu=0.3, seed=7)
    exact = ARIMA(y, order=(0, 0, 2), trend="c").smooth(np.r_[0.3, theta, 1.0])
    conditional = MAMean(y, lags=2).resids(np.r_[0.3, theta])
    assert_allclose(np.asarray(exact.resid)[60:], conditional[60:], atol=1e-10)
    assert np.abs(np.asarray(exact.resid)[1:4] - conditional[1:4]).max() > 1e-4


def test_resids_explosive_are_finite():
    y = simulate_ma([0.4], 6000, seed=3)
    model = MAMean(y, lags=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resids = model.resids(np.array([0.0, 3.0, 2.0, 1.5]))
        resids = np.asarray(resids)
        assert np.all(np.isfinite(resids))
        assert np.abs(resids).max() <= 1e100
        # squares used by the likelihood do not overflow
        assert np.all(np.isfinite(resids**2))
    # and are unchanged when the residuals are not large
    normal = model.resids(np.array([0.0, 0.3, 0.2, 0.1]))
    assert_allclose(normal, ma_resids(y, 0.0, [0.3, 0.2, 0.1]), atol=1e-12)


# Simulation
@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("lags", [1, 2, 3])
def test_simulate_recovers_errors(lags, constant):
    theta = np.array([0.5, -0.3, 0.2])[:lags]
    dist = StudentsT(seed=RandomState(lags))
    model = MAMean(lags=lags, constant=constant, volatility=GARCH(), distribution=dist)
    mean_params = np.r_[0.25, theta] if constant else theta
    params = np.r_[mean_params, [0.05, 0.1, 0.85], [6.0]]
    sim = model.simulate(params, 300, burn=0)
    assert list(sim.columns) == ["data", "volatility", "errors"]
    assert sim.shape == (300, 3)
    errors = sim["errors"].to_numpy()
    y = sim["data"].to_numpy()
    mu = 0.25 if constant else 0.0
    assert_allclose(y, ma_process(errors, mu, theta), atol=1e-12)
    # With burn 0, the innovations are recovered exactly by the residuals
    assert_allclose(model.resids(mean_params, y), errors, atol=1e-10)


def test_simulate_volatility_is_recursion_of_shocks():
    # The variance depends on the simulated innovations, which is not the case
    # if the shocks are drawn separately from the variance path
    model = MAMean(lags=2, volatility=GARCH(), distribution=Normal(seed=5))
    params = np.r_[MA2_PARAMS, GARCH_PARAMS]
    sim = model.simulate(params, 500, burn=0)
    errors = sim["errors"].to_numpy()
    sigma2 = sim["volatility"].to_numpy() ** 2
    omega, alpha, beta = GARCH_PARAMS
    expected = omega + alpha * errors[:-1] ** 2 + beta * sigma2[:-1]
    assert_allclose(sigma2[1:], expected, rtol=1e-10)
    # shocks are standardized by the volatility
    assert abs(np.std(errors / np.sqrt(sigma2)) - 1.0) < 0.1


def test_simulate_moments():
    theta = np.array([0.6, 0.2])
    model = MAMean(lags=2, distribution=Normal(seed=RandomState(8)))
    sim = model.simulate(np.r_[0.1, theta, 2.0], 400000)
    y = sim["data"].to_numpy()
    var = 2.0 * (1 + theta @ theta)
    assert abs(y.mean() - 0.1) < 0.01
    assert abs(y.var() - var) < 0.03
    yc = y - y.mean()
    acov = [yc[k:] @ yc[: yc.shape[0] - k] / yc.shape[0] for k in range(1, 4)]
    assert abs(acov[0] - 2.0 * (theta[0] + theta[0] * theta[1])) < 0.03
    assert abs(acov[1] - 2.0 * theta[1]) < 0.03
    assert abs(acov[2]) < 0.03


def test_simulate_reproducible():
    params = np.r_[MA2_PARAMS, GARCH_PARAMS]
    first = MAMean(lags=2, volatility=GARCH(), distribution=Normal(seed=11))
    second = MAMean(lags=2, volatility=GARCH(), distribution=Normal(seed=11))
    third = MAMean(lags=2, volatility=GARCH(), distribution=Normal(seed=12))
    sim = first.simulate(params, 50)
    pd.testing.assert_frame_equal(sim, second.simulate(params, 50))
    assert not np.allclose(sim["data"], third.simulate(params, 50)["data"])


def test_simulate_errors():
    model = MAMean(lags=2)
    params = np.r_[MA2_PARAMS, 1.0]
    with pytest.raises(ValueError, match="Both initial value and x must be none"):
        model.simulate(params, 10, initial_value=1.0)
    with pytest.raises(ValueError, match="Both initial value and x must be none"):
        model.simulate(params, 10, x=np.zeros((510, 1)))
    with pytest.raises(ValueError, match="params has the wrong number of elements"):
        model.simulate(params[:-1], 10)
    with pytest.raises(ValueError, match="params has the wrong number of elements"):
        MAMean(lags=2, constant=False).simulate(params, 10)


# Invertibility, starting values, bounds and constraints
def test_make_invertible_examples():
    assert_allclose(_make_invertible(np.array([2.0])), [0.5])
    assert_allclose(_make_invertible(np.array([-2.0])), [-0.5])
    assert_allclose(_make_invertible(np.array([0.5])), [0.5])
    # A unit root is moved inside the allowed region
    assert_allclose(_make_invertible(np.array([1.0])), [1 / 1.01])
    # Roots at -4 and 0.5 for 1 + t1 z + t2 z^2 = (1 + z / 4) (1 - 2z): the
    # second root is reflected from 0.5 to 2
    theta = np.convolve([1.0, 0.25], [1.0, -2.0])[1:]
    assert not is_invertible(theta)
    assert_allclose(_make_invertible(theta), np.convolve([1, 0.25], [1, -0.5])[1:])
    # Zero coefficients and trailing zeros
    assert_allclose(_make_invertible(np.zeros(3)), np.zeros(3))
    assert_allclose(_make_invertible(np.array([2.0, 0.0])), [0.5, 0.0])
    assert_allclose(_make_invertible(np.array([np.nan, 0.1])), np.zeros(2))
    assert _make_invertible(np.empty(0)).shape == (0,)


@pytest.mark.parametrize("lags", [1, 2, 3, 4])
def test_make_invertible_random(lags):
    rs = RandomState(lags)
    for _ in range(200):
        theta = rs.uniform(-1.5, 1.5, lags)
        fixed = _make_invertible(theta)
        assert is_invertible(fixed, 0.0099)
        # idempotent
        assert_allclose(_make_invertible(fixed), fixed, atol=1e-12)
        if is_invertible(theta, 0.011):
            assert_allclose(fixed, theta)


def acf(theta, nlags):
    full = np.r_[1.0, theta]
    gamma = [full[: full.shape[0] - k] @ full[k:] for k in range(nlags + 1)]
    return np.array(gamma) / gamma[0]


def test_make_invertible_preserves_autocorrelation():
    # Reflecting roots does not change the autocorrelation of the process
    theta = np.convolve([1.0, 0.25], [1.0, -2.0])[1:]
    assert_allclose(acf(_make_invertible(theta), 2), acf(theta, 2), atol=1e-12)
    theta = np.convolve(np.convolve([1.0, 3.0], [1.0, 0.4]), [1.0, -1.7])[1:]
    assert_allclose(acf(_make_invertible(theta), 3), acf(theta, 3), atol=1e-12)


@pytest.mark.parametrize("theta", [[0.5], [0.5, 0.2], [0.4, 0.3, 0.2]])
def test_starting_values_recover(theta):
    lags = len(theta)
    y = simulate_ma(theta, 30000, mu=0.2, seed=21)
    start = _ma_starting_values(y, lags, True)
    assert start.shape == (lags + 1,)
    assert abs(start[0] - y.mean()) < 1e-12
    assert_allclose(start[1:], theta, atol=0.05)
    model = MAMean(y, lags=lags)
    assert_allclose(model.starting_values(), start)
    no_const = _ma_starting_values(y - y.mean(), lags, False)
    assert no_const.shape == (lags,)
    assert_allclose(no_const, theta, atol=0.05)


def test_starting_values_invertible():
    # Data from a non-invertible process: theta = 2 is equivalent to 0.5
    y = simulate_ma([2.0], 5000, seed=31)
    start = _ma_starting_values(y, 1, True)
    assert abs(start[1]) < 1
    assert_allclose(start[1], 0.5, atol=0.05)
    for lags in (1, 2, 3):
        for seed in range(10):
            y = RandomState(seed).standard_normal(300).cumsum()
            start = _ma_starting_values(np.diff(y), lags, False)
            assert is_invertible(start)


@pytest.mark.parametrize(
    "theta", [[0.99], list(np.convolve([1.0, 0.98], [1.0, 0.5])[1:])]
)
def test_starting_values_near_unit_root(theta):
    # With a root close to the unit circle and a small sample the regression
    # estimates are often not invertible, but the starting values always are
    lags = len(theta)
    nobs = 80 if lags == 1 else 120
    for seed in range(60):
        y = simulate_ma(theta, nobs, seed=seed)
        start = _ma_starting_values(y, lags, True)
        assert is_invertible(start[1:], 0.0099)
    model = MAMean(simulate_ma(theta, nobs, seed=1), lags=lags)
    assert is_invertible(model.starting_values()[1:], 0.0099)


def test_starting_values_small_sample():
    # Too small to estimate the regressions
    start = _ma_starting_values(RandomState(0).standard_normal(5), 3, True)
    assert start.shape == (4,)
    assert_allclose(start[1:], 0.0)
    with pytest.raises(ValueError, match="Insufficient data"):
        MAMean(RandomState(0).standard_normal(3), lags=3).fit()


def test_bounds():
    assert MAMean(lags=1).bounds() == [(-np.inf, np.inf), (-1.0, 1.0)]
    assert MAMean(lags=1, constant=False).bounds() == [(-1.0, 1.0)]
    assert MAMean(lags=2).bounds() == [(-np.inf, np.inf)] * 3
    assert MAMean(lags=3).bounds() == [(-np.inf, np.inf)] * 4


@pytest.mark.parametrize("constant", [True, False])
def test_constraints_are_invertibility(constant):
    model = MAMean(lags=2, constant=constant)
    a, b = model.constraints()
    assert a.shape == (3, model.num_params)
    assert b.shape == (3,)
    rs = RandomState(77)
    count = 0
    for _ in range(2000):
        theta = rs.uniform(-2.2, 2.2, 2)
        params = np.r_[rs.standard_normal(), theta] if constant else theta
        satisfied = bool(np.all(a @ params - b >= 0))
        assert satisfied == is_invertible(theta)
        count += satisfied
    # The region is a triangle with area 4 in a square with area 19.36
    assert 0.15 < count / 2000 < 0.27


@pytest.mark.parametrize("lags", [1, 3, 4])
def test_no_constraints(lags):
    a, b = MAMean(lags=lags).constraints()
    assert a.shape == (0, lags + 1)
    assert b.shape == (0,)


# Estimation
def gaussian_llf(resids, sigma2):
    return -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + resids**2 / sigma2)


def garch_variance(resids, params, backcast):
    omega, alpha, beta = params
    sigma2 = np.empty(resids.shape[0])
    previous_resid2 = previous_sigma2 = backcast
    for t in range(resids.shape[0]):
        sigma2[t] = omega + alpha * previous_resid2 + beta * previous_sigma2
        previous_resid2 = resids[t] ** 2
        previous_sigma2 = sigma2[t]
    return sigma2


def split(y, location):
    if isinstance(y, np.ndarray):
        return y[:location], y[location:]
    return y.iloc[:location], y.iloc[location:]


@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("theta", [[0.6], [0.6, 0.2]])
def test_fit_matches_direct_likelihood(theta, constant):
    # Maximize the conditional Gaussian likelihood directly, concentrating out
    # the innovation variance, using loops for the innovations
    lags = len(theta)
    y = simulate_ma(theta, 800, mu=0.3 if constant else 0.0, seed=lags + 40)
    res = MAMean(y, lags=lags, constant=constant).fit(disp="off")
    assert res.convergence_flag == 0

    def negative_llf(p):
        mu = p[0] if constant else 0.0
        theta_p = p[int(constant) :]
        if not is_invertible(theta_p):
            return 1e10
        e = ma_resids(y, mu, theta_p)
        sigma2 = e @ e / e.shape[0]
        return -gaussian_llf(e, sigma2)

    start = np.r_[y.mean(), np.zeros(lags)] if constant else np.zeros(lags)
    direct = optimize.minimize(
        negative_llf,
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000, "maxfev": 4000},
    )
    mean_params = res.params.to_numpy()[:-1]
    assert_allclose(mean_params, direct.x, atol=2e-3)
    assert_allclose(res.loglikelihood, -direct.fun, atol=1e-4)
    # The likelihood computed with the estimated parameters
    mu = mean_params[0] if constant else 0.0
    e = ma_resids(y, mu, mean_params[int(constant) :])
    assert_allclose(res.loglikelihood, gaussian_llf(e, res.params["sigma2"]), rtol=1e-9)
    assert_allclose(res.params["sigma2"], e @ e / e.shape[0], rtol=1e-3)
    assert_allclose(res.resid, e, atol=1e-10)


def test_fit_loglikelihood_garch(garch_ma2):
    model = MAMean(garch_ma2, lags=2, volatility=GARCH())
    res = model.fit(disp="off")
    assert res.convergence_flag == 0
    y = np.asarray(garch_ma2, dtype=float)
    mu, theta = res.params.iloc[0], res.params.iloc[1:3].to_numpy()
    e = ma_resids(y, mu, theta)
    sigma2 = garch_variance(e, res.params.iloc[3:].to_numpy(), model._backcast)
    assert_allclose(res.loglikelihood, gaussian_llf(e, sigma2), rtol=1e-9)
    assert_allclose(np.asarray(res.resid), e, atol=1e-10)
    assert_allclose(np.asarray(res.conditional_volatility), np.sqrt(sigma2))
    assert_allclose(np.asarray(res.std_resid), e / np.sqrt(sigma2))
    # Close to the process that generated the data
    true = np.concatenate((MA2_PARAMS, GARCH_PARAMS))
    assert np.all(np.abs(res.params.to_numpy() - true) < 4 * res.std_err.to_numpy())
    # and no worse than the true parameters
    fixed = model.fix(true)
    assert res.loglikelihood >= fixed.loglikelihood


def test_fix_equals_fit(garch_ma2):
    model = MAMean(garch_ma2, lags=2, volatility=GARCH())
    res = model.fit(disp="off")
    fixed = model.fix(res.params)
    assert_allclose(fixed.loglikelihood, res.loglikelihood, rtol=1e-12)
    assert_allclose(np.asarray(fixed.resid), np.asarray(res.resid))
    assert_allclose(
        np.asarray(fixed.conditional_volatility),
        np.asarray(res.conditional_volatility),
    )


@pytest.mark.parametrize("vol", [ConstantVariance(), GARCH(), ARCH(2)])
def test_zero_theta_is_constant_mean(garch_ma2, vol):
    # An MA with theta = 0 is a constant mean model
    if isinstance(vol, ConstantVariance):
        vol_params = np.array([1.0])
    elif isinstance(vol, ARCH):
        vol_params = np.array([0.5, 0.1, 0.2])
    else:
        vol_params = GARCH_PARAMS
    ma = MAMean(garch_ma2, lags=2, volatility=vol).fix(np.r_[0.2, 0.0, 0.0, vol_params])
    cm = ConstantMean(garch_ma2, volatility=vol).fix(np.r_[0.2, vol_params])
    assert_allclose(np.asarray(ma.resid), np.asarray(cm.resid))
    # The backcast of the variance is computed using the residuals at the
    # starting values, which differ from the residuals with theta = 0
    rtol = 1e-12 if isinstance(vol, ConstantVariance) else 5e-3
    assert_allclose(ma.loglikelihood, cm.loglikelihood, rtol=rtol)


def test_non_invertible_data():
    # theta = 2 and sigma2 = 1 has the same distribution as theta = 0.5 and
    # sigma2 = 4, and only the second is invertible
    y = simulate_ma([2.0], 4000, seed=19)
    res = MAMean(y, lags=1).fit(disp="off")
    assert abs(res.params["theta[1]"] - 0.5) < 0.05
    assert abs(res.params["sigma2"] - 4.0) < 0.4


@pytest.mark.parametrize("theta", [[0.4, 0.3, 0.2], [0.4, 0.3, 0.2, 0.1]])
def test_fit_long_ma(theta):
    # No invertibility constraints, and no warnings when moving outside of it
    lags = len(theta)
    y = simulate_ma(theta, 6000, mu=0.2, seed=lags)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = MAMean(y, lags=lags).fit(disp="off")
    assert res.convergence_flag == 0
    assert_allclose(res.params.iloc[1 : lags + 1], theta, atol=0.06)
    assert is_invertible(res.params.iloc[1 : lags + 1])
    assert np.all(res.std_err.to_numpy() > 0)


@pytest.mark.parametrize(
    "dist", [Normal, StudentsT, SkewStudent, GeneralizedError], ids=lambda d: d.__name__
)
@pytest.mark.parametrize("vol", [ConstantVariance, GARCH, EGARCH])
def test_distributions_and_volatility(garch_ma2, vol, dist):
    model = MAMean(garch_ma2, lags=2, volatility=vol(), distribution=dist())
    res = model.fit(disp="off", show_warning=False)
    assert np.isfinite(res.loglikelihood)
    assert isinstance(res, ARCHModelResult)
    fixed = model.fix(res.params)
    assert_allclose(fixed.loglikelihood, res.loglikelihood, rtol=1e-10)
    assert_allclose(res.params.iloc[:3], MA2_PARAMS, atol=0.2)


def test_other_volatility_processes(garch_ma2):
    nobs = np.asarray(garch_ma2).shape[0]
    # EWMA has no parameters, so there is nothing to estimate for the variance
    res = MAMean(garch_ma2, lags=2, volatility=EWMAVariance(0.94)).fit(disp="off")
    assert res.params.shape[0] == 3
    # FixedVariance uses the sample of the mean model
    model = MAMean(garch_ma2, lags=2, volatility=FixedVariance(np.full(nobs, 2.0)))
    assert model.fit(disp="off").params.shape[0] == 4
    res = model.fit(disp="off", first_obs=100, last_obs=1000)
    assert res.nobs == 900


def test_summary(garch_ma2):
    res = MAMean(garch_ma2, lags=2, volatility=GARCH()).fit(disp="off")
    text = str(res.summary())
    assert "MA - GARCH Model Results" in text
    assert "Mean Model:" in text
    for name in ("mu", "theta[1]", "theta[2]", "omega", "alpha[1]", "beta[1]"):
        assert name in text
    assert "Df Model:                            3" in text


def test_rsquared():
    y = simulate_ma([0.6], 1000, mu=0.3, seed=2)
    res = MAMean(y, lags=1).fit(disp="off")
    e = np.asarray(res.resid)
    assert_allclose(res.rsquared, 1 - e @ e / ((y - y.mean()) @ (y - y.mean())))
    res = MAMean(y, lags=1, constant=False).fit(disp="off")
    e = np.asarray(res.resid)
    assert_allclose(res.rsquared, 1 - e @ e / (y @ y))
    # No variation in the data
    assert np.isnan(MAMean(np.ones(100), lags=1)._r2(np.r_[0.0, 0.0] + 1.0))


@pytest.mark.slow
def test_standard_errors_and_bias():
    # Monte Carlo: the reported standard errors describe the sampling variation
    theta, nobs, reps = 0.5, 600, 300
    estimates = np.empty(reps)
    errors = np.empty(reps)
    covered = 0
    for i in range(reps):
        y = simulate_ma([theta], nobs, mu=0.2, seed=1000 + i)
        res = MAMean(y, lags=1).fit(disp="off", cov_type="classic")
        estimates[i] = res.params["theta[1]"]
        errors[i] = res.std_err["theta[1]"]
        low, high = res.conf_int().loc["theta[1]"]
        covered += low <= theta <= high
    assert abs(estimates.mean() - theta) < 0.01
    assert 0.85 < errors.mean() / estimates.std() < 1.15
    assert 0.90 < covered / reps < 0.99


# Samples
def test_fit_sample_equals_subsample(garch_ma2):
    # The innovations start at 0 in the first observation of the estimation sample
    first, last = 80, 1100
    full = MAMean(garch_ma2, lags=2, volatility=GARCH())
    res = full.fit(disp="off", first_obs=first, last_obs=last)
    sub = (
        garch_ma2[first:last]
        if isinstance(garch_ma2, np.ndarray)
        else garch_ma2.iloc[first:last]
    )
    expected = MAMean(sub, lags=2, volatility=GARCH()).fit(disp="off")
    assert_allclose(res.params, expected.params, rtol=1e-8)
    assert_allclose(res.loglikelihood, expected.loglikelihood, rtol=1e-10)
    assert (res.fit_start, res.fit_stop, res.nobs) == (first, last, last - first)
    resid = np.asarray(res.resid)
    assert np.all(np.isnan(resid[:first]))
    assert np.all(np.isnan(resid[last:]))
    mu, theta = res.params.iloc[0], res.params.iloc[1:3].to_numpy()
    y = np.asarray(garch_ma2, dtype=float)
    assert_allclose(resid[first:last], ma_resids(y[first:last], mu, theta), atol=1e-10)


@pytest.mark.parametrize("hold_back", [5, 20])
def test_hold_back(garch_ma2, hold_back):
    held = MAMean(garch_ma2, lags=2, hold_back=hold_back).fit(disp="off")
    first = MAMean(garch_ma2, lags=2).fit(disp="off", first_obs=hold_back)
    assert_allclose(held.params, first.params, rtol=1e-10)
    assert held.fit_start == hold_back
    assert held.nobs == np.asarray(garch_ma2).shape[0] - hold_back
    # in addition to first_obs
    both = MAMean(garch_ma2, lags=2, hold_back=hold_back).fit(
        disp="off", first_obs=hold_back
    )
    assert both.fit_start == 2 * hold_back


def test_fit_date_cutoffs():
    y = pd.Series(
        simulate_ma([0.5], 700, seed=9),
        index=pd.date_range("2000-01-03", periods=700, freq="B"),
    )
    model = MAMean(y, lags=1)
    by_date = model.fit(disp="off", first_obs="2000-06-01", last_obs="2001-12-31")
    first = int(np.searchsorted(y.index, pd.Timestamp("2000-06-01")))
    last = int(np.searchsorted(y.index, pd.Timestamp("2001-12-31")))
    by_index = model.fit(disp="off", first_obs=first, last_obs=last)
    assert_allclose(by_date.params, by_index.params)
    assert by_date.nobs == by_index.nobs
    with pytest.raises(ValueError, match="first_obs and last_obs produce an empty"):
        model.fit(disp="off", first_obs=100, last_obs=50)


def test_rescale():
    y = simulate_ma([0.5], 2000, mu=0.0003, seed=4, scale=0.001)
    with pytest.warns(DataScaleWarning, match="y is poorly scaled"):
        MAMean(y, lags=1).fit(disp="off")
    res = MAMean(y, lags=1, rescale=True).fit(disp="off")
    assert res.scale == 1000
    expected = MAMean(1000 * y, lags=1).fit(disp="off")
    assert_allclose(res.params, expected.params, rtol=1e-8)
    assert_allclose(res.params["theta[1]"], 0.5, atol=0.06)


def test_input_types():
    y = simulate_ma([0.5], 400, seed=14)
    expected = MAMean(y, lags=1).fit(disp="off")
    for data in (list(y), pd.Series(y), pd.DataFrame(y), y.astype(np.float32)):
        res = MAMean(data, lags=1).fit(disp="off")
        assert_allclose(res.params, expected.params, rtol=1e-4)
    with pytest.raises(ValueError, match="NaN or inf values found in y"):
        MAMean(np.r_[y, np.nan], lags=1)


def test_append(garch_ma2):
    y = garch_ma2
    nobs = np.asarray(y).shape[0]
    head, tail = split(y, 1000)
    model = MAMean(head, lags=2, volatility=GARCH())
    res = model.fit(disp="off")
    model.append(tail)
    assert model._fit_indices == [0, nobs]
    assert model._fit_y.shape[0] == nobs
    full = MAMean(y, lags=2, volatility=GARCH())
    appended = model.forecast(res.params, horizon=3, reindex=False)
    expected = full.forecast(res.params, horizon=3, reindex=False)
    assert_allclose(appended.mean, expected.mean)
    assert_allclose(appended.variance, expected.variance)
    # The innovations use the new data
    mean_params = res.params.to_numpy()[:3]
    assert_allclose(model.resids(mean_params), full.resids(mean_params))
    fresh = MAMean(head, lags=2)
    with pytest.raises(ValueError, match="MAMean does not include exogenous"):
        fresh.append(tail, x=np.ones((tail.shape[0], 1)))
    assert fresh._y.shape[0] == nobs - tail.shape[0]


# Forecasts
def psi_weights(theta, horizon):
    psi = np.zeros(horizon)
    psi[0] = 1.0
    count = min(len(theta), horizon - 1)
    psi[1 : count + 1] = theta[:count]
    return psi


def reference_innovations(y, mu, theta, first):
    """Innovations that start at 0 in the observation first"""
    e = np.zeros(y.shape[0])
    e[first:] = ma_resids(y[first:], mu, theta)
    return e


def reference_mean_forecasts(y, mu, theta, horizon, first, start):
    """E[y_{i+h} | F_i] = mu + sum_{j >= h} theta_j e_{i+h-j}"""
    e = reference_innovations(y, mu, theta, first)
    nobs = y.shape[0]
    out = np.full((nobs - start, horizon), float(mu))
    for row, i in enumerate(range(start, nobs)):
        for h in range(1, horizon + 1):
            for j in range(h, len(theta) + 1):
                if i + h - j >= 0:
                    out[row, h - 1] += theta[j - 1] * e[i + h - j]
    return out


def reference_garch_forecasts(
    y, mu, theta, vol_params, backcast, horizon, first, start
):
    """Variances of future innovations and of future y using GARCH(1,1)"""
    omega, alpha, beta = vol_params
    e = reference_innovations(y, mu, theta, first)
    sigma2 = np.full(y.shape[0], np.nan)
    sigma2[first:] = garch_variance(e[first:], vol_params, backcast)
    psi = psi_weights(theta, horizon)
    nobs = y.shape[0]
    resid_var = np.full((nobs - start, horizon), np.nan)
    total_var = np.full((nobs - start, horizon), np.nan)
    for row, i in enumerate(range(start, nobs)):
        if i < first:
            continue
        resid_var[row, 0] = omega + alpha * e[i] ** 2 + beta * sigma2[i]
        for h in range(1, horizon):
            resid_var[row, h] = omega + (alpha + beta) * resid_var[row, h - 1]
        for h in range(horizon):
            # Var(sum_{j<=h} psi_j e_{i+h+1-j})
            total_var[row, h] = sum(
                psi[j] ** 2 * resid_var[row, h - j] for j in range(h + 1)
            )
    return resid_var, total_var


@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("lags", [1, 2, 3])
def test_forecast_mean_closed_form(lags, constant):
    # Previously the forecast was mu at every horizon
    theta = np.array([0.5, -0.3, 0.2])[:lags]
    mu = 0.25 if constant else 0.0
    y = simulate_ma(theta, 120, mu=mu, seed=lags)
    model = MAMean(y, lags=lags, constant=constant, volatility=GARCH())
    mean_params = np.r_[mu, theta] if constant else theta
    params = np.r_[mean_params, GARCH_PARAMS]
    horizon = lags + 3
    fc = model.forecast(params, horizon=horizon, start=0, reindex=False)
    assert isinstance(fc, ARCHModelForecast)
    expected = reference_mean_forecasts(y, mu, theta, horizon, 0, 0)
    assert fc.mean.shape == (120, horizon)
    assert_allclose(fc.mean.to_numpy(), expected, atol=1e-12)
    # The MA terms matter for the first q steps and not after
    assert np.abs(expected[:, :lags] - mu).max() > 0.1
    assert_allclose(fc.mean.to_numpy()[:, lags:], mu)
    assert list(fc.mean.columns) == [f"h.{h}" for h in range(1, horizon + 1)]


def test_forecast_default_is_last_observation(garch_ma2):
    model = MAMean(garch_ma2, lags=2, volatility=GARCH())
    res = model.fit(disp="off")
    default = res.forecast(horizon=4, reindex=False)
    assert default.mean.shape == (1, 4)
    everything = res.forecast(horizon=4, start=0, reindex=False)
    assert_allclose(default.mean.to_numpy()[0], everything.mean.to_numpy()[-1])
    assert_allclose(default.variance.to_numpy()[0], everything.variance.to_numpy()[-1])
    direct = model.forecast(res.params, horizon=4, reindex=False)
    assert_allclose(default.mean, direct.mean)
    assert_allclose(default.variance, direct.variance)
    # From the last residuals
    y = np.asarray(garch_ma2, dtype=float)
    mu, theta = res.params.iloc[0], res.params.iloc[1:3].to_numpy()
    e = ma_resids(y, mu, theta)
    expected = [
        mu + theta[0] * e[-1] + theta[1] * e[-2],
        mu + theta[1] * e[-1],
        mu,
        mu,
    ]
    assert_allclose(default.mean.to_numpy()[0], expected)


@pytest.mark.parametrize("lags", [1, 2])
def test_forecast_variance_closed_form_garch(lags):
    theta = np.array([0.6, 0.25])[:lags]
    horizon = 5
    mean_sim = MAMean(
        lags=lags, volatility=GARCH(), distribution=Normal(seed=RandomState(lags))
    )
    y = mean_sim.simulate(np.r_[0.1, theta, GARCH_PARAMS], 200, burn=0)[
        "data"
    ].to_numpy()
    model = MAMean(y, lags=lags, volatility=GARCH())
    params = np.r_[0.12, theta, 0.04, 0.12, 0.8]
    backcast = model.volatility.backcast(model.resids(params[: lags + 1]))
    fc = model.forecast(params, horizon=horizon, start=0, reindex=False)
    resid_var, total_var = reference_garch_forecasts(
        y, 0.12, theta, params[lags + 1 :], backcast, horizon, 0, 0
    )
    assert_allclose(fc.residual_variance.to_numpy(), resid_var, rtol=1e-10)
    assert_allclose(fc.variance.to_numpy(), total_var, rtol=1e-10)
    # The variance of the data is larger than the variance of the innovation
    assert np.all(
        fc.variance.to_numpy()[:, 1:] > fc.residual_variance.to_numpy()[:, 1:]
    )
    assert_allclose(fc.variance.to_numpy()[:, 0], fc.residual_variance.to_numpy()[:, 0])


def test_forecast_variance_constant_variance():
    theta = np.array([0.6, 0.2])
    y = simulate_ma(theta, 80, seed=5)
    model = MAMean(y, lags=2)
    fc = model.forecast(np.r_[0.0, theta, 1.7], horizon=5, start=10, reindex=False)
    psi = psi_weights(theta, 5)
    assert_allclose(fc.residual_variance.to_numpy(), 1.7)
    expected = 1.7 * np.cumsum(psi**2)
    assert_allclose(fc.variance.to_numpy(), np.tile(expected, (70, 1)))
    assert_allclose(expected, 1.7 * np.array([1, 1.36, 1.4, 1.4, 1.4]))


@pytest.mark.parametrize("sample", ["all", "hold_back", "first_last_obs"])
def test_forecast_uses_fit_sample_start(garch_ma2, sample):
    # Innovations start at 0 in the first observation of the estimation sample,
    # and forecasts can be made from any later observation
    y = np.asarray(garch_ma2, dtype=float)
    kwargs = {}
    init = {}
    first = 0
    if sample == "hold_back":
        init["hold_back"] = 30
        first = 30
    elif sample == "first_last_obs":
        kwargs = {"first_obs": 50, "last_obs": 900}
        first = 50
    model = MAMean(garch_ma2, lags=2, volatility=GARCH(), **init)
    res = model.fit(disp="off", **kwargs)
    mu, theta = res.params.iloc[0], res.params.iloc[1:3].to_numpy()
    last = kwargs.get("last_obs", y.shape[0])
    backcast = model.volatility.backcast(ma_resids(y[first:last], mu, theta))
    # Default origin is the last observation of the estimation sample
    fc = res.forecast(horizon=3, reindex=False)
    start = last - 1
    assert fc.mean.shape == (y.shape[0] - start, 3)
    expected = reference_mean_forecasts(y, mu, theta, 3, first, start)
    assert_allclose(fc.mean.to_numpy(), expected, atol=1e-10)
    _, total_var = reference_garch_forecasts(
        y, mu, theta, res.params.iloc[3:].to_numpy(), backcast, 3, first, start
    )
    assert_allclose(fc.variance.to_numpy(), total_var, rtol=1e-8)
    # And from the observation before the sample, where nothing is known
    earliest = first - 1
    if earliest >= 0:
        fc = res.forecast(horizon=3, start=earliest, reindex=False)
        expected = reference_mean_forecasts(y, mu, theta, 3, first, earliest)
        assert_allclose(fc.mean.to_numpy(), expected, atol=1e-10)
        assert_allclose(fc.mean.to_numpy()[0], mu)
        assert np.all(np.isnan(fc.variance.to_numpy()[0]))
        assert not np.any(np.isnan(fc.variance.to_numpy()[1:]))
        with pytest.raises(ValueError, match="start cannot be less than the index"):
            res.forecast(horizon=3, start=earliest - 1)


def test_forecast_start_and_alignment(garch_ma2):
    model = MAMean(garch_ma2, lags=2, volatility=GARCH())
    res = model.fit(disp="off")
    nobs = np.asarray(garch_ma2).shape[0]
    full = res.forecast(horizon=3, start=0, reindex=False)
    assert full.mean.shape == (nobs, 3)
    for start in (1, 700, nobs - 3):
        fc = res.forecast(horizon=3, start=start, reindex=False)
        assert_allclose(fc.mean.to_numpy(), full.mean.to_numpy()[start:], atol=1e-12)
        assert_allclose(fc.variance.to_numpy(), full.variance.to_numpy()[start:])
    if isinstance(garch_ma2, pd.Series):
        label = garch_ma2.index[700]
        by_label = res.forecast(horizon=3, start=label, reindex=False)
        assert_allclose(by_label.mean.to_numpy(), full.mean.to_numpy()[700:])
        assert by_label.mean.index[0] == label
    # In target alignment row t, column h.k is the forecast made at t-k
    origin = res.forecast(horizon=3, start=nobs - 5, reindex=False)
    target = res.forecast(horizon=3, start=nobs - 5, align="target", reindex=False)
    for k in (1, 2, 3):
        column = origin.mean.to_numpy()[:, k - 1]
        shifted = target.mean.to_numpy()[:, k - 1]
        assert_allclose(shifted[k:], column[: column.shape[0] - k])
        assert np.all(np.isnan(shifted[:k]))
    # reindex pads to the full sample with missing values
    reindexed = res.forecast(horizon=3, start=nobs - 5, reindex=True)
    assert reindexed.mean.shape == (nobs, 3)
    assert reindexed.mean.iloc[: nobs - 5].isna().all().all()
    assert_allclose(reindexed.mean.to_numpy()[nobs - 5 :], origin.mean.to_numpy())


def test_forecast_errors(garch_ma2):
    model = MAMean(garch_ma2, lags=2, volatility=GARCH(), hold_back=10)
    res = model.fit(disp="off")
    for horizon in (0, -1, 1.5, "1"):
        with pytest.raises(ValueError, match="horizon must be an integer >= 1"):
            res.forecast(horizon=horizon)
    assert res.forecast(horizon=np.int64(2), reindex=False).mean.shape[1] == 2
    with pytest.raises(TypeError, match="x is not None but the model does not"):
        res.forecast(x=np.zeros((10, 1)))
    with pytest.raises(ValueError, match="start cannot be less than the index"):
        res.forecast(start=5)
    with pytest.raises(ValueError, match="start cannot be less than the index"):
        model.forecast(res.params, start=8)
    assert res.forecast(start=9, reindex=False).mean.shape[0] == (
        np.asarray(garch_ma2).shape[0] - 9
    )


@pytest.mark.parametrize("method", ["simulation", "bootstrap"])
@pytest.mark.parametrize("lags", [1, 2, 3])
def test_forecast_paths_structure(garch_ma2, method, lags):
    # Each path is the mean forecast plus the accumulated shocks,
    # sum_{j<h} psi_j shock_{h-j}. This depends on the innovations before the
    # forecast origin entering the first q steps.
    res = MAMean(garch_ma2, lags=lags, volatility=GARCH()).fit(disp="off")
    nobs = np.asarray(garch_ma2).shape[0]
    horizon = lags + 2
    fc = res.forecast(
        horizon=horizon,
        start=nobs - 6,
        method=method,
        simulations=40,
        random_state=RandomState(11),
        reindex=False,
    )
    paths = fc.simulations.values
    shocks = fc.simulations.residuals
    assert paths.shape == (6, 40, horizon)
    assert shocks.shape == (6, 40, horizon)
    theta = res.params.iloc[1 : lags + 1].to_numpy()
    psi = psi_weights(theta, horizon)
    mean = fc.mean.to_numpy()
    for h in range(horizon):
        accumulated = sum(psi[j] * shocks[:, :, h - j] for j in range(h + 1))
        expected = np.broadcast_to(mean[:, h : h + 1], accumulated.shape)
        assert_allclose(paths[:, :, h] - accumulated, expected, atol=1e-10)
    # Variances of the paths relate to the variances of the shocks
    variances = fc.simulations.variances
    resid_variances = fc.simulations.residual_variances
    assert variances.shape == resid_variances.shape == paths.shape
    for h in range(horizon):
        expected = sum(psi[j] ** 2 * resid_variances[:, :, h - j] for j in range(h + 1))
        assert_allclose(variances[:, :, h], expected, rtol=1e-10)


def test_forecast_paths_moments():
    theta = np.array([0.6, 0.3])
    y = simulate_ma(theta, 300, mu=0.2, seed=61)
    model = MAMean(y, lags=2)
    params = np.r_[0.2, theta, 2.0]
    horizon = 4
    fc = model.forecast(
        params,
        horizon=horizon,
        start=295,
        method="simulation",
        simulations=40000,
        random_state=RandomState(5),
        reindex=False,
    )
    paths = fc.simulations.values
    assert_allclose(paths.mean(1), fc.mean.to_numpy(), atol=0.03)
    assert_allclose(paths.var(1), fc.variance.to_numpy(), rtol=0.03)


def test_forecast_paths_before_sample(garch_ma2):
    model = MAMean(garch_ma2, lags=2, volatility=GARCH(), hold_back=10)
    res = model.fit(disp="off")
    fc = res.forecast(
        horizon=3,
        start=9,
        method="simulation",
        simulations=20,
        random_state=RandomState(1),
        reindex=False,
    )
    # Nothing is known about the variance at the first origin
    assert np.all(np.isnan(fc.simulations.values[0]))
    assert np.all(np.isnan(fc.simulations.residuals[0]))
    assert not np.any(np.isnan(fc.simulations.values[1:]))


def test_fixed_result_forecast(garch_ma2):
    model = MAMean(garch_ma2, lags=2, volatility=GARCH())
    res = model.fit(disp="off")
    fixed = model.fix(res.params)
    assert_allclose(
        fixed.forecast(horizon=3, reindex=False).mean,
        res.forecast(horizon=3, reindex=False).mean,
        atol=1e-12,
    )


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plots(garch_ma2, agg_backend):
    res = MAMean(garch_ma2, lags=2, volatility=GARCH()).fit(disp="off")
    res.plot()
    res.hedgehog_plot(start=1400)
    res.hedgehog_plot(start=1400, plot_type="mean")
    res.hedgehog_plot(start=1400, method="simulation", simulations=50)


def test_append_hold_back(garch_ma2):
    # The estimation sample after appending excludes the held back observations
    nobs = np.asarray(garch_ma2).shape[0]
    head, tail = split(garch_ma2, 1000)
    variance = np.ones(nobs)
    model = MAMean(head, lags=2, hold_back=15, volatility=FixedVariance(variance))
    model.append(tail)
    assert model._fit_indices == [15, nobs]
    assert model._fit_y.shape[0] == nobs - 15
    assert (model.volatility.start, model.volatility.stop) == (15, nobs)
    res = model.fit(disp="off")
    assert res.fit_start == 15
    assert res.nobs == nobs - 15
