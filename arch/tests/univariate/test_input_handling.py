import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest
from scipy.special import gammaln

from arch.univariate import (
    APARCH,
    ARCH,
    ARX,
    EGARCH,
    GARCH,
    HARCH,
    HARX,
    LS,
    ConstantMean,
    EWMAVariance,
    MIDASHyperbolic,
    Normal,
    StudentsT,
    ZeroMean,
    arch_model,
)


def midas_weights(theta, m):
    # phi_i(theta) proportional to Gamma(i + theta) / (Gamma(i + 1) Gamma(theta))
    j = np.arange(1.0, m + 1)
    w = np.exp(gammaln(theta + j) - gammaln(j + 1) - gammaln(theta))
    return w / w.sum()


def midas_reference(params, errors, nobs, burn, m, asym):
    """Direct implementation of the MIDAS Hyperbolic recursion"""
    if asym:
        omega, alpha, gamma, theta = params
    else:
        omega, alpha, theta = params
        gamma = 0.0
    weights = midas_weights(theta, m)
    # The pre-sample values are the unconditional variance
    initial = omega / (1 - alpha - gamma / 2)
    total = nobs + burn
    sigma2 = np.empty(total)
    data = np.empty(total)
    sigma2[:m] = initial
    data[:m] = np.sqrt(initial)
    for t in range(m, total):
        sigma2[t] = omega
        for i in range(m):
            lagged = data[t - 1 - i]
            if t - 1 - i < m:
                coef = alpha + gamma / 2
            else:
                coef = alpha + gamma * (lagged < 0)
            sigma2[t] += coef * weights[i] * lagged**2
        data[t] = errors[t] * np.sqrt(sigma2[t])
    return data[burn:], sigma2[burn:]


@pytest.mark.parametrize("burn", [0, 5, 21, 22, 30])
@pytest.mark.parametrize("asym", [False, True], ids=["symmetric", "asymmetric"])
def test_midas_simulate_short_burn(burn, asym):
    m, nobs = 22, 40
    params = np.array([0.1, 0.3, 0.4, 0.8] if asym else [0.1, 0.5, 0.8])
    errors = np.random.default_rng(3).standard_normal(nobs + 500)

    def rng(size):
        return errors[:size]

    process = MIDASHyperbolic(m, asym=asym)
    data, sigma2 = process.simulate(params, nobs, rng, burn=burn)
    # The first m observations are always discarded
    expected_data, expected_sigma2 = midas_reference(
        params, errors, nobs, max(burn, m), m, asym
    )
    assert data.shape == (nobs,)
    assert sigma2.shape == (nobs,)
    assert_allclose(data, expected_data, rtol=1e-10)
    assert_allclose(sigma2, expected_sigma2, rtol=1e-10)


def test_midas_simulate_burn_zero_mean_model():
    mod = ConstantMean(volatility=MIDASHyperbolic(22), distribution=Normal(seed=None))
    sim = mod.simulate([0.0, 0.1, 0.5, 0.8], 50, burn=0)
    assert sim.shape[0] == 50
    assert np.all(np.isfinite(sim["data"]))


def test_midas_str():
    assert str(MIDASHyperbolic(22)) == "MIDAS Hyperbolic(lags: 22, asym: False)"
    assert str(MIDASHyperbolic(5, asym=True)) == "MIDAS Hyperbolic(lags: 5, asym: True)"


@pytest.mark.parametrize(
    "lags",
    [
        np.array([1, 5, 22]),
        np.array([1, 5, 22], dtype=np.int32),
        pd.Series([1, 5, 22]),
        (1, 5, 22),
        [1, 5, 22],
    ],
    ids=["int64-array", "int32-array", "series", "tuple", "list"],
)
def test_harch_lags_formats(lags):
    harch = HARCH(lags)
    assert_array_equal(harch.lags, [1, 5, 22])
    assert harch.num_params == 4
    assert str(harch) == "HARCH(lags: 1, 5, 22)"


@pytest.mark.parametrize(
    "lags", [3, np.int64(3), np.array(3)], ids=["int", "numpy-int", "0d-array"]
)
def test_harch_scalar_lags(lags):
    assert_array_equal(HARCH(lags).lags, [1, 2, 3])


def test_harch_array_lags_arch_model():
    y = np.random.default_rng(5).standard_normal(500)
    res_array = arch_model(y, vol="harch", p=np.array([1, 5])).fit(disp="off")
    res_list = arch_model(y, vol="harch", p=[1, 5]).fit(disp="off")
    assert_allclose(res_array.params, res_list.params)
    assert_allclose(res_array.loglikelihood, res_list.loglikelihood)


@pytest.mark.parametrize(
    "lags",
    [0, -1, [0, 1], [1, -5], [], np.array([0, 1]), np.ones((2, 2), dtype=int)],
    ids=[
        "zero",
        "negative",
        "zero-in-list",
        "negative-in-list",
        "empty",
        "array",
        "2d",
    ],
)
def test_harch_invalid_lags(lags):
    with pytest.raises(ValueError, match="lags must be a positive integer"):
        HARCH(lags)


@pytest.mark.parametrize("mean", [ConstantMean, ZeroMean])
@pytest.mark.parametrize("dist", [Normal, StudentsT])
def test_simulate_params_length(mean, dist):
    # [mu] + [omega, alpha, beta] + [nu]
    nmean = 1 if mean is ConstantMean else 0
    valid = [0.5] * nmean + [0.05, 0.1, 0.8] + ([8.0] if dist is StudentsT else [])
    mod = mean(volatility=GARCH(), distribution=dist(seed=np.random.RandomState(0)))
    sim = mod.simulate(valid, 25, burn=10)
    assert sim.shape == (25, 3)
    if nmean:
        assert_allclose(sim["data"] - 0.5, sim["errors"])
    else:
        assert_allclose(sim["data"], sim["errors"])
    for extra in (1, 2):
        with pytest.raises(ValueError, match=f"Expected {len(valid)}, got "):
            mod.simulate([*valid, *([0.5] * extra)], 25)
    for missing in (1, 2):
        with pytest.raises(ValueError, match=f"Expected {len(valid)}, got "):
            mod.simulate(valid[:-missing], 25)
    with pytest.raises(ValueError, match="Expected"):
        mod.simulate([], 25)


@pytest.mark.parametrize("mean", [ConstantMean, ZeroMean])
def test_simulate_params_shape(mean):
    mod = mean(volatility=GARCH())
    n = 4 if mean is ConstantMean else 3
    params = np.array([0.5, 0.05, 0.1, 0.8])[-n:]
    # a column or row vector is fine
    assert mod.simulate(params[:, None], 10, burn=5).shape[0] == 10
    assert mod.simulate(params[None, :], 10, burn=5).shape[0] == 10
    with pytest.raises(ValueError, match=r"1D|squeezable"):
        mod.simulate(np.tile(params, (2, 2)), 10)


def lagged_regressors(y, lags):
    """Regressors [1, mean of the last l observations for each l] and target"""
    max_lag = max(lags)
    nobs = y.shape[0]
    cols = [np.ones(nobs - max_lag)]
    cols.extend(
        np.array([y[t - lag : t].mean() for t in range(max_lag, nobs)]) for lag in lags
    )
    return np.column_stack(cols), y[max_lag:]


@pytest.fixture(scope="module")
def ar_data():
    rs = np.random.RandomState(8765)
    e = rs.standard_normal(300)
    y = np.empty(300)
    y[0] = e[0]
    for t in range(1, 300):
        y[t] = 0.2 + 0.5 * y[t - 1] + e[t]
    return y


@pytest.mark.parametrize("lags", [[1], [1, 5]], ids=["ar1", "har-1-5"])
def test_unfitted_starting_values_and_resids(ar_data, lags):
    x, target = lagged_regressors(ar_data, lags)
    ols = np.linalg.lstsq(x, target, rcond=None)[0]
    cls = ARX if lags == [1] else HARX
    mod = cls(ar_data, lags=lags[0] if lags == [1] else lags)
    sv = mod.starting_values()
    assert_allclose(sv[: ols.shape[0]], ols, rtol=1e-8)
    params = np.r_[0.1, 0.4, np.zeros(len(lags) - 1)]
    assert_allclose(mod.resids(params), target - x @ params, rtol=1e-10)


def test_unfitted_ls_starting_values():
    rs = np.random.RandomState(98)
    x = rs.standard_normal((100, 2))
    y = 1.0 + x @ np.array([0.5, -0.5]) + rs.standard_normal(100)
    mod = LS(y, x=x)
    design = np.column_stack([np.ones(100), x])
    ols = np.linalg.lstsq(design, y, rcond=None)[0]
    assert_allclose(mod.starting_values()[:3], ols, rtol=1e-8)


def test_unfitted_arx_forecast(ar_data):
    params = np.array([0.2, 0.5, 1.0])
    mod = ARX(ar_data, lags=1)
    fcast = mod.forecast(params, horizon=4)
    # Recursion for the mean of an AR(1)
    expected = []
    last = ar_data[-1]
    for _ in range(4):
        last = 0.2 + 0.5 * last
        expected.append(last)
    assert_allclose(fcast.mean.to_numpy().ravel(), expected, rtol=1e-10)
    assert_allclose(fcast.residual_variance.to_numpy().ravel()[0], 1.0)
    # same as the forecast from a fixed model
    fixed = ARX(ar_data, lags=1).fix(params)
    assert_allclose(fcast.mean.to_numpy(), fixed.forecast(horizon=4).mean.to_numpy())


def test_unfitted_resids_with_hold_back():
    y = np.random.RandomState(7).standard_normal(30)
    mod = HARX(y, lags=[1, 5], hold_back=8)
    # The first 8 observations are held back, as when fitting
    expected = y[8:] - (0.0 + 1.0 * y[7:-1])
    assert_allclose(mod.resids(np.array([0.0, 1.0, 0.0])), expected, atol=1e-12)
    # No impact on the model used for estimation
    assert mod.fit(disp="off").nobs == 22


VOLS = ["ARCH", "GARCH", "EGARCH", "FIGARCH", "APARCH", "FIAPARCH"]


@pytest.mark.parametrize("vol", VOLS)
@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.uint8])
def test_arch_model_numpy_integers(vol, dtype):
    y = np.random.default_rng(10).standard_normal(100)
    order = 1 if vol in ("FIGARCH", "FIAPARCH") else 2
    np_kwargs = {"vol": vol, "p": dtype(order)}
    if vol != "ARCH":
        np_kwargs["q"] = dtype(1)
    if vol in ("GARCH", "EGARCH", "APARCH", "FIAPARCH"):
        np_kwargs["o"] = dtype(1)
    mod = arch_model(y, **np_kwargs)
    ref = arch_model(
        y, **{k: v if k == "vol" else int(v) for k, v in np_kwargs.items()}
    )
    assert str(mod.volatility) == str(ref.volatility)
    assert mod.volatility.num_params == ref.volatility.num_params
    assert mod.volatility.p == order
    assert type(mod.volatility.p) is int


def test_arch_model_numpy_integers_fit_identical():
    y = np.random.default_rng(11).standard_normal(400)
    res = arch_model(y, p=np.int64(1), o=np.int64(1), q=np.int64(1)).fit(disp="off")
    ref = arch_model(y, p=1, o=1, q=1).fit(disp="off")
    assert_allclose(res.params, ref.params)
    assert_allclose(res.loglikelihood, ref.loglikelihood)


def test_arch_model_loop_over_numpy_ints():
    y = np.random.default_rng(12).standard_normal(100)
    for p in np.arange(1, 3):
        assert arch_model(y, p=p).volatility.p == p


@pytest.mark.parametrize("vol", VOLS)
@pytest.mark.parametrize(
    "p", [1.0, "1", [1, 5], None], ids=["float", "str", "list", "none"]
)
def test_arch_model_invalid_p(vol, p):
    y = np.random.default_rng(10).standard_normal(100)
    with pytest.raises(TypeError, match="p must be a scalar int"):
        arch_model(y, vol=vol, p=p)


@pytest.mark.parametrize("vol", ["GARCH", "EGARCH", "APARCH", "FIAPARCH"])
def test_arch_model_invalid_o(vol):
    y = np.random.default_rng(10).standard_normal(100)
    with pytest.raises(TypeError, match="o must be a scalar int"):
        arch_model(y, vol=vol, o=1.0)


@pytest.mark.parametrize("vol", ["GARCH", "EGARCH", "FIGARCH", "APARCH", "FIAPARCH"])
def test_arch_model_invalid_q(vol):
    y = np.random.default_rng(10).standard_normal(100)
    with pytest.raises(TypeError, match="q must be a scalar int"):
        arch_model(y, vol=vol, q=1.5)


def test_arch_model_unused_orders_not_validated():
    y = np.random.default_rng(10).standard_normal(100)
    # ARCH has no o or q, constant variance and HARCH ignore o and q
    assert arch_model(y, vol="ARCH", o=0.5, q=0.5).volatility.p == 1
    arch_model(y, vol="HARCH", p=[1, 5], o=0.5, q=0.5)
    arch_model(y, vol="HARCH", p=np.array([1, 5]))
    arch_model(y, vol="HARCH", p=np.int64(3))


def test_fit_options_not_modified(capsys):
    y = np.random.default_rng(13).standard_t(5, 500)
    mod = arch_model(y)
    options = {"maxiter": 100}
    res = mod.fit(options=options, update_freq=0)
    # disp="final" is the default and is not written to the caller's options
    assert options == {"maxiter": 100}
    capsys.readouterr()
    # A new call with the same options and disp="off" does not print
    res_off = mod.fit(disp="off", options=options)
    assert capsys.readouterr().out == ""
    assert options == {"maxiter": 100}
    assert_allclose(res_off.params, res.params, rtol=1e-6)


def test_fit_options_disp_in_options_respected(capsys):
    y = np.random.default_rng(13).standard_t(5, 500)
    options = {"maxiter": 100, "disp": False}
    arch_model(y).fit(options=options, update_freq=0)
    assert capsys.readouterr().out == ""
    assert options == {"maxiter": 100, "disp": False}


@pytest.mark.parametrize(
    ("process", "params", "expected"),
    [
        pytest.param(GARCH(), [0.1, 0.1, 0.8], lambda v: v, id="garch"),
        pytest.param(ARCH(2), [0.1, 0.1, 0.1], lambda v: v, id="arch"),
        pytest.param(HARCH([1, 5]), [0.1, 0.1, 0.1], lambda v: v, id="harch"),
        pytest.param(EWMAVariance(0.94), [], lambda v: v, id="ewma"),
        pytest.param(
            GARCH(power=1.0), [0.1, 0.1, 0.8], lambda v: v**2, id="garch-power-1"
        ),
        pytest.param(
            GARCH(power=1.5), [0.1, 0.1, 0.8], lambda v: v ** (2 / 1.5), id="power-1.5"
        ),
        pytest.param(
            APARCH(), [0.1, 0.1, 0.1, 0.8, 1.5], lambda v: v ** (2 / 1.5), id="aparch"
        ),
        pytest.param(EGARCH(), [0.1, 0.1, 0.8], np.exp, id="egarch"),
    ],
)
def test_simulate_initial_value_units(process, params, expected):
    # The documented units of initial_value are those of the quantity the
    # process updates: sigma^2, sigma^power, sigma^delta or ln(sigma^2).
    initial_value = 1.7
    rng = np.random.default_rng(0).standard_normal
    _, sigma2 = process.simulate(params, 20, rng, burn=0, initial_value=initial_value)
    assert_allclose(sigma2[0], expected(initial_value))


@pytest.mark.parametrize(
    ("mean", "message"),
    [
        (ConstantMean, "constant mean"),
        (ZeroMean, "zero mean"),
    ],
)
def test_simulate_initial_value_message(mean, message):
    mod = mean(volatility=GARCH())
    params = [0.0, 0.05, 0.1, 0.8][-mod.num_params - 3 :]
    for kwargs in ({"initial_value": 1.0}, {"x": np.ones(10)}):
        with pytest.raises(ValueError, match=f"simulating a {message} process"):
            mod.simulate(params, 10, **kwargs)
