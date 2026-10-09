import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.tsatools import lagmat

from arch._typing import Float64Array
from arch.covariance import kernel as kernel_module
from arch.covariance.kernel import CovarianceEstimate
from arch.covariance.var import PreWhitenedRecolored
from arch.data import default
from arch.tests.covariance.sandwich_results import SANDWICH_LONG_RUN

KERNELS = [
    "Bartlett",
    "Parzen",
    "ParzenCauchy",
    "ParzenGeometric",
    "ParzenRiesz",
    "TukeyHamming",
    "TukeyHanning",
    "TukeyParzen",
    "QuadraticSpectral",
    "Andrews",
    "Gallant",
    "NeweyWest",
]


@pytest.fixture(params=KERNELS)
def kernel(request):
    return request.param


def direct_var(
    x, const: bool, full_order: int, diag_order: int, max_order: int | None = None
) -> tuple[Float64Array, Float64Array]:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    c = int(const)
    nobs, nvar = x.shape
    order = max_order if max_order is not None else max(full_order, diag_order)
    rhs = np.empty((nobs - order, c + nvar * order))
    lhs = np.empty((nobs - order, nvar))
    offset = 0
    if const:
        rhs[:, 0] = 1
        offset = 1
    for i in range(nvar):
        idx = offset + i + nvar * np.arange(order)
        rhs[:, idx], lhs[:, i : i + 1] = lagmat(
            x[:, i : i + 1], order, "both", "sep", False
        )
    idx = []
    if const:
        idx += [0]
    idx += (c + np.arange(full_order * nvar)).tolist()
    idx += [-9999] * (diag_order - full_order)
    locs = np.array(idx, dtype=int)
    diag_start = int(const) + full_order * nvar
    params = np.zeros((nvar, rhs.shape[1]))
    resids = np.empty_like(lhs)
    for i in range(nvar):
        if diag_order > full_order:
            locs[diag_start:] = c + i + nvar * np.arange(full_order, diag_order)
        _rhs = rhs[:, locs]
        if _rhs.shape[1] > 0:
            p = np.linalg.lstsq(_rhs, lhs[:, i : i + 1], rcond=None)[0]
            params[i : i + 1, locs] = p.T
            resids[:, i : i + 1] = lhs[:, i : i + 1] - _rhs @ p
        else:
            # Branch is a workaround of NumPy 1.15
            # TODO: Remove after NumPy 1.15 dropped
            resids[:, i : i + 1] = lhs[:, i : i + 1]
    return params, resids


def direct_ic(
    x,
    ic: str,
    const: bool,
    full_order: int,
    diag_order: int,
    max_order: int | None = None,
) -> float:
    _, resids = direct_var(x, const, full_order, diag_order, max_order)
    nobs, nvar = resids.shape
    sigma = resids.T @ resids / nobs
    ndiag = max(0, diag_order - full_order)
    nparams = (int(const) + full_order * nvar + ndiag) * nvar
    if ic == "aic":
        penalty = 2
    elif ic == "hqc":
        penalty = 2 * np.log(np.log(nobs))
    else:  # bic
        penalty = np.log(nobs)
    _, ld = np.linalg.slogdet(sigma)
    return ld + penalty * nparams / nobs


@pytest.mark.parametrize("const", [True, False])
@pytest.mark.parametrize("full_order", [1, 3])
@pytest.mark.parametrize("diag_order", [3, 5])
@pytest.mark.parametrize("max_order", [None, 10])
@pytest.mark.parametrize("ic", ["aic", "bic", "hqc"])
def test_direct_var(covariance_data, const, full_order, diag_order, max_order, ic):
    direct_ic(covariance_data, ic, const, full_order, diag_order, max_order)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("diagonal", [True, False])
@pytest.mark.parametrize("method", ["aic", "bic", "hqc"])
def test_ic(covariance_data, center, diagonal, method):
    pwrc = PreWhitenedRecolored(
        covariance_data,
        center=center,
        diagonal=diagonal,
        method=method,
        bandwidth=0.0,
    )
    cov = pwrc.cov
    expected_type = (
        np.ndarray if isinstance(covariance_data, np.ndarray) else pd.DataFrame
    )
    assert isinstance(cov.short_run, expected_type)
    expected_max_lag = int(covariance_data.shape[0] ** (1 / 3))
    assert pwrc._max_lag == expected_max_lag
    expected_ics = {}
    for full_order in range(expected_max_lag + 1):
        diag_limit = expected_max_lag + 1 if diagonal else full_order + 1
        if covariance_data.ndim == 1 or covariance_data.shape[1] == 1:
            diag_limit = full_order + 1
        for diag_order in range(full_order, diag_limit):
            key = (full_order, diag_order)
            expected_ics[key] = direct_ic(
                covariance_data,
                method,
                center,
                full_order,
                diag_order,
                max_order=expected_max_lag,
            )
    assert tuple(sorted(pwrc._ics.keys())) == tuple(sorted(expected_ics.keys()))
    for key, value in expected_ics.items():
        assert_allclose(pwrc._ics[key], value)
    expected_order = pd.Series(expected_ics).idxmin()
    assert pwrc._order == expected_order


def theoretical_autocov(
    coefs: list[Float64Array],
    sigma: Float64Array,
    lag: int | list[int],
    terms: int = 600,
) -> Float64Array | list[Float64Array]:
    """
    Gamma_lag = E[x_t x_{t-lag}'] of a VAR with the coefficients and innovation
    covariance, computed from the MA(infinity) representation
    x_t = sum_k Psi_k e_{t-k} so that Gamma_j = sum_k Psi_{k+j} Sigma Psi_k'.
    """
    nvar = sigma.shape[0]
    psi = np.zeros((terms, nvar, nvar))
    psi[0] = np.eye(nvar)
    for k in range(1, terms):
        for i, coef in enumerate(coefs, 1):
            if i <= k:
                psi[k] += coef @ psi[k - i]

    def gamma(j: int) -> Float64Array:
        return np.einsum("kab,bc,kdc->ad", psi[j:], sigma, psi[: terms - j])

    if isinstance(lag, int):
        return gamma(lag)
    return [gamma(j) for j in lag]


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("diagonal", [True, False])
@pytest.mark.parametrize("method", ["aic", "bic", "hqc"])
@pytest.mark.parametrize("lags", [0, 1, 3])
def test_short_long_run(covariance_data, center, diagonal, method, lags):
    pwrc = PreWhitenedRecolored(
        covariance_data,
        center=center,
        diagonal=diagonal,
        method=method,
        lags=lags,
        bandwidth=0.0,
    )
    cov = pwrc.cov
    full_order, diag_order = pwrc._order
    params, resids = direct_var(covariance_data, center, full_order, diag_order)
    nvar = resids.shape[1]
    # The residual covariance divides by all observations in x, not by the
    # number of residuals
    nobs = np.asarray(covariance_data).shape[0]
    resid_cov = resids.T @ resids / nobs
    c = int(center)
    order = max(full_order, diag_order)
    coefs = [params[:, c + i * nvar : c + (i + 1) * nvar] for i in range(order)]
    d = np.eye(nvar) - sum(coefs, np.zeros((nvar, nvar)))
    d_inv = np.linalg.inv(d)
    assert_allclose(cov.long_run, d_inv @ resid_cov @ d_inv.T)
    # Without a kernel, the short run is the variance of x implied by the VAR
    assert_allclose(cov.short_run, theoretical_autocov(coefs, resid_cov, 0))
    # and so the covariance identities hold
    one_sided_strict = np.asarray(cov.one_sided_strict)
    assert_allclose(cov.one_sided, np.asarray(cov.short_run) + one_sided_strict)
    assert_allclose(
        cov.long_run,
        np.asarray(cov.short_run) + one_sided_strict + one_sided_strict.T,
        atol=1e-10,
    )


@pytest.mark.parametrize("force_int", [True, False])
def test_pwrc_attributes(covariance_data, force_int):
    pwrc = PreWhitenedRecolored(covariance_data, force_int=force_int)
    assert isinstance(pwrc.bandwidth_scale, float)
    assert isinstance(pwrc.kernel_const, float)
    assert isinstance(pwrc.rate, float)
    assert isinstance(pwrc._weights(), np.ndarray)
    assert pwrc.force_int == force_int
    expected_type = (
        np.ndarray if isinstance(covariance_data, np.ndarray) else pd.DataFrame
    )
    assert isinstance(pwrc.cov.short_run, expected_type)
    assert isinstance(pwrc.cov.long_run, expected_type)
    assert isinstance(pwrc.cov.one_sided, expected_type)
    assert isinstance(pwrc.cov.one_sided_strict, expected_type)


@pytest.mark.parametrize("sample_autocov", [True, False])
def test_data(covariance_data, sample_autocov, kernel):
    pwrc = PreWhitenedRecolored(
        covariance_data, sample_autocov=sample_autocov, kernel=kernel, bandwidth=0.0
    )
    assert isinstance(pwrc.cov, CovarianceEstimate)


def test_pwrc_errors():
    x = np.random.default_rng(0).standard_normal((500, 2))
    with pytest.raises(ValueError, match="lags must be a"):
        PreWhitenedRecolored(x, lags=-1)
    with pytest.raises(ValueError, match="lags must be a"):
        PreWhitenedRecolored(x, lags=np.array([2]))
    with pytest.raises(ValueError, match="lags must be a"):
        PreWhitenedRecolored(x, lags=3.5)


def test_pwrc_warnings():
    x = np.random.default_rng(1).standard_normal((9, 5))
    with pytest.warns(RuntimeWarning, match="The maximum number of lags is 0"):
        assert isinstance(PreWhitenedRecolored(x).cov, CovarianceEstimate)


@pytest.mark.parametrize("method", ["aic", "hqc", "bic"])
def test_method_case(var_data, method):
    lower = PreWhitenedRecolored(var_data, method=method)
    upper = PreWhitenedRecolored(var_data, method=method.upper())
    assert_allclose(upper.cov.long_run, lower.cov.long_run)
    assert upper._order == lower._order
    assert upper._ics == lower._ics


@pytest.mark.parametrize("method", ["unknown", "t-stat", "", None, 1])
def test_unknown_method(var_data, method):
    # Previously any value other than "aic" and "hqc" used BIC
    with pytest.raises(ValueError, match="method must be one of"):
        PreWhitenedRecolored(var_data, method=method)


def test_unknown_kernel(covariance_data):
    with pytest.raises(ValueError, match="kernel is not a known"):
        PreWhitenedRecolored(covariance_data, kernel="unknown")


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("bandwidth", [2.0, 7.5])
def test_recolored_kernel_long_run(covariance_data, center, bandwidth, kernel):
    # Andrews & Monahan (1992): the long run is D Omega_e D' where Omega_e is
    # the kernel long-run covariance of the VAR residuals and
    # D = (I - A_1 - ... - A_p)^-1. Computed here without the estimator.
    lags = 2
    x = np.asarray(covariance_data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    nobs_full, nvar = x.shape
    lhs = x[lags:]
    rhs = [x[lags - i : nobs_full - i] for i in range(1, lags + 1)]
    if center:
        rhs = [np.ones((nobs_full - lags, 1))] + rhs
    rhs = np.hstack(rhs)
    params = np.linalg.lstsq(rhs, lhs, rcond=None)[0].T
    resids = lhs - rhs @ params.T
    coef_sum = np.zeros((nvar, nvar))
    c = int(center)
    for i in range(lags):
        coef_sum += params[:, c + i * nvar : c + (i + 1) * nvar]
    d = np.linalg.inv(np.eye(nvar) - coef_sum)
    kern_est = getattr(kernel_module, kernel)
    omega_e = kern_est(resids, bandwidth=bandwidth, center=False).cov.long_run
    # The kernel divides by the number of residuals, the estimator divides by
    # the number of observations in x
    scale = resids.shape[0] / nobs_full
    expected = scale * d @ omega_e @ d.T

    pwrc = PreWhitenedRecolored(
        covariance_data, lags=lags, kernel=kernel, bandwidth=bandwidth, center=center
    )
    cov = pwrc.cov
    assert_allclose(np.asarray(cov.long_run), expected, rtol=1e-8, atol=1e-10)
    # The short run is the variance of x implied by the VAR for all kernels
    coefs = [params[:, c + i * nvar : c + (i + 1) * nvar] for i in range(lags)]
    resid_cov = resids.T @ resids / nobs_full
    assert_allclose(
        np.asarray(cov.short_run),
        theoretical_autocov(coefs, resid_cov, 0),
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.parametrize("center", [True, False])
def test_var_hac_kernel_none(covariance_data, center):
    # kernel=None is VAR-HAC: the long run is D Sigma_e D' where Sigma_e is the
    # residual covariance, with no kernel applied to the residuals.
    lags = 2
    x = np.asarray(covariance_data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    nobs_full, nvar = x.shape
    lhs = x[lags:]
    rhs = [x[lags - i : nobs_full - i] for i in range(1, lags + 1)]
    if center:
        rhs = [np.ones((nobs_full - lags, 1))] + rhs
    rhs = np.hstack(rhs)
    params = np.linalg.lstsq(rhs, lhs, rcond=None)[0].T
    resids = lhs - rhs @ params.T
    coef_sum = np.zeros((nvar, nvar))
    c = int(center)
    for i in range(lags):
        coef_sum += params[:, c + i * nvar : c + (i + 1) * nvar]
    d = np.linalg.inv(np.eye(nvar) - coef_sum)
    sigma_e = resids.T @ resids / nobs_full
    expected = d @ sigma_e @ d.T

    pwrc = PreWhitenedRecolored(covariance_data, lags=lags, kernel=None, center=center)
    assert_allclose(np.asarray(pwrc.cov.long_run), expected, rtol=1e-8, atol=1e-10)
    assert pwrc.kernel_const == 1.0
    assert pwrc.bandwidth_scale == 0.0
    assert pwrc.rate == 0.0
    assert_allclose(pwrc._weights(), np.ones(1))

    pwrc_zero = PreWhitenedRecolored(
        covariance_data, lags=lags, kernel=None, bandwidth=0.0, center=center
    )
    assert_allclose(pwrc_zero.cov.long_run, pwrc.cov.long_run)


def test_zero_lag_kernel():
    x = np.random.RandomState(0).standard_normal((250, 2))
    cov = kernel_module.ZeroLag(x).cov
    assert_allclose(cov.long_run, cov.short_run)
    assert_allclose(cov.one_sided_strict, np.zeros((2, 2)))


def test_kernel_none_bandwidth_error():
    x = np.random.default_rng(2).standard_normal((500, 2))
    with pytest.raises(ValueError, match="bandwidth must be None"):
        PreWhitenedRecolored(x, kernel=None, bandwidth=3.0)


def test_nonstationary_var_error():
    rs = np.random.RandomState(0)
    e = rs.standard_normal((250, 2))
    x = np.zeros_like(e)
    for t in range(1, x.shape[0]):
        x[t] = 1.05 * x[t - 1] + e[t]
    pwrc = PreWhitenedRecolored(x, lags=1)
    with pytest.raises(ValueError, match="not compatible with covariance"):
        _ = pwrc.cov


@pytest.mark.parametrize("center", [True, False])
def test_sample_autocov_center(covariance_data, center):
    pwrc = PreWhitenedRecolored(
        covariance_data, lags=2, sample_autocov=True, center=center
    )
    assert isinstance(pwrc.cov, CovarianceEstimate)


VAR_COEFS = (
    np.array([[0.5, 0.2, 0.0], [0.0, 0.4, 0.3], [0.1, 0.0, 0.3]]),
    np.array([[0.1, 0.0, 0.2], [0.0, 0.1, 0.0], [0.0, 0.1, 0.1]]),
    np.array([[0.05, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.05]]),
)
INNOVATION_SCALE = np.array([1.0, 2.0, 0.5])


def simulate_var(
    nobs: int, coefs: tuple[Float64Array, ...] = VAR_COEFS, seed: int = 20261009
) -> Float64Array:
    """Simulate a Gaussian VAR with asymmetric coefficient matrices"""
    rng = np.random.default_rng(seed)
    burn = 200
    nvar = coefs[0].shape[0]
    eps = rng.standard_normal((nobs + burn, nvar)) * INNOVATION_SCALE
    x = np.zeros_like(eps)
    for t in range(len(coefs), nobs + burn):
        x[t] = eps[t]
        for j, coef in enumerate(coefs):
            x[t] += coef @ x[t - 1 - j]
    return x[burn:]


@pytest.fixture(scope="module")
def var_data() -> Float64Array:
    return simulate_var(500)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("nlag", [1, 2, 3])
def test_sample_cov_stacked(center, nlag):
    # The stacked vector is [x_t, x_{t-1}, ...] so that block (r, c) of its
    # covariance is E[x_{t-r} x_{t-c}'] = Gamma_{c-r}. The cross-covariances
    # are not symmetric so a transposed layout is detectable.
    x = simulate_var(4000)
    nobs, nvar = x.shape
    xc = x - x.mean(0) if center else x

    def gamma(j: int) -> Float64Array:
        # E[x_t x_{t-j}'] using the definition
        if j < 0:
            return gamma(-j).T
        return sum(np.outer(xc[t], xc[t - j]) for t in range(j, nobs)) / nobs

    expected = np.block([[gamma(c - r) for c in range(nlag)] for r in range(nlag)])
    pwrc = PreWhitenedRecolored(x, center=center)
    assert_allclose(pwrc._estimate_sample_cov(nvar, nlag), expected, atol=1e-12)
    # Same up to end effects as the second moment of the stacked data. The
    # transposed layout differs by about 0.3.
    stacked = np.hstack([xc[nlag - 1 - j : nobs - j] for j in range(nlag)])
    assert_allclose(expected, stacked.T @ stacked / stacked.shape[0], atol=0.02)


def residual_kernel(x, kernel, center, order, **kwargs):
    # The kernel estimator applied to the residuals of a directly estimated VAR
    full_order, diag_order = (order, order) if np.isscalar(order) else order
    _, resids = direct_var(x, center, full_order, diag_order)
    return getattr(kernel_module, kernel)(resids, center=False, **kwargs)


@pytest.mark.parametrize("force_int", [True, False])
@pytest.mark.parametrize("center", [True, False])
def test_bandwidth_is_residual_bandwidth(var_data, kernel, center, force_int):
    # The kernel runs on the VAR residuals, so the bandwidth is the optimal
    # bandwidth estimated using the residuals and not using x
    lags = 2
    expected = residual_kernel(var_data, kernel, center, lags, force_int=force_int)
    pwrc = PreWhitenedRecolored(
        var_data, lags=lags, kernel=kernel, center=center, force_int=force_int
    )
    assert_allclose(pwrc.bandwidth, expected.bandwidth)
    assert_allclose(pwrc.opt_bandwidth, expected.opt_bandwidth)
    assert_allclose(pwrc.kernel_weights, expected.kernel_weights)
    assert f"Bandwidth: {pwrc.bandwidth}\n" in str(pwrc)
    assert "Automatic Bandwidth: True" in str(pwrc)
    if force_int:
        assert pwrc.bandwidth == np.ceil(pwrc.bandwidth)


def test_bandwidth_differs_from_x_bandwidth(var_data):
    # Guards against reporting the bandwidth estimated using x
    lags = 2
    x_kernel = kernel_module.Bartlett(var_data)
    resid_kernel = residual_kernel(var_data, "Bartlett", True, lags)
    pwrc = PreWhitenedRecolored(var_data, lags=lags)
    assert abs(x_kernel.bandwidth - resid_kernel.bandwidth) > 1.0
    assert pwrc.bandwidth != pytest.approx(x_kernel.bandwidth, abs=0.5)
    assert pwrc.bandwidth == pytest.approx(resid_kernel.bandwidth)


@pytest.mark.parametrize("force_int", [True, False])
@pytest.mark.parametrize("bandwidth", [0.0, 2.5, 7.0])
def test_bandwidth_user_provided(var_data, bandwidth, force_int):
    pwrc = PreWhitenedRecolored(
        var_data, lags=1, bandwidth=bandwidth, force_int=force_int
    )
    expected = np.ceil(bandwidth) if force_int else bandwidth
    assert pwrc.bandwidth == expected
    assert "Automatic Bandwidth: False" in str(pwrc)
    # Weights are those of the bandwidth that is reported
    direct = residual_kernel(
        var_data, "Bartlett", True, 1, bandwidth=bandwidth, force_int=force_int
    )
    assert_allclose(pwrc.kernel_weights, direct.kernel_weights)


def test_bandwidth_access_order(var_data):
    # Reading the bandwidth before cov does not change cov
    first = PreWhitenedRecolored(var_data, lags=2)
    bandwidth = first.bandwidth
    second = PreWhitenedRecolored(var_data, lags=2)
    assert_allclose(first.cov.long_run, second.cov.long_run)
    assert second.bandwidth == bandwidth


@pytest.mark.parametrize("method", ["aic", "bic"])
def test_bandwidth_automatic_order(var_data, method):
    # The bandwidth is for the residuals of the VAR with the selected order
    pwrc = PreWhitenedRecolored(var_data, kernel="Parzen", method=method)
    bandwidth = pwrc.bandwidth
    assert pwrc._order != (0, 0)
    expected = residual_kernel(var_data, "Parzen", True, pwrc._order)
    assert_allclose(bandwidth, expected.bandwidth)


def test_bandwidth_nonstationary_var():
    # The bandwidth only needs the VAR residuals
    rs = np.random.RandomState(0)
    e = rs.standard_normal((250, 2))
    x = np.zeros_like(e)
    for t in range(1, x.shape[0]):
        x[t] = 1.05 * x[t - 1] + e[t]
    pwrc = PreWhitenedRecolored(x, lags=1)
    assert pwrc.bandwidth > 0
    with pytest.raises(ValueError, match="not compatible with covariance"):
        _ = pwrc.cov


@pytest.mark.parametrize("bandwidth", [None, 0.0])
def test_bandwidth_kernel_none(var_data, bandwidth):
    pwrc = PreWhitenedRecolored(var_data, lags=2, kernel=None, bandwidth=bandwidth)
    assert pwrc.bandwidth == 0.0
    assert pwrc.opt_bandwidth == 0.0
    assert_allclose(pwrc.kernel_weights, np.ones(1))
    assert "Bandwidth: 0.0\n" in str(pwrc)
    assert "Automatic Bandwidth: False" in str(pwrc)
    assert kernel_module.ZeroLag(var_data).bandwidth == 0.0
    assert kernel_module.ZeroLag(var_data, force_int=True).bandwidth == 0.0


@pytest.mark.parametrize("sample_autocov", [True, False])
@pytest.mark.parametrize(("kernel", "bandwidth"), [(None, None), ("Bartlett", 5.0)])
@pytest.mark.parametrize("lags", [0, 2])
@pytest.mark.parametrize("df_adjust", [1, 3, 10])
def test_df_adjust(var_data, df_adjust, lags, kernel, bandwidth, sample_autocov):
    # Every covariance divides by the number of observations in x less
    # df_adjust, as the other covariance estimators do.
    kwargs = {
        "lags": lags,
        "kernel": kernel,
        "bandwidth": bandwidth,
        "sample_autocov": sample_autocov,
    }
    base = PreWhitenedRecolored(var_data, **kwargs)
    adjusted = PreWhitenedRecolored(var_data, df_adjust=df_adjust, **kwargs)
    nobs = var_data.shape[0]
    factor = nobs / (nobs - df_adjust)
    for attr in ("long_run", "short_run", "one_sided", "one_sided_strict"):
        assert_allclose(getattr(adjusted.cov, attr), factor * getattr(base.cov, attr))
    assert adjusted.bandwidth == base.bandwidth
    assert f"Degree of Freedom Adjustment: {df_adjust}" in str(adjusted)


@pytest.mark.parametrize("df_adjust", [-1, 500, 600])
def test_df_adjust_errors(var_data, df_adjust):
    with pytest.raises(ValueError, match=r"df_adjust|Degrees of freedom"):
        PreWhitenedRecolored(var_data, df_adjust=df_adjust)


def fitted_var(x, lags, center):
    """VAR coefficients and the residual covariance dividing by all of x"""
    params, resids = direct_var(x, center, lags, lags)
    nvar = resids.shape[1]
    c = int(center)
    coefs = [params[:, c + i * nvar : c + (i + 1) * nvar] for i in range(lags)]
    return coefs, resids.T @ resids / np.asarray(x).shape[0]


@pytest.mark.parametrize(("kernel", "bandwidth"), [(None, None), ("Parzen", 0.0)])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("lags", [1, 2, 3])
def test_one_sided_model_implied(var_data, lags, center, kernel, bandwidth):
    # One-sided covariances are sums of the autocovariances of the VAR:
    # os_strict = sum_{j>=1} Gamma_j, os = sum_{j>=0} Gamma_j. The Gamma_j come
    # from the MA(infinity) representation of the estimated VAR.
    pwrc = PreWhitenedRecolored(
        var_data, lags=lags, kernel=kernel, bandwidth=bandwidth, center=center
    )
    cov = pwrc.cov
    coefs, resid_cov = fitted_var(var_data, lags, center)
    gamma = theoretical_autocov(coefs, resid_cov, list(range(300)))
    one_sided_strict = sum(gamma[1:], np.zeros_like(resid_cov))
    assert_allclose(cov.short_run, gamma[0], rtol=1e-9, atol=1e-10)
    assert_allclose(cov.one_sided_strict, one_sided_strict, rtol=1e-9, atol=1e-10)
    assert_allclose(cov.one_sided, gamma[0] + one_sided_strict, rtol=1e-9, atol=1e-10)
    assert_allclose(
        cov.long_run, gamma[0] + one_sided_strict + one_sided_strict.T, rtol=1e-9
    )


def test_one_sided_scalar_ar1():
    # x_t = a x_{t-1} + e_t has Gamma_0 = s2 / (1 - a^2), Gamma_j = a^j Gamma_0
    # so that os_strict = a Gamma_0 / (1 - a), and long run = s2 / (1 - a)^2.
    a, nobs = 0.6, 100000
    rng = np.random.default_rng(1234)
    eps = rng.standard_normal(nobs + 200)
    x = np.zeros(nobs + 200)
    for t in range(1, x.shape[0]):
        x[t] = a * x[t - 1] + eps[t]
    x = x[200:]
    pwrc = PreWhitenedRecolored(x, lags=1, kernel=None, center=False)
    cov = pwrc.cov
    coefs, resid_cov = fitted_var(x, 1, False)
    a_hat = coefs[0][0, 0]
    assert a_hat == pytest.approx(a, abs=0.01)
    gamma0 = resid_cov[0, 0] / (1 - a_hat**2)
    assert_allclose(np.squeeze(cov.short_run), gamma0)
    assert_allclose(np.squeeze(cov.one_sided_strict), a_hat * gamma0 / (1 - a_hat))
    assert_allclose(np.squeeze(cov.one_sided), gamma0 / (1 - a_hat))
    assert_allclose(np.squeeze(cov.long_run), resid_cov[0, 0] / (1 - a_hat) ** 2)


@pytest.mark.parametrize("sample_autocov", [True, False])
@pytest.mark.parametrize("kernel", ["Bartlett", "QuadraticSpectral"])
@pytest.mark.parametrize("center", [True, False])
def test_one_sided_identity_with_kernel(var_data, center, kernel, sample_autocov):
    # one_sided = short_run + one_sided_strict holds for every configuration,
    # also when the kernel and sample autocovariance are used
    pwrc = PreWhitenedRecolored(
        var_data,
        lags=2,
        kernel=kernel,
        bandwidth=6.0,
        center=center,
        sample_autocov=sample_autocov,
    )
    cov = pwrc.cov
    assert_allclose(cov.one_sided, cov.short_run + cov.one_sided_strict, atol=1e-12)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("lags", [1, 2, 3])
def test_sample_autocov_values(var_data, lags, center):
    # With sample_autocov, Gamma_0 is the sample covariance of the stacked
    # [x_t, ..., x_{t-P+1}], so that short_run is the sample variance of x and
    # os_strict is the upper-left block of sum_{m>=1} F^m Gamma_0.
    nobs, nvar = var_data.shape
    xc = var_data - var_data.mean(0) if center else var_data

    def gamma(j: int) -> Float64Array:
        if j < 0:
            return gamma(-j).T
        return sum(np.outer(xc[t], xc[t - j]) for t in range(j, nobs)) / nobs

    stacked_cov = np.block([[gamma(c - r) for c in range(lags)] for r in range(lags)])
    coefs, _ = fitted_var(var_data, lags, center)
    comp = np.zeros((nvar * lags, nvar * lags))
    comp[:nvar] = np.hstack(coefs)
    comp[nvar:, :-nvar] = np.eye(nvar * (lags - 1))
    total = np.zeros_like(comp)
    power = np.eye(nvar * lags)
    for _ in range(500):
        power = power @ comp
        total += power @ stacked_cov
    expected_strict = total[:nvar, :nvar]

    pwrc = PreWhitenedRecolored(
        var_data, lags=lags, center=center, kernel=None, sample_autocov=True
    )
    cov = pwrc.cov
    assert_allclose(cov.short_run, gamma(0), rtol=1e-9)
    assert_allclose(cov.one_sided_strict, expected_strict, rtol=1e-9, atol=1e-10)
    assert_allclose(cov.one_sided, gamma(0) + expected_strict, rtol=1e-9, atol=1e-10)
    # The long run does not depend on the source of the autocovariances
    model = PreWhitenedRecolored(var_data, lags=lags, center=center, kernel=None)
    assert_allclose(cov.long_run, model.cov.long_run)
    assert not np.allclose(cov.one_sided_strict, model.cov.one_sided_strict)


def stable_companion(nvar: int, nlag: int, seed: int) -> Float64Array:
    rng = np.random.default_rng(seed)
    dim = nvar * nlag
    comp = np.zeros((dim, dim))
    comp[:nvar] = rng.standard_normal((nvar, dim)) * 0.8 / dim
    comp[nvar:, :-nvar] = np.eye(dim - nvar)
    assert np.abs(np.linalg.eigvals(comp)).max() < 1
    return comp


@pytest.mark.parametrize(("nvar", "nlag"), [(1, 1), (1, 4), (2, 3), (3, 2), (3, 12)])
def test_estimate_model_cov(nvar, nlag):
    # Gamma = F Gamma F' + Sigma has the closed form vec(Gamma) =
    # (I - F kron F)^{-1} vec(Sigma) which is too large to use for big VARs
    comp = stable_companion(nvar, nlag, 100 * nvar + nlag)
    rng = np.random.default_rng(nvar)
    root = rng.standard_normal((nvar, nvar))
    resid_cov = root @ root.T + np.eye(nvar)
    dim = nvar * nlag
    sigma = np.zeros((dim, dim))
    sigma[:nvar, :nvar] = resid_cov
    vec = np.linalg.solve(np.eye(dim**2) - np.kron(comp, comp), sigma.ravel())
    pwrc = PreWhitenedRecolored(np.zeros((10, nvar)), lags=1)
    result = pwrc._estimate_model_cov(nvar, nlag, comp, resid_cov)
    assert_allclose(result, vec.reshape(dim, dim), atol=1e-12)
    assert_allclose(result, result.T, atol=0, rtol=0)


def test_estimate_model_cov_large_var():
    # 120 by 120 companion matrix, which a Kronecker product formulation
    # cannot handle, satisfies the discrete Lyapunov equation
    nvar, nlag = 3, 40
    comp = stable_companion(nvar, nlag, 11)
    resid_cov = np.diag([1.0, 2.0, 0.5])
    result = PreWhitenedRecolored._estimate_model_cov(nvar, nlag, comp, resid_cov)
    sigma = np.zeros_like(comp)
    sigma[:nvar, :nvar] = resid_cov
    assert_allclose(result, comp @ result @ comp.T + sigma, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(result) > 0)


def test_large_order_runs():
    # A VAR(30) with 3 series has a 90 by 90 companion form
    x = simulate_var(1500)
    cov = PreWhitenedRecolored(x, lags=30).cov
    assert np.all(np.isfinite(np.asarray(cov.long_run)))
    assert_allclose(cov.one_sided, cov.short_run + cov.one_sided_strict)


@pytest.fixture(scope="module")
def yield_changes() -> pd.DataFrame:
    # Real, weakly dependent data with a stationary VAR
    changes = default.load().diff().dropna()
    return changes - changes.mean()


@pytest.mark.parametrize("df_adjust", [0, 2])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize(
    ("kernel", "bandwidth"),
    [("bartlett", 6.0), ("parzen", 8.0), ("quadratic-spectral", 5.0), (None, None)],
)
def test_sandwich_reference(yield_changes, order, kernel, bandwidth, df_adjust):
    # R sandwich::kernHAC (meatHAC for kernel=None) with prewhite=order. The
    # values in R use R's kernel weights, VAR and recoloring, and adjust=TRUE
    # is df_adjust=2, the number of series. See sandwich_results.py.
    expected = np.array(SANDWICH_LONG_RUN[(bool(df_adjust), order, kernel, bandwidth)])
    pwrc = PreWhitenedRecolored(
        yield_changes,
        lags=order,
        kernel=kernel,
        bandwidth=bandwidth,
        center=False,
        df_adjust=df_adjust,
    )
    assert_allclose(pwrc.cov.long_run, expected, rtol=1e-10)
    assert list(pwrc.cov.long_run.columns) == ["AAA", "BAA"]


@pytest.mark.parametrize("kind", ["ndarray", "frame", "series"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("kernel", ["bartlett", None])
def test_non_finite_x(var_data, kind, bad, kernel):
    x = var_data.copy()
    x[17, 1] = bad
    if kind == "frame":
        x = pd.DataFrame(x, columns=["a", "b", "c"])
    elif kind == "series":
        x = pd.Series(x[:, 1], name="b")
    with pytest.raises(ValueError, match="x must not contain NaN or infinite"):
        PreWhitenedRecolored(x, lags=1, kernel=kernel)
    with pytest.raises(ValueError, match="x must not contain NaN or infinite"):
        PreWhitenedRecolored(x)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize(("nobs", "nvar"), [(250, 2), (60, 3), (40, 1), (12, 4)])
def test_lags_limit(nobs, nvar, center):
    # A VAR(P) has nobs - P observations and P * nvar + center parameters per
    # equation, and needs more observations than parameters
    x = np.random.default_rng(nobs).standard_normal((nobs, nvar))
    feasible = [p for p in range(nobs) if nobs - p - (p * nvar + int(center)) > 0]
    largest = max(feasible)
    assert feasible == list(range(largest + 1))
    PreWhitenedRecolored(x, lags=largest, center=center)
    message = f"lags must be at most {largest} when x has {nobs} observations"
    with pytest.raises(ValueError, match=message):
        PreWhitenedRecolored(x, lags=largest + 1, center=center)


@pytest.mark.parametrize("lags", [300, 250, 249, 125])
def test_lags_too_large(lags):
    # Previously "negative dimensions are not allowed" or a 458 GiB allocation
    x = np.random.default_rng(0).standard_normal((250, 2))
    with pytest.raises(ValueError, match="lags must be at most 82 when x has 250"):
        PreWhitenedRecolored(x, lags=lags)


@pytest.mark.parametrize("center", [True, False])
def test_lags_limit_estimates(center):
    # The VAR with the largest allowed order can be estimated, with a
    # positive number of degrees of freedom in each equation
    x = simulate_var(60)
    pwrc = PreWhitenedRecolored(x, center=center)
    largest = pwrc._largest_lag()
    var_mod, _ = PreWhitenedRecolored(x, lags=largest, center=center)._setup()
    nobs, nvar = x.shape
    assert var_mod.resids.shape == (nobs - largest, nvar)
    assert var_mod.params.shape[1] == int(center) + largest * nvar
    assert nobs - largest > int(center) + largest * nvar


@pytest.mark.parametrize("lags", ["2", True, False, np.inf, np.nan, -1, 2.5, [2]])
def test_lags_invalid(var_data, lags):
    with pytest.raises(ValueError, match="lags must be a non-negative integer"):
        PreWhitenedRecolored(var_data, lags=lags)


@pytest.mark.parametrize(
    "lags",
    [2, 2.0, np.int64(2), np.int8(2), np.float64(2.0)],
    ids=["int", "float", "int64", "int8", "float64"],
)
def test_lags_valid_types(var_data, lags):
    expected = PreWhitenedRecolored(var_data, lags=2).cov.long_run
    assert_allclose(PreWhitenedRecolored(var_data, lags=lags).cov.long_run, expected)


@pytest.mark.parametrize("max_lag", ["2", True, np.inf, np.nan, -1, 2.5, [2]])
def test_max_lag_invalid(var_data, max_lag):
    with pytest.raises(ValueError, match="max_lag must be a non-negative integer"):
        PreWhitenedRecolored(var_data, max_lag=max_lag)


def test_max_lag_limited():
    # A larger max_lag is limited to what can be estimated instead of
    # producing a VAR with more parameters than observations
    x = np.random.default_rng(0).standard_normal((250, 2))
    pwrc = PreWhitenedRecolored(x, max_lag=300, diagonal=False)
    assert pwrc._select_lags()[0] <= 82
    assert pwrc._max_lag == 82
    pwrc = PreWhitenedRecolored(x, max_lag=5, diagonal=False)
    pwrc._select_lags()
    assert pwrc._max_lag == 5
    pwrc = PreWhitenedRecolored(x, max_lag=0)
    assert pwrc._select_lags() == (0, 0)
    assert pwrc._max_lag == 0


def test_default_max_lag_limited_small_sample():
    # Default max_lag of int(3 ** (1 / 3)) = 1 needs more observations than
    # parameters; with 3 observations and a constant it is not estimable
    x = np.random.default_rng(0).standard_normal((3, 1))
    pwrc = PreWhitenedRecolored(x)
    with pytest.warns(RuntimeWarning, match="The maximum number of lags is 0"):
        order = pwrc._select_lags()
    assert order == (0, 0)


def simulate_diagonal_var(nobs: int = 1500) -> Float64Array:
    # Own-lag dependence of series 0 at lag 1 and of series 1 at lag 3, so that
    # diagonal lags are selected: (0, 3)
    rng = np.random.default_rng(3)
    eps = rng.standard_normal((nobs + 100, 3))
    x = np.zeros_like(eps)
    for t in range(3, nobs + 100):
        x[t] = eps[t] + np.array([0.5, 0.0, 0.0]) * x[t - 1]
        x[t] += np.array([0.0, 0.3, 0.0]) * x[t - 3]
    return x[100:]


def as_kind(x, kind):
    index = pd.date_range("2001-01-01", periods=x.shape[0], freq="D")
    if kind == "frame":
        return pd.DataFrame(x, index=index, columns=["a", "b", "c"])
    elif kind == "series":
        return pd.Series(x[:, 1], index=index, name="b")
    return x


@pytest.mark.parametrize("kind", ["ndarray", "frame", "series"])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("lags", [0, 1, 2, 3])
def test_resid(var_data, lags, center, kind):
    # Compared to the residuals of a VAR estimated directly
    x = as_kind(var_data, kind)
    pwrc = PreWhitenedRecolored(x, lags=lags, center=center)
    resid = pwrc.resid
    _, expected = direct_var(x, center, lags, lags)
    assert_allclose(resid, expected, atol=1e-12)
    assert resid.shape == (var_data.shape[0] - lags, expected.shape[1])
    if kind == "ndarray":
        assert isinstance(resid, np.ndarray)
    else:
        assert isinstance(resid, pd.DataFrame)
        assert resid.index.equals(x.index[lags:])
        assert list(resid.columns) == list(pd.DataFrame(x).columns)


@pytest.mark.parametrize("center", [True, False])
def test_resid_normal_equations(var_data, center):
    # OLS residuals are orthogonal to the regressors, and have mean zero if
    # there is a constant
    lags = 2
    nobs = var_data.shape[0]
    resid = PreWhitenedRecolored(var_data, lags=lags, center=center).resid
    regressors = [var_data[lags - i : nobs - i] for i in range(1, lags + 1)]
    if center:
        regressors = [np.ones((nobs - lags, 1))] + regressors
    regressors = np.hstack(regressors)
    assert_allclose(regressors.T @ resid, 0, atol=1e-8)
    if center:
        assert_allclose(resid.mean(0), 0, atol=1e-12)


def test_resid_order_zero(var_data):
    centered = PreWhitenedRecolored(var_data, lags=0).resid
    assert_allclose(centered, var_data - var_data.mean(0))
    uncentered = PreWhitenedRecolored(var_data, lags=0, center=False).resid
    assert_allclose(uncentered, var_data)


def test_resid_copy(var_data):
    pwrc = PreWhitenedRecolored(var_data, lags=2)
    resid = pwrc.resid
    resid[:] = 0.0
    assert np.all(pwrc.resid != 0.0)
    assert_allclose(
        pwrc.cov.long_run, PreWhitenedRecolored(var_data, lags=2).cov.long_run
    )


@pytest.mark.parametrize("lags", [0, 1, 4])
def test_order_provided(var_data, lags):
    order = PreWhitenedRecolored(var_data, lags=lags).order
    assert order == (lags, lags)
    assert all(type(value) is int for value in order)


@pytest.mark.parametrize("method", ["aic", "bic", "hqc"])
def test_order_selected(var_data, method):
    pwrc = PreWhitenedRecolored(var_data, method=method, diagonal=False)
    # Available before the covariance, and the same afterwards
    order = pwrc.order
    assert pwrc.order == order
    _ = pwrc.cov
    assert pwrc.order == order
    assert order[0] == order[1]
    ics = {p: direct_ic(var_data, method, True, p, p, pwrc._max_lag) for p in range(8)}
    assert order[0] == min(ics, key=ics.get)
    assert all(type(value) is int for value in order)


def test_order_diagonal():
    x = simulate_diagonal_var()
    pwrc = PreWhitenedRecolored(x)
    assert pwrc.order == (0, 3)
    assert all(type(value) is int for value in pwrc.order)
    # Residuals are those of a VAR with diagonal lags 1 to 3 only
    _, expected = direct_var(x, True, 0, 3)
    assert_allclose(pwrc.resid, expected, atol=1e-12)
    assert pwrc.resid.shape == (x.shape[0] - 3, 3)
    # which is also the case when the VAR is not restricted to diagonal lags
    unrestricted = PreWhitenedRecolored(x, diagonal=False)
    assert unrestricted.order[0] == unrestricted.order[1]


def test_order_single_series(var_data):
    order = PreWhitenedRecolored(var_data[:, 1]).order
    assert order[0] == order[1]


@pytest.mark.parametrize("kind", ["ndarray", "frame", "series"])
@pytest.mark.parametrize("df_adjust", [0, 3])
@pytest.mark.parametrize("lags", [0, 2])
def test_resid_cov(var_data, lags, df_adjust, kind):
    x = as_kind(var_data, kind)
    pwrc = PreWhitenedRecolored(x, lags=lags, df_adjust=df_adjust)
    resid_cov = pwrc.resid_cov
    _, resids = direct_var(x, True, lags, lags)
    # divides by the number of observations in x less df_adjust
    expected = resids.T @ resids / (var_data.shape[0] - df_adjust)
    assert_allclose(resid_cov, expected, atol=1e-12)
    if kind == "ndarray":
        assert isinstance(resid_cov, np.ndarray)
    else:
        assert isinstance(resid_cov, pd.DataFrame)
        columns = list(pd.DataFrame(x).columns)
        assert list(resid_cov.columns) == columns
        assert list(resid_cov.index) == columns
    if lags == 0:
        # no VAR so the residual covariance is the short run
        assert_allclose(resid_cov, pwrc.cov.short_run, atol=1e-12)


@pytest.mark.parametrize(("kernel", "bandwidth"), [(None, None), ("Parzen", 0.0)])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("lags", [1, 2, 3])
def test_resid_cov_recolored(var_data, lags, center, kernel, bandwidth):
    # Without a kernel, long_run is the residual covariance recolored
    pwrc = PreWhitenedRecolored(
        var_data, lags=lags, kernel=kernel, bandwidth=bandwidth, center=center
    )
    coefs, _ = fitted_var(var_data, lags, center)
    d_inv = np.linalg.inv(np.eye(var_data.shape[1]) - sum(coefs))
    assert_allclose(pwrc.cov.long_run, d_inv @ pwrc.resid_cov @ d_inv.T)


@pytest.mark.parametrize("center", [True, False])
def test_resid_cov_var1_lyapunov(var_data, center):
    # For a VAR(1), Gamma_0 = A Gamma_0 A' + Sigma
    pwrc = PreWhitenedRecolored(var_data, lags=1, center=center)
    coefs, _ = fitted_var(var_data, 1, center)
    gamma0 = np.asarray(pwrc.cov.short_run)
    assert_allclose(pwrc.resid_cov, gamma0 - coefs[0] @ gamma0 @ coefs[0].T)


def test_resid_cov_sample_autocov(var_data):
    # Only the VAR determines the residual covariance
    base = PreWhitenedRecolored(var_data, lags=2)
    sample = PreWhitenedRecolored(var_data, lags=2, sample_autocov=True)
    assert_allclose(sample.resid_cov, base.resid_cov)


def test_var_results_nonstationary_var():
    # No stationarity is needed for the results of the VAR
    rs = np.random.RandomState(0)
    e = rs.standard_normal((250, 2))
    x = np.zeros_like(e)
    for t in range(1, x.shape[0]):
        x[t] = 1.05 * x[t - 1] + e[t]
    pwrc = PreWhitenedRecolored(x, lags=1)
    assert pwrc.order == (1, 1)
    assert pwrc.resid.shape == (249, 2)
    assert pwrc.resid_cov.shape == (2, 2)
    with pytest.raises(ValueError, match="not compatible with covariance"):
        _ = pwrc.cov
