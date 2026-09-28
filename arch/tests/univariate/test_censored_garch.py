import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy.stats import norm

from arch.univariate import ConstantMean
from arch.univariate.censored_garch import CensoredGARCH


@pytest.fixture(scope="module")
def setup():
    rng = RandomState(1234)
    t = 1000
    resids = rng.standard_normal(t)
    threshold = 1.5
    censored = np.abs(resids) >= threshold
    # Observed residuals are truncated at the threshold, as they would be
    # for an actual censored return series.
    observed = np.where(censored, np.sign(resids) * threshold, resids)
    return {
        "rng": rng,
        "t": t,
        "resids": observed,
        "censored": censored,
        "threshold": threshold,
        "sigma2": np.zeros_like(observed),
    }


def test_bounds_and_names(setup):
    vol = CensoredGARCH(censored=setup["censored"], threshold=setup["threshold"])
    resids = setup["resids"]

    bounds = vol.bounds(resids)
    v = np.mean(resids**2.0)
    assert_allclose(bounds[0], (1.0e-8 * v, 10.0 * v))
    assert_equal(bounds[1], (0.0, 1.0))
    assert_equal(bounds[2], (0.0, 1.0))

    assert vol.parameter_names() == ["omega", "alpha[1]", "beta[1]"]
    assert vol.num_params == 3


def test_constraints(setup):
    vol = CensoredGARCH(censored=setup["censored"], threshold=setup["threshold"])
    a, b = vol.constraints()
    # omega >= 0, alpha >= 0, beta >= 0, alpha + beta <= 1
    a_target = np.vstack((np.eye(3), np.array([[0, -1.0, -1.0]])))
    b_target = np.array([0.0, 0.0, 0.0, -1.0])
    assert_allclose(a, a_target)
    assert_allclose(b, b_target)


def test_invalid_orders():
    censored = np.zeros(10, dtype=bool)
    with pytest.raises(ValueError, match="One of p or q must be strictly positive"):
        CensoredGARCH(censored=censored, threshold=1.0, p=0, q=0)


def test_invalid_threshold():
    censored = np.array([False, True, False])
    with pytest.raises(ValueError, match="threshold must be strictly positive"):
        CensoredGARCH(censored=censored, threshold=np.array([1.0, -1.0, 1.0]))


def test_mismatched_threshold_length():
    censored = np.array([False, True, False])
    with pytest.raises(ValueError, match="same length"):
        CensoredGARCH(censored=censored, threshold=np.array([1.0, 1.0]))


def test_no_censoring_matches_plain_garch(setup):
    """With censored all-False, the recursion must reduce exactly to a plain
    GARCH(1, 1) recursion, since the correction term is never applied."""
    from arch.univariate.volatility import GARCH

    resids = setup["resids"]
    t = setup["t"]
    censored = np.zeros(t, dtype=bool)

    vol = CensoredGARCH(censored=censored, threshold=1.0, p=1, q=1)
    vol._start, vol._stop = 0, t
    backcast = vol.backcast(resids)
    var_bounds = vol.variance_bounds(resids)
    parameters = np.array([0.1, 0.1, 0.8])

    sigma2_censored = np.zeros(t)
    vol.compute_variance(parameters, resids, sigma2_censored, backcast, var_bounds)

    garch = GARCH(p=1, q=1)
    sigma2_plain = np.zeros(t)
    garch.compute_variance(parameters, resids, sigma2_plain, backcast, var_bounds)

    assert_allclose(sigma2_censored, sigma2_plain)


def test_censoring_increases_variance(setup):
    """A censored observation's effective squared residual must be at least
    as large as the naive (truncated) squared residual -- the whole point of
    the correction -- and strictly larger whenever the threshold truly binds
    below the model's own conditional standard deviation."""
    resids = setup["resids"]
    censored = setup["censored"]
    threshold = setup["threshold"]
    t = setup["t"]

    vol_censored = CensoredGARCH(censored=censored, threshold=threshold, p=1, q=1)
    vol_censored._start, vol_censored._stop = 0, t
    vol_naive = CensoredGARCH(
        censored=np.zeros(t, dtype=bool), threshold=threshold, p=1, q=1
    )
    vol_naive._start, vol_naive._stop = 0, t

    backcast = vol_censored.backcast(resids)
    var_bounds = vol_censored.variance_bounds(resids)
    parameters = np.array([0.05, 0.1, 0.85])

    sigma2_censored = np.zeros(t)
    vol_censored.compute_variance(
        parameters, resids, sigma2_censored, backcast, var_bounds
    )
    sigma2_naive = np.zeros(t)
    vol_naive.compute_variance(parameters, resids, sigma2_naive, backcast, var_bounds)

    # sigma2 at t depends on sigma2/eff2 at t-1, so compare the *next*
    # period's variance following a censored observation.
    censored_idx = np.where(censored[:-1])[0]
    assert len(censored_idx) > 0
    assert np.all(sigma2_censored[censored_idx + 1] >= sigma2_naive[censored_idx + 1])
    assert np.any(sigma2_censored[censored_idx + 1] > sigma2_naive[censored_idx + 1])


def test_correction_matches_truncated_normal_identity():
    """Directly check the recursion's censoring correction against the
    truncated-normal second-moment identity
    E[Z^2 | |Z| >= a] = 1 + a * phi(a) / (1 - Phi(a))
    for a single censored step with a known sigma2 carried in from omega."""
    from arch.univariate.censored_garch import censored_garch_recursion

    omega = 1.0
    threshold = 1.2
    parameters = np.array([omega, 0.0, 0.0])  # p=1, q=1, both zeroed out
    resids = np.array([threshold])
    censored = np.array([True])
    thresh_arr = np.array([threshold])
    sigma2 = np.zeros(1)
    var_bounds = np.array([[1e-12, 1e12]])

    censored_garch_recursion(
        parameters, resids, censored, thresh_arr, sigma2, 1, 1, 1, 1.0, var_bounds
    )
    # sigma2[0] = omega = 1.0 regardless of censoring (alpha=beta=0 here);
    # the correction only shows up in the *next* period's effective
    # variance input, which we can't observe directly from sigma2 alone in
    # this minimal example, so instead verify the corr factor formula used
    # inside the recursion analytically.
    a = threshold / np.sqrt(sigma2[0])
    expected_corr = 1.0 + a * norm.pdf(a) / norm.sf(a)

    # Re-run with a second observation to pull the correction into sigma2.
    parameters2 = np.array([0.0, 1.0, 0.0])  # sigma2[1] = alpha * eff2[0]
    resids2 = np.array([threshold, 0.0])
    censored2 = np.array([True, False])
    thresh2 = np.array([threshold, 0.0])
    sigma2_2 = np.zeros(2)
    var_bounds2 = np.array([[1e-12, 1e12], [1e-12, 1e12]])
    censored_garch_recursion(
        parameters2, resids2, censored2, thresh2, sigma2_2, 1, 1, 2, 1.0, var_bounds2
    )
    assert_allclose(sigma2_2[1], expected_corr, rtol=1e-10)


def test_fit_and_forecast_end_to_end():
    """Smoke test the full estimation pipeline: ConstantMean + CensoredGARCH
    should fit without error and produce a finite one-step forecast."""
    import pandas as pd

    rng = RandomState(42)
    nobs = 500
    threshold = 1.8
    resids = rng.standard_normal(nobs)
    censored = np.abs(resids) >= threshold
    observed = np.where(censored, np.sign(resids) * threshold, resids)
    y = pd.Series(observed)

    model = ConstantMean(y)
    model.volatility = CensoredGARCH(censored=censored, threshold=threshold, p=1, q=1)
    res = model.fit(disp="off")

    assert np.all(np.isfinite(res.params.values))
    assert res.params["alpha[1]"] >= 0
    assert res.params["beta[1]"] >= 0

    forecast = res.forecast(horizon=1, reindex=False)
    assert np.isfinite(forecast.variance.values[-1, 0])
    assert forecast.variance.values[-1, 0] > 0


def test_simulate_shapes():
    censored = np.zeros(10, dtype=bool)
    vol = CensoredGARCH(censored=censored, threshold=1.0, p=1, q=1)
    rng = RandomState(0)
    parameters = np.array([0.05, 0.1, 0.85])
    data, sigma2 = vol.simulate(parameters, 250, rng.standard_normal, burn=50)
    assert data.shape == (250,)
    assert sigma2.shape == (250,)
    assert np.all(sigma2 > 0)


def test_unsupported_forecast_options():
    censored = np.zeros(10, dtype=bool)
    vol = CensoredGARCH(censored=censored, threshold=1.0, p=1, q=1)
    with pytest.raises(NotImplementedError, match="horizon=1"):
        vol._check_forecasting_method("analytic", horizon=2)
    with pytest.raises(NotImplementedError, match="analytic"):
        vol._check_forecasting_method("simulation", horizon=1)
    with pytest.raises(NotImplementedError, match="Simulation-based forecasts"):
        vol._simulation_forecast(
            np.array([0.05, 0.1, 0.85]),
            np.zeros(10),
            1.0,
            np.tile([1e-8, 1e8], (10, 1)),
            0,
            1,
            10,
            RandomState(0).standard_normal,
        )
