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


def test_recursion_deep_tail_underflow_branch():
    """When the threshold is many standard deviations out, 1 - Phi(a)
    underflows to 0 in floating point before a itself does. The recursion
    should fall back to the asymptotic correction 1 + a^2 rather than
    dividing by (effectively) zero."""
    from arch.univariate.censored_garch import (
        censored_garch_recursion,
        censored_garch_recursion_python,
    )
    from scipy.stats import norm

    a = 40.0  # norm.sf(40) underflows to exactly 0.0 in float64
    assert norm.sf(a) == 0.0

    threshold = a
    # omega=0, alpha=1, backcast=1.0 => sigma2[0] = 0 + 1*backcast = 1.0
    # exactly, so a = threshold / sigma_t = threshold / 1.0 = a as intended.
    parameters = np.array([0.0, 1.0, 0.0])
    resids = np.array([threshold, 0.0])
    censored = np.array([True, False])
    thresh_arr = np.array([threshold, 0.0])
    var_bounds = np.array([[1e-12, 1e12], [1e-12, 1e12]])

    # Exercise both the (possibly jitted) recursion and the pure-Python source.
    for recursion in (censored_garch_recursion, censored_garch_recursion_python):
        sigma2 = np.zeros(2)
        recursion(
            parameters, resids, censored, thresh_arr, sigma2, 1, 1, 2, 1.0, var_bounds
        )
        assert_allclose(sigma2[0], 1.0)
        # sigma2[1] = alpha * eff2[0] = 1.0 * (1 + a^2) * sigma2[0] = 1 + a^2
        assert_allclose(sigma2[1], 1.0 + a * a, rtol=1e-10)


def test_recursion_zero_threshold_branch():
    """A censored observation with a zero threshold (a degenerate edge
    case not reachable through the public API, which requires threshold >
    0 wherever censored is True) should fall back to the uncorrected
    (corr=1.0) branch rather than raising or dividing by zero."""
    from arch.univariate.censored_garch import (
        censored_garch_recursion,
        censored_garch_recursion_python,
    )

    # omega=0, alpha=1, backcast=1.0 => sigma2[0] = 1.0 exactly (isolates the
    # step cleanly, same trick as the deep-tail test above).
    parameters = np.array([0.0, 1.0, 0.0])
    resids = np.array([0.0, 0.0])
    censored = np.array([True, False])
    threshold = np.array([0.0, 0.0])
    var_bounds = np.array([[1e-12, 1e12], [1e-12, 1e12]])

    # Exercise both the (possibly jitted) recursion and the pure-Python source.
    for recursion in (censored_garch_recursion, censored_garch_recursion_python):
        sigma2 = np.zeros(2)
        recursion(
            parameters, resids, censored, threshold, sigma2, 1, 1, 2, 1.0, var_bounds
        )
        assert_allclose(sigma2[0], 1.0)
        # corr = 1.0 (a <= 0 branch), so eff2[0] = sigma2[0] = 1.0 and
        # sigma2[1] = alpha * eff2[0] = 1.0
        assert_allclose(sigma2[1], 1.0)


def test_recursion_zero_variance_branch():
    """When the model's own conditional variance is driven to exactly the
    (zero) lower bound at a censored step, sigma_t is 0 and a would
    otherwise be a division by zero; the recursion should fall back to
    a = 0 (corr = 1.0) instead."""
    from arch.univariate.censored_garch import censored_garch_recursion

    parameters = np.array([0.0, 0.0, 0.0])  # omega=0 -> sigma2[0] forced to 0
    resids = np.array([1.5])
    censored = np.array([True])
    threshold = np.array([1.5])
    sigma2 = np.zeros(1)
    var_bounds = np.array([[0.0, 1e12]])  # lower bound of exactly 0

    censored_garch_recursion(
        parameters, resids, censored, threshold, sigma2, 1, 1, 1, 0.0, var_bounds
    )
    assert sigma2[0] == 0.0


def test_norm_helpers_match_scipy():
    """The hand-rolled, numba-jittable _norm_pdf/_norm_sf (built from
    math.exp/math.erfc rather than scipy.stats.norm, so they can be
    inlined into the jitted recursion) must agree with scipy across a
    wide range of inputs, including the deep tail where naive 1 - Phi(x)
    would underflow long before erfc-based _norm_sf does."""
    from arch.univariate.censored_garch import (
        _norm_pdf,
        _norm_pdf_python,
        _norm_sf,
        _norm_sf_python,
    )

    xs = np.array([0.0, 0.1, 0.5, 1.0, 1.96, 3.0, 5.0, 8.0, 15.0, 30.0])
    for x in xs:
        # Check both the (possibly jitted) helpers and their pure-Python source.
        for pdf in (_norm_pdf, _norm_pdf_python):
            assert_allclose(pdf(x), norm.pdf(x), rtol=1e-12, atol=1e-300)
        # scipy's own norm.sf underflows well before erfc-based _norm_sf
        # does, so only compare where scipy itself hasn't hit zero.
        scipy_sf = norm.sf(x)
        if scipy_sf > 0:
            for sf in (_norm_sf, _norm_sf_python):
                assert_allclose(sf(x), scipy_sf, rtol=1e-8)


def test_jitted_recursion_matches_pure_python():
    """The numba-jitted censored_garch_recursion (used by compute_variance)
    must produce bit-identical output to the pure-Python source it's
    compiled from, censored_garch_recursion_python, across a realistic
    GARCH(1,1)-then-censored path."""
    from arch.univariate.censored_garch import (
        censored_garch_recursion,
        censored_garch_recursion_python,
    )

    rng = RandomState(7)
    nobs = 2000
    threshold = 1.8
    resids = rng.standard_normal(nobs)
    censored = np.abs(resids) >= threshold
    resids = np.where(censored, np.sign(resids) * threshold, resids)
    thresh_arr = threshold * np.ones(nobs) * censored
    parameters = np.array([0.05, 0.1, 0.85])
    var_bounds = np.tile([1e-12, 1e12], (nobs, 1))

    sigma2_jit = np.zeros(nobs)
    sigma2_py = np.zeros(nobs)
    censored_garch_recursion(
        parameters, resids, censored, thresh_arr, sigma2_jit, 1, 1, nobs, 1.0, var_bounds
    )
    censored_garch_recursion_python(
        parameters, resids, censored, thresh_arr, sigma2_py, 1, 1, nobs, 1.0, var_bounds
    )
    assert_allclose(sigma2_jit, sigma2_py, rtol=0, atol=0)


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


def test_compute_variance_length_mismatch():
    """A censored/threshold series that doesn't line up with the data being
    fit (neither nobs nor nobs - 1) must raise rather than silently misalign."""
    vol = CensoredGARCH(censored=np.zeros(10, dtype=bool), threshold=1.0)
    vol._start, vol._stop = 0, 10
    resids = np.zeros(5)
    var_bounds = np.tile([1e-8, 1e8], (5, 1))
    with pytest.raises(ValueError, match="does not match the data"):
        vol.compute_variance(
            np.array([0.05, 0.1, 0.85]), resids, np.zeros(5), 1.0, var_bounds
        )


@pytest.mark.parametrize("p, q", [(1, 0), (0, 1), (2, 2)])
def test_starting_values_orders(setup, p, q):
    """starting_values must handle models with no ARCH term (p=0) or no
    GARCH term (q=0), as well as higher orders."""
    resids = setup["resids"]
    t = setup["t"]
    vol = CensoredGARCH(
        censored=setup["censored"], threshold=setup["threshold"], p=p, q=q
    )
    vol._start, vol._stop = 0, t
    sv = vol.starting_values(resids)
    assert sv.shape == (1 + p + q,)
    assert np.all(np.isfinite(sv))
    assert sv[0] > 0
    assert np.all(sv[1:] >= 0)


def test_simulate_initial_value():
    """A user-supplied initial_value is used as-is for the first variance."""
    vol = CensoredGARCH(censored=np.zeros(10, dtype=bool), threshold=1.0)
    rng = RandomState(0)
    parameters = np.array([0.05, 0.1, 0.85])
    data, sigma2 = vol.simulate(
        parameters, 20, rng.standard_normal, burn=0, initial_value=0.5
    )
    assert data.shape == (20,)
    assert_allclose(sigma2[0], 0.5)


def test_simulate_default_initial_value_stationary():
    """With persistence < 1 the default start is the unconditional variance."""
    vol = CensoredGARCH(censored=np.zeros(10, dtype=bool), threshold=1.0)
    rng = RandomState(0)
    parameters = np.array([0.05, 0.1, 0.85])
    _, sigma2 = vol.simulate(parameters, 20, rng.standard_normal, burn=0)
    assert_allclose(sigma2[0], 0.05 / (1.0 - 0.95))


def test_simulate_default_initial_value_unit_root():
    """With persistence >= 1 there is no unconditional variance, so the
    default start falls back to omega."""
    vol = CensoredGARCH(censored=np.zeros(10, dtype=bool), threshold=1.0)
    rng = RandomState(0)
    parameters = np.array([0.05, 0.5, 0.5])
    data, sigma2 = vol.simulate(parameters, 20, rng.standard_normal, burn=0)
    assert_allclose(sigma2[0], 0.05)
    assert np.all(np.isfinite(data))
    assert np.all(sigma2 > 0)
