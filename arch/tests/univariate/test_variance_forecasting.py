from math import exp, lgamma

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import integrate, stats
from scipy.special import gammaln

from arch.univariate import arch_model
from arch.univariate.distribution import (
    GeneralizedError,
    Normal,
    SkewStudent,
    StudentsT,
)
from arch.univariate.mean import ARX, ConstantMean, MAMean

try:
    from arch.univariate.recursions import figarch_weights
except ImportError:
    from arch.univariate.recursions_python import figarch_weights

from arch.univariate.volatility import (
    APARCH,
    EGARCH,
    FIAPARCH,
    FIGARCH,
    GARCH,
    HARCH,
    BootstrapRng,
    ConstantVariance,
    EWMAVariance,
    FixedVariance,
    MIDASHyperbolic,
    RiskMetrics2006,
)

try:
    from arch.univariate.recursions import garch_recursion
except ImportError:
    from arch.univariate.recursions_python import garch_recursion


def _compare_truncated_forecasts(full, trunc, start):
    assert np.all(np.isfinite(trunc.forecasts[:start]))
    assert_allclose(trunc.forecasts, full.forecasts[start:])

    if full.forecast_paths is None:
        assert trunc.forecast_paths is None
        assert trunc.shocks is None
        return


def _aparch_hand_next(e, sigma_delta, params, p, o, q, delta, backcast):
    """
    One step of the APARCH recursion written out term by term.

    ``e`` holds the shocks and ``sigma_delta`` the conditional variances to the
    power ``delta / 2`` up to (and excluding) the period being computed.
    """
    t = len(e)
    omega = params[0]
    alpha = params[1 : p + 1]
    gamma = params[p + 1 : p + o + 1]
    beta = params[p + o + 1 : p + o + q + 1]
    value = omega
    for j in range(p):
        if t - 1 - j < 0:
            shock = backcast**0.5
        else:
            shock = abs(e[t - 1 - j])
            if j < o:
                shock = shock - gamma[j] * e[t - 1 - j]
        value += alpha[j] * shock**delta
    for j in range(q):
        if t - 1 - j < 0:
            value += beta[j] * backcast ** (delta / 2.0)
        else:
            value += beta[j] * sigma_delta[t - 1 - j]
    return value


def _aparch_hand_history(resids, params, p, o, q, delta, backcast):
    sigma_delta: list[float] = []
    for t in range(len(resids)):
        value = _aparch_hand_next(
            [float(r) for r in resids[:t]],
            sigma_delta,
            params,
            p,
            o,
            q,
            delta,
            backcast,
        )
        sigma_delta.append(value)
    return sigma_delta


def _aparch_hand_simulation(
    resids, history_sd, params, p, o, q, delta, backcast, origin, draws
):
    """
    Simulated APARCH variance paths for one forecast origin.

    ``draws`` has shape (simulations, horizon) and contains the standardized
    shocks. Every path extends the history and applies ``_aparch_hand_next``.
    """
    history_e = [float(r) for r in resids[: origin + 1]]
    sims, horizon = draws.shape
    paths = np.empty((sims, horizon))
    shocks = np.empty((sims, horizon))
    for s in range(sims):
        e = list(history_e)
        sd = list(history_sd[: origin + 1])
        for h in range(horizon):
            value = _aparch_hand_next(e, sd, params, p, o, q, delta, backcast)
            sd.append(value)
            e.append(draws[s, h] * value ** (1.0 / delta))
            paths[s, h] = value ** (2.0 / delta)
            shocks[s, h] = e[-1]
    return paths, shocks


class PreservedState:
    """
    Context manager that will save NumPy's random generator's state when entering and
    restore the original state when exiting.
    """

    def __init__(self, random_state):
        self._random_state = random_state
        self._state = None

    def __enter__(self):
        self._state = self._random_state.get_state()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._random_state.set_state(self._state)


preserved_state = PreservedState


def _simple_direct_gjrgarch_forecaster(
    resids, params, p, o, q, backcast, var_bounds, horizon
):
    """
    Simple GARCH forecasting for use when testing model-based forecasts

    Parameters
    ----------
    resids : ndarray
    params : ndarray
    p : int
    o : int
    q : int
    backcast : float
    var_bounds : ndarray
    horizon : int

    Returns
    -------
    forecasts : ndarray

    """
    m = max([p, o, q])
    t = resids.shape[0]
    _resids = resids
    _sigma2 = np.empty(t)
    garch_recursion(
        params,
        resids**2.0,
        np.sign(resids),
        _sigma2,
        p,
        o,
        q,
        t,
        backcast,
        var_bounds,
    )
    resids = np.empty((t, m + horizon))
    asymresids = resids.copy()
    sigma2 = np.empty((t, m + horizon))

    resids[:, :m] = np.sqrt(backcast)
    asymresids[:, :m] = np.sqrt(0.5 * backcast)
    sigma2[:, :m] = backcast

    for i in range(m):
        resids[m - 1 - i :, i] = _resids[: (t - (m - 1) + i)]
        asymresids[m - 1 - i :, i] = _resids[: (t - (m - 1) + i)] * (
            _resids[: (t - (m - 1) + i)] < 0
        )
        sigma2[m - 1 - i :, i] = _sigma2[: (t - (m - 1) + i)]

    for i in range(m, m + horizon):
        sigma2[:, i] = params[0]
        ind = 1
        for j in range(p):
            sigma2[:, i] += params[ind] * resids[:, i - j - 1] ** 2.0
            ind += 1

        for j in range(o):
            sigma2[:, i] += params[ind] * asymresids[:, i - j - 1] ** 2.0
            ind += 1

        for j in range(q):
            sigma2[:, i] += params[ind] * sigma2[:, i - j - 1]
            ind += 1
        resids[:, i] = np.sqrt(sigma2[:, i])
        asymresids[:, i] = np.sqrt(0.5 * sigma2[:, i])

    return sigma2[:, m:]


class TestVarianceForecasts:
    @classmethod
    def setup_class(cls):
        cls.rng = RandomState(12345)
        cls.t = 1000
        cls.resid = cls.rng.standard_normal(cls.t) * np.sqrt(10)

    def test_constant_variance_forecast(self):
        vol = ConstantVariance()
        params = np.array([10.0])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(params, self.resid, backcast, var_bounds)
        assert np.all(np.isfinite(forecast.forecasts))
        assert forecast.forecasts[-1] == params[0]
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        forecast = vol.forecast(params, self.resid, backcast, var_bounds, horizon=5)
        assert forecast.forecasts.shape == (1, 5)
        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(forecast.forecasts == params[0])
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        forecast = vol.forecast(params, self.resid, backcast, var_bounds, start=100)
        assert forecast.forecasts.shape == (900, 1)
        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(forecast.forecasts == params[0])
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        forecast = vol.forecast(
            params, self.resid, backcast, var_bounds, start=100, horizon=3
        )
        assert forecast.forecasts.shape == (900, 3)
        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(forecast.forecasts == params[0])
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        with pytest.raises(
            ValueError, match=r"Bootstrap forecasting requires at least"
        ):
            vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=5,
                method="bootstrap",
                start=0,
                simulations=2000,
            )

    def test_arch_1_forecast(self):
        t = self.t
        vol = GARCH(p=1, o=0, q=0)
        params = np.array([10.0, 0.4])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(params, self.resid, backcast, var_bounds)
        expected = np.empty((t, 1))
        expected.fill(np.nan)
        expected[-1] = params[0] + params[1] * self.resid[-1] ** 2
        assert_allclose(forecast.forecasts, expected[-1:])
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        forecast = vol.forecast(params, self.resid, backcast, var_bounds, horizon=5)
        assert forecast.forecasts.shape == (1, 5)
        assert np.all(np.isfinite(forecast.forecasts))
        assert forecast.forecast_paths is None
        assert forecast.shocks is None
        expected = np.zeros(5)
        expected[0] = params[0] + params[1] * self.resid[-1] ** 2
        for i in range(1, 5):
            expected[i] = params[0] + params[1] * expected[i - 1]
        assert_allclose(forecast.forecasts[-1], expected)

        forecast = vol.forecast(params, self.resid, backcast, var_bounds, start=0)
        assert forecast.forecasts.shape == (1000, 1)
        assert np.all(np.isfinite(forecast.forecasts))
        assert forecast.forecast_paths is None
        assert forecast.shocks is None
        expected = params[0] + params[1] * self.resid**2
        try:
            expected = np.reshape(expected, (1000, 1), copy=False)
        except TypeError:
            expected = np.reshape(expected, (1000, 1))
        assert_allclose(forecast.forecasts, expected)

        forecast = vol.forecast(
            params, self.resid, backcast, var_bounds, start=0, horizon=3
        )
        assert forecast.forecasts.shape == (1000, 3)
        assert np.all(np.isfinite(forecast.forecasts))
        assert forecast.forecast_paths is None
        assert forecast.shocks is None
        expected = np.zeros((1000, 3))
        expected[:, 0] = params[0] + params[1] * self.resid**2
        for i in range(1, 3):
            expected[:, i] = params[0] + params[1] * expected[:, i - 1]
        assert_allclose(forecast.forecasts, expected)

        forecast = vol.forecast(
            params, self.resid, backcast, var_bounds, start=100, horizon=3
        )
        assert_allclose(forecast.forecasts, expected[100:])
        with pytest.raises(ValueError, match=r"horizon must be an integer >= 1"):
            vol.forecast(params, self.resid, backcast, var_bounds, start=0, horizon=0)

    def test_arch_1_forecast_simulation(self):
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        vol = GARCH(p=1, o=0, q=0)
        params = np.array([10.0, 0.4])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=1,
                method="simulation",
                rng=rng,
            )
        assert forecast.forecasts.shape == (1, 1)
        assert forecast.forecast_paths.shape == (1, 1000, 1)
        assert forecast.shocks.shape == (1, 1000, 1)
        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(np.isfinite(forecast.forecast_paths))
        expected = params[0] + params[1] * self.resid[-1] ** 2.0
        assert_allclose(forecast.forecasts[-1], expected)
        assert_allclose(forecast.forecast_paths[-1], expected * np.ones((1000, 1)))
        assert_allclose(forecast.shocks[-1], np.sqrt(expected) * rng((1000, 1)))

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=5,
                method="simulation",
                rng=rng,
            )

        paths = np.zeros((1000, 5))
        paths[:, 0] = expected
        std_shocks = rng((1000, 5))
        shocks = np.zeros((1000, 5))
        shocks[:, 0] = np.sqrt(paths[:, 0]) * std_shocks[:, 0]
        for i in range(1, 5):
            paths[:, i] = params[0] + params[1] * shocks[:, i - 1] ** 2
            shocks[:, i] = np.sqrt(paths[:, i]) * std_shocks[:, i]

        assert forecast.forecasts.shape == (1, 5)
        assert forecast.forecast_paths.shape == (1, 1000, 5)
        assert forecast.shocks.shape == (1, 1000, 5)
        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(np.isfinite(forecast.forecast_paths))
        assert np.all(np.isfinite(forecast.shocks))
        assert_allclose(forecast.forecasts[-1], paths.mean(0))
        assert_allclose(forecast.forecast_paths[-1], paths)
        assert_allclose(forecast.shocks[-1], shocks)

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=5,
                start=100,
                method="simulation",
                simulations=2000,
                rng=rng,
            )

        paths = np.zeros((1000, 2000, 5))
        paths.fill(np.nan)
        shocks = np.zeros((1000, 2000, 5))
        shocks.fill(np.nan)
        for i in range(100, 1000):
            std_shocks = rng((2000, 5))
            paths[i, :, 0] = params[0] + params[1] * self.resid[i] ** 2.0
            shocks[i, :, 0] = np.sqrt(paths[i, :, 0]) * std_shocks[:, 0]
            for j in range(1, 5):
                paths[i, :, j] = params[0] + params[1] * shocks[i, :, j - 1] ** 2
                shocks[i, :, j] = np.sqrt(paths[i, :, j]) * std_shocks[:, j]

        assert_allclose(forecast.shocks, shocks[100:])
        assert_allclose(forecast.forecast_paths, paths[100:])
        assert_allclose(forecast.forecasts, paths[100:].mean(1))

    def test_arch_1_forecast_bootstrap(self):
        vol = GARCH(p=1, o=0, q=0)
        params = np.array([10.0, 0.4])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=5,
                method="bootstrap",
                random_state=self.rng,
            )
        sigma2 = np.zeros(1000)
        sigma2[0] = params[0] + params[1] * backcast
        for i in range(1, 1000):
            sigma2[i] = params[0] + params[1] * self.resid[i - 1] ** 2.0
        std_resids = self.resid / np.sqrt(sigma2)
        locs = np.floor(1000 * self.rng.random_sample((1000, 5))).astype(np.int64)
        std_shocks = std_resids[locs]

        paths = np.zeros((1000, 5))
        paths[:, 0] = params[0] + params[1] * self.resid[-1] ** 2.0
        shocks = np.zeros((1000, 5))
        shocks[:, 0] = np.sqrt(paths[:, 0]) * std_shocks[:, 0]
        for i in range(1, 5):
            paths[:, i] = params[0] + params[1] * shocks[:, i - 1] ** 2
            shocks[:, i] = np.sqrt(paths[:, i]) * std_shocks[:, i]

        assert forecast.forecasts.shape == (1, 5)
        assert forecast.forecast_paths.shape == (1, 1000, 5)
        assert forecast.shocks.shape == (1, 1000, 5)
        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(np.isfinite(forecast.forecast_paths))
        assert np.all(np.isfinite(forecast.shocks))
        assert_allclose(forecast.forecasts[-1], paths.mean(0))
        assert_allclose(forecast.forecast_paths[-1], paths)
        assert_allclose(forecast.shocks[-1], shocks)

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=5,
                method="bootstrap",
                start=333,
                simulations=2000,
                random_state=self.rng,
            )

        paths = np.zeros((1000, 2000, 5))
        paths.fill(np.nan)
        shocks = np.zeros((1000, 2000, 5))
        shocks.fill(np.nan)
        for i in range(333, 1000):
            locs = self.rng.random_sample((2000, 5))
            int_locs = np.floor((i + 1) * locs).astype(np.int64)
            std_shocks = std_resids[int_locs]
            paths[i, :, 0] = params[0] + params[1] * self.resid[i] ** 2.0
            shocks[i, :, 0] = np.sqrt(paths[i, :, 0]) * std_shocks[:, 0]
            for j in range(1, 5):
                paths[i, :, j] = params[0] + params[1] * shocks[i, :, j - 1] ** 2
                shocks[i, :, j] = np.sqrt(paths[i, :, j]) * std_shocks[:, j]

        assert_allclose(forecast.forecast_paths, paths[333:])
        assert_allclose(forecast.forecasts, paths[333:].mean(1))
        assert_allclose(forecast.shocks, shocks[333:])

    def test_arch_2_forecast(self):
        vol = GARCH(p=2, o=0, q=0)
        params = np.array([10.0, 0.4, 0.3])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(
            params, self.resid, backcast, var_bounds, horizon=10, start=0
        )
        expected = _simple_direct_gjrgarch_forecaster(
            self.resid, params, 2, 0, 0, backcast, var_bounds, 10
        )
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_garch_11_forecast(self):
        vol = GARCH(p=1, o=0, q=1)
        params = np.array([10.0, 0.1, 0.85])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(
            params, self.resid, backcast, var_bounds, horizon=10, start=0
        )
        expected = _simple_direct_gjrgarch_forecaster(
            self.resid, params, 1, 0, 1, backcast, var_bounds, 10
        )
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_gjrgarch_111_forecast(self):
        vol = GARCH(p=1, o=1, q=1)
        params = np.array([10.0, 0.05, 0.1, 0.85])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(
            params, self.resid, backcast, var_bounds, horizon=10, start=0
        )
        expected = _simple_direct_gjrgarch_forecaster(
            self.resid, params, 1, 1, 1, backcast, var_bounds, 10
        )
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_garch_21_forecast(self):
        vol = GARCH(p=2, o=0, q=1)
        params = np.array([10.0, 0.05, 0.1, 0.85])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(
            params, self.resid, backcast, var_bounds, horizon=10, start=0
        )
        expected = _simple_direct_gjrgarch_forecaster(
            self.resid, params, 2, 0, 1, backcast, var_bounds, 10
        )
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_garch_12_forecast(self):
        vol = GARCH(p=1, o=0, q=2)
        params = np.array([10.0, 0.1, 0.55, 0.3])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(
            params, self.resid, backcast, var_bounds, horizon=10, start=0
        )
        expected = _simple_direct_gjrgarch_forecaster(
            self.resid, params, 1, 0, 2, backcast, var_bounds, 10
        )
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_garch_22_forecast(self):
        vol = GARCH(p=2, o=0, q=2)
        params = np.array([10.0, 0.1, 0.05, 0.4, 0.3])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(
            params, self.resid, backcast, var_bounds, horizon=10, start=0
        )
        expected = _simple_direct_gjrgarch_forecaster(
            self.resid, params, 2, 0, 2, backcast, var_bounds, 10
        )
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_harch_forecast(self):
        vol = HARCH(lags=[1, 5, 22])
        params = np.array([10.0, 0.4, 0.3, 0.2])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        forecast = vol.forecast(
            params, self.resid, backcast, var_bounds, horizon=10, start=0
        )
        trans_params = np.zeros(23)
        trans_params[0] = params[0]
        trans_params[1:2] += params[1]
        trans_params[1:6] += params[2] / 5
        trans_params[1:23] += params[3] / 22
        expected = _simple_direct_gjrgarch_forecaster(
            self.resid, trans_params, 22, 0, 0, backcast, var_bounds, 10
        )
        assert_allclose(forecast.forecasts, expected)

        assert forecast.forecast_paths is None
        assert forecast.shocks is None

    def test_tarch_111_forecast(self):
        t = self.t
        vol = GARCH(p=1, o=1, q=1, power=1.0)
        params = np.array([3.0, 0.1, 0.1, 0.80])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid, power=1.0)

        forecast = vol.forecast(
            params, self.resid, backcast, var_bounds, horizon=1, start=0
        )
        sigma = np.zeros((t + 1, 1))
        sigma[0] = params[0] + (params[1] + 0.5 * params[2] + params[3]) * backcast

        resid = self.resid
        abs_resid = np.abs(resid)
        sresid = resid < 0
        for i in range(1, t + 1):
            sigma[i] = params[0]
            sigma[i] += params[1] * abs_resid[i - 1]
            sigma[i] += params[2] * abs_resid[i - 1] * sresid[i - 1]
            sigma[i] += params[3] * sigma[i - 1]

        expected = sigma[1:] ** 2.0
        assert_allclose(forecast.forecasts, expected)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        with pytest.raises(
            ValueError, match=r"Analytic forecasts not available for horizon"
        ):
            vol.forecast(params, self.resid, backcast, var_bounds, horizon=10, start=0)

    def test_tarch_111_forecast_simulation(self):
        t = self.t
        vol = GARCH(p=1, o=1, q=1, power=1.0)
        dist = StudentsT(seed=self.rng)
        rng = dist.simulate([8.0])
        params = np.array([3.0, 0.1, 0.1, 0.80])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids, power=1.0)
        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=10,
                start=0,
                rng=rng,
                method="simulation",
            )
        one_step = vol.forecast(
            params, self.resid, backcast, var_bounds, horizon=1, start=0
        )

        paths = np.zeros((1000, 10))
        shocks = np.zeros((1000, 10))
        for j in range(t):
            std_shocks = rng((1000, 10))
            paths[:, 0] = one_step.forecasts[j, 0]
            shocks[:, 0] = std_shocks[:, 0] * np.sqrt(paths[:, 0])
            sqrt_path = np.sqrt(paths[:, 0])
            for i in range(1, 10):
                abs_err = np.abs(shocks[:, i - 1])
                err_neg = shocks[:, i - 1] < 0
                sqrt_path = (
                    params[0]
                    + params[1] * abs_err
                    + params[2] * abs_err * err_neg
                    + params[3] * sqrt_path
                )
                paths[:, i] = sqrt_path**2
                shocks[:, i] = sqrt_path * std_shocks[:, i]
            assert_allclose(paths.mean(0), forecast.forecasts[j])
            assert_allclose(paths, forecast.forecast_paths[j])
            assert_allclose(shocks, forecast.shocks[j])

    def test_tarch_111_forecast_bootstrap(self):
        t = self.t
        vol = GARCH(p=1, o=1, q=1, power=1.0)
        dist = StudentsT(seed=self.rng)
        rng = dist.simulate([8.0])
        params = np.array([3.0, 0.1, 0.1, 0.80])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids, power=1.0)
        sigma2 = np.zeros(t)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=10,
                start=100,
                rng=rng,
                method="bootstrap",
                random_state=self.rng,
            )

        std_resids = resids / np.sqrt(sigma2)

        one_step = vol.forecast(
            params, self.resid, backcast, var_bounds, horizon=1, start=100
        )
        paths = np.zeros((1000, 10))
        shocks = np.zeros((1000, 10))
        for j in range(100, t):
            locs = self.rng.random_sample((1000, 10))
            int_locs = np.floor(locs * (j + 1)).astype(np.int_)
            std_shocks = std_resids[int_locs]

            paths[:, 0] = one_step.forecasts[j - 100, 0]
            shocks[:, 0] = std_shocks[:, 0] * np.sqrt(paths[:, 0])
            sqrt_path = np.sqrt(paths[:, 0])
            for i in range(1, 10):
                abs_err = np.abs(shocks[:, i - 1])
                err_neg = shocks[:, i - 1] < 0
                sqrt_path = (
                    params[0]
                    + params[1] * abs_err
                    + params[2] * abs_err * err_neg
                    + params[3] * sqrt_path
                )
                paths[:, i] = sqrt_path**2
                shocks[:, i] = sqrt_path * std_shocks[:, i]
            assert_allclose(shocks, forecast.shocks[j - 100])
            assert_allclose(paths.mean(0), forecast.forecasts[j - 100])
            assert_allclose(paths, forecast.forecast_paths[j - 100])

        with pytest.raises(
            ValueError, match=r"start must include more than 100 observations"
        ):
            vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=2,
                start=20,
                rng=rng,
                method="bootstrap",
                random_state=self.rng,
            )
        with pytest.raises(ValueError, match=r"unknown is not a known forecasting"):
            vol.forecast(params, self.resid, backcast, var_bounds, method="unknown")

    def test_harch_forecast_simulation(self):
        t = self.t
        vol = HARCH(lags=[1, 5, 22])
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        params = np.array([3.0, 0.4, 0.3, 0.2])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=0,
                rng=rng,
                method="simulation",
            )

        resids2 = np.zeros((t, 22 + 10))
        resids2.fill(backcast)
        for i in range(22):
            resids2[21 - i :, i] = resids[: (t - 21 + i)] ** 2.0
        paths = np.zeros((t, 1000, 10))
        shocks = np.zeros((t, 1000, 10))
        const = 3.0
        arch = np.zeros(22)
        arch[0] = 0.4
        arch[:5] += 0.3 / 5
        arch[:22] += 0.2 / 22
        for i in range(t):
            std_shocks = rng((1000, 10))
            temp = np.empty((1000, 32))
            temp[:, :22] = resids2[i : i + 1, :22]
            for j in range(10):
                paths[i, :, j] = const + temp[:, j : 22 + j].dot(arch[::-1])
                shocks[i, :, j] = std_shocks[:, j] * np.sqrt(paths[i, :, j])
                temp[:, 22 + j] = shocks[i, :, j] ** 2.0

        forecasts = paths.mean(1)

        assert_allclose(forecast.forecasts, forecasts)
        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.shocks, shocks)

    def test_harch_forecast_bootstrap(self):
        t = self.t
        vol = HARCH(lags=[1, 5, 22])
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        params = np.array([3.0, 0.4, 0.3, 0.2])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        sigma2 = np.empty_like(resids)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=10,
                start=100,
                rng=rng,
                method="bootstrap",
                random_state=self.rng,
            )

        std_resid = resids / np.sqrt(sigma2)
        resids2 = np.zeros((t, 22 + 10))
        resids2.fill(backcast)
        for i in range(22):
            resids2[21 - i :, i] = resids[: (t - 21 + i)] ** 2.0
        paths = np.zeros((t, 1000, 10))
        paths.fill(np.nan)
        shocks = np.zeros((t, 1000, 10))
        shocks.fill(np.nan)
        const = 3.0
        arch = np.zeros(22)
        arch[0] = 0.4
        arch[:5] += 0.3 / 5
        arch[:22] += 0.2 / 22
        for i in range(100, t):
            locs = self.rng.random_sample((1000, 10))
            locs *= i + 1
            int_locs = np.floor(locs).astype(np.int64)
            std_shocks = std_resid[int_locs]
            temp = np.empty((1000, 32))
            temp[:, :22] = resids2[i : i + 1, :22]
            for j in range(10):
                paths[i, :, j] = const + temp[:, j : 22 + j].dot(arch[::-1])
                shocks[i, :, j] = std_shocks[:, j] * np.sqrt(paths[i, :, j])
                temp[:, 22 + j] = shocks[i, :, j] ** 2.0

        forecasts = paths.mean(1)

        assert_allclose(forecast.forecasts, forecasts[100:])
        assert_allclose(forecast.forecast_paths, paths[100:])
        assert_allclose(forecast.shocks, shocks[100:])

    def test_egarch_111_forecast(self):
        t = self.t
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        vol = EGARCH(p=1, o=1, q=1)
        params = np.array([0.0, 0.1, 0.1, 0.95])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)

        forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=1, start=0
        )

        _resids = np.array(resids.tolist() + [0])
        _sigma2 = np.empty_like(_resids)
        _var_bounds = np.array(var_bounds.tolist() + [[0, np.inf]])
        vol.compute_variance(params, _resids, _sigma2, backcast, _var_bounds)
        sigma2 = _sigma2[:-1]
        one_step = _sigma2[1:][:, None]

        assert_allclose(forecast.forecasts, one_step)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        expected = forecast.forecasts.copy()
        expected[:-1] = np.nan
        forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=1)
        assert_allclose(forecast.forecasts, expected[-1:])
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=5,
                start=0,
                method="simulation",
                rng=rng,
            )
        paths = np.empty((t, 1000, 5))
        shocks = np.empty((t, 1000, 5))
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(t):
            std_shocks = rng((1000, 5))
            paths[i, :, 0] = np.log(one_step[i])
            for j in range(1, 5):
                paths[i, :, j] = (
                    params[0]
                    + params[1] * (np.abs(std_shocks[:, j - 1]) - sqrt2pi)
                    + params[2] * std_shocks[:, j - 1]
                    + params[3] * paths[i, :, j - 1]
                )
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths.mean(1))
        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.shocks, shocks)

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=5,
                start=100,
                method="bootstrap",
                random_state=self.rng,
            )

        std_resids = resids / np.sqrt(sigma2)
        paths = np.empty((t, 1000, 5))
        paths.fill(np.nan)
        shocks = np.empty((t, 1000, 5))
        shocks.fill(np.nan)
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(100, t):
            locs = self.rng.random_sample((1000, 5))
            int_locs = np.floor((i + 1) * locs).astype(np.int64)
            std_shocks = std_resids[int_locs]
            paths[i, :, 0] = np.log(one_step[i])
            for j in range(1, 5):
                paths[i, :, j] = (
                    params[0]
                    + params[1] * (np.abs(std_shocks[:, j - 1]) - sqrt2pi)
                    + params[2] * std_shocks[:, j - 1]
                    + params[3] * paths[i, :, j - 1]
                )
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths[100:].mean(1))
        assert_allclose(forecast.forecast_paths, paths[100:])
        assert_allclose(forecast.shocks, shocks[100:])

        with pytest.raises(ValueError, match=r"Analytic forecasts not available"):
            vol.forecast(params, resids, backcast, var_bounds, horizon=5)

    def test_egarch_101_forecast(self):
        t = self.t
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        vol = EGARCH(p=1, o=0, q=1)
        params = np.array([0.0, 0.1, 0.95])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=1, start=0
        )

        _resids = np.array(resids.tolist() + [0])
        _sigma2 = np.empty_like(_resids)
        _var_bounds = np.array(var_bounds.tolist() + [[0, np.inf]])
        vol.compute_variance(params, _resids, _sigma2, backcast, _var_bounds)
        sigma2 = _sigma2[:-1]
        one_step = _sigma2[1:][:, None]

        assert_allclose(forecast.forecasts, one_step)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        expected = forecast.forecasts.copy()
        expected[:-1] = np.nan
        forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=1)
        assert_allclose(forecast.forecasts, expected[-1:])
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=5,
                start=0,
                method="simulation",
                rng=rng,
            )
        paths = np.empty((t, 1000, 5))
        shocks = np.empty((t, 1000, 5))
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(t):
            std_shocks = rng((1000, 5))
            paths[i, :, 0] = np.log(one_step[i])
            for j in range(1, 5):
                paths[i, :, j] = (
                    params[0]
                    + params[1] * (np.abs(std_shocks[:, j - 1]) - sqrt2pi)
                    + params[2] * paths[i, :, j - 1]
                )
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths.mean(1))
        assert_allclose(forecast.forecast_paths, paths)
        assert_allclose(forecast.shocks, shocks)

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=5,
                start=100,
                method="bootstrap",
                random_state=self.rng,
            )

        std_resids = resids / np.sqrt(sigma2)
        paths = np.empty((t, 1000, 5))
        paths.fill(np.nan)
        shocks = np.empty((t, 1000, 5))
        shocks.fill(np.nan)
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(100, t):
            locs = self.rng.random_sample((1000, 5))
            int_locs = np.floor((i + 1) * locs).astype(np.int64)
            std_shocks = std_resids[int_locs]
            paths[i, :, 0] = np.log(one_step[i])
            for j in range(1, 5):
                paths[i, :, j] = (
                    params[0]
                    + params[1] * (np.abs(std_shocks[:, j - 1]) - sqrt2pi)
                    + params[2] * paths[i, :, j - 1]
                )
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths[100:].mean(1))
        assert_allclose(forecast.forecast_paths, paths[100:])
        assert_allclose(forecast.shocks, shocks[100:])

    def test_egarch_211_forecast(self):
        t = self.t
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        vol = EGARCH(p=2, o=1, q=1)
        params = np.array([0.0, 0.15, 0.05, 0.1, 0.95])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)

        forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=1, start=0
        )

        _resids = np.array(resids.tolist() + [0])
        _sigma2 = np.empty_like(_resids)
        _var_bounds = np.array(var_bounds.tolist() + [[0, np.inf]])
        vol.compute_variance(params, _resids, _sigma2, backcast, _var_bounds)
        one_step = _sigma2[1:][:, None]

        assert_allclose(forecast.forecasts, one_step)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        expected = forecast.forecasts.copy()
        expected[:-1] = np.nan
        forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=1)
        assert_allclose(forecast.forecasts, expected[-1:])
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=7,
                start=433,
                method="simulation",
                rng=rng,
            )

        sigma2 = np.empty_like(self.resid)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        std_resids = resids / np.sqrt(sigma2)
        paths = np.empty((t, 1000, 7))
        paths.fill(np.nan)
        shocks = np.empty((t, 1000, 7))
        shocks.fill(np.nan)
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(433, t):
            std_shocks = rng((1000, 7))
            paths[i, :, 0] = np.log(one_step[i])

            for j in range(1, 7):
                lag_1_sym = np.abs(std_shocks[:, j - 1]) - sqrt2pi
                if j > 1:
                    lag_2_sym = np.abs(std_shocks[:, j - 2]) - sqrt2pi
                else:
                    lag_2_sym = np.abs(std_resids[i]) - sqrt2pi

                paths[i, :, j] = (
                    params[0]
                    + params[1] * lag_1_sym
                    + params[2] * lag_2_sym
                    + params[3] * std_shocks[:, j - 1]
                    + params[4] * paths[i, :, j - 1]
                )
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths[433:].mean(1))
        assert_allclose(forecast.forecast_paths, paths[433:])
        assert_allclose(forecast.shocks, shocks[433:])

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=3,
                start=731,
                method="bootstrap",
                simulations=1111,
                random_state=self.rng,
            )
        paths = np.empty((t, 1111, 3))
        paths.fill(np.nan)
        shocks = np.empty((t, 1111, 3))
        shocks.fill(np.nan)
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(731, t):
            locs = self.rng.random_sample((1111, 3))
            int_locs = np.floor((i + 1) * locs).astype(np.int64)
            std_shocks = std_resids[int_locs]
            paths[i, :, 0] = np.log(one_step[i])

            for j in range(1, 3):
                lag_1_sym = np.abs(std_shocks[:, j - 1]) - sqrt2pi
                if j > 1:
                    lag_2_sym = np.abs(std_shocks[:, j - 2]) - sqrt2pi
                else:
                    lag_2_sym = np.abs(std_resids[i]) - sqrt2pi

                paths[i, :, j] = (
                    params[0]
                    + params[1] * lag_1_sym
                    + params[2] * lag_2_sym
                    + params[3] * std_shocks[:, j - 1]
                    + params[4] * paths[i, :, j - 1]
                )
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths[731:].mean(1))
        assert_allclose(forecast.forecast_paths, paths[731:])
        assert_allclose(forecast.shocks, shocks[731:])

    def test_egarch_212_forecast_smoke(self):
        t = self.t
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        vol = EGARCH(p=2, o=1, q=2)
        params = np.array([0.0, 0.15, 0.05, 0.1, 0.55, 0.4])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)

        forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=1, start=0
        )

        _resids = np.array(resids.tolist() + [0])
        _sigma2 = np.empty_like(_resids)
        _var_bounds = np.array(var_bounds.tolist() + [[0, np.inf]])
        vol.compute_variance(params, _resids, _sigma2, backcast, _var_bounds)
        one_step = _sigma2[1:][:, None]

        assert_allclose(forecast.forecasts, one_step)
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        expected = forecast.forecasts.copy()
        expected[:-1] = np.nan
        forecast = vol.forecast(params, resids, backcast, var_bounds, horizon=1)
        assert_allclose(forecast.forecasts, expected[-1:])
        assert forecast.forecast_paths is None
        assert forecast.shocks is None

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=7,
                start=433,
                method="simulation",
                rng=rng,
            )

        sigma2 = np.empty_like(self.resid)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        std_resids = resids / np.sqrt(sigma2)
        paths = np.empty((t, 1000, 7))
        paths.fill(np.nan)
        shocks = np.empty((t, 1000, 7))
        shocks.fill(np.nan)
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(433, t):
            std_shocks = rng((1000, 7))
            paths[i, :, 0] = np.log(one_step[i])

            for j in range(1, 7):
                lag_1_sym = np.abs(std_shocks[:, j - 1]) - sqrt2pi
                lag_1_path = paths[i, :, j - 1]
                if j > 1:
                    lag_2_sym = np.abs(std_shocks[:, j - 2]) - sqrt2pi
                    lag_2_path = paths[i, :, j - 2]
                else:
                    lag_2_sym = np.abs(std_resids[i]) - sqrt2pi
                    lag_2_path = np.log(sigma2[i])

                paths[i, :, j] = (
                    params[0]
                    + params[1] * lag_1_sym
                    + params[2] * lag_2_sym
                    + params[3] * std_shocks[:, j - 1]
                    + params[4] * lag_1_path
                    + params[5] * lag_2_path
                )
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths[433:].mean(1))
        assert_allclose(forecast.forecast_paths, paths[433:])
        assert_allclose(forecast.shocks, shocks[433:])

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=3,
                start=731,
                method="bootstrap",
                simulations=1111,
                random_state=self.rng,
            )
        paths = np.empty((t, 1111, 3))
        paths.fill(np.nan)
        shocks = np.empty((t, 1111, 3))
        shocks.fill(np.nan)
        sqrt2pi = np.sqrt(2 / np.pi)
        for i in range(731, t):
            locs = self.rng.random_sample((1111, 3))
            int_locs = np.floor((i + 1) * locs).astype(np.int64)
            std_shocks = std_resids[int_locs]
            paths[i, :, 0] = np.log(one_step[i])

            for j in range(1, 3):
                lag_1_sym = np.abs(std_shocks[:, j - 1]) - sqrt2pi
                lag_1_path = paths[i, :, j - 1]
                if j > 1:
                    lag_2_sym = np.abs(std_shocks[:, j - 2]) - sqrt2pi
                    lag_2_path = paths[i, :, j - 2]
                else:
                    lag_2_sym = np.abs(std_resids[i]) - sqrt2pi
                    lag_2_path = np.log(sigma2[i])

                paths[i, :, j] = (
                    params[0]
                    + params[1] * lag_1_sym
                    + params[2] * lag_2_sym
                    + params[3] * std_shocks[:, j - 1]
                    + params[4] * lag_1_path
                    + params[5] * lag_2_path
                )
            shocks[i, :, :] = np.exp(paths[i, :, :] / 2) * std_shocks
        paths = np.exp(paths)
        assert_allclose(forecast.forecasts, paths[731:].mean(1))
        assert_allclose(forecast.forecast_paths, paths[731:])
        assert_allclose(forecast.shocks, shocks[731:])

    def test_constant_variance_simulation(self):
        t = self.t
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        vol = ConstantVariance()
        params = np.array([10.0])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=1,
                method="simulation",
                rng=rng,
            )
        assert forecast.forecasts.shape == (1, 1)
        assert forecast.forecast_paths.shape == (1, 1000, 1)
        assert forecast.shocks.shape == (1, 1000, 1)
        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(forecast.forecasts == params[0])
        assert np.all(np.isfinite(forecast.forecast_paths))
        assert np.all(forecast.forecast_paths == params[0])
        assert np.all(np.isfinite(forecast.shocks))
        assert_allclose(forecast.shocks[-1], np.sqrt(params[0]) * rng((1000, 1)))

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=5,
                method="simulation",
                rng=rng,
            )
        assert forecast.forecasts.shape == (1, 5)
        assert forecast.forecast_paths.shape == (1, 1000, 5)
        assert forecast.shocks.shape == (1, 1000, 5)
        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(forecast.forecasts[-1] == params[0])
        assert np.all(np.isfinite(forecast.forecast_paths))
        assert np.all(forecast.forecast_paths[-1] == params[0])
        assert_allclose(forecast.shocks[-1], np.sqrt(params[0]) * rng((1000, 5)))

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=5,
                start=100,
                method="simulation",
                simulations=2000,
                rng=rng,
            )
        # TODO: This is not correct.  Should be (900,5)
        assert forecast.forecasts.shape == (900, 5)
        assert forecast.forecast_paths.shape == (900, 2000, 5)
        assert forecast.shocks.shape == (900, 2000, 5)
        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(forecast.forecasts == params[0])
        assert np.all(np.isfinite(forecast.forecast_paths))
        assert np.all(forecast.forecast_paths == params[0])
        expected = np.sqrt(params[0]) * rng((t - 100, 2000, 5))
        expected = np.concatenate((np.empty((100, 2000, 5)), expected))
        assert_allclose(forecast.shocks, expected[100:])

    def test_constant_variance_bootstrap(self):
        t = self.t
        vol = ConstantVariance()
        params = np.array([10.0])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=5,
                method="bootstrap",
                random_state=self.rng,
            )
        assert forecast.forecasts.shape == (1, 5)
        assert forecast.forecast_paths.shape == (1, 1000, 5)
        assert forecast.shocks.shape == (1, 1000, 5)
        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(forecast.forecasts[-1] == params[0])
        assert np.all(np.isfinite(forecast.forecast_paths))
        assert np.all(forecast.forecast_paths[-1] == params[0])
        index = np.floor(self.rng.random_sample((1000, 5)) * t)
        index = index.astype(np.int64)
        assert_allclose(forecast.shocks[-1], self.resid[index])

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=5,
                method="bootstrap",
                start=100,
                simulations=2000,
                random_state=self.rng,
            )
        assert forecast.forecasts.shape == (900, 5)
        assert forecast.forecast_paths.shape == (900, 2000, 5)
        assert forecast.shocks.shape == (900, 2000, 5)
        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(forecast.forecasts == params[0])
        assert np.all(np.isfinite(forecast.forecast_paths))
        assert np.all(forecast.forecast_paths == params[0])
        expected = np.empty((1000, 2000, 5))
        expected.fill(np.nan)
        for i in range(100, 1000):
            index = self.rng.random_sample((2000, 5))
            int_index = np.floor((i + 1) * index).astype(np.int64)
            expected[i] = self.resid[int_index]
        assert_allclose(forecast.shocks, expected[100:])

    def test_garch11_simulation(self):
        t = self.t
        vol = GARCH(p=1, o=0, q=1)
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        params = np.array([10.0, 0.1, 0.85])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=10,
                method="simulation",
                rng=rng,
            )
        assert forecast.forecasts.shape == (1, 10)
        assert forecast.forecast_paths.shape == (1, 1000, 10)
        assert forecast.shocks.shape == (1, 1000, 10)
        assert np.all(np.isfinite(forecast.forecast_paths))
        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(np.isfinite(forecast.shocks))
        std_shocks = rng((1000, 10))
        one_step = vol.forecast(
            params, self.resid, backcast, var_bounds, horizon=1, start=0
        )
        paths = np.zeros((1000, 10))
        shocks = np.zeros((1000, 10))
        paths[:, 0] = one_step.forecasts[-1]
        shocks[:, 0] = np.sqrt(paths[:, 0]) * std_shocks[:, 0]
        for i in range(1, 10):
            paths[:, i] = params[0]
            paths[:, i] += params[1] * shocks[:, i - 1] ** 2.0
            paths[:, i] += params[2] * paths[:, i - 1]
            shocks[:, i] = np.sqrt(paths[:, i]) * std_shocks[:, i]
        forecasts = paths.mean(0)
        assert_allclose(forecast.forecasts[-1], forecasts)
        assert_allclose(forecast.forecast_paths[-1], paths)
        assert_allclose(forecast.shocks[-1], shocks)

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=10,
                method="simulation",
                simulations=2000,
                rng=rng,
            )
        assert np.all(np.isfinite(forecast.forecast_paths))
        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(np.isfinite(forecast.shocks))
        assert forecast.forecasts.shape == (1, 10)
        assert forecast.forecast_paths.shape == (1, 2000, 10)
        assert forecast.shocks.shape == (1, 2000, 10)
        std_shocks = rng((2000, 10))
        paths = np.zeros((2000, 10))
        shocks = np.zeros((2000, 10))
        paths[:, 0] = one_step.forecasts[-1]
        shocks[:, 0] = np.sqrt(paths[:, 0]) * std_shocks[:, 0]
        for i in range(1, 10):
            paths[:, i] = params[0]
            paths[:, i] += params[1] * shocks[:, i - 1] ** 2.0
            paths[:, i] += params[2] * paths[:, i - 1]
            shocks[:, i] = np.sqrt(paths[:, i]) * std_shocks[:, i]
        forecasts = paths.mean(0)
        assert_allclose(forecast.forecasts[-1], forecasts)
        assert_allclose(forecast.forecast_paths[-1], paths)
        assert_allclose(forecast.shocks[-1], shocks)

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=3,
                method="simulation",
                start=0,
                rng=rng,
            )
        paths = np.zeros((1000, 3))
        shocks = np.zeros((1000, 3))
        for j in range(t):
            std_shocks = rng((1000, 3))
            paths[:, 0] = one_step.forecasts[j]
            shocks[:, 0] = np.sqrt(paths[:, 0]) * std_shocks[:, 0]
            for i in range(1, 3):
                paths[:, i] = params[0]
                paths[:, i] += params[1] * shocks[:, i - 1] ** 2.0
                paths[:, i] += params[2] * paths[:, i - 1]
                shocks[:, i] = np.sqrt(paths[:, i]) * std_shocks[:, i]
            forecasts = paths.mean(0)
            assert_allclose(forecast.forecasts[j], forecasts)
            assert_allclose(forecast.forecast_paths[j], paths)
            assert_allclose(forecast.shocks[j], shocks)

    def test_garch11_bootstrap(self):
        t = self.t
        vol = GARCH(p=1, o=0, q=1)
        params = np.array([10.0, 0.1, 0.85])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=10,
                method="bootstrap",
                start=100,
                random_state=self.rng,
            )

        paths = np.empty((t, 1000, 10))
        paths.fill(np.nan)
        shocks = np.empty((t, 1000, 10))
        shocks.fill(np.nan)

        _sigma2 = np.empty(t + 1)
        _resids = np.concatenate((self.resid, [0]))
        _var_bounds = np.concatenate((var_bounds, [[0, np.inf]]))
        vol.compute_variance(params, _resids, _sigma2, backcast, _var_bounds)
        std_resid = self.resid / np.sqrt(_sigma2[:-1])
        one_step = _sigma2[1:]
        omega, alpha, beta = params
        for i in range(100, t):
            locs = np.floor((i + 1) * self.rng.random_sample((1000, 10))).astype(
                np.int64
            )
            std_shocks = std_resid[locs]
            paths[i, :, 0] = one_step[i]
            shocks[i, :, 0] = np.sqrt(paths[i, :, 0]) * std_shocks[:, 0]
            for j in range(1, 10):
                paths[i, :, j] = (
                    omega
                    + alpha * shocks[i, :, j - 1] ** 2.0
                    + beta * paths[i, :, j - 1]
                )
                shocks[i, :, j] = np.sqrt(paths[i, :, j]) * std_shocks[:, j]

        assert_allclose(forecast.forecasts, paths[100:].mean(1))
        assert_allclose(forecast.forecast_paths, paths[100:])
        assert_allclose(forecast.shocks, shocks[100:])

        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=10,
                method="bootstrap",
                simulations=2000,
                start=100,
            )

        assert forecast.forecasts.shape == (t - 100, 10)
        assert forecast.forecast_paths.shape == (t - 100, 2000, 10)
        assert forecast.shocks.shape == (t - 100, 2000, 10)

        assert np.all(np.isfinite(forecast.forecasts))
        assert np.all(np.isfinite(forecast.forecast_paths))
        assert np.all(np.isfinite(forecast.shocks))

    def test_gjrgarch222_simulation(self):
        vol = GARCH(p=2, o=2, q=2)
        params = np.array([10.0, 0.05, 0.03, 0.1, 0.05, 0.3, 0.2])
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        backcast = vol.backcast(self.resid)
        var_bounds = vol.variance_bounds(self.resid)
        one_step = vol.forecast(
            params, self.resid, backcast, var_bounds, horizon=1, start=0
        )
        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                self.resid,
                backcast,
                var_bounds,
                horizon=3,
                method="simulation",
                start=0,
                rng=rng,
                simulations=100,
            )
        paths = np.zeros((100, 3))
        shocks = np.zeros((100, 3))

        resids = self.resid
        sigma2_0 = (
            params[0] + (np.sum(params[1:]) - 0.5 * np.sum(params[3:5])) * backcast
        )
        _sigma2 = np.concatenate([[backcast], [sigma2_0], one_step.forecasts[:-1, 0]])
        _resids = np.concatenate([[np.sqrt(backcast)], resids])
        _asymresids = np.concatenate([[np.sqrt(0.5 * backcast)], resids * (resids < 0)])

        for t in range(resids.shape[0]):
            std_shocks = rng((100, 3))

            j = 0
            tau = t + 1 + j + 1
            r1, r2 = _resids[tau - 1], _resids[tau - 2]
            a1, a2 = _asymresids[tau - 1], _asymresids[tau - 2]
            s21, s22 = _sigma2[tau - 1], _sigma2[tau - 2]

            fcast = (
                params[0]
                + params[1] * r1**2
                + params[2] * r2**2
                + params[3] * a1**2
                + params[4] * a2**2
                + params[5] * s21
                + params[6] * s22
            )
            paths[:, j] = fcast
            shocks[:, j] = std_shocks[:, j] * np.sqrt(fcast)

            j = 1
            r1, r2 = shocks[:, 0], _resids[tau - 1]
            a1, a2 = shocks[:, 0] * (shocks[:, 0] < 0), _asymresids[tau - 1]
            s21, s22 = paths[:, 0], _sigma2[tau - 1]
            fcast = (
                params[0]
                + params[1] * r1**2
                + params[2] * r2**2
                + params[3] * a1**2
                + params[4] * a2**2
                + params[5] * s21
                + params[6] * s22
            )
            paths[:, j] = fcast
            shocks[:, j] = std_shocks[:, j] * np.sqrt(fcast)

            j = 2
            r1, r2 = shocks[:, 1], shocks[:, 0]
            a1, a2 = (
                shocks[:, 1] * (shocks[:, 1] < 0),
                shocks[:, 0] * (shocks[:, 0] < 0),
            )
            s21, s22 = paths[:, 1], paths[:, 0]
            fcast = (
                params[0]
                + params[1] * r1**2
                + params[2] * r2**2
                + params[3] * a1**2
                + params[4] * a2**2
                + params[5] * s21
                + params[6] * s22
            )
            paths[:, j] = fcast
            shocks[:, j] = std_shocks[:, j] * np.sqrt(fcast)

            forecasts = paths.mean(0)
            assert_allclose(forecast.forecasts[t], forecasts)
            assert_allclose(forecast.forecast_paths[t], paths)
            assert_allclose(forecast.shocks[t], shocks)

    def test_ewma_forecast(self):
        t = self.t
        vol = EWMAVariance()
        params = np.array([])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)

        forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=10, start=0
        )

        expected = np.zeros(t + 1)
        expected[0] = backcast
        lam = 0.94
        for i in range(1, t + 1):
            expected[i] = lam * expected[i - 1] + (1 - lam) * resids[i - 1] ** 2
        for i in range(10):
            assert_allclose(forecast.forecasts[:, i], expected[1:])

        alt_forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=10, start=500
        )
        _compare_truncated_forecasts(forecast, alt_forecast, 500)

        vol_estim = EWMAVariance(lam=None)
        param_estim = np.array([0.94])
        forecast_estim = vol_estim.forecast(
            param_estim, resids, backcast, var_bounds, horizon=10, start=0
        )
        for i in range(10):
            assert_allclose(forecast.forecasts[:, i], forecast_estim.forecasts[:, i])

    def test_ewma_simulation(self):
        t = self.t
        vol = EWMAVariance()
        params = np.array([])
        resids = self.resid
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        with preserved_state(self.rng):
            forecasts = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=0,
                method="simulation",
                rng=rng,
            )

        vol_estim = EWMAVariance(lam=None)
        param_estim = np.array([0.94])
        with preserved_state(self.rng):
            forecasts_estim = vol_estim.forecast(
                param_estim,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=0,
                method="simulation",
                rng=rng,
            )

        one_step = np.empty(t + 1)
        one_step[0] = backcast
        for i in range(1, t + 1):
            one_step[i] = vol.lam * one_step[i - 1] + (1 - vol.lam) * resids[i - 1] ** 2
        one_step = one_step[1:]
        paths = np.empty((t, 1000, 10))
        shocks = np.empty((t, 1000, 10))
        for i in range(t):
            std_shocks = rng((1000, 10))
            paths[i, :, 0] = one_step[i]
            shocks[i, :, 0] = np.sqrt(paths[i, :, 0]) * std_shocks[:, 0]
            for j in range(1, 10):
                paths[i, :, j] = (
                    vol.lam * paths[i, :, j - 1]
                    + (1 - vol.lam) * shocks[i, :, j - 1] ** 2
                )
                shocks[i, :, j] = np.sqrt(paths[i, :, j]) * std_shocks[:, j]

        assert_allclose(forecasts.forecasts, paths.mean(1))
        assert_allclose(forecasts.forecast_paths, paths)
        assert_allclose(forecasts.shocks, shocks)

        assert_allclose(forecasts_estim.forecasts, paths.mean(1))
        assert_allclose(forecasts_estim.forecast_paths, paths)
        assert_allclose(forecasts_estim.shocks, shocks)

        forecasts = vol.forecast(
            params,
            resids,
            backcast,
            var_bounds,
            horizon=10,
            start=252,
            method="simulation",
            rng=rng,
        )

        assert forecasts.forecasts.shape == (1000 - 252, 10)
        assert forecasts.forecast_paths.shape == (1000 - 252, 1000, 10)
        assert forecasts.shocks.shape == (1000 - 252, 1000, 10)

        assert np.all(np.isfinite(forecasts.forecasts))
        assert np.all(np.isfinite(forecasts.forecast_paths))
        assert np.all(np.isfinite(forecasts.shocks))

    def test_ewma_bootstrap(self):
        t = self.t
        vol = EWMAVariance()
        params = np.array([])
        resids = self.resid
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        with preserved_state(self.rng):
            forecasts = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=131,
                method="bootstrap",
                rng=rng,
                random_state=self.rng,
            )

        sigma2 = np.empty_like(self.resid)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        std_resids = resids / np.sqrt(sigma2)

        one_step = np.empty(t + 1)
        one_step[0] = backcast
        for i in range(1, t + 1):
            one_step[i] = vol.lam * one_step[i - 1] + (1 - vol.lam) * resids[i - 1] ** 2
        one_step = one_step[1:]
        paths = np.empty((t, 1000, 10))
        paths.fill(np.nan)
        shocks = np.empty((t, 1000, 10))
        shocks.fill(np.nan)
        for i in range(131, t):
            locs = self.rng.random_sample((1000, 10))
            int_locs = np.floor(locs * (i + 1)).astype(np.int_)
            std_shocks = std_resids[int_locs]
            paths[i, :, 0] = one_step[i]
            shocks[i, :, 0] = np.sqrt(paths[i, :, 0]) * std_shocks[:, 0]
            for j in range(1, 10):
                paths[i, :, j] = (
                    vol.lam * paths[i, :, j - 1]
                    + (1 - vol.lam) * shocks[i, :, j - 1] ** 2
                )
                shocks[i, :, j] = np.sqrt(paths[i, :, j]) * std_shocks[:, j]

        assert_allclose(forecasts.shocks, shocks[131:])
        assert_allclose(forecasts.forecasts, paths[131:].mean(1))
        assert_allclose(forecasts.forecast_paths, paths[131:])

        forecasts = vol.forecast(
            params,
            resids,
            backcast,
            var_bounds,
            horizon=10,
            start=252,
            method="simulation",
            rng=rng,
            random_state=self.rng,
        )
        assert forecasts.forecasts.shape == (1000 - 252, 10)
        assert forecasts.forecast_paths.shape == (1000 - 252, 1000, 10)
        assert forecasts.shocks.shape == (1000 - 252, 1000, 10)
        assert np.all(np.isfinite(forecasts.forecasts))
        assert np.all(np.isfinite(forecasts.forecast_paths))
        assert np.all(np.isfinite(forecasts.shocks))

    def test_rm2006_forecast(self):
        vol = RiskMetrics2006()
        params = np.array([])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        forecasts = vol.forecast(
            params, resids, backcast, var_bounds, horizon=10, start=0
        )
        _resids = np.array(resids.tolist() + [0])
        _sigma2 = np.empty_like(_resids)
        _var_bounds = np.array(var_bounds.tolist() + [[0, np.inf]])
        vol.compute_variance(params, _resids, _sigma2, backcast, _var_bounds)
        assert forecasts.forecast_paths is None
        assert forecasts.shocks is None
        one_step = _sigma2[1:]
        assert_allclose(forecasts.forecasts[:, 0], one_step, rtol=1e-4)
        # The components are EWMAs of squared residuals, and the expected
        # squared residual is the combined variance
        w = vol._ewma_combination_weights()
        mus = vol._ewma_smoothing_parameters()
        resids2 = np.asarray(resids) ** 2
        components = np.empty((resids2.shape[0] + 1, mus.shape[0]))
        components[0] = backcast
        for t in range(resids2.shape[0]):
            components[t + 1] = mus * components[t] + (1 - mus) * resids2[t]
        expected = components[1:]
        for i in range(10):
            assert_allclose(forecasts.forecasts[:, i], expected @ w, rtol=1e-4)
            expected = mus * expected + (1 - mus) * (expected @ w)[:, None]

        alt_forecasts = vol.forecast(
            params, resids, backcast, var_bounds, horizon=10, start=500
        )
        _compare_truncated_forecasts(forecasts, alt_forecasts, 500)

    def test_rm2006_simulation_smoke(self):
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        vol = RiskMetrics2006()
        params = np.array([])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        with preserved_state(self.rng):
            vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=0,
                method="simulation",
                rng=rng,
            )

    def test_rm2006_bootstrap_smoke(self):
        vol = RiskMetrics2006()
        params = np.array([])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        with preserved_state(self.rng):
            vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=100,
                method="bootstrap",
            )

    @pytest.mark.parametrize("method", ["simulation", "bootstrap"])
    def test_rm2006_simulation_one_step(self, method):
        vol = RiskMetrics2006()
        params = np.array([])
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        analytic = vol.forecast(
            params, resids, backcast, var_bounds, horizon=3, start=900
        )
        kwargs = {}
        if method == "simulation":
            kwargs["rng"] = Normal(seed=RandomState(12345)).simulate([])
        else:
            kwargs["random_state"] = RandomState(12345)
        sim = vol.forecast(
            params,
            resids,
            backcast,
            var_bounds,
            horizon=3,
            start=900,
            method=method,
            simulations=10,
            **kwargs,
        )
        # The one-step-ahead variance is known at the forecast origin
        assert_allclose(sim.forecasts[:, 0], analytic.forecasts[:, 0])
        assert_allclose(
            sim.forecast_paths[:, :, 0],
            np.tile(analytic.forecasts[:, :1], (1, 10)),
        )

    def test_aparch_one_step(self):
        vol = APARCH()
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        params = np.array([0.1, 0.1, -0.5, 0.8, 1.5])
        forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=1, start=0
        )
        sigma2 = np.empty_like(resids)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        assert_allclose(sigma2[1:], forecast.forecasts[:-1, 0])
        delta = params[-1]
        final = params[0]
        final += params[1] * ((np.abs(resids[-1]) - params[2] * resids[-1]) ** delta)
        final += params[3] * (sigma2[-1] ** (delta / 2.0))
        final **= 2.0 / delta
        assert_allclose(final, forecast.forecasts[-1, 0])

        delta = 2.0
        vol = APARCH(delta=delta)
        params = np.array([0.1, 0.1, -0.5, 0.8])
        sigma2 = np.empty_like(resids)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=1, start=0
        )
        assert_allclose(sigma2[1:], forecast.forecasts[:-1, 0])
        final = params[0]
        final += params[1] * ((np.abs(resids[-1]) - params[2] * resids[-1]) ** delta)
        final += params[3] * (sigma2[-1] ** (delta / 2.0))
        final **= 2.0 / delta
        assert_allclose(final, forecast.forecasts[-1, 0])

    @pytest.mark.parametrize(
        ("p", "o", "q", "params", "delta"),
        [
            (2, 1, 1, [0.1, 0.1, 0.05, 0.3, 0.3, 1.5], None),
            (2, 2, 1, [0.1, 0.1, 0.05, 0.3, -0.2, 0.3, 1.5], None),
            (3, 1, 0, [0.1, 0.1, 0.08, 0.05, 0.3, 1.3], None),
            (2, 1, 1, [0.1, 0.1, 0.05, 0.3, 0.3], 1.5),
            (1, 1, 1, [0.1, 0.1, 0.3, 0.3, 1.5], None),
        ],
        ids=["211", "221", "310", "211-fixed-delta", "111"],
    )
    @pytest.mark.parametrize("method", ["simulation", "bootstrap"])
    def test_aparch_simulation_matches_hand_recursion(
        self, p, o, q, params, delta, method
    ):
        # Forecast paths must follow the APARCH recursion written out by hand
        # when fed the same standardized shocks. In particular |e| - gamma * e
        # may only be used in the period it is applied to
        params = np.array(params)
        delta_value = params[-1] if delta is None else delta
        horizon = 4
        sims = 12
        n = 130
        resids = np.random.default_rng(20240101).standard_t(8, n) * 1.2
        vol = APARCH(p=p, o=o, q=q, delta=delta)
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        args = (params, p, o, q, delta_value, backcast)
        history_sd = _aparch_hand_history(resids, *args)
        starts = [0, 1, 110] if method == "simulation" else [110]
        for start in starts:
            if method == "simulation":
                generator = np.random.default_rng(start + 1)
                draws = [generator.standard_normal((sims, horizon)) for _ in range(n)]
                calls = iter(draws[start:])

                def rng(size, calls=calls):
                    return next(calls)

                kwargs = {"rng": rng}
            else:
                # Standardized residuals from the hand recursion, replayed
                std_resid = resids / np.array(history_sd) ** (1.0 / delta_value)
                replay = BootstrapRng(std_resid, start, RandomState(7)).rng()
                draws = [None] * start
                draws += [replay((sims, horizon)) for _ in range(n - start)]
                kwargs = {"random_state": RandomState(7)}
            forecast = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=horizon,
                start=start,
                method=method,
                simulations=sims,
                **kwargs,
            )
            for origin in range(start, n):
                paths, shocks = _aparch_hand_simulation(
                    resids, history_sd, *args, origin, draws[origin]
                )
                loc = origin - start
                assert_allclose(forecast.forecast_paths[loc], paths, rtol=1e-10)
                assert_allclose(forecast.shocks[loc], shocks, rtol=1e-10)
                assert_allclose(forecast.forecasts[loc], paths.mean(0), rtol=1e-10)
            # The one-step variance is known at the origin
            analytic = vol.forecast(
                params, resids, backcast, var_bounds, horizon=1, start=start
            )
            assert_allclose(
                forecast.forecast_paths[:, :, 0],
                np.tile(analytic.forecasts, (1, sims)),
                rtol=1e-10,
            )

    @pytest.mark.parametrize("o", [0, 1])
    @pytest.mark.parametrize("delta", [None, 1.5])
    def test_aparch_simulation_smoke(self, o, delta):
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        vol = APARCH(o=o, delta=delta)
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        params = np.array([0.1, 0.1, -0.5, 0.8])
        if o == 0:
            params = np.array([0.1, 0.1, 0.8])
        if delta is None:
            params = np.r_[params, 1.5]
        forecast = vol.forecast(
            params,
            resids,
            backcast,
            var_bounds,
            horizon=10,
            start=0,
            method="simulation",
            rng=rng,
            simulations=100,
        )
        sigma2 = np.empty_like(resids)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        assert_allclose(sigma2[1:], forecast.forecasts[:-1, 0])
        delta = 1.5 if delta is None else delta
        final = params[0]
        gamma = 0.0 if o == 0 else params[2]
        final += params[1] * ((np.abs(resids[-1]) - gamma * resids[-1]) ** delta)
        beta = params[2 + int(o > 0)]
        final += beta * (sigma2[-1] ** (delta / 2.0))
        final **= 2.0 / delta
        assert_allclose(final, forecast.forecasts[-1, 0])
        with pytest.raises(ValueError, match=r"Analytic forecasts not"):
            vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=0,
                method="analytic",
            )

    def test_fiaparch_one_step(self):
        trunc = 50
        vol = FIAPARCH(truncation=trunc)
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        params = np.array([0.1, 0.2, 0.4, 0.2, -0.3, 1.5])
        sigma2 = np.empty_like(resids)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=1, start=0
        )
        assert_allclose(sigma2[1:], forecast.forecasts[:-1, 0])

        delta = 1.5
        vol_fixed = FIAPARCH(truncation=trunc, delta=delta)
        params_fixed = np.array([0.1, 0.2, 0.4, 0.2, -0.3])
        sigma2_f = np.empty_like(resids)
        vol_fixed.compute_variance(params_fixed, resids, sigma2_f, backcast, var_bounds)
        forecast_f = vol_fixed.forecast(
            params_fixed, resids, backcast, var_bounds, horizon=1, start=0
        )
        assert_allclose(sigma2_f[1:], forecast_f.forecasts[:-1, 0])

        with pytest.raises(
            ValueError, match=r"Analytic forecasts not available for horizon"
        ):
            vol.forecast(
                params, resids, backcast, var_bounds, horizon=2, method="analytic"
            )

    @pytest.mark.parametrize("o", [0, 1])
    @pytest.mark.parametrize("delta", [None, 1.5])
    def test_fiaparch_simulation_smoke(self, o, delta):
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        trunc = 50
        vol = FIAPARCH(o=o, delta=delta, truncation=trunc)
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        if o == 0:
            params_ = [0.1, 0.2, 0.4, 0.2]
        else:
            params_ = [0.1, 0.2, 0.4, 0.2, -0.3]
        if delta is None:
            params_ += [1.5]
        params = np.array(params_)
        sigma2 = np.empty_like(resids)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        forecast = vol.forecast(
            params,
            resids,
            backcast,
            var_bounds,
            horizon=10,
            start=0,
            method="simulation",
            rng=rng,
            simulations=100,
        )
        assert_allclose(sigma2[1:], forecast.forecasts[:-1, 0])
        assert forecast.forecast_paths is not None
        assert forecast.shocks is not None
        with pytest.raises(ValueError, match=r"Analytic forecasts not"):
            vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=0,
                method="analytic",
            )

    @pytest.mark.parametrize(("p", "q"), [(0, 1), (1, 0), (0, 0)])
    def test_fiaparch_one_step_reduced(self, p, q):
        trunc = 50
        vol = FIAPARCH(p=p, q=q, truncation=trunc)
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        params = [0.1]
        if p:
            params.append(0.2)
        params.append(0.4)
        if q:
            params.append(0.2)
        params.extend([-0.3, 1.5])
        params = np.array(params)
        sigma2 = np.empty_like(resids)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=1, start=0
        )
        assert_allclose(sigma2[1:], forecast.forecasts[:-1, 0])

    @pytest.mark.parametrize(("p", "q"), [(0, 1), (1, 0), (0, 0)])
    def test_fiaparch_simulation_reduced(self, p, q):
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        trunc = 50
        vol = FIAPARCH(p=p, q=q, truncation=trunc)
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        params = [0.1]
        if p:
            params.append(0.2)
        params.append(0.4)
        if q:
            params.append(0.2)
        params.extend([-0.3, 1.5])
        params = np.array(params)
        sigma2 = np.empty_like(resids)
        vol.compute_variance(params, resids, sigma2, backcast, var_bounds)
        forecast = vol.forecast(
            params,
            resids,
            backcast,
            var_bounds,
            horizon=10,
            start=0,
            method="simulation",
            rng=rng,
            simulations=100,
        )
        assert_allclose(sigma2[1:], forecast.forecasts[:-1, 0])
        assert forecast.forecast_paths is not None
        assert forecast.shocks is not None

    def test_midas_analytical(self):
        vol = MIDASHyperbolic()
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        params = np.array([0.1, 0.9, 0.4])
        forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=10, start=0
        )
        weights = vol._weights(params)
        arch_params = np.r_[params[0], params[1] * weights]
        expected = _simple_direct_gjrgarch_forecaster(
            resids, arch_params, 22, 0, 0, backcast, var_bounds, 10
        )
        assert_allclose(forecast.forecasts, expected)

    def test_midas_asym_analytical(self):
        vol = MIDASHyperbolic(asym=True)
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        params = np.array([0.1, 0.3, 1.0, 0.4])
        forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=10, start=0
        )
        weights = vol._weights(params)
        arch_params = np.r_[params[0], params[1] * weights, params[2] * weights]
        expected = _simple_direct_gjrgarch_forecaster(
            resids, arch_params, 22, 22, 0, backcast, var_bounds, 10
        )
        assert_allclose(forecast.forecasts, expected)

    def test_midas_simulation(self):
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        vol = MIDASHyperbolic()
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        params = np.array([0.1, 0.9, 0.4])
        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=0,
                method="simulation",
                rng=rng,
            )

        weights = vol._weights(params)
        arch = GARCH(p=22, o=0, q=0)
        arch_params = np.r_[params[0], params[1] * weights]
        with preserved_state(self.rng):
            arch_forecast = arch.forecast(
                arch_params,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=0,
                method="simulation",
                rng=rng,
            )

        assert_allclose(forecast.forecasts, arch_forecast.forecasts)
        assert_allclose(forecast.forecast_paths, arch_forecast.forecast_paths)
        assert_allclose(forecast.shocks, arch_forecast.shocks)

    def test_figarch_analytical(self):
        vol = FIGARCH(truncation=50)
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        params = np.array([0.1, 0.2, 0.4, 0.2])
        forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=10, start=0
        )
        lam = figarch_weights(params[1:], 1, 1, vol.truncation)
        arch_params = np.r_[params[0] / (1 - params[-1]), lam]
        expected = _simple_direct_gjrgarch_forecaster(
            resids, arch_params, 50, 0, 0, backcast, var_bounds, 10
        )
        assert_allclose(forecast.forecasts, expected)

        forecast = vol.forecast(
            params, resids, backcast, var_bounds, horizon=1, start=0
        )
        assert forecast.forecasts.shape[1] == 1

        vol = FIGARCH(truncation=50, power=1.0)
        with pytest.raises(
            ValueError, match=r"Analytic forecasts not available for horizon"
        ):
            vol.forecast(
                params, resids, backcast, var_bounds, horizon=2, method="analytic"
            )

    def test_figarch_simulation(self):
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        vol = FIGARCH(truncation=51)
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        params = np.array([0.1, 0.2, 0.4, 0.2])
        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=0,
                method="simulation",
                rng=rng,
            )

        lam = figarch_weights(params[1:], 1, 1, vol.truncation)
        arch = GARCH(p=vol.truncation, o=0, q=0)
        arch_params = np.r_[params[0] / (1 - params[-1]), lam]
        with preserved_state(self.rng):
            arch_forecast = arch.forecast(
                arch_params,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=0,
                method="simulation",
                rng=rng,
            )

        assert_allclose(forecast.forecasts, arch_forecast.forecasts)
        assert_allclose(forecast.forecast_paths, arch_forecast.forecast_paths)
        assert_allclose(forecast.shocks, arch_forecast.shocks)

    def test_midas_asym_simulation(self):
        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        vol = MIDASHyperbolic(asym=True)
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        params = np.array([0.1, 0.3, 1.0, 0.4])
        with preserved_state(self.rng):
            forecast = vol.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=0,
                method="simulation",
                rng=rng,
            )

        weights = vol._weights(params)
        arch = GARCH(p=22, o=22, q=0)
        arch_params = np.r_[params[0], params[1] * weights, params[2] * weights]
        with preserved_state(self.rng):
            arch_forecast = arch.forecast(
                arch_params,
                resids,
                backcast,
                var_bounds,
                horizon=10,
                start=0,
                method="simulation",
                rng=rng,
            )

        assert_allclose(forecast.forecasts, arch_forecast.forecasts)
        assert_allclose(forecast.forecast_paths, arch_forecast.forecast_paths)
        assert_allclose(forecast.shocks, arch_forecast.shocks)

    def test_midas_bootstrap_smoke(self):
        # Note strictly needed if simulation passes since just a special case
        vol = MIDASHyperbolic()
        resids = self.resid
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        params = np.array([0.1, 0.9, 0.4])
        vol.forecast(
            params,
            resids,
            backcast,
            var_bounds,
            horizon=10,
            start=100,
            method="bootstrap",
        )

    def test_fixed_variance(self):
        parameters = np.array([1.0])
        resids = self.resid
        variance = np.arange(resids.shape[0]) + 1.0
        vol = FixedVariance(variance)
        vol.start = 0
        vol.stop = 1000
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        forecasts = vol.forecast(parameters, resids, backcast, var_bounds, horizon=1)
        assert np.all(np.isnan(forecasts.forecasts))
        assert forecasts.forecast_paths is None
        assert forecasts.shocks is None

        forecasts = vol.forecast(parameters, resids, backcast, var_bounds, horizon=7)
        assert np.all(np.isnan(forecasts.forecasts))
        assert forecasts.forecast_paths is None
        assert forecasts.shocks is None

        dist = Normal(seed=self.rng)
        rng = dist.simulate([])
        forecasts = vol.forecast(
            parameters,
            resids,
            backcast,
            var_bounds,
            333,
            horizon=4,
            method="simulation",
            simulations=100,
            rng=rng,
        )
        assert np.all(np.isnan(forecasts.forecasts))
        assert np.all(np.isnan(forecasts.forecast_paths))
        assert np.all(np.isnan(forecasts.shocks))

        forecasts = vol.forecast(
            parameters, resids, backcast, var_bounds, 100, 2, "bootstrap", 500
        )
        assert np.all(np.isnan(forecasts.forecasts))
        assert np.all(np.isnan(forecasts.forecast_paths))
        assert np.all(np.isnan(forecasts.shocks))


class TestBootstrapRng:
    @classmethod
    def setup_class(cls):
        cls.rng = RandomState(12345)

    def test_bs_rng(self):
        y = self.rng.random_sample(1000)
        bs_rng = BootstrapRng(y, 100, random_state=self.rng)
        rng = bs_rng.rng()
        size = (1231, 13)
        with preserved_state(self.rng):
            output = {i: rng(size) for i in range(100, 1000)}

        expected = {}
        for i in range(100, 1000):
            locs = self.rng.random_sample(size)
            expected[i] = y[np.floor(locs * (i + 1)).astype(np.int64)]

        for i in range(100, 1000):
            assert_allclose(expected[i], output[i])

    def test_bs_rng_errors(self):
        y = self.rng.random_sample(1000)
        bs_rng = BootstrapRng(y, 100)
        rng = bs_rng.rng()
        for _ in range(100, 1000):
            rng(1)
        with pytest.raises(IndexError, match=r"not enough data points"):
            rng(1)

        y = self.rng.random_sample(1000)
        with pytest.raises(ValueError, match=r"start must be > 0 and"):
            BootstrapRng(y, 0)

    def test_bootstrap_rng(self):
        resid = np.arange(1000.0)
        rs = np.random.RandomState(1)
        state = rs.get_state()
        bs_rng = BootstrapRng(resid, start=100, random_state=rs)
        rng = bs_rng.rng()
        sim = rng(10000)
        assert np.max(sim) <= 100.5
        assert sim.shape == (10000,)
        rs.set_state(state)
        bs_rs = bs_rng.random_state
        assert bs_rs is rs

        with pytest.raises(
            TypeError, match=r"random_state must be a NumPy RandomState instance"
        ):
            BootstrapRng(resid, start=100, random_state=1234)


def test_external_rng():
    arch_mod = arch_model(None, mean="Constant", vol="GARCH", p=1, q=1)
    data = arch_mod.simulate(np.array([0.1, 0.1, 0.1, 0.88]), 1000)
    data.index = pd.date_range("2000-01-01", periods=data.index.shape[0])

    rand_state = np.random.RandomState(1)
    state = rand_state.get_state()
    volatility = GARCH(1, 0, 1)
    distribution = Normal(seed=rand_state)
    mod = ConstantMean(data.data, volatility=volatility, distribution=distribution)
    res = mod.fit(disp="off")

    rand_state.set_state(state)
    fcast_1 = res.forecast(
        res.params,
        horizon=5,
        method="simulation",
        start=900,
        simulations=250,
    )
    rand_state.set_state(state)
    rng = rand_state.standard_normal
    fcast_2 = res.forecast(
        res.params,
        horizon=5,
        method="simulation",
        start=900,
        simulations=250,
        rng=rng,
    )
    assert_allclose(fcast_1.residual_variance, fcast_2.residual_variance)


def _midas_reference_forecast(resids, params, m, horizon, origin):
    """
    Direct implementation of the asymmetric MIDAS Hyperbolic variance

    sigma2_t = omega + sum_i (alpha + gamma I[e_{t-i} < 0]) phi_i(theta) e_{t-i}^2

    with phi_i(theta) proportional to Gamma(i + theta) / (Gamma(i + 1) Gamma(theta)),
    normalized to sum to 1. Pre-sample squared residuals are set to the exponentially
    weighted (0.94) backcast of the first 75 observations and pre-sample indicators to
    1/2. For h > 1 the unknown squared shocks are replaced by their forecasts and
    their indicators by 1/2.
    """
    omega, alpha, gamma, theta = params
    lags = np.arange(1, m + 1)
    phi = np.array(
        [exp(lgamma(i + theta) - lgamma(i + 1) - lgamma(theta)) for i in lags]
    )
    phi = phi / phi.sum()

    tau = min(75, resids.shape[0])
    ew = 0.94 ** np.arange(tau)
    ew = ew / ew.sum()
    backcast = float(np.sum(resids[:tau] ** 2 * ew))

    sq = [backcast] * m + list(resids[: origin + 1] ** 2)
    ind = [0.5] * m + [float(r < 0) for r in resids[: origin + 1]]
    out = []
    for _ in range(horizon):
        # most recent value first: lag 1 is the last element
        past_sq = np.array(sq[-m:][::-1])
        past_ind = np.array(ind[-m:][::-1])
        sigma2 = omega + np.sum((alpha + gamma * past_ind) * phi * past_sq)
        out.append(sigma2)
        sq.append(sigma2)
        ind.append(0.5)
    return np.array(out)


@pytest.fixture(scope="module")
def midas_asym_fit():
    m = 22
    params = np.array([0.0, 0.05, 0.05, 0.25, 0.6])
    sim_mod = ConstantMean(
        None,
        volatility=MIDASHyperbolic(m=m, asym=True),
        distribution=Normal(seed=np.random.RandomState(2)),
    )
    y = sim_mod.simulate(params, 600, burn=300)["data"].to_numpy()
    mod = ConstantMean(y, volatility=MIDASHyperbolic(m=m, asym=True))
    res = mod.fix(params)
    return mod, res, params, m


@pytest.mark.parametrize("start", [0, 1, 300, 590, 599], ids=lambda s: f"start-{s}")
def test_midas_asym_analytic_start(midas_asym_fit, start):
    # Analytic forecasts of asymmetric MIDAS models used to raise for start > 0
    _, res, params, m = midas_asym_fit
    resids = np.asarray(res.resid)
    nobs = resids.shape[0]
    horizon = 5
    fcast = res.forecast(horizon=horizon, start=start, reindex=False)
    expected = np.array(
        [
            _midas_reference_forecast(resids, params[1:], m, horizon, origin)
            for origin in range(start, nobs)
        ]
    )
    assert fcast.variance.shape == expected.shape
    assert_allclose(fcast.variance.to_numpy(), expected, rtol=1e-10)


def test_midas_asym_analytic_default_start(midas_asym_fit):
    # The default forecasts only the final observation
    _, res, params, m = midas_asym_fit
    resids = np.asarray(res.resid)
    horizon = 5
    fcast = res.forecast(horizon=horizon, reindex=False)
    expected = _midas_reference_forecast(
        resids, params[1:], m, horizon, resids.shape[0] - 1
    )
    assert fcast.variance.shape == (1, horizon)
    assert_allclose(fcast.variance.to_numpy()[0], expected, rtol=1e-10)


def test_midas_asym_analytic_rows_agree_across_start(midas_asym_fit):
    _, res, _, _ = midas_asym_fit
    full = res.forecast(horizon=4, start=0, reindex=False).variance.to_numpy()
    part = res.forecast(horizon=4, start=500, reindex=False).variance.to_numpy()
    assert_allclose(part, full[500:], rtol=1e-12)


def test_midas_asym_analytic_matches_simulation(midas_asym_fit):
    # Multi-step analytic forecasts replace the indicator by 1/2, which is exact in
    # expectation for symmetric standardized shocks, so the simulation mean must
    # agree up to Monte Carlo error
    _, res, _, _ = midas_asym_fit
    horizon = 5
    analytic = res.forecast(horizon=horizon, reindex=False).variance.to_numpy()[0]
    sim = res.forecast(
        horizon=horizon,
        method="simulation",
        simulations=20000,
        rng=np.random.RandomState(12345).standard_normal,
        reindex=False,
    )
    paths = sim.simulations.variances[0]
    mc_mean = paths.mean(0)
    mc_se = paths.std(0, ddof=1) / np.sqrt(paths.shape[0])
    # one-step forecast is deterministic
    assert_allclose(mc_mean[0], analytic[0], rtol=1e-10)
    assert np.all(np.abs(mc_mean[1:] - analytic[1:]) < 5 * mc_se[1:])


def _skew_t_constants(eta, lam):
    c = np.exp(gammaln((eta + 1) / 2) - gammaln(eta / 2)) / np.sqrt(np.pi * (eta - 2))
    a = 4 * lam * c * (eta - 2) / (eta - 1)
    b = np.sqrt(1 + 3 * lam**2 - a**2)
    return a, b, c


def _skew_t_pdf(z, eta, lam):
    """Hansen's (1994) skew-t density, written from the paper's definitions"""
    a, b, c = _skew_t_constants(eta, lam)
    scale = np.where(z < -a / b, 1 - lam, 1 + lam)
    return b * c * (1 + ((b * z + a) / scale) ** 2 / (eta - 2)) ** (-(eta + 1) / 2)


def _skew_t_kink(eta, lam):
    a, b, _ = _skew_t_constants(eta, lam)
    return -a / b


def _integrate_pdf(pdf, power, upper, kink=0.0):
    """Integrate z**power * pdf(z) over (-inf, upper], splitting at a kink"""

    def integrand(z):
        return z**power * pdf(z)

    options = {"epsabs": 1e-13, "epsrel": 1e-13}
    if kink < upper:
        lower = integrate.quad(integrand, -np.inf, kink, **options)[0]
        return lower + integrate.quad(integrand, kink, upper, **options)[0]
    return integrate.quad(integrand, -np.inf, upper, **options)[0]


def _students_t_pdf(z, nu):
    scale = np.sqrt(nu / (nu - 2))
    return stats.t.pdf(z * scale, nu) * scale


def _ged_pdf(z, nu):
    # gennorm with unit scale has variance gamma(3/nu) / gamma(1/nu)
    scale = np.sqrt(np.exp(gammaln(1 / nu) - gammaln(3 / nu)))
    return stats.gennorm.pdf(z, nu, scale=scale)


# name: (distribution, distribution parameters)
SKEW_CASES = {
    "skew-t-7.6--0.16": (SkewStudent, [7.6, -0.16]),
    "skew-t-5-0.35": (SkewStudent, [5.0, 0.35]),
    "skew-t-12--0.6": (SkewStudent, [12.0, -0.6]),
}
SYMMETRIC_CASES = {
    "normal": (Normal, []),
    "students-t-6": (StudentsT, [6.0]),
    "ged-1.5": (GeneralizedError, [1.5]),
}
DIST_CASES = {**SKEW_CASES, **SYMMETRIC_CASES}


def _reference_negative_moment(name):
    """E[z**2 1(z<0)] by numerical integration of the standardized density"""
    dist, dp = DIST_CASES[name]
    kink = 0.0
    if dist is SkewStudent:
        eta, lam = dp
        kink = _skew_t_kink(eta, lam)

        def pdf(z):
            return _skew_t_pdf(z, eta, lam)

    elif dist is Normal:
        pdf = stats.norm.pdf
    elif dist is StudentsT:

        def pdf(z):
            return _students_t_pdf(z, dp[0])

    else:

        def pdf(z):
            return _ged_pdf(z, dp[0])

    return _integrate_pdf(pdf, 2, 0.0, kink)


@pytest.mark.parametrize("name", list(DIST_CASES))
def test_negative_second_moment_reference(name):
    # Checks the reference used in the GJR forecast tests below
    dist, dp = DIST_CASES[name]
    k = _reference_negative_moment(name)
    assert_allclose(k, dist().partial_moment(2, 0.0, dp), rtol=1e-8)
    if name in SKEW_CASES:
        assert abs(k - 0.5) > 0.01
    else:
        assert_allclose(k, 0.5, rtol=1e-8)


def test_skew_t_reference_density():
    eta, lam = 7.6, -0.16
    kink = _skew_t_kink(eta, lam)

    def pdf(z):
        return _skew_t_pdf(z, eta, lam)

    assert_allclose(_integrate_pdf(pdf, 0, np.inf, kink), 1.0, rtol=1e-8)
    assert_allclose(_integrate_pdf(pdf, 2, np.inf, kink), 1.0, rtol=1e-8)
    # E[z^2 1(z<0)] for this (eta, lambda), from the integral above
    assert_allclose(_reference_negative_moment("skew-t-7.6--0.16"), 0.5525668775)


GJR_PARAMS = np.array([0.02, 0.0, 0.20, 0.88])  # omega, alpha, gamma, beta


def _gjr_data(dist_name, nobs=400):
    dist, dp = DIST_CASES[dist_name]
    params = np.r_[0.05, GJR_PARAMS, dp]
    sim = ConstantMean(None, volatility=GARCH(p=1, o=1, q=1), distribution=dist(seed=1))
    return sim.simulate(params, nobs, burn=100)["data"].to_numpy()


def _gjr_recursion(first, k, horizon):
    omega, alpha, gamma, beta = GJR_PARAMS
    ref = np.empty((first.shape[0], horizon))
    ref[:, 0] = first
    for h in range(1, horizon):
        ref[:, h] = omega + (alpha + gamma * k + beta) * ref[:, h - 1]
    return ref


def _mean_model(kind, data, dist_name):
    dist, dp = DIST_CASES[dist_name]
    vol = GARCH(p=1, o=1, q=1)
    if kind == "constant":
        model = ConstantMean(data, volatility=vol, distribution=dist(seed=2))
        mean_params = [0.05]
    elif kind == "arx":
        model = ARX(data, lags=1, volatility=vol, distribution=dist(seed=2))
        mean_params = [0.05, 0.1]
    else:
        model = MAMean(data, lags=1, volatility=vol, distribution=dist(seed=2))
        mean_params = [0.05, 0.1]
    return model, np.r_[mean_params, GJR_PARAMS, dp]


@pytest.mark.parametrize("name", list(DIST_CASES))
def test_gjr_analytic_forecast_recursion(name):
    data = _gjr_data(name)
    model, params = _mean_model("constant", data, name)
    res = model.fix(params)
    k = _reference_negative_moment(name)
    horizon = 5
    fcast = res.forecast(horizon=horizon, start=len(data) - 4, reindex=False)
    analytic = fcast.residual_variance.to_numpy()
    assert analytic.shape == (4, horizon)
    ref = _gjr_recursion(analytic[:, 0], k, horizon)
    assert_allclose(analytic, ref, rtol=1e-10)


@pytest.mark.parametrize("kind", ["constant", "arx", "ma"])
@pytest.mark.parametrize("method", ["analytic", "Analytic", "ANALYTIC"])
def test_gjr_analytic_forecast_models_and_method_case(kind, method):
    name = "skew-t-7.6--0.16"
    data = _gjr_data(name)
    model, params = _mean_model(kind, data, name)
    res = model.fix(params)
    k = _reference_negative_moment(name)
    horizon = 5
    fcast = res.forecast(
        horizon=horizon, start=len(data) - 3, method=method, reindex=False
    )
    analytic = fcast.residual_variance.to_numpy()
    ref = _gjr_recursion(analytic[:, 0], k, horizon)
    assert_allclose(analytic, ref, rtol=1e-10)
    # The half-moment recursion (k = 0.5) is clearly different
    half = _gjr_recursion(analytic[:, 0], 0.5, horizon)
    assert np.all(np.abs(analytic[:, -1] / half[:, -1] - 1) > 0.003)


def test_gjr_analytic_forecast_matches_simulation():
    name = "skew-t-7.6--0.16"
    data = _gjr_data(name, nobs=1000)
    model, params = _mean_model("constant", data, name)
    res = model.fix(params)
    horizon = 20
    analytic = res.forecast(horizon=horizon, reindex=False).variance.to_numpy()
    sim = res.forecast(
        horizon=horizon, method="simulation", simulations=100_000, reindex=False
    ).variance.to_numpy()
    half = _gjr_recursion(analytic[:, 0], 0.5, horizon)
    # The Monte Carlo error is below 1%; the half-moment recursion is 13% too low
    # at h=20
    assert_allclose(sim, analytic, rtol=0.02)
    assert np.abs(half[:, -1] / sim[:, -1] - 1).max() > 0.1


def _hyperbolic_weights(theta, m):
    # w_j = theta (theta + 1) ... (theta + j - 1) / j!, normalized to sum to 1
    w = np.empty(m)
    w[0] = theta
    for j in range(1, m):
        w[j] = w[j - 1] * (theta + j) / (j + 1)
    return w / w.sum()


@pytest.mark.parametrize("method", ["analytic", "ANALYTIC"])
@pytest.mark.parametrize("last", [1, 3])
def test_midas_asym_analytic_forecast_skew_moment(method, last):
    eta, lam = 7.6, -0.16
    k = _reference_negative_moment("skew-t-7.6--0.16")
    omega, alpha, gamma, theta = 0.02, 0.5, 0.6, 0.7
    m = 22
    params = np.array([0.05, omega, alpha, gamma, theta, eta, lam])
    vol = MIDASHyperbolic(m=m, asym=True)
    sim = ConstantMean(None, volatility=vol, distribution=SkewStudent(seed=1))
    data = sim.simulate(params, 500, burn=100)["data"].to_numpy()
    model = ConstantMean(data, volatility=vol, distribution=SkewStudent(seed=2))
    res = model.fix(params)
    nobs = data.shape[0]
    horizon = 5
    # start > 0 used to raise a ValueError for asymmetric MIDAS models
    fcast = res.forecast(
        horizon=horizon, start=nobs - last, method=method, reindex=False
    )
    analytic = fcast.variance.to_numpy()
    assert analytic.shape == (last, horizon)

    resid = data - params[0]
    weights = _hyperbolic_weights(theta, m)
    for row, origin in enumerate(range(nobs - last, nobs)):
        # x: squared residuals, ind: squared negative residuals. Future values are
        # replaced by their conditional expectations.
        past = resid[origin - m + 1 : origin + 1]
        x = list(past**2)
        ind = list((past**2) * (past < 0))
        expected = []
        for _ in range(horizon):
            lags_x = np.array(x[::-1][:m])
            lags_i = np.array(ind[::-1][:m])
            f = omega + alpha * weights @ lags_x + gamma * weights @ lags_i
            expected.append(f)
            x.append(f)
            ind.append(k * f)
        assert_allclose(analytic[row], expected, rtol=1e-10)
