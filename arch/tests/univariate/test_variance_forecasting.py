import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.univariate import arch_model
from arch.univariate.distribution import Normal, StudentsT
from arch.univariate.mean import ConstantMean

try:
    from arch.univariate.recursions import figarch_weights
except ImportError:
    from arch.univariate.recursions_python import figarch_weights

from arch.univariate.volatility import (
    APARCH,
    EGARCH,
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
        expected.shape = (1000, 1)
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
        for i in range(10):
            assert_allclose(forecasts.forecasts[:, i], one_step, rtol=1e-4)

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
