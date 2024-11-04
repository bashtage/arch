import datetime as dt
from itertools import product

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from arch.data import sp500
from arch.tests.univariate.test_variance_forecasting import preserved_state
from arch.univariate import (
    APARCH,
    ARX,
    EGARCH,
    FIGARCH,
    GARCH,
    HARCH,
    HARX,
    ConstantMean,
    ConstantVariance,
    EWMAVariance,
    MIDASHyperbolic,
    RiskMetrics2006,
    ZeroMean,
    arch_model,
)
from arch.univariate.mean import _ar_forecast, _ar_to_impulse

SP500 = 100 * sp500.load()["Adj Close"].pct_change().dropna()

MEAN_MODELS = [
    HARX(SP500, lags=[1, 5]),
    ARX(SP500, lags=2),
    ConstantMean(SP500),
    ZeroMean(SP500),
]

VOLATILITIES = [
    ConstantVariance(),
    GARCH(),
    FIGARCH(),
    EWMAVariance(lam=0.94),
    MIDASHyperbolic(),
    HARCH(lags=[1, 5, 22]),
    RiskMetrics2006(),
    APARCH(),
    EGARCH(),
]

ANALYTICAL_VOLATILITIES = [
    ConstantVariance(),
    GARCH(),
    FIGARCH(),
    EWMAVariance(lam=0.94),
    MIDASHyperbolic(),
    HARCH(lags=[1, 5, 22]),
    RiskMetrics2006(),
]


MODEL_SPECS = list(product(MEAN_MODELS, VOLATILITIES))
ANALYTICAL_MODEL_SPECS = list(product(MEAN_MODELS, ANALYTICAL_VOLATILITIES))

IDS = [
    f"{str(mean).split('(')[0]}-{str(vol).split('(')[0]}" for mean, vol in MODEL_SPECS
]
ANALYTICAL_IDS = [
    f"{str(mean).split('(')[0]}-{str(vol).split('(')[0]}"
    for mean, vol in ANALYTICAL_MODEL_SPECS
]


@pytest.fixture(params=MODEL_SPECS, ids=IDS)
def model_spec(request):
    mean, vol = request.param
    mean.volatility = vol
    return mean


@pytest.fixture(params=ANALYTICAL_MODEL_SPECS, ids=ANALYTICAL_IDS)
def analytical_model_spec(request):
    mean, vol = request.param
    mean.volatility = vol
    return mean


class TestForecasting:
    @classmethod
    def setup_class(cls):
        cls.rng = RandomState(12345)
        am = arch_model(None, mean="Constant", vol="Constant")
        data = am.simulate(np.array([0.0, 10.0]), 1000)
        data.index = pd.date_range("2000-01-01", periods=data.index.shape[0])
        cls.zero_mean = data.data

        am = arch_model(None, mean="AR", vol="Constant", lags=[1])
        data = am.simulate(np.array([1.0, 0.9, 2]), 1000)
        data.index = pd.date_range("2000-01-01", periods=data.index.shape[0])
        cls.ar1 = data.data

        am = arch_model(None, mean="AR", vol="Constant", lags=[1, 2])
        data = am.simulate(np.array([1.0, 1.9, -0.95, 2]), 1000)
        data.index = pd.date_range("2000-01-01", periods=data.index.shape[0])
        cls.ar2 = data.data

        am = arch_model(None, mean="HAR", vol="Constant", lags=[1, 5, 22])
        data = am.simulate(np.array([1.0, 0.4, 0.3, 0.2, 2]), 1000)
        data.index = pd.date_range("2000-01-01", periods=data.index.shape[0])
        cls.har3 = data.data

        am = arch_model(None, mean="AR", vol="GARCH", lags=[1, 2], p=1, q=1)
        data = am.simulate(np.array([1.0, 1.9, -0.95, 0.05, 0.1, 0.88]), 1000)
        data.index = pd.date_range("2000-01-01", periods=data.index.shape[0])
        cls.ar2_garch = data.data

    def test_ar_forecasting(self):
        params = np.array([0.9])
        forecasts = _ar_forecast(
            self.zero_mean, 5, 0, 0.0, params, np.empty(0), np.empty(0)
        )
        expected = np.zeros((1000, 5))
        expected[:, 0] = 0.9 * self.zero_mean.values
        for i in range(1, 5):
            expected[:, i] = 0.9 * expected[:, i - 1]
        assert_allclose(forecasts, expected)

        params = np.array([0.5, -0.3, 0.2])
        forecasts = _ar_forecast(
            self.zero_mean, 5, 2, 0.0, params, np.empty(0), np.empty(0)
        )
        expected = np.zeros((998, 8))
        expected[:, 0] = self.zero_mean.iloc[0:-2]
        expected[:, 1] = self.zero_mean.iloc[1:-1]
        expected[:, 2] = self.zero_mean.iloc[2:]
        for i in range(3, 8):
            expected[:, i] = (
                0.5 * expected[:, i - 1]
                - 0.3 * expected[:, i - 2]
                + 0.2 * expected[:, i - 3]
            )
        fill = np.empty((2, 5))
        fill.fill(np.nan)
        expected = np.concatenate((fill, expected[:, 3:]))
        assert_allclose(forecasts, expected[2:])

    def test_ar_to_impulse(self):
        arp = np.array([0.9])
        impulses = _ar_to_impulse(20, arp)
        expected = 0.9 ** np.arange(20)
        assert_allclose(impulses, expected)

        arp = np.array([0.5, 0.3])
        impulses = _ar_to_impulse(20, arp)
        comp = np.array([arp, [1, 0]])
        a = comp.copy()
        expected = np.ones(20)
        for i in range(1, 20):
            expected[i] = a[0, 0]
            a = a.dot(comp)
        assert_allclose(impulses, expected)

        arp = np.array([1.5, 0.0, -0.7])
        impulses = _ar_to_impulse(20, arp)
        comp = np.array([arp, [1, 0, 0], [0, 1, 0]])
        a = comp.copy()
        expected = np.ones(20)
        for i in range(1, 20):
            expected[i] = a[0, 0]
            a = a.dot(comp)
        assert_allclose(impulses, expected)

    def test_zero_mean_forecast(self):
        am = arch_model(self.zero_mean, mean="Zero", vol="Constant")
        res = am.fit()
        fcast = res.forecast(res.params, horizon=3)
        alt_fcast = res.forecast(horizon=3)
        assert_frame_equal(fcast.mean, alt_fcast.mean)
        assert_frame_equal(fcast.variance, alt_fcast.variance)
        assert_frame_equal(fcast.residual_variance, alt_fcast.residual_variance)

        with pytest.raises(ValueError, match="horizon must"):
            res.forecast(res.params, horizon=3.0)
        with pytest.raises(ValueError, match="horizon must"):
            res.forecast(res.params, horizon=-1)
        with pytest.raises(ValueError, match="horizon must"):
            res.forecast(res.params, horizon="3")

        fcast_reindex = res.forecast(res.params, horizon=3, reindex=True)
        assert_frame_equal(fcast.mean, fcast_reindex.mean.iloc[-1:])
        assert_frame_equal(fcast.variance, fcast_reindex.variance.iloc[-1:])
        assert_frame_equal(
            fcast.residual_variance, fcast_reindex.residual_variance.iloc[-1:]
        )
        assert fcast_reindex.mean.shape[0] == self.zero_mean.shape[0]

        assert np.all(np.asarray(np.isnan(fcast.mean[:-1])))
        assert np.all(np.asarray(np.isnan(fcast.variance[:-1])))
        assert np.all(np.asarray(np.isnan(fcast.residual_variance[:-1])))

        params = np.asarray(res.params)
        assert np.all(0.0 == fcast.mean.iloc[-1])
        assert_allclose(fcast.variance.iloc[-1], np.ones(3) * params[0])
        assert_allclose(fcast.residual_variance.iloc[-1], np.ones(3) * params[0])

        res = am.fit(last_obs=500)
        params = np.asarray(res.params)
        fcast = res.forecast(horizon=3)
        assert fcast.mean.shape == (501, 3)
        assert fcast.variance.shape == (501, 3)
        assert fcast.residual_variance.shape == (501, 3)
        assert np.all(np.asarray(np.isfinite(fcast.mean)))
        assert np.all(np.asarray(np.isfinite(fcast.variance)))
        assert np.all(np.asarray(np.isfinite(fcast.residual_variance)))

        assert np.all(np.asarray(0.0 == fcast.mean))
        assert_allclose(fcast.variance, np.ones((501, 3)) * params[0])
        assert_allclose(fcast.residual_variance, np.ones((501, 3)) * params[0])
        with pytest.raises(ValueError, match="horizon must be an integer >= 1"):
            res.forecast(horizon=0)

    def test_frame_labels(self):
        am = arch_model(self.zero_mean, mean="Zero", vol="Constant")
        res = am.fit()
        fcast = res.forecast(horizon=12)
        assert fcast.mean.shape[1] == 12
        assert fcast.variance.shape[1] == 12
        assert fcast.residual_variance.shape[1] == 12
        for i in range(1, 13):
            if i < 10:
                col = "h.0" + str(i)
            else:
                col = "h." + str(i)

            assert col in fcast.mean.columns
            assert col in fcast.variance.columns
            assert col in fcast.residual_variance.columns

    def test_ar1_forecast(self):
        am = arch_model(self.ar1, mean="AR", vol="Constant", lags=[1])
        res = am.fit()

        fcast = res.forecast(horizon=5, start=0)
        params = np.asarray(res.params)
        direct = self.ar1.values

        for i in range(5):
            direct = params[0] + params[1] * direct
            assert_allclose(direct, fcast.mean.iloc[:, i])
            scale = np.sum((params[1] ** np.arange(i + 1)) ** 2.0)
            var = fcast.variance.iloc[1:, i]
            assert_allclose(var, scale * params[2] * np.ones_like(var))

        assert np.all(np.asarray(fcast.residual_variance[1:] == params[2]))

        fcast = res.forecast(horizon=5)
        params = np.asarray(res.params)
        assert np.all(np.asarray(np.isnan(fcast.mean[:-1])))
        assert np.all(np.asarray(np.isnan(fcast.variance[:-1])))
        assert np.all(np.asarray(np.isnan(fcast.residual_variance[:-1])))
        assert np.all(np.asarray(fcast.residual_variance.iloc[-1] == params[-1]))
        means = np.zeros(5)
        means[0] = params[0] + params[1] * self.ar1.iloc[-1]
        for i in range(1, 5):
            means[i] = params[0] + params[1] * means[i - 1]
        assert_allclose(means, fcast.mean.iloc[-1].values)

    def test_constant_mean_forecast(self):
        am = arch_model(self.zero_mean, mean="Constant", vol="Constant")
        res = am.fit()
        fcast = res.forecast(horizon=5)

        assert np.all(np.asarray(np.isnan(fcast.mean[:-1])))
        assert np.all(np.asarray(np.isnan(fcast.variance[:-1])))
        assert np.all(np.asarray(np.isnan(fcast.residual_variance[:-1])))
        params = np.asarray(res.params)
        assert_allclose(params[0] * np.ones(5), fcast.mean.iloc[-1])
        assert_allclose(params[1] * np.ones(5), fcast.variance.iloc[-1])
        assert_allclose(params[1] * np.ones(5), fcast.residual_variance.iloc[-1])

        assert fcast.mean.shape == (1, 5)
        assert fcast.variance.shape == (1, 5)
        assert fcast.residual_variance.shape == (1, 5)

    def test_ar2_forecast(self):
        am = arch_model(self.ar2, mean="AR", vol="Constant", lags=[1, 2])
        res = am.fit()

        fcast = res.forecast(horizon=5)
        params = np.asarray(res.params)
        expected = np.zeros(7)
        expected[:2] = self.ar2.iloc[-2:]
        for i in range(2, 7):
            expected[i] = (
                params[0] + params[1] * expected[i - 1] + params[2] * expected[i - 2]
            )

        expected = expected[2:]
        assert np.all(np.asarray(np.isnan(fcast.mean.iloc[:-1])))
        assert_allclose(fcast.mean.iloc[-1], expected)

        expected = np.zeros(5)
        comp = np.array([res.params.iloc[1:3], [1, 0]])
        a = np.eye(2)
        for i in range(5):
            expected[i] = a[0, 0]
            a = a.dot(comp)
        expected = res.params.iloc[-1] * np.cumsum(expected**2)
        assert_allclose(fcast.variance.iloc[-1], expected)

        expected = np.empty((1000, 5))
        expected[:2] = np.nan
        expected[2:] = res.params.iloc[-1]

        fcast = res.forecast(horizon=5, start=1)
        expected = np.zeros((999, 7))
        expected[:, 0] = self.ar2.iloc[0:-1]
        expected[:, 1] = self.ar2.iloc[1:]
        for i in range(2, 7):
            expected[:, i] = (
                params[0]
                + params[1] * expected[:, i - 1]
                + params[2] * expected[:, i - 2]
            )
        fill = np.empty((1, 5))
        fill.fill(np.nan)
        expected = np.concatenate((fill, expected[:, 2:]))
        assert_allclose(np.asarray(fcast.mean), expected[1:])

        expected = np.empty((1000, 5))
        expected[:2] = np.nan
        expected[2:] = res.params.iloc[-1]
        assert_allclose(np.asarray(fcast.residual_variance), expected[1:])

        with pytest.raises(ValueError):
            res.forecast(horizon=5, start=0)

    def test_har_forecast(self):
        am = arch_model(self.har3, mean="HAR", vol="Constant", lags=[1, 5, 22])
        res = am.fit()
        fcast_1 = res.forecast(horizon=1)
        fcast_5 = res.forecast(horizon=5)
        assert_allclose(fcast_1.mean, fcast_5.mean.iloc[:, :1])

        with pytest.raises(ValueError):
            res.forecast(horizon=1, start=0)
        with pytest.raises(ValueError):
            res.forecast(horizon=1, start=20)

        fcast_66 = res.forecast(horizon=66, start=21)
        expected = np.empty((1000, 66 + 22))
        expected.fill(np.nan)
        for i in range(22):
            if i < 21:
                expected[21:, i] = self.har3.iloc[i : (-21 + i)]
            else:
                expected[21:, i] = self.har3.iloc[i:]
        params = np.asarray(res.params)
        const = params[0]
        arp = np.zeros(22)
        arp[0] = params[1]
        arp[:5] += params[2] / 5
        arp[:22] += params[3] / 22
        arp_rev = arp[::-1]
        for i in range(22, 88):
            expected[:, i] = const + expected[:, i - 22 : i].dot(arp_rev)
        expected = expected[:, 22:]
        assert_allclose(fcast_66.mean, expected[21:])

        expected[:22] = np.nan
        expected[22:] = res.params.iloc[-1]
        assert_allclose(fcast_66.residual_variance, expected[21:])

        impulse = _ar_to_impulse(66, arp)
        expected = expected * np.cumsum(impulse**2)
        assert_allclose(fcast_66.variance, expected[21:])

    def test_forecast_start_alternatives(self):
        am = arch_model(self.har3, mean="HAR", vol="Constant", lags=[1, 5, 22])
        res = am.fit()
        date = self.har3.index[21]
        fcast_1 = res.forecast(start=21)
        fcast_2 = res.forecast(start=date)
        for field in ("mean", "variance", "residual_variance"):
            assert_frame_equal(getattr(fcast_1, field), getattr(fcast_2, field))
        pydt = dt.datetime(date.year, date.month, date.day)
        fcast_2 = res.forecast(start=pydt)
        for field in ("mean", "variance", "residual_variance"):
            assert_frame_equal(getattr(fcast_1, field), getattr(fcast_2, field))

        strdt = pydt.strftime("%Y-%m-%d")
        fcast_2 = res.forecast(start=strdt)
        for field in ("mean", "variance", "residual_variance"):
            assert_frame_equal(getattr(fcast_1, field), getattr(fcast_2, field))

        npydt = np.datetime64(pydt).astype("M8[ns]")
        fcast_2 = res.forecast(start=npydt)
        for field in ("mean", "variance", "residual_variance"):
            assert_frame_equal(getattr(fcast_1, field), getattr(fcast_2, field))

        with pytest.raises(ValueError):
            date = self.har3.index[20]
            res.forecast(start=date)

        with pytest.raises(ValueError):
            date = self.har3.index[0]
            res.forecast(start=date)

        fcast_0 = res.forecast()
        fcast_1 = res.forecast(start=999)
        fcast_2 = res.forecast(start=self.har3.index[999])
        for field in ("mean", "variance", "residual_variance"):
            assert_frame_equal(getattr(fcast_0, field), getattr(fcast_1, field))

            assert_frame_equal(getattr(fcast_0, field), getattr(fcast_2, field))

    def test_fit_options(self):
        am = arch_model(self.zero_mean, mean="Constant", vol="Constant")
        res = am.fit(first_obs=100)
        res.forecast()
        res = am.fit(last_obs=900)
        res.forecast()
        res = am.fit(first_obs=100, last_obs=900)
        res.forecast()
        res.forecast(start=100)
        res.forecast(start=200)
        am = arch_model(self.zero_mean, mean="Constant", vol="Constant", hold_back=20)
        res = am.fit(first_obs=100)
        res.forecast()

    def test_ar1_forecast_simulation_one(self):
        # Bug found when simulation=1
        am = arch_model(self.ar1, mean="AR", vol="GARCH", lags=[1])
        res = am.fit(disp="off")
        forecast = res.forecast(horizon=10, method="simulation", simulations=1)
        assert forecast.simulations.variances.shape == (1, 1, 10)

    def test_ar1_forecast_simulation(self):
        am = arch_model(self.ar1, mean="AR", vol="GARCH", lags=[1])
        res = am.fit(disp="off")

        with preserved_state(self.rng):
            forecast = res.forecast(
                horizon=5,
                start=0,
                method="simulation",
            )
            forecast_reindex = res.forecast(
                horizon=5, start=10, method="simulation", reindex=True
            )
        assert forecast.simulations.index.shape[0] == self.ar1.shape[0]
        assert (
            forecast.simulations.index.shape[0] == forecast.simulations.values.shape[0]
        )

        with preserved_state(self.rng):
            forecast_reindex = res.forecast(
                horizon=5, start=10, method="simulation", reindex=True
            )
        assert forecast_reindex.mean.shape[0] == self.ar1.shape[0]
        assert forecast_reindex.simulations.index.shape[0] == self.ar1.shape[0]

        y = np.asarray(self.ar1)
        index = self.ar1.index
        t = y.shape[0]
        params = np.array(res.params)
        resids = np.asarray(y[1:] - params[0] - params[1] * y[:-1])
        vol = am.volatility
        params = np.array(res.params)
        backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        rng = am.distribution.simulate([])
        vfcast = vol.forecast(
            params[2:],
            resids,
            backcast,
            var_bounds,
            start=0,
            method="simulation",
            rng=rng,
            horizon=5,
        )
        const, ar = params[0], params[1]
        means = np.zeros((t, 5))
        means[:, 0] = const + ar * y
        for i in range(1, 5):
            means[:, i] = const + ar * means[:, i - 1]
        means = pd.DataFrame(
            means, index=index, columns=[f"h.{j}" for j in range(1, 6)]
        )
        assert_frame_equal(means, forecast.mean)
        var = np.concatenate([[[np.nan] * 5], vfcast.forecasts])
        rv = pd.DataFrame(var, index=index, columns=[f"h.{j}" for j in range(1, 6)])
        assert_frame_equal(rv, forecast.residual_variance)

        lrv = rv.copy()
        for i in range(5):
            weights = (ar ** np.arange(i + 1)) ** 2
            weights = weights[:, None]
            lrv.iloc[:, i : i + 1] = rv.values[:, : i + 1].dot(weights[::-1])
        assert_frame_equal(lrv, forecast.variance)

    def test_ar1_forecast_bootstrap(self):
        am = arch_model(self.ar1, mean="AR", vol="GARCH", lags=[1])
        res = am.fit(disp="off")
        rs = np.random.RandomState(98765432)
        state = rs.get_state()
        forecast = res.forecast(
            horizon=5,
            start=900,
            method="bootstrap",
            random_state=rs,
        )
        rs.set_state(state)
        repeat = res.forecast(
            horizon=5,
            start=900,
            method="bootstrap",
            random_state=rs,
        )
        assert_frame_equal(forecast.mean, repeat.mean)
        assert_frame_equal(forecast.variance, repeat.variance)

    def test_ar2_garch11(self):
        pass

    def test_first_obs(self):
        y = self.ar2_garch
        mod = arch_model(y)
        res = mod.fit(disp="off", first_obs=y.index[100])
        mod = arch_model(y[100:])
        res2 = mod.fit(disp="off")
        assert_allclose(res.params, res2.params)
        mod = arch_model(y)
        res3 = mod.fit(disp="off", first_obs=100)
        assert res.fit_start == 100
        assert_allclose(res.params, res3.params)

        forecast = res.forecast(horizon=3)
        assert np.all(np.asarray(np.isfinite(forecast.mean)))
        assert np.all(np.asarray(np.isfinite(forecast.variance)))

        forecast = res.forecast(horizon=3, start=y.index[100])
        assert np.all(np.asarray(np.isfinite(forecast.mean)))
        assert np.all(np.asarray(np.isfinite(forecast.variance)))

        forecast = res.forecast(horizon=3, start=100)
        assert np.all(np.asarray(np.isfinite(forecast.mean)))
        assert np.all(np.asarray(np.isfinite(forecast.variance)))

        with pytest.raises(ValueError):
            res.forecast(horizon=3, start=y.index[98])

        res = mod.fit(disp="off")
        forecast = res.forecast(horizon=3)
        assert np.all(np.asarray(np.isfinite(forecast.mean)))
        assert np.all(np.asarray(np.isfinite(forecast.variance)))

        forecast = res.forecast(horizon=3, start=y.index[100])
        assert np.all(np.asarray(np.isfinite(forecast.mean)))
        assert np.all(np.asarray(np.isfinite(forecast.variance)))
        forecast = res.forecast(horizon=3, start=0)
        assert np.all(np.asarray(np.isfinite(forecast.mean)))
        assert np.all(np.asarray(np.isfinite(forecast.variance)))

        mod = arch_model(y, mean="AR", lags=[1, 2])
        res = mod.fit(disp="off")
        with pytest.raises(ValueError):
            res.forecast(horizon=3, start=0)

        forecast = res.forecast(horizon=3, start=1)
        assert np.all(np.asarray(np.isfinite(forecast.mean)))
        assert np.all(np.asarray(np.isnan(forecast.variance.iloc[:1])))
        assert np.all(np.asarray(np.isfinite(forecast.variance.iloc[1:])))

    def test_last_obs(self):
        y = self.ar2_garch
        mod = arch_model(y)
        res = mod.fit(disp="off", last_obs=y.index[900])
        res_2 = mod.fit(disp="off", last_obs=900)
        assert_allclose(res.params, res_2.params)
        mod = arch_model(y[:900])
        res_3 = mod.fit(disp="off")
        assert_allclose(res.params, res_3.params)

    def test_first_last_obs(self):
        y = self.ar2_garch
        mod = arch_model(y)
        res = mod.fit(disp="off", first_obs=y.index[100], last_obs=y.index[900])
        res_2 = mod.fit(disp="off", first_obs=100, last_obs=900)
        assert_allclose(res.params, res_2.params)
        mod = arch_model(y.iloc[100:900])
        res_3 = mod.fit(disp="off")
        assert_allclose(res.params, res_3.params)

        mod = arch_model(y)
        res_4 = mod.fit(disp="off", first_obs=100, last_obs=y.index[900])
        assert_allclose(res.params, res_4.params)

    def test_holdback_first_obs(self):
        y = self.ar2_garch
        mod = arch_model(y, hold_back=20)
        res_holdback = mod.fit(disp="off")
        mod = arch_model(y)
        res_first_obs = mod.fit(disp="off", first_obs=20)
        assert_allclose(res_holdback.params, res_first_obs.params)

        with pytest.raises(ValueError):
            res_holdback.forecast(start=18)

    def test_holdback_lastobs(self):
        y = self.ar2_garch
        mod = arch_model(y, hold_back=20)
        res_holdback_last_obs = mod.fit(disp="off", last_obs=800)
        mod = arch_model(y)
        res_first_obs_last_obs = mod.fit(disp="off", first_obs=20, last_obs=800)
        assert_allclose(res_holdback_last_obs.params, res_first_obs_last_obs.params)
        mod = arch_model(y[20:800])
        res_direct = mod.fit(disp="off")
        assert_allclose(res_direct.params, res_first_obs_last_obs.params)

        with pytest.raises(ValueError):
            res_holdback_last_obs.forecast(start=18)

    def test_holdback_ar(self):
        y = self.ar2_garch
        mod = arch_model(y, mean="AR", lags=1, hold_back=1)
        res_holdback = mod.fit(disp="off")
        mod = arch_model(y, mean="AR", lags=1)
        res = mod.fit(disp="off")
        assert_allclose(res_holdback.params, res.params, rtol=1e-4, atol=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("first_obs", [None, 250])
@pytest.mark.parametrize("last_obs", [None, 2500, 2750])
@pytest.mark.parametrize("reindex", [True, False])
def test_reindex(model_spec, reindex, first_obs, last_obs):
    reindex_dim = SP500.shape[0] - last_obs + 1 if last_obs is not None else 1
    dim0 = SP500.shape[0] if reindex else reindex_dim
    res = model_spec.fit(disp="off", first_obs=first_obs, last_obs=last_obs)
    fcast = res.forecast(horizon=1, reindex=reindex)
    assert fcast.mean.shape == (dim0, 1)
    fcast = res.forecast(
        horizon=2, method="simulation", simulations=25, reindex=reindex
    )
    assert fcast.mean.shape == (dim0, 2)
    fcast = res.forecast(horizon=2, method="bootstrap", simulations=25, reindex=reindex)
    assert fcast.mean.shape == (dim0, 2)
    assert fcast.simulations.values.shape == (dim0, 25, 2)
    with pytest.raises(ValueError, match="horizon must be"):
        res.forecast(horizon=0, reindex=reindex)


def test_invalid_horizon():
    res = arch_model(SP500).fit(disp="off")
    with pytest.raises(ValueError, match="horizon must be"):
        res.forecast(horizon=-1)
    with pytest.raises(ValueError, match="horizon must be"):
        res.forecast(horizon=1.0)
    with pytest.raises(ValueError, match="horizon must be"):
        res.forecast(horizon="5")


def test_arx_no_lags():
    mod = ARX(SP500, volatility=GARCH())
    res = mod.fit(disp="off")
    assert res.params.shape[0] == 4
    assert "lags" not in mod._model_description(include_lags=False)


EXOG_PARAMS = product(
    ["pandas", "dict", "numpy"], [(10,), (1, 10), (1, 1, 10), (2, 1, 10)], [True, False]
)


@pytest.fixture(scope="function", params=EXOG_PARAMS)
def exog_format(request):
    xtyp, shape, full = request.param
    rng = RandomState(123456)
    x_fcast = rng.standard_normal(shape)
    orig = x_fcast.copy()
    nobs = SP500.shape[0]
    if full:
        if x_fcast.ndim == 2:
            _x = np.full((nobs, shape[1]), np.nan)
            _x[-1:] = x_fcast
            x_fcast = _x
        elif x_fcast.ndim == 3:
            _x = np.full((shape[0], nobs, shape[-1]), np.nan)
            _x[:, -1:] = x_fcast
            x_fcast = _x
        else:
            # No full 1d
            return None, None
    if xtyp == "pandas":
        if x_fcast.ndim == 3:
            return None, None
        if x_fcast.ndim == 1:
            x_fcast = pd.Series(x_fcast)
        else:
            x_fcast = pd.DataFrame(x_fcast)
        x_fcast.index = SP500.index[-x_fcast.shape[0] :]
    elif xtyp == "dict":
        if x_fcast.ndim == 3:
            keys = [f"x{i}" for i in range(1, x_fcast.shape[0] + 1)]
            x_fcast = {k: x_fcast[i] for i, k in enumerate(keys)}
        else:
            x_fcast = {"x1": x_fcast}
    return x_fcast, orig


def test_x_reformat_1var(exog_format):
    # (10,)
    # (1,10)
    # (n, 10)
    # (1,1,10)
    # (1,n,10)
    # {"x1"} : (10,)
    # {"x1"} : (1,10)
    # {"x1"} : (n,10)
    exog, ref = exog_format
    if exog is None:
        return
    if isinstance(exog, dict):
        nexog = len(exog)
    else:
        if np.ndim(exog) == 3:
            nexog = exog.shape[0]
        else:
            nexog = 1
    cols = [f"x{i}" for i in range(1, nexog + 1)]
    rng = RandomState(12345)
    x = pd.DataFrame(
        rng.standard_normal((SP500.shape[0], nexog)), columns=cols, index=SP500.index
    )
    mod = ARX(SP500, lags=1, x=x)
    res = mod.fit()
    fcasts = res.forecast(horizon=10, x=exog)
    ref = res.forecast(horizon=10, x=ref)
    assert_allclose(fcasts.mean, ref.mean)


@pytest.mark.parametrize("nexog", [1, 2])
def test_x_forecasting(nexog):
    rng = RandomState(12345)
    mod = arch_model(None, mean="ARX", lags=2)
    data = mod.simulate([0.1, 1.2, -0.6, 0.1, 0.1, 0.8], nobs=1000)
    cols = [f"x{i}" for i in range(1, nexog + 1)]
    x = pd.DataFrame(
        rng.standard_normal((data.data.shape[0], nexog)),
        columns=cols,
        index=data.data.index,
    )
    b = np.array([0.25, 0.5]) if x.shape[1] == 2 else np.array([0.25])
    y = data.data + x @ b
    y.name = "y"
    mod = arch_model(y, x, mean="ARX", lags=2)
    res = mod.fit(disp="off")
    x_fcast = np.zeros((x.shape[1], 1, 10))
    for i in range(x_fcast.shape[0]):
        x_fcast[i] = np.arange(100 * i, 100 * i + 10)

    forecasts = res.forecast(x=x_fcast, horizon=10)
    direct = np.zeros(12)
    direct[:2] = y.iloc[-2:]
    p0, p1, p2, *bs = res.params
    b0 = bs[0]
    b1 = bs[1] if x.shape[1] == 2 else 0.0
    for i in range(10):
        direct[i + 2] = p0 + p1 * direct[i + 1] + p2 * direct[i]
        direct[i + 2] += b0 * (i)
        direct[i + 2] += b1 * (100 + i)
    assert_allclose(forecasts.mean.iloc[0], direct[2:])


@pytest.mark.parametrize("nexog", [1, 2])
def test_x_forecasting_simulation_smoke(nexog):
    rng = RandomState(12345)
    mod = arch_model(None, mean="ARX", lags=2)
    data = mod.simulate([0.1, 1.2, -0.6, 0.1, 0.1, 0.8], nobs=1000)
    cols = [f"x{i}" for i in range(1, nexog + 1)]
    x = pd.DataFrame(
        rng.standard_normal((data.data.shape[0], nexog)),
        columns=cols,
        index=data.data.index,
    )
    b = np.array([0.25, 0.5]) if x.shape[1] == 2 else np.array([0.25])
    y = data.data + x @ b
    y.name = "y"
    mod = arch_model(y, x, mean="ARX", lags=2)
    res = mod.fit(disp="off")
    x_fcast = np.zeros((x.shape[1], 1, 10))
    for i in range(x_fcast.shape[0]):
        x_fcast[i] = np.arange(100 * i, 100 * i + 10)

    res.forecast(x=x_fcast, horizon=10, method="simulation", simulations=10)


def test_x_exceptions():
    res = ARX(SP500, lags=1).fit(disp="off")
    with pytest.raises(TypeError, match="x is not None but"):
        res.forecast(x=SP500)
    x = SP500.copy()
    x[:] = np.random.standard_normal(SP500.shape)
    x.name = "Exog"
    res = ARX(SP500, lags=1, x=x).fit(disp="off")
    with pytest.raises(TypeError, match="x is None but the model"):
        res.forecast()
    res = ARX(SP500, lags=1, x=x).fit(disp="off")
    with pytest.raises(ValueError, match="x must have the same"):
        res.forecast(x={})
    with pytest.raises(ValueError, match="x must have the same"):
        res.forecast(x={"x0": x, "x1": x})
    with pytest.raises(KeyError, match="The keys of x must exactly"):
        res.forecast(x={"z": x})
    with pytest.raises(ValueError, match="The arrays contained in the dictionary"):
        _x = np.asarray(x).reshape((1, x.shape[0], 1))
        res.forecast(x={"Exog": _x})
    x2 = pd.concat([x, x], axis=1)
    x2.columns = ["x0", "x1"]
    x2.iloc[:, 1] = np.random.standard_normal(SP500.shape)
    res = ARX(SP500, lags=1, x=x2).fit(disp="off")
    with pytest.raises(ValueError, match="The shapes of the arrays contained"):
        res.forecast(x={"x0": x2.iloc[:, 0], "x1": x2.iloc[10:, 1:]})
    with pytest.raises(ValueError, match="1- and 2-dimensional x values"):
        res.forecast(x=x2)
    with pytest.raises(ValueError, match="The leading dimension of x"):
        _x2 = np.asarray(x2)
        _x2 = _x2.reshape((1, -1, 2))
        res.forecast(x=_x2)
    with pytest.raises(ValueError, match="The number of values passed"):
        res.forecast(x=np.empty((2, SP500.shape[0], 3)))
    with pytest.raises(ValueError, match="The shape of x does not satisfy the"):
        res.forecast(x=np.empty((2, SP500.shape[0] // 2, 1)))


def test_model_forecast():
    mod = arch_model(SP500)
    res = mod.fit(disp=False)
    res_fcast = res.forecast(horizon=10, reindex=True)
    params = np.asarray(res.params, dtype=float)
    mod_fcast = mod.forecast(params, horizon=10, reindex=True)
    new_mod = arch_model(SP500)
    new_mod_fcast = new_mod.forecast(params, horizon=10, reindex=True)
    assert_allclose(res_fcast.variance, mod_fcast.variance)
    assert_allclose(res_fcast.variance, new_mod_fcast.variance)


def test_model_forecast_recursive():
    vol = GARCH()
    base = ConstantMean(SP500.iloc[:-10], volatility=vol).fit(disp=False)
    params = base.params
    fcasts = {}
    for i in range(11):
        end = SP500.shape[0] - 10 + i
        mod = ConstantMean(SP500.iloc[:end], volatility=vol)
        fcasts[i] = mod.forecast(params, reindex=True)


@pytest.mark.parametrize("lags", [0, 1, 2, [2]])
@pytest.mark.parametrize("constant", [True, False])
def test_forecast_ar0(constant, lags):
    burn = 250

    x_mod = ARX(None, lags=1)
    x0 = x_mod.simulate([1, 0.8, 1], nobs=1000 + burn).data
    x1 = x_mod.simulate([2.5, 0.5, 1], nobs=1000 + burn).data

    resid_mod = ZeroMean(volatility=GARCH())
    resids = resid_mod.simulate([0.1, 0.1, 0.8], nobs=1000 + burn).data

    phi1 = 0.7
    phi0 = 3
    y = 10 + resids.copy()
    for i in range(1, y.shape[0]):
        y[i] = phi0 + phi1 * y[i - 1] + 2 * x0[i] - 2 * x1[i] + resids[i]

    x0 = x0.iloc[-1000:]
    x1 = x1.iloc[-1000:]
    y = y.iloc[-1000:]
    y.index = x0.index = x1.index = np.arange(1000)

    x0_oos = np.empty((1000, 10))
    x1_oos = np.empty((1000, 10))
    for i in range(10):
        if i == 0:
            last = x0
        else:
            last = x0_oos[:, i - 1]
        x0_oos[:, i] = 1 + 0.8 * last
        if i == 0:
            last = x1
        else:
            last = x1_oos[:, i - 1]
        x1_oos[:, i] = 2.5 + 0.5 * last

    exog = pd.DataFrame({"x0": x0, "x1": x1})
    mod = ARX(y, x=exog, lags=lags, constant=constant, volatility=GARCH())
    res = mod.fit(disp="off")
    exog_fcast = {"x0": x0_oos[-1:], "x1": x1_oos[-1:]}
    forecasts = res.forecast(
        horizon=10,
        x=exog_fcast,
        method="simulation",
        simulations=100,
    )
    assert forecasts.mean.shape == (1, 10)
    assert forecasts.simulations.values.shape == (1, 100, 10)


def test_simulation_exog():
    # GH 551
    burn = 250
    from arch.univariate import Normal

    rs = np.random.RandomState(3382983)
    normal = Normal(seed=rs)
    x_mod = ARX(None, lags=1, distribution=normal)
    x0 = x_mod.simulate([1, 0.8, 1], nobs=1000 + burn).data
    x1 = x_mod.simulate([2.5, 0.5, 1], nobs=1000 + burn).data

    rs = np.random.RandomState(33829831)
    normal = Normal(seed=rs)
    resid_mod = ZeroMean(volatility=GARCH(), distribution=normal)
    resids = resid_mod.simulate([0.1, 0.1, 0.8], nobs=1000 + burn).data

    phi1 = 0.7
    phi0 = 3
    y = 10 + resids.copy()
    for i in range(1, y.shape[0]):
        y[i] = phi0 + phi1 * y[i - 1] + 2 * x0[i] - 2 * x1[i] + resids[i]

    x0 = x0.iloc[-1000:]
    x1 = x1.iloc[-1000:]
    y = y.iloc[-1000:]
    y.index = x0.index = x1.index = np.arange(1000)

    x0_oos = np.empty((1000, 10))
    x1_oos = np.empty((1000, 10))
    for i in range(10):
        if i == 0:
            last = x0
        else:
            last = x0_oos[:, i - 1]
        x0_oos[:, i] = 1 + 0.8 * last
        if i == 0:
            last = x1
        else:
            last = x1_oos[:, i - 1]
        x1_oos[:, i] = 2.5 + 0.5 * last

    exog = pd.DataFrame({"x0": x0, "x1": x1})
    mod = arch_model(y, x=exog, mean="ARX", lags=0)
    res = mod.fit(disp="off")

    nforecast = 3

    # DECOMMENT ONE OF THE FOLLOWING LINES
    exog_fcast = {
        "x0": np.zeros_like(x0_oos[-nforecast:]),
        "x1": np.zeros_like(x1_oos[-nforecast:]),
    }
    forecasts = res.forecast(
        horizon=10,
        x=exog_fcast,
        start=1000 - nforecast,
        method="simulation",
    )

    exog_fcast = {"x0": np.zeros_like(x0_oos), "x1": np.zeros_like(x0_oos)}
    forecasts_alt = res.forecast(
        horizon=10,
        x=exog_fcast,
        start=1000 - nforecast,
        method="simulation",
    )
    assert_allclose(forecasts.mean, forecasts_alt.mean)

    exog_fcast = {
        "x0": 10 + np.zeros_like(x0_oos[-nforecast:]),
        "x1": 10 + np.zeros_like(x1_oos[-nforecast:]),
    }  # case with shape (nforecast, horizon)
    # exog_fcast = {"x0": x0_oos, "x1": x1_oos} # case with shape (nobs, horizon)

    forecasts_10 = res.forecast(
        horizon=10,
        x=exog_fcast,
        start=1000 - nforecast,
        method="simulation",
    )

    delta = forecasts_10.mean - forecasts.mean
    expected = (10 * res.params[["x0", "x1"]]).sum() + np.zeros_like(delta)
    assert_allclose(delta, expected)


def test_rescale():
    # GH 632

    rets = 100 * SP500.copy()

    model = arch_model(
        rets,
        p=1,
        o=0,
        q=1,
        mean="AR",
        lags=1,
        vol="GARCH",
        dist="normal",
        rescale=False,
    )
    res = model.fit(disp="off")

    model_rescale = arch_model(
        rets,
        p=1,
        o=0,
        q=1,
        mean="AR",
        lags=1,
        vol="GARCH",
        dist="normal",
        rescale=True,
    )
    res_rescale = model_rescale.fit(disp="off")

    lr_fcast = res.forecast(horizon=10000)
    lr_fcast_rescale = res_rescale.forecast(horizon=10000)

    omega_bar = res.params.omega / (1 - res.params["alpha[1]"] - res.params["beta[1]"])
    assert_allclose(lr_fcast.residual_variance.iloc[0, -1], omega_bar, rtol=1e-5)

    omega_bar_rescale = res_rescale.params.omega / (
        1 - res_rescale.params["alpha[1]"] - res_rescale.params["beta[1]"]
    )
    assert_allclose(
        lr_fcast_rescale.residual_variance.iloc[0, -1], omega_bar_rescale, rtol=1e-5
    )

    rets = 10000 * SP500.copy()
    model_rescale = arch_model(
        rets,
        p=1,
        o=0,
        q=1,
        mean="AR",
        lags=1,
        vol="GARCH",
        dist="normal",
        rescale=True,
    )
    res_rescale = model_rescale.fit(disp="off")
    omega_bar_rescale = res_rescale.params.omega / (
        1 - res_rescale.params["alpha[1]"] - res_rescale.params["beta[1]"]
    )
    assert_allclose(
        lr_fcast_rescale.residual_variance.iloc[0, -1], omega_bar_rescale, rtol=1e-5
    )


def test_rescale_ar():
    # GH 632
    am = arch_model(None, mean="AR", lags=3, vol="GARCH", p=1, o=0, q=1)
    data = am.simulate([10, 0.4, 0.3, 0.2, 10, 0.1, 0.85], 10000).data / 1000
    vol = GARCH()
    mod = ARX(data, lags=3, constant=True, volatility=vol, rescale=True)
    res = mod.fit()

    mod_no_rs = ARX(100 * data, lags=3, constant=True, volatility=vol, rescale=False)
    res_no_rs = mod_no_rs.fit()
    assert_allclose(res_no_rs.params.iloc[1:4], res.params.iloc[1:4], rtol=1e-5)

    fcasts = res.forecast(horizon=100).variance
    fcasts_no_rs = res_no_rs.forecast(horizon=100).variance
    assert_allclose(fcasts.iloc[0, -10:], fcasts_no_rs.iloc[0, -10:], rtol=1e-5)


def test_figarch_multistep():
    # GH 670
    mod = ConstantMean(SP500, volatility=FIGARCH())
    res = mod.fit(disp="off")
    fcasts = res.forecast(horizon=10)
    rv = fcasts.residual_variance
    assert np.all(np.isfinite(rv))
    assert rv.shape == (1, 10)
    fcasts_ri = res.forecast(horizon=10, reindex=True)
    rv_ri = fcasts_ri.residual_variance
    assert_frame_equal(rv, rv_ri.iloc[-1:])
    assert rv_ri.shape == (SP500.shape[0], 10)


def test_multistep(analytical_model_spec):
    # GH 670
    # Ensure all work as expected
    res = analytical_model_spec.fit(disp="off")
    fcasts = res.forecast(horizon=10)
    rv = fcasts.residual_variance
    assert np.all(np.isfinite(rv))
    assert rv.shape == (1, 10)
    fcasts_ri = res.forecast(horizon=10, reindex=True)
    rv_ri = fcasts_ri.residual_variance
    assert_frame_equal(rv, rv_ri.iloc[-1:])
    assert rv_ri.shape == (SP500.shape[0], 10)


def test_forecast_exog_single_exog():
    rg = np.random.default_rng(0)
    y = rg.standard_normal(100)
    x = pd.DataFrame(rg.standard_normal((100, 1)), columns=["x"])
    x_oos = rg.standard_normal((1, 5))
    mod = ARX(y, x=x, lags=1)
    res = mod.fit()
    # Direct forecast
    c, p, b, _ = res.params
    oos = np.zeros((1, 5))
    oos[0, 0] = c + p * y[-1] + b * x_oos[0, 0]
    oos[0, 1] = c + p * oos[0, 0] + b * x_oos[0, 1]
    oos[0, 2] = c + p * oos[0, 1] + b * x_oos[0, 2]
    oos[0, 3] = c + p * oos[0, 2] + b * x_oos[0, 3]
    oos[0, 4] = c + p * oos[0, 3] + b * x_oos[0, 4]
    fcast = res.forecast(horizon=5, x=x_oos)
    assert_allclose(oos, fcast.mean)

    x_oos2 = np.tile(x_oos, (100, 1))
    fcast2 = res.forecast(horizon=5, x=x_oos2)
    assert_allclose(fcast.mean, fcast2.mean)

    x_oos3 = np.tile(x_oos, (1, 100, 1))
    fcast3 = res.forecast(horizon=5, x=x_oos3)
    assert_allclose(fcast.mean, fcast3.mean)

    x_oos4 = {"x": x_oos}
    fcast4 = res.forecast(horizon=5, x=x_oos4)
    assert_allclose(fcast.mean, fcast4.mean)


def test_forecast_exog_multi_exog():
    rg = np.random.default_rng(0)
    y = rg.standard_normal(100)
    x = pd.DataFrame(rg.standard_normal((100, 2)), columns=["x1", "x2"])
    x_oos = rg.standard_normal((2, 1, 5))
    mod = ARX(y, x=x, lags=1)
    res = mod.fit()

    c, p, b1, b2, _ = res.params
    oos = np.zeros((1, 5))
    oos[0, 0] = c + p * y[-1] + b1 * x_oos[0, 0, 0] + b2 * x_oos[1, 0, 0]
    oos[0, 1] = c + p * oos[0, 0] + b1 * x_oos[0, 0, 1] + b2 * x_oos[1, 0, 1]
    oos[0, 2] = c + p * oos[0, 1] + b1 * x_oos[0, 0, 2] + b2 * x_oos[1, 0, 2]
    oos[0, 3] = c + p * oos[0, 2] + b1 * x_oos[0, 0, 3] + b2 * x_oos[1, 0, 3]
    oos[0, 4] = c + p * oos[0, 3] + b1 * x_oos[0, 0, 4] + b2 * x_oos[1, 0, 4]

    fcast = res.forecast(horizon=5, x=x_oos)
    assert_allclose(oos, fcast.mean)

    x_oos2 = np.tile(x_oos, (100, 1))
    fcast2 = res.forecast(horizon=5, x=x_oos2)
    assert_allclose(fcast.mean, fcast2.mean)

    x_oos3 = {"x1": x_oos[0], "x2": x_oos[1]}
    fcast3 = res.forecast(horizon=5, x=x_oos3)
    assert_allclose(fcast.mean, fcast3.mean)


def test_forecast_exog_single_exog_limited_sample():
    rg = np.random.default_rng(0)
    y = rg.standard_normal(100)
    x = pd.DataFrame(rg.standard_normal((100, 1)), columns=["x"])
    x_oos = rg.standard_normal((3, 5))
    mod = ARX(y, x=x, lags=1)
    res = mod.fit(first_obs=0, last_obs=98)
    oos = np.zeros((3, 5))
    # Direct forecast
    c, p, b, _ = res.params
    for idx in range(3):
        oos[idx, 0] = c + p * y[(-3 + idx)] + b * x_oos[idx, 0]
        oos[idx, 1] = c + p * oos[idx, 0] + b * x_oos[idx, 1]
        oos[idx, 2] = c + p * oos[idx, 1] + b * x_oos[idx, 2]
        oos[idx, 3] = c + p * oos[idx, 2] + b * x_oos[idx, 3]
        oos[idx, 4] = c + p * oos[idx, 3] + b * x_oos[idx, 4]
    fcast = res.forecast(horizon=5, x=x_oos)
    assert_allclose(oos, fcast.mean)

    x_oos2 = np.concatenate([np.zeros((97, 5)), x_oos], axis=0)
    fcast2 = res.forecast(horizon=5, x=x_oos2)
    assert_allclose(fcast.mean, fcast2.mean)

    x_oos3 = x_oos2[None, :, :]
    fcast3 = res.forecast(horizon=5, x=x_oos3)
    assert_allclose(fcast.mean, fcast3.mean)

    x_oos4 = {"x": x_oos}
    fcast4 = res.forecast(horizon=5, x=x_oos4)
    assert_allclose(fcast.mean, fcast4.mean)


def test_forecast_simulation_horizon_1():
    rg = np.random.default_rng(0)
    y = rg.standard_normal(100)
    x = pd.DataFrame(rg.standard_normal((100, 1)), columns=["x"])
    mod = ARX(y, x=x, lags=1)
    res = mod.fit(first_obs=0, last_obs=98)
    res.forecast(start=1, x=x, method="simulation", simulations=2)
    res.forecast(start=1, x=x, method="simulation", simulations=1)


def test_forecast_start():
    rg = np.random.default_rng(0)
    y = rg.standard_normal(10)
    x = pd.DataFrame(rg.standard_normal((10, 1)), columns=["x"])
    mod = ARX(y, x=x, lags=3)
    res = mod.fit(first_obs=0, last_obs=98)
    fcast = res.forecast(start=2, x=x.shift(-1))
    fcast2 = res.forecast(start=2, method="simulation", simulations=1, x=x.shift(-1))
    assert_allclose(fcast.mean, fcast2.mean)

    c, p1, p2, p3, b, _ = res.params
    oos = np.full((8, 1), np.nan)
    for i in range(2, 9):
        oos[i - 2, 0] = (
            c + p1 * y[i] + p2 * y[i - 1] + p3 * y[i - 2] + b * x.iloc[i + 1, 0]
        )
    assert_allclose(fcast.mean, oos)
