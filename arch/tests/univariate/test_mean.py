from arch.compat.matplotlib import HAS_MATPLOTLIB
from arch.compat.pandas import MONTH_END

from io import StringIO
import itertools
from itertools import product
from string import ascii_lowercase
import struct
import sys
import types
import warnings

import numpy as np
from numpy.random import RandomState
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_equal,
)
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
from scipy import stats
from scipy.optimize import OptimizeResult
import statsmodels.regression.linear_model as smlm
import statsmodels.tools as smtools

from arch._typing import Literal
from arch.data import sp500
from arch.univariate.base import (
    ARCHModel,
    ARCHModelFixedResult,
    ARCHModelForecast,
    ARCHModelResult,
    _align_forecast,
)
from arch.univariate.distribution import (
    GeneralizedError,
    Normal,
    SkewStudent,
    StudentsT,
)
from arch.univariate.mean import ARX, HARX, LS, ConstantMean, ZeroMean, arch_model
from arch.univariate.volatility import (
    APARCH,
    ARCH,
    EGARCH,
    FIGARCH,
    GARCH,
    HARCH,
    ConstantVariance,
    EWMAVariance,
    FixedVariance,
    MIDASHyperbolic,
    RiskMetrics2006,
    VolatilityProcess,
)
from arch.utility.exceptions import ConvergenceWarning, DataScaleWarning

USE_CYTHON = False
try:
    import arch.univariate.recursions

    USE_CYTHON = True
except ImportError:
    import arch.univariate.recursions_python

if USE_CYTHON:
    rec: types.ModuleType = arch.univariate.recursions
else:
    rec = arch.univariate.recursions_python


RTOL = 1e-4 if struct.calcsize("P") < 8 else 1e-6
DISPLAY: Literal["off", "final"] = "off"
UPDATE_FREQ = 0 if DISPLAY == "off" else 3
SP500 = 100 * sp500.load()["Adj Close"].pct_change().dropna()
rs = np.random.RandomState(20241029)
X = SP500 * 0.01 + SP500.std() * rs.standard_normal(SP500.shape)


def close_plots():
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.close("all")


@pytest.fixture(scope="module", params=[True, False])
def simulated_data(request):
    rs = np.random.RandomState(1)
    zm = ZeroMean(volatility=GARCH(), distribution=Normal(seed=rs))
    sim_data = zm.simulate(np.array([0.1, 0.1, 0.88]), 1000)
    return np.asarray(sim_data.data) if request.param else sim_data.data


simple_mean_models = [
    ARX(SP500, lags=1),
    HARX(SP500, lags=[1, 5]),
    ConstantMean(SP500),
    ZeroMean(SP500),
]

mean_models = [
    ARX(SP500, x=X, lags=1),
    HARX(SP500, x=X, lags=[1, 5]),
    LS(SP500, X),
] + simple_mean_models

analytic_volatility_processes = [
    ARCH(3),
    FIGARCH(1, 1),
    GARCH(1, 1, 1),
    HARCH([1, 5, 22]),
    ConstantVariance(),
    EWMAVariance(0.94),
    FixedVariance(np.full_like(SP500, SP500.var())),
    MIDASHyperbolic(),
    RiskMetrics2006(),
]

other_volatility_processes = [
    APARCH(1, 1, 1, 1.5),
    EGARCH(1, 1, 1),
]

volatility_processes = analytic_volatility_processes + other_volatility_processes


@pytest.fixture(
    scope="module",
    params=list(itertools.product(simple_mean_models, analytic_volatility_processes)),
    ids=[
        f"{a.__class__.__name__}-{b}"
        for a, b in itertools.product(simple_mean_models, analytic_volatility_processes)
    ],
)
def forecastable_model(request) -> tuple[ARCHModelResult, ARCHModelFixedResult]:
    mod: ARCHModel
    vol: VolatilityProcess
    mod, vol = request.param
    mod.volatility = vol
    res = mod.fit()
    return res, mod.fix(res.params)


FIT_FIXED_PARAMS = []
count = 0
for model, vol in itertools.product(mean_models, volatility_processes):
    count += isinstance(vol, FIGARCH)
    marks = pytest.mark.slow if isinstance(vol, FIGARCH) and count > 1 else ()
    FIT_FIXED_PARAMS.append(pytest.param((model, vol), marks=marks))

FIT_FIXED_IDS = [
    f"{param[0][0].__class__.__name__}-{param[0][1]}{' (SLOW)' if len(mark) else ''}"
    for param, mark, _ in FIT_FIXED_PARAMS
]


@pytest.fixture(
    scope="module",
    params=FIT_FIXED_PARAMS,
    ids=FIT_FIXED_IDS,
)
def fit_fixed_models(request):
    mod: ARCHModel
    vol: VolatilityProcess
    mod, vol = request.param
    mod.volatility = vol
    res = mod.fit()
    return res, mod.fix(res.params)


class TestMeanModel:
    @classmethod
    def setup_class(cls):
        cls.rng = RandomState(1234)
        cls.T = 1000
        cls.resids = cls.rng.standard_normal(cls.T)
        zm = ZeroMean()
        zm.volatility = GARCH()
        seed = 12345
        random_state = np.random.RandomState(seed)
        zm.distribution = Normal(seed=random_state)
        sim_data = zm.simulate(np.array([0.1, 0.1, 0.8]), 1000)
        with pytest.raises(ValueError, match=r"Both initial value and x must"):
            zm.simulate(np.array([0.1, 0.1, 0.8]), 1000, initial_value=3.0)
        date_index = pd.date_range("2000-12-31", periods=1000, freq="W")
        cls.y = sim_data.data.values
        cls.y_df = pd.DataFrame(
            cls.y[:, None], columns=["LongVariableName"], index=date_index
        )

        cls.y_series = pd.Series(
            cls.y, name="VeryVeryLongLongVariableName", index=date_index
        )
        x = cls.resids + cls.rng.standard_normal(cls.T)
        cls.x = x[:, None]
        cls.x_df = pd.DataFrame(cls.x, columns=["LongExogenousName"])
        cls.resid_var = np.var(cls.resids)
        cls.sigma2 = np.zeros_like(cls.resids)
        cls.backcast = 1.0

    def test_constant_mean(self):
        cm = ConstantMean(self.y)
        parameters = np.array([5.0, 1.0])
        cm.simulate(parameters, self.T)
        assert_equal(cm.num_params, 1)
        with pytest.raises(ValueError, match=r"Both initial value and x must"):
            cm.simulate(parameters, self.T, x=np.array(10))
        bounds = cm.bounds()
        assert_equal(bounds, [(-np.inf, np.inf)])
        assert_equal(cm.constant, True)
        a, b = cm.constraints()
        assert_equal(a, np.empty((0, 1)))
        assert_equal(b, np.empty((0,)))
        assert isinstance(cm.volatility, ConstantVariance)
        assert isinstance(cm.distribution, Normal)
        assert cm.lags is None
        res = cm.fit(disp=DISPLAY)
        expected = np.array([self.y.mean(), self.y.var()])
        assert_almost_equal(res.params, expected)

        forecasts = res.forecast(horizon=20, start=20)
        direct = pd.DataFrame(
            index=np.arange(self.y.shape[0]),
            columns=[f"h.{i + 1:>02d}" for i in range(20)],
            dtype="double",
        )
        direct.iloc[20:, :] = res.params.iloc[0]
        # TODO
        # assert_frame_equal(direct, forecasts)
        assert isinstance(forecasts, ARCHModelForecast)
        assert isinstance(cm.__repr__(), str)
        assert isinstance(cm.__str__(), str)
        assert "<strong>" in cm._repr_html_()
        with pytest.raises(ValueError, match=r"horizon must be an integer >= 1"):
            res.forecast(horizon=0, start=20)

    def test_zero_mean(self):
        zm = ZeroMean(self.y)
        parameters = np.array([1.0])
        data = zm.simulate(parameters, self.T)
        assert_equal(data.shape, (self.T, 3))
        assert_equal(data["data"].shape[0], self.T)
        assert_equal(zm.num_params, 0)
        bounds = zm.bounds()
        assert_equal(bounds, [])
        assert_equal(zm.constant, False)
        a, b = zm.constraints()
        assert_equal(a, np.empty((0, 0)))
        assert_equal(b, np.empty((0,)))
        assert isinstance(zm.volatility, ConstantVariance)
        assert isinstance(zm.distribution, Normal)
        assert zm.lags is None
        res = zm.fit(disp=DISPLAY)
        assert_almost_equal(res.params, np.array([np.mean(self.y**2)]))

        forecasts = res.forecast(horizon=99)
        direct = pd.DataFrame(
            index=np.arange(self.y.shape[0]),
            columns=[f"h.{i + 1:>02d}" for i in range(99)],
            dtype="double",
        )
        direct.iloc[:, :] = 0.0
        assert isinstance(forecasts, ARCHModelForecast)
        # TODO
        # assert_frame_equal(direct, forecasts)
        garch = GARCH()
        zm.volatility = garch
        zm.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        assert isinstance(zm.__repr__(), str)
        assert isinstance(zm.__str__(), str)
        assert "<strong>" in zm._repr_html_()

    def test_harx(self):
        harx = HARX(self.y, self.x, lags=[1, 5, 22])
        assert harx.x is self.x
        params = np.array([1.0, 0.4, 0.3, 0.2, 1.0, 1.0])
        harx.simulate(params, self.T, x=self.rng.randn(self.T + 500, 1))
        iv = self.rng.randn(22, 1)
        x = self.rng.randn(self.T + 500, 1)
        alt_iv_data = harx.simulate(params, self.T, x=x, initial_value=1.0)
        assert_equal(alt_iv_data.shape, (self.T, 3))
        data = harx.simulate(params, self.T, x=x, initial_value=iv)
        assert_equal(data.shape, (self.T, 3))
        cols = ["data", "volatility", "errors"]
        for c in cols:
            assert c in data

        bounds = harx.bounds()
        for b in bounds:
            assert_equal(b[0], -np.inf)
            assert_equal(b[1], np.inf)
        assert_equal(len(bounds), 5)

        assert_equal(harx.num_params, 1 + 3 + self.x.shape[1])
        assert_equal(harx.constant, True)
        a, b = harx.constraints()
        assert_equal(a, np.empty((0, 5)))
        assert_equal(b, np.empty(0))
        res = harx.fit(disp=DISPLAY)
        with pytest.raises(ValueError, match=r"params have incorrect"):
            res.forecast(params=np.array([1.0, 1.0]))
        nobs = self.T - 22
        rhs = np.ones((nobs, 5))
        y = self.y
        lhs = y[22:]
        for i in range(self.T - 22):
            rhs[i, 1] = y[i + 21]
            rhs[i, 2] = np.mean(y[i + 17 : i + 22])
            rhs[i, 3] = np.mean(y[i : i + 22])
        rhs[:, 4] = self.x[22:, 0]
        params = np.linalg.pinv(rhs).dot(lhs)
        assert_almost_equal(params, res.params[:-1])

        assert harx.hold_back is None
        assert_equal(harx.lags, [1, 5, 22])
        assert_equal(harx.name, "HAR-X")
        assert_equal(harx.use_rotated, False)
        assert isinstance(harx.__repr__(), str)
        harx._repr_html_()
        res = harx.fit(cov_type="classic", disp=DISPLAY)
        assert isinstance(res.__repr__(), str)

    def test_harx_error(self):
        with pytest.raises(ValueError, match=r"Input to lags must be non-negative"):
            HARX(self.y, self.x, lags=[1, -5, 22])
        with pytest.raises(ValueError, match=r"When using the 1-d format of lags"):
            HARX(self.y, self.x, lags=[0, 1, 5, 22])
        with pytest.raises(ValueError, match=r"nput to lags must be non-negative"):
            HARX(self.y, self.x, lags=[[-1], [3]])
        with pytest.raises(ValueError, match=r"When using a 2-d array, all values"):
            HARX(self.y, self.x, lags=[[0], [0]])
        with pytest.raises(ValueError, match=r"lags contains redundant entries"):
            HARX(self.y, self.x, lags=[[1, 1, 3], [2, 3, 3]])
        with pytest.raises(ValueError, match=r"Incorrect format for lags"):
            HARX(self.y, self.x, lags=[[[1], [3]]])

    @pytest.mark.parametrize("constant", [True, False])
    def test_har(self, constant):
        har = HARX(self.y, lags=[1, 5, 22], constant=constant)
        params = np.array([1.0, 0.4, 0.3, 0.2, 1.0])
        if not constant:
            params = params[1:]
        data = har.simulate(params, self.T)
        assert_equal(data.shape, (self.T, 3))
        cols = ["data", "volatility", "errors"]
        for c in cols:
            assert c in data

        bounds = har.bounds()
        for b in bounds:
            assert_equal(b[0], -np.inf)
            assert_equal(b[1], np.inf)
        assert_equal(len(bounds), 3 + int(constant))

        assert_equal(har.num_params, 3 + int(constant))
        assert_equal(har.constant, constant)
        a, b = har.constraints()
        assert_equal(a, np.empty((0, 3 + int(constant))))
        assert_equal(b, np.empty(0))
        res = har.fit(disp=DISPLAY)
        nobs = self.T - 22
        rhs = np.ones((nobs, 4))
        y = self.y
        lhs = y[22:]
        for i in range(self.T - 22):
            rhs[i, 1] = y[i + 21]
            rhs[i, 2] = np.mean(y[i + 17 : i + 22])
            rhs[i, 3] = np.mean(y[i : i + 22])
        if not constant:
            rhs = rhs[:, 1:]
        params = np.linalg.pinv(rhs).dot(lhs)
        assert_almost_equal(params, res.params[:-1])

        with pytest.raises(ValueError, match=r"Due to backcasting and"):
            res.forecast(horizon=6, start=0)
        forecasts = res.forecast(horizon=6)
        t = self.y.shape[0]
        direct = pd.DataFrame(
            index=np.arange(t),
            columns=["h." + str(i + 1) for i in range(6)],
            dtype=float,
        )

        params = np.asarray(res.params)
        fcast = np.zeros(t + 6)
        for i in range(21, t):
            fcast[: i + 1] = self.y[: i + 1]
            fcast[i + 1 :] = 0.0
            for h in range(6):
                fcast[i + h + 1] = params[0]
                fcast[i + h + 1] += params[1] * fcast[i + h]
                fcast[i + h + 1] += params[2] * fcast[i + h - 4 : i + h + 1].mean()
                fcast[i + h + 1] += params[3] * fcast[i + h - 21 : i + h + 1].mean()
            direct.iloc[i, :] = fcast[i + 1 : i + 7]
        assert isinstance(forecasts, ARCHModelForecast)
        # TODO
        # assert_frame_equal(direct, forecasts)
        forecasts = res.forecast(res.params, horizon=6)
        assert isinstance(forecasts, ARCHModelForecast)
        # TODO
        # assert_frame_equal(direct, forecasts)

        assert har.hold_back is None
        assert_equal(har.lags, [1, 5, 22])
        assert_equal(har.name, "HAR")
        assert_equal(har.use_rotated, False)

        har = HARX(self.y_series, lags=[1, 5, 22])
        res = har.fit(disp=DISPLAY)
        direct = pd.DataFrame(
            index=self.y_series.index,
            columns=["h." + str(i + 1) for i in range(6)],
            dtype=float,
        )
        forecasts = res.forecast(horizon=6)
        params = np.asarray(res.params)
        fcast = np.zeros(t + 6)
        for i in range(21, t):
            fcast[: i + 1] = self.y[: i + 1]
            fcast[i + 1 :] = 0.0
            for h in range(6):
                fcast[i + h + 1] = params[0]
                fcast[i + h + 1] += params[1] * fcast[i + h]
                fcast[i + h + 1] += params[2] * fcast[i + h - 4 : i + h + 1].mean()
                fcast[i + h + 1] += params[3] * fcast[i + h - 21 : i + h + 1].mean()
            direct.iloc[i, :] = fcast[i + 1 : i + 7]
        assert isinstance(forecasts, ARCHModelForecast)
        # TODO
        # assert_frame_equal(direct, forecasts)

    def test_arx(self):
        arx = ARX(self.y, self.x, lags=3, hold_back=10, constant=False)
        params = np.array([0.4, 0.3, 0.2, 1.0, 1.0])
        data = arx.simulate(params, self.T, x=self.rng.randn(self.T + 500, 1))
        assert isinstance(data, pd.DataFrame)
        bounds = arx.bounds()
        for b in bounds:
            assert_equal(b[0], -np.inf)
            assert_equal(b[1], np.inf)
        assert_equal(len(bounds), 4)

        assert_equal(arx.num_params, 4)
        assert not arx.constant
        a, b = arx.constraints()
        assert_equal(a, np.empty((0, 4)))
        assert_equal(b, np.empty(0))
        res = arx.fit(last_obs=900, disp=DISPLAY)
        assert res.fit_stop == 900

        nobs = 900 - 10
        rhs = np.zeros((nobs, 4))
        y = self.y
        lhs = y[10:900]
        for i in range(10, 900):
            rhs[i - 10, 0] = y[i - 1]
            rhs[i - 10, 1] = y[i - 2]
            rhs[i - 10, 2] = y[i - 3]
        rhs[:, 3] = self.x[10:900, 0]
        params = np.linalg.pinv(rhs).dot(lhs)
        assert_almost_equal(params, res.params[:-1])
        assert_equal(arx.hold_back, 10)
        assert_equal(arx.lags, np.array([[1, 2, 3], [1, 2, 3]]))
        assert_equal(arx.name, "AR-X")
        assert_equal(arx.use_rotated, False)
        assert isinstance(arx.__repr__(), str)
        arx._repr_html_()

    def test_ar(self):
        ar = ARX(self.y, lags=3)
        params = np.array([1.0, 0.4, 0.3, 0.2, 1.0])
        data = ar.simulate(params, self.T)
        assert len(data) == self.T
        assert_equal(self.y, ar.y)

        bounds = ar.bounds()
        for b in bounds:
            assert_equal(b[0], -np.inf)
            assert_equal(b[1], np.inf)
        assert_equal(len(bounds), 4)

        assert_equal(ar.num_params, 4)
        assert ar.constant
        a, b = ar.constraints()
        assert_equal(a, np.empty((0, 4)))
        assert_equal(b, np.empty(0))
        res = ar.fit(disp=DISPLAY)

        nobs = 1000 - 3
        rhs = np.ones((nobs, 4))
        y = self.y
        lhs = y[3:1000]
        for i in range(3, 1000):
            rhs[i - 3, 1] = y[i - 1]
            rhs[i - 3, 2] = y[i - 2]
            rhs[i - 3, 3] = y[i - 3]
        params = np.linalg.pinv(rhs).dot(lhs)
        assert_almost_equal(params, res.params[:-1])

        forecasts = res.forecast(horizon=5)
        direct = pd.DataFrame(
            index=np.arange(y.shape[0]),
            columns=["h." + str(i + 1) for i in range(5)],
            dtype=float,
        )
        params = res.params.iloc[:-1]
        for i in range(2, y.shape[0]):
            fcast = np.zeros(y.shape[0] + 5)
            fcast[: y.shape[0]] = y.copy()
            for h in range(1, 6):
                reg = np.array(
                    [1.0, fcast[i + h - 1], fcast[i + h - 2], fcast[i + h - 3]]
                )
                fcast[i + h] = reg.dot(params)
            direct.iloc[i, :] = fcast[i + 1 : i + 6]
        assert isinstance(forecasts, ARCHModelForecast)
        # TODO
        # assert_frame_equal(direct, forecasts)

        assert ar.hold_back is None
        assert_equal(ar.lags, np.array([[1, 2, 3], [1, 2, 3]]))
        assert_equal(ar.name, "AR")
        assert_equal(ar.use_rotated, False)
        ar.__repr__()
        ar._repr_html_()

        ar = ARX(self.y_df, lags=5)
        ar.__repr__()
        ar = ARX(self.y_series, lags=5)
        ar.__repr__()
        res = ar.fit(disp=DISPLAY)
        assert isinstance(res.resid, pd.Series)
        assert isinstance(res.conditional_volatility, pd.Series)
        std_resid = res.resid / res.conditional_volatility
        std_resid.name = "std_resid"
        assert_series_equal(res.std_resid, std_resid)
        # Smoke bootstrap
        summ = ar.fit(disp=DISPLAY).summary()
        assert "Df Model:                            6" in str(summ)
        assert "Constant Variance" in str(summ)
        ar = ARX(self.y, lags=1, volatility=GARCH(), distribution=StudentsT())
        res = ar.fit(disp=DISPLAY, update_freq=UPDATE_FREQ, cov_type="classic")
        assert isinstance(res.param_cov, pd.DataFrame)
        sims = res.forecast(horizon=5, method="simulation")
        assert isinstance(sims.simulations.residual_variances, np.ndarray)
        assert isinstance(sims.simulations.residuals, np.ndarray)
        assert isinstance(sims.simulations.values, np.ndarray)
        assert isinstance(sims.simulations.variances, np.ndarray)

    def test_ar_no_lags(self):
        ar = ARX(self.y, lags=0)
        assert ar.lags is None
        res = ar.fit(disp=DISPLAY)
        param0, *_ = res.params
        assert_almost_equal(param0, self.y.mean())
        assert "lags: none" in ar.__str__()

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_ar_plot(self, agg_backend):
        ar = ARX(self.y, lags=1, volatility=GARCH(), distribution=StudentsT())
        res = ar.fit(disp=DISPLAY, update_freq=UPDATE_FREQ, cov_type="mle")
        res.plot()
        res.plot(annualize="D")
        res.plot(annualize="W")
        res.plot(annualize="M")
        with pytest.raises(ValueError, match=r"annualize not recognized"):
            res.plot(annualize="unknown")

        close_plots()

        res.plot(scale=360)
        res.hedgehog_plot(start=500)
        res.hedgehog_plot(start=500, plot_type="mean")
        res.hedgehog_plot(plot_type="volatility")
        res.hedgehog_plot(start=500, method="simulation", simulations=100)
        res.hedgehog_plot(plot_type="volatility", method="bootstrap")

        close_plots()

    def test_arch_arx(self):
        self.rng.seed(12345)
        x = self.rng.randn(500, 3)
        y = x.sum(1) + 3 * self.rng.standard_normal(500)

        am = ARX(y=y, x=x)
        res = am.fit(disp=DISPLAY)
        res.summary()
        assert isinstance(res.optimization_result, OptimizeResult)
        am.volatility = ARCH(p=2)
        results = am.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        assert isinstance(results.pvalues, pd.Series)
        assert_equal(
            list(results.pvalues.index),
            ["Const", "x0", "x1", "x2", "omega", "alpha[1]", "alpha[2]"],
        )

        am = ARX(y=y, lags=2, x=x)
        res = am.fit(disp=DISPLAY)
        summ = res.summary().as_text()
        res_repr = res.__repr__()
        assert str(hex(id(res))) in res_repr
        assert summ[:10] == res_repr[:10]

        am.volatility = ARCH(p=2)
        results = am.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        assert isinstance(results.pvalues, pd.Series)
        assert_equal(
            list(results.pvalues.index),
            [
                "Const",
                "y[1]",
                "y[2]",
                "x0",
                "x1",
                "x2",
                "omega",
                "alpha[1]",
                "alpha[2]",
            ],
        )

        x = pd.DataFrame(x, columns=["x0", "x1", "x2"])
        y = pd.Series(y, name="y")
        am = ARX(y=y, x=x)
        am.fit(disp=DISPLAY).summary()
        am.volatility = ARCH(p=2)
        results = am.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        assert isinstance(results.pvalues, pd.Series)
        assert_equal(
            list(results.pvalues.index),
            ["Const", "x0", "x1", "x2", "omega", "alpha[1]", "alpha[2]"],
        )

    def test_arch_model(self):
        am = arch_model(self.y)
        assert isinstance(am, ConstantMean)
        assert isinstance(am.volatility, GARCH)
        assert isinstance(am.distribution, Normal)

        am = arch_model(self.y, mean="harx", lags=[1, 5, 22])
        assert isinstance(am, HARX)
        assert isinstance(am.volatility, GARCH)

        am = arch_model(self.y, mean="har", lags=[1, 5, 22])
        assert isinstance(am, HARX)
        assert isinstance(am.volatility, GARCH)

        am = arch_model(self.y, self.x, mean="ls")
        assert isinstance(am, LS)
        assert isinstance(am.volatility, GARCH)
        am.__repr__()

        am = arch_model(self.y, mean="arx", lags=[1, 5, 22])
        assert isinstance(am, ARX)
        assert isinstance(am.volatility, GARCH)

        am = arch_model(self.y, mean="ar", lags=[1, 5, 22])
        assert isinstance(am, ARX)
        assert isinstance(am.volatility, GARCH)

        am = arch_model(self.y, mean="ar", lags=None)
        assert isinstance(am, ARX)
        assert isinstance(am.volatility, GARCH)

        am = arch_model(self.y, mean="zero")
        assert isinstance(am, ZeroMean)
        assert isinstance(am.volatility, GARCH)

        am = arch_model(self.y, vol="Harch")
        assert isinstance(am, ConstantMean)
        assert isinstance(am.volatility, HARCH)

        am = arch_model(self.y, vol="Constant")
        assert isinstance(am, ConstantMean)
        assert isinstance(am.volatility, ConstantVariance)

        am = arch_model(self.y, vol="arch")
        assert isinstance(am.volatility, ARCH)

        am = arch_model(self.y, vol="egarch")
        assert isinstance(am.volatility, EGARCH)

        am = arch_model(self.y, vol="figarch")
        assert isinstance(am.volatility, FIGARCH)

        am = arch_model(self.y, vol="aparch")
        assert isinstance(am.volatility, APARCH)

        with pytest.raises(ValueError, match=r"Unknown model type in mean"):
            arch_model(self.y, mean="unknown")
        with pytest.raises(ValueError, match=r"Unknown model type in vol"):
            arch_model(self.y, vol="unknown")
        with pytest.raises(ValueError, match=r"Unknown model type in dist"):
            arch_model(self.y, dist="unknown")

        am.fit(disp=DISPLAY)

    def test_pandas(self):
        am = arch_model(self.y_df, self.x_df, mean="ls")
        assert isinstance(am, LS)

    def test_summary(self):
        am = arch_model(self.y, mean="ar", lags=[1, 3, 5])
        res = am.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        res.summary()

        am = arch_model(self.y, mean="ar", lags=[1, 3, 5], dist="studentst")
        assert isinstance(am.distribution, StudentsT)
        res = am.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        res.summary()

        am = arch_model(self.y, mean="ar", lags=[1, 3, 5], dist="ged")
        assert isinstance(am.distribution, GeneralizedError)
        res = am.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        res.summary()

        am = arch_model(self.y, mean="ar", lags=[1, 3, 5], dist="skewt")
        res = am.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        assert isinstance(am.distribution, SkewStudent)
        res.summary()

    def test_errors(self):
        with pytest.raises(
            ValueError, match=r"lags does not follow a supported format"
        ):
            ARX(self.y, lags=np.array([[1, 2], [3, 4]]))
        x = self.rng.randn(self.y.shape[0] + 1, 1)
        with pytest.raises(ValueError, match=r"x must be nobs by n"):
            ARX(self.y, x=x)
        with pytest.raises(
            ValueError, match=r"When using a 2-d array, lags must by k by 2"
        ):
            HARX(self.y, lags=np.eye(3))
        with pytest.raises(ValueError, match=r"lags must be a positive integer"):
            ARX(self.y, lags=-1)
        with pytest.raises(ValueError, match=r"lags must be a positive integer"):
            ARX(self.y, x=self.rng.randn(1, 1), lags=-1)

        ar = ARX(self.y, lags=1)
        d = Normal()
        with pytest.raises(ValueError, match=r"Must subclass VolatilityProcess"):
            ar.volatility = d

        v = GARCH()
        with pytest.raises(ValueError, match=r"Must subclass Distribution"):
            ar.distribution = v
        x = self.rng.randn(1000, 1)
        with pytest.raises(ValueError, match=r"x must have nobs \+ burn rows"):
            ar.simulate(np.ones(5), 100, x=x)
        with pytest.raises(
            ValueError, match=r"params has the wrong number of elements"
        ):
            ar.simulate(np.ones(5), 100)
        with pytest.raises(ValueError, match=r"initial_value has the wrong shape"):
            ar.simulate(np.ones(3), 100, initial_value=self.rng.standard_normal(10))

        ar.volatility = ConstantVariance()
        with pytest.raises(ValueError, match=r"Unknown cov_type"):
            ar.fit(cov_type="unknown")

    def test_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            ARX(self.y, lags=[1, 2, 3, 12], hold_back=5)
            assert_equal(len(w), 1)

        with warnings.catch_warnings(record=True) as w:
            HARX(self.y, lags=[[1, 1, 1], [2, 5, 22]], use_rotated=True)
            assert_equal(len(w), 1)

    def test_har_lag_specifications(self):
        """Test equivalence of alternative lag specifications"""
        har = HARX(self.y, lags=[1, 2, 3])
        har_r = HARX(self.y, lags=[1, 2, 3], use_rotated=True)
        har_r_v2 = HARX(self.y, lags=3, use_rotated=True)
        ar = ARX(self.y, lags=[1, 2, 3])
        ar_v2 = ARX(self.y, lags=3)

        res_har = har.fit(disp=DISPLAY)
        res_har_r = har_r.fit(disp=DISPLAY)
        res_har_r_v2 = har_r_v2.fit(disp=DISPLAY)
        res_ar = ar.fit(disp=DISPLAY)
        res_ar_v2 = ar_v2.fit(disp=DISPLAY)
        assert_almost_equal(res_har.rsquared, res_har_r.rsquared)
        assert_almost_equal(res_har_r_v2.rsquared, res_har_r.rsquared)
        assert_almost_equal(np.asarray(res_ar.params), np.asarray(res_ar_v2.params))
        assert_almost_equal(np.asarray(res_ar.params), np.asarray(res_har_r_v2.params))
        assert_almost_equal(
            np.asarray(res_ar.param_cov), np.asarray(res_har_r_v2.param_cov)
        )
        assert_almost_equal(
            res_ar.conditional_volatility, res_har_r_v2.conditional_volatility
        )
        assert_almost_equal(res_ar.resid, res_har_r_v2.resid)

    def test_starting_values(self):
        am = arch_model(self.y, mean="ar", lags=[1, 3, 5])
        res = am.fit(cov_type="classic", update_freq=UPDATE_FREQ, disp=DISPLAY)
        res2 = am.fit(starting_values=res.params, update_freq=UPDATE_FREQ, disp=DISPLAY)
        assert isinstance(res, ARCHModelResult)
        assert isinstance(res2, ARCHModelResult)
        assert len(res.params) == 7
        assert len(res2.params) == 7

        am = arch_model(self.y, mean="zero")
        sv = np.array([1.0, 0.3, 0.8])
        with warnings.catch_warnings(record=True) as w:
            am.fit(starting_values=sv, update_freq=UPDATE_FREQ, disp=DISPLAY)
            assert len(w) == 1, str(w)

    def test_no_param_volatility(self):
        cm = ConstantMean(self.y)
        cm.volatility = EWMAVariance()
        cm.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        cm.volatility = RiskMetrics2006()
        cm.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)

        ar = ARX(self.y, lags=5)
        ar.volatility = EWMAVariance()
        ar.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        ar.volatility = RiskMetrics2006()
        ar.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        assert "tau0" in str(ar.volatility)
        assert "tau1" in str(ar.volatility)
        assert "kmax" in str(ar.volatility)

    def test_egarch(self):
        cm = ConstantMean(self.y)
        cm.volatility = EGARCH()
        res = cm.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        summ = res.summary()
        assert "Df Model:                            1" in str(summ)
        cm.distribution = StudentsT()
        cm.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)

    def test_multiple_lags(self):
        """Smoke test to ensure models estimate with multiple lags"""
        vp = {"garch": GARCH, "egarch": EGARCH, "harch": HARCH, "arch": ARCH}
        cm = ConstantMean(self.y)
        for name, process in vp.items():
            cm.volatility = process()
            cm.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
            for p in [1, 2, 3]:
                for o in [1, 2, 3]:
                    for q in [1, 2, 3]:
                        if name in ("arch",):
                            cm.volatility = process(p=p + o + q)
                            cm.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
                        elif name in ("harch",):
                            cm.volatility = process(lags=[p, p + o, p + o + q])
                            cm.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
                        else:
                            cm.volatility = process(p=p, o=o, q=q)
                            cm.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)

    def test_first_last_obs(self):
        ar = ARX(self.y, lags=5, hold_back=100)
        res = ar.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        resids = res.resid
        resid_copy = resids.copy()
        resid_copy[:100] = np.nan
        assert_equal(resids, resid_copy)

        ar.volatility = GARCH()
        res = ar.fit(update_freq=UPDATE_FREQ, disp=DISPLAY)
        resids = res.resid
        resid_copy = resids.copy()
        resid_copy[:100] = np.nan
        assert_equal(resids, resid_copy)

        ar = ARX(self.y, lags=5)
        ar.volatility = GARCH()
        res = ar.fit(update_freq=UPDATE_FREQ, last_obs=500, disp=DISPLAY)
        resids = res.resid
        resid_copy = resids.copy()
        resid_copy[500:] = np.nan
        assert_equal(resids, resid_copy)

        ar = ARX(self.y, lags=5, hold_back=100)
        ar.volatility = GARCH()
        res = ar.fit(update_freq=UPDATE_FREQ, last_obs=500, disp=DISPLAY)
        resids = res.resid
        resid_copy = resids.copy()
        resid_copy[:100] = np.nan
        resid_copy[500:] = np.nan
        assert_equal(resids, resid_copy)

        vol = res.conditional_volatility
        vol_copy = vol.copy()
        vol_copy[:100] = np.nan
        vol_copy[500:] = np.nan
        assert_equal(vol, vol_copy)
        assert_equal(self.y.shape[0], vol.shape[0])

        ar = ARX(self.y, lags=5)
        ar.volatility = GARCH()
        res = ar.fit(update_freq=UPDATE_FREQ, last_obs=500, disp=DISPLAY)
        resids = res.resid
        resid_copy = resids.copy()
        resid_copy[:5] = np.nan
        resid_copy[500:] = np.nan
        assert_equal(resids, resid_copy)

    def test_date_first_last_obs(self):
        y = self.y_series

        cm = ConstantMean(y)
        res = cm.fit(last_obs=y.index[900], disp=DISPLAY)

        cm = ConstantMean(y)
        res2 = cm.fit(last_obs=900, disp=DISPLAY)

        assert_equal(res.resid.values, res2.resid.values)

    def test_align(self):
        dates = pd.date_range("2000-01-01", "2010-01-01", freq=MONTH_END)
        columns = ["h." + f"{h + 1:>02}" for h in range(10)]
        forecasts = pd.DataFrame(self.rng.randn(120, 10), index=dates, columns=columns)

        aligned = _align_forecast(forecasts.copy(), align="origin")
        assert_frame_equal(aligned, forecasts)

        aligned = _align_forecast(forecasts.copy(), align="target")
        direct = forecasts.copy()
        for i in range(10):
            direct.iloc[(i + 1) :, i] = direct.iloc[: (120 - i - 1), i].values
            direct.iloc[: (i + 1), i] = np.nan
        assert_frame_equal(aligned, direct)

        with pytest.raises(ValueError, match=r"Unknown alignment"):
            _align_forecast(forecasts, align="unknown")

    def test_fixed_user_parameters(self):
        am = arch_model(self.y_series)
        res = am.fit(disp=DISPLAY)
        fixed_res = am.fix(res.params)
        assert_series_equal(
            res.conditional_volatility, fixed_res.conditional_volatility
        )
        assert_series_equal(res.params, fixed_res.params)
        assert_equal(res.aic, fixed_res.aic)
        assert_equal(res.bic, fixed_res.bic)
        assert_equal(res.loglikelihood, fixed_res.loglikelihood)
        assert_equal(res.num_params, fixed_res.num_params)
        # Smoke for summary
        fixed_res.summary()

    def test_fixed_user_parameters_new_model(self):
        am = arch_model(self.y_series)
        res = am.fit(disp=DISPLAY)
        new_am = arch_model(self.y_series)
        fixed_res = new_am.fix(res.params)
        assert_series_equal(
            res.conditional_volatility, fixed_res.conditional_volatility
        )
        assert_series_equal(res.params, fixed_res.params)
        assert_equal(res.aic, fixed_res.aic)
        assert_equal(res.bic, fixed_res.bic)
        assert_equal(res.loglikelihood, fixed_res.loglikelihood)
        assert_equal(res.num_params, fixed_res.num_params)

        # Test first and last dates
        am = arch_model(self.y_series)
        res = am.fit(disp=DISPLAY, first_obs=100, last_obs=900)
        new_am = arch_model(self.y_series)
        fixed_res = new_am.fix(res.params, first_obs=100, last_obs=900)
        assert_series_equal(res.params, fixed_res.params)
        assert_equal(res.aic, fixed_res.aic)
        assert_equal(res.bic, fixed_res.bic)
        assert_equal(res.loglikelihood, fixed_res.loglikelihood)
        assert_equal(res.num_params, fixed_res.num_params)

    @pytest.mark.parametrize("display", ["off", "final"])
    def test_output_options(self, display):
        am = arch_model(self.y_series)
        orig_stdout = sys.stdout

        try:
            sio = StringIO()
            sys.stdout = sio
            am.fit(disp=display)
            sio.seek(0)
            output = sio.read()
            if display == "off":
                assert len(output) == 0
            else:
                assert len(output) > 0
        finally:
            sys.stdout = orig_stdout

    def test_convergence_warning(self):
        y = np.array(
            [
                0.83277114,
                0.45194014,
                -0.33475561,
                -0.49463896,
                0.54715787,
                1.11895382,
                1.31280266,
                0.81464021,
                0.8532107,
                1.0967188,
                0.9346354,
                0.92289249,
                1.01339085,
                1.071065,
                1.42413486,
                1.15392453,
                1.10929691,
                0.96162061,
                0.96489515,
                0.93250153,
                1.34509807,
                1.80951607,
                1.66313783,
                1.38610821,
                1.26381761,
            ]
        )
        am = arch_model(y, mean="ARX", lags=10, p=5, q=0)

        with pytest.warns(DataScaleWarning, match=r"y is poorly scaled"):
            am.fit(disp=DISPLAY)

        with pytest.warns(DataScaleWarning, match=r"y is poorly scaled"):
            am.fit(show_warning=True, disp=DISPLAY)

        with pytest.warns(DataScaleWarning, match=r"y is poorly scaled"):
            am.fit(show_warning=False, disp=DISPLAY)

    def test_first_after_last(self):
        am = arch_model(self.y_series)
        with pytest.raises(ValueError, match=r"first_obs and last_obs produce"):
            am.fit(disp=DISPLAY, first_obs=500, last_obs=480)

        with pytest.raises(ValueError, match=r"first_obs and last_obs produce"):
            am.fit(
                disp=DISPLAY,
                first_obs=self.y_series.index[500],
                last_obs=self.y_series.index[480],
            )

    def test_sample_adjustment(self):
        am = arch_model(self.y_series, vol="Constant")
        res = am.fit(disp=DISPLAY)

        res_adj = am.fit(disp=DISPLAY, first_obs=0, last_obs=self.y_series.shape[0] + 1)
        assert_equal(res.resid.values, res_adj.resid.values)
        assert_equal(res.params.values, res_adj.params.values)

        res = am.fit(disp=DISPLAY, first_obs=100)
        assert res.fit_start == 100
        res_adj = am.fit(disp=DISPLAY, first_obs=self.y_series.index[100])
        assert_equal(res.params.values, res_adj.params.values)
        assert_equal(res.resid.values, res_adj.resid.values)

        res = am.fit(disp=DISPLAY, last_obs=900)
        res2 = am.fit(disp=DISPLAY, last_obs=self.y_series.index[900])
        assert_equal(res.params.values, res2.params.values)
        assert_equal(res.resid.values, res2.resid.values)

        res = am.fit(disp=DISPLAY, first_obs=100, last_obs=900)
        res2 = am.fit(
            disp=DISPLAY,
            first_obs=self.y_series.index[100],
            last_obs=self.y_series.index[900],
        )
        assert_equal(res.params.values, res2.params.values)
        assert_equal(res.resid.values, res2.resid.values)

    def test_model_obs_equivalence(self):
        """Tests models that should use the same observation"""
        am = arch_model(self.y_series.iloc[100:900])
        res = am.fit(disp=DISPLAY)
        am = arch_model(self.y_series)
        res2 = am.fit(disp=DISPLAY, first_obs=100, last_obs=900)
        index = self.y_series.index
        res3 = am.fit(disp=DISPLAY, first_obs=index[100], last_obs=index[900])
        assert_equal(res.params.values, res2.params.values)
        assert_equal(res2.params.values, res3.params.values)

        am = arch_model(self.y_series, hold_back=100)
        res4 = am.fit(disp=DISPLAY, last_obs=900)
        assert_equal(res.params.values, res4.params.values)

    def test_model_obs_equivalence_ar(self):
        """Tests models that should use the same observation"""
        am = arch_model(self.y_series.iloc[100:900], mean="AR", lags=[1, 2, 4])
        res = am.fit(disp=DISPLAY)
        am = arch_model(self.y_series, mean="AR", lags=[1, 2, 4])
        res2 = am.fit(disp=DISPLAY, first_obs=100, last_obs=900)
        index = self.y_series.index
        res3 = am.fit(disp=DISPLAY, first_obs=index[100], last_obs=index[900])
        assert_almost_equal(res.params.values, res2.params.values, decimal=4)
        assert_almost_equal(res2.params.values, res3.params.values, decimal=4)

        am = arch_model(self.y_series, mean="AR", lags=[1, 2, 4], hold_back=100)
        res4 = am.fit(disp=DISPLAY, first_obs=4, last_obs=900)
        assert_almost_equal(res.params.values, res4.params.values, decimal=4)
        assert am.hold_back == 100

    def test_constant_mean_fixed_variance(self):
        rng = RandomState(1234)
        variance = 2 + rng.standard_normal(self.y.shape[0]) ** 2.0
        std = np.sqrt(variance)
        y = pd.Series(
            std * rng.standard_normal(self.y_series.shape[0]), index=self.y_series.index
        )

        mod = ConstantMean(y, volatility=FixedVariance(variance))
        res = mod.fit(disp=DISPLAY)
        res.summary()
        assert len(res.params) == 2
        assert "scale" in res.params.index

        mod = ARX(self.y_series, lags=[1, 2, 3], volatility=FixedVariance(variance))
        res = mod.fit(disp=DISPLAY)
        assert len(res.params) == 5
        assert "scale" in res.params.index

        mod = ARX(
            self.y_series,
            lags=[1, 2, 3],
            volatility=FixedVariance(variance, unit_scale=True),
        )
        res = mod.fit(disp=DISPLAY)
        assert len(res.params) == 4
        assert "scale" not in res.params.index

    def test_optimization_options(self):
        norm = Normal(seed=RandomState([12891298, 843084]))
        am = arch_model(None)
        am.distribution = norm
        data = am.simulate(np.array([0.0, 0.1, 0.1, 0.85]), 2500)
        am = arch_model(data.data)
        std = am.fit(disp=DISPLAY)
        loose = am.fit(tol=1e-2, disp=DISPLAY)
        assert std.loglikelihood >= loose.loglikelihood
        with warnings.catch_warnings(record=True) as w:
            short = am.fit(options={"maxiter": 3}, disp=DISPLAY)
        assert len(w) == 1, str(w)
        assert std.loglikelihood >= short.loglikelihood
        assert short.convergence_flag != 0

    def test_little_or_no_data(self):
        mod = HARX(self.y[:24], lags=[1, 5, 22])
        with pytest.raises(ValueError, match=r"Insufficient data, 4 regressors"):
            mod.fit(disp=DISPLAY)
        mod = HARX(None, lags=[1, 5, 22])
        with pytest.raises(RuntimeError, match=r"Cannot estimate model without"):
            mod.fit(disp=DISPLAY)

    def test_empty_mean(self):
        mod = HARX(
            self.y,
            None,
            None,
            False,
            volatility=ConstantVariance(),
            distribution=Normal(),
        )
        res = mod.fit(disp=DISPLAY)

        mod = ZeroMean(self.y, volatility=ConstantVariance(), distribution=Normal())
        res_z = mod.fit(disp=DISPLAY)

        assert res.num_params == res_z.num_params
        assert_series_equal(res.params, res_z.params)
        assert res.loglikelihood == res_z.loglikelihood


@pytest.mark.parametrize(
    "volatility",
    [GARCH, EGARCH, RiskMetrics2006, EWMAVariance, HARCH, ConstantVariance],
)
def test_backcast(volatility, simulated_data):
    zm = ZeroMean(simulated_data, volatility=volatility())
    res = zm.fit(disp=DISPLAY)
    bc = zm.volatility.backcast(np.asarray(res.resid))
    if volatility is EGARCH:
        bc = np.exp(bc)
    res2 = zm.fit(backcast=bc, disp=DISPLAY)
    assert_array_almost_equal(res.params, res2.params)
    if volatility is RiskMetrics2006:
        zm.fit(backcast=bc[0], disp=DISPLAY)


def test_backcast_error(simulated_data):
    zm = ZeroMean(simulated_data, volatility=GARCH())
    with pytest.raises(ValueError, match=r"User backcast value"):
        zm.fit(backcast=-1, disp=DISPLAY)
    zm = ZeroMean(simulated_data, volatility=RiskMetrics2006())
    with pytest.raises(ValueError, match=r"User backcast must be either"):
        zm.fit(backcast=np.ones(100), disp=DISPLAY)


@pytest.mark.parametrize(
    "volatility",
    [
        ConstantVariance,
        GARCH,
        EGARCH,
        FIGARCH,
        APARCH,
        HARCH,
        MIDASHyperbolic,
        RiskMetrics2006,
        EWMAVariance,
    ],
)
def test_fit_smoke(simulated_data, volatility):
    zm = ZeroMean(simulated_data, volatility=volatility())
    zm.fit(disp=DISPLAY)


def test_arch_lm(simulated_data):
    zm = ZeroMean(simulated_data, volatility=GARCH())
    res = zm.fit(disp=DISPLAY)
    wald = res.arch_lm_test()
    nobs = simulated_data.shape[0]
    df = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
    assert wald.df == df
    assert "Standardized" not in wald.null
    assert "Standardized" not in wald.alternative
    assert "H0: Standardized" not in wald.__repr__()
    assert "heteroskedastic" in wald.__repr__()

    resids2 = pd.Series(res.resid**2)
    data = [resids2.shift(i) for i in range(df + 1)]
    data = pd.concat(data, axis=1).dropna()
    lhs = data.iloc[:, 0]
    rhs = smtools.add_constant(data.iloc[:, 1:])
    ols_res = smlm.OLS(lhs, rhs).fit()
    assert_almost_equal(wald.stat, nobs * ols_res.rsquared)
    assert len(wald.critical_values) == 3
    assert "10%" in wald.critical_values

    wald = res.arch_lm_test(lags=5)
    assert wald.df == 5
    assert_almost_equal(wald.pval, 1 - stats.chi2(5).cdf(wald.stat))

    wald = res.arch_lm_test(standardized=True)
    assert wald.df == df
    assert "Standardized" in wald.null
    assert "Standardized" in wald.alternative
    assert_almost_equal(wald.pval, 1 - stats.chi2(df).cdf(wald.stat))
    assert "H0: Standardized" in wald.__repr__()


def test_autoscale():
    rs = np.random.RandomState(34254321)
    dist = Normal(seed=rs)
    am = arch_model(None)
    am.distribution = dist
    data = am.simulate([0, 0.0001, 0.05, 0.94], nobs=1000)
    am = arch_model(data.data)
    with pytest.warns(DataScaleWarning, match=r"y is poorly scaled"):
        res = am.fit(disp=DISPLAY)
    assert_almost_equal(res.scale, 1.0)

    am = arch_model(data.data, rescale=True)
    res_auto = am.fit(disp=DISPLAY)
    assert_almost_equal(res_auto.scale, 10.0)

    am = arch_model(10 * data.data)
    res_manual = am.fit(disp=DISPLAY)
    assert_series_equal(res_auto.params, res_manual.params)

    res_no = arch_model(data.data, rescale=False).fit(disp=DISPLAY)
    assert res_no.scale == 1.0

    am = arch_model(10000 * data.data, rescale=True)
    res_big = am.fit(disp=DISPLAY)
    assert_almost_equal(res_big.scale, 0.1)


def test_no_variance():
    mod = arch_model(np.ones(100))
    with pytest.warns(ConvergenceWarning, match=r"The optimizer returned"):
        mod.fit(disp=DISPLAY)


def test_1d_exog():
    rs = np.random.RandomState(329302)
    y = rs.standard_normal(300)
    x = rs.standard_normal(300)
    am = arch_model(y, x, mean="ARX", lags=2, vol="ARCH", q=0)
    res = am.fit()
    am = arch_model(y, x[:, None], mean="ARX", lags=2, vol="ARCH", q=0)
    res2 = am.fit()
    assert_series_equal(res.params, res2.params)


def test_harx_lag_spec(simulated_data):
    harx_1 = HARX(simulated_data, lags=[1, 5, 22])
    harx_2 = HARX(simulated_data, lags=[1, 5, 22], use_rotated=True)
    harx_3 = HARX(simulated_data, lags=[[1, 1, 1], [1, 5, 22]])
    harx_4 = HARX(simulated_data, lags=[[1, 2, 6], [1, 5, 22]])

    r2 = harx_1.fit().rsquared
    assert_almost_equal(harx_2.fit().rsquared, r2)
    assert_almost_equal(harx_3.fit().rsquared, r2)
    assert_almost_equal(harx_4.fit().rsquared, r2)


def test_backcast_restricted(simulated_data):
    # GH 440
    mod = arch_model(simulated_data)
    res = mod.fit(disp="off")
    subset = (
        simulated_data[100:600]
        if isinstance(simulated_data, np.ndarray)
        else simulated_data.iloc[100:600]
    )
    mod_restricted = arch_model(subset)
    res_restricted = mod_restricted.fit(disp="off")
    res_limited = mod.fit(first_obs=100, last_obs=600, disp="off")
    assert_almost_equal(res_restricted.model._backcast, res_restricted.model._backcast)
    assert np.abs(res.model._backcast - res_limited.model._backcast) > 1e-8


def test_missing_data_exception():
    y = np.random.standard_normal(1000)
    y[::29] = np.nan
    with pytest.raises(ValueError, match=r"NaN or inf values"):
        arch_model(y)
    y = np.random.standard_normal(1000)
    y[::53] = np.inf
    with pytest.raises(ValueError, match=r"NaN or inf values"):
        arch_model(y)
    y[::29] = np.nan
    y[::53] = np.inf
    with pytest.raises(ValueError, match=r"NaN or inf values"):
        arch_model(y)


@pytest.mark.parametrize("first_obs", [None, 250])
@pytest.mark.parametrize("last_obs", [None, 2750])
@pytest.mark.parametrize("vol", [RiskMetrics2006(), EWMAVariance()])
def test_parameterless_fit(first_obs, last_obs, vol):
    base = ConstantMean(SP500, volatility=vol)
    base_res = base.fit(first_obs=first_obs, last_obs=last_obs, disp="off")
    mod = ZeroMean(SP500, volatility=vol)
    res = mod.fit(first_obs=first_obs, last_obs=last_obs, disp="off")
    assert res.conditional_volatility.shape == base_res.conditional_volatility.shape


def test_invalid_vol_dist():
    with pytest.raises(TypeError, match=r"volatility must inherit"):
        ConstantMean(SP500, volatility="GARCH")
    with pytest.raises(TypeError, match=r"distribution must inherit"):
        ConstantMean(SP500, distribution="Skew-t")


def test_param_cov():
    mod = ConstantMean(SP500)
    res = mod.fit(disp="off")
    mod._backcast = None
    cov = mod.compute_param_cov(res.params)
    k = res.params.shape[0]
    assert cov.shape == (k, k)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plot_bad_index(agg_backend):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    idx = sorted(f"{a}{b}{c}" for a, b, c in product(*([ascii_lowercase] * 3)))
    sp500_copy = SP500.copy()
    sp500_copy.index = idx[: sp500_copy.shape[0]]
    res = ConstantMean(sp500_copy).fit(disp=False)
    fig = res.plot()
    assert isinstance(fig, plt.Figure)


def test_false_reindex():
    res = ConstantMean(SP500, volatility=GARCH()).fit(disp="off")
    fcast = res.forecast(start=0, reindex=True)
    assert fcast.mean.shape[0] == SP500.shape[0]
    assert_series_equal(pd.Series(fcast.mean.index), pd.Series(SP500.index))


def test_invalid_arch_model():
    with pytest.raises(AssertionError):
        arch_model(SP500, p="3")


def test_last_obs_equiv():
    y = SP500.iloc[:-100]
    res1 = arch_model(y).fit(disp=False)
    res2 = arch_model(SP500).fit(last_obs=SP500.index[-100], disp=False)
    assert_allclose(res1.model._backcast, res2.model._backcast, rtol=1e-6)
    assert_allclose(res1.params, res2.params, rtol=1e-6)


@pytest.mark.parametrize("first", [0, 250])
@pytest.mark.parametrize("last", [0, 250])
@pytest.mark.parametrize("mean", ["Constant", "AR"])
def test_last_obs_equiv_param(first, last, mean):
    lags = None if mean == "constant" else 2
    nobs = SP500.shape[0]
    last_obs = SP500.index[-last] if last else None
    y = SP500.iloc[first : nobs - last]
    res1 = arch_model(y, mean=mean, lags=lags).fit(disp=False)
    res2 = arch_model(SP500, mean=mean, lags=lags).fit(
        first_obs=y.index[0], last_obs=last_obs, disp=False
    )
    cv1 = res1.conditional_volatility
    cv2 = res2.conditional_volatility
    assert np.isfinite(cv1).sum() == np.isfinite(cv2).sum()
    r1 = res1.resid
    r2 = res2.resid
    assert np.isfinite(r1).sum() == np.isfinite(r2).sum()

    assert_allclose(res1.model._backcast, res2.model._backcast, rtol=RTOL)
    assert_allclose(res1.params, res2.params, rtol=RTOL)
    assert_allclose(cv1[np.isfinite(cv1)], cv2[np.isfinite(cv2)], rtol=RTOL)


@pytest.mark.parametrize("use_pandas", [True, False])
def test_all_attr_numpy_pandas(use_pandas):
    data = SP500
    if not use_pandas:
        data = np.asanyarray(data)
    mod = arch_model(data, p=1, o=1, q=1)
    res = mod.fit(disp="off")
    for attr in dir(res):
        if not attr.startswith("_"):
            getattr(res, attr)


@pytest.mark.slow
def test_figarch_power():
    base = ConstantMean(SP500, volatility=FIGARCH())
    fiavgarch = ConstantMean(SP500, volatility=FIGARCH(power=1.0))
    base_res = base.fit(disp=DISPLAY)
    fiavgarch_res = fiavgarch.fit(disp=DISPLAY, update_freq=UPDATE_FREQ)
    assert np.abs(base_res.loglikelihood - fiavgarch_res.loglikelihood) > 1.0
    alt_fiavgarch = arch_model(SP500, vol="FIGARCH", power=1.0)
    alt_fiavgarch_res = alt_fiavgarch.fit(disp=DISPLAY, update_freq=UPDATE_FREQ)
    assert np.abs(alt_fiavgarch_res.loglikelihood - fiavgarch_res.loglikelihood) < 1.0


@pytest.mark.parametrize("lags", [1, 5, [1, 4]])
def test_arch_lm_ar_model(lags):
    rs = RandomState(1234)
    y = rs.standard_normal(1000)
    model = arch_model(y, mean="AR", lags=lags, vol="GARCH", rescale=True)
    fit = model.fit()
    val = fit.arch_lm_test()
    assert val.stat > 0
    assert val.pval <= 1


@pytest.mark.parametrize("use_numpy", [True, False])
def test_non_contiguous_input(use_numpy):
    # GH 740
    if use_numpy:
        y = np.array(SP500, copy=True)[::2]
        assert not y.flags["C_CONTIGUOUS"]
    else:
        y = SP500.iloc[::2]
    mod = arch_model(y, mean="Zero")
    res = mod.fit()
    assert res.params.shape[0] == 3


@pytest.mark.slow
def test_fixed_equivalence(fit_fixed_models):
    res, res_fixed = fit_fixed_models

    assert_allclose(res.aic, res_fixed.aic)
    assert_allclose(res.bic, res_fixed.bic)
    assert_allclose(res.loglikelihood, res_fixed.loglikelihood)
    assert res.nobs == res_fixed.nobs
    assert res.num_params == res_fixed.num_params
    assert_allclose(res.params, res_fixed.params)
    assert_allclose(res.conditional_volatility, res_fixed.conditional_volatility)
    assert_allclose(res.std_resid, res_fixed.std_resid)
    assert_allclose(res.resid, res_fixed.resid)
    assert_allclose(res.arch_lm_test(5).stat, res_fixed.arch_lm_test(5).stat)
    assert res.model.__class__ is res_fixed.model.__class__
    assert res.model.volatility.__class__ is res_fixed.model.volatility.__class__
    assert isinstance(res.summary(), type(res_fixed.summary()))
    if res.num_params > 0:
        assert "std err" in str(res.summary())
        assert "std err" not in str(res_fixed.summary())


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_fixed_equivalence_plots(fit_fixed_models, agg_backend):
    import matplotlib as mpl  # noqa: PLC0415

    backend = mpl.get_backend()
    mpl.use("agg")

    res, res_fixed = fit_fixed_models

    fig = res.plot()
    fixed_fig = res_fixed.plot()
    assert isinstance(fig, type(fixed_fig))

    close_plots()
    mpl.use(backend)


@pytest.mark.slow
@pytest.mark.parametrize("simulations", [1, 100])
def test_fixed_equivalence_forecastable(forecastable_model, simulations):
    res, res_fixed = forecastable_model
    f1 = res.forecast(horizon=5)
    f2 = res_fixed.forecast(horizon=5)
    assert isinstance(f1, type(f2))
    assert_allclose(f1.mean, f2.mean)
    assert_allclose(f1.variance, f2.variance)

    f1 = res.forecast(horizon=5, method="simulation", simulations=simulations)
    f2 = res_fixed.forecast(horizon=5, method="simulation", simulations=simulations)
    assert isinstance(f1, type(f2))
    f1 = res.forecast(horizon=5, method="bootstrap", simulations=simulations)
    f2 = res_fixed.forecast(horizon=5, method="bootstrap", simulations=simulations)
    assert isinstance(f1, type(f2))


@pytest.mark.slow
@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_fixed_equivalence_forecastable_plots(forecastable_model, agg_backend):
    res, res_fixed = forecastable_model
    fig1 = res.hedgehog_plot(start=SP500.shape[0] - 25)
    fig2 = res_fixed.hedgehog_plot(start=SP500.shape[0] - 25)
    assert isinstance(fig1, type(fig2))
    close_plots()
