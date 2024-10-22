# TODO: Tests for features that are just called
# TODO: Test for trend='ctt'
from arch.compat.statsmodels import dataset_loader

import os
from typing import NamedTuple, Optional
import warnings

import numpy as np
from numpy import ceil, diff, log, polyval
from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
import scipy.stats as stats
from statsmodels.datasets import macrodata, modechoice, nile, randhie, sunspots
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import _autolag, lagmat

from arch.unitroot import ADF, DFGLS, KPSS, PhillipsPerron, VarianceRatio, ZivotAndrews
from arch.unitroot.critical_values.dickey_fuller import tau_2010
from arch.unitroot.unitroot import (
    _autolag_ols,
    _autolag_ols_low_memory,
    _is_reduced_rank,
    auto_bandwidth,
    mackinnoncrit,
    mackinnonp,
)
from arch.utility.exceptions import InfeasibleTestException, PerformanceWarning

DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1

BASE_PATH = os.path.split(os.path.abspath(__file__))[0]
DATA_PATH = os.path.join(BASE_PATH, "data")
ZIVOT_ANDREWS_DATA = pd.read_csv(
    os.path.join(DATA_PATH, "zivot-andrews.csv"), index_col=0
)

# Time series to test the autobandwidth method against its implementation under R
REAL_TIME_SERIES = [8, 9, 2, 4, 8, 9, 9, 4, 4, 9, 7, 1, 1, 9, 4, 9, 3]
TRUE_BW_FROM_R_BA = 3.033886
TRUE_BW_FROM_R_PA = 7.75328
TRUE_BW_FROM_R_QS = 3.851586


class TestUnitRoot:
    @classmethod
    def setup_class(cls):
        cls.rng = RandomState(12345)

        data = dataset_loader(macrodata)
        cls.cpi = log(data["cpi"])
        cls.realgdp = data["realgdp"]
        cls.inflation = diff(cls.cpi)
        cls.inflation_change = diff(cls.inflation)

    def test_adf_no_options(self):
        adf = ADF(self.inflation)
        assert_almost_equal(adf.stat, -3.09310, DECIMAL_4)
        assert_equal(adf.lags, 2)
        assert_almost_equal(adf.pvalue, 0.027067, DECIMAL_4)
        adf.regression.summary()
        adf2 = ADF(self.inflation, low_memory=True)
        assert_equal(adf2.lags, 2)

    def test_adf_no_lags(self):
        adf = ADF(self.inflation, lags=0).stat
        assert_almost_equal(adf, -6.56880, DECIMAL_4)

    def test_adf_nc_no_lags(self):
        adf = ADF(self.inflation, trend="n", lags=0)
        assert_almost_equal(adf.stat, -3.88845, DECIMAL_4)
        # 16.239

    def test_adf_c_no_lags(self):
        adf = ADF(self.inflation, trend="c", lags=0)
        assert_almost_equal(adf.stat, -6.56880, DECIMAL_4)
        assert_equal(adf.nobs, self.inflation.shape[0] - adf.lags - 1)

    def test_adf_ct_no_lags(self):
        adf = ADF(self.inflation, trend="ct", lags=0)
        assert_almost_equal(adf.stat, -6.66705, DECIMAL_4)

    def test_adf_lags_10(self):
        adf = ADF(self.inflation, lags=10)
        assert_almost_equal(adf.stat, -2.28375, DECIMAL_4)
        adf.summary()

    def test_adf_auto_bic(self):
        adf = ADF(self.inflation, method="bic")
        assert_equal(adf.lags, 2)
        adf2 = ADF(self.inflation, method="bic", low_memory=True)
        assert_equal(adf2.lags, 2)

    def test_adf_critical_value(self):
        adf = ADF(self.inflation, trend="c", lags=3)
        adf_cv = adf.critical_values
        temp = polyval(tau_2010["c"][0, :, ::-1].T, 1.0 / adf.nobs)
        cv = {"1%": temp[0], "5%": temp[1], "10%": temp[2]}
        for k, v in cv.items():
            assert_almost_equal(v, adf_cv[k])

    def test_adf_auto_t_stat(self):
        adf = ADF(self.inflation, method="t-stat")
        assert_equal(adf.lags, 11)
        adf2 = ADF(self.inflation, method="t-stat", low_memory=True)
        assert_equal(adf2.lags, 11)
        old_stat = adf.stat
        adf = ADF(self.inflation, method="t-stat", lags=adf.lags + 1)
        assert adf.stat != old_stat
        old_stat = adf.stat
        assert_equal(adf.y, self.inflation)
        adf = ADF(self.inflation, method="t-stat", trend="ctt", lags=adf.lags)
        assert adf.stat != old_stat
        assert adf.trend == "ctt"
        assert len(adf.valid_trends) == len(("n", "c", "ct", "ctt"))
        for d in adf.valid_trends:
            assert d in ("n", "c", "ct", "ctt")
        assert adf.null_hypothesis == "The process contains a unit root."
        assert adf.alternative_hypothesis == "The process is weakly stationary."

    def test_kpss_auto(self):
        kpss = KPSS(self.inflation, lags=-1)
        m = self.inflation.shape[0]
        lags = np.ceil(12.0 * (m / 100) ** (1.0 / 4))
        assert_equal(kpss.lags, lags)

    def test_kpss(self):
        kpss = KPSS(self.inflation, trend="ct", lags=12)
        assert_almost_equal(kpss.stat, 0.235581902996454, DECIMAL_4)
        assert_equal(self.inflation.shape[0], kpss.nobs)
        kpss.summary()

    def test_kpss_c(self):
        kpss = KPSS(self.inflation, trend="c", lags=12)
        assert_almost_equal(kpss.stat, 0.3276290340191141, DECIMAL_4)

    def test_pp(self):
        pp = PhillipsPerron(self.inflation, lags=12)
        assert_almost_equal(pp.stat, -7.8076512, DECIMAL_4)
        assert pp.test_type == "tau"
        pp = PhillipsPerron(self.inflation, lags=12, test_type="rho")
        assert_almost_equal(pp.stat, -108.1552688, DECIMAL_2)
        pp.summary()

    def test_pp_regression(self):
        pp = PhillipsPerron(self.inflation, lags=12)
        reg = pp.regression
        assert len(reg.params) == 2
        assert "(HAC) using 12 lags" in str(reg.summary())

    def test_pp_bad_type(self):
        pp = PhillipsPerron(self.inflation, lags=12)
        assert isinstance(pp.test_type, str)

    def test_pp_auto(self):
        pp = PhillipsPerron(self.inflation)
        n = self.inflation.shape[0] - 1
        lags = ceil(12.0 * ((n / 100.0) ** (1.0 / 4.0)))
        assert_equal(pp.lags, lags)
        assert_almost_equal(pp.stat, -8.135547778, DECIMAL_4)
        pp = PhillipsPerron(self.inflation, test_type="rho")
        assert_almost_equal(pp.stat, -118.7746451, DECIMAL_2)

    def test_dfgls_c(self):
        dfgls = DFGLS(self.inflation, trend="c", lags=0)
        assert_almost_equal(dfgls.stat, -6.017304, DECIMAL_4)
        dfgls.summary()
        dfgls.regression.summary()
        assert dfgls.trend == "c"
        dfgls = DFGLS(self.inflation, trend="ct", lags=0)
        assert dfgls.trend == "ct"
        dfgls = DFGLS(self.inflation, trend="c")
        assert dfgls.trend == "c"
        dfgls_hm = DFGLS(self.inflation, trend="c", lags=0, low_memory=False)
        assert_almost_equal(dfgls_hm.stat, -6.017304, DECIMAL_4)
        dfgls_lm = DFGLS(self.inflation, trend="c", lags=0, low_memory=True)
        assert_almost_equal(dfgls_lm.stat, -6.017304, DECIMAL_4)

    def test_dfgls(self):
        dfgls = DFGLS(self.inflation, trend="ct", lags=0)
        assert_almost_equal(dfgls.stat, -6.300927, DECIMAL_4)
        dfgls.summary()
        dfgls.regression.summary()

    def test_dfgls_auto(self):
        dfgls = DFGLS(self.inflation, trend="ct", method="bic", max_lags=3)
        assert_equal(dfgls.lags, 2)
        assert_equal(dfgls.max_lags, 3)
        assert_almost_equal(dfgls.stat, -2.9035369, DECIMAL_4)
        dfgls = DFGLS(self.inflation, trend="ct", method="bic", max_lags=1)
        assert_equal(dfgls.lags, 1)

    def test_dfgls_bad_trend(self):
        dfgls = DFGLS(self.inflation, trend="ct", method="bic", max_lags=3)
        with pytest.raises(ValueError):
            DFGLS(self.inflation, method="bic", max_lags=3, trend="n")

        assert dfgls != 0.0

    def test_dfgls_auto_low_memory(self):
        y = np.cumsum(self.rng.standard_normal(200000))
        dfgls = DFGLS(y, trend="c", method="bic", low_memory=None)
        assert isinstance(dfgls.stat, float)
        assert dfgls._low_memory

    def test_negative_lag(self):
        with pytest.raises(ValueError):
            ADF(self.inflation, lags=-1)

    def test_invalid_determinstic(self):
        with pytest.raises(ValueError):
            ADF(self.inflation, trend="bad-value")

    def test_variance_ratio(self):
        vr = VarianceRatio(self.inflation, debiased=False)
        y = self.inflation
        dy = np.diff(y)
        mu = dy.mean()
        dy2 = y[2:] - y[:-2]
        nq = dy.shape[0]
        denom = np.sum((dy - mu) ** 2.0) / nq
        num = np.sum((dy2 - 2 * mu) ** 2.0) / (nq * 2)
        ratio = num / denom

        assert_almost_equal(ratio, vr.vr)
        assert "Variance-Ratio Test" in str(vr)
        vr = VarianceRatio(self.inflation, debiased=True)
        assert vr.debiased is True

    def test_variance_ratio_no_overlap(self):
        vr = VarianceRatio(self.inflation, overlap=False)

        with warnings.catch_warnings(record=True) as w:
            computed_value = vr.vr
            assert_equal(len(w), 1)

        y = self.inflation
        # Adjust due ot sample size
        y = y[:-1]
        dy = np.diff(y)
        mu = dy.mean()
        dy2 = y[2::2] - y[:-2:2]
        nq = dy.shape[0]
        denom = np.sum((dy - mu) ** 2.0) / nq
        num = np.sum((dy2 - 2 * mu) ** 2.0) / nq
        ratio = num / denom
        assert_allclose(ratio, computed_value)

        vr = VarianceRatio(self.inflation, overlap=True)
        assert_equal(vr.overlap, True)
        vr2 = VarianceRatio(self.inflation)
        assert_almost_equal(vr.stat, vr2.stat)

    def test_variance_ratio_non_robust(self):
        vr = VarianceRatio(self.inflation, robust=False, debiased=False)
        y = self.inflation
        dy = np.diff(y)
        mu = dy.mean()
        dy2 = y[2:] - y[:-2]
        nq = dy.shape[0]
        denom = np.sum((dy - mu) ** 2.0) / nq
        num = np.sum((dy2 - 2 * mu) ** 2.0) / (nq * 2)
        ratio = num / denom
        variance = 3.0 / 3.0
        stat = np.sqrt(nq) * (ratio - 1) / np.sqrt(variance)
        assert_almost_equal(stat, vr.stat)
        orig_stat = vr.stat
        vr = VarianceRatio(self.inflation, robust=True, debiased=False)
        assert_equal(vr.robust, True)
        assert vr.stat != orig_stat

    def test_variance_ratio_no_constant(self):
        y = self.rng.standard_normal(100)
        vr = VarianceRatio(y, trend="n", debiased=False)
        dy = np.diff(y)
        mu = 0.0
        dy2 = y[2:] - y[:-2]
        nq = dy.shape[0]
        denom = np.sum((dy - mu) ** 2.0) / nq
        num = np.sum((dy2 - 2 * mu) ** 2.0) / (nq * 2)
        ratio = num / denom
        assert_almost_equal(ratio, vr.vr)
        assert_equal(vr.debiased, False)

    def test_variance_ratio_invalid_lags(self):
        y = self.inflation
        with pytest.raises(ValueError):
            VarianceRatio(y, lags=1)

    def test_variance_ratio_generic(self):
        # TODO: Currently not a test, just makes sure code runs at all
        vr = VarianceRatio(self.inflation, lags=24)
        assert isinstance(vr, VarianceRatio)


class TestAutolagOLS:
    @classmethod
    def setup_class(cls):
        cls.rng = RandomState(12345)
        t = 1100
        y = np.zeros(t)
        e = cls.rng.standard_normal(t)
        y[:2] = e[:2]
        for i in range(3, t):
            y[i] = 1.5 * y[i - 1] - 0.8 * y[i - 2] + 0.2 * y[i - 3] + e[i]
        cls.y = y[100:]
        cls.x = cls.y.std() * cls.rng.randn(t, 2)
        cls.x = cls.x[100:]
        cls.z = cls.y + cls.x.sum(1)

        cls.cpi = log(dataset_loader(macrodata)["cpi"])
        cls.inflation = diff(cls.cpi)
        cls.inflation_change = diff(cls.inflation)

    def test_aic(self):
        exog, endog = lagmat(self.inflation, 12, original="sep", trim="both")
        _, sel_lag = _autolag(OLS, endog, exog, 1, 11, "aic")
        icbest2, sel_lag2 = _autolag_ols(endog, exog, 0, 12, "aic")
        assert np.isscalar(icbest2)
        assert np.isscalar(sel_lag2)
        assert sel_lag == sel_lag2

        exog, endog = lagmat(self.y, 12, original="sep", trim="both")
        _, sel_lag = _autolag(OLS, endog, exog, 1, 11, "aic")
        icbest2, sel_lag2 = _autolag_ols(endog, exog, 0, 12, "aic")
        assert np.isscalar(icbest2)
        assert np.isscalar(sel_lag2)
        assert sel_lag == sel_lag2

    def test_bic(self):
        exog, endog = lagmat(self.inflation, 12, original="sep", trim="both")
        _, sel_lag = _autolag(OLS, endog, exog, 1, 11, "bic")
        icbest2, sel_lag2 = _autolag_ols(endog, exog, 0, 12, "bic")
        assert np.isscalar(icbest2)
        assert np.isscalar(sel_lag2)
        assert sel_lag == sel_lag2

        exog, endog = lagmat(self.y, 12, original="sep", trim="both")
        _, sel_lag = _autolag(OLS, endog, exog, 1, 11, "bic")
        icbest2, sel_lag2 = _autolag_ols(endog, exog, 0, 12, "bic")
        assert np.isscalar(icbest2)
        assert np.isscalar(sel_lag2)
        assert sel_lag == sel_lag2

    def test_tstat(self):
        exog, endog = lagmat(self.inflation, 12, original="sep", trim="both")
        _, sel_lag = _autolag(OLS, endog, exog, 1, 11, "t-stat")
        icbest2, sel_lag2 = _autolag_ols(endog, exog, 0, 12, "t-stat")
        assert np.isscalar(icbest2)
        assert np.isscalar(sel_lag2)
        assert sel_lag == sel_lag2

        exog, endog = lagmat(self.y, 12, original="sep", trim="both")
        _, sel_lag = _autolag(OLS, endog, exog, 1, 11, "t-stat")
        icbest2, sel_lag2 = _autolag_ols(endog, exog, 0, 12, "t-stat")
        assert np.isscalar(icbest2)
        assert np.isscalar(sel_lag2)
        assert sel_lag == sel_lag2

    def test_aic_exogenous(self):
        exog, endog = lagmat(self.z, 12, original="sep", trim="both")
        exog = np.concatenate([self.x[12:], exog], axis=1)
        _, sel_lag = _autolag_ols(endog, exog, 2, 12, "aic")
        direct = np.zeros(exog.shape[1])
        direct.fill(np.inf)
        for i in range(3, exog.shape[1]):
            res = OLS(endog, exog[:, :i]).fit()
            direct[i] = res.aic
        assert np.argmin(direct[2:]) == sel_lag

    def test_bic_exogenous(self):
        exog, endog = lagmat(self.z, 12, original="sep", trim="both")
        exog = np.concatenate([self.x[12:], exog], axis=1)
        _, sel_lag = _autolag_ols(endog, exog, 2, 12, "bic")
        direct = np.zeros(exog.shape[1])
        direct.fill(np.inf)
        for i in range(3, exog.shape[1]):
            res = OLS(endog, exog[:, :i]).fit()
            direct[i] = res.bic
        assert np.argmin(direct[2:]) == sel_lag

    def test_tstat_exogenous(self):
        exog, endog = lagmat(self.z, 12, original="sep", trim="both")
        exog = np.concatenate([self.x[12:], exog], axis=1)
        _, sel_lag = _autolag_ols(endog, exog, 2, 12, "t-stat")
        direct = np.zeros(exog.shape[1])
        for i in range(3, exog.shape[1]):
            res = OLS(endog, exog[:, :i]).fit()
            direct[i] = res.tvalues[-1]
        crit = stats.norm.ppf(0.95)
        assert np.max(np.argwhere(np.abs(direct[2:]) > crit)) == sel_lag


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
def test_trends_low_memory(trend):
    rnd = np.random.RandomState(12345)
    y = np.cumsum(rnd.standard_normal(250))
    adf = ADF(y, trend=trend, max_lags=16)
    adf2 = ADF(y, trend=trend, low_memory=True, max_lags=16)
    assert adf.lags == adf2.lags
    assert adf.max_lags == 16
    adf = ADF(y, trend=trend, max_lags=1)
    assert_equal(adf.lags, 1)
    assert_equal(adf.max_lags, 1)


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
def test_representations(trend):
    rnd = np.random.RandomState(12345)
    y = np.cumsum(rnd.standard_normal(250))
    adf = ADF(y, trend=trend, max_lags=16)
    check = "Constant"
    if trend == "n":
        check = "No Trend"
    assert check in adf.__repr__()
    assert check in adf.__repr__()
    assert check in adf._repr_html_()
    assert 'class="simpletable"' in adf._repr_html_()


def test_unknown_method():
    rnd = np.random.RandomState(12345)
    y = np.cumsum(rnd.standard_normal(250))
    with pytest.raises(ValueError):
        assert np.isfinite(ADF(y, method="unknown").stat)


def test_auto_low_memory():
    rnd = np.random.RandomState(12345)
    y = np.cumsum(rnd.standard_normal(250))
    adf = ADF(y, trend="ct")
    assert adf._low_memory is False
    y = np.cumsum(rnd.standard_normal(1000000))
    adf = ADF(y, trend="ct")
    assert adf._low_memory is True


def test_mackinnonp_errors():
    with pytest.raises(ValueError):
        mackinnonp(-1.0, regression="c", num_unit_roots=2, dist_type="ADF-z")
    with pytest.raises(ValueError):
        mackinnonp(-1.0, dist_type="unknown")


def test_mackinnonp_small():
    val_large = mackinnonp(-7.0, regression="c", num_unit_roots=1, dist_type="adf-z")
    val = mackinnonp(-10.0, regression="c", num_unit_roots=1, dist_type="adf-z")
    assert val < val_large


def test_mackinnonp_large():
    val = mackinnonp(100.0, regression="c", num_unit_roots=1)
    assert val == 1.0


def test_mackinnoncrit_errors():
    with pytest.raises(ValueError):
        mackinnoncrit(regression="ttc")
    with pytest.raises(ValueError):
        mackinnoncrit(dist_type="unknown")
    cv_50 = mackinnoncrit(nobs=50)
    cv_inf = mackinnoncrit()
    assert np.all(cv_50 <= cv_inf)


def test_adf_buggy_timeseries1():
    x = np.asarray([0])
    adf = ADF(x)
    # ValueError: maxlag should be < nobs
    msg = "A minimum of 4 observations are needed"
    with pytest.raises(InfeasibleTestException, match=msg):
        assert np.isfinite(adf.stat)


def test_adf_buggy_timeseries2():
    x = np.asarray([0, 0])
    adf = ADF(x)
    # IndexError: index 0 is out of bounds for axis 0 with size 0
    msg = "A minimum of 4 observations are needed"
    with pytest.raises(InfeasibleTestException, match=msg):
        assert np.isfinite(adf.stat)


def test_adf_buggy_timeseries3():
    x = np.asarray([1] * 1000)
    adf = ADF(x)
    # AssertionError: Number of manager items must equal union of block items
    # # manager items: 1, # tot_items: 0
    with pytest.raises(InfeasibleTestException, match="The maximum lag you are"):
        assert np.isfinite(adf.stat)


def test_kpss_buggy_timeseries1():
    x = np.asarray([0])
    adf = KPSS(x, lags=0)
    # ValueError: cannot convert float NaN to integer
    with pytest.raises(InfeasibleTestException, match="A minimum of 2 observations"):
        assert np.isfinite(adf.stat)


kpss_autolag_data = (
    (dataset_loader(macrodata)["realgdp"], "c", 9),
    (dataset_loader(sunspots)["SUNACTIVITY"], "c", 7),
    (dataset_loader(nile)["volume"], "c", 5),
    (dataset_loader(randhie)["lncoins"], "ct", 75),
    (dataset_loader(modechoice)["invt"], "ct", 18),
)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("data,trend,lags", kpss_autolag_data)
def test_kpss_data_dependent_lags(data, trend, lags):
    # real GDP from macrodata data set
    kpss = KPSS(data, trend=trend)
    assert_equal(kpss.lags, lags)


class ZATestResult(NamedTuple):
    stat: float
    pvalue: float
    lags: Optional[int]
    trend: str
    max_lags: Optional[int]
    method: Optional[str]
    actual_lags: int


series = {
    "REAL_GNP": ZATestResult(
        stat=-5.57615,
        pvalue=0.00312,
        lags=8,
        trend="c",
        max_lags=None,
        method=None,
        actual_lags=8,
    ),
    "GNP_DEFLATOR": ZATestResult(
        stat=-4.12155,
        pvalue=0.28024,
        lags=None,
        trend="c",
        max_lags=8,
        method="t-stat",
        actual_lags=5,
    ),
    "STOCK_PRICES": ZATestResult(
        stat=-5.60689,
        pvalue=0.00894,
        lags=None,
        trend="ct",
        max_lags=8,
        method="t-stat",
        actual_lags=1,
    ),
    "REAL_GNP_QTR": ZATestResult(
        stat=-3.02761,
        pvalue=0.63993,
        lags=None,
        trend="t",
        max_lags=12,
        method="t-stat",
        actual_lags=12,
    ),
    "RAND10000": ZATestResult(
        stat=-3.48223,
        pvalue=0.69111,
        lags=None,
        trend="c",
        max_lags=None,
        method="t-stat",
        actual_lags=25,
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("series_name", series.keys())
def test_zivot_andrews(series_name):
    # Test results from package urca.ur.za (1.13-0)
    y = ZIVOT_ANDREWS_DATA[series_name].dropna()
    result = series[series_name]
    za = ZivotAndrews(
        y,
        lags=result.lags,
        trend=result.trend,
        max_lags=result.max_lags,
        method=result.method,
    )
    assert_almost_equal(za.stat, result.stat, decimal=3)
    assert_almost_equal(za.pvalue, result.pvalue, decimal=3)
    assert_equal(za.lags, result.actual_lags)
    assert isinstance(za.__repr__(), str)


def test_zivot_andrews_error():
    series_name = "REAL_GNP"
    y = ZIVOT_ANDREWS_DATA[series_name].dropna()
    with pytest.raises(ValueError):
        ZivotAndrews(y, trim=0.5)


def test_zivot_andrews_reduced_rank():
    y = np.random.standard_normal(1000)
    y[1:] = 3.0
    with pytest.raises(InfeasibleTestException, match="The regressor matrix is"):
        assert np.isfinite(ZivotAndrews(y, lags=1).stat)


def test_bw_selection():
    bw_ba = round(auto_bandwidth(REAL_TIME_SERIES, kernel="ba"), 7)
    assert_allclose(bw_ba, TRUE_BW_FROM_R_BA)

    bw_pa = round(auto_bandwidth(REAL_TIME_SERIES, kernel="pa"), 6)
    assert_allclose(bw_pa, TRUE_BW_FROM_R_PA)

    bw_qs = round(auto_bandwidth(REAL_TIME_SERIES, kernel="qs"), 6)
    assert_allclose(bw_qs, TRUE_BW_FROM_R_QS)

    with pytest.raises(ValueError):
        auto_bandwidth(REAL_TIME_SERIES, kernel="err")

    with pytest.raises(ValueError):
        auto_bandwidth([1])


def test_invalid_trend():
    with pytest.raises(ValueError, match="trend not understood"):
        ADF(np.random.standard_normal(100), trend="unknown")


@pytest.mark.filterwarnings("ignore:Lag selection has changed:DeprecationWarning")
@pytest.mark.parametrize("nobs", np.arange(1, 11).tolist())
@pytest.mark.parametrize("stat", [ADF, PhillipsPerron, KPSS, ZivotAndrews, DFGLS])
@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
def test_wrong_exceptions(stat, nobs, trend):
    skip = trend == "ctt" and stat in (PhillipsPerron, KPSS, ZivotAndrews, DFGLS)
    skip |= trend == "n" and stat in (KPSS, ZivotAndrews, DFGLS)
    if skip:
        return
    y = np.random.standard_normal((nobs,))
    try:
        assert np.isfinite(stat(y, trend=trend).stat)
    except InfeasibleTestException:
        pass


@pytest.mark.filterwarnings("ignore:Lag selection has changed:DeprecationWarning")
@pytest.mark.parametrize("nobs", [2, 10, 100])
@pytest.mark.parametrize("stat", [ADF, PhillipsPerron, KPSS, ZivotAndrews, DFGLS])
@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
def test_wrong_exceptions_nearly_constant_series(stat, nobs, trend):
    skip = trend == "ctt" and stat in (PhillipsPerron, KPSS, ZivotAndrews, DFGLS)
    skip |= trend == "n" and stat in (KPSS, ZivotAndrews, DFGLS)
    if skip:
        return
    y = np.zeros((nobs,))
    y[-1] = 1.0
    try:
        assert np.isfinite(stat(y, trend=trend).stat)
    except InfeasibleTestException:
        pass


def test_phillips_perron_specifed_lag():
    y = np.zeros((10,))
    with pytest.raises(InfeasibleTestException, match="A minimum of 12 observations"):
        assert np.isfinite(PhillipsPerron(y, lags=12).stat)


def test_kpss_legacy():
    y = np.random.standard_normal(4)
    with pytest.raises(InfeasibleTestException, match="The number of observations 4"):
        assert np.isfinite(KPSS(y, lags=-1).stat)


@pytest.mark.parametrize(
    "x", [np.ones((2, 10)), np.full((20, 2), np.nan), np.ones((20, 2))]
)
def test_rank_checker(x):
    assert _is_reduced_rank(x)


@pytest.mark.parametrize("nobs", list(range(1, 11)))
@pytest.mark.parametrize("trend", ["c", "ct", "t"])
def test_wrong_exceptions_nearly_constant_series_za_lags(nobs, trend):
    y = np.zeros((nobs,))
    y[-1] = 1.0
    try:
        assert np.isfinite(ZivotAndrews(y, lags=2, trend=trend).stat)
    except InfeasibleTestException:
        pass


@pytest.mark.filterwarnings("ignore::arch.utility.exceptions.InvalidLengthWarning")
@pytest.mark.parametrize("nobs", np.arange(1, 11).tolist())
@pytest.mark.parametrize("trend", ["n", "c"])
@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("debiased", [True, False])
def test_wrong_exceptions_variance_ratio(nobs, trend, overlap, debiased):
    y = np.random.standard_normal((nobs,))
    try:
        vr = VarianceRatio(y, trend=trend, lags=4, overlap=overlap, debiased=debiased)
        assert np.isfinite(vr.stat)
    except InfeasibleTestException:
        pass


def test_low_memory_singular():
    x = np.zeros(1000)
    x[:3] = np.random.standard_normal()
    x[-3:] = np.random.standard_normal()
    match = "The maximum lag you are"
    with pytest.raises(InfeasibleTestException, match=match):
        ADF(x, max_lags=10, low_memory=True).stat


@pytest.mark.parametrize("method", ["aic", "bic", "t-stat"])
@pytest.mark.parametrize("trend", ["c", "t", "ct", "ctt"])
def test_autolag_ols_low_memory_smoke(trend, method):
    data = dataset_loader(macrodata)
    realgdp = np.log(data["realgdp"])
    _autolag_ols_low_memory(realgdp, maxlag=4, trend=trend, method=method)


def test_autolag_warning():
    y = np.random.standard_normal(1000000)
    with pytest.warns(PerformanceWarning, match=""):
        assert isinstance(ADF(y).stat, float)
