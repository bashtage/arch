"""String options of the unit root and cointegration tests ignore case"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.unitroot import (
    ADF,
    DFGLS,
    KPSS,
    PhillipsPerron,
    VarianceRatio,
    ZivotAndrews,
    auto_bandwidth,
    engle_granger,
)
from arch.unitroot._engle_granger import engle_granger_cv, engle_granger_pval
from arch.unitroot._phillips_ouliaris import (
    phillips_ouliaris_cv,
    phillips_ouliaris_pval,
)
from arch.unitroot.cointegration import (
    CanonicalCointegratingReg,
    DynamicOLS,
    FullyModifiedOLS,
    phillips_ouliaris,
)
from arch.unitroot.unitroot import kpss_crit, mackinnoncrit, mackinnonp

# A string that is not an option is a ValueError and anything else a TypeError
BAD_VALUES = [
    pytest.param("bogus", ValueError, id="bogus"),
    pytest.param("", ValueError, id="empty"),
    pytest.param(None, TypeError, id="none"),
    pytest.param(1, TypeError, id="int"),
]


def variants(value):
    return [value.upper(), value.capitalize(), value.title()]


@pytest.fixture(scope="module")
def series():
    return pd.Series(np.cumsum(np.random.default_rng(5).standard_normal(300)))


@pytest.fixture(scope="module")
def cointegrated():
    rs = np.random.default_rng(6)
    x = pd.DataFrame(
        np.cumsum(rs.standard_normal((300, 2)), axis=0), columns=["a", "b"]
    )
    y = pd.Series(0.5 * x["a"] - 0.2 * x["b"] + rs.standard_normal(300), name="y")
    return y, x


def summary_of(test):
    return (test.stat, test.pvalue, test.lags, test.trend)


UNIT_ROOT_TRENDS = [
    (ADF, ["n", "c", "ct", "ctt"]),
    (DFGLS, ["c", "ct"]),
    (ZivotAndrews, ["c", "t", "ct"]),
    (PhillipsPerron, ["n", "c", "ct"]),
    (VarianceRatio, ["n", "c"]),
    (KPSS, ["c", "ct"]),
]
TREND_CASES = [(cls, trend) for cls, trends in UNIT_ROOT_TRENDS for trend in trends]
TREND_IDS = [f"{cls.__name__}-{trend}" for cls, trend in TREND_CASES]


def build(cls, series, **kwargs):
    if cls is KPSS:
        kwargs["lags"] = 4
    return cls(series, **kwargs)


@pytest.mark.parametrize(("cls", "trend"), TREND_CASES, ids=TREND_IDS)
def test_trend_case(series, cls, trend):
    expected = build(cls, series, trend=trend)
    for spelling in variants(trend):
        test = build(cls, series, trend=spelling)
        # The canonical lower case value is stored
        assert test.trend == trend
        assert_allclose(test.stat, expected.stat)
        assert_allclose(test.pvalue, expected.pvalue)


@pytest.mark.parametrize("cls", [c for c, _ in UNIT_ROOT_TRENDS])
@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_trend_invalid(series, cls, bad, exc):
    with pytest.raises(exc, match=r"trend must be one of '.*'; got "):
        build(cls, series, trend=bad)


def test_dfgls_trend_changes_statistic(series):
    # The DFGLS detrending constant depends on the trend, which was read
    # before the trend was converted to lower case
    lower = DFGLS(series, trend="ct")
    upper = DFGLS(series, trend="CT")
    assert_allclose(upper.stat, lower.stat)
    assert upper.stat != DFGLS(series, trend="c").stat


METHODS = ["aic", "bic", "t-stat", "maic"]


@pytest.mark.parametrize("cls", [ADF, DFGLS, ZivotAndrews])
@pytest.mark.parametrize("method", METHODS + ["t", "tstat"])
def test_lag_method_case(series, cls, method):
    expected = cls(series, method=method, max_lags=6)
    for spelling in variants(method):
        test = cls(series, method=spelling, max_lags=6)
        assert test.lags == expected.lags
        assert_allclose(test.stat, expected.stat)


@pytest.mark.parametrize("cls", [ADF, DFGLS, ZivotAndrews])
@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_lag_method_invalid(series, cls, bad, exc):
    match = r"method must be one of 'aic', 'bic', 't-stat' or 'maic'; got "
    with pytest.raises(exc, match=match):
        cls(series, method=bad)


@pytest.mark.parametrize("test_type", ["tau", "rho"])
def test_pp_test_type_case(series, test_type):
    expected = PhillipsPerron(series, test_type=test_type)
    for spelling in variants(test_type):
        test = PhillipsPerron(series, test_type=spelling)
        assert_allclose(test.stat, expected.stat)
        assert test.summary().as_text() == expected.summary().as_text()
    assert (
        expected.stat
        != PhillipsPerron(series, test_type="rho" if test_type == "tau" else "tau").stat
    )


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_pp_test_type_invalid(series, bad, exc):
    # Any other value used to be treated as "tau"
    with pytest.raises(exc, match=r"test_type must be one of 'tau' or 'rho'; got "):
        PhillipsPerron(series, test_type=bad)


@pytest.mark.parametrize("dist_type", ["adf-t", "adf-z", "dfgls"])
def test_mackinnon_dist_type_case(dist_type):
    expected_p = mackinnonp(-2.5, dist_type=dist_type)
    expected_cv = mackinnoncrit(dist_type=dist_type, nobs=100)
    for spelling in variants(dist_type):
        assert_allclose(mackinnonp(-2.5, dist_type=spelling), expected_p)
        assert_allclose(mackinnoncrit(dist_type=spelling, nobs=100), expected_cv)


@pytest.mark.parametrize("regression", ["c", "n", "ct", "ctt"])
def test_mackinnon_regression_case(regression):
    expected_p = mackinnonp(-2.5, regression=regression)
    expected_cv = mackinnoncrit(regression=regression, nobs=100)
    assert_allclose(mackinnonp(-2.5, regression=regression.upper()), expected_p)
    assert_allclose(mackinnoncrit(regression=regression.upper(), nobs=100), expected_cv)


@pytest.mark.parametrize("regression", ["n", "ctt"])
def test_mackinnonp_dfgls_regression(regression):
    # Was a KeyError
    with pytest.raises(ValueError, match=r"regression must be one of 'c' or 'ct'"):
        mackinnonp(-2.5, regression=regression, dist_type="dfgls")
    with pytest.raises(ValueError, match=r"regression must be one of 'c' or 'ct'"):
        mackinnoncrit(regression=regression, dist_type="dfgls")


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_mackinnon_invalid(bad, exc):
    for func in (mackinnonp, mackinnoncrit):
        with pytest.raises(exc, match=r"regression must be one of '.*'; got "):
            func(*((-2.5,) if func is mackinnonp else ()), regression=bad)
        with pytest.raises(
            exc, match=r"dist_type must be one of 'adf-t', 'adf-z' or 'dfgls'; got "
        ):
            func(*((-2.5,) if func is mackinnonp else ()), dist_type=bad)


@pytest.mark.parametrize("trend", ["c", "ct"])
def test_kpss_crit_case(trend):
    expected = kpss_crit(0.3, trend)
    for spelling in variants(trend):
        pvalue, cv = kpss_crit(0.3, spelling)
        assert_allclose(pvalue, expected[0])
        assert_allclose(cv, expected[1])


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_kpss_crit_invalid(bad, exc):
    with pytest.raises(exc, match=r"trend must be one of 'c' or 'ct'; got "):
        kpss_crit(0.3, bad)


@pytest.mark.parametrize(
    "kernel", ["ba", "bartlett", "nw", "pa", "parzen", "gallant", "qs", "andrews"]
)
def test_auto_bandwidth_kernel_case(series, kernel):
    data = series.diff().dropna()
    expected = auto_bandwidth(data, kernel=kernel)
    for spelling in variants(kernel):
        assert_allclose(auto_bandwidth(data, kernel=spelling), expected)


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_auto_bandwidth_invalid(series, bad, exc):
    with pytest.raises(exc, match=r"kernel must be one of 'ba', 'bartlett', .*; got "):
        auto_bandwidth(series.diff().dropna(), kernel=bad)


TRENDS4 = ["n", "c", "ct", "ctt"]


@pytest.mark.parametrize("trend", TRENDS4)
def test_engle_granger_trend_case(cointegrated, trend):
    y, x = cointegrated
    expected = engle_granger(y, x, trend=trend)
    for spelling in variants(trend):
        test = engle_granger(y, x, trend=spelling)
        assert test.trend == trend
        assert_allclose(test.stat, expected.stat)
        assert_allclose(test.pvalue, expected.pvalue)
        assert_allclose(test.critical_values, expected.critical_values)
        assert test.summary().as_text() == expected.summary().as_text()


@pytest.mark.parametrize("method", ["aic", "bic", "t-stat", "maic"])
def test_engle_granger_method_case(cointegrated, method):
    y, x = cointegrated
    expected = engle_granger(y, x, method=method, max_lags=4)
    test = engle_granger(y, x, method=method.upper(), max_lags=4)
    assert test.lags == expected.lags
    assert_allclose(test.stat, expected.stat)


@pytest.mark.parametrize("trend", TRENDS4)
def test_engle_granger_cv_pval_case(trend):
    assert_allclose(
        engle_granger_cv(trend.upper(), 2, 200), engle_granger_cv(trend, 2, 200)
    )
    assert_allclose(
        engle_granger_pval(-3.0, trend.upper(), 2), engle_granger_pval(-3.0, trend, 2)
    )


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_engle_granger_invalid(cointegrated, bad, exc):
    y, x = cointegrated
    match = r"trend must be one of 'n', 'c', 'ct' or 'ctt'; got "
    with pytest.raises(exc, match=match):
        engle_granger(y, x, trend=bad)
    with pytest.raises(exc, match=match):
        engle_granger_cv(bad, 2, 200)
    with pytest.raises(exc, match=match):
        engle_granger_pval(-3.0, bad, 2)
    with pytest.raises(exc, match=r"method must be one of 'aic'"):
        engle_granger(y, x, method=bad)


@pytest.mark.parametrize("test_type", ["Za", "Zt", "Pu", "Pz"])
def test_phillips_ouliaris_case(cointegrated, test_type):
    y, x = cointegrated
    expected = phillips_ouliaris(y, x, test_type=test_type)
    for spelling in (test_type.lower(), test_type.upper()):
        test = phillips_ouliaris(y, x, test_type=spelling)
        assert_allclose(test.stat, expected.stat)
        assert_allclose(test.pvalue, expected.pvalue)
        # The canonical spelling is kept in the name
        assert test.name == expected.name
        assert test_type in test.name


@pytest.mark.parametrize("trend", TRENDS4)
def test_phillips_ouliaris_trend_case(cointegrated, trend):
    y, x = cointegrated
    expected = phillips_ouliaris(y, x, trend=trend)
    test = phillips_ouliaris(y, x, trend=trend.upper())
    assert_allclose(test.stat, expected.stat)
    assert_allclose(test.pvalue, expected.pvalue)


@pytest.mark.parametrize("test_type", ["Za", "Zt", "Pu", "Pz"])
def test_phillips_ouliaris_cv_pval_case(test_type):
    assert_allclose(
        phillips_ouliaris_cv(test_type.lower(), "C", 3, 200),
        phillips_ouliaris_cv(test_type, "c", 3, 200),
    )
    assert_allclose(
        phillips_ouliaris_pval(-3.0, test_type.upper(), "CT", 3),
        phillips_ouliaris_pval(-3.0, test_type, "ct", 3),
    )


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_phillips_ouliaris_invalid(cointegrated, bad, exc):
    y, x = cointegrated
    tt = r"test_type must be one of 'Za', 'Zt', 'Pu' or 'Pz'; got "
    tr = r"trend must be one of 'n', 'c', 'ct' or 'ctt'; got "
    with pytest.raises(exc, match=tt):
        phillips_ouliaris(y, x, test_type=bad)
    with pytest.raises(exc, match=tr):
        phillips_ouliaris(y, x, trend=bad)
    with pytest.raises(exc, match=tt):
        phillips_ouliaris_cv(bad, "c", 3, 200)
    with pytest.raises(exc, match=tr):
        phillips_ouliaris_cv("Za", bad, 3, 200)
    with pytest.raises(exc, match=tt):
        phillips_ouliaris_pval(-3.0, bad, "c", 3)
    with pytest.raises(exc, match=tr):
        phillips_ouliaris_pval(-3.0, "Za", bad, 3)


@pytest.mark.parametrize("method", ["aic", "bic", "hqic"])
def test_dols_method_case(cointegrated, method):
    y, x = cointegrated
    expected = DynamicOLS(y, x, method=method, max_lag=4, max_lead=4).fit()
    for spelling in variants(method):
        res = DynamicOLS(y, x, method=spelling, max_lag=4, max_lead=4).fit()
        # An upper case method used to select the leads and lags with BIC
        assert (res.leads, res.lags) == (expected.leads, expected.lags)
        assert_allclose(res.params, expected.params)


def test_dols_method_changes_selection():
    rs = np.random.default_rng(2)
    x = pd.DataFrame(
        np.cumsum(rs.standard_normal((300, 2)), axis=0), columns=["a", "b"]
    )
    shocks = rs.standard_normal(300)
    errors = np.zeros(300)
    for t in range(1, 300):
        errors[t] = 0.6 * errors[t - 1] + shocks[t]
    y = pd.Series(0.5 * x["a"] - 0.2 * x["b"] + errors, name="y")
    selected = {
        method: DynamicOLS(y, x, method=method, max_lag=6, max_lead=6)._leads_and_lags()
        for method in ("aic", "bic")
    }
    # AIC and BIC select different orders in this sample, so that using BIC for
    # "AIC", as an upper case method used to, is visible
    assert selected["aic"] != selected["bic"]
    for method in ("AIC", "Aic"):
        mod = DynamicOLS(y, x, method=method, max_lag=6, max_lead=6)
        assert mod._leads_and_lags() == selected["aic"]


@pytest.mark.parametrize(
    "cov_type", ["unadjusted", "homoskedastic", "robust", "kernel"]
)
def test_dols_cov_type_case(cointegrated, cov_type):
    y, x = cointegrated
    mod = DynamicOLS(y, x, lags=1, leads=1)
    expected = mod.fit(cov_type=cov_type)
    for spelling in variants(cov_type):
        res = mod.fit(cov_type=spelling)
        assert res.cov_type == cov_type
        assert_allclose(res.cov, expected.cov)


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_dols_invalid(cointegrated, bad, exc):
    y, x = cointegrated
    with pytest.raises(
        exc, match=r"method must be one of 'aic', 'bic' or 'hqic'; got "
    ):
        DynamicOLS(y, x, method=bad)
    with pytest.raises(exc, match=r"cov_type must be one of 'unadjusted', "):
        DynamicOLS(y, x, lags=1, leads=1).fit(cov_type=bad)
    with pytest.raises(exc, match=r"kernel must be one of "):
        DynamicOLS(y, x, lags=1, leads=1).fit(kernel=bad)


@pytest.mark.parametrize("kernel", ["Bartlett", "Parzen", "QuadraticSpectral"])
@pytest.mark.parametrize("estimator", [FullyModifiedOLS, CanonicalCointegratingReg])
def test_kernel_case(cointegrated, estimator, kernel):
    y, x = cointegrated
    expected = estimator(y, x).fit(kernel=kernel)
    for spelling in (kernel.lower(), kernel.upper()):
        res = estimator(y, x).fit(kernel=spelling)
        assert_allclose(res.params, expected.params)
        assert res.kernel == expected.kernel


@pytest.mark.parametrize("estimator", [FullyModifiedOLS, CanonicalCointegratingReg])
@pytest.mark.parametrize("x_trend", ["n", "c", "ct", "ctt"])
def test_x_trend_case(cointegrated, estimator, x_trend):
    y, x = cointegrated
    trend = "n" if x_trend == "n" else "c"
    expected = estimator(y, x, trend=trend, x_trend=x_trend).fit()
    res = estimator(y, x, trend=trend.upper(), x_trend=x_trend.upper()).fit()
    assert_allclose(res.params, expected.params)


@pytest.mark.parametrize("estimator", [FullyModifiedOLS, CanonicalCointegratingReg])
@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_fmols_ccr_invalid(cointegrated, estimator, bad, exc):
    y, x = cointegrated
    with pytest.raises(exc, match=r"trend must be one of "):
        estimator(y, x, trend=bad)
    with pytest.raises(exc, match=r"kernel must be one of "):
        estimator(y, x).fit(kernel=bad)
    if bad is not None:
        with pytest.raises(exc, match=r"trend must be one of "):
            estimator(y, x, x_trend=bad).fit()
