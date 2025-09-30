from arch.compat.matplotlib import HAS_MATPLOTLIB

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.iolib.summary import Summary

from arch.unitroot._engle_granger import engle_granger_cv, engle_granger_pval
from arch.unitroot._shared import ResidualCointegrationTestResult, _cross_section
from arch.unitroot.cointegration import engle_granger


@pytest.fixture(scope="module", params=[True, False])
def data(request):
    use_pandas = request.param
    rs = np.random.RandomState([392032, 302931, 20913210, 1023910])
    e = rs.standard_normal((500, 3))
    xyz = e.cumsum(0)
    data = pd.DataFrame(xyz, columns=["x", "y", "z"])
    e = rs.standard_normal(500)
    elag = e.copy()
    elag[1:] = e[:-1]
    elag[0] = rs.standard_normal()
    data["w"] = 2 * data.y + e + 0.5 * elag
    data.index = pd.date_range("1980-1-1", periods=data.shape[0], freq="MS")
    if not use_pandas:
        return np.asarray(data)
    return data


NULL_TEST_VALUES = {
    ("n", "x"): [-2.737185, 0.05355030199278942, -0.02963672702057441],
    ("n", "y"): [-2.354023, 0.1244587477290734, -0.02222826106845954],
    ("c", "x"): [-2.734094, 0.1884300320178474, -0.029484],
    ("c", "y"): [-2.245916, 0.3999527607346568, -0.017121],
    ("ct", "x"): [-2.953992475678419, 0.2882033700258219, -0.03476973409119411],
    ("ct", "y"): [-2.118891881009684, 0.7200656040655355, -0.02174058424648551],
    ("ctt", "x"): [-2.876051, 0.5316, -0.033659],
    ("ctt", "y"): [-3.016448, 0.4538, -0.035873],
}


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("method", ["aic", "bic"])
def test_bivariate_eg_null(data, trend, method):
    if isinstance(data, pd.DataFrame):
        y, x = data.y, data.x
    else:
        x = data[:, 0]
        y = data[:, 1]
    test_yx = engle_granger(y, x, trend=trend, method=method)
    key = (trend, "y")
    assert_allclose(test_yx.stat, NULL_TEST_VALUES[key][0], rtol=1e-4)
    assert_allclose(test_yx.rho, 1 + NULL_TEST_VALUES[key][2], rtol=1e-4)
    assert_allclose(test_yx.pvalue, NULL_TEST_VALUES[key][1], rtol=1e-2)

    test_xy = engle_granger(x, y, trend=trend, method=method)
    key = (trend, "x")
    assert_allclose(test_xy.stat, NULL_TEST_VALUES[key][0], rtol=1e-4)
    assert_allclose(test_xy.rho, 1 + NULL_TEST_VALUES[key][2], rtol=1e-4)
    assert_allclose(test_xy.pvalue, NULL_TEST_VALUES[key][1], rtol=1e-2)
    s = str(test_xy)
    assert "Engle-Granger" in s
    assert "Statistic:" in s
    assert "ID:" not in s
    r = test_yx.__repr__()
    assert "ID:" in r
    assert test_yx.null_hypothesis == "No Cointegration"
    assert test_yx.alternative_hypothesis == "Cointegration"
    assert 1 in test_yx.critical_values
    assert 5 in test_yx.critical_values
    assert 10 in test_yx.critical_values


ALT_TEST_VALUES = {
    ("aic", "n", "w"): [-7.817307, 0.00, -0.615927, 5],
    ("aic", "n", "y"): [-7.802172, 0.00, -0.613858, 5],
    ("bic", "n", "w"): [-14.29805, 0.0001, -0.696661, 1],
    ("bic", "n", "y"): [-14.28754, 0.0001, -0.695925, 1],
    ("aic", "c", "w"): [-9.329104, 0.00, -0.692495, 4],
    ("aic", "c", "y"): [-7.966146, 0.00, -0.636725, 5],
    ("bic", "c", "w"): [-14.47295, 0.0000, -0.708606, 1],
    ("bic", "c", "y"): [-14.41683, 0.0000, -0.704814, 1],
    ("aic", "ct", "w"): [-9.343354, 0.0000, -0.694055, 4],
    ("aic", "ct", "y"): [-7.900950, 0.0000, -0.628138, 5],
    ("bic", "ct", "w"): [-14.47904, 0.0000, -0.709048, 1],
    ("bic", "ct", "y"): [-14.37048, 0.0000, -0.701621, 1],
    ("aic", "ctt", "w"): [-9.416100, 0.0000, -0.703868, 4],
    ("aic", "ctt", "y"): [-9.057349, 0.0000, -0.659752, 4],
    ("bic", "ctt", "w"): [-14.55772, 0.0001, -0.714249, 1],
    ("bic", "ctt", "y"): [-14.30506, 0.0001, -0.696309, 1],
}


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("method", ["aic", "bic"])
def test_bivariate_eg_alternative(data, trend, method):
    if isinstance(data, pd.DataFrame):
        y, w = data.y, data.w
    else:
        w = data[:, -1]
        y = data[:, 1]

    test_yw = engle_granger(y, w, trend=trend, method=method, max_lags=17)
    key = (method, trend, "y")
    assert_allclose(test_yw.stat, ALT_TEST_VALUES[key][0], rtol=1e-4)
    assert_allclose(test_yw.rho, 1 + ALT_TEST_VALUES[key][2], rtol=1e-4)
    assert_allclose(test_yw.pvalue, ALT_TEST_VALUES[key][1], atol=1e-3)
    assert test_yw.lags == ALT_TEST_VALUES[key][3]
    assert test_yw.max_lags == 17

    test_wy = engle_granger(w, y, trend=trend, method=method, max_lags=17)
    key = (method, trend, "w")
    assert_allclose(test_wy.stat, ALT_TEST_VALUES[key][0], rtol=1e-4)
    assert_allclose(test_wy.rho, 1 + ALT_TEST_VALUES[key][2], rtol=1e-4)
    assert_allclose(test_wy.pvalue, ALT_TEST_VALUES[key][1], atol=1e-3)
    assert test_wy.lags == ALT_TEST_VALUES[key][3]


TRIVARIATE = {
    ("x", "c"): [-3.181996, 0.1774, -0.040199, 0],
    ("y", "c"): [-2.873134, 0.2988, -0.036931, 0],
    ("z", "c"): [-2.276025, 0.6023, -0.025554, 0],
    ("x", "ct"): [-3.18879, 0.3374, -0.040343, 0],
    ("y", "ct"): [-3.009687, 0.4285, -0.039642, 0],
    ("z", "ct"): [-3.1807, 0.3414, -0.039713, 0],
    ("x", "n"): [-3.13981, 0.0747, -0.039509, 0],
    ("y", "n"): [-2.929378, 0.1185, -0.038409, 0],
    ("z", "n"): [-2.289225, 0.3551, -0.026503, 0],
    ("x", "ctt"): [-3.183624, 0.5258, -0.041258, 0],
    ("y", "ctt"): [-3.181277, 0.5271, -0.040282, 0],
    ("z", "ctt"): [-3.161019, 0.5382, -0.039849, 0],
}


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("method", ["aic", "bic"])
@pytest.mark.parametrize("lhs", ["x", "y", "z"])
def test_trivariate(data, trend, method, lhs, agg_backend):
    rhs = ["x", "y", "z"]
    if isinstance(data, pd.DataFrame):
        rhs.remove(lhs)
        dep, exog = data[lhs], data[rhs]
    else:
        dep_loc = rhs.index(lhs)
        exog_locs = [0, 1, 2]
        exog_locs.remove(dep_loc)
        dep = data[:, dep_loc]
        exog = data[:, exog_locs]

    test = engle_granger(dep, exog, trend=trend, method=method)
    key = (lhs, trend)
    assert_allclose(test.stat, TRIVARIATE[key][0], rtol=1e-4)
    assert_allclose(test.rho, 1 + TRIVARIATE[key][2], rtol=1e-4)
    assert_allclose(test.pvalue, TRIVARIATE[key][1], rtol=2e-2)
    assert test.lags == 0
    assert isinstance(test.summary(), Summary)
    assert isinstance(test._repr_html_(), str)

    ci = test.cointegrating_vector
    assert isinstance(ci, pd.Series)

    if HAS_MATPLOTLIB:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig = test.plot()
        assert isinstance(fig, plt.Figure)

        plt.close("all")
    assert isinstance(test.resid, pd.Series)
    assert test.resid.shape[0] == dep.shape[0]


def test_exceptions_pvals():
    with pytest.raises(ValueError, match=r"Trend must by one"):
        engle_granger_cv("unknown", 2, 500)
    with pytest.raises(ValueError, match=r"The number of cross-sectional"):
        engle_granger_cv("n", 25, 500)


def test_exceptions_critvals():
    with pytest.raises(ValueError, match=r"Trend must by one"):
        engle_granger_pval(-3.0, "unknown", 500)
    with pytest.raises(ValueError, match=r"The number of cross-sectional"):
        engle_granger_pval(-3.0, "n", 500)


def test_pval_max_min():
    assert engle_granger_pval(300.0, "n", 2) == 1.0
    assert engle_granger_pval(-300.0, "n", 2) == 0.0


def test_exceptions(data):
    if isinstance(data, pd.DataFrame):
        y, x = data.y, data.x
    else:
        y, x = data[:, :2].T
    with pytest.raises(ValueError, match=r"Unknown trend. Must be one of"):
        engle_granger(y, x, trend="nc")


def test_name_ci_vector(data):
    if not isinstance(data, pd.DataFrame):
        return
    eg = engle_granger(data.w, data[["x", "y", "z"]])
    ci = eg.cointegrating_vector
    assert list(ci.index) == ["w", "x", "y", "z", "const"]


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
def test_plot(data, agg_backend):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rhs = ["x", "y", "z"]
    lhs = "y"
    if isinstance(data, pd.DataFrame):
        rhs.remove(lhs)
        dep, exog = data[lhs], data[rhs]
    else:
        dep_loc = rhs.index(lhs)
        exog_locs = [0, 1, 2]
        exog_locs.remove(dep_loc)
        dep = data[:, dep_loc]
        exog = data[:, exog_locs]
    test = engle_granger(dep, exog)
    assert isinstance(test.plot(), plt.Figure)


def test_cross_section_exceptions():
    y = np.random.standard_normal(1000)
    x = np.random.standard_normal((1000, 2))
    with pytest.raises(ValueError, match=r"trend must be one of "):
        _cross_section(y, x, "unknown")


def test_base_summary():
    cv = pd.Series([1.0, 2, 3], index=[1, 5, 10])
    y = np.random.standard_normal(1000)
    x = np.random.standard_normal((1000, 2))
    xsection = _cross_section(y, x, "ct")
    res = ResidualCointegrationTestResult(1.0, 0.05, cv, xsection=xsection)
    assert isinstance(res.summary(), Summary)
