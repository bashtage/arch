from typing import NamedTuple

from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.iolib.summary import Summary

from arch.unitroot.cointegration import CanonicalCointegratingReg, FullyModifiedOLS


@pytest.mark.parametrize("trend", ["n", "ct", "ctt"])
@pytest.mark.parametrize("x_trend", [None, "c", "ct", "ctt"])
@pytest.mark.parametrize("diff", [True, False])
@pytest.mark.parametrize("kernel", ["bartlett", "gallant", "andrews"])
@pytest.mark.parametrize("bandwidth", [1, 10, None])
@pytest.mark.parametrize("force_int", [True, False])
def test_fmols_smoke(
    trivariate_data, trend, x_trend, diff, kernel, bandwidth, force_int
):
    y, x = trivariate_data
    if x_trend is not None and len(x_trend) < len(trend):
        x_trend = trend
    mod = FullyModifiedOLS(y, x, trend, x_trend)
    res = mod.fit(kernel, bandwidth, force_int, diff)
    assert isinstance(res.summary(), Summary)


@pytest.mark.parametrize("trend", ["n", "ct", "ctt"])
@pytest.mark.parametrize("x_trend", [None, "c", "ct", "ctt"])
@pytest.mark.parametrize("diff", [True, False])
@pytest.mark.parametrize("kernel", ["bartlett", "gallant", "andrews"])
@pytest.mark.parametrize("bandwidth", [1, 10, None])
@pytest.mark.parametrize("force_int", [True, False])
def test_ccr_smoke(trivariate_data, trend, x_trend, diff, kernel, bandwidth, force_int):
    y, x = trivariate_data
    if x_trend is not None and len(x_trend) < len(trend):
        x_trend = trend
    mod = CanonicalCointegratingReg(y, x, trend, x_trend)
    res = mod.fit(kernel, bandwidth, force_int, diff)
    assert isinstance(res.summary(), Summary)


@pytest.mark.parametrize("estimator", [CanonicalCointegratingReg, FullyModifiedOLS])
def test_exceptions(trivariate_data, estimator):
    y, x = trivariate_data
    with pytest.raises(
        ValueError, match="The number of observations in y and x differ"
    ):
        estimator(y[1:], x)
    mod = estimator(y, x)
    with pytest.raises(ValueError, match="kernel is not a known kernel estimator"):
        mod.fit(kernel="unknown")


FMOLS_RES = {
    ("n", "n", False): {
        "Y2": [-4.71837422166802, 0.0925862508285919, -50.961932030311],
        "Y3": [1.14019246045587, 0.0165662901980161, 68.8260586303394],
        "R2": [0.984879, 0.984864],
        "VARIANCE": [14.81346, 1417.654],
    },
    ("c", "c", False): {
        "Y2": [-11.72714, 0.18051, -64.96664],
        "Y3": [2.313736, 0.030284, 76.40188],
        "C": [92.66459, 2.369368, 39.10942],
        "R2": [0.993952, 0.99394],
        "VARIANCE": [9.373602, 140.0409],
    },
    ("c", "c", True): {
        "Y2": [-11.72714, 0.18051, -64.96664],
        "Y3": [2.313736, 0.030284, 76.40188],
        "C": [92.66459, 2.369368, 39.10942],
        "R2": [0.993952, 0.99394],
        "VARIANCE": [9.373602, 140.0409],
    },
    ("c", "ct", False): {
        "Y2": [-11.72416, 0.180386, -64.99479],
        "Y3": [2.313268, 0.030263, 76.43893],
        "C": [92.50686, 2.36774, 39.06969],
        "R2": [0.993956, 0.993944],
        "VARIANCE": [9.370457, 139.8486],
    },
    ("c", "ct", True): {
        "Y2": [-11.72399, 0.180377, -64.99724],
        "Y3": [2.313243, 0.030261, 76.44207],
        "C": [92.4476, 2.367617, 39.04669],
        "R2": [0.993956, 0.993944],
        "VARIANCE": [9.370195, 139.834],
    },
    ("c", "ctt", False): {
        "Y2": [-11.7438, 0.181134, -64.83482],
        "Y3": [2.316309, 0.030388, 76.22337],
        "C": [92.7662, 2.377558, 39.01743],
        "R2": [0.993941, 0.993929],
        "VARIANCE": [9.381736, 141.0108],
    },
    ("c", "ctt", True): {
        "Y2": [-11.74995, 0.181467, -64.74992],
        "Y3": [2.317245, 0.030444, 76.11439],
        "C": [92.80805, 2.381924, 38.96349],
        "R2": [0.993937, 0.993924],
        "VARIANCE": [9.385349, 141.5291],
    },
    ("ct", "ct", False): {
        "Y2": [-11.11086, 0.28259, -39.31791],
        "Y3": [2.215921, 0.045542, 48.6566],
        "C": [87.88988, 2.785441, 31.55331],
        "TREND": [-0.007352, 0.003787, -1.9416],
        "R2": [0.994426, 0.994409],
        "VARIANCE": [9.003393, 175.426],
    },
    ("ct", "ct", True): {
        "Y2": [-11.11026, 0.282556, -39.32051],
        "Y3": [2.215828, 0.045537, 48.66044],
        "C": [87.79994, 2.785105, 31.52483],
        "TREND": [-0.007358, 0.003786, -1.943444],
        "R2": [0.994426, 0.99441],
        "VARIANCE": [9.002708, 175.3836],
    },
    ("ct", "ctt", False): {
        "Y2": [-11.10796, 0.282445, -39.32781],
        "Y3": [2.215473, 0.045519, 48.67175],
        "C": [88.09649, 2.784012, 31.64372],
        "TREND": [-0.007841, 0.003785, -2.071791],
        "R2": [0.99443, 0.994413],
        "VARIANCE": [9.000175, 175.246],
    },
    ("ct", "ctt", True): {
        "Y2": [-11.10728, 0.282406, -39.33092],
        "Y3": [2.215371, 0.045512, 48.67632],
        "C": [88.10724, 2.783622, 31.65201],
        "TREND": [-0.008056, 0.003784, -2.128764],
        "R2": [0.994431, 0.994414],
        "VARIANCE": [8.99922, 175.197],
    },
    ("ctt", "ctt", False): {
        "Y2": [-11.08107, 0.280433, -39.51418],
        "Y3": [2.212796, 0.045144, 49.01611],
        "C": [86.76076, 2.85819, 30.35514],
        "TREND": [0.001084, 0.006209, 0.174662],
        "QUAD": [-0.0000109, 0.00000608, -1.783747],
        "R2": [0.994498, 0.994476],
        "VARIANCE": [8.949234, 172.0974],
    },
    ("ctt", "ctt", True): {
        "Y2": [-11.08068, 0.28041, -39.51603],
        "Y3": [2.212738, 0.045141, 49.01881],
        "C": [86.77389, 2.857956, 30.36221],
        "TREND": [0.000892, 0.006209, 0.14375],
        "QUAD": [-0.0000108, 0.00000608, -1.781621],
        "R2": [0.994499, 0.994477],
        "VARIANCE": [8.948617, 172.0693],
    },
}


class EviewsTestResult(NamedTuple):
    params: pd.Series
    se: pd.Series
    tstats: pd.Series
    rsquared: float
    rsquared_adj: float
    short_run: float
    long_run: float


def setup_test_values(d: dict):
    d = d.copy()
    r2 = d.pop("R2")
    variance = d.pop("VARIANCE")
    results = pd.DataFrame(d, index=["params", "se", "tstats"]).T
    return EviewsTestResult(
        results.params,
        results.se,
        results.tstats,
        r2[0],
        r2[1],
        variance[0] ** 2,
        variance[1],
    )


FMOLS_RES_IDS = [f"tr: {v[0]}, xtr: {v[1]}, diff:{v[2]}" for v in FMOLS_RES.keys()]


@pytest.mark.parametrize("test_key", list(FMOLS_RES.keys()), ids=FMOLS_RES_IDS)
def test_fmols_eviews(trivariate_data, test_key):
    y, x = trivariate_data
    trend, x_trend, diff = test_key
    key = (trend, x_trend, diff)
    test_res = setup_test_values(FMOLS_RES[key])
    mod = FullyModifiedOLS(y, x, trend, x_trend)
    # BW is one less than what Eviews reports
    res = mod.fit(bandwidth=6, force_int=True, diff=diff, df_adjust=True)
    if trend != "ctt":
        assert_allclose(res.params, test_res.params, rtol=1e-4)
        assert_allclose(res.std_errors, test_res.se, rtol=1e-3)
        assert_allclose(res.tvalues, test_res.tstats, rtol=1e-3)
    else:
        # Do not match results when ctt since Eviews adds the trend and trend**2
        # after shortening the series, so that the trend for the second observations
        # is 1, while we use the trend fo rthe second observation as 2 (and tt as 4)
        assert_allclose(res.params.iloc[:3], test_res.params.iloc[:3], rtol=1e-4)
        assert_allclose(res.std_errors.iloc[:3], test_res.se.iloc[:3], rtol=1e-3)
        assert_allclose(res.tvalues.iloc[:3], test_res.tstats.iloc[:3], rtol=1e-3)
    if trend != "n":
        # Eviews seems to center even with no trend
        assert_allclose(res.rsquared, test_res.rsquared, rtol=1e-4)
        assert_allclose(res.rsquared_adj, test_res.rsquared_adj, rtol=1e-4)
    # Loose tolerance since I use 1000 obs while Eviews drops 1
    assert_allclose(res.residual_variance, test_res.short_run, rtol=2e-3)
    assert_allclose(res.long_run_variance, test_res.long_run, rtol=1e-4)

    assert isinstance(res.summary(), Summary)


CCR_RES = {
    ("n", "n", False): {
        "Y2": [-4.386332, 0.092346, -47.499],
        "Y3": [1.091409, 0.016501, 66.14003],
        "R2": [0.982615, 0.982597],
        "VARIANCE": [15.88413, 1417.654],
    },
    ("c", "c", False): {
        "Y2": [-11.67466, 0.177017, -65.95228],
        "Y3": [2.304937, 0.029697, 77.6161],
        "C": [91.99825, 2.326175, 39.54914],
        "R2": [0.993993, 0.993981],
        "VARIANCE": [9.341236, 140.0409],
    },
    ("c", "c", True): {
        "Y2": [-11.67466, 0.177017, -65.95228],
        "Y3": [2.304937, 0.029697, 77.6161],
        "C": [91.99825, 2.326175, 39.54914],
        "R2": [0.993993, 0.993981],
        "VARIANCE": [9.341236, 140.0409],
    },
    ("c", "ct", False): {
        "Y2": [-11.67128, 0.176879, -65.9847],
        "Y3": [2.304396, 0.029674, 77.65796],
        "C": [91.82754, 2.323642, 39.51879],
        "R2": [0.993998, 0.993986],
        "VARIANCE": [9.337652, 139.8486],
    },
    ("c", "ct", True): {
        "Y2": [-11.67119, 0.176878, -65.98442],
        "Y3": [2.304384, 0.029673, 77.65799],
        "C": [91.7642, 2.32307, 39.50127],
        "R2": [0.993998, 0.993986],
        "VARIANCE": [9.337402, 139.834],
    },
    ("c", "ctt", False): {
        "Y2": [-11.69285, 0.177785, -65.76976],
        "Y3": [2.307762, 0.029826, 77.37343],
        "C": [92.11232, 2.335479, 39.44043],
        "R2": [0.993982, 0.99397],
        "VARIANCE": [9.3497, 141.0108],
    },
    ("c", "ctt", True): {
        "Y2": [-11.70232, 0.178347, -65.61549],
        "Y3": [2.309247, 0.02992, 77.18072],
        "C": [92.19184, 2.342172, 39.36169],
        "R2": [0.993975, 0.993963],
        "VARIANCE": [9.355204, 141.5291],
    },
    ("ct", "ct", False): {
        "Y2": [-11.11296, 0.282904, -39.2818],
        "Y3": [2.21602, 0.045559, 48.64069],
        "C": [87.83715, 2.777905, 31.61992],
        "TREND": [-0.00718, 0.003799, -1.890093],
        "R2": [0.994426, 0.994409],
        "VARIANCE": [9.003305, 175.426],
    },
    ("ct", "ct", True): {
        "Y2": [-11.11269, 0.282879, -39.28424],
        "Y3": [2.215977, 0.045555, 48.64402],
        "C": [87.76041, 2.778078, 31.59033],
        "TREND": [-0.007181, 0.003798, -1.890627],
        "R2": [0.994426, 0.994409],
        "VARIANCE": [9.002874, 175.3836],
    },
    ("ct", "ctt", False): {
        "Y2": [-11.11157, 0.282798, -39.2916],
        "Y3": [2.215801, 0.045542, 48.65429],
        "C": [88.01424, 2.775084, 31.71589],
        "TREND": [-0.007569, 0.003814, -1.98461],
        "R2": [0.994428, 0.994411],
        "VARIANCE": [9.001295, 175.246],
    },
    ("ct", "ctt", True): {
        "Y2": [-11.11138, 0.282772, -39.29449],
        "Y3": [2.215772, 0.045538, 48.65816],
        "C": [88.03172, 2.775329, 31.71938],
        "TREND": [-0.00776, 0.003818, -2.03218],
        "R2": [0.994429, 0.994412],
        "VARIANCE": [9.000717, 175.197],
    },
    ("ctt", "ctt", False): {
        "Y2": [-11.08231, 0.280624, -39.49173],
        "Y3": [2.21279, 0.045147, 49.01355],
        "C": [86.6552, 2.849482, 30.41086],
        "TREND": [0.00143, 0.006228, 0.229653],
        "QUAD": [-0.000011, 0.00000605, -1.818512],
        "R2": [0.994498, 0.994476],
        "VARIANCE": [8.949107, 172.0974],
    },
    ("ctt", "ctt", True): {
        "Y2": [-11.08228, 0.280611, -39.49339],
        "Y3": [2.212786, 0.045145, 49.0155],
        "C": [86.67368, 2.849716, 30.41485],
        "TREND": [0.001265, 0.006229, 0.203093],
        "QUAD": [-0.000011, 0.00000605, -1.817444],
        "R2": [0.994499, 0.994476],
        "VARIANCE": [8.948774, 172.0693],
    },
}


@pytest.mark.parametrize("test_key", list(FMOLS_RES.keys()), ids=FMOLS_RES_IDS)
def test_ccr_eviews(trivariate_data, test_key):
    y, x = trivariate_data
    trend, x_trend, diff = test_key
    key = (trend, x_trend, diff)
    test_res = setup_test_values(CCR_RES[key])
    mod = CanonicalCointegratingReg(y, x, trend, x_trend)
    # BW is one less than what Eviews reports
    res = mod.fit(bandwidth=6, force_int=True, diff=diff, df_adjust=True)
    if trend == "n":
        # TODO: Determine reason for difference here
        return
    if trend != "ctt":
        assert_allclose(res.params, test_res.params, rtol=1e-4)
        assert_allclose(res.std_errors, test_res.se, rtol=1e-3)
        assert_allclose(res.tvalues, test_res.tstats, rtol=1e-3)
    else:
        # Do not match results when ctt since Eviews adds the trend and trend**2
        # after shortening the series, so that the trend for the second observations
        # is 1, while we use the trend fo rthe second observation as 2 (and tt as 4)
        assert_allclose(res.params.iloc[:3], test_res.params.iloc[:3], rtol=1e-4)
        assert_allclose(res.std_errors.iloc[:3], test_res.se.iloc[:3], rtol=1e-3)
        assert_allclose(res.tvalues.iloc[:3], test_res.tstats.iloc[:3], rtol=1e-3)
        # Eviews seems to center even with no trend
    assert_allclose(res.rsquared, test_res.rsquared, rtol=1e-4)
    assert_allclose(res.rsquared_adj, test_res.rsquared_adj, rtol=1e-4)
    # Loose tolerance since I use 1000 obs while Eviews drops 1
    assert_allclose(res.residual_variance, test_res.short_run, rtol=2e-3)
    assert_allclose(res.long_run_variance, test_res.long_run, rtol=1e-3)

    assert isinstance(res.summary(), Summary)
