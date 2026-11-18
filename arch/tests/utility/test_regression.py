import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS

from arch.utility._regression import centered_ols
from arch.utility.timeseries import add_trend

NOBS = 300


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20261118)
    x = pd.DataFrame(
        rng.standard_normal((NOBS, 3)) + [5.0, -2.0, 100.0], columns=["a", "b", "c"]
    )
    y = pd.Series(x @ [0.5, -1.0, 0.01] + rng.standard_normal(NOBS), name="y")
    return y, x


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize(
    "fit_kwargs",
    [{}, {"cov_type": "HC0"}, {"cov_type": "HAC", "cov_kwds": {"maxlags": 4}}],
    ids=["nonrobust", "HC0", "HAC"],
)
def test_centered_ols_matches_ols(data, trend, fit_kwargs):
    y, x = data
    rhs = x if trend == "n" else add_trend(x, trend)
    expected = OLS(y, rhs).fit(**fit_kwargs)
    res = centered_ols(y, rhs, **fit_kwargs)
    assert list(res.params.index) == list(rhs.columns)
    assert res.cov_type == expected.cov_type
    for attr in ("params", "bse", "tvalues", "resid", "fittedvalues"):
        assert_allclose(
            getattr(res, attr), getattr(expected, attr), rtol=1e-8, atol=1e-12
        )
    assert_allclose(res.cov_params(), expected.cov_params(), rtol=1e-8, atol=1e-14)
    for attr in ("rsquared", "rsquared_adj", "ssr", "llf", "condition_number"):
        assert_allclose(getattr(res, attr), getattr(expected, attr), rtol=1e-10)
    for attr in ("nobs", "df_model", "df_resid"):
        assert getattr(res, attr) == getattr(expected, attr)
    if trend != "n":
        assert_allclose(res.fvalue, expected.fvalue, rtol=1e-10)
    assert int(res.model.rank) == rhs.shape[1]
    res.summary()


def test_centered_ols_without_other_columns(data):
    y, _ = data
    const = pd.DataFrame({"const": np.full(NOBS, 2.0)}, index=y.index)
    res = centered_ols(y, const)
    assert_allclose(res.params, OLS(y, const).fit().params, rtol=1e-14)


def test_centered_ols_constant_not_one(data):
    # The constant can have any value, and its coefficient is that of the regression
    y, x = data
    rhs = x.copy()
    rhs["two"] = 2.0
    assert_allclose(
        centered_ols(y, rhs).params, OLS(y, rhs).fit().params, rtol=1e-8, atol=1e-12
    )


def test_centered_ols_level_with_tiny_variation():
    # The lagged level of a series that varies little around 5.3 is nearly 5.3 times
    # the constant. The smallest singular value of the regressors is 1.7e-14 times
    # the largest, which statsmodels can treat as 0. After centering, the regression
    # is far from singular, and its estimates are reported for the level.
    y = 5.3 + 1e-12 * np.random.default_rng(7).standard_normal(250)
    dy = np.diff(y)
    rhs = pd.DataFrame(
        {"level": y[2:-1], "diff1": dy[1:-1], "diff2": dy[:-2], "const": 1.0}
    )
    lhs = dy[2:]
    res = centered_ols(lhs, rhs)
    assert int(res.model.rank) == 4
    # The reference centers and normalizes the columns, which makes them
    # orthogonal to the constant and of the same size
    centered = rhs.to_numpy().copy()
    centered[:, :3] -= centered[:, :3].mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    beta = np.linalg.lstsq(centered / norms, lhs, rcond=None)[0] / norms
    assert_allclose(res.params.iloc[:3], beta[:3], rtol=1e-4)
    # The intercept of the centered regression is the mean of lhs, and the constant
    # is the intercept of the regression on the level
    means = rhs.iloc[:, :3].mean(axis=0).to_numpy()
    params = res.params.to_numpy()
    assert_allclose(params[3], lhs.mean() - means @ params[:3], rtol=1e-12)
    assert np.isfinite(res.tvalues).all()
    # The residuals are those of the centered regression, up to the rounding error
    # of the regressors, whose level is 5.3
    assert_allclose(res.resid, lhs - centered @ beta, rtol=0, atol=1e-14)
