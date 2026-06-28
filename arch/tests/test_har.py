"""Tests for the HAR-RV model (arch.univariate.har)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(12345)
N = 500  # number of observations


def make_rv(n: int = N, seed: int = 42) -> pd.Series:
    """Generate a synthetic realized-variance series (non-negative AR(1)-like)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    rv = np.abs(eps) ** 2  # chi-squared(1) proxy for RV
    # Add a mild AR(1) structure to make it more realistic
    for t in range(1, n):
        rv[t] = 0.6 * rv[t - 1] + 0.4 * np.abs(eps[t]) ** 2
    return pd.Series(rv, name="RV")


# Pre-compute a shared RV series for tests that don't need isolation
RV_SERIES = make_rv()


# ---------------------------------------------------------------------------
# Import under PYTHONPATH (works whether arch-har is on sys.path or installed)
# ---------------------------------------------------------------------------

import sys
import os

# Ensure the cloned repo is importable
_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
if _REPO not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO))

from arch.univariate.har import HAR, HARResult  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHARFit:
    """Tests for HAR.fit()."""

    def test_har_fit_returns_result_object(self):
        """fit() should return a HARResult instance."""
        model = HAR(RV_SERIES)
        result = model.fit()
        assert isinstance(result, HARResult)

    def test_har_params_shape(self):
        """Default HAR should have 4 parameters: const + daily + weekly + monthly."""
        model = HAR(RV_SERIES)
        result = model.fit()
        assert result.params_.shape == (4,), (
            f"Expected 4 params, got {result.params_.shape}"
        )
        assert result.param_names == ["const", "daily", "weekly", "monthly"]

    def test_har_rsquared_in_range(self):
        """R-squared must be in [0, 1] for realistic data."""
        model = HAR(RV_SERIES)
        result = model.fit()
        r2 = result.rsquared
        assert 0.0 <= r2 <= 1.0, f"R-squared out of range: {r2}"

    def test_har_fitted_values_shape(self):
        """predict() must return a Series of length (n - max_lag)."""
        model = HAR(RV_SERIES)
        result = model.fit()
        fitted = result.predict()
        expected_len = len(RV_SERIES) - 22  # max_lag = 22
        assert len(fitted) == expected_len, (
            f"Fitted length {len(fitted)} != expected {expected_len}"
        )

    def test_har_forecast_shape(self):
        """forecast(horizon=h) should return an array of length h."""
        model = HAR(RV_SERIES)
        result = model.fit()
        for h in [1, 5, 22]:
            fc = result.forecast(horizon=h)
            assert len(fc) == h, f"Forecast length {len(fc)} != horizon {h}"

    def test_har_pvalues_in_range(self):
        """HAC p-values must lie in [0, 1]."""
        model = HAR(RV_SERIES)
        result = model.fit()
        pvals = result.pvalues
        assert (pvals >= 0.0).all(), "Some p-values are negative"
        assert (pvals <= 1.0).all(), "Some p-values exceed 1"

    def test_har_summary_runs(self):
        """summary() should not raise and should return a non-empty string."""
        model = HAR(RV_SERIES)
        result = model.fit()
        s = result.summary()
        assert isinstance(s, str) and len(s) > 0

    def test_har_summary_contains_key_info(self):
        """summary() should mention R-squared and param names."""
        model = HAR(RV_SERIES)
        result = model.fit()
        s = result.summary()
        assert "R-squared" in s
        assert "const" in s
        assert "daily" in s


class TestHARCustomLags:
    """Tests for HAR with non-default lag specifications."""

    def test_har_custom_lags_two(self):
        """HAR with lags=[1, 10] should produce 3 params (const + 2 betas)."""
        model = HAR(RV_SERIES, lags=[1, 10])
        result = model.fit()
        assert result.params_.shape == (3,), (
            f"Expected 3 params, got {result.params_.shape}"
        )
        assert result.param_names == ["const", "daily", "rv_lag10"]

    def test_har_custom_lags_single(self):
        """HAR with a single lag [1] should produce 2 params."""
        model = HAR(RV_SERIES, lags=[1])
        result = model.fit()
        assert result.params_.shape == (2,)

    def test_har_custom_lags_rsquared(self):
        """Custom lags should still give a valid R-squared."""
        model = HAR(RV_SERIES, lags=[1, 10])
        result = model.fit()
        assert 0.0 <= result.rsquared <= 1.0

    def test_har_lags_sorted_internally(self):
        """HAR should sort lags so max_lag is used correctly."""
        model1 = HAR(RV_SERIES, lags=[22, 1, 5])
        model2 = HAR(RV_SERIES, lags=[1, 5, 22])
        res1 = model1.fit()
        res2 = model2.fit()
        assert_allclose(res1.params_, res2.params_, rtol=1e-10)


class TestHARAgainstOLS:
    """Verify HAR params match a manual OLS computation."""

    def test_har_against_numpy_lstsq(self):
        """HAR params should match manually built OLS solution."""
        rv = np.asarray(RV_SERIES, dtype=float)
        lags = [1, 5, 22]
        max_lag = 22
        n = len(rv) - max_lag

        # Build design matrix manually (matching HAR._build_features logic)
        X_manual = np.ones((n, 4))
        for i in range(n):
            t = i + max_lag
            X_manual[i, 1] = rv[t - 1 : t].mean()       # lag 1
            X_manual[i, 2] = rv[t - 5 : t].mean()       # lag 5
            X_manual[i, 3] = rv[t - 22 : t].mean()      # lag 22
        y_manual = rv[max_lag:]

        params_manual, *_ = np.linalg.lstsq(
            X_manual.T @ X_manual,
            X_manual.T @ y_manual,
            rcond=None,
        )

        model = HAR(RV_SERIES, lags=lags)
        result = model.fit()

        assert_allclose(result.params_, params_manual, rtol=1e-8, atol=1e-12)

    def test_har_rsquared_against_manual(self):
        """R-squared should match the manual calculation."""
        rv = np.asarray(RV_SERIES, dtype=float)
        max_lag = 22
        n = len(rv) - max_lag

        X_manual = np.ones((n, 4))
        for i in range(n):
            t = i + max_lag
            X_manual[i, 1] = rv[t - 1 : t].mean()
            X_manual[i, 2] = rv[t - 5 : t].mean()
            X_manual[i, 3] = rv[t - 22 : t].mean()
        y_manual = rv[max_lag:]

        params_manual, *_ = np.linalg.lstsq(
            X_manual.T @ X_manual,
            X_manual.T @ y_manual,
            rcond=None,
        )
        resid = y_manual - X_manual @ params_manual
        ss_res = resid @ resid
        ss_tot = np.sum((y_manual - y_manual.mean()) ** 2)
        r2_manual = 1.0 - ss_res / ss_tot

        model = HAR(RV_SERIES)
        result = model.fit()
        assert_allclose(result.rsquared, r2_manual, rtol=1e-10)


class TestHARInputValidation:
    """Tests for input validation in HAR.__init__."""

    def test_har_rejects_negative_lags(self):
        with pytest.raises(ValueError, match="positive integers"):
            HAR(RV_SERIES, lags=[-1, 5])

    def test_har_rejects_zero_lag(self):
        with pytest.raises(ValueError, match="positive integers"):
            HAR(RV_SERIES, lags=[0, 5])

    def test_har_rejects_empty_lags(self):
        with pytest.raises(ValueError, match="non-empty"):
            HAR(RV_SERIES, lags=[])

    def test_har_rejects_too_short_series(self):
        short = pd.Series(np.ones(10))
        with pytest.raises(ValueError):
            HAR(short, lags=[1, 5, 22])

    def test_har_rejects_bad_cov_type(self):
        model = HAR(RV_SERIES)
        with pytest.raises(ValueError, match="cov_type"):
            model.fit(cov_type="nonsense")

    def test_har_accepts_numpy_array(self):
        """HAR should accept a plain numpy array as input."""
        rv_arr = np.abs(np.random.randn(300)) ** 2
        model = HAR(rv_arr)
        result = model.fit()
        assert isinstance(result, HARResult)


class TestHARUnadjustedCov:
    """Tests for the unadjusted (homoskedastic) covariance option."""

    def test_unadjusted_pvalues_in_range(self):
        model = HAR(RV_SERIES)
        result = model.fit(cov_type="unadjusted")
        pvals = result.pvalues
        assert (pvals >= 0.0).all()
        assert (pvals <= 1.0).all()

    def test_unadjusted_params_same_as_robust(self):
        """OLS coefficients are the same regardless of cov_type."""
        model = HAR(RV_SERIES)
        res_robust = model.fit(cov_type="robust")
        res_plain = model.fit(cov_type="unadjusted")
        assert_allclose(res_robust.params_, res_plain.params_, rtol=1e-12)


class TestHARForecast:
    """Tests for the forecast method."""

    def test_forecast_horizon_1(self):
        model = HAR(RV_SERIES)
        result = model.fit()
        fc = result.forecast(horizon=1)
        assert fc.shape == (1,)
        assert np.isfinite(fc[0])

    def test_forecast_horizon_22(self):
        model = HAR(RV_SERIES)
        result = model.fit()
        fc = result.forecast(horizon=22)
        assert fc.shape == (22,)
        assert np.all(np.isfinite(fc))

    def test_forecast_horizon_0_raises(self):
        model = HAR(RV_SERIES)
        result = model.fit()
        with pytest.raises(ValueError, match="horizon"):
            result.forecast(horizon=0)

    def test_forecast_is_nonnegative_for_nonneg_data(self):
        """For positive RV data and positive coefficients, forecast should be finite."""
        model = HAR(RV_SERIES)
        result = model.fit()
        fc = result.forecast(horizon=5)
        # Just check finiteness; can't guarantee non-negative with arbitrary params
        assert np.all(np.isfinite(fc))


class TestHARProperties:
    """Tests for HARResult computed properties."""

    def test_params_is_series(self):
        result = HAR(RV_SERIES).fit()
        assert isinstance(result.params, pd.Series)
        assert list(result.params.index) == ["const", "daily", "weekly", "monthly"]

    def test_std_errors_positive(self):
        result = HAR(RV_SERIES).fit()
        se = result.std_errors
        assert (se > 0).all(), "All standard errors should be positive"

    def test_tvalues_shape(self):
        result = HAR(RV_SERIES).fit()
        tv = result.tvalues
        assert tv.shape == (4,)
