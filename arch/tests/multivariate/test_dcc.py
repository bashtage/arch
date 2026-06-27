"""
Tests for arch.multivariate: CCCModel, DCCModel, and associated result classes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arch.multivariate import CCCModel, DCCModel, CCCResult, DCCResult, simulate_dcc
from arch.multivariate.dcc import (
    MultivariateForecasts,
    _check_returns,
    _dcc_recursion,
    _dcc_loglikelihood,
    _ccc_loglikelihood,
    _nearest_pd,
    _nearest_pd_correlation,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
T_SMALL = 300   # fast tests
T_LARGE = 800   # statistical tests


def _make_returns(T: int, n: int, rho: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Simulate GARCH(1,1) returns with constant correlation rho."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(
        (1 - rho) * np.eye(n) + rho * np.ones((n, n))
    )
    z = rng.standard_normal((T, n)) @ L.T
    h = np.ones((T, n)) * 0.0004
    for t in range(1, T):
        h[t] = 0.0001 + 0.05 * z[t - 1] ** 2 + 0.90 * h[t - 1]
    returns = z * np.sqrt(h) * 100  # scale to percentage
    cols = [f"asset_{i}" for i in range(n)]
    return pd.DataFrame(returns, columns=cols)


@pytest.fixture(scope="module")
def returns_2d() -> pd.DataFrame:
    return _make_returns(T_SMALL, 2, rho=0.6)


@pytest.fixture(scope="module")
def returns_3d() -> pd.DataFrame:
    return _make_returns(T_SMALL, 3, rho=0.4)


@pytest.fixture(scope="module")
def ccc_result_2d(returns_2d: pd.DataFrame) -> CCCResult:
    return CCCModel(returns_2d).fit(disp=False)


@pytest.fixture(scope="module")
def dcc_result_2d(returns_2d: pd.DataFrame) -> DCCResult:
    return DCCModel(returns_2d).fit(disp=False)


@pytest.fixture(scope="module")
def ccc_result_3d(returns_3d: pd.DataFrame) -> CCCResult:
    return CCCModel(returns_3d).fit(disp=False)


@pytest.fixture(scope="module")
def dcc_result_3d(returns_3d: pd.DataFrame) -> DCCResult:
    return DCCModel(returns_3d).fit(disp=False)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_check_returns_dataframe_passthrough(self) -> None:
        df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        out = _check_returns(df)
        assert isinstance(out, pd.DataFrame)

    def test_check_returns_ndarray_becomes_dataframe(self) -> None:
        arr = np.ones((10, 3))
        out = _check_returns(arr)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["asset_0", "asset_1", "asset_2"]

    def test_check_returns_1d_ndarray_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            _check_returns(np.ones(10))

    def test_check_returns_single_column_raises(self) -> None:
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="at least 2"):
            _check_returns(df)

    def test_check_returns_nan_raises(self) -> None:
        df = pd.DataFrame({"A": [1.0, np.nan], "B": [2.0, 3.0]})
        with pytest.raises(ValueError, match="missing"):
            _check_returns(df)

    def test_check_returns_bad_type_raises(self) -> None:
        with pytest.raises(TypeError):
            _check_returns([[1, 2], [3, 4]])  # type: ignore[arg-type]

    def test_nearest_pd_preserves_pd_matrix(self) -> None:
        A = np.array([[2.0, 0.5], [0.5, 3.0]])
        B = _nearest_pd(A)
        np.testing.assert_allclose(A, B, atol=1e-10)

    def test_nearest_pd_fixes_indefinite_matrix(self) -> None:
        A = np.array([[1.0, 2.0], [2.0, 1.0]])  # not PD (eigenvalues: -1, 3)
        B = _nearest_pd(A)
        eigvals = np.linalg.eigvalsh(B)
        assert np.all(eigvals > 0)

    def test_nearest_pd_correlation_unit_diagonal(self) -> None:
        R = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]])
        out = _nearest_pd_correlation(R)
        np.testing.assert_allclose(np.diag(out), 1.0, atol=1e-12)

    def test_nearest_pd_correlation_is_pd(self) -> None:
        R = np.array([[1.0, 0.99, 0.99], [0.99, 1.0, 0.99], [0.99, 0.99, 1.0]])
        out = _nearest_pd_correlation(R)
        eigvals = np.linalg.eigvalsh(out)
        assert np.all(eigvals > 0)

    def test_dcc_recursion_shapes(self) -> None:
        T, n = 50, 3
        eps = np.random.default_rng(0).standard_normal((T, n))
        Q_bar = eps.T @ eps / T
        Q_t, R_t = _dcc_recursion(eps, Q_bar, alpha=0.05, beta=0.90)
        assert Q_t.shape == (T, n, n)
        assert R_t.shape == (T, n, n)

    def test_dcc_recursion_R_unit_diagonal(self) -> None:
        T, n = 50, 2
        eps = np.random.default_rng(1).standard_normal((T, n))
        Q_bar = eps.T @ eps / T
        _, R_t = _dcc_recursion(eps, Q_bar, alpha=0.05, beta=0.90)
        for t in range(T):
            np.testing.assert_allclose(np.diag(R_t[t]), 1.0, atol=1e-12)

    def test_dcc_recursion_R_symmetric(self) -> None:
        T, n = 50, 3
        eps = np.random.default_rng(2).standard_normal((T, n))
        Q_bar = eps.T @ eps / T
        _, R_t = _dcc_recursion(eps, Q_bar, alpha=0.05, beta=0.90)
        for t in range(T):
            np.testing.assert_allclose(R_t[t], R_t[t].T, atol=1e-12)

    def test_dcc_recursion_R_bounded(self) -> None:
        """Off-diagonal elements of R_t must be in (-1, 1)."""
        T, n = 100, 2
        eps = np.random.default_rng(3).standard_normal((T, n))
        Q_bar = eps.T @ eps / T
        _, R_t = _dcc_recursion(eps, Q_bar, alpha=0.05, beta=0.90)
        off_diag = R_t[:, 0, 1]
        assert np.all(np.abs(off_diag) <= 1.0 + 1e-10)

    def test_dcc_recursion_alpha_zero_constant_correlation(self) -> None:
        """With alpha=0, beta=0, R_t = normalise(Q_bar) for all t."""
        T, n = 30, 2
        eps = np.random.default_rng(4).standard_normal((T, n))
        Q_bar = eps.T @ eps / T
        _, R_t = _dcc_recursion(eps, Q_bar, alpha=0.0, beta=0.0)
        d = 1.0 / np.sqrt(np.diag(Q_bar))
        R_expected = Q_bar * np.outer(d, d)
        for t in range(T):
            np.testing.assert_allclose(R_t[t], R_expected, atol=1e-12)

    def test_dcc_loglikelihood_returns_finite(self) -> None:
        T, n = 50, 2
        eps = np.random.default_rng(5).standard_normal((T, n))
        Q_bar = eps.T @ eps / T
        _, R_t = _dcc_recursion(eps, Q_bar, 0.05, 0.90)
        llf = _dcc_loglikelihood(eps, R_t)
        assert np.isfinite(llf)

    def test_ccc_loglikelihood_returns_finite(self) -> None:
        T, n = 50, 2
        eps = np.random.default_rng(6).standard_normal((T, n))
        R_bar = np.corrcoef(eps.T)
        llf = _ccc_loglikelihood(eps, R_bar)
        assert np.isfinite(llf)

    def test_ccc_loglikelihood_identity_correlation(self) -> None:
        """With R = I the CCC LLF equals -T/2 * sum(eps^2 - eps^2) = 0."""
        T, n = 100, 2
        eps = np.random.default_rng(7).standard_normal((T, n))
        R_bar = np.eye(n)
        llf = _ccc_loglikelihood(eps, R_bar)
        # log|I| = 0, eps'I^{-1}eps - eps'eps = 0 → LLF = 0
        assert abs(llf) < 1e-10


# ---------------------------------------------------------------------------
# CCCModel instantiation and validation
# ---------------------------------------------------------------------------

class TestCCCModelInstantiation:
    def test_accepts_dataframe(self, returns_2d: pd.DataFrame) -> None:
        model = CCCModel(returns_2d)
        assert model.n_assets == 2
        assert model.nobs == T_SMALL

    def test_accepts_ndarray(self) -> None:
        arr = np.random.default_rng(0).standard_normal((100, 3))
        model = CCCModel(arr)
        assert model.n_assets == 3

    def test_y_property_is_dataframe(self, returns_2d: pd.DataFrame) -> None:
        model = CCCModel(returns_2d)
        assert isinstance(model.y, pd.DataFrame)

    def test_p_less_than_1_raises(self, returns_2d: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="p must"):
            CCCModel(returns_2d, p=0)

    def test_q_less_than_1_raises(self, returns_2d: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="q must"):
            CCCModel(returns_2d, q=0)

    def test_bad_dist_raises(self, returns_2d: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="dist must"):
            CCCModel(returns_2d, dist="student")

    def test_single_column_raises(self) -> None:
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="at least 2"):
            CCCModel(df)


# ---------------------------------------------------------------------------
# CCCResult properties
# ---------------------------------------------------------------------------

class TestCCCResult:
    def test_r_bar_shape(self, ccc_result_2d: CCCResult) -> None:
        assert ccc_result_2d.R_bar.shape == (2, 2)

    def test_r_bar_is_correlation(self, ccc_result_2d: CCCResult) -> None:
        R = ccc_result_2d.R_bar
        np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-12)

    def test_r_bar_is_symmetric(self, ccc_result_2d: CCCResult) -> None:
        R = ccc_result_2d.R_bar
        np.testing.assert_allclose(R, R.T, atol=1e-12)

    def test_r_bar_is_positive_definite(self, ccc_result_2d: CCCResult) -> None:
        eigvals = np.linalg.eigvalsh(ccc_result_2d.R_bar)
        assert np.all(eigvals > 0)

    def test_r_bar_3d(self, ccc_result_3d: CCCResult) -> None:
        assert ccc_result_3d.R_bar.shape == (3, 3)

    def test_conditional_correlations_shape(self, ccc_result_2d: CCCResult) -> None:
        R_t = ccc_result_2d.conditional_correlations
        assert R_t.shape == (T_SMALL, 2, 2)

    def test_conditional_correlations_constant(self, ccc_result_2d: CCCResult) -> None:
        """CCC: all correlation matrices equal R_bar."""
        R_t = ccc_result_2d.conditional_correlations
        for t in range(R_t.shape[0]):
            np.testing.assert_allclose(R_t[t], ccc_result_2d.R_bar, atol=1e-12)

    def test_conditional_covariances_shape(self, ccc_result_2d: CCCResult) -> None:
        H_t = ccc_result_2d.conditional_covariances
        assert H_t.shape == (T_SMALL, 2, 2)

    def test_conditional_covariances_positive_definite(
        self, ccc_result_2d: CCCResult
    ) -> None:
        H_t = ccc_result_2d.conditional_covariances
        for t in range(0, T_SMALL, 50):
            eigvals = np.linalg.eigvalsh(H_t[t])
            assert np.all(eigvals > 0), f"H_t[{t}] not PD"

    def test_conditional_volatilities_shape(self, ccc_result_2d: CCCResult) -> None:
        vols = ccc_result_2d.conditional_volatilities
        assert vols.shape == (T_SMALL, 2)
        assert isinstance(vols, pd.DataFrame)

    def test_std_resids_shape(self, ccc_result_2d: CCCResult) -> None:
        eps = ccc_result_2d.std_resids
        assert eps.shape == (T_SMALL, 2)
        assert isinstance(eps, pd.DataFrame)

    def test_nobs(self, ccc_result_2d: CCCResult) -> None:
        assert ccc_result_2d.nobs == T_SMALL

    def test_n_assets(self, ccc_result_2d: CCCResult) -> None:
        assert ccc_result_2d.n_assets == 2

    def test_loglikelihood_finite(self, ccc_result_2d: CCCResult) -> None:
        assert np.isfinite(ccc_result_2d.loglikelihood)

    def test_aic_gt_bic_for_large_T(self, ccc_result_2d: CCCResult) -> None:
        """BIC penalises more heavily than AIC when T > e^2 ≈ 7.4."""
        assert ccc_result_2d.bic >= ccc_result_2d.aic

    def test_aic_formula(self, ccc_result_2d: CCCResult) -> None:
        expected = -2 * ccc_result_2d.loglikelihood + 2 * ccc_result_2d.num_params
        assert abs(ccc_result_2d.aic - expected) < 1e-8

    def test_bic_formula(self, ccc_result_2d: CCCResult) -> None:
        expected = (
            -2 * ccc_result_2d.loglikelihood
            + ccc_result_2d.num_params * np.log(T_SMALL)
        )
        assert abs(ccc_result_2d.bic - expected) < 1e-8

    def test_summary_is_series(self, ccc_result_2d: CCCResult) -> None:
        s = ccc_result_2d.summary()
        assert isinstance(s, pd.Series)

    def test_summary_contains_rho(self, ccc_result_2d: CCCResult) -> None:
        s = ccc_result_2d.summary()
        assert any("rho" in k for k in s.index)

    def test_summary_contains_loglikelihood(self, ccc_result_2d: CCCResult) -> None:
        s = ccc_result_2d.summary()
        assert "LogLikelihood" in s.index

    def test_marginal_results_length(self, ccc_result_2d: CCCResult) -> None:
        assert len(ccc_result_2d.marginal_results) == 2

    def test_correlation_recovers_true_rho(self) -> None:
        """With T=800 and rho=0.7, R_bar[0,1] should be within 0.15 of truth."""
        returns = _make_returns(T_LARGE, 2, rho=0.7, seed=99)
        result = CCCModel(returns).fit(disp=False)
        assert abs(result.R_bar[0, 1] - 0.7) < 0.15

    def test_off_diagonal_positive_for_positive_rho(self) -> None:
        returns = _make_returns(T_SMALL, 2, rho=0.6)
        result = CCCModel(returns).fit(disp=False)
        assert result.R_bar[0, 1] > 0


# ---------------------------------------------------------------------------
# CCCResult.forecast
# ---------------------------------------------------------------------------

class TestCCCForecast:
    def test_forecast_shape(self, ccc_result_2d: CCCResult) -> None:
        fc = ccc_result_2d.forecast(horizon=5)
        assert fc.covariance.shape == (5, 2, 2)
        assert fc.correlation.shape == (5, 2, 2)
        assert fc.volatility.shape == (5, 2)

    def test_forecast_correlation_constant(self, ccc_result_2d: CCCResult) -> None:
        """CCC: all forecast correlations equal R_bar."""
        fc = ccc_result_2d.forecast(horizon=3)
        for h in range(3):
            np.testing.assert_allclose(
                fc.correlation[h], ccc_result_2d.R_bar, atol=1e-12
            )

    def test_forecast_covariance_pd(self, ccc_result_2d: CCCResult) -> None:
        fc = ccc_result_2d.forecast(horizon=1)
        eigvals = np.linalg.eigvalsh(fc.covariance[0])
        assert np.all(eigvals > 0)

    def test_forecast_horizon_1(self, ccc_result_2d: CCCResult) -> None:
        fc = ccc_result_2d.forecast(horizon=1)
        assert fc.horizon == 1

    def test_forecast_invalid_horizon_raises(self, ccc_result_2d: CCCResult) -> None:
        with pytest.raises(ValueError, match="horizon"):
            ccc_result_2d.forecast(horizon=0)

    def test_covariance_dataframe(self, ccc_result_2d: CCCResult) -> None:
        fc = ccc_result_2d.forecast(horizon=3)
        df = fc.covariance_dataframe(step=2)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_covariance_dataframe_bad_step_raises(
        self, ccc_result_2d: CCCResult
    ) -> None:
        fc = ccc_result_2d.forecast(horizon=3)
        with pytest.raises(ValueError):
            fc.covariance_dataframe(step=4)

    def test_correlation_dataframe(self, ccc_result_2d: CCCResult) -> None:
        fc = ccc_result_2d.forecast(horizon=2)
        df = fc.correlation_dataframe(step=1)
        assert isinstance(df, pd.DataFrame)
        np.testing.assert_allclose(np.diag(df.values), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# DCCModel instantiation
# ---------------------------------------------------------------------------

class TestDCCModelInstantiation:
    def test_accepts_dataframe(self, returns_2d: pd.DataFrame) -> None:
        model = DCCModel(returns_2d)
        assert model.n_assets == 2

    def test_accepts_ndarray(self) -> None:
        arr = np.random.default_rng(0).standard_normal((100, 2))
        model = DCCModel(arr)
        assert model.n_assets == 2

    def test_bad_starting_values_shape_raises(self, returns_2d: pd.DataFrame) -> None:
        model = DCCModel(returns_2d)
        with pytest.raises(ValueError, match="starting_values"):
            model.fit(disp=False, starting_values=np.array([0.05]))

    def test_p_less_than_1_raises(self, returns_2d: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="p must"):
            DCCModel(returns_2d, p=0)


# ---------------------------------------------------------------------------
# DCCResult properties
# ---------------------------------------------------------------------------

class TestDCCResult:
    def test_alpha_positive(self, dcc_result_2d: DCCResult) -> None:
        assert dcc_result_2d.alpha >= 0

    def test_beta_positive(self, dcc_result_2d: DCCResult) -> None:
        assert dcc_result_2d.beta >= 0

    def test_persistence_less_than_one(self, dcc_result_2d: DCCResult) -> None:
        assert dcc_result_2d.persistence < 1.0

    def test_half_life_positive(self, dcc_result_2d: DCCResult) -> None:
        assert dcc_result_2d.half_life > 0

    def test_conditional_correlations_shape(self, dcc_result_2d: DCCResult) -> None:
        R_t = dcc_result_2d.conditional_correlations
        assert R_t.shape == (T_SMALL, 2, 2)

    def test_conditional_correlations_unit_diagonal(
        self, dcc_result_2d: DCCResult
    ) -> None:
        R_t = dcc_result_2d.conditional_correlations
        for t in range(0, T_SMALL, 30):
            np.testing.assert_allclose(np.diag(R_t[t]), 1.0, atol=1e-10)

    def test_conditional_correlations_symmetric(
        self, dcc_result_2d: DCCResult
    ) -> None:
        R_t = dcc_result_2d.conditional_correlations
        for t in range(0, T_SMALL, 30):
            np.testing.assert_allclose(R_t[t], R_t[t].T, atol=1e-12)

    def test_conditional_correlations_bounded(
        self, dcc_result_2d: DCCResult
    ) -> None:
        off_diag = dcc_result_2d.conditional_correlations[:, 0, 1]
        assert np.all(np.abs(off_diag) <= 1.0 + 1e-10)

    def test_conditional_covariances_shape(self, dcc_result_2d: DCCResult) -> None:
        H_t = dcc_result_2d.conditional_covariances
        assert H_t.shape == (T_SMALL, 2, 2)

    def test_conditional_covariances_pd_sampled(
        self, dcc_result_2d: DCCResult
    ) -> None:
        H_t = dcc_result_2d.conditional_covariances
        for t in range(0, T_SMALL, 50):
            eigvals = np.linalg.eigvalsh(H_t[t])
            assert np.all(eigvals > 0), f"H_t[{t}] not PD"

    def test_dcc_3d_shapes(self, dcc_result_3d: DCCResult) -> None:
        assert dcc_result_3d.conditional_correlations.shape == (T_SMALL, 3, 3)
        assert dcc_result_3d.conditional_covariances.shape == (T_SMALL, 3, 3)

    def test_loglikelihood_finite(self, dcc_result_2d: DCCResult) -> None:
        assert np.isfinite(dcc_result_2d.loglikelihood)

    def test_aic_bic_ordering(self, dcc_result_2d: DCCResult) -> None:
        assert dcc_result_2d.bic >= dcc_result_2d.aic

    def test_summary_contains_alpha_beta(self, dcc_result_2d: DCCResult) -> None:
        s = dcc_result_2d.summary()
        assert "alpha[DCC]" in s.index
        assert "beta[DCC]" in s.index

    def test_summary_contains_persistence(self, dcc_result_2d: DCCResult) -> None:
        s = dcc_result_2d.summary()
        assert "persistence" in s.index

    def test_summary_persistence_matches_alpha_beta(
        self, dcc_result_2d: DCCResult
    ) -> None:
        s = dcc_result_2d.summary()
        assert abs(s["persistence"] - (s["alpha[DCC]"] + s["beta[DCC]"])) < 1e-10

    def test_summary_is_series(self, dcc_result_2d: DCCResult) -> None:
        assert isinstance(dcc_result_2d.summary(), pd.Series)

    def test_q_bar_shape(self, dcc_result_2d: DCCResult) -> None:
        assert dcc_result_2d.Q_bar.shape == (2, 2)

    def test_q_bar_positive_definite(self, dcc_result_2d: DCCResult) -> None:
        eigvals = np.linalg.eigvalsh(dcc_result_2d.Q_bar)
        assert np.all(eigvals > 0)

    def test_dcc_llf_no_worse_than_ccc(
        self,
        returns_2d: pd.DataFrame,
        ccc_result_2d: CCCResult,
        dcc_result_2d: DCCResult,
    ) -> None:
        """DCC has at least as many free params as CCC, so LLF ≥ CCC LLF."""
        assert dcc_result_2d.loglikelihood >= ccc_result_2d.loglikelihood - 1.0


# ---------------------------------------------------------------------------
# DCCResult.forecast
# ---------------------------------------------------------------------------

class TestDCCForecast:
    def test_forecast_shape(self, dcc_result_2d: DCCResult) -> None:
        fc = dcc_result_2d.forecast(horizon=5)
        assert fc.covariance.shape == (5, 2, 2)
        assert fc.correlation.shape == (5, 2, 2)
        assert fc.volatility.shape == (5, 2)

    def test_forecast_correlation_reverts_to_q_bar(
        self, dcc_result_2d: DCCResult
    ) -> None:
        """At long horizons the DCC correlation forecast converges to Q_bar-implied R."""
        fc = dcc_result_2d.forecast(horizon=200)
        q_bar = dcc_result_2d.Q_bar
        d = 1.0 / np.sqrt(np.diag(q_bar))
        R_inf = q_bar * np.outer(d, d)
        np.testing.assert_allclose(fc.correlation[-1], R_inf, atol=0.05)

    def test_forecast_horizon_1(self, dcc_result_2d: DCCResult) -> None:
        fc = dcc_result_2d.forecast(horizon=1)
        assert fc.horizon == 1

    def test_forecast_invalid_horizon_raises(self, dcc_result_2d: DCCResult) -> None:
        with pytest.raises(ValueError):
            dcc_result_2d.forecast(horizon=0)

    def test_forecast_covariance_pd(self, dcc_result_2d: DCCResult) -> None:
        fc = dcc_result_2d.forecast(horizon=1)
        eigvals = np.linalg.eigvalsh(fc.covariance[0])
        assert np.all(eigvals > 0)

    def test_forecast_correlation_unit_diagonal(
        self, dcc_result_2d: DCCResult
    ) -> None:
        fc = dcc_result_2d.forecast(horizon=3)
        for h in range(3):
            np.testing.assert_allclose(
                np.diag(fc.correlation[h]), 1.0, atol=1e-10
            )

    def test_forecast_covariance_dataframe_shape(
        self, dcc_result_2d: DCCResult
    ) -> None:
        fc = dcc_result_2d.forecast(horizon=5)
        df = fc.covariance_dataframe(step=3)
        assert df.shape == (2, 2)

    def test_forecast_correlation_dataframe_shape(
        self, dcc_result_2d: DCCResult
    ) -> None:
        fc = dcc_result_2d.forecast(horizon=5)
        df = fc.correlation_dataframe(step=1)
        assert df.shape == (2, 2)


# ---------------------------------------------------------------------------
# Statistical / economic correctness
# ---------------------------------------------------------------------------

class TestStatisticalProperties:
    def test_ccc_recovers_positive_correlation(self) -> None:
        """CCC should estimate positive off-diagonal rho when true rho=0.7."""
        returns = _make_returns(T_LARGE, 2, rho=0.7, seed=10)
        result = CCCModel(returns).fit(disp=False)
        assert result.R_bar[0, 1] > 0.4  # at least 0.4 from 0.7

    def test_ccc_recovers_negative_correlation(self) -> None:
        """CCC off-diagonal should be negative when inputs are negatively correlated."""
        rng_s = np.random.default_rng(20)
        T = T_LARGE
        z = rng_s.standard_normal((T, 2))
        z[:, 1] = -z[:, 0] * 0.8 + np.sqrt(1 - 0.64) * z[:, 1]
        df = pd.DataFrame(z, columns=["A", "B"])
        result = CCCModel(df).fit(disp=False)
        assert result.R_bar[0, 1] < 0

    def test_dcc_detects_correlation_change(self) -> None:
        """DCC should show higher corr in first half vs second half for regime data."""
        rng_s = np.random.default_rng(30)
        T = 800
        e = np.empty((T, 2))
        for t in range(T):
            rho = 0.8 if t < 400 else -0.2
            L = np.array([[1.0, 0.0], [rho, np.sqrt(1.0 - rho**2)]])
            e[t] = L @ rng_s.standard_normal(2)
        df = pd.DataFrame(e, columns=["A", "B"])
        result = DCCModel(df).fit(disp=False)
        corr_early = result.conditional_correlations[:300, 0, 1].mean()
        corr_late = result.conditional_correlations[500:, 0, 1].mean()
        assert corr_early > corr_late

    def test_dcc_llf_ge_ccc_llf_on_dcc_data(self) -> None:
        """DCC should fit DCC data better than CCC (higher LLF)."""
        rng_s = np.random.default_rng(40)
        T = 500
        e = np.empty((T, 2))
        for t in range(T):
            rho = 0.8 if t < 250 else -0.2
            L = np.array([[1.0, 0.0], [rho, np.sqrt(1.0 - rho**2)]])
            e[t] = L @ rng_s.standard_normal(2)
        df = pd.DataFrame(e, columns=["A", "B"])
        ccc_llf = CCCModel(df).fit(disp=False).loglikelihood
        dcc_llf = DCCModel(df).fit(disp=False).loglikelihood
        assert dcc_llf >= ccc_llf - 0.5  # DCC at least as good

    def test_dcc_vs_ccc_aic_on_static_data(self) -> None:
        """On truly static-correlation data CCC AIC should be ≤ DCC AIC."""
        returns = _make_returns(T_LARGE, 2, rho=0.5, seed=50)
        ccc_aic = CCCModel(returns).fit(disp=False).aic
        dcc_aic = DCCModel(returns).fit(disp=False).aic
        # CCC uses fewer params so AIC should not be much worse
        assert ccc_aic <= dcc_aic + 10  # allow small tolerance

    def test_std_resids_unit_variance(self, ccc_result_2d: CCCResult) -> None:
        """Standardised residuals should have variance ≈ 1."""
        eps = ccc_result_2d.std_resids.values
        var = np.var(eps, axis=0)
        np.testing.assert_allclose(var, 1.0, atol=0.2)


# ---------------------------------------------------------------------------
# Imports and exports
# ---------------------------------------------------------------------------

class TestImportsAndExports:
    def test_import_ccc(self) -> None:
        from arch.multivariate import CCCModel as _CCCModel
        assert _CCCModel is CCCModel

    def test_import_dcc(self) -> None:
        from arch.multivariate import DCCModel as _DCCModel
        assert _DCCModel is DCCModel

    def test_import_ccc_result(self) -> None:
        from arch.multivariate import CCCResult as _CCCResult
        assert _CCCResult is CCCResult

    def test_import_dcc_result(self) -> None:
        from arch.multivariate import DCCResult as _DCCResult
        assert _DCCResult is DCCResult

    def test_all_exports(self) -> None:
        import arch.multivariate as mv
        for name in ["CCCModel", "DCCModel", "CCCResult", "DCCResult", "simulate_dcc"]:
            assert hasattr(mv, name)
            assert name in mv.__all__


# ---------------------------------------------------------------------------
# simulate_dcc tests
# ---------------------------------------------------------------------------

class TestSimulateDCC:
    def test_output_shape(self) -> None:
        Q_bar = np.array([[1.0, 0.5], [0.5, 1.0]])
        sim = simulate_dcc(200, 2, alpha=0.05, beta=0.90, Q_bar=Q_bar,
                           rng=np.random.default_rng(0))
        assert sim.shape == (200, 2)

    def test_returns_dataframe(self) -> None:
        Q_bar = np.array([[1.0, 0.3], [0.3, 1.0]])
        sim = simulate_dcc(100, 2, alpha=0.05, beta=0.90, Q_bar=Q_bar)
        assert isinstance(sim, pd.DataFrame)

    def test_column_names(self) -> None:
        Q_bar = np.eye(3)
        sim = simulate_dcc(50, 3, alpha=0.05, beta=0.90, Q_bar=Q_bar)
        assert list(sim.columns) == ["asset_0", "asset_1", "asset_2"]

    def test_no_nan(self) -> None:
        Q_bar = np.array([[1.0, 0.6], [0.6, 1.0]])
        sim = simulate_dcc(300, 2, alpha=0.05, beta=0.90, Q_bar=Q_bar,
                           rng=np.random.default_rng(1))
        assert not sim.isnull().any().any()

    def test_finite_values(self) -> None:
        Q_bar = np.array([[1.0, 0.4], [0.4, 1.0]])
        sim = simulate_dcc(300, 2, alpha=0.05, beta=0.90, Q_bar=Q_bar,
                           rng=np.random.default_rng(2))
        assert np.all(np.isfinite(sim.values))

    def test_invalid_alpha_beta_raises(self) -> None:
        Q_bar = np.eye(2)
        with pytest.raises(ValueError, match="alpha and beta"):
            simulate_dcc(100, 2, alpha=0.5, beta=0.6, Q_bar=Q_bar)

    def test_bad_alpha_raises(self) -> None:
        Q_bar = np.eye(2)
        with pytest.raises(ValueError):
            simulate_dcc(100, 2, alpha=-0.1, beta=0.9, Q_bar=Q_bar)

    def test_bad_q_bar_shape_raises(self) -> None:
        Q_bar = np.eye(3)
        with pytest.raises(ValueError, match="Q_bar"):
            simulate_dcc(100, 2, alpha=0.05, beta=0.90, Q_bar=Q_bar)

    def test_reproducibility(self) -> None:
        Q_bar = np.array([[1.0, 0.5], [0.5, 1.0]])
        s1 = simulate_dcc(100, 2, 0.05, 0.90, Q_bar, rng=np.random.default_rng(99))
        s2 = simulate_dcc(100, 2, 0.05, 0.90, Q_bar, rng=np.random.default_rng(99))
        pd.testing.assert_frame_equal(s1, s2)

    def test_custom_garch_params(self) -> None:
        Q_bar = np.array([[1.0, 0.3], [0.3, 1.0]])
        gp = {"omega": 0.02, "alpha_garch": 0.05, "beta_garch": 0.90}
        sim = simulate_dcc(200, 2, 0.05, 0.90, Q_bar, garch_params=gp,
                           rng=np.random.default_rng(3))
        assert sim.shape == (200, 2)

    def test_simulated_correlation_positive(self) -> None:
        """Simulated series with rho=0.8 should have positive sample correlation."""
        Q_bar = np.array([[1.0, 0.8], [0.8, 1.0]])
        sim = simulate_dcc(1000, 2, 0.05, 0.90, Q_bar, rng=np.random.default_rng(4))
        assert np.corrcoef(sim.T)[0, 1] > 0
