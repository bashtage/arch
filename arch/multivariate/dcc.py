"""
Multivariate GARCH models: CCC-GARCH and DCC-GARCH.

References
----------
Bollerslev, T. (1990). Modelling the coherence in short-run nominal exchange
    rates: a multivariate generalized ARCH model. *The Review of Economics
    and Statistics*, 72(3), 498-505.

Engle, R. F. (2002). Dynamic conditional correlation: a simple class of
    multivariate generalized autoregressive conditional heteroskedasticity
    models. *Journal of Business & Economic Statistics*, 20(3), 339-350.

Engle, R. F., & Sheppard, K. (2001). Theoretical and empirical properties
    of dynamic conditional correlation multivariate GARCH. NBER Working Paper
    No. 8554.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Union
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

__all__ = [
    "CCCModel",
    "DCCModel",
    "CCCResult",
    "DCCResult",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_returns(y: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce returns input."""
    if not isinstance(y, (pd.DataFrame, np.ndarray)):
        raise TypeError(
            "y must be a pandas DataFrame or a 2-D NumPy array."
        )
    if isinstance(y, np.ndarray):
        if y.ndim != 2:
            raise ValueError("y must be 2-D when passed as a NumPy array.")
        y = pd.DataFrame(y, columns=[f"asset_{i}" for i in range(y.shape[1])])
    if y.shape[1] < 2:
        raise ValueError(
            "y must contain at least 2 assets (columns). "
            "Use arch.univariate for single-asset GARCH."
        )
    if y.isnull().any().any():
        raise ValueError(
            "y contains missing values. Remove or fill NaNs before fitting."
        )
    return y


def _fit_marginals(
    y: pd.DataFrame,
    p: int,
    q: int,
    dist: str,
    disp: bool,
) -> tuple[list, np.ndarray, np.ndarray]:
    """
    Fit independent GARCH(p,q) to each column of y.

    Returns
    -------
    marginal_results : list of ARCHModelResult
    std_resids : (T, n) array of standardised residuals
    cond_vols : (T, n) array of conditional standard deviations
    """
    try:
        from arch import arch_model
    except ImportError as e:  # pragma: no cover
        raise ImportError("arch package required for marginal GARCH fits.") from e

    disp_str = "off" if not disp else "final"
    marginal_results = []
    std_resid_list = []
    cond_vol_list = []

    for col in y.columns:
        am = arch_model(y[col], vol="GARCH", p=p, q=q, dist=dist, rescale=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp=disp_str, show_warning=False)
        marginal_results.append(res)
        std_resid_list.append(res.std_resid.values)
        cond_vol_list.append(res.conditional_volatility.values)

    std_resids = np.column_stack(std_resid_list)
    cond_vols = np.column_stack(cond_vol_list)
    return marginal_results, std_resids, cond_vols


def _dcc_recursion(
    std_resids: np.ndarray,
    Q_bar: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute DCC Q and R matrices for all time steps.

    Parameters
    ----------
    std_resids : (T, n) array
    Q_bar : (n, n) unconditional covariance of std_resids
    alpha, beta : DCC scalar parameters

    Returns
    -------
    Q_t : (T, n, n) — raw DCC auxiliary matrices
    R_t : (T, n, n) — conditional correlation matrices
    """
    T, n = std_resids.shape
    one_minus_ab = 1.0 - alpha - beta

    Q_t = np.empty((T, n, n))
    R_t = np.empty((T, n, n))

    Q_t[0] = Q_bar.copy()

    for t in range(T):
        if t > 0:
            eps_prev = std_resids[t - 1, :]
            Q_t[t] = (
                one_minus_ab * Q_bar
                + alpha * np.outer(eps_prev, eps_prev)
                + beta * Q_t[t - 1]
            )
        # Normalise Q to correlation
        q_diag_inv_sqrt = 1.0 / np.sqrt(np.diag(Q_t[t]))
        R_t[t] = Q_t[t] * np.outer(q_diag_inv_sqrt, q_diag_inv_sqrt)

    return Q_t, R_t


def _dcc_loglikelihood(
    std_resids: np.ndarray,
    R_t: np.ndarray,
) -> float:
    """
    Compute the DCC quasi-log-likelihood (step 2) for given R_t.

    The step-2 objective (Engle & Sheppard 2001, eq. 14):

        L_DCC = -1/2 Σ_t [ log|R_t| + ε_t' R_t^{-1} ε_t - ε_t' ε_t ]

    The -ε_t'ε_t term removes the contribution already counted in step 1.
    """
    T = std_resids.shape[0]
    llf = 0.0
    for t in range(T):
        R = R_t[t]
        eps = std_resids[t]
        sign, log_det = np.linalg.slogdet(R)
        if sign <= 0:
            return -1e12
        Rinv_eps = np.linalg.solve(R, eps)
        llf += log_det + float(eps @ Rinv_eps) - float(eps @ eps)
    return -0.5 * llf


def _ccc_loglikelihood(
    std_resids: np.ndarray,
    R_bar: np.ndarray,
) -> float:
    """Step-2 log-likelihood for constant correlation."""
    sign, log_det = np.linalg.slogdet(R_bar)
    if sign <= 0:
        return -1e12
    Rinv = np.linalg.inv(R_bar)
    T = std_resids.shape[0]
    llf = 0.0
    for t in range(T):
        eps = std_resids[t]
        llf += log_det + float(eps @ Rinv @ eps) - float(eps @ eps)
    return -0.5 * llf


# ---------------------------------------------------------------------------
# Result classes
# ---------------------------------------------------------------------------

class _MultivariateResult(metaclass=ABCMeta):
    """Shared base for CCC and DCC result objects."""

    def __init__(
        self,
        model: "_MultivariateModel",
        marginal_results: list,
        std_resids: np.ndarray,
        cond_vols: np.ndarray,
        R_t: np.ndarray,
        loglikelihood: float,
        nobs: int,
        n_assets: int,
    ) -> None:
        self._model = model
        self.marginal_results = marginal_results
        self._std_resids = std_resids
        self._cond_vols = cond_vols
        self._R_t = R_t
        self._loglikelihood = loglikelihood
        self._nobs = nobs
        self._n_assets = n_assets

    @property
    def nobs(self) -> int:
        """Number of observations."""
        return self._nobs

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self._n_assets

    @property
    def loglikelihood(self) -> float:
        """Total quasi-log-likelihood (step 1 + step 2)."""
        return self._loglikelihood

    @property
    def num_params(self) -> int:
        """Total number of estimated parameters."""
        return self._count_params()

    @abstractmethod
    def _count_params(self) -> int:
        ...

    @property
    def aic(self) -> float:
        """Akaike information criterion: -2·LLF + 2·k."""
        return -2.0 * self._loglikelihood + 2.0 * self.num_params

    @property
    def bic(self) -> float:
        """Bayesian information criterion: -2·LLF + k·log(T)."""
        return -2.0 * self._loglikelihood + self.num_params * np.log(self._nobs)

    @property
    def conditional_correlations(self) -> np.ndarray:
        """
        Conditional correlation matrices, shape (T, n, n).

        For CCC this is a constant matrix broadcast to shape (T, n, n).
        For DCC it varies each period.
        """
        return self._R_t

    @property
    def conditional_covariances(self) -> np.ndarray:
        """
        Conditional covariance matrices, shape (T, n, n).

        Computed as H_t = D_t R_t D_t where D_t = diag(σ_{1,t}, …, σ_{n,t}).
        """
        T = self._nobs
        n = self._n_assets
        H_t = np.empty((T, n, n))
        for t in range(T):
            D = self._cond_vols[t, :]
            H_t[t] = np.outer(D, D) * self._R_t[t]
        return H_t

    @property
    def conditional_volatilities(self) -> pd.DataFrame:
        """
        Conditional volatilities (marginal), shape (T, n).

        Each column corresponds to one asset.
        """
        return pd.DataFrame(
            self._cond_vols,
            index=self._model.y.index,
            columns=self._model.y.columns,
        )

    @property
    def std_resids(self) -> pd.DataFrame:
        """
        Standardised residuals (demeaned returns divided by conditional vol),
        shape (T, n).
        """
        return pd.DataFrame(
            self._std_resids,
            index=self._model.y.index,
            columns=self._model.y.columns,
        )

    @abstractmethod
    def summary(self) -> pd.Series:
        """Parameter summary."""
        ...

    @abstractmethod
    def forecast(self, horizon: int = 1) -> "MultivariateForecasts":
        ...


class CCCResult(_MultivariateResult):
    """
    Result from fitting a CCC-GARCH model.

    Attributes
    ----------
    marginal_results : list of ARCHModelResult
        Fitted univariate GARCH results for each asset.
    R_bar : ndarray, shape (n, n)
        Estimated constant conditional correlation matrix.
    conditional_correlations : ndarray, shape (T, n, n)
        Constant R_bar repeated T times.
    conditional_covariances : ndarray, shape (T, n, n)
        Time-varying H_t = D_t R_bar D_t.
    loglikelihood : float
        Total quasi-log-likelihood.
    aic, bic : float
        Information criteria.
    """

    def __init__(
        self,
        model: "CCCModel",
        marginal_results: list,
        std_resids: np.ndarray,
        cond_vols: np.ndarray,
        R_bar: np.ndarray,
        loglikelihood: float,
        nobs: int,
        n_assets: int,
    ) -> None:
        R_t = np.broadcast_to(R_bar[None, :, :], (nobs, n_assets, n_assets)).copy()
        super().__init__(
            model, marginal_results, std_resids, cond_vols,
            R_t, loglikelihood, nobs, n_assets,
        )
        self.R_bar = R_bar

    def _count_params(self) -> int:
        # Each GARCH(p,q): omega, alpha_1..p, beta_1..q + mean = 1+p+q+1 = p+q+2
        # Correlation matrix R_bar: n*(n-1)/2 unique off-diagonals
        n = self._n_assets
        marginal_params = sum(len(r.params) for r in self.marginal_results)
        corr_params = n * (n - 1) // 2
        return marginal_params + corr_params

    def summary(self) -> pd.Series:
        """Parameter summary as a pandas Series."""
        rows: dict[str, float] = {}
        for i, res in enumerate(self.marginal_results):
            asset = self._model.y.columns[i]
            for name, val in res.params.items():
                rows[f"{asset}[{name}]"] = float(val)
        n = self._n_assets
        cols = list(self._model.y.columns)
        for i in range(n):
            for j in range(i + 1, n):
                rows[f"rho[{cols[i]},{cols[j]}]"] = float(self.R_bar[i, j])
        rows["LogLikelihood"] = self._loglikelihood
        rows["AIC"] = self.aic
        rows["BIC"] = self.bic
        return pd.Series(rows, name="CCCModel")

    def forecast(self, horizon: int = 1) -> "MultivariateForecasts":
        """
        Produce h-step-ahead covariance forecasts.

        For CCC the correlation is constant, so only the marginal variance
        forecasts evolve.

        Parameters
        ----------
        horizon : int
            Number of steps ahead.

        Returns
        -------
        MultivariateForecasts
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")

        n = self._n_assets
        cov_fc = np.empty((horizon, n, n))
        corr_fc = np.broadcast_to(
            self.R_bar[None, :, :], (horizon, n, n)
        ).copy()

        # Marginal variance forecasts
        vol_fc = np.empty((horizon, n))
        for i, res in enumerate(self.marginal_results):
            fc = res.forecast(horizon=horizon, reindex=False)
            vol_fc[:, i] = np.sqrt(fc.variance.values[-1])

        for h in range(horizon):
            D = vol_fc[h]
            cov_fc[h] = np.outer(D, D) * self.R_bar

        return MultivariateForecasts(
            covariance=cov_fc,
            correlation=corr_fc,
            volatility=vol_fc,
            asset_names=list(self._model.y.columns),
        )


class DCCResult(_MultivariateResult):
    """
    Result from fitting a DCC-GARCH model.

    Attributes
    ----------
    marginal_results : list of ARCHModelResult
        Fitted univariate GARCH results for each asset.
    alpha : float
        DCC innovation parameter (α).
    beta : float
        DCC persistence parameter (β).
    Q_bar : ndarray, shape (n, n)
        Unconditional covariance of standardised residuals.
    conditional_correlations : ndarray, shape (T, n, n)
        Time-varying conditional correlation matrices R_t.
    conditional_covariances : ndarray, shape (T, n, n)
        H_t = D_t R_t D_t.
    loglikelihood : float
        Total quasi-log-likelihood (step 1 + step 2).
    aic, bic : float
        Information criteria.
    """

    def __init__(
        self,
        model: "DCCModel",
        marginal_results: list,
        std_resids: np.ndarray,
        cond_vols: np.ndarray,
        R_t: np.ndarray,
        Q_t: np.ndarray,
        Q_bar: np.ndarray,
        alpha: float,
        beta: float,
        loglikelihood: float,
        nobs: int,
        n_assets: int,
        optimize_result: OptimizeResult,
    ) -> None:
        super().__init__(
            model, marginal_results, std_resids, cond_vols,
            R_t, loglikelihood, nobs, n_assets,
        )
        self.alpha = alpha
        self.beta = beta
        self.Q_bar = Q_bar
        self._Q_t = Q_t
        self._optimize_result = optimize_result

    def _count_params(self) -> int:
        # Marginal GARCH params + 2 DCC params (alpha, beta)
        marginal_params = sum(len(r.params) for r in self.marginal_results)
        return marginal_params + 2

    @property
    def persistence(self) -> float:
        """DCC persistence: α + β. Values < 1 ensure stationarity."""
        return self.alpha + self.beta

    @property
    def half_life(self) -> float:
        """
        Half-life of correlation shocks in periods.

        Computed as log(0.5) / log(α + β).  Infinite when α + β = 1.
        """
        ab = self.alpha + self.beta
        if ab >= 1.0:
            return float("inf")
        return float(np.log(0.5) / np.log(ab))

    def summary(self) -> pd.Series:
        """Parameter summary as a pandas Series."""
        rows: dict[str, float] = {}
        for i, res in enumerate(self.marginal_results):
            asset = self._model.y.columns[i]
            for name, val in res.params.items():
                rows[f"{asset}[{name}]"] = float(val)
        rows["alpha[DCC]"] = self.alpha
        rows["beta[DCC]"] = self.beta
        rows["persistence"] = self.persistence
        rows["half_life"] = self.half_life
        rows["LogLikelihood"] = self._loglikelihood
        rows["AIC"] = self.aic
        rows["BIC"] = self.bic
        return pd.Series(rows, name="DCCModel")

    def plot_correlations(
        self,
        pairs: Optional[list[tuple[int, int]]] = None,
    ) -> "object":
        """
        Plot time-varying conditional correlations.

        Parameters
        ----------
        pairs : list of (int, int), optional
            Asset-index pairs to plot.  Defaults to all unique pairs.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plot_correlations()."
            ) from e

        n = self._n_assets
        cols = list(self._model.y.columns)
        if pairs is None:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        n_pairs = len(pairs)
        fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 3 * n_pairs), squeeze=False)

        index = self._model.y.index
        for ax, (i, j) in zip(axes[:, 0], pairs):
            corr_ij = self._R_t[:, i, j]
            ax.plot(index, corr_ij, lw=0.8)
            ax.axhline(
                corr_ij.mean(), color="r", ls="--", lw=0.8,
                label=f"mean={corr_ij.mean():.3f}",
            )
            ax.set_ylabel(f"ρ({cols[i]}, {cols[j]})")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlabel("Time")

        fig.suptitle("DCC-GARCH Conditional Correlations", fontsize=12)
        fig.tight_layout()
        return fig

    def forecast(self, horizon: int = 1) -> "MultivariateForecasts":
        """
        Produce h-step-ahead covariance and correlation forecasts.

        Correlation forecasts use the mean-reversion formula:

        .. math::

            E[Q_{T+h} \\mid \\mathcal{F}_T] =
                \\bar{Q} + (\\alpha+\\beta)^{h-1}(Q_T - \\bar{Q})

        normalised to yield R-forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead.

        Returns
        -------
        MultivariateForecasts
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")

        n = self._n_assets
        cov_fc = np.empty((horizon, n, n))
        corr_fc = np.empty((horizon, n, n))

        Q_T = self._Q_t[-1]
        ab = self.alpha + self.beta

        for h in range(1, horizon + 1):
            Q_fc = self.Q_bar + (ab ** (h - 1)) * (Q_T - self.Q_bar)
            q_diag_inv_sqrt = 1.0 / np.sqrt(np.diag(Q_fc))
            R_fc = Q_fc * np.outer(q_diag_inv_sqrt, q_diag_inv_sqrt)
            corr_fc[h - 1] = R_fc

        # Marginal variance forecasts
        vol_fc = np.empty((horizon, n))
        for i, res in enumerate(self.marginal_results):
            fc = res.forecast(horizon=horizon, reindex=False)
            vol_fc[:, i] = np.sqrt(fc.variance.values[-1])

        for h in range(horizon):
            D = vol_fc[h]
            cov_fc[h] = np.outer(D, D) * corr_fc[h]

        return MultivariateForecasts(
            covariance=cov_fc,
            correlation=corr_fc,
            volatility=vol_fc,
            asset_names=list(self._model.y.columns),
        )


class MultivariateForecasts:
    """
    Multivariate GARCH forecasts container.

    Parameters
    ----------
    covariance : ndarray, shape (horizon, n, n)
        Forecast conditional covariance matrices.
    correlation : ndarray, shape (horizon, n, n)
        Forecast conditional correlation matrices.
    volatility : ndarray, shape (horizon, n)
        Forecast conditional volatilities (sqrt of marginal variances).
    asset_names : list of str
        Asset names.
    """

    def __init__(
        self,
        covariance: np.ndarray,
        correlation: np.ndarray,
        volatility: np.ndarray,
        asset_names: list[str],
    ) -> None:
        self.covariance = covariance
        self.correlation = correlation
        self.volatility = volatility
        self.asset_names = asset_names

    @property
    def horizon(self) -> int:
        """Forecast horizon."""
        return self.covariance.shape[0]

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self.covariance.shape[1]

    def covariance_dataframe(self, step: int = 1) -> pd.DataFrame:
        """
        Return the h-step-ahead covariance matrix as a DataFrame.

        Parameters
        ----------
        step : int
            Forecast step (1-indexed).
        """
        if step < 1 or step > self.horizon:
            raise ValueError(f"step must be in [1, {self.horizon}].")
        return pd.DataFrame(
            self.covariance[step - 1],
            index=self.asset_names,
            columns=self.asset_names,
        )

    def correlation_dataframe(self, step: int = 1) -> pd.DataFrame:
        """Return the h-step-ahead correlation matrix as a DataFrame."""
        if step < 1 or step > self.horizon:
            raise ValueError(f"step must be in [1, {self.horizon}].")
        return pd.DataFrame(
            self.correlation[step - 1],
            index=self.asset_names,
            columns=self.asset_names,
        )


def simulate_dcc(
    nobs: int,
    n_assets: int,
    alpha: float,
    beta: float,
    Q_bar: np.ndarray,
    garch_params: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Simulate returns from a DCC-GARCH(1,1) model.

    Parameters
    ----------
    nobs : int
        Number of observations to simulate.
    n_assets : int
        Number of assets.
    alpha : float
        DCC innovation parameter.
    beta : float
        DCC persistence parameter.
    Q_bar : ndarray, shape (n_assets, n_assets)
        Unconditional covariance used for the DCC process.  Should be
        close to a correlation matrix (diagonal elements ≈ 1).
    garch_params : dict, optional
        GARCH(1,1) parameters for marginal variance processes.
        Keys: ``'omega'``, ``'alpha_garch'``, ``'beta_garch'``.
        Defaults to ``{'omega': 0.05, 'alpha_garch': 0.10, 'beta_garch': 0.85}``.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    DataFrame, shape (nobs, n_assets)
        Simulated return series.

    Examples
    --------
    >>> import numpy as np
    >>> from arch.multivariate.dcc import simulate_dcc
    >>> Q_bar = np.array([[1.0, 0.6], [0.6, 1.0]])
    >>> sim = simulate_dcc(500, 2, alpha=0.05, beta=0.90, Q_bar=Q_bar, rng=np.random.default_rng(0))
    >>> print(sim.shape)
    (500, 2)
    """
    if alpha <= 0 or beta <= 0 or alpha + beta >= 1:
        raise ValueError("alpha and beta must be positive and alpha+beta < 1.")
    if Q_bar.shape != (n_assets, n_assets):
        raise ValueError(f"Q_bar must be ({n_assets}, {n_assets}).")

    gp = garch_params or {}
    omega = float(gp.get("omega", 0.05))
    alpha_g = float(gp.get("alpha_garch", 0.10))
    beta_g = float(gp.get("beta_garch", 0.85))

    if rng is None:
        rng = np.random.default_rng()

    one_minus_ab = 1.0 - alpha - beta
    Q_t = Q_bar.copy()
    h = np.ones(n_assets) * (omega / (1.0 - alpha_g - beta_g))
    returns = np.empty((nobs, n_assets))
    eps_prev = np.zeros(n_assets)

    for t in range(nobs):
        # Update Q and compute R
        Q_t = (
            one_minus_ab * Q_bar
            + alpha * np.outer(eps_prev, eps_prev)
            + beta * Q_t
        )
        q_diag_inv_sqrt = 1.0 / np.sqrt(np.diag(Q_t))
        R_t = Q_t * np.outer(q_diag_inv_sqrt, q_diag_inv_sqrt)

        # Draw correlated standard normals via Cholesky of R_t
        try:
            L = np.linalg.cholesky(R_t)
        except np.linalg.LinAlgError:
            R_t = _nearest_pd_correlation(R_t)
            L = np.linalg.cholesky(R_t)

        z = L @ rng.standard_normal(n_assets)

        # Scale by marginal conditional volatilities
        D = np.sqrt(h)
        e = D * z
        returns[t] = e

        # Update marginal variances
        h = omega + alpha_g * e**2 + beta_g * h

        # Standardise for next DCC step
        eps_prev = z  # already standard normal given R_t

    cols = [f"asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=cols)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class _MultivariateModel(metaclass=ABCMeta):
    """Abstract base for multivariate GARCH models."""

    def __init__(
        self,
        y: Union[pd.DataFrame, np.ndarray],
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
    ) -> None:
        self._y_raw = _check_returns(y)
        self._p = int(p)
        self._q = int(q)
        self._dist = dist
        if p < 1:
            raise ValueError("p must be >= 1.")
        if q < 1:
            raise ValueError("q must be >= 1.")
        if dist not in ("normal", "t", "skewt", "ged"):
            raise ValueError(
                "dist must be one of 'normal', 't', 'skewt', 'ged'."
            )

    @property
    def y(self) -> pd.DataFrame:
        """Returns data (always a DataFrame)."""
        return self._y_raw

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self._y_raw.shape[1]

    @property
    def nobs(self) -> int:
        """Number of observations."""
        return self._y_raw.shape[0]

    @abstractmethod
    def fit(self, disp: bool = True) -> "_MultivariateResult":
        ...


class CCCModel(_MultivariateModel):
    """
    Constant Conditional Correlation GARCH model (Bollerslev 1990).

    Models a multivariate return series as:

    .. math::

        \\mathbf{r}_t = \\boldsymbol{\\mu} + \\mathbf{e}_t, \\quad
        \\mathbf{e}_t = H_t^{1/2} \\mathbf{z}_t

        H_t = D_t \\bar{R} D_t

    where :math:`D_t = \\text{diag}(h_{1,t}^{1/2}, \\ldots, h_{n,t}^{1/2})`
    contains the marginal GARCH conditional volatilities and :math:`\\bar{R}` is
    a constant correlation matrix estimated from the sample correlations of
    standardised residuals.

    Estimation uses the two-step QMLE of Engle & Sheppard (2001):

    1. Fit an independent GARCH(p, q) to each asset.
    2. Estimate :math:`\\bar{R}` as the sample correlation of standardised
       residuals.

    Parameters
    ----------
    y : DataFrame or ndarray, shape (T, n)
        Return series. Must have at least 2 columns.
    p : int, optional
        GARCH lag order for innovations (default 1).
    q : int, optional
        GARCH lag order for lagged variance (default 1).
    dist : str, optional
        Marginal error distribution: ``'normal'`` (default), ``'t'``,
        ``'skewt'``, ``'ged'``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from arch.multivariate import CCCModel
    >>> rng = np.random.default_rng(42)
    >>> T, n = 500, 3
    >>> returns = pd.DataFrame(rng.standard_normal((T, n)),
    ...                        columns=["SPX", "FTSE", "DAX"])
    >>> model = CCCModel(returns)
    >>> result = model.fit(disp=False)
    >>> print(result.R_bar)
    >>> print(result.aic)
    """

    def fit(self, disp: bool = True) -> CCCResult:
        """
        Fit the CCC-GARCH model.

        Parameters
        ----------
        disp : bool, optional
            Display fitting progress for each marginal GARCH (default True).

        Returns
        -------
        CCCResult
        """
        marginal_results, std_resids, cond_vols = _fit_marginals(
            self._y_raw, self._p, self._q, self._dist, disp
        )

        # Estimate constant correlation as sample correlation of std residuals
        R_bar = np.corrcoef(std_resids.T)
        # Ensure positive definiteness via clipping eigenvalues
        R_bar = _nearest_pd_correlation(R_bar)

        # Compute total log-likelihood
        step1_llf = float(sum(r.loglikelihood for r in marginal_results))
        step2_llf = _ccc_loglikelihood(std_resids, R_bar)
        total_llf = step1_llf + step2_llf

        return CCCResult(
            model=self,
            marginal_results=marginal_results,
            std_resids=std_resids,
            cond_vols=cond_vols,
            R_bar=R_bar,
            loglikelihood=total_llf,
            nobs=self.nobs,
            n_assets=self.n_assets,
        )


def _nearest_pd(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return the nearest positive-definite matrix by flooring eigenvalues."""
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _nearest_pd_correlation(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Project a symmetric matrix onto the nearest positive-definite correlation
    matrix by flooring eigenvalues at eps and re-scaling to unit diagonal.
    """
    R_pd = _nearest_pd(R, eps)
    # Re-scale to correlation matrix (unit diagonal)
    d = np.sqrt(np.diag(R_pd))
    return R_pd / np.outer(d, d)


class DCCModel(_MultivariateModel):
    """
    Dynamic Conditional Correlation GARCH model (Engle 2002).

    Extends CCC-GARCH by allowing the correlation matrix to vary over time
    according to the DCC process:

    .. math::

        Q_t = (1 - \\alpha - \\beta)\\bar{Q}
              + \\alpha \\boldsymbol{\\varepsilon}_{t-1}
                \\boldsymbol{\\varepsilon}_{t-1}^\\top
              + \\beta Q_{t-1}

        R_t = \\text{diag}(Q_t)^{-1/2} Q_t \\text{diag}(Q_t)^{-1/2}

        H_t = D_t R_t D_t

    where :math:`\\boldsymbol{\\varepsilon}_t = D_t^{-1} \\mathbf{e}_t` are
    the standardised residuals from the marginal GARCH models and
    :math:`\\bar{Q} = T^{-1} \\sum_t \\boldsymbol{\\varepsilon}_t
    \\boldsymbol{\\varepsilon}_t^\\top` is the unconditional covariance.

    Estimation uses the two-step QMLE of Engle & Sheppard (2001):

    1. Fit an independent GARCH(p, q) to each asset.
    2. Given the standardised residuals from step 1, maximise the DCC
       quasi-log-likelihood over :math:`(\\alpha, \\beta)` subject to
       :math:`\\alpha \\geq 0,\\, \\beta \\geq 0,\\, \\alpha + \\beta < 1`.

    Parameters
    ----------
    y : DataFrame or ndarray, shape (T, n)
        Return series. Must have at least 2 columns.
    p : int, optional
        GARCH lag order for innovations (default 1).
    q : int, optional
        GARCH lag order for lagged variance (default 1).
    dist : str, optional
        Marginal error distribution: ``'normal'`` (default), ``'t'``,
        ``'skewt'``, ``'ged'``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from arch.multivariate import DCCModel
    >>> rng = np.random.default_rng(42)
    >>> T, n = 1000, 2
    >>> returns = pd.DataFrame(rng.standard_normal((T, n)),
    ...                        columns=["SPX", "FTSE"])
    >>> model = DCCModel(returns)
    >>> result = model.fit(disp=False)
    >>> print(f"alpha={result.alpha:.4f}, beta={result.beta:.4f}")
    >>> print(f"half-life={result.half_life:.1f} periods")
    >>> forecasts = result.forecast(horizon=5)
    >>> print(forecasts.covariance_dataframe(step=1))
    """

    def fit(
        self,
        disp: bool = True,
        starting_values: Optional[np.ndarray] = None,
    ) -> DCCResult:
        """
        Fit the DCC-GARCH model via two-step QMLE.

        Parameters
        ----------
        disp : bool, optional
            Display fitting progress (default True).
        starting_values : ndarray of shape (2,), optional
            Initial values for DCC parameters [alpha, beta].
            Defaults to [0.05, 0.90].

        Returns
        -------
        DCCResult
        """
        # ---- Step 1: marginal GARCH fits ----
        marginal_results, std_resids, cond_vols = _fit_marginals(
            self._y_raw, self._p, self._q, self._dist, disp
        )

        # ---- Step 2: DCC estimation ----
        T, n = std_resids.shape
        # Q_bar: unconditional covariance of standardised residuals.
        # Under correct specification this ≈ the correlation matrix since
        # each ε_{i,t} has unit variance, but we keep it as a covariance
        # (not projected to unit diagonal) per Engle (2002).
        Q_bar = std_resids.T @ std_resids / T
        Q_bar = _nearest_pd(Q_bar)  # ensure positive definite

        sv = np.array([0.05, 0.90] if starting_values is None else starting_values,
                      dtype=float)
        if sv.shape != (2,):
            raise ValueError("starting_values must have shape (2,): [alpha, beta].")

        def neg_step2_llf(params: np.ndarray) -> float:
            a, b = float(params[0]), float(params[1])
            if a <= 0.0 or b <= 0.0 or a + b >= 1.0:
                return 1e12
            _, R_t = _dcc_recursion(std_resids, Q_bar, a, b)
            return -_dcc_loglikelihood(std_resids, R_t)

        bounds = [(1e-6, 0.9999 - 1e-6), (1e-6, 0.9999 - 1e-6)]
        constraints = [{"type": "ineq", "fun": lambda x: 0.9999 - x[0] - x[1]}]

        opt = minimize(
            neg_step2_llf,
            sv,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )

        alpha_hat = float(np.clip(opt.x[0], 1e-8, 1.0))
        beta_hat = float(np.clip(opt.x[1], 1e-8, 1.0))

        if not opt.success and disp:
            warnings.warn(
                f"DCC optimiser did not converge: {opt.message}. "
                "Results may be unreliable. Try different starting_values.",
                UserWarning,
                stacklevel=2,
            )

        Q_t, R_t = _dcc_recursion(std_resids, Q_bar, alpha_hat, beta_hat)

        step1_llf = float(sum(r.loglikelihood for r in marginal_results))
        step2_llf = _dcc_loglikelihood(std_resids, R_t)
        total_llf = step1_llf + step2_llf

        return DCCResult(
            model=self,
            marginal_results=marginal_results,
            std_resids=std_resids,
            cond_vols=cond_vols,
            R_t=R_t,
            Q_t=Q_t,
            Q_bar=Q_bar,
            alpha=alpha_hat,
            beta=beta_hat,
            loglikelihood=total_llf,
            nobs=self.nobs,
            n_assets=self.n_assets,
            optimize_result=opt,
        )
