"""
HAR (Heterogeneous Autoregressive) model for realized volatility.

Implements the Corsi (2009) HAR-RV model via OLS:

    RV_t = c + beta_d * RV_{t-1} + beta_w * RV^(w)_{t-1} + beta_m * RV^(m)_{t-1} + eps_t

where RV^(w) and RV^(m) are rolling averages over 5 and 22 days respectively.

Reference
---------
Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility.
*Journal of Financial Econometrics*, 7(2), 174-196.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["HAR", "HARResult"]


def _newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Compute Newey-West HAC covariance matrix.

    Parameters
    ----------
    X : ndarray of shape (n, k)
        Regressor matrix.
    resid : ndarray of shape (n,)
        OLS residuals.
    lags : int
        Number of lags for Newey-West kernel.

    Returns
    -------
    ndarray of shape (k, k)
        HAC covariance matrix for the OLS estimator.
    """
    n, k = X.shape
    # Score matrix: (n, k)
    scores = X * resid[:, np.newaxis]

    # Sandwich meat: S = Gamma_0 + sum_{l=1}^{lags} w_l * (Gamma_l + Gamma_l')
    # w_l = 1 - l / (lags + 1)  (Bartlett kernel)
    S = scores.T @ scores  # Gamma_0

    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1)
        gamma = scores[lag:].T @ scores[:-lag]
        S += w * (gamma + gamma.T)

    # Bread: (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)

    # Sandwich: (X'X)^{-1} S (X'X)^{-1}
    cov = XtX_inv @ S @ XtX_inv
    return cov


class HARResult:
    """Result object returned by :meth:`HAR.fit`.

    Attributes
    ----------
    params : ndarray
        OLS coefficient estimates in order: [const, beta_1, beta_2, ...].
    param_names : list of str
        Names corresponding to each parameter.
    resid_ : ndarray
        OLS residuals.
    nobs : int
        Number of observations used in estimation.
    cov_params_ : ndarray
        Estimated covariance matrix of parameters.
    """

    def __init__(
        self,
        params: np.ndarray,
        param_names: list[str],
        resid: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        cov_params: np.ndarray,
        lags: list[int],
        y_original: pd.Series,
    ) -> None:
        self.params_ = params
        self.param_names = param_names
        self.resid_ = resid
        self._X = X
        self._y = y
        self.cov_params_ = cov_params
        self.lags = lags
        self._y_original = y_original
        self.nobs = len(y)

    @property
    def rsquared(self) -> float:
        """R-squared of the OLS fit."""
        ss_res = float(self.resid_ @ self.resid_)
        ss_tot = float(np.sum((self._y - self._y.mean()) ** 2))
        if ss_tot == 0:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    @property
    def pvalues(self) -> pd.Series:
        """HAC-robust two-sided p-values for each parameter (Newey-West)."""
        se = np.sqrt(np.diag(self.cov_params_))
        t_stats = self.params_ / se
        pvals = 2.0 * stats.t.sf(np.abs(t_stats), df=self.nobs - len(self.params_))
        return pd.Series(pvals, index=self.param_names)

    @property
    def std_errors(self) -> pd.Series:
        """HAC-robust standard errors."""
        se = np.sqrt(np.diag(self.cov_params_))
        return pd.Series(se, index=self.param_names)

    @property
    def tvalues(self) -> pd.Series:
        """t-statistics using HAC standard errors."""
        return pd.Series(
            self.params_ / np.sqrt(np.diag(self.cov_params_)),
            index=self.param_names,
        )

    @property
    def params(self) -> pd.Series:
        """Estimated parameters as a pandas Series."""
        return pd.Series(self.params_, index=self.param_names)

    def predict(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> pd.Series:
        """Return in-sample fitted values.

        Parameters
        ----------
        start : int, optional
            Starting index (default: 0).
        end : int, optional
            Ending index exclusive (default: nobs).

        Returns
        -------
        pd.Series
            Fitted values.
        """
        fitted = self._X @ self.params_
        start = start if start is not None else 0
        end = end if end is not None else self.nobs
        return pd.Series(fitted[start:end], name="fitted_values")

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Multi-step ahead forecast using recursive substitution.

        For HAR-RV, the h-step forecast iterates forward by appending the
        one-step forecast to the end of the series and recomputing the
        rolling averages.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        ndarray of shape (horizon,)
            Forecasted values.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        # Reconstruct the series including estimation sample
        max_lag = max(self.lags)
        # We need the raw rv series to build the feature rows for new steps.
        # _y_original contains the full original series; we take the tail we need.
        rv = np.asarray(self._y_original, dtype=float)

        # Extract the coefficients: [const, beta_1, beta_2, ...]
        const = self.params_[0]
        betas = self.params_[1:]  # one per lag

        forecasts = []
        # Working copy: extend with forecasted values iteratively
        rv_ext = rv.copy().tolist()

        for _ in range(horizon):
            row = [1.0]
            for lag in self.lags:
                # Average of the last `lag` values
                window = rv_ext[-lag:]
                row.append(float(np.mean(window)))
            yhat = const + float(np.array(row[1:]) @ betas)
            forecasts.append(yhat)
            rv_ext.append(yhat)

        return np.array(forecasts)

    def summary(self) -> str:
        """Return a text summary table of the HAR estimation results."""
        lines = []
        lines.append("=" * 60)
        lines.append("HAR-RV Model Results (Corsi 2009)")
        lines.append("=" * 60)
        lines.append(f"No. Observations: {self.nobs:>10d}")
        lines.append(f"Lags:             {str(self.lags):>10s}")
        lines.append(f"R-squared:        {self.rsquared:>10.4f}")
        lines.append("-" * 60)
        lines.append(
            f"{'Parameter':<20} {'Coef':>10} {'Std Err':>10} "
            f"{'t-stat':>10} {'p-value':>10}"
        )
        lines.append("-" * 60)
        ses = np.sqrt(np.diag(self.cov_params_))
        tstats = self.params_ / ses
        pvals = self.pvalues.values
        for name, coef, se, t, p in zip(
            self.param_names, self.params_, ses, tstats, pvals
        ):
            lines.append(
                f"{name:<20} {coef:>10.6f} {se:>10.6f} {t:>10.4f} {p:>10.4f}"
            )
        lines.append("=" * 60)
        lines.append("Std. Errors: Newey-West HAC")
        return "\n".join(lines)


class HAR:
    """Heterogeneous Autoregressive model for realized volatility (HAR-RV).

    Implements the Corsi (2009) model via OLS:

        RV_t = c + beta_d * RV_{t-1} + beta_w * RV^(w)_{t-1}
               + beta_m * RV^(m)_{t-1} + epsilon_t

    where the component averages are defined as:

        RV^(w)_{t-1} = mean(RV_{t-1}, ..., RV_{t-5})
        RV^(m)_{t-1} = mean(RV_{t-1}, ..., RV_{t-22})

    Parameters
    ----------
    y : pd.Series or array-like
        Time series of realized variances (must be non-negative).
    lags : list of int, optional
        Lag horizons for the HAR components. Default is [1, 5, 22]
        (daily, weekly, monthly). Each entry ``L`` contributes one
        regressor: the rolling mean of the previous ``L`` observations.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> rv = pd.Series(np.abs(rng.standard_normal(500)) ** 2)
    >>> model = HAR(rv)
    >>> result = model.fit()
    >>> print(result.params)
    >>> print(result.rsquared)
    """

    def __init__(
        self,
        y: Union[pd.Series, np.ndarray],
        lags: Optional[list[int]] = None,
    ) -> None:
        if lags is None:
            lags = [1, 5, 22]
        if not all(isinstance(lag, (int, np.integer)) and lag >= 1 for lag in lags):
            raise ValueError("All lags must be positive integers.")
        if len(lags) == 0:
            raise ValueError("lags must be non-empty.")

        self.lags: list[int] = sorted(int(lag) for lag in lags)
        self._max_lag = max(self.lags)

        if isinstance(y, pd.Series):
            self._y = y
        else:
            self._y = pd.Series(np.asarray(y, dtype=float))

        if len(self._y) <= self._max_lag:
            raise ValueError(
                f"y must have more than {self._max_lag} observations "
                f"(max lag = {self._max_lag})."
            )

    def _build_features(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build the OLS regressor matrix and response vector.

        Returns
        -------
        X : ndarray of shape (T - max_lag, 1 + len(lags))
            Design matrix: [1, rolling_mean_lag1, rolling_mean_lag2, ...].
        y_dep : ndarray of shape (T - max_lag,)
            Dependent variable: RV_t for t = max_lag, ..., T-1.
        names : list of str
            Column names for X.
        """
        rv = np.asarray(self._y, dtype=float)
        T = len(rv)
        n = T - self._max_lag  # effective sample size

        # Response: rv[max_lag], rv[max_lag+1], ..., rv[T-1]
        y_dep = rv[self._max_lag :]

        # Build one column per lag
        # For lag L at time t, the feature is mean(rv[t-L], ..., rv[t-1])
        # i.e. the mean of the L observations ending at t-1.
        cols = [np.ones(n)]
        names = ["const"]

        # Precompute lag names
        lag_labels = {1: "daily", 5: "weekly", 22: "monthly"}

        for lag in self.lags:
            col = np.empty(n)
            for i in range(n):
                t = i + self._max_lag  # current row corresponds to rv[t]
                col[i] = rv[t - lag : t].mean()
            cols.append(col)

            if lag in lag_labels:
                names.append(lag_labels[lag])
            else:
                names.append(f"rv_lag{lag}")

        X = np.column_stack(cols)
        return X, y_dep, names

    def fit(self, cov_type: str = "robust") -> HARResult:
        """Fit the HAR model by OLS.

        Parameters
        ----------
        cov_type : {'robust', 'unadjusted'}
            Covariance estimator for standard errors.

            * ``'robust'``: Newey-West HAC with automatic bandwidth
              (``floor(4 * (n/100)^(2/9))`` lags, per convention).
            * ``'unadjusted'``: Homoskedastic OLS covariance.

        Returns
        -------
        HARResult
        """
        if cov_type not in ("robust", "unadjusted"):
            raise ValueError("cov_type must be 'robust' or 'unadjusted'.")

        X, y_dep, names = self._build_features()
        n, k = X.shape

        # OLS: beta = (X'X)^{-1} X'y
        XtX = X.T @ X
        Xty = X.T @ y_dep
        params, *_ = np.linalg.lstsq(XtX, Xty, rcond=None)
        resid = y_dep - X @ params

        if cov_type == "robust":
            # Newey-West bandwidth: floor(4 * (n/100)^(2/9))
            nw_lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
            cov_params = _newey_west_cov(X, resid, nw_lags)
        else:
            s2 = float(resid @ resid) / (n - k)
            cov_params = s2 * np.linalg.inv(XtX)

        return HARResult(
            params=params,
            param_names=names,
            resid=resid,
            X=X,
            y=y_dep,
            cov_params=cov_params,
            lags=self.lags,
            y_original=self._y,
        )
