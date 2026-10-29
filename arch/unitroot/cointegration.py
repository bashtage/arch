from collections.abc import Sequence
from functools import cached_property
from typing import cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import stats
from statsmodels.iolib.summary import Summary, fmt_2cols, fmt_params
from statsmodels.iolib.table import SimpleTable
from statsmodels.regression.linear_model import OLS, RegressionResults

from arch._typing import ArrayLike1D, ArrayLike2D, Float64Array, Literal, UnitRootTrend
from arch.covariance.kernel import (
    CovarianceEstimate,
    CovarianceEstimator,
    _check_bandwidth_method,
    get_kernel_estimator,
)
from arch.unitroot._engle_granger import EngleGrangerTestResults, engle_granger
from arch.unitroot._phillips_ouliaris import (
    CriticalValueWarning,
    PhillipsOuliarisTestResults,
    phillips_ouliaris,
)
from arch.unitroot._shared import (
    _TREND_TERMS,
    _check_cointegrating_regression,
    _check_finite_estimates,
    _cross_section,
    _infeasible_on_error,
)
from arch.unitroot.unitroot import SHORT_TREND_DESCRIPTION
from arch.utility._validation import (
    check_bool,
    check_float,
    check_int,
    check_option,
)
from arch.utility.array import ensure2d
from arch.utility.io import pval_format, str_format
from arch.utility.timeseries import add_trend
from arch.vendor._decorators import Appender, Substitution

__all__ = [
    "CriticalValueWarning",
    "DynamicOLS",
    "DynamicOLSResults",
    "EngleGrangerTestResults",
    "PhillipsOuliarisTestResults",
    "engle_granger",
    "phillips_ouliaris",
]


class _CommonCointegrationResults:
    def __init__(
        self,
        params: Series,
        cov: DataFrame,
        resid: Series,
        kernel_est: CovarianceEstimator,
        num_x: int,
        trend: UnitRootTrend,
        df_adjust: bool,
        r2: float,
        adj_r2: float,
        estimator_type: str,
    ):
        self._params = params
        self._cov = cov
        self._resid = resid
        self._bandwidth = kernel_est.bandwidth
        self._kernel = kernel_est.__class__.__name__
        self._kernel_est = kernel_est
        self._num_x = num_x
        self._trend = trend
        self._df_adjust = df_adjust
        self._ci_size = params.shape[0]
        self._rsquared = r2
        self._rsquared_adj = adj_r2
        self._estimator_type = estimator_type

    @property
    def params(self) -> Series:
        """The estimated parameters of the cointegrating vector."""
        return self._params.iloc[: self._ci_size]

    @cached_property
    def std_errors(self) -> Series:
        """
        Standard errors of the parameters in the cointegrating vector.
        """
        se = np.sqrt(np.diag(self.cov))
        return Series(se, index=self.params.index, name="std_errors")

    @cached_property
    def tvalues(self) -> Series:
        """
        T-statistics of the parameters in the cointegrating vector.
        """
        return Series(self.params / self.std_errors, name="tvalues")

    @cached_property
    def pvalues(self) -> Series:
        """
        P-values of the parameters in the cointegrating vector.
        """
        return Series(2 * stats.norm.sf(np.abs(self.tvalues)), name="pvalues")

    @property
    def cov(self) -> pd.DataFrame:
        """The estimated parameter covariance of the cointegrating vector."""
        return self._cov.iloc[: self._ci_size, : self._ci_size]

    @property
    def resid(self) -> Series:
        """The model residuals."""
        return self._resid

    @property
    def kernel(self) -> str:
        """The kernel used to estimate the covariance."""
        return self._kernel

    @property
    def bandwidth(self) -> float:
        """The bandwidth used in the parameter covariance estimation."""
        return self._bandwidth

    @property
    def rsquared(self) -> float:
        """The model R²."""
        return self._rsquared

    @property
    def rsquared_adj(self) -> float:
        """The degree-of-freedom adjusted R²."""
        return self._rsquared_adj

    @cached_property
    def _cov_est(self) -> CovarianceEstimate:
        r = np.asarray(self._resid)
        kern_class = self._kernel_est.__class__
        bw = self._bandwidth
        force_int = self._kernel_est.force_int
        cov_est = kern_class(r, bandwidth=bw, center=False, force_int=force_int)
        return cov_est.cov

    @property
    def _df_scale(self) -> float:
        if not self._df_adjust:
            return 1.0
        nobs = self._resid.shape[0]
        nvar = self.params.shape[0]
        return nobs / (nobs - nvar)

    @property
    def residual_variance(self) -> float:
        r"""
        The variance of the regression residual.

        Returns
        -------
        float
            The estimated residual variance.

        Notes
        -----
        The residual variance only accounts for the short-run variance of the
        residual and does not account for any autocorrelation. It is defined
        as

        .. math::

            \hat{\sigma}^2 = T^{-1} \sum _{t=p}^{T-q} \hat{\epsilon}_t^2

        If `df_adjust` is True, then the estimator is rescaled by T/(T-m) where
        m is the number of regressors in the model.
        """
        return float(np.asarray(self._df_scale * self._cov_est.short_run)[0, 0])

    @property
    def long_run_variance(self) -> float:
        """
        The long-run variance of the regression residual.

        Returns
        -------
        float
            The estimated long-run variance of the residual.

        Notes
        -----
        The long-run variance is estimated from the model residuals
        using the same kernel used to estimate the parameter
        covariance.

        If `df_adjust` is True, then the estimator is rescaled by T/(T-m) where
        m is the number of regressors in the model.
        """
        return float(np.asarray(self._df_scale * self._cov_est.long_run)[0, 0])

    @staticmethod
    def _top_table(
        top_left: Sequence[tuple[str, str]],
        top_right: Sequence[tuple[str, str]],
        title: str,
    ) -> SimpleTable:
        stubs = []
        vals = []
        for stub, val in top_left:
            stubs.append(stub)
            vals.append([val])
        table = SimpleTable(vals, txt_fmt=fmt_2cols, title=title, stubs=stubs)

        fmt = fmt_2cols.copy()
        fmt["data_fmts"][1] = "%18s"

        top_right = [("%-21s" % ("  " + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))

        return table

    def _top_right(self) -> list[tuple[str, str]]:
        top_right = [
            ("No. Observations:", str(self._resid.shape[0])),
            ("R²:", str_format(self.rsquared)),
            ("Adjusted. R²:", str_format(self.rsquared_adj)),
            ("Residual Variance:", str_format(self.residual_variance)),
            ("Long-run Variance:", str_format(self.long_run_variance)),
            ("", ""),
        ]
        return top_right

    @staticmethod
    def _param_table(
        params: Float64Array,
        se: Float64Array,
        tstats: Float64Array,
        pvalues: Float64Array,
        stubs: list[str],
        title: str,
    ) -> SimpleTable:
        ci = params[:, None] + se[:, None] * stats.norm.ppf([[0.025, 0.975]])
        param_data = np.column_stack([params, se, tstats, pvalues, ci])
        data = []
        for row in param_data:
            txt_row = []
            for i, v in enumerate(row):
                f = str_format
                if i == 3:
                    f = pval_format
                txt_row.append(f(v))
            data.append(txt_row)

        header = ["Parameter", "Std. Err.", "T-stat", "P-value", "Lower CI", "Upper CI"]
        table = SimpleTable(
            data, stubs=stubs, txt_fmt=fmt_params, headers=header, title=title
        )
        return table

    def summary(self) -> Summary:
        """
        Summary of the model, containing estimated parameters and std. errors.

        Returns
        -------
        statsmodels.iolib.summary.Summary
            A summary instance with methods that support export to text, csv
            or latex.
        """
        if self._bandwidth != int(self._bandwidth):
            bw = str_format(self._bandwidth)
        else:
            bw = str(int(self._bandwidth))

        top_left = [
            ("Trend:", SHORT_TREND_DESCRIPTION[self._trend]),
            ("Kernel:", str(self._kernel)),
            ("Bandwidth:", bw),
            ("", ""),
            ("", ""),
            ("", ""),
        ]

        top_right = self._top_right()
        smry = Summary()
        title = self._estimator_type
        table = self._top_table(top_left, top_right, title)
        # Top Table
        # Parameter table
        smry.tables.append(table)
        params = np.asarray(self.params)
        stubs = list(self.params.index)
        se = np.asarray(self.std_errors)
        tstats = np.asarray(self.tvalues)
        pvalues = np.asarray(self.pvalues)

        title = "Cointegrating Vector"
        table = self._param_table(params, se, tstats, pvalues, stubs, title)
        smry.tables.append(table)

        return smry


class DynamicOLSResults(_CommonCointegrationResults):
    """
    Estimation results for Dynamic OLS models.

    Parameters
    ----------
    params : Series
        The estimated model parameters.
    cov : DataFrame
        The estimated parameter covariance.
    resid : Series
        The model residuals.
    lags : int
        The number of lags included in the model.
    leads : int
        The number of leads included in the model.
    cov_type : str
        The type of the parameter covariance estimator used.
    kernel_est : CovarianceEstimator
        The covariance estimator instance used to estimate the parameter
        covariance.
    num_x : int
        The number of right-hand-side variables in the cointegrating regression.
    trend : {"n", "c", "ct", "ctt"}
        The trend included in the cointegrating regression.
    reg_results : RegressionResults
        Regression results from fitting statsmodels OLS.
    df_adjust : bool
        Whether to degree of freedom adjust the estimator.
    """

    def __init__(
        self,
        params: Series,
        cov: DataFrame,
        resid: Series,
        lags: int,
        leads: int,
        cov_type: str,
        kernel_est: CovarianceEstimator,
        num_x: int,
        trend: UnitRootTrend,
        reg_results: RegressionResults,
        df_adjust: bool,
    ) -> None:
        super().__init__(
            params,
            cov,
            resid,
            kernel_est,
            num_x,
            trend,
            df_adjust,
            r2=reg_results.rsquared,
            adj_r2=reg_results.rsquared_adj,
            estimator_type="Dynamic OLS",
        )
        self._leads = leads
        self._lags = lags
        self._cov_type = cov_type
        self._ci_size = params.shape[0] - self._num_x * (leads + lags + 1)

    @property
    def full_params(self) -> Series:
        """The complete set of parameters, including leads and lags."""
        return self._params

    @property
    def full_cov(self) -> pd.DataFrame:
        """
        Parameter covariance of all the model parameters, incl. leads and lags.
        """
        return self._cov

    @property
    def lags(self) -> int:
        """The number of lags included in the model."""
        return self._lags

    @property
    def leads(self) -> int:
        """The number of leads included in the model."""
        return self._leads

    @property
    def cov_type(self) -> str:
        """The type of parameter covariance estimator used."""
        return self._cov_type

    @property
    def _df_scale(self) -> float:
        if not self._df_adjust:
            return 1.0
        nobs = self._resid.shape[0]
        nvar = self.full_params.shape[0]
        return nobs / (nobs - nvar)

    def summary(self, full: bool = False) -> Summary:
        """
        Summary of the model, containing estimated parameters and std. errors.

        Parameters
        ----------
        full : bool, optional
            Flag indicating whether to include all estimated parameters
            (True) or only the parameters of the cointegrating vector.

        Returns
        -------
        statsmodels.iolib.summary.Summary
            A summary instance with methods that support export to text, csv
            or latex.

        Raises
        ------
        TypeError
            If full is not a bool.
        """
        full = check_bool("full", full)
        if self._bandwidth != int(self._bandwidth):
            bw = str_format(self._bandwidth)
        else:
            bw = str(int(self._bandwidth))

        top_left = [
            ("Trend:", SHORT_TREND_DESCRIPTION[self._trend]),
            ("Leads:", str(self._leads)),
            ("Lags:", str(self._lags)),
            ("Cov Type:", str(self._cov_type)),
            ("Kernel:", str(self._kernel)),
            ("Bandwidth:", bw),
        ]

        top_right = self._top_right()
        smry = Summary()
        typ = "Cointegrating Vector" if not full else "Model"
        title = f"Dynamic OLS {typ} Summary"
        table = self._top_table(top_left, top_right, title)
        # Top Table
        # Parameter table
        smry.tables.append(table)
        if full:
            params = np.asarray(self.full_params)
            stubs = list(self.full_params.index)
            se = np.sqrt(np.diag(self.full_cov))
            tstats = params / se
            pvalues = 2 * stats.norm.sf(np.abs(tstats))
        else:
            params = np.asarray(self.params)
            stubs = list(self.params.index)
            se = np.asarray(self.std_errors)
            tstats = np.asarray(self.tvalues)
            pvalues = np.asarray(self.pvalues)

        title = "Cointegrating Vector" if not full else "Model Parameters"
        assert isinstance(se, np.ndarray)
        table = self._param_table(params, se, tstats, pvalues, stubs, title)
        smry.tables.append(table)

        return smry


class DynamicOLS:
    r"""
    Dynamic OLS (DOLS) cointegrating vector estimation.

    Parameters
    ----------
    y : array_like
        The left-hand-side variable in the cointegrating regression. It must be
        1-d and contain only finite values.
    x : array_like
        The right-hand-side variables in the cointegrating regression. It must
        be 1-d or 2-d with the same number of rows as y, at least 1 column,
        the same index as y if both are pandas objects, and contain only finite
        values. The number of rows must be at least two more than the number of
        parameters in the cointegrating regression (the number of columns of x
        and the number of terms in trend).
    trend : {"n", "c", "ct", "ctt"}, optional
        Trend to include in the cointegrating regression. Not case sensitive.
        Trends are:

        * "n": No deterministic terms
        * "c": Constant
        * "ct": Constant and linear trend
        * "ctt": Constant, linear and quadratic trends

    lags : int, optional
        The number of lags to include in the model.  If None, the optimal
        number of lags is chosen using method. It must be an integer between 0
        and nobs - 2.
    leads : int, optional
        The number of leads to include in the model.  If None, the optimal
        number of leads is chosen using method. It must be an integer between 0
        and nobs - 2.
    common : bool, optional
        Flag indicating that lags and leads should be restricted to the same
        value. When common is True, lags must equal leads and max_lag must
        equal max_lead.
    max_lag : int, optional
        The maximum lag to consider. See Notes for value used when None. It
        must be an integer between 0 and nobs - 2.
    max_lead : int, optional
        The maximum lead to consider. See Notes for value used when None. It
        must be an integer between 0 and nobs - 2.
    method : {"aic", "bic", "hqic"}, optional
        The method used to select lag length when lags or leads is None. Not
        case sensitive.

        * "aic" - Akaike Information Criterion
        * "hqic" - Hannan-Quinn Information Criterion
        * "bic" - Schwartz/Bayesian Information Criterion

    Raises
    ------
    TypeError
        If y or x cannot be converted to float64, lags, leads, max_lag or
        max_lead is not an integer or None, common is not a bool, or trend or
        method is not a string.
    ValueError
        If y or x contains NaN or inf, does not have the required dimensions, the
        same number of rows or the same index, or has too few rows. Also if lags,
        leads, max_lag or max_lead is not between 0 and nobs - 2, if common is
        True and lags and leads (or max_lag and max_lead) differ, if the sample
        that remains after the differences, lags and leads is not larger than
        the number of parameters of the smallest model that is estimated, or if
        trend or method is not one of the options.
    InfeasibleTestException
        If y is constant. This is also a ``ValueError``.

    Notes
    -----
    The cointegrating vector is estimated from the regression

    .. math ::

       Y_t = D_t \delta + X_t \beta + \Delta X_{t} \gamma
             + \sum_{i=1}^p \Delta X_{t-i} \kappa_i
             + \sum _{j=1}^q \Delta X_{t+j} \lambda_j + \epsilon_t

    where p is the lag length and q is the lead length.  :math:`D_t` is a
    vector containing the deterministic terms, if any. All specifications
    include the contemporaneous difference :math:`\Delta X_{t}`.

    When lag lengths are not provided, the optimal lag length is chosen to
    minimize an Information Criterion of the form

    .. math::

        \ln\left(\hat{\sigma}^2\right) + k\frac{c}{T}

    where c is 2 for Akaike, :math:`2\ln\ln T` for Hannan-Quinn and
    :math:`\ln T` for Schwartz/Bayesian.

    See [1]_ and [2]_ for further details.

    References
    ----------
    .. [1] Saikkonen, P. (1992). Estimation and testing of cointegrated
       systems by an autoregressive approximation. Econometric theory,
       8(1), 1-27.
    .. [2] Stock, J. H., & Watson, M. W. (1993). A simple estimator of
       cointegrating vectors in higher order integrated systems.
       Econometrica: Journal of the Econometric Society, 783-820.
    """

    def __init__(
        self,
        y: ArrayLike1D,
        x: ArrayLike2D,
        trend: UnitRootTrend = "c",
        lags: int | None = None,
        leads: int | None = None,
        common: bool = False,
        max_lag: int | None = None,
        max_lead: int | None = None,
        method: Literal["aic", "bic", "hqic"] = "bic",
    ) -> None:
        setup = _check_cointegrating_regression(y, x, trend)
        self._y = setup.y
        self._x = setup.x
        self._trend = setup.trend
        # At least one observation must remain after the first difference
        upper = self._y.shape[0] - 2
        self._lags = check_int("lags", lags, lower=0, upper=upper, allow_none=True)
        self._leads = check_int("leads", leads, lower=0, upper=upper, allow_none=True)
        self._max_lag = check_int(
            "max_lag", max_lag, lower=0, upper=upper, allow_none=True
        )
        self._max_lead = check_int(
            "max_lead", max_lead, lower=0, upper=upper, allow_none=True
        )
        self._method = check_option("method", method, ("aic", "bic", "hqic"))
        self._common = check_bool("common", common)
        self._y_df = pd.DataFrame(self._y)
        self._check_inputs()

    def _check_inputs(self) -> None:
        """Validate the inputs"""
        max_lag, max_lead = self._max_lag, self._max_lead
        if self._common and self._leads != self._lags:
            raise ValueError(
                "common is specified but leads and lags have different values"
            )
        if self._common and self._max_lead != self._max_lag:
            raise ValueError(
                "common is specified but max_lead and max_lag have different values"
            )
        max_ll = self._max_lead_lag()

        nobs = self._y.shape[0]
        obs_remaining = nobs - 1
        obs_remaining -= max_ll if max_lag is None else max_lag
        obs_remaining -= max_ll if max_lead is None else max_lead
        if obs_remaining <= 0:
            raise ValueError(
                "max_lag and max_lead are too large for the amount of "
                "data. The largest model specification in the search "
                "cannot be estimated."
            )
        # The regressions use the sample of the largest model. The smallest model
        # that is estimated must have fewer parameters than the sample has rows.
        most_lags, most_leads = self._lags, self._leads
        if most_lags is None:
            most_lags = max_ll if max_lag is None else max_lag
        if most_leads is None:
            most_leads = max_ll if max_lead is None else max_lead
        rows = nobs - 1 - most_lags - most_leads
        fewest = (self._lags or 0) + (self._leads or 0)
        num_x = self._x.shape[1]
        # x, the trend, and the difference of x with its lags and leads
        num_params = num_x * (fewest + 2) + _TREND_TERMS[self._trend]
        if rows <= num_params:
            raise ValueError(
                "lags, leads, max_lag and max_lead are too large for the amount of "
                f"data. {rows} observations remain after the differences, lags and "
                f"leads, which is not more than the {num_params} parameters of the "
                "smallest model that is estimated."
            )

    def _format_variables(
        self, leads: int, lags: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Format the variables for the regression"""
        x = self._x
        y = self._y_df
        delta_x = x.diff()
        data = [y, x]

        for lag in range(-lags, leads + 1):
            lag_data = delta_x.shift(-lag)
            typ = "LAG" if lag < 0 else "LEAD"
            lag_data.columns = pd.Index(
                [f"D.{c}.{typ}{abs(lag)}" for c in lag_data.columns]
            )
            if lag == 0:
                lag_data.columns = pd.Index([f"D.{c}" for c in lag_data.columns])
            data.append(lag_data)
        data_df: pd.DataFrame = pd.concat(data, axis=1).dropna()
        lhs, rhs = data_df.iloc[:, :1], data_df.iloc[:, 1:]
        nrhs = rhs.shape[1]
        with_trend = add_trend(rhs, trend=self._trend, prepend=True)
        assert isinstance(with_trend, pd.DataFrame)
        rhs = with_trend
        ntrend = rhs.shape[1] - nrhs
        if ntrend:
            nx = x.shape[1]
            trend = rhs.iloc[:, :ntrend]
            rhs = pd.concat(
                [rhs.iloc[:, ntrend : ntrend + nx], trend, rhs.iloc[:, ntrend + nx :]],
                axis=1,
            )
        return lhs, rhs

    def _ic(self, resids: Float64Array, nparam: int) -> float:
        """Compute an info criterion"""
        nobs = resids.shape[0]
        sigma2 = float(resids.T @ resids / nobs)
        if self._method == "aic":
            penalty = 2.0
        elif self._method == "hqic":
            penalty = 2.0 * float(np.log(np.log(nobs)))
        else:  # bic
            penalty = float(np.log(nobs))
        return np.log(sigma2) + nparam * penalty / nobs

    def _max_lead_lag(self) -> int:
        nobs = self._y.shape[0]
        return int(np.ceil(12.0 * (nobs / 100) ** (1 / 4)))

    def _leads_and_lags(self) -> tuple[int, int]:
        """Select the optimal number of leads and lags"""
        if self._lags is not None and self._leads is not None:
            return self._leads, self._lags
        nobs = self._y.shape[0]
        max_lead_lag = int(np.ceil(12.0 * (nobs / 100) ** (1 / 4)))
        if self._lags is None:
            max_lag = max_lead_lag if self._max_lag is None else self._max_lag
            min_lag = 0
        else:
            min_lag = max_lag = self._lags
        if self._leads is None:
            max_lead = max_lead_lag if self._max_lead is None else self._max_lead
            min_lead = 0
        else:
            min_lead = max_lead = self._leads
        variables = self._format_variables(max_lead, max_lag)
        lhs = np.asarray(variables[0])
        rhs = np.asarray(variables[1])
        nx = self._x.shape[1]
        # +1 to account for the Delta X(t) (not a lead or a lag)
        lead_lag_offset = rhs.shape[1] - (max_lead + max_lag + 1) * nx
        always_loc = np.arange(lead_lag_offset)
        best_ic = np.inf
        best_leads_and_lags = (0, 0)
        for lag in range(min_lag, max_lag + 1):
            for lead in range(min_lead, max_lead + 1):
                if self._common and lag != lead:
                    continue
                lag_start = max_lag - lag
                # +1 to get LAG0 in all regressions
                lead_end = max_lag + 1 + lead
                lead_lag_locs = np.arange(lag_start * nx, lead_end * nx)
                lead_lag_locs += lead_lag_offset
                locs = np.r_[always_loc, lead_lag_locs]
                _rhs = rhs[:, locs]
                params = np.linalg.lstsq(_rhs, lhs, rcond=None)[0]
                resid = np.squeeze(lhs - _rhs @ params)
                ic = self._ic(resid, params.shape[0])
                if ic < best_ic:
                    best_ic = ic
                    best_leads_and_lags = (lead, lag)
        return best_leads_and_lags

    @_infeasible_on_error("Dynamic OLS", "estimates")
    def fit(
        self,
        cov_type: Literal[
            "unadjusted", "homoskedastic", "robust", "kernel"
        ] = "unadjusted",
        kernel: str = "bartlett",
        bandwidth: float | None = None,
        force_int: bool = False,
        df_adjust: bool = False,
        bandwidth_method: str = "newey-west",
    ) -> DynamicOLSResults:
        r"""
        Estimate the Dynamic OLS regression.

        Parameters
        ----------
        cov_type : {"unadjusted", "homoskedastic", "robust", "kernel"}, optional
            Either "unadjusted" (or its equivalent "homoskedastic") or "robust"
            (or its equivalent "kernel"). Not case sensitive.
        kernel : str, optional
            The string name of any known kernel-based long-run
            covariance estimator. Common choices are "bartlett" for the
            Bartlett kernel (Newey-West), "parzen" for the Parzen kernel
            and "quadratic-spectral" for the Quadratic Spectral kernel. Not case
            sensitive.
        bandwidth : float, optional
            The bandwidth to use. It must be a finite non-negative number. If
            not provided, the optimal bandwidth is estimated from the data.
            Setting the bandwidth to 0 and using "unadjusted" produces the
            classic OLS covariance estimator. Setting the bandwidth to 0 and
            using "robust" produces White's covariance estimator.
        force_int : bool, optional
            Whether to force the estimated optimal bandwidth to be an integer.
        df_adjust : bool, optional
            Whether to adjust the parameter covariance to account for the
            number of parameters estimated in the regression. If true, the
            parameter covariance estimator is multiplied by T/(T-k) where
            k is the number of regressors in the model.
        bandwidth_method : {"newey-west", "andrews"}, optional
            The method used to estimate the bandwidth when bandwidth is None.
            "newey-west" uses the nonparametric estimate of Newey and West (1994)
            and "andrews" uses the AR(1) plug-in estimate of Andrews (1991), see
            ``opt_bandwidth`` of the kernel estimators. Not case sensitive, and
            hyphens and underscores are ignored as in the names of kernels.
            Ignored if bandwidth is provided.

        Returns
        -------
        DynamicOLSResults
            The estimation results.

        Raises
        ------
        TypeError
            If bandwidth is not a real number or None, force_int or df_adjust is
            not a bool, or cov_type, kernel or bandwidth_method is not a string.
        ValueError
            If bandwidth is negative, NaN or inf, cov_type, kernel or
            bandwidth_method is not one of the options, or bandwidth_method is
            "andrews" and the AR(1) cannot be fitted to the residuals or scores.

        See Also
        --------
        arch.unitroot.cointegration.engle_granger
            Cointegration testing using the Engle-Granger methodology.
        statsmodels.regression.linear_model.OLS
            Ordinary Least Squares regression.

        Notes
        -----
        When using the unadjusted covariance, the parameter covariance is
        estimated as

        .. math::

            T^{-1} \hat{\sigma}^2_{HAC} \hat{\Sigma}_{ZZ}^{-1}

        where :math:`\hat{\sigma}^2_{HAC}` is an estimator of the long-run
        variance of the regression error and
        :math:`\hat{\Sigma}_{ZZ}=T^{-1}Z'Z`. :math:`Z_t` is a vector the
        includes all terms in the regression (i.e., deterministics,
        cross-sectional, leads and lags) When using the robust covariance,
        the parameter covariance is estimated as

        .. math::

            T^{-1} \hat{\Sigma}_{ZZ}^{-1} \hat{S}_{HAC} \hat{\Sigma}_{ZZ}^{-1}

        where :math:`\hat{S}_{HAC}` is a Heteroskedasticity-Autocorrelation
        Consistent estimator of the covariance of the regression scores
        :math:`Z_t\epsilon_t`.
        """
        cov_type = cast(
            "Literal['unadjusted', 'homoskedastic', 'robust', 'kernel']",
            check_option(
                "cov_type",
                cov_type,
                ("unadjusted", "homoskedastic", "robust", "kernel"),
            ),
        )
        get_kernel_estimator(kernel)
        bandwidth = check_float("bandwidth", bandwidth, lower=0.0, allow_none=True)
        force_int = check_bool("force_int", force_int)
        df_adjust = check_bool("df_adjust", df_adjust)
        bandwidth_method = _check_bandwidth_method(bandwidth_method)
        leads, lags = self._leads_and_lags()
        # TODO: Rank check and drop??
        lhs, rhs = self._format_variables(leads, lags)
        mod = OLS(lhs, rhs)
        res = mod.fit()
        coeffs = np.asarray(res.params)
        resid = pd.Series(
            lhs.squeeze() - np.asarray(rhs @ coeffs).squeeze(), name="resid"
        )
        cov, est = self._cov(
            cov_type,
            kernel,
            bandwidth,
            force_int,
            df_adjust,
            rhs,
            resid,
            bandwidth_method,
        )
        params = Series(np.squeeze(coeffs), index=rhs.columns, name="params")
        _check_finite_estimates(params, cov, "Dynamic OLS")
        num_x = self._x.shape[1]
        return DynamicOLSResults(
            params,
            cov,
            resid,
            lags,
            leads,
            cov_type,
            est,
            num_x,
            self._trend,
            res,
            df_adjust,
        )

    @staticmethod
    def _cov(
        cov_type: Literal["unadjusted", "homoskedastic", "robust", "kernel"],
        kernel: str,
        bandwidth: float | None,
        force_int: bool,
        df_adjust: bool,
        rhs: pd.DataFrame,
        resids: Series,
        bandwidth_method: str = "newey-west",
    ) -> tuple[pd.DataFrame, CovarianceEstimator]:
        """Estimate the covariance"""
        kernel_est = get_kernel_estimator(kernel)
        x = np.asarray(rhs)
        eps = ensure2d(np.asarray(resids), "eps")
        nobs, nx = x.shape
        sigma_xx = x.T @ x / nobs
        sigma_xx_inv = np.linalg.inv(sigma_xx)
        scale = nobs / (nobs - nx) if df_adjust else 1.0
        if cov_type in ("unadjusted", "homoskedastic"):
            est = kernel_est(
                eps,
                bandwidth,
                center=False,
                force_int=force_int,
                bandwidth_method=bandwidth_method,
            )
            sigma2 = np.squeeze(est.cov.long_run)
            cov = (scale * sigma2) * sigma_xx_inv / nobs
        elif cov_type in ("robust", "kernel"):
            scores = x * eps
            est = kernel_est(
                scores,
                bandwidth,
                center=False,
                force_int=force_int,
                bandwidth_method=bandwidth_method,
            )
            s = est.cov.long_run
            cov = scale * sigma_xx_inv @ s @ sigma_xx_inv / nobs
        else:
            raise ValueError(f"Unknown cov_type: {cov_type}")
        cov_df = pd.DataFrame(cov, columns=rhs.columns, index=rhs.columns)
        return cov_df, est


class CointegrationAnalysisResults(_CommonCointegrationResults):
    """
    Estimation results for cointegrating vector estimators.

    Returned by the ``fit`` method of FullyModifiedOLS and CanonicalCointegratingReg.

    Parameters
    ----------
    params : Series
        The estimated model parameters.
    cov : DataFrame
        The estimated parameter covariance.
    resid : Series
        The model residuals.
    omega_112 : float
        The long-run variance estimate used in the parameter covariance estimator.
    kernel_est : CovarianceEstimator
        The covariance estimator instance used to estimate the long-run variance
        of the residuals.
    num_x : int
        The number of right-hand-side variables in the cointegrating regression.
    trend : {"n", "c", "ct", "ctt"}
        The trend included in the cointegrating regression.
    df_adjust : bool
        Whether the parameter covariance was adjusted for degrees of freedom.
    rsquared : float
        The model R-squared.
    rsquared_adj : float
        The degree-of-freedom adjusted R-squared.
    estimator_type : str
        The name of the estimator, which is used in the summary.
    """

    def __init__(
        self,
        params: Series,
        cov: DataFrame,
        resid: Series,
        omega_112: float,
        kernel_est: CovarianceEstimator,
        num_x: int,
        trend: UnitRootTrend,
        df_adjust: bool,
        rsquared: float,
        rsquared_adj: float,
        estimator_type: str,
    ):
        super().__init__(
            params,
            cov,
            resid,
            kernel_est,
            num_x,
            trend,
            df_adjust,
            rsquared,
            rsquared_adj,
            estimator_type,
        )
        self._omega_112 = omega_112

    @property
    def long_run_variance(self) -> float:
        """
        Long-run variance estimate used in the parameter covariance estimator.
        """
        return self._omega_112


COMMON_DOCSTRING = r"""
    %(method)s cointegrating vector estimation.

    Parameters
    ----------
    y : array_like
        The left-hand-side variable in the cointegrating regression. It must be
        1-d and contain only finite values.
    x : array_like
        The right-hand-side variables in the cointegrating regression. It must
        be 1-d or 2-d with the same number of rows as y, at least 1 column,
        the same index as y if both are pandas objects, and contain only finite
        values. The number of rows must be at least two more than the number of
        parameters in the cointegrating regression (the number of columns of x
        and the number of terms in trend).
    trend : {"n", "c", "ct", "ctt"}, optional
        Trend to include in the cointegrating regression. Not case sensitive.
        Trends are:

        * "n": No deterministic terms
        * "c": Constant
        * "ct": Constant and linear trend
        * "ctt": Constant, linear and quadratic trends

    x_trend : {"c", "ct", "ctt"}, optional
        Trends that affect the x-data but do not appear in the
        cointegrating regression. x_trend must be at least as large as
        trend, so that if trend is "ct", x_trend must be either "ct" or
        "ctt". If None, x_trend is the same as trend. Not case sensitive.

    Raises
    ------
    TypeError
        If y or x cannot be converted to float64, or trend or x_trend is not a
        string.
    ValueError
        If y or x contains NaN or inf, does not have the required dimensions, the
        same number of rows or the same index, or has too few rows. Also if trend
        or x_trend is not one of the options.
    InfeasibleTestException
        If y is constant. This is also a ``ValueError``.

    Notes
    -----
    The cointegrating vector is estimated from the regressions

    .. math::

       Y_t & = D_{1t} \delta + X_t \beta + \eta_{1t} \\
       X_t & = D_{1t} \Gamma_1 + D_{2t}\Gamma_2 + \epsilon_{2t} \\
       \eta_{2t} & = \Delta \epsilon_{2t}

    or if estimated in differences, the last two lines are

    .. math::

       \Delta X_t = \Delta D_{1t} \Gamma_1 + \Delta D_{2t} \Gamma_2 + \eta_{2t}

    Define the vector of residuals as :math:`\eta = (\eta_{1t},\eta'_{2t})'`, and the
    long-run covariance

    .. math::

       \Omega = \sum_{h=-\infty}^{\infty} E[\eta_t\eta_{t-h}']

    and the one-sided long-run covariance matrix

    .. math::

       \Lambda_0 = \sum_{h=0}^\infty E[\eta_t\eta_{t-h}']

    The covariance matrices are partitioned into a block form

    .. math::

       \Omega = \left[\begin{array}{cc}
                         \omega_{11} & \omega_{12} \\
                         \omega'_{12} & \Omega_{22}
                \end{array} \right]

    The cointegrating vector is then estimated using modified data

%(estimator)s
    """

CCR_METHOD = "Canonical Cointegrating Regression"

CCR_ESTIMATOR = r"""
    .. math::

        X^\star_t & = X_t - \hat{\Lambda}_2'\hat{\Sigma}^{-1}\hat{\eta}_t \\
        Y^\star_t & = Y_t - (\hat{\Sigma}^{-1} \hat{\Lambda}_2 \hat{\beta}
                           + \hat{\kappa})' \hat{\eta}_t

    where :math:`\hat{\kappa} = (0,\hat{\Omega}_{22}^{-1}\hat{\Omega}'_{12})` and
    the regression

    .. math::

       Y^\star_t = D_{1t} \delta + X^\star_t \beta + \eta^\star_{1t}

    See [1]_ for further details.

    References
    ----------
    .. [1] Park, J. Y. (1992). Canonical cointegrating regressions. Econometrica:
       Journal of the Econometric Society, 119-143.
"""

FMOLS_METHOD = "Fully Modified OLS"

FMOLS_ESTIMATOR = r"""
    .. math::

       Y^\star_t = Y_t - \hat{\omega}_{12}\hat{\Omega}_{22}\hat{\eta}_{2t}

    as

    .. math::

        \hat{\theta} = \left[\begin{array}{c}\hat{\gamma}_1 \\
                       \hat{\beta} \end{array}\right]
                     = \left(\sum_{t=2}^T Z_tZ'_t\right)^{-1}
                       \left(\sum_{t=2}^t Z_t Y^\star_t -
                       T \left[\begin{array}{c} 0 \\ \lambda^{\star\prime}_{12}
                       \end{array}\right]\right)

    where the bias term is defined

    .. math::

       \lambda^\star_{12} = \hat{\lambda}_{12}
                            - \hat{\omega}_{12}\hat{\Omega}_{22}\hat{\omega}_{21}

    See [1]_ for further details.

    References
    ----------
    .. [1] Hansen, B. E., & Phillips, P. C. (1990). Estimation and inference in
       models of cointegration: A simulation study. Advances in Econometrics,
       8(1989), 225-248.
"""


def _check_fit_arguments(
    bandwidth: float | None,
    force_int: bool,
    diff: bool,
    df_adjust: bool,
    bandwidth_method: str,
) -> tuple[float | None, bool, bool, bool, str]:
    """Check the arguments of the fit method of FMOLS and CCR"""
    return (
        check_float("bandwidth", bandwidth, lower=0.0, allow_none=True),
        cast("bool", check_bool("force_int", force_int)),
        cast("bool", check_bool("diff", diff)),
        cast("bool", check_bool("df_adjust", df_adjust)),
        _check_bandwidth_method(bandwidth_method),
    )


@Substitution(method=FMOLS_METHOD, estimator=FMOLS_ESTIMATOR)
@Appender(COMMON_DOCSTRING)
class FullyModifiedOLS:
    def __init__(
        self,
        y: ArrayLike1D,
        x: ArrayLike2D,
        trend: UnitRootTrend = "c",
        x_trend: UnitRootTrend | None = None,
    ) -> None:
        setup = _check_cointegrating_regression(y, x, trend)
        self._y = setup.y
        self._x = setup.x
        self._trend = setup.trend
        self._x_trend: UnitRootTrend | None = None
        if x_trend is not None:
            # The documented options are "c", "ct" and "ctt", "n" and "t" also work
            self._x_trend = cast(
                "UnitRootTrend",
                check_option(
                    "x_trend",
                    x_trend,
                    ("n", "c", "ct", "ctt", "t"),
                    display=("c", "ct", "ctt"),
                ),
            )
        self._y_df = pd.DataFrame(self._y)

    def _common_fit(
        self,
        kernel: str,
        bandwidth: float | None,
        force_int: bool,
        diff: bool,
        bandwidth_method: str = "newey-west",
    ) -> tuple[CovarianceEstimator, Float64Array, Float64Array]:
        kern_est = get_kernel_estimator(kernel)
        res = _cross_section(self._y, self._x, self._trend)
        x = np.asarray(self._x)
        eta_1 = np.asarray(res.resid)
        if self._x_trend is not None:
            x_trend = self._x_trend
        else:
            x_trend = self._trend
        tr = add_trend(nobs=x.shape[0], trend=x_trend)
        if tr.shape[1] > 1 and diff:
            delta_tr = np.diff(tr[:, 1:], axis=0)
            delta_x = np.diff(x, axis=0)
            gamma = np.linalg.lstsq(delta_tr, delta_x, rcond=None)[0]
            eta_2 = delta_x - delta_tr @ gamma
        else:
            if tr.shape[1]:
                gamma = np.linalg.lstsq(tr, x, rcond=None)[0]
                eps = x - tr @ gamma
            else:
                eps = x
            eta_2 = np.diff(eps, axis=0)
        eta = np.column_stack([eta_1[1:], eta_2])
        cov_est = kern_est(
            eta,
            bandwidth=bandwidth,
            center=False,
            force_int=force_int,
            bandwidth_method=bandwidth_method,
        )
        beta = np.asarray(res.params)[: x.shape[1]]
        return cov_est, eta, beta

    def _final_statistics(self, theta: Series) -> tuple[Series, float, float]:
        z = add_trend(self._x, self._trend)
        nobs, nvar = z.shape
        resid = self._y - np.asarray(z @ theta)
        resid.name = "resid"
        center = 0.0
        tss_df = 0
        if "c" in self._trend:
            center = float(self._y.mean())
            tss_df = 1
        y_centered = self._y - center
        ssr = float(cast("float", resid.T @ resid))
        tss = float(cast("float", y_centered.T @ y_centered))
        r2 = 1.0 - ssr / tss
        r2_adj = 1.0 - (ssr / (nobs - nvar)) / (tss / (nobs - tss_df))
        return resid, r2, r2_adj

    @_infeasible_on_error("Fully Modified OLS", "estimates")
    def fit(
        self,
        kernel: str = "bartlett",
        bandwidth: float | None = None,
        force_int: bool = True,
        diff: bool = False,
        df_adjust: bool = False,
        bandwidth_method: str = "newey-west",
    ) -> CointegrationAnalysisResults:
        """
        Estimate the cointegrating vector.

        Parameters
        ----------
        kernel : str, optional
            The string name of any known kernel-based long-run
            covariance estimator. Common choices are "bartlett" for the
            Bartlett kernel (Newey-West), "parzen" for the Parzen kernel
            and "quadratic-spectral" for the Quadratic Spectral kernel. Not case
            sensitive.
        bandwidth : float, optional
            The bandwidth to use. It must be a finite non-negative number. If
            not provided, the optimal bandwidth is estimated from the data.
        force_int : bool, optional
            Whether to force the estimated optimal bandwidth to be an integer.
        diff : bool, optional
            Use differenced data to estimate the residuals.
        df_adjust : bool, optional
            Whether to adjust the parameter covariance to account for the
            number of parameters estimated in the regression. If true, the
            parameter covariance estimator is multiplied by T/(T-k) where
            k is the number of regressors in the model.
        bandwidth_method : {"newey-west", "andrews"}, optional
            The method used to estimate the bandwidth when bandwidth is None.
            "newey-west" uses the nonparametric estimate of Newey and West (1994)
            and "andrews" uses the AR(1) plug-in estimate of Andrews (1991), see
            ``opt_bandwidth`` of the kernel estimators. Not case sensitive, and
            hyphens and underscores are ignored as in the names of kernels.
            Ignored if bandwidth is provided.

        Returns
        -------
        CointegrationAnalysisResults
            The estimation results instance.

        Raises
        ------
        TypeError
            If bandwidth is not a real number or None, force_int, diff or
            df_adjust is not a bool, or kernel or bandwidth_method is not a
            string.
        ValueError
            If bandwidth is negative, NaN or inf, kernel or bandwidth_method is
            not one of the options, or bandwidth_method is "andrews" and the
            AR(1) cannot be fitted to the residuals.
        """
        bandwidth, force_int, diff, df_adjust, bandwidth_method = _check_fit_arguments(
            bandwidth, force_int, diff, df_adjust, bandwidth_method
        )
        cov_est, eta, _ = self._common_fit(
            kernel, bandwidth, force_int, diff, bandwidth_method
        )
        omega = np.asarray(cov_est.cov.long_run)
        lmbda = np.asarray(cov_est.cov.one_sided)
        omega_12 = omega[:1, 1:]
        omega_22 = omega[1:, 1:]
        omega_22_inv = np.linalg.inv(omega_22)
        eta_2 = eta[:, 1:]
        y, x = np.asarray(self._y_df), np.asarray(self._x)
        y_dot = y[1:] - eta_2 @ omega_22_inv @ omega_12.T

        lmbda_12 = lmbda[:1, 1:]
        lmbda_22 = lmbda[1:, 1:]
        lmbda_12_dot = lmbda_12 - omega_12 @ omega_22_inv @ lmbda_22
        with_trend = add_trend(self._x, trend=self._trend)
        assert isinstance(with_trend, pd.DataFrame)
        z_df = with_trend
        z_df = z_df.iloc[1:]
        z = np.asarray(z_df)
        zpz = z.T @ z

        nobs, nvar = z.shape
        bias = np.zeros((nvar, 1))
        kx = x.shape[1]
        bias[:kx] = lmbda_12_dot.T
        zpydot = z.T @ y_dot - nobs * bias

        params = np.squeeze(np.linalg.solve(zpz, zpydot))
        omega_11 = omega[:1, :1]
        scale = 1.0 if not df_adjust else nobs / (nobs - nvar)
        omega_112 = scale * (omega_11 - omega_12 @ omega_22_inv @ omega_12.T)
        zpz_inv = np.linalg.inv(zpz)
        param_cov = omega_112 * zpz_inv
        cols = z_df.columns
        params_s = Series(params.squeeze(), index=cols, name="params")
        param_cov = pd.DataFrame(param_cov, columns=cols, index=cols)
        _check_finite_estimates(params_s, param_cov, "Fully Modified OLS")
        resid, r2, r2_adj = self._final_statistics(params_s)
        resid_kern = type(cov_est)(
            resid,
            bandwidth=cov_est.bandwidth,
            force_int=cov_est.force_int,
            bandwidth_method=cov_est.bandwidth_method,
        )
        return CointegrationAnalysisResults(
            params_s,
            param_cov,
            resid,
            omega_112[0, 0],
            resid_kern,
            kx,
            self._trend,
            df_adjust,
            r2,
            r2_adj,
            "Fully Modified OLS",
        )


@Substitution(method=CCR_METHOD, estimator=CCR_ESTIMATOR)
@Appender(COMMON_DOCSTRING)
class CanonicalCointegratingReg(FullyModifiedOLS):
    def __init__(
        self,
        y: ArrayLike1D,
        x: ArrayLike2D,
        trend: UnitRootTrend = "c",
        x_trend: UnitRootTrend | None = None,
    ) -> None:
        super().__init__(y, x, trend, x_trend)

    @Appender(FullyModifiedOLS.fit.__doc__)
    @_infeasible_on_error("Canonical Cointegrating Regression", "estimates")
    def fit(
        self,
        kernel: str = "bartlett",
        bandwidth: float | None = None,
        force_int: bool = True,
        diff: bool = False,
        df_adjust: bool = False,
        bandwidth_method: str = "newey-west",
    ) -> CointegrationAnalysisResults:
        bandwidth, force_int, diff, df_adjust, bandwidth_method = _check_fit_arguments(
            bandwidth, force_int, diff, df_adjust, bandwidth_method
        )
        cov_est, eta, beta = self._common_fit(
            kernel, bandwidth, force_int, diff, bandwidth_method
        )
        omega = np.asarray(cov_est.cov.long_run)
        lmbda = np.asarray(cov_est.cov.one_sided)
        sigma = np.asarray(cov_est.cov.short_run)

        lmbda2 = lmbda[:, 1:]
        sigma_inv = np.linalg.inv(sigma)
        y, x = np.asarray(self._y_df), np.asarray(self._x)
        x_star = x[1:] - eta @ (sigma_inv @ lmbda2)

        kx = x.shape[1]
        omega_12 = omega[:1, 1:]
        omega_22 = omega[1:, 1:]
        omega_22_inv = np.linalg.inv(omega_22)
        bias = np.zeros((kx + 1, 1))
        bias[1:] = omega_22_inv @ omega_12.T
        # K x K        K by 1
        #  K by 1
        y_star = y[1:] - eta @ (sigma_inv @ lmbda2 @ beta[:, None] + bias)
        z_star = add_trend(x_star, trend=self._trend)
        params = np.linalg.lstsq(z_star, y_star, rcond=None)[0]

        omega_11 = omega[:1, :1]
        nobs, nvar = z_star.shape
        scale = 1.0 if not df_adjust else nobs / (nobs - nvar)
        omega_112 = scale * (omega_11 - omega_12 @ omega_22_inv @ omega_12.T)
        param_cov = omega_112 * np.linalg.inv(z_star.T @ z_star)
        with_trend = add_trend(self._x.iloc[:10], self._trend)
        assert isinstance(with_trend, pd.DataFrame)
        cols = with_trend.columns
        params = Series(params.squeeze(), index=cols, name="params")
        param_cov = pd.DataFrame(param_cov, columns=cols, index=cols)
        _check_finite_estimates(params, param_cov, "Canonical Cointegrating Regression")
        resid, r2, r2_adj = self._final_statistics(params)
        resid_kern = type(cov_est)(
            resid,
            bandwidth=cov_est.bandwidth,
            force_int=cov_est.force_int,
            bandwidth_method=cov_est.bandwidth_method,
        )
        return CointegrationAnalysisResults(
            params,
            param_cov,
            resid,
            omega_112[0, 0],
            resid_kern,
            kx,
            self._trend,
            df_adjust,
            r2,
            r2_adj,
            "Canonical Cointegrating Regression",
        )
