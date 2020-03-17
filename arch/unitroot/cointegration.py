from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.iolib.summary import Summary, fmt_2cols, fmt_params
from statsmodels.iolib.table import SimpleTable
from statsmodels.regression.linear_model import OLS, RegressionResults

import arch.covariance.kernel as lrcov
from arch.typing import ArrayLike1D, ArrayLike2D, NDArray
from arch.unitroot.critical_values.engle_granger import (
    CV_PARAMETERS,
    LARGE_PARAMETERS,
    SMALL_PARAMETERS,
    TAU_MAX,
    TAU_MIN,
    TAU_STAR,
)
from arch.unitroot.unitroot import ADF, SHORT_TREND_DESCRIPTION, TREND_DESCRIPTION
from arch.utility.array import ensure1d, ensure2d
from arch.utility.io import pval_format, str_format
from arch.utility.timeseries import add_trend
from arch.vendor import cached_property

__all__ = [
    "engle_granger",
    "EngleGrangerCointegrationTestResult",
    "engle_granger_cv",
    "engle_granger_pval",
    "DynamicOLS",
    "DynamicOLSResults",
]

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


KERNEL_ESTIMATORS: Dict[str, Type[lrcov.CovarianceEstimator]] = {
    kernel.lower(): getattr(lrcov, kernel) for kernel in lrcov.KERNELS
}
KERNEL_ESTIMATORS.update({kernel: getattr(lrcov, kernel) for kernel in lrcov.KERNELS})


def _cross_section(y: ArrayLike1D, x: ArrayLike2D, trend: str) -> RegressionResults:
    if trend not in ("n", "c", "ct", "ctt"):
        raise ValueError('trend must be one of "n", "c", "ct" or "ctt"')
    y = ensure1d(y, "y", True)
    x = ensure2d(x, "x")

    if not isinstance(x, pd.DataFrame):
        cols = [f"x{i}" for i in range(1, x.shape[1] + 1)]
        x = pd.DataFrame(x, columns=cols, index=y.index)
    x = add_trend(x, trend)
    res = OLS(y, x).fit()
    return res


def engle_granger(
    y: ArrayLike1D,
    x: ArrayLike2D,
    trend: str = "c",
    *,
    lags: Optional[int] = None,
    max_lags: Optional[int] = None,
    method: str = "bic",
) -> "EngleGrangerCointegrationTestResult":
    r"""
    Test for cointegration within a set of time series.

    Parameters
    ----------
    y : array_like
        The left-hand-side variable in the cointegrating regression.
    x : array_like
        The right-hand-side variables in the cointegrating regression.
    trend : {"n","c","ct","ctt"}, default "c"
        Trend to include in the cointegrating regression. Trends are:

        * "n": No deterministic terms
        * "c": Constant
        * "ct": Constant and linear trend
        * "ctt": Constant, linear and quadratic trends
    lags : int, default None
        The number of lagged differences to include in the Augmented
        Dickey-Fuller test used on the residuals of the
    max_lags : int, default None
        The maximum number of lags to consider when using automatic
        lag-length in the Augmented Dickey-Fuller regression.
    method: {"aic", "bic", "tstat"}, default "bic"
        The method used to select the number of lags included in the
        Augmented Dickey-Fuller regression.

    Returns
    -------
    EngleGrangerCointegrationTestResult
        Results of the Engle-Granger test.

    See Also
    --------
    arch.unitroot.ADF
        Augmented Dickey-Fuller testing.

    Notes
    -----
    The model estimated is

    .. math::

       Y_t = X_t \beta + D_t \gamma + \epsilon_t

    where :math:`Z_t = [Y_t,X_t]` is being tested for cointegration.
    :math:`D_t` is a set of deterministic terms that may include a
    constant, a time trend or a quadratic time trend.

    The null hypothesis is that the series are not cointegrated.

    The test is implemented as an ADF of the estimated residuals from the
    cross-sectional regression using a set of critical values that is
    determined by the number of assumed stochastic trends when the null
    hypothesis is true.
    """
    x = ensure2d(x, "x")
    xsection = _cross_section(y, x, trend)
    resid = xsection.resid
    # Never pass in the trend here since only used in x-section
    adf = ADF(resid, lags, trend="n", max_lags=max_lags, method=method)
    stat = adf.stat
    nobs = resid.shape[0] - adf.lags - 1
    num_x = x.shape[1]
    cv = engle_granger_cv(trend, num_x, nobs)
    pv = engle_granger_pval(stat, trend, num_x)
    return EngleGrangerCointegrationTestResult(
        stat, pv, cv, order=num_x, adf=adf, xsection=xsection
    )


class CointegrationTestResult(object):
    """
    Base results class for cointegration tests.

    Parameters
    ----------
    stat : float
        The Engle-Granger test statistic.
    pvalue : float
        The pvalue of the Engle-Granger test statistic.
    crit_vals : Series
        The critical values of the Engle-Granger specific to the sample size
        and model dimension.
    null : str, default "No Cointegration"
        The null hypothesis.
    alternative : str, default "Cointegration"
        The alternative hypothesis.
    """

    def __init__(
        self,
        stat: float,
        pvalue: float,
        crit_vals: pd.Series,
        null: str = "No Cointegration",
        alternative: str = "Cointegration",
    ) -> None:
        self._stat = stat
        self._pvalue = pvalue
        self._crit_vals = crit_vals
        self._name = ""
        self._null = null
        self._alternative = alternative
        self._additional_info: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Sets or gets the name of the cointegration test"""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def stat(self) -> float:
        """The test statistic."""
        return self._stat

    @property
    def pvalue(self) -> float:
        """The p-value of the test statistic."""
        return self._pvalue

    @property
    def critical_values(self) -> pd.Series:
        """
        Critical Values

        Returns
        -------
        Series
            Series with three keys, 1, 5 and 10 containing the critical values
            of the test statistic.
        """
        return self._crit_vals

    @property
    def null_hypothesis(self) -> str:
        """The null hypothesis"""
        return self._null

    @property
    def alternative_hypothesis(self) -> str:
        """The alternative hypothesis"""
        return self._alternative

    def __str__(self) -> str:
        out = f"{self.name}\n"
        out += f"Statistic: {self.stat}\n"
        out += f"P-value: {self.pvalue}\n"
        out += f"Null: {self.null_hypothesis}, "
        out += f"Alternative: {self.alternative_hypothesis}"
        for key in self._additional_info:
            out += f"\n{key}: {self._additional_info[key]}"
        return out

    def __repr__(self) -> str:
        return self.__str__() + f"\nID: {hex(id(self))}"


class EngleGrangerCointegrationTestResult(CointegrationTestResult):
    """
    Results class for Engle-Granger cointegration tests.

    Parameters
    ----------
    stat : float
        The Engle-Granger test statistic.
    pvalue : float
        The pvalue of the Engle-Granger test statistic.
    crit_vals : Series
        The critical values of the Engle-Granger specific to the sample size
        and model dimension.
    null : str
        The null hypothesis.
    alternative : str
        The alternative hypothesis.
    trend : str
        The model's trend description.
    order : int
        The number of stochastic trends in the null distribution.
    adf : ADF
        The ADF instance used to perform the test and lag selection.
    xsection : RegressionResults
        The OLS results used in the cross-sectional regression.
    """

    def __init__(
        self,
        stat: float,
        pvalue: float,
        crit_vals: pd.Series,
        null: str = "No Cointegration",
        alternative: str = "Cointegration",
        trend: str = "c",
        order: int = 2,
        adf: Optional[ADF] = None,
        xsection: Optional[RegressionResults] = None,
    ) -> None:
        super().__init__(stat, pvalue, crit_vals, null, alternative)
        self.name = "Engle-Granger Cointegration Test"
        assert adf is not None
        self._adf = adf
        assert xsection is not None
        self._xsection = xsection
        self._order = order
        self._trend = trend
        self._additional_info = {
            "ADF Lag length": self.lags,
            "Trend": self.trend,
            "Estimated Root ρ (γ+1)": self.rho,
            "Distribution Order": self.distribution_order,
        }

    @property
    def trend(self) -> str:
        """The trend used in the cointegrating regression"""
        return self._trend

    @property
    def lags(self) -> int:
        """The number of lags used in the Augmented Dickey-Fuller regression."""
        return self._adf.lags

    @property
    def max_lags(self) -> Optional[int]:
        """The maximum number of lags used in the lag-length selection."""
        return self._adf.max_lags

    @property
    def rho(self) -> float:
        r"""
        The estimated coefficient in the Dickey-Fuller Test

        Returns
        -------
        float
            The coefficient.

        Notes
        -----
        The value returned is :math:`\hat{\rho}=\hat{\gamma}+1` from the ADF
        regression

        .. math::

            \Delta y_t = \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i}
                         + \epsilon_t
        """

        return 1 + self._adf.regression.params[0]

    @property
    def distribution_order(self) -> int:
        """The number of stochastic trends under the null hypothesis."""
        return self._order

    @property
    def cointegrating_vector(self) -> pd.Series:
        """
        The estimated cointegrating vector.
        """
        params = self._xsection.params
        index = [self._xsection.model.data.ynames]
        return pd.concat([pd.Series([1], index=index), -params], axis=0)

    @property
    def resid(self) -> pd.Series:
        """The residual from the cointegrating regression."""
        resid = self._xsection.resid
        resid.name = "Cointegrating Residual"
        return resid

    def plot(
        self, axes: Optional["plt.Axes"] = None, title: Optional[str] = None
    ) -> "plt.Figure":
        """
        Plot the cointegration residuals.

        Parameters
        ----------
        axes : Axes, default None
            Matplotlib axes instance to hold the figure.
        title : str, default None
            Title for the figure.

        Returns
        -------
        Figure
            The matplotlib Figure instance.
        """
        resid = self.resid
        df = pd.DataFrame(resid)
        title = "Cointegrating Residual" if title is None else title
        ax = df.plot(legend=False, ax=axes, title=title)
        ax.set_xlim(df.index.min(), df.index.max())
        fig = ax.get_figure()
        fig.tight_layout(pad=1.0)

        return fig

    def summary(self) -> Summary:
        """Summary of test, containing statistic, p-value and critical values"""
        table_data = [
            ("Test Statistic", f"{self.stat:0.3f}"),
            ("P-value", f"{self.pvalue:0.3f}"),
            ("ADF Lag length", f"{self.lags:d}"),
            ("Estimated Root ρ (γ+1)", f"{self.rho:0.3f}"),
        ]
        title = self.name

        table = SimpleTable(
            table_data,
            stubs=None,
            title=title,
            colwidths=18,
            datatypes=[0, 1],
            data_aligns=("l", "r"),
        )

        smry = Summary()
        smry.tables.append(table)

        cv_string = "Critical Values: "
        for val in self.critical_values.keys():
            p = str(int(val)) + "%"
            cv_string += f"{self.critical_values[val]:0.2f}"
            cv_string += " (" + p + ")"
            cv_string += ", "
        # Remove trailing ,<space>
        cv_string = cv_string[:-2]

        extra_text = [
            "Trend: " + TREND_DESCRIPTION[self._trend],
            cv_string,
            "Null Hypothesis: " + self.null_hypothesis,
            "Alternative Hypothesis: " + self.alternative_hypothesis,
            "Distribution Order: " + str(self.distribution_order),
        ]

        smry.add_extra_txt(extra_text)
        return smry

    def _repr_html_(self) -> str:
        """Display as HTML for IPython notebook."""
        return self.summary().as_html()


def engle_granger_cv(trend: str, num_x: int, nobs: int) -> pd.Series:
    """
    Critical Values for Engle-Granger t-tests

    Parameters
    ----------
    trend : {"n", "c", "ct", "ctt"}
        The trend included in the model
    num_x : The number of cross-sectional regressors in the model.
        Must be between 1 and 12.
    nobs : int
        The number of observations in the time series.

    Returns
    -------
    Series
        The critical values for 1, 5 and 10%
    """
    trends = ("n", "c", "ct", "ctt")
    if trend not in trends:
        valid = ",".join(trends)
        raise ValueError(f"Trend must by one of: {valid}")
    if not 1 <= num_x <= 12:
        raise ValueError(
            "The number of cross-sectional variables must be between 1 and "
            "12 (inclusive)"
        )
    tbl = CV_PARAMETERS[trend]

    crit_vals = {}
    for size in (10, 5, 1):
        params = tbl[size][num_x]
        x = 1.0 / (nobs ** np.arange(4.0))
        crit_vals[size] = x @ params
    return pd.Series(crit_vals)


def engle_granger_pval(stat: float, trend: str, num_x: int) -> float:
    """
    Asymptotic P-values for Engle-Granger t-tests

    Parameters
    ----------
    stat : float
        The test statistic
    trend : {"n", "c", "ct", "ctt"}
        The trend included in the model
    num_x : The number of cross-sectional regressors in the model.
        Must be between 1 and 12.

    Returns
    -------
    Series
        The critical values for 1, 5 and 10%
    """
    trends = ("n", "c", "ct", "ctt")
    if trend not in trends:
        valid = ",".join(trends)
        raise ValueError(f"Trend must by one of: {valid}")
    if not 1 <= num_x <= 12:
        raise ValueError(
            "The number of cross-sectional variables must be between 1 and "
            "12 (inclusive)"
        )
    key = (trend, num_x)
    if stat > TAU_MAX[key]:
        return 1.0
    elif stat < TAU_MIN[key]:
        return 0.0
    if stat > TAU_STAR[key]:
        params = np.array(LARGE_PARAMETERS[key])
    else:
        params = np.array(SMALL_PARAMETERS[key])
    order = params.shape[0]
    x = stat ** np.arange(order)
    return stats.norm().cdf(params @ x)


class DynamicOLSResults:
    """
    Estimation results for Dynamic OLS models

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
    reg_results : RegressionResults
        Regression results from fitting statsmodels OLS.
    df_adjust : bool
        Whether to degree of freedom adjust the estimator.
    """

    def __init__(
        self,
        params: pd.Series,
        cov: pd.DataFrame,
        resid: pd.Series,
        lags: int,
        leads: int,
        cov_type: str,
        kernel_est: lrcov.CovarianceEstimator,
        num_x: int,
        trend: str,
        reg_results: RegressionResults,
        df_adjust: bool,
    ) -> None:
        self._params = params
        self._cov = cov
        self._resid = resid
        self._bandwidth = kernel_est.bandwidth
        self._kernel = kernel_est.__class__.__name__
        self._kernel_est = kernel_est
        self._leads = leads
        self._lags = lags
        self._cov_type = cov_type
        self._num_x = num_x
        self._ci_size = params.shape[0] - num_x * (leads + lags + 1)
        self._trend = trend
        self._rsquared = reg_results.rsquared
        self._rsquared_adj = reg_results.rsquared_adj
        self._df_adjust = df_adjust

    @property
    def params(self) -> pd.Series:
        """The estimated parameters of the cointegrating vector"""
        return self._params.iloc[: self._ci_size]

    @cached_property
    def std_errors(self) -> pd.Series:
        """
        Standard errors  of the parameters in the cointegrating vector
        """
        se = np.sqrt(np.diag(self.cov))
        return pd.Series(se, index=self.params.index, name="std_errors")

    @cached_property
    def tvalues(self) -> pd.Series:
        """
        T-statistics of the parameters in the cointegrating vector
        """
        return pd.Series(self.params / self.std_errors, name="tvalues")

    @cached_property
    def pvalues(self) -> pd.Series:
        """
        P-value of the parameters in the cointegrating vector
        """
        return pd.Series(2 * (1 - stats.norm.cdf(np.abs(self.tvalues))), name="pvalues")

    @property
    def cov(self) -> pd.DataFrame:
        """The estimated parameter covariance of the cointegrating vector"""
        return self._cov.iloc[: self._ci_size, : self._ci_size]

    @property
    def full_params(self) -> pd.Series:
        """The complete set of parameters, including leads and lags"""
        return self._params

    @property
    def full_cov(self) -> pd.DataFrame:
        """
        Parameter covariance of the all model parameters, incl. leads and lags
        """
        return self._cov

    @property
    def resid(self) -> pd.Series:
        """The model residuals"""
        return self._resid

    @property
    def lags(self) -> int:
        """The number of lags included in the model"""
        return self._lags

    @property
    def leads(self) -> int:
        """The number of leads included in the model"""
        return self._leads

    @property
    def kernel(self) -> str:
        """The kernel used to estimate the covariance"""
        return self._kernel

    @property
    def bandwidth(self) -> float:
        """The bandwidth used in the parameter covariance estimation"""
        return self._bandwidth

    @property
    def cov_type(self) -> str:
        """The type of parameter covariance estimator used"""
        return self._cov_type

    @property
    def rsquared(self) -> float:
        """The model R²"""
        return self._rsquared

    @property
    def rsquared_adj(self) -> float:
        """The degree-of-freedom adjusted R²"""
        return self._rsquared_adj

    @cached_property
    def _cov_est(self) -> lrcov.CovarianceEstimate:
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
        nvar = self.full_params.shape[0]
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
        return self._df_scale * self._cov_est.short_run[0, 0]

    @property
    def long_run_variance(self) -> float:
        """
        The long-run variance of the regression residual.

        Returns
        -------
        float
            The estimated long-run variance of the residual.

        The long-run variance is estimated from the model residuals
        using the same kernel used to estimate the parameter
        covariance.

        If `df_adjust` is True, then the estimator is rescaled by T/(T-m) where
        m is the number of regressors in the model.
        """
        return self._df_scale * self._cov_est.long_run[0, 0]

    def summary(self, full: bool = False) -> Summary:
        """
        Summary of the model, containing estimated parameters and std. errors

        Parameters
        ----------
        full : bool, default False
            Flag indicating whether to include all estimated parameters
            (True) or only the parameters of the cointegrating vector

        Returns
        -------
        Summary
            A summary instance with method that support export to text, csv
            or latex.
        """
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
        # TODO: Need missing values
        top_right = [
            ("No. Observations:", str(self._resid.shape[0])),
            ("R²:", str_format(self.rsquared)),
            ("Adjusted. R²:", str_format(self.rsquared_adj)),
            ("Residual Variance:", str_format(self.residual_variance)),
            ("Long-run Variance:", str_format(self.long_run_variance)),
            ("", ""),
        ]
        smry = Summary()
        typ = "Cointegrating Vector" if not full else "Model"
        title = f"Dynamic OLS {typ} Summary"
        stubs = []
        vals = []
        for stub, val in top_left:
            stubs.append(stub)
            vals.append([val])
        table = SimpleTable(vals, txt_fmt=fmt_2cols, title=title, stubs=stubs)

        # Top Table
        # Parameter table
        fmt = fmt_2cols.copy()
        fmt["data_fmts"][1] = "%18s"

        top_right = [("%-21s" % ("  " + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))
        smry.tables.append(table)
        if full:
            params = np.asarray(self.full_params)
            stubs = list(self.full_params.index)
            se = np.sqrt(np.diag(self.full_cov))
            tstats = params / se
            pvalues = 2 * (1 - stats.norm.cdf(np.abs(tstats)))
        else:
            params = np.asarray(self.params)
            stubs = list(self.params.index)
            se = np.asarray(self.std_errors)
            tstats = np.asarray(self.tvalues)
            pvalues = np.asarray(self.pvalues)
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
        title = "Cointegrating Vector" if not full else "Model Parameters"
        header = ["Parameter", "Std. Err.", "T-stat", "P-value", "Lower CI", "Upper CI"]
        table = SimpleTable(
            data, stubs=stubs, txt_fmt=fmt_params, headers=header, title=title
        )
        smry.tables.append(table)

        return smry


class DynamicOLS(object):
    r"""
    Estimate Cointegrating Vector using Dynamic OLS

    Parameters
    ----------
    y : array_like
        The left-hand-side variable in the cointegrating regression.
    x : array_like
        The right-hand-side variables in the cointegrating regression.
    trend : {"n","c","ct","ctt"}, default "c"
        Trend to include in the cointegrating regression. Trends are:

        * "n": No deterministic terms
        * "c": Constant
        * "ct": Constant and linear trend
        * "ctt": Constant, linear and quadratic trends
    lags : int, default None
        The number of lags to include in the model.  If None, the optimal
        number of lags is chosen using method.
    leads : int, default None
        The number of leads to include in the model.  If None, the optimal
        number of leads is chosen using method.
    common : bool, default False
        Flag indicating that lags and leads should be restricted to the same
        value. When common is None, lags must equal leads and max_lag must
        equal max_lead.
    max_lag : int, default None
        The maximum lag to consider. See Notes for value used when None.
    max_lead : int, default None
        The maximum lead to consider. See Notes for value used when None.
    method : {"aic","bic","hqic"}, default "bic"
        The method used to select lag length when lags or leads is None.

        * "aic" - Akaike Information Criterion
        * "hqic" - Hannan-Quinn Information Criterion
        * "bic" - Schwartz/Bayesian Information Criterion

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
    """

    def __init__(
        self,
        y: ArrayLike1D,
        x: ArrayLike2D,
        trend: str = "c",
        lags: Optional[int] = None,
        leads: Optional[int] = None,
        common: bool = False,
        max_lag: Optional[int] = None,
        max_lead: Optional[int] = None,
        method: str = "bic",
    ) -> None:
        self._y = ensure1d(y, "y", True)
        assert isinstance(self._y, pd.Series)
        self._x = ensure2d(x, "x")
        self._trend = trend
        self._lags = lags
        self._leads = leads
        self._max_lag = max_lag
        self._max_lead = max_lead
        self._method = method
        self._common = bool(common)
        if not isinstance(self._x, pd.DataFrame):
            cols = [f"x_{i}" for i in range(1, self._x.shape[1] + 1)]
            self._x_df = pd.DataFrame(self._x, columns=cols, index=self._y.index)
        else:
            self._x_df = self._x
        self._y_df = pd.DataFrame(self._y)
        self._check_inputs()

    def _check_inputs(self) -> None:
        """Validate the inputs"""
        if not isinstance(self._method, str) or self._method.lower() not in (
            "aic",
            "bic",
            "hqic",
        ):
            raise ValueError('method must be one of "aic", "bic", or "hqic"')
        if self._trend not in ("n", "c", "ct", "ctt"):
            raise ValueError('trend must of be one of "n","c","ct", or "ctt"')
        max_lag = self._max_lag
        self._max_lag = int(max_lag) if max_lag is not None else max_lag
        max_lead = self._max_lead
        self._max_lead = int(max_lead) if max_lead is not None else max_lead
        self._leads = int(self._leads) if self._leads is not None else self._leads
        self._lags = int(self._lags) if self._lags is not None else self._lags

        if self._common and self._leads != self._lags:
            raise ValueError(
                "common is specified but leads and lags have different values"
            )
        if self._common and self._max_lead != self._max_lag:
            raise ValueError(
                "common is specified but max_lead and max_lag have different values"
            )

    def _format_variables(
        self, leads: int, lags: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Format the variables for the regression"""
        x = self._x_df
        y = self._y_df
        delta_x = x.diff()
        data = [y, x]

        for lag in range(-lags, leads + 1):
            lag_data = delta_x.shift(-lag)
            typ = "LAG" if lag < 0 else "LEAD"
            lag_data.columns = [f"D.{c}.{typ}{abs(lag)}" for c in lag_data.columns]
            if lag == 0:
                lag_data.columns = [f"D.{c}" for c in lag_data.columns]
            data.append(lag_data)
        data_df: pd.DataFrame = pd.concat(data, axis=1).dropna()
        lhs, rhs = data_df.iloc[:, :1], data_df.iloc[:, 1:]
        nrhs = rhs.shape[1]
        rhs = add_trend(rhs, trend=self._trend, prepend=True)
        ntrend = rhs.shape[1] - nrhs
        if ntrend:
            nx = x.shape[1]
            trend = rhs.iloc[:, :ntrend]
            rhs = pd.concat(
                [rhs.iloc[:, ntrend : ntrend + nx], trend, rhs.iloc[:, ntrend + nx :]],
                axis=1,
            )
        return lhs, rhs

    def _ic(self, resids: NDArray, nparam: int) -> float:
        """Compute an info criterion"""
        nobs = resids.shape[0]
        sigma2 = resids.T @ resids / nobs
        if self._method == "aic":
            penalty = 2
        elif self._method == "hqic":
            penalty = 2 * np.log(np.log(nobs))
        else:  # bic
            penalty = np.log(nobs)
        return np.log(sigma2) + nparam * penalty / nobs

    def _leads_and_lags(self) -> Tuple[int, int]:
        """Select the optimal number of leads and lags"""
        if self._lags is not None and self._leads is not None:
            return self._leads, self._lags
        nobs = self._y.shape[0]
        max_lead_lag = int(np.ceil(12.0 * (nobs / 100) ** (1 / 4)))
        # TODO: Make sure max lags is estimable
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
        nx = self._x_df.shape[1]
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

    def fit(
        self,
        cov_type: str = "unadjusted",
        kernel: str = "bartlett",
        bandwidth: Optional[int] = None,
        force_int: bool = False,
        df_adjust: bool = False,
    ) -> DynamicOLSResults:
        r"""
        Estimate the Dynamic OLS regression

        Parameters
        ----------
        cov_type : str, default "unadjusted"
            Either "unadjusted" (or is equivalent "homoskedastic") or "robust"
            (or its equivalent "kernel").
        kernel : str, default "bartlett"
            The string name of any of any known kernel-based long-run
            covariance estimators. Common choices are "bartlett" for the
            Bartlett kernel (Newey-West), "parzen" for the Parzen kernel
            and "quadratic-spectral" for Quadratic Spectral kernel.
        bandwidth : int, default None
            The bandwidth to use. If not provided, the optimal bandwidth is
            estimated from the data. Setting the bandwidth to 0 and using
            "unadjusted" produces the classic OLS covariance estimator.
            Setting the bandwidth to 0 and using "robust" produces White's
            covariance estimator.
        force_int : bool, default False
            Whether the force the estimated optimal bandwidth to be an integer.
        df_adjust : bool, default False
            Whether the adjust the parameter covariance to account for the
            number of parameters estimated in the regression. If true, the
            parameter covariance estimator is multiplied by T/(T-k) where
            k is the number of regressors in the model.

        Returns
        -------
        DynamicOLSResults
            The estimation results.

        See Also
        --------
        arch.unitroot.cointegration.engle_granger
            Cointegration testing using the Engle-Granger methodology
        statsmodels.regression.linear_model.OLS
            Ordinal Least Squares regression.

        Notes
        -----
        When using the unadjusted covariance, the parameter covariance is
        estimated as

        .. math::

            T^{-1} \hat{\sigma}^2_{HAC} \hat{\Sigma}_{ZZ}^{-1}

        where :math:`\hat{\sigma}^2_{HAC}` is an estimator of the long-run
        variance of the regression error and
        :math:`\hat{\Sigma}_{ZZ}=T^{-1}Z'Z`. :math:`Z_t` is a vector the
        includes all terms in the regression (i.e., determinstics,
        cross-sectional, leads and lags) When using the robust covariance,
        the parameter covariance is estimated as

        .. math::

            T^{-1} \hat{\Sigma}_{ZZ}^{-1} \hat{S}_{HAC} \hat{\Sigma}_{ZZ}^{-1}

        where :math:`\hat{S}_{HAC}` is a Heteroskedasticity-Autocorrelation
        Consistent estimator of the covariance of the regression scores
        :math:`Z_t\epsilon_t`.
        """
        leads, lags = self._leads_and_lags()
        # TODO: Rank check and drop??
        lhs, rhs = self._format_variables(leads, lags)
        mod = OLS(lhs, rhs)
        res = mod.fit()
        coeffs = np.asarray(res.params)
        resid = lhs.squeeze() - (rhs @ coeffs).squeeze()
        resid.name = "resid"
        cov, est = self._cov(
            cov_type, kernel, bandwidth, force_int, df_adjust, rhs, resid
        )
        params = pd.Series(np.squeeze(coeffs), index=rhs.columns, name="params")
        num_x = self._x_df.shape[1]
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
        cov_type: str,
        kernel: str,
        bandwidth: Optional[int],
        force_int: bool,
        df_adjust: bool,
        rhs: pd.DataFrame,
        resids: pd.Series,
    ) -> Tuple[pd.DataFrame, lrcov.CovarianceEstimator]:
        """Estimate the covariance"""
        x = np.asarray(rhs)
        eps = ensure2d(np.asarray(resids), "eps")
        nobs, nx = x.shape
        sigma_xx = x.T @ x / nobs
        sigma_xx_inv = np.linalg.inv(sigma_xx)
        kernel = kernel.lower().replace("-", "")
        if kernel not in KERNEL_ESTIMATORS:
            estimators = "\n".join(sorted([k for k in KERNEL_ESTIMATORS]))
            raise ValueError(
                f"kernel is not a known kernel estimator. Must be one of:\n {estimators}"
            )
        kernel_est = KERNEL_ESTIMATORS[kernel]
        scale = nobs / (nobs - nx) if df_adjust else 1.0
        if cov_type in ("unadjusted", "homoskedastic"):
            est = kernel_est(eps, bandwidth, center=False, force_int=force_int)
            sigma2 = np.squeeze(est.cov.long_run)
            cov = (scale * sigma2) * sigma_xx_inv / nobs
        elif cov_type in ("robust", "kernel"):
            scores = x * eps
            est = kernel_est(scores, bandwidth, center=False, force_int=force_int)
            s = est.cov.long_run
            cov = scale * sigma_xx_inv @ s @ sigma_xx_inv / nobs
        else:
            raise ValueError("Unknown cov_type")
        cov_df = pd.DataFrame(cov, columns=rhs.columns, index=rhs.columns)
        return cov_df, est
