from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union, cast
import warnings

from numpy import (
    abs,
    amax,
    amin,
    any as npany,
    arange,
    argwhere,
    array,
    asarray,
    ceil,
    cumsum,
    diag,
    diff,
    empty,
    full,
    hstack,
    inf,
    interp,
    isnan,
    log,
    nan,
    ndarray,
    ones,
    pi,
    polyval,
    power,
    sort,
    sqrt,
    squeeze,
    sum as npsum,
)
from numpy.linalg import LinAlgError, inv, lstsq, matrix_rank, pinv, qr, solve
from pandas import DataFrame
from scipy.stats import norm
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.regression.linear_model import OLS, RegressionResults
from statsmodels.tsa.tsatools import lagmat

from arch.typing import (
    ArrayLike,
    ArrayLike1D,
    ArrayLike2D,
    Float64Array,
    Literal,
    UnitRootTrend,
)
from arch.unitroot.critical_values.dfgls import (
    dfgls_cv_approx,
    dfgls_large_p,
    dfgls_small_p,
    dfgls_tau_max,
    dfgls_tau_min,
    dfgls_tau_star,
)
from arch.unitroot.critical_values.dickey_fuller import (
    adf_z_cv_approx,
    adf_z_large_p,
    adf_z_max,
    adf_z_min,
    adf_z_small_p,
    adf_z_star,
    tau_2010,
    tau_large_p,
    tau_max,
    tau_min,
    tau_small_p,
    tau_star,
)
from arch.unitroot.critical_values.kpss import kpss_critical_values
from arch.unitroot.critical_values.zivot_andrews import za_critical_values
from arch.utility import cov_nw
from arch.utility.array import AbstractDocStringInheritor, ensure1d, ensure2d
from arch.utility.exceptions import (
    InfeasibleTestException,
    InvalidLengthWarning,
    PerformanceWarning,
    invalid_length_doc,
)
from arch.utility.timeseries import add_trend

__all__ = [
    "ADF",
    "DFGLS",
    "PhillipsPerron",
    "KPSS",
    "VarianceRatio",
    "kpss_crit",
    "mackinnoncrit",
    "mackinnonp",
    "ZivotAndrews",
    "auto_bandwidth",
    "TREND_DESCRIPTION",
    "SHORT_TREND_DESCRIPTION",
]

TREND_MAP = {None: "n", 0: "c", 1: "ct", 2: "ctt"}

TREND_DESCRIPTION = {
    "n": "No Trend",
    "c": "Constant",
    "ct": "Constant and Linear Time Trend",
    "ctt": "Constant, Linear and Quadratic Time Trends",
    "t": "Linear Time Trend (No Constant)",
}

SHORT_TREND_DESCRIPTION = {
    "n": "No Trend",
    "c": "Constant",
    "ct": "Const and Linear Trend",
    "ctt": "Const., Lin. and Quad. Trends",
    "t": "Linear Time Trend (No Const.)",
}


def _is_reduced_rank(
    x: Union[Float64Array, DataFrame]
) -> tuple[bool, Union[int, None]]:
    """
    Check if a matrix has reduced rank preferring quick checks
    """
    if x.shape[1] > x.shape[0]:
        return True, None
    elif npany(isnan(x)):
        return True, None
    elif sum(amax(x, axis=0) == amin(x, axis=0)) > 1:
        return True, None
    else:
        x_rank = matrix_rank(x)
        return x_rank < x.shape[1], x_rank


def _select_best_ic(
    method: Literal["aic", "bic", "t-stat"],
    nobs: float,
    sigma2: Float64Array,
    tstat: Float64Array,
) -> tuple[float, int]:
    """
    Computes the best information criteria

    Parameters
    ----------
    method : {"aic", "bic", "t-stat"}
        Method to use when finding the lag length
    nobs : float
        Number of observations in time series
    sigma2 : ndarray
        maxlag + 1 array containing MLE estimates of the residual variance
    tstat : ndarray
        maxlag + 1 array containing t-statistic values. Only used if method
        is "t-stat"

    Returns
    -------
    icbest : float
        Minimum value of the information criteria
    lag : int
        The lag length that maximizes the information criterion.
    """
    llf = -nobs / 2.0 * (log(2 * pi) + log(sigma2) + 1)
    maxlag = len(sigma2) - 1
    if method == "aic":
        crit = -2 * llf + 2 * arange(float(maxlag + 1))
        icbest, lag = min(zip(crit, arange(maxlag + 1)))
    elif method == "bic":
        crit = -2 * llf + log(nobs) * arange(float(maxlag + 1))
        icbest, lag = min(zip(crit, arange(maxlag + 1)))
    elif method == "t-stat":
        stop = 1.6448536269514722
        large_tstat = abs(tstat) >= stop
        lag = int(squeeze(max(argwhere(large_tstat))))
        icbest = float(tstat[lag])
    else:
        raise ValueError("Unknown method")

    return icbest, lag


singular_array_error: str = """\
The maximum lag you are considering ({max_lags}) results in an ADF regression with a
singular regressor matrix after including {lag} lags, and so a specification test be
run. This may occur if your series have little variation and so is locally constant,
or may occur if you are attempting to test a very short series. You can manually set
maximum lag length to consider smaller models.\
"""


def _autolag_ols_low_memory(
    y: Float64Array,
    maxlag: int,
    trend: UnitRootTrend,
    method: Literal["aic", "bic", "t-stat"],
) -> tuple[float, int]:
    """
    Computes the lag length that minimizes an info criterion .

    Parameters
    ----------
    y : ndarray
        Variable being tested for a unit root
    maxlag : int
        The highest lag order for lag length selection.
    trend : {"n", "c", "ct", "ctt"}
        Trend in the model
    method : {"aic", "bic", "t-stat"}
        Method to use when finding the lag length

    Returns
    -------
    icbest : float
        Minimum value of the information criteria
    lag : int
        The lag length that maximizes the information criterion.

    Notes
    -----
    Minimizes creation of large arrays. Uses approx 6 * nobs temporary values
    """
    y = asarray(y)
    lower_method = method.lower()
    deltay = diff(y)
    deltay = deltay / sqrt(deltay @ deltay)
    lhs = deltay[maxlag:][:, None]
    level = y[maxlag:-1]
    level = level / sqrt(level @ level)
    trendx: list[Float64Array] = []
    nobs = lhs.shape[0]
    if trend == "n":
        trendx.append(empty((nobs, 0)))
    else:
        if "tt" in trend:
            tt = arange(1, nobs + 1, dtype=float)[:, None] ** 2
            tt *= sqrt(5) / float(nobs) ** (5 / 2)
            trendx.append(tt)
        if "t" in trend:
            t = arange(1, nobs + 1, dtype=float)[:, None]
            t *= sqrt(3) / float(nobs) ** (3 / 2)
            trendx.append(t)
        if trend.startswith("c"):
            trendx.append(ones((nobs, 1)) / sqrt(nobs))
    rhs = hstack([level[:, None], hstack(trendx)])
    m = rhs.shape[1]
    xpx = full((m + maxlag, m + maxlag), nan)
    xpy = full((m + maxlag, 1), nan)
    assert isinstance(xpx, ndarray)
    assert isinstance(xpy, ndarray)
    xpy[:m] = rhs.T @ lhs
    xpx[:m, :m] = rhs.T @ rhs
    for i in range(maxlag):
        x1 = deltay[maxlag - i - 1 : -(1 + i)]
        block = rhs.T @ x1
        xpx[m + i, :m] = block
        xpx[:m, m + i] = block
        xpy[m + i] = x1 @ lhs
        for j in range(i, maxlag):
            x2 = deltay[maxlag - j - 1 : -(1 + j)]
            x1px2 = x1 @ x2
            xpx[m + i, m + j] = x1px2
            xpx[m + j, m + i] = x1px2
    ypy = lhs.T @ lhs
    sigma2 = empty(maxlag + 1)

    tstat = empty(maxlag + 1)
    tstat[0] = inf
    for i in range(m, m + maxlag + 1):
        xpx_sub = xpx[:i, :i]
        try:
            b = solve(xpx_sub, xpy[:i])
        except LinAlgError:
            raise InfeasibleTestException(
                singular_array_error.format(max_lags=maxlag, lag=m - i)
            )
        sigma2[i - m] = squeeze(ypy - b.T @ xpx_sub @ b) / nobs
        if lower_method == "t-stat":
            xpxi = inv(xpx_sub)
            stderr = sqrt(sigma2[i - m] * xpxi[-1, -1])
            tstat[i - m] = squeeze(b[-1]) / stderr

    return _select_best_ic(method, nobs, sigma2, tstat)


def _autolag_ols(
    endog: ArrayLike1D,
    exog: ArrayLike2D,
    startlag: int,
    maxlag: int,
    method: Literal["aic", "bic", "t-stat"],
) -> tuple[float, int]:
    """
    Returns the results for the lag length that maximizes the info criterion.

    Parameters
    ----------
    endog : {ndarray, Series}
        nobs array containing endogenous variable
    exog : {ndarray, DataFrame}
        nobs by (startlag + maxlag) array containing lags and possibly other
        variables
    startlag : int
        The first zero-indexed column to hold a lag.  See Notes.
    maxlag : int
        The highest lag order for lag length selection.
    method : {"aic", "bic", "t-stat"}

        * aic - Akaike Information Criterion
        * bic - Bayes Information Criterion
        * t-stat - Based on last lag

    Returns
    -------
    icbest : float
        Minimum value of the information criteria
    lag : int
        The lag length that maximizes the information criterion.

    Notes
    -----
    Does estimation like mod(endog, exog[:,:i]).fit()
    where i goes from lagstart to lagstart + maxlag + 1.  Therefore, lags are
    assumed to be in contiguous columns from low to high lag length with
    the highest lag in the last column.
    """
    lower_method = method.lower()
    exog_singular, exog_rank = _is_reduced_rank(exog)
    if exog_singular:
        if exog_rank is None:
            exog_rank = matrix_rank(exog)
        raise InfeasibleTestException(
            singular_array_error.format(
                max_lags=maxlag, lag=max(exog_rank - startlag, 0)
            )
        )
    q, r = qr(exog)
    qpy = q.T @ endog
    ypy = endog.T @ endog
    xpx = exog.T @ exog

    sigma2 = empty(maxlag + 1)
    tstat = empty(maxlag + 1)
    nobs = float(endog.shape[0])
    tstat[0] = inf
    for i in range(startlag, startlag + maxlag + 1):
        b = solve(r[:i, :i], qpy[:i])
        sigma2[i - startlag] = squeeze(ypy - b.T @ xpx[:i, :i] @ b) / nobs
        if lower_method == "t-stat" and i > startlag:
            xpxi = inv(xpx[:i, :i])
            stderr = sqrt(sigma2[i - startlag] * xpxi[-1, -1])
            tstat[i - startlag] = squeeze(b[-1]) / stderr

    return _select_best_ic(method, nobs, sigma2, tstat)


def _df_select_lags(
    y: Float64Array,
    trend: Literal["n", "c", "ct", "ctt"],
    max_lags: Optional[int],
    method: Literal["aic", "bic", "t-stat"],
    low_memory: bool = False,
) -> tuple[float, int]:
    """
    Helper method to determine the best lag length in DF-like regressions

    Parameters
    ----------
    y : ndarray
        The data for the lag selection exercise
    trend : {"n","c","ct","ctt"}
        The trend order
    max_lags : int
        The maximum number of lags to check.  This setting affects all
        estimation since the sample is adjusted by max_lags when
        fitting the models
    method : {"aic", "bic", "t-stat"}
        The method to use when estimating the model
    low_memory : bool
        Flag indicating whether to use the low-memory algorithm for
        lag-length selection.

    Returns
    -------
    best_ic : float
        The information criteria at the selected lag
    best_lag : int
        The selected lag

    Notes
    -----
    If max_lags is None, the default value of 12 * (nobs/100)**(1/4) is used.
    """
    nobs = y.shape[0]
    # This is the absolute maximum number of lags possible,
    # only needed to very short time series.
    max_max_lags = max((nobs - 1) // 2 - 1, 0)
    if trend != "n":
        max_max_lags -= len(trend)
    if max_lags is None:
        max_lags = int(ceil(12.0 * power(nobs / 100.0, 1 / 4.0)))
        max_lags = max(min(max_lags, max_max_lags), 0)
        if max_lags > 119:
            warnings.warn(
                "The value of max_lags was not specified and has been calculated as "
                f"{max_lags}. Searching over a large lag length with a sample size "
                f"of {nobs} is likely to be slow. Consider directly setting "
                "``max_lags`` to a small value to avoid this performance issue.",
                PerformanceWarning,
            )
    assert max_lags is not None
    if low_memory:
        out = _autolag_ols_low_memory(y, max_lags, trend, method)
        return out
    delta_y = diff(y)
    rhs = lagmat(delta_y[:, None], max_lags, trim="both", original="in")
    nobs = rhs.shape[0]
    rhs[:, 0] = y[-nobs - 1 : -1]  # replace 0 with level of y
    lhs = delta_y[-nobs:]

    if trend != "n":
        full_rhs = add_trend(rhs, trend, prepend=True)
    else:
        full_rhs = rhs

    start_lag = full_rhs.shape[1] - rhs.shape[1] + 1
    ic_best, best_lag = _autolag_ols(lhs, full_rhs, start_lag, max_lags, method)
    return ic_best, best_lag


def _add_column_names(rhs: Float64Array, lags: int) -> DataFrame:
    """Return a DataFrame with named columns"""
    lag_names = [f"Diff.L{i}" for i in range(1, lags + 1)]
    return DataFrame(rhs, columns=["Level.L1"] + lag_names)


def _estimate_df_regression(
    y: Float64Array, trend: Literal["n", "c", "ct", "ctt"], lags: int
) -> RegressionResults:
    """Helper function that estimates the core (A)DF regression

    Parameters
    ----------
    y : ndarray
        The data for the lag selection
    trend : {"n","c","ct","ctt"}
        The trend order
    lags : int
        The number of lags to include in the ADF regression

    Returns
    -------
    ols_res : OLSResults
        A results class object produced by OLS.fit()

    Notes
    -----
    See statsmodels.regression.linear_model.OLS for details on the results
    returned
    """
    delta_y = diff(y)

    rhs = lagmat(delta_y[:, None], lags, trim="both", original="in")
    nobs = rhs.shape[0]
    lhs = rhs[:, 0].copy()  # lag-0 values are lhs, Is copy() necessary?
    rhs[:, 0] = y[-nobs - 1 : -1]  # replace lag 0 with level of y
    rhs = _add_column_names(rhs, lags)

    if trend != "n":
        rhs = add_trend(rhs.iloc[:, : lags + 1], trend)

    return OLS(lhs, rhs).fit()


class UnitRootTest(metaclass=ABCMeta):
    """Base class to be used for inheritance in unit root bootstrap"""

    def __init__(
        self,
        y: ArrayLike,
        lags: Optional[int],
        trend: Union[UnitRootTrend, Literal["t"]],
        valid_trends: Union[list[str], tuple[str, ...]],
    ) -> None:
        self._y = ensure1d(y, "y", series=False)
        self._delta_y = diff(y)
        self._nobs = self._y.shape[0]
        self._lags = int(lags) if lags is not None else lags
        if self._lags is not None and self._lags < 0:
            raise ValueError("lags must be non-negative.")
        self._valid_trends = list(valid_trends)
        if trend not in self.valid_trends:
            raise ValueError("trend not understood")
        self._trend = trend
        self._stat: Optional[float] = None
        self._critical_values: dict[str, float] = {}
        self._pvalue: Optional[float] = None
        self._null_hypothesis = "The process contains a unit root."
        self._alternative_hypothesis = "The process is weakly stationary."
        self._test_name = ""
        self._title = ""
        self._summary_text: list[str] = []

    def __str__(self) -> str:
        return self.summary().__str__()

    def __repr__(self) -> str:
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self) -> str:
        """Display as HTML for IPython notebook."""
        return self.summary().as_html()

    @abstractmethod
    def _check_specification(self) -> None:
        """
        Check that the data are compatible with running a test.
        """

    @abstractmethod
    def _compute_statistic(self) -> None:
        """This is the core routine that computes the test statistic, computes
        the p-value and constructs the critical values.
        """

    def _compute_if_needed(self) -> None:
        """Checks whether the statistic needs to be computed, and computed if
        needed
        """
        if self._stat is None:
            self._check_specification()
            self._compute_statistic()

    @property
    def null_hypothesis(self) -> str:
        """The null hypothesis"""
        return self._null_hypothesis

    @property
    def alternative_hypothesis(self) -> str:
        """The alternative hypothesis"""
        return self._alternative_hypothesis

    @property
    def nobs(self) -> int:
        """The number of observations used when computing the test statistic.
        Accounts for loss of data due to lags for regression-based bootstrap."""
        return self._nobs

    @property
    def valid_trends(self) -> list[str]:
        """List of valid trend terms."""
        return self._valid_trends

    @property
    def pvalue(self) -> float:
        """Returns the p-value for the test statistic"""
        self._compute_if_needed()
        assert self._pvalue is not None
        return self._pvalue

    @property
    def stat(self) -> float:
        """The test statistic for a unit root"""
        self._compute_if_needed()
        assert self._stat is not None
        return self._stat

    @property
    def critical_values(self) -> dict[str, float]:
        """Dictionary containing critical values specific to the test, number of
        observations and included deterministic trend terms.
        """
        self._compute_if_needed()
        return self._critical_values

    def summary(self) -> Summary:
        """Summary of test, containing statistic, p-value and critical values"""
        table_data = [
            ("Test Statistic", f"{self.stat:0.3f}"),
            ("P-value", f"{self.pvalue:0.3f}"),
            ("Lags", f"{self.lags:d}"),
        ]
        title = self._title

        if not title:
            title = self._test_name + " Results"
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
        cv = self._critical_values.keys()
        cv_numeric = array([float(x.split("%")[0]) for x in cv])
        cv_numeric = sort(cv_numeric)
        for val in cv_numeric:
            p = str(int(val)) + "%"
            cv_string += f"{self._critical_values[p]:0.2f}"
            cv_string += " (" + p + ")"
            if val != cv_numeric[-1]:
                cv_string += ", "

        extra_text = [
            "Trend: " + TREND_DESCRIPTION[self._trend],
            cv_string,
            "Null Hypothesis: " + self.null_hypothesis,
            "Alternative Hypothesis: " + self.alternative_hypothesis,
        ]

        smry.add_extra_txt(extra_text)
        if self._summary_text:
            smry.add_extra_txt(self._summary_text)
        return smry

    @property
    def lags(self) -> int:
        """Sets or gets the number of lags used in the model.
        When bootstrap use DF-type regressions, lags is the number of lags in the
        regression model.  When bootstrap use long-run variance estimators, lags
        is the number of lags used in the long-run variance estimator.
        """
        self._compute_if_needed()
        assert self._lags is not None
        return self._lags

    @property
    def y(self) -> ArrayLike:
        """Returns the data used in the test statistic"""
        return self._y

    @property
    def trend(self) -> str:
        """Sets or gets the deterministic trend term used in the test. See
        valid_trends for a list of supported trends
        """
        return self._trend


class ADF(UnitRootTest, metaclass=AbstractDocStringInheritor):
    """
    Augmented Dickey-Fuller unit root test

    Parameters
    ----------
    y : {ndarray, Series}
        The data to test for a unit root
    lags : int, optional
        The number of lags to use in the ADF regression.  If omitted or None,
        `method` is used to automatically select the lag length with no more
        than `max_lags` are included.
    trend : {"n", "c", "ct", "ctt"}, optional
        The trend component to include in the test

        - "n" - No trend components
        - "c" - Include a constant (Default)
        - "ct" - Include a constant and linear time trend
        - "ctt" - Include a constant and linear and quadratic time trends

    max_lags : int, optional
        The maximum number of lags to use when selecting lag length
    method : {"AIC", "BIC", "t-stat"}, optional
        The method to use when selecting the lag length

        - "AIC" - Select the minimum of the Akaike IC
        - "BIC" - Select the minimum of the Schwarz/Bayesian IC
        - "t-stat" - Select the minimum of the Schwarz/Bayesian IC

    low_memory : bool
        Flag indicating whether to use a low memory implementation of the
        lag selection algorithm. The low memory algorithm is slower than
        the standard algorithm but will use 2-4% of the memory required for
        the standard algorithm. This options allows automatic lag selection
        to be used in very long time series. If None, use automatic selection
        of algorithm.

    Notes
    -----
    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then the null cannot be rejected that there
    and the series appears to be a unit root.

    The p-values are obtained through regression surface approximation from
    MacKinnon (1994) using the updated 2010 tables.
    If the p-value is close to significant, then the critical values should be
    used to judge whether to reject the null.

    The autolag option and maxlag for it are described in Greene [1]_.
    See Hamilton [2]_ for more on ADF tests. Critical value simulation based
    on MacKinnon [3]_ abd [4]_.

    Examples
    --------
    >>> from arch.unitroot import ADF
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load().data
    >>> inflation = np.diff(np.log(data["cpi"]))
    >>> adf = ADF(inflation)
    >>> print(f"{adf.stat:0.4f}")
    -3.0931
    >>> print(f"{adf.pvalue:0.4f}")
    0.0271
    >>> adf.lags
    2
    >>> adf.trend="ct"
    >>> print(f"{adf.stat:0.4f}")
    -3.2111
    >>> print(f"{adf.pvalue:0.4f}")
    0.0822

    References
    ----------
    .. [1] Greene, W. H. 2011. Econometric Analysis. Prentice Hall: Upper
       Saddle River, New Jersey.

    .. [2] Hamilton, J. D. 1994. Time Series Analysis. Princeton: Princeton
       University Press.

    .. [3] MacKinnon, J.G. 1994.  "Approximate asymptotic distribution
       functions for unit-root and cointegration bootstrap.  `Journal of
       Business and Economic Statistics` 12, 167-76.

    .. [4] MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."
       Queen's University, Dept of Economics, Working Papers.  Available at
       https://ideas.repec.org/p/qed/wpaper/1227.html
    """

    def __init__(
        self,
        y: ArrayLike,
        lags: Optional[int] = None,
        trend: UnitRootTrend = "c",
        max_lags: Optional[int] = None,
        method: Literal["aic", "bic", "t-stat"] = "aic",
        low_memory: Optional[bool] = None,
    ) -> None:
        valid_trends = ("n", "c", "ct", "ctt")
        super().__init__(y, lags, trend, valid_trends)
        self._max_lags = max_lags
        self._method = method
        self._test_name = "Augmented Dickey-Fuller"
        self._regression = None
        self._low_memory = bool(low_memory)
        if low_memory is None:
            self._low_memory = True if self.y.shape[0] > 1e5 else False

    def _select_lag(self) -> None:
        ic_best, best_lag = _df_select_lags(
            self._y,
            cast(UnitRootTrend, self._trend),
            self._max_lags,
            self._method,
            low_memory=self._low_memory,
        )
        self._ic_best = ic_best
        self._lags = best_lag

    def _check_specification(self) -> None:
        trend_order = len(self._trend) if self._trend not in ("n", "nc") else 0
        lag_len = 0 if self._lags is None else self._lags
        required = 3 + trend_order + lag_len
        if self._y.shape[0] < required:
            raise InfeasibleTestException(
                f"A minimum of {required} observations are needed to run an ADF with "
                f"trend {self.trend} and the user-specified number of lags."
            )

    def _compute_statistic(self) -> None:
        if self._lags is None:
            self._select_lag()
        assert self._lags is not None
        y, trend, lags = self._y, self._trend, self._lags
        resols = _estimate_df_regression(y, cast(UnitRootTrend, trend), lags)
        self._regression = resols
        (self._stat, *_) = (stat, *_) = resols.tvalues
        self._nobs = int(resols.nobs)
        self._pvalue = mackinnonp(
            stat,
            regression=cast(Literal["n", "c", "ct", "ctt"], trend),
            num_unit_roots=1,
        )
        critical_values = mackinnoncrit(
            num_unit_roots=1,
            regression=cast(Literal["n", "c", "ct", "ctt"], trend),
            nobs=resols.nobs,
        )
        self._critical_values = {
            "1%": critical_values[0],
            "5%": critical_values[1],
            "10%": critical_values[2],
        }

    @property
    def regression(self) -> RegressionResults:
        """Returns the OLS regression results from the ADF model estimated"""
        self._compute_if_needed()
        return self._regression

    @property
    def max_lags(self) -> Union[int, None]:
        """Sets or gets the maximum lags used when automatically selecting lag
        length"""
        return self._max_lags


class DFGLS(UnitRootTest, metaclass=AbstractDocStringInheritor):
    """
    Elliott, Rothenberg and Stock's ([ers]_) GLS detrended Dickey-Fuller

    Parameters
    ----------
    y : {ndarray, Series}
        The data to test for a unit root
    lags : int, optional
        The number of lags to use in the ADF regression.  If omitted or None,
        `method` is used to automatically select the lag length with no more
        than `max_lags` are included.
    trend : {"c", "ct"}, optional
        The trend component to include in the test

        - "c" - Include a constant (Default)
        - "ct" - Include a constant and linear time trend

    max_lags : int, optional
        The maximum number of lags to use when selecting lag length. When using
        automatic lag length selection, the lag is selected using OLS
        detrending rather than GLS detrending ([pq]_).
    method : {"AIC", "BIC", "t-stat"}, optional
        The method to use when selecting the lag length

        - "AIC" - Select the minimum of the Akaike IC
        - "BIC" - Select the minimum of the Schwarz/Bayesian IC
        - "t-stat" - Select the minimum of the Schwarz/Bayesian IC

    Notes
    -----
    The null hypothesis of the Dickey-Fuller GLS is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then the null cannot be rejected and the series
    appears to be a unit root.

    DFGLS differs from the ADF test in that an initial GLS detrending step
    is used before a trend-less ADF regression is run.

    Critical values and p-values when trend is "c" are identical to
    the ADF.  When trend is set to "ct", they are from novel simulations.

    Examples
    --------
    >>> from arch.unitroot import DFGLS
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load().data
    >>> inflation = np.diff(np.log(data["cpi"]))
    >>> dfgls = DFGLS(inflation)
    >>> print(f"{dfgls.stat:0.4f}")
    -2.7611
    >>> print(f"{dfgls.pvalue:0.4f}")
    0.0059
    >>> dfgls.lags
    2
    >>> dfgls = DFGLS(inflation, trend = "ct")
    >>> print(f"{dfgls.stat:0.4f}")
    -2.9036
    >>> print(f"{dfgls.pvalue:0.4f}")
    0.0447

    References
    ----------
    .. [ers] Elliott, G. R., T. J. Rothenberg, and J. H. Stock. 1996. Efficient
           bootstrap for an autoregressive unit root. Econometrica 64: 813-836
    .. [pq] Perron, P., & Qu, Z. (2007). A simple modification to improve the
           finite sample properties of Ng and Perron's unit root tests.
           Economics letters, 94(1), 12-19.
    """

    def __init__(
        self,
        y: ArrayLike,
        lags: Optional[int] = None,
        trend: Literal["c", "ct"] = "c",
        max_lags: Optional[int] = None,
        method: Literal["aic", "bic", "t-stat"] = "aic",
        low_memory: Optional[bool] = None,
    ) -> None:
        valid_trends = ("c", "ct")
        super().__init__(y, lags, trend, valid_trends)
        self._max_lags = max_lags
        self._method = method
        self._regression = None
        self._low_memory = low_memory
        if low_memory is None:
            self._low_memory = True if self.y.shape[0] >= 1e5 else False
        self._test_name = "Dickey-Fuller GLS"
        if trend == "c":
            self._c = -7.0
        else:
            self._c = -13.5

    def _check_specification(self) -> None:
        trend_order = len(self._trend)
        lag_len = 0 if self._lags is None else self._lags
        required = 3 + trend_order + lag_len
        if self._y.shape[0] < required:
            raise InfeasibleTestException(
                f"A minimum of {required} observations are needed to run an ADF with "
                f"trend {self.trend} and the user-specified number of lags."
            )

    def _compute_statistic(self) -> None:
        """Core routine to estimate DF-GLS test statistic"""
        # 1. GLS detrend
        trend, c = self._trend, self._c

        nobs = self._y.shape[0]
        ct = c / nobs
        z = add_trend(nobs=nobs, trend=trend)

        delta_z = z.copy()
        delta_z[1:, :] = delta_z[1:, :] - (1 + ct) * delta_z[:-1, :]
        delta_y = self._y.copy()[:, None]
        delta_y[1:] = delta_y[1:] - (1 + ct) * delta_y[:-1]
        detrend_coef = pinv(delta_z) @ delta_y
        y = self._y
        y_detrended = y - (z @ detrend_coef).ravel()

        # 2. determine lag length, if needed
        if self._lags is None:
            max_lags, method = self._max_lags, self._method
            assert self._low_memory is not None
            self._lags = ADF(self._y, method=method, max_lags=max_lags).lags
            ols_detrend_coef = lstsq(z, y, rcond=None)[0]
            y_ols_detrend = y - z @ ols_detrend_coef
            icbest, bestlag = _df_select_lags(
                y_ols_detrend, "n", max_lags, method, low_memory=self._low_memory
            )
            self._lags = bestlag

        # 3. Run Regression
        lags = self._lags

        resols = _estimate_df_regression(y_detrended, lags=lags, trend="n")
        self._regression = resols
        self._nobs = int(resols.nobs)
        self._stat, *_ = resols.tvalues
        assert self._stat is not None
        self._pvalue = mackinnonp(
            self._stat, regression=cast(Literal["c", "ct"], trend), dist_type="dfgls"
        )
        critical_values = mackinnoncrit(
            regression=cast(Literal["c", "ct"], trend),
            nobs=self._nobs,
            dist_type="dfgls",
        )
        self._critical_values = {
            "1%": critical_values[0],
            "5%": critical_values[1],
            "10%": critical_values[2],
        }

    @property
    def trend(self) -> str:
        return self._trend

    @property
    def regression(self) -> RegressionResults:
        """Returns the OLS regression results from the ADF model estimated"""
        self._compute_if_needed()
        return self._regression

    @property
    def max_lags(self) -> Union[int, None]:
        """Sets or gets the maximum lags used when automatically selecting lag
        length"""
        return self._max_lags


class PhillipsPerron(UnitRootTest, metaclass=AbstractDocStringInheritor):
    """
    Phillips-Perron unit root test

    Parameters
    ----------
    y : {ndarray, Series}
        The data to test for a unit root
    lags : int, optional
        The number of lags to use in the Newey-West estimator of the long-run
        covariance.  If omitted or None, the lag length is set automatically to
        12 * (nobs/100) ** (1/4)
    trend : {"n", "c", "ct"}, optional
        The trend component to include in the test

        - "n" - No trend components
        - "c" - Include a constant (Default)
        - "ct" - Include a constant and linear time trend

    test_type : {"tau", "rho"}
        The test to use when computing the test statistic. "tau" is based on
        the t-stat and "rho" uses a test based on nobs times the re-centered
        regression coefficient

    Notes
    -----
    The null hypothesis of the Phillips-Perron (PP) test is that there is a
    unit root, with the alternative that there is no unit root. If the pvalue
    is above a critical size, then the null cannot be rejected that there
    and the series appears to be a unit root.

    Unlike the ADF test, the regression estimated includes only one lag of
    the dependant variable, in addition to trend terms. Any serial
    correlation in the regression errors is accounted for using a long-run
    variance estimator (currently Newey-West).

    See Philips and Perron for details [3]_. See Hamilton [1]_ for more on
    PP tests. Newey and West contains information about long-run variance
    estimation [2]_. The p-values are obtained through regression surface
    approximation using the mathodology of MacKinnon [4]_ and [5]_, only
    using many more simulations.

    If the p-value is close to significant, then the critical values should be
    used to judge whether to reject the null.

    Examples
    --------
    >>> from arch.unitroot import PhillipsPerron
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load().data
    >>> inflation = np.diff(np.log(data["cpi"]))
    >>> pp = PhillipsPerron(inflation)
    >>> print(f"{pp.stat:0.4f}")
    -8.1356
    >>> print(f"{pp.pvalue:0.4f}")
    0.0000
    >>> pp.lags
    15
    >>> pp.trend = "ct"
    >>> print(f"{pp.stat:0.4f}")
    -8.2022
    >>> print(f"{pp.pvalue:0.4f}")
    0.0000
    >>> pp.test_type = "rho"
    >>> print(f"{pp.stat:0.4f}")
    -120.3271
    >>> print(f"{pp.pvalue:0.4f}")
    0.0000

    References
    ----------
    .. [1] Hamilton, J. D. 1994. Time Series Analysis. Princeton: Princeton
           University Press.

    .. [2] Newey, W. K., and K. D. West. 1987. "A simple, positive
           semidefinite, heteroskedasticity and autocorrelation consistent covariance
           matrix". Econometrica 55, 703-708.

    .. [3] Phillips, P. C. B., and P. Perron. 1988. "Testing for a unit root in
           time series regression". Biometrika 75, 335-346.

    .. [4] MacKinnon, J.G. 1994.  "Approximate asymptotic distribution
           functions for unit-root and cointegration bootstrap".  Journal of
           Business and  Economic Statistics. 12, 167-76.

    .. [5] MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."
           Queen's University, Dept of Economics, Working Papers.  Available at
           https://ideas.repec.org/p/qed/wpaper/1227.html
    """

    def __init__(
        self,
        y: ArrayLike,
        lags: Optional[int] = None,
        trend: Literal["n", "c", "ct"] = "c",
        test_type: Literal["tau", "rho"] = "tau",
    ) -> None:
        valid_trends = ("n", "c", "ct")
        super().__init__(y, lags, trend, valid_trends)
        self._test_type = test_type
        self._stat_rho = None
        self._stat_tau = None
        self._test_name = "Phillips-Perron Test"
        self._lags = lags
        self._regression = None

    def _check_specification(self) -> None:
        trend_order = len(self._trend) if self._trend not in ("n", "nc") else 0
        lag_len = 0 if self._lags is None else self._lags
        required = max(3 + trend_order, lag_len)
        if self._y.shape[0] < required:
            raise InfeasibleTestException(
                f"A minimum of {required} observations are needed to run an ADF with "
                f"trend {self.trend} and the user-specified number of lags."
            )

    def _compute_statistic(self) -> None:
        """Core routine to estimate PP test statistics"""
        # 1. Estimate Regression
        y, trend = self._y, self._trend
        nobs = y.shape[0]

        if self._lags is None:
            self._lags = int(ceil(12.0 * power(nobs / 100.0, 1 / 4.0)))
        lags = self._lags

        rhs = asarray(y, dtype=float)[:-1, None]
        rhs_df = _add_column_names(rhs, 0)
        lhs = y[1:, None]
        if trend != "n":
            rhs_df = add_trend(rhs_df, trend)

        mod = OLS(lhs, rhs_df)
        resols = mod.fit()
        self._regression = mod.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        k = rhs_df.shape[1]
        n, u = resols.nobs, resols.resid
        if u.shape[0] < lags:
            raise InfeasibleTestException(
                f"The number of observations {u.shape[0]} is less than the number of"
                f"lags in the long-run covariance estimator, {lags}. You must have "
                "lags <= nobs."
            )
        lam2 = cov_nw(u, lags, demean=False)
        lam = sqrt(lam2)
        # 2. Compute components
        s2 = u @ u / (n - k)
        s = sqrt(s2)
        gamma0 = s2 * (n - k) / n
        sigma, *_ = resols.bse
        sigma2 = sigma**2.0
        if sigma <= 0:
            raise InfeasibleTestException(
                "The estimated variance of the coefficient in the Phillips-Perron "
                "regression is 0. This may occur if the series contains constant "
                "values or the residual variance in the regression is 0."
            )
        rho, *_ = resols.params
        # 3. Compute statistics
        self._stat_tau = sqrt(gamma0 / lam2) * ((rho - 1) / sigma) - 0.5 * (
            (lam2 - gamma0) / lam
        ) * (n * sigma / s)
        self._stat_rho = n * (rho - 1) - 0.5 * (n**2.0 * sigma2 / s2) * (lam2 - gamma0)

        self._nobs = int(resols.nobs)
        if self._test_type == "rho":
            self._stat = self._stat_rho
            dist_type = "adf-z"
        else:
            self._stat = self._stat_tau
            dist_type = "adf-t"
        assert self._stat is not None
        self._pvalue = mackinnonp(self._stat, regression=trend, dist_type=dist_type)
        critical_values = mackinnoncrit(regression=trend, nobs=n, dist_type=dist_type)
        self._critical_values = {
            "1%": critical_values[0],
            "5%": critical_values[1],
            "10%": critical_values[2],
        }

        self._title = self._test_name + " (Z-" + self._test_type + ")"

    @property
    def test_type(self) -> str:
        """
        Gets or sets the test type returned by stat.
        Valid values are "tau" or "rho"
        """
        return self._test_type

    @property
    def regression(self) -> RegressionResults:
        """
        Returns OLS regression results for the specification used in the test

        The results returned use a Newey-West covariance matrix with the same
        number of lags as are used in the test statistic.
        """
        self._compute_if_needed()
        return self._regression


class KPSS(UnitRootTest, metaclass=AbstractDocStringInheritor):
    """
    Kwiatkowski, Phillips, Schmidt and Shin (KPSS) stationarity test

    Parameters
    ----------
    y : {ndarray, Series}
        The data to test for stationarity
    lags : int, optional
        The number of lags to use in the Newey-West estimator of the long-run
        covariance.  If omitted or None, the number of lags is calculated
        with the data-dependent method of Hobijn et al. (1998). See also
        Andrews (1991), Newey & West (1994), and Schwert (1989).
        Set lags=-1 to use the old method that only depends on the sample
        size, 12 * (nobs/100) ** (1/4).
    trend : {"c", "ct"}, optional
        The trend component to include in the ADF test
            "c" - Include a constant (Default)
            "ct" - Include a constant and linear time trend

    Notes
    -----
    The null hypothesis of the KPSS test is that the series is weakly
    stationary and the alternative is that it is non-stationary.
    If the p-value is above a critical size, then the null cannot be
    rejected that there and the series appears stationary.

    The p-values and critical values were computed using an extensive
    simulation based on 100,000,000 replications using series with 2,000
    observations. See [3]_ for the initial description of the KPSS test.
    Further details are available in [2]_ and [5]_. Details about the long-run
    covariance estimation can be found in [1]_ and [4]_.

    Examples
    --------
    >>> from arch.unitroot import KPSS
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load().data
    >>> inflation = np.diff(np.log(data["cpi"]))
    >>> kpss = KPSS(inflation)
    >>> print(f"{kpss.stat:0.4f}")
    0.2870
    >>> print(f"{kpss.pvalue:0.4f}")
    0.1473
    >>> kpss.trend = "ct"
    >>> print(f"{kpss.stat:0.4f}")
    0.2075
    >>> print(f"{kpss.pvalue:0.4f}")
    0.0128

    References
    ----------
    .. [1] Andrews, D.W.K. (1991). "Heteroskedasticity and autocorrelation
           consistent covariance matrix estimation". Econometrica, 59: 817-858.

    .. [2] Hobijn, B., Frances, B.H., & Ooms, M. (2004). Generalizations
           of the KPSS-test for stationarity. Statistica Neerlandica, 52: 483-502.

    .. [3] Kwiatkowski, D.; Phillips, P. C. B.; Schmidt, P.; Shin, Y. (1992).
           "Testing the null hypothesis of stationarity against the alternative of
           a unit root". Journal of Econometrics 54 (1-3), 159-178

    .. [4] Newey, W.K., & West, K.D. (1994). "Automatic lag selection in
           covariance matrix estimation". Review of Economic Studies, 61: 631-653.

    .. [5] Schwert, G. W. (1989). "Tests for unit roots: A Monte Carlo
           investigation". Journal of Business and Economic Statistics, 7 (2):
           147-159.
    """

    def __init__(
        self, y: ArrayLike, lags: Optional[int] = None, trend: Literal["c", "ct"] = "c"
    ) -> None:
        valid_trends = ("c", "ct")
        if lags is None:
            warnings.warn(
                "Lag selection has changed to use a data-dependent method. To use the "
                "old method that only depends on time, set lags=-1",
                DeprecationWarning,
            )
        self._legacy_lag_selection = False
        if lags == -1:
            self._legacy_lag_selection = True
            lags = None
        super().__init__(y, lags, trend, valid_trends)
        self._test_name = "KPSS Stationarity Test"
        self._null_hypothesis = "The process is weakly stationary."
        self._alternative_hypothesis = "The process contains a unit root."
        self._resids: Union[ArrayLike1D, None] = None

    def _check_specification(self) -> None:
        trend_order = len(self._trend)
        lag_len = 0 if self._lags is None else self._lags
        required = max(1 + trend_order, lag_len)
        if self._y.shape[0] < required:
            raise InfeasibleTestException(
                f"A minimum of {required} observations are needed to run an ADF with "
                f"trend {self.trend} and the user-specified number of lags."
            )

    def _compute_statistic(self) -> None:
        # 1. Estimate model with trend
        nobs, y, trend = self._nobs, self._y, self._trend
        z = add_trend(nobs=nobs, trend=trend)
        res = OLS(y, z).fit()
        # 2. Compute KPSS test
        self._resids = u = res.resid
        if self._lags is None:
            if self._legacy_lag_selection:
                self._lags = int(ceil(12.0 * power(nobs / 100.0, 1 / 4.0)))
            else:
                self._autolag()
        assert self._lags is not None
        if u.shape[0] < self._lags:
            raise InfeasibleTestException(
                f"The number of observations {u.shape[0]} is less than the number of"
                f"lags in the long-run covariance estimator, {self._lags}. You must "
                "have lags <= nobs."
            )
        lam = cov_nw(u, self._lags, demean=False)
        s = cumsum(u)
        self._stat = 1 / (nobs**2.0) * (s**2.0).sum() / lam
        self._nobs = u.shape[0]
        assert self._stat is not None
        if trend == "c":
            lit_trend: Literal["c", "ct"] = "c"
        else:
            lit_trend = "ct"
        self._pvalue, critical_values = kpss_crit(self._stat, lit_trend)
        self._critical_values = {
            "1%": critical_values[0],
            "5%": critical_values[1],
            "10%": critical_values[2],
        }

    def _autolag(self) -> None:
        """
        Computes the number of lags for covariance matrix estimation in KPSS
        test using method of Hobijn et al (1998). See also Andrews (1991),
        Newey & West (1994), and Schwert (1989). Assumes Bartlett / Newey-West
        kernel.

        Written by Jim Varanelli
        """
        resids = self._resids
        assert resids is not None
        covlags = int(power(self._nobs, 2.0 / 9.0))
        s0 = sum(resids**2) / self._nobs
        s1 = 0
        for i in range(1, covlags + 1):
            resids_prod = resids[i:] @ resids[: self._nobs - i]
            resids_prod /= self._nobs / 2
            s0 += resids_prod
            s1 += i * resids_prod
        if s0 <= 0:
            raise InfeasibleTestException(
                "Residuals are all zero and so automatic bandwidth selection cannot "
                "be used. This is usually an indication that the series being testes "
                "is too small or have constant values."
            )
        s_hat = s1 / s0
        pwr = 1.0 / 3.0
        gamma_hat = 1.1447 * power(s_hat * s_hat, pwr)
        autolags = amin([self._nobs, int(gamma_hat * power(self._nobs, pwr))])
        self._lags = int(autolags)


class ZivotAndrews(UnitRootTest, metaclass=AbstractDocStringInheritor):
    """
    Zivot-Andrews structural-break unit-root test

    The Zivot-Andrews test can be used to test for a unit root in a
    univariate process in the presence of serial correlation and a
    single structural break.

    Parameters
    ----------
    y : array_like
        data series
    lags : int, optional
        The number of lags to use in the ADF regression.  If omitted or None,
        `method` is used to automatically select the lag length with no more
        than `max_lags` are included.
    trend : {"c", "t", "ct"}, optional
        The trend component to include in the test

        - "c" - Include a constant (Default)
        - "t" - Include a linear time trend
        - "ct" - Include a constant and linear time trend

    trim : float
        percentage of series at begin/end to exclude from break-period
        calculation in range [0, 0.333] (default=0.15)
    max_lags : int, optional
        The maximum number of lags to use when selecting lag length
    method : {"AIC", "BIC", "t-stat"}, optional
        The method to use when selecting the lag length

        - "AIC" - Select the minimum of the Akaike IC
        - "BIC" - Select the minimum of the Schwarz/Bayesian IC
        - "t-stat" - Select the minimum of the Schwarz/Bayesian IC

    Notes
    -----
    H0 = unit root with a single structural break

    Algorithm follows Baum (2004/2015) approximation to original
    Zivot-Andrews method. Rather than performing an autolag regression at
    each candidate break period (as per the original paper), a single
    autolag regression is run up-front on the base model (constant + trend
    with no dummies) to determine the best lag length. This lag length is
    then used for all subsequent break-period regressions. This results in
    significant run time reduction but also slightly more pessimistic test
    statistics than the original Zivot-Andrews method,

    No attempt has been made to characterize the size/power trade-off.

    Based on the description in Zivot and Andrews [3]_. See [2]_ for
    a general discussion of unit root tests. Code tested against Baum [1]_.

    References
    ----------
    .. [1] Baum, C.F. (2004). ZANDREWS: Stata module to calculate Zivot-Andrews
           unit root test in presence of structural break," Statistical Software
           Components S437301, Boston College Department of Economics, revised
           2015.

    .. [2] Schwert, G.W. (1989). Tests for unit roots: A Monte Carlo
           investigation. Journal of Business & Economic Statistics, 7: 147-159.

    .. [3] Zivot, E., and Andrews, D.W.K. (1992). Further evidence on the great
           crash, the oil-price shock, and the unit-root hypothesis. Journal of
           Business & Economic Studies, 10: 251-270.
    """

    def __init__(
        self,
        y: ArrayLike,
        lags: Optional[int] = None,
        trend: Literal["c", "ct", "t"] = "c",
        trim: float = 0.15,
        max_lags: Optional[int] = None,
        method: Literal["aic", "bic", "t-stat"] = "aic",
    ) -> None:
        super().__init__(y, lags, trend, ("c", "t", "ct"))
        if not isinstance(trim, float) or not 0 <= trim <= (1 / 3):
            raise ValueError("trim must be a float in range [0, 1/3]")
        self._trim = trim
        self._max_lags = max_lags
        self._method = method
        self._test_name = "Zivot-Andrews"
        self._all_stats = full(self._y.shape[0], nan)
        self._null_hypothesis = (
            "The process contains a unit root with a single structural break."
        )
        self._alternative_hypothesis = "The process is trend and break stationary."

    @staticmethod
    def _quick_ols(endog: Float64Array, exog: Float64Array) -> Float64Array:
        """
        Minimal implementation of LS estimator for internal use
        """
        xpxi = inv(exog.T @ exog)
        xpy = exog.T @ endog
        nobs, k_exog = exog.shape
        b = xpxi @ xpy
        e = endog - exog @ b
        sigma2 = e.T @ e / (nobs - k_exog)
        return b / sqrt(diag(sigma2 * xpxi))

    def _check_specification(self) -> None:
        trend_order = len(self._trend)
        lag_len = 0 if self._lags is None else self._lags
        required = 3 + trend_order + lag_len
        if self._y.shape[0] < required:
            raise InfeasibleTestException(
                f"A minimum of {required} observations are needed to run an ADF with "
                f"trend {self.trend} and the user-specified number of lags."
            )

    def _compute_statistic(self) -> None:
        """This is the core routine that computes the test statistic, computes
        the p-value and constructs the critical values.
        """
        trim = self._trim
        trend = self._trend

        y = self._y
        y_2d = ensure2d(y, "y")
        nobs = y_2d.shape[0]

        if self._lags is not None:
            baselags = self._lags
        else:
            adf = ADF(self._y, max_lags=self._max_lags, trend="ct", method=self._method)
            self._lags = baselags = adf.lags

        trimcnt = int(nobs * trim)
        start_period = trimcnt
        end_period = nobs - trimcnt
        if trend == "ct":
            basecols = 5
        else:
            basecols = 4
        # first-diff y and standardize for numerical stability
        dy = diff(y_2d, axis=0)[:, 0]
        dy /= sqrt(dy.T @ dy)
        y_2d = y_2d / sqrt(y_2d.T @ y_2d)
        # reserve exog space
        exog = empty((dy[baselags:].shape[0], basecols + baselags))
        # normalize constant for stability in long time series
        c_const = 1 / sqrt(nobs)  # Normalize
        exog[:, 0] = c_const
        # lagged y and dy
        exog[:, basecols - 1] = y_2d[baselags : (nobs - 1), 0]
        exog[:, basecols:] = lagmat(dy, baselags, trim="none")[
            baselags : exog.shape[0] + baselags
        ]
        # better time trend: t_const @ t_const = 1 for large nobs
        t_const = arange(1.0, nobs + 2)
        t_const *= sqrt(3) / nobs ** (3 / 2)
        # iterate through the time periods
        stats = full(end_period + 1, inf)
        for bp in range(start_period + 1, end_period + 1):
            # update intercept dummy / trend / trend dummy
            cutoff = bp - (baselags + 1)
            if cutoff <= 0:
                raise InfeasibleTestException(
                    f"The number of observations is too small to use the Zivot-Andrews "
                    f"test with trend {trend} and {self._lags} lags."
                )
            if trend != "t":
                exog[:cutoff, 1] = 0
                exog[cutoff:, 1] = c_const
                exog[:, 2] = t_const[(baselags + 2) : (nobs + 1)]
                if trend == "ct":
                    exog[:cutoff, 3] = 0
                    exog[cutoff:, 3] = t_const[1 : (nobs - bp + 1)]
            else:
                exog[:, 1] = t_const[(baselags + 2) : (nobs + 1)]
                exog[: (cutoff - 1), 2] = 0
                exog[(cutoff - 1) :, 2] = t_const[0 : (nobs - bp + 1)]
            # check exog rank on first iteration
            if bp == start_period + 1:
                rank = matrix_rank(exog)
                if rank < exog.shape[1]:
                    raise InfeasibleTestException(
                        f"The regressor matrix is singular. The can happen if the data "
                        "contains regions of constant observations, if the number of "
                        f"lags ({self._lags}) is too large, or if the series is very "
                        "short."
                    )
            stats[bp] = self._quick_ols(dy[baselags:], exog)[basecols - 1]
        # return best seen
        self._all_stats[start_period + 1 : end_period + 1] = stats[
            start_period + 1 : end_period + 1
        ]
        self._stat = float(amin(stats))
        self._cv_interpolate()

    def _cv_interpolate(self) -> None:
        """
        Linear interpolation for Zivot-Andrews p-values and critical values

        Notes
        -----
        The p-values are linearly interpolated from the quantiles of the
        simulated ZA test statistic distribution
        """
        table = za_critical_values[self._trend]
        y = table[:, 0]
        x = table[:, 1]
        # ZA cv table contains quantiles multiplied by 100
        self._pvalue = float(interp(self.stat, x, y)) / 100.0
        cv = [1.0, 5.0, 10.0]
        crit_value = interp(cv, y, x)
        self._critical_values = {
            "1%": crit_value[0],
            "5%": crit_value[1],
            "10%": crit_value[2],
        }


class VarianceRatio(UnitRootTest, metaclass=AbstractDocStringInheritor):
    """
    Variance Ratio test of a random walk.

    Parameters
    ----------
    y : {ndarray, Series}
        The data to test for a random walk
    lags : int
        The number of periods to used in the multi-period variance, which is
        the numerator of the test statistic.  Must be at least 2
    trend : {"n", "c"}, optional
        "c" allows for a non-zero drift in the random walk, while "n" requires
        that the increments to y are mean 0
    overlap : bool, optional
        Indicates whether to use all overlapping blocks.  Default is True.  If
        False, the number of observations in y minus 1 must be an exact
        multiple of lags.  If this condition is not satisfied, some values at
        the end of y will be discarded.
    robust : bool, optional
        Indicates whether to use heteroskedasticity robust inference. Default
        is True.
    debiased : bool, optional
        Indicates whether to use a debiased version of the test. Default is
        True. Only applicable if overlap is True.

    Notes
    -----
    The null hypothesis of a VR is that the process is a random walk, possibly
    plus drift.  Rejection of the null with a positive test statistic
    indicates the presence of positive serial correlation in the time series.
    See [1]_ for details about variance ratio testing.

    Examples
    --------
    >>> from arch.unitroot import VarianceRatio
    >>> import pandas_datareader as pdr
    >>> data = pdr.get_data_fred("DJIA", start="2010-1-1", end="2020-12-31")
    >>> data = np.log(data.resample("M").last())  # End of month
    >>> vr = VarianceRatio(data, lags=12)
    >>> print(f"{vr.pvalue:0.4f}")
    0.1370

    References
    ----------
    .. [1] Campbell, John Y., Lo, Andrew W. and MacKinlay, A. Craig. (1997) The
       Econometrics of Financial Markets. Princeton, NJ: Princeton University
       Press.
    """

    def __init__(
        self,
        y: ArrayLike,
        lags: int = 2,
        trend: Literal["n", "c"] = "c",
        debiased: bool = True,
        robust: bool = True,
        overlap: bool = True,
    ) -> None:
        if lags < 2:
            raise ValueError("lags must be an integer larger than 2")
        valid_trends = ("n", "c")
        super().__init__(y, lags, trend, valid_trends)
        self._test_name = "Variance-Ratio Test"
        self._null_hypothesis = "The process is a random walk."
        self._alternative_hypothesis = "The process is not a random walk."
        self._robust = robust
        self._debiased = debiased
        self._overlap = overlap
        self._vr: Optional[float] = None
        self._stat_variance: Optional[float] = None
        quantiles = array([0.01, 0.05, 0.1, 0.9, 0.95, 0.99])
        for q, cv in zip(quantiles, norm.ppf(quantiles)):
            self._critical_values[str(int(100 * q)) + "%"] = cv

    @property
    def vr(self) -> float:
        """The ratio of the long block lags-period variance
        to the 1-period variance"""
        self._compute_if_needed()
        assert self._vr is not None
        return self._vr

    @property
    def overlap(self) -> bool:
        """Sets of gets the indicator to use overlapping returns in the
        long-period variance estimator"""
        return self._overlap

    @property
    def robust(self) -> bool:
        """Sets of gets the indicator to use a heteroskedasticity robust
        variance estimator"""
        return self._robust

    @property
    def debiased(self) -> bool:
        """Sets of gets the indicator to use debiased variances in the ratio"""
        return self._debiased

    def _check_specification(self) -> None:
        assert self._lags is not None
        lags = self._lags
        required = 2 * lags if not self._overlap else lags + 1 + int(self.debiased)
        if self._y.shape[0] < required:
            raise InfeasibleTestException(
                f"A minimum of {required} observations are needed to run an ADF with "
                f"trend {self.trend} and the user-specified number of lags."
            )

    def _compute_statistic(self) -> None:
        overlap, debiased, robust = self._overlap, self._debiased, self._robust
        y, nobs, q, trend = self._y, self._nobs, self._lags, self._trend
        assert q is not None
        nq = nobs - 1
        if not overlap:
            # Check length of y
            if nq % q != 0:
                extra = nq % q
                y = y[:-extra]
                warnings.warn(
                    invalid_length_doc.format(var="y", block=q, drop=extra),
                    InvalidLengthWarning,
                )

        nobs = y.shape[0]
        if trend == "n":
            mu = 0
        else:
            mu = (y[-1] - y[0]) / (nobs - 1)

        delta_y = diff(y)
        nq = delta_y.shape[0]
        sigma2_1 = sum((delta_y - mu) ** 2.0) / nq

        if not overlap:
            delta_y_q = y[q::q] - y[0:-q:q]
            sigma2_q = sum((delta_y_q - q * mu) ** 2.0) / nq
            self._summary_text = ["Computed with non-overlapping blocks"]
        else:
            delta_y_q = y[q:] - y[:-q]
            sigma2_q = sum((delta_y_q - q * mu) ** 2.0) / (nq * q)
            self._summary_text = ["Computed with overlapping blocks"]

        if debiased and overlap:
            sigma2_1 *= nq / (nq - 1)
            m = q * (nq - q + 1) * (1 - (q / nq))
            sigma2_q *= (nq * q) / m
            self._summary_text = ["Computed with overlapping blocks (de-biased)"]

        if not overlap:
            self._stat_variance = 2.0 * (q - 1)
        elif not robust:
            # GH 286, CLM 2.4.39
            self._stat_variance = (2 * (2 * q - 1) * (q - 1)) / (3 * q)
        else:
            z2 = (delta_y - mu) ** 2.0
            scale = sum(z2) ** 2.0
            theta = 0.0
            for k in range(1, q):
                delta = nq * z2[k:] @ z2[:-k] / scale
                # GH 286, CLM 2.4.43
                theta += 4 * (1 - k / q) ** 2.0 * delta
            self._stat_variance = theta
        self._vr = sigma2_q / sigma2_1
        assert self._vr is not None

        self._stat = sqrt(nq) * (self._vr - 1) / sqrt(self._stat_variance)
        assert self._stat is not None
        abs_stat = float(abs(self._stat))
        self._pvalue = 2 - 2 * norm.cdf(abs(abs_stat))


def mackinnonp(
    stat: float,
    regression: Literal["c", "n", "ct", "ctt"] = "c",
    num_unit_roots: int = 1,
    dist_type: Literal["adf-t", "adf-z", "dfgls"] = "adf-t",
) -> float:
    """
    Returns MacKinnon's approximate p-value for test stat.

    Parameters
    ----------
    stat : float
        "T-value" from an Augmented Dickey-Fuller or DFGLS regression.
    regression : {"c", "n", "ct", "ctt"}
        This is the method of regression that was used.  Following MacKinnon's
        notation, this can be "c" for constant, "n" for no constant, "ct" for
        constant and trend, and "ctt" for constant, trend, and trend-squared.
    num_unit_roots : int
        The number of series believed to be I(1).  For (Augmented) Dickey-
        Fuller N = 1.
    dist_type : {"adf-t", "adf-z", "dfgls"}
        The test type to use when computing p-values.  Options include
        "ADF-t" - ADF t-stat based bootstrap
        "ADF-z" - ADF z bootstrap
        "DFGLS" - GLS detrended Dickey Fuller

    Returns
    -------
    p-value : float
        The p-value for the ADF statistic estimated using MacKinnon 1994.

    References
    ----------
    MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution Functions for
        Unit-Root and Cointegration Tests." Journal of Business & Economics
        Statistics, 12.2, 167-76.

    Notes
    -----
    Most values are from MacKinnon (1994).  Values for DFGLS test statistics
    and the "n" version of the ADF z test statistic were computed following
    the methodology of MacKinnon (1994).
    """
    dist_type = cast(Literal["adf-t", "adf-z", "dfgls"], dist_type.lower())
    if num_unit_roots > 1 and dist_type.lower() != "adf-t":
        raise ValueError(
            "Cointegration results (num_unit_roots > 1) are"
            + "only available for ADF-t values"
        )
    if dist_type == "adf-t":
        maxstat = tau_max[regression][num_unit_roots - 1]
        minstat = tau_min[regression][num_unit_roots - 1]
        starstat = tau_star[regression][num_unit_roots - 1]
        small_p = tau_small_p[regression][num_unit_roots - 1]
        large_p = tau_large_p[regression][num_unit_roots - 1]
    elif dist_type == "adf-z":
        maxstat = adf_z_max[regression]
        minstat = adf_z_min[regression]
        starstat = adf_z_star[regression]
        small_p = adf_z_small_p[regression]
        large_p = adf_z_large_p[regression]
    elif dist_type == "dfgls":
        maxstat = dfgls_tau_max[regression]
        minstat = dfgls_tau_min[regression]
        starstat = dfgls_tau_star[regression]
        small_p = dfgls_small_p[regression]
        large_p = dfgls_large_p[regression]
    else:
        raise ValueError(f"Unknown test type {dist_type}")

    if stat > maxstat:
        return 1.0
    elif stat < minstat:
        return 0.0
    if stat <= starstat:
        poly_coef = small_p
        if dist_type == "adf-z":
            stat = float(log(abs(stat)))  # Transform stat for small p ADF-z
    else:
        poly_coef = large_p
    return norm.cdf(polyval(poly_coef[::-1], stat))


def mackinnoncrit(
    num_unit_roots: int = 1,
    regression: Literal["c", "n", "ct", "ctt"] = "c",
    nobs: float = inf,
    dist_type: Literal["adf-t", "adf-z", "dfgls"] = "adf-t",
) -> Float64Array:
    """
    Returns the critical values for cointegrating and the ADF test.

    In 2010 MacKinnon updated the values of his 1994 paper with critical values
    for the augmented Dickey-Fuller bootstrap.  These new values are to be
    preferred and are used here.

    Parameters
    ----------
    num_unit_roots : int
        The number of series of I(1) series for which the null of
        non-cointegration is being tested.  For N > 12, the critical values
        are linearly interpolated (not yet implemented).  For the ADF test,
        N = 1.
    regression : {"c", "ct", "ctt", "n"}, optional
        Following MacKinnon (1996), these stand for the type of regression run.
        "c" for constant and no trend, "ct" for constant with a linear trend,
        "ctt" for constant with a linear and quadratic trend, and "n" for
        no constant.  The values for the no constant case are taken from the
        1996 paper, as they were not updated for 2010 due to the unrealistic
        assumptions that would underlie such a case.
    nobs : {int, np.inf}, optional
        This is the sample size.  If the sample size is numpy.inf, then the
        asymptotic critical values are returned.
    dist_type : {"adf-t", "adf-z", "dfgls"}, optional
        Type of test statistic

    Returns
    -------
    crit_vals : ndarray
        Three critical values corresponding to 1%, 5% and 10% cut-offs.

    Notes
    -----
    Results for ADF t-stats from MacKinnon (1994,2010).  Results for DFGLS and
    ADF z-bootstrap use the same methodology as MacKinnon.

    References
    ----------
    MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution Functions for
        Unit-Root and Cointegration Tests." Journal of Business & Economics
        Statistics, 12.2, 167-76.
    MacKinnon, J.G. 2010.  "Critical Values for Cointegration Tests."
        Queen's University, Dept of Economics Working Papers 1227.
        https://ideas.repec.org/p/qed/wpaper/1227.html
    """
    lower_dist_type = dist_type.lower()
    valid_regression = ["c", "ct", "n", "ctt"]
    if lower_dist_type == "dfgls":
        valid_regression = ["c", "ct"]
    if regression not in valid_regression:
        raise ValueError(f"regression keyword {regression} not understood")

    if lower_dist_type == "adf-t":
        asymptotic_cv = tau_2010[regression][num_unit_roots - 1, :, 0]
        poly_coef = tau_2010[regression][num_unit_roots - 1, :, :].T
    elif lower_dist_type == "adf-z":
        poly_coef = array(adf_z_cv_approx[regression]).T
        asymptotic_cv = array(adf_z_cv_approx[regression])[:, 0]
    elif lower_dist_type == "dfgls":
        poly_coef = dfgls_cv_approx[regression].T
        asymptotic_cv = dfgls_cv_approx[regression][:, 0]
    else:
        raise ValueError(f"Unknown test type {dist_type}")

    if nobs is inf:
        return asymptotic_cv
    else:
        # Flip so that highest power to lowest power
        return polyval(poly_coef[::-1], 1.0 / nobs)


def kpss_crit(
    stat: float, trend: Literal["c", "ct"] = "c"
) -> tuple[float, Float64Array]:
    """
    Linear interpolation for KPSS p-values and critical values

    Parameters
    ----------
    stat : float
        The KPSS test statistic.
    trend : {"c","ct"}
        The trend used when computing the KPSS statistic

    Returns
    -------
    pvalue : float
        The interpolated p-value
    crit_val : ndarray
        Three element array containing the 10%, 5% and 1% critical values,
        in order

    Notes
    -----
    The p-values are linear interpolated from the quantiles of the simulated
    KPSS test statistic distribution using 100,000,000 replications and 2000
    data points.
    """
    table = kpss_critical_values[trend]
    y = table[:, 0]
    x = table[:, 1]
    # kpss.py contains quantiles multiplied by 100
    pvalue = float(interp(stat, x, y)) / 100.0
    cv = [1.0, 5.0, 10.0]
    crit_value = interp(cv, y[::-1], x[::-1])

    return pvalue, crit_value


def auto_bandwidth(
    y: Union[Sequence[Union[float, int]], ArrayLike1D],
    kernel: Literal[
        "ba", "bartlett", "nw", "pa", "parzen", "gallant", "qs", "andrews"
    ] = "ba",
) -> float:
    """
    Automatic bandwidth selection of Andrews (1991) and Newey & West (1994).

    Parameters
    ----------
    y : {ndarray, Series}
        Data on which to apply the bandwidth selection
    kernel : str
        The kernel function to use for selecting the bandwidth

        - "ba", "bartlett", "nw": Bartlett kernel (default)
        - "pa", "parzen", "gallant": Parzen kernel
        - "qs", "andrews":  Quadratic Spectral kernel

    Returns
    -------
    float
        The estimated optimal bandwidth.
    """
    y = ensure1d(y, "y")
    if y.shape[0] < 2:
        raise ValueError("Data must contain more than one observation")

    lower_kernel = kernel.lower()
    if lower_kernel in ("ba", "bartlett", "nw"):
        kernel = "ba"
        n_power = 2 / 9
    elif lower_kernel in ("pa", "parzen", "gallant"):
        kernel = "pa"
        n_power = 4 / 25
    elif lower_kernel in ("qs", "andrews"):
        kernel = "qs"
        n_power = 2 / 25
    else:
        raise ValueError("Unknown kernel")

    n = int(4 * ((len(y) / 100) ** n_power))
    sig = (n + 1) * [0]

    for i in range(n + 1):
        a = list(y[i:])
        b = list(y[: len(y) - i])
        sig[i] = int(npsum([i * j for (i, j) in zip(a, b)]))

    sigma_m1 = sig[1 : len(sig)]  # sigma without the 1st element
    s0 = sig[0] + 2 * sum(sigma_m1)

    if kernel == "ba":
        s1 = 0
        for j in range(len(sigma_m1)):
            s1 += (j + 1) * sigma_m1[j]
        s1 *= 2
        q = 1
        t_power = 1 / (2 * q + 1)
        gamma = 1.1447 * (((s1 / s0) ** 2) ** t_power)
    else:
        s2 = 0
        for j in range(len(sigma_m1)):
            s2 += ((j + 1) ** 2) * sigma_m1[j]
        s2 *= 2
        q = 2
        t_power = 1 / (2 * q + 1)
        if kernel == "pa":
            gamma = 2.6614 * (((s2 / s0) ** 2) ** t_power)
        else:  # kernel == "qs":
            gamma = 1.3221 * (((s2 / s0) ** 2) ** t_power)

    bandwidth = gamma * power(len(y), t_power)

    return bandwidth
