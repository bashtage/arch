import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.regression.linear_model import RegressionResults

from arch._typing import ArrayLike1D, ArrayLike2D, Literal, UnitRootTrend
from arch.unitroot._shared import (
    ResidualCointegrationTestResult,
    _check_cointegrating_regression,
    _cross_section,
)
from arch.unitroot.critical_values.engle_granger import (
    CV_PARAMETERS,
    LARGE_PARAMETERS,
    SMALL_PARAMETERS,
    TAU_MAX,
    TAU_MIN,
    TAU_STAR,
)
from arch.unitroot.unitroot import ADF, TREND_DESCRIPTION
from arch.utility._validation import check_bool, check_int, check_option


def _check_num_x(num_x: int) -> None:
    if not 1 <= num_x <= 12:
        raise ValueError(
            "The number of cross-sectional variables must be between 1 and "
            "12 (inclusive)"
        )


def engle_granger(
    y: ArrayLike1D,
    x: ArrayLike2D,
    trend: UnitRootTrend = "c",
    *,
    lags: int | None = None,
    max_lags: int | None = None,
    method: Literal["aic", "bic", "t-stat", "maic"] = "bic",
    low_memory: bool | None = None,
) -> "EngleGrangerTestResults":
    r"""
    Test for cointegration within a set of time series.

    Parameters
    ----------
    y : array_like
        The left-hand-side variable in the cointegrating regression.
    x : ndarray, Series or DataFrame
        The right-hand-side variables in the cointegrating regression.
    trend : {"n", "c", "ct", "ctt"}, optional
        Trend to include in the cointegrating regression. Trends are:

        * "n": No deterministic terms
        * "c": Constant
        * "ct": Constant and linear trend
        * "ctt": Constant, linear and quadratic trends

    lags : int, optional
        The number of lagged differences to include in the Augmented
        Dickey-Fuller test used on the residuals of the cointegrating
        regression. If None, the number of lags is selected using ``method``.
    max_lags : int, optional
        The maximum number of lags to consider when using automatic
        lag-length in the Augmented Dickey-Fuller regression.
    method : {"aic", "bic", "t-stat", "maic"}, optional
        The method used to select the number of lags included in the
        Augmented Dickey-Fuller regression.

        * "aic": Select the minimum of the Akaike IC
        * "bic": Select the minimum of the Schwarz/Bayesian IC
        * "t-stat": Select the largest lag length for which the t-statistic
          of the last lag is significant, using a cut-off of 1.645 for the
          absolute value of the t-statistic
        * "maic": Select the minimum of Ng and Perron's modified Akaike IC

    low_memory : bool, optional
        Flag indicating whether to use the low memory implementation of the
        lag selection algorithm in the Augmented Dickey-Fuller regression.
        If None, the algorithm is selected automatically. See
        :class:`arch.unitroot.ADF`.

    Returns
    -------
    EngleGrangerTestResults
        Results of the Engle-Granger test.

    See Also
    --------
    arch.unitroot.ADF
        Augmented Dickey-Fuller testing.
    arch.unitroot.PhillipsPerron
        Phillips & Perron's unit root test.
    arch.unitroot.cointegration.phillips_ouliaris
        Phillips-Ouliaris tests of cointegration.

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
    setup = _check_cointegrating_regression(y, x, trend)
    nobs = setup.y.shape[0]
    # The ADF regression of the residual cannot use more than nobs - 2 lags
    lags = check_int("lags", lags, lower=0, allow_none=True)
    max_lags = check_int("max_lags", max_lags, lower=0, upper=nobs - 2, allow_none=True)
    low_memory = check_bool("low_memory", low_memory, allow_none=True)
    _check_num_x(setup.x.shape[1])
    xsection = _cross_section(setup.y, setup.x, setup.trend)
    resid = xsection.resid
    # Never pass in the trend here since only used in x-section
    adf = ADF(
        resid,
        lags,
        trend="n",
        max_lags=max_lags,
        method=method,
        low_memory=low_memory,
    )
    stat = adf.stat
    nobs = resid.shape[0] - adf.lags - 1
    num_x = setup.x.shape[1]
    cv = engle_granger_cv(setup.trend, num_x, nobs)
    pv = engle_granger_pval(stat, setup.trend, num_x)
    return EngleGrangerTestResults(
        stat, pv, cv, order=num_x, adf=adf, xsection=xsection, trend=setup.trend
    )


class EngleGrangerTestResults(ResidualCointegrationTestResult):
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
    null : str, optional
        The null hypothesis.
    alternative : str, optional
        The alternative hypothesis.
    trend : str, optional
        The model's trend description.
    order : int, optional
        The number of stochastic trends in the null distribution.
    adf : ADF, optional
        The ADF instance used to perform the test and lag selection. It must
        be provided.
    xsection : RegressionResults, optional
        The OLS results used in the cross-sectional regression. They must be
        provided.
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
        adf: ADF | None = None,
        xsection: RegressionResults | None = None,
    ) -> None:
        super().__init__(
            stat, pvalue, crit_vals, null, alternative, trend, order, xsection
        )
        self.name = "Engle-Granger Cointegration Test"
        assert adf is not None
        self._adf = adf
        self._additional_info.update(
            {
                "ADF Lag length": self.lags,
                "Trend": self.trend,
                "Estimated Root ρ (γ+1)": self.rho,
                "Distribution Order": self.distribution_order,
            }
        )

    @property
    def lags(self) -> int:
        """The number of lags used in the Augmented Dickey-Fuller regression."""
        return self._adf.lags

    @property
    def max_lags(self) -> int | None:
        """The maximum number of lags used in the lag-length selection."""
        return self._adf.max_lags

    @property
    def rho(self) -> float:
        r"""
        The estimated coefficient in the Dickey-Fuller Test.

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

        param0, *_ = self._adf.regression.params
        return 1 + param0

    def summary(self) -> Summary:
        """
        Summary of test, containing statistic, p-value and critical values.

        Returns
        -------
        statsmodels.iolib.summary.Summary
            A summary instance with methods that support export to text, csv
            or latex.
        """
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


def engle_granger_cv(trend: str, num_x: int, nobs: int) -> pd.Series:
    """
    Critical values for Engle-Granger t-tests.

    Parameters
    ----------
    trend : {"n", "c", "ct", "ctt"}
        The trend included in the model.
    num_x : int
        The number of cross-sectional regressors in the model.
        Must be between 1 and 12.
    nobs : int
        The number of observations in the time series.

    Returns
    -------
    Series
        The critical values for 1, 5 and 10%.
    """
    trend = check_option("trend", trend, ("n", "c", "ct", "ctt"))
    _check_num_x(num_x)
    tbl = CV_PARAMETERS[trend]

    crit_vals = {}
    for size in (10, 5, 1):
        params = tbl[size][num_x]
        x = 1.0 / (nobs ** np.arange(4.0))
        crit_vals[size] = x @ params
    return pd.Series(crit_vals)


def engle_granger_pval(stat: float, trend: str, num_x: int) -> float:
    """
    Asymptotic P-values for Engle-Granger t-tests.

    Parameters
    ----------
    stat : float
        The test statistic.
    trend : {"n", "c", "ct", "ctt"}
        The trend included in the model.
    num_x : int
        The number of cross-sectional regressors in the model.
        Must be between 1 and 12.

    Returns
    -------
    float
        The asymptotic p-value.
    """
    trend = check_option("trend", trend, ("n", "c", "ct", "ctt"))
    _check_num_x(num_x)
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
