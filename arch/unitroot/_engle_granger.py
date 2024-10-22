from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.regression.linear_model import RegressionResults

from arch.typing import ArrayLike1D, ArrayLike2D, Literal, UnitRootTrend
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


def engle_granger(
    y: ArrayLike1D,
    x: ArrayLike2D,
    trend: UnitRootTrend = "c",
    *,
    lags: Optional[int] = None,
    max_lags: Optional[int] = None,
    method: Literal["aic", "bic", "t-stat"] = "bic",
) -> "EngleGrangerTestResults":
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
    xsection = _cross_section(setup.y, setup.x, setup.trend)
    resid = xsection.resid
    # Never pass in the trend here since only used in x-section
    adf = ADF(resid, lags, trend="n", max_lags=max_lags, method=method)
    stat = adf.stat
    nobs = resid.shape[0] - adf.lags - 1
    num_x = setup.x.shape[1]
    cv = engle_granger_cv(trend, num_x, nobs)
    pv = engle_granger_pval(stat, trend, num_x)
    return EngleGrangerTestResults(
        stat, pv, cv, order=num_x, adf=adf, xsection=xsection
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
    def max_lags(self) -> Union[int, None]:
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

        param0, *_ = self._adf.regression.params
        return 1 + param0

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
