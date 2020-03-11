from arch.compat.statsmodels import add_trend

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.regression.linear_model import OLS, RegressionResults

from arch.typing import ArrayLike1D, ArrayLike2D
from arch.unitroot.critical_values.engle_granger import (
    CV_PARAMETERS,
    LARGE_PARAMETERS,
    SMALL_PARAMETERS,
    TAU_MAX,
    TAU_MIN,
    TAU_STAR,
)
from arch.unitroot.unitroot import ADF, TREND_DESCRIPTION
from arch.utility.array import ensure1d, ensure2d

__all__ = [
    "engle_granger",
    "EngleGrangerCointegrationTestResult",
    "engle_granger_cv",
    "engle_granger_pval",
]

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


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
