
from statsmodels.regression.linear_model import OLS
from arch.utility.timeseries import add_trend
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.data import _is_using_pandas
from arch.unitroot import ADF, DFGLS, PhillipsPerron
import pandas as pd
import numpy as np
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


# TODO Phillips-Ouliaris-Hansen Test For Cointegration
# TODO IntOrder Function


class Cointegration(object):
    """
    Class to test and estimate cointegration coefficents

    Cointegration:
    engle_granger_coef
    dynamic_coefs
    """
    def engle_granger_coef(self, y0, y1, trend='c',
                           max_lags=None, autolag='AIC',
                           method='ADF',
                           normalize=True, check=True):
        """
        Engle-Granger Cointegration Coefficient Calculations.

        This equation takes a linear combination of two L(1) time series to
        create a L(0) or stationary time series.

        This is useful if the two series have a similar stochastic long-term
        trend, as it eliminates them and allows you

        Parameters
        ----------
        y0 : array_like
            The first element in cointegrated system. Must be 1-d.
        y1 : array_like
            The remaining elements in cointegrated system.
        trend : str {'c', 'ct'}
            The trend term included in regression for cointegrating equation.

            * 'c' : constant.
            * 'ct' : constant and linear trend.
            * also available quadratic trend 'ctt', and no constant 'nc'.
        maxlag : None or int
            Argument for `adfuller`, largest or given number of lags.
        method: string
            Determines what method is used to check for a stationary process
            The available are DFGLS, PhillipsPerron, ADF.
        autolag : str
            * If 'AIC' (default) or 'BIC', then the number of lags is chosen
            to minimize the corresponding information criterion.
        normalize: boolean, optional
            As there are infinite scalar combinations that will produce the
            factor, this normalizes the first entry to be 1.
        check: boolean, optional
            Checks if the series has a possible cointegration factor using the
            Engle-Granger Cointegration Test

        Returns
        -------
        coefs: array
            A vector that will create a L(0) time series if a combination
            exists.

        Notes
        -----
        The series should be checked independently for their integration
        order. The series must be L(1) to get consistent results. You can
        check this by using the int_order function.

        References
        ----------
        .. [1] MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution
        Functions for Unit-Root and Cointegration Tests." Journal of
        Business & Economics Statistics, 12.2, 167-76.
        .. [2] MacKinnon, J.G. 2010.  "Critical Values for Cointegration
        Tests." Queen's University, Dept of Economics Working Papers 1227.
        http://ideas.repec.org/p/qed/wpaper/1227.html
        .. [3] Hamilton, J. D. (1994). Time series analysis
        (Vol. 2, pp. 690-696). Princeton, NJ: Princeton university press.
        """
        nobs, k_vars = y1.shape

        y1 = add_trend(y1, trend=trend, prepend=False)

        eg_model = OLS(y0, y1).fit()
        coefs = eg_model.params[0: k_vars]

        if check:
            if method == 'DFGLS':
                unit_root = DFGLS(
                                eg_model.resid,
                                trend=trend,
                                max_lags=max_lags,
                                method=method)
            elif method == 'PhillipsPerron':
                unit_root = PhillipsPerron(eg_model.resid, trend=trend)
            else:
                unit_root = ADF(eg_model.resid,
                                trend=trend,
                                max_lags=max_lags,
                                method=method)

            if unit_root.pvalue > .1:
                print('The null hypothesis cannot be rejected')

        if normalize:
            coefs = coefs / coefs[0]

        return coefs

    def dynamic_coefs(
                    self, y0, y1, n_lags=None,
                    trend='c', auto_lag='bic',
                    normalize=True, reverse=False):
        """
        Dynamic Cointegration Coefficient Calculation.

        This equation takes a linear combination of multiple L(1) time
        series to create a L(0) or stationary time series.

        This is useful if the two series have a similar stochastic long-term
        trend, as it eliminates them and allows you.

        Unlike Engle-Granger, this method uses dynamic regression - taking
        an equal combination of lags and leads of the differences of the
        series - to create a more accurate parameter vector. This method
        calculates the lag-lead matricies for the given lag values or searches
        for the best amount of lags using BIC calculations. Once the optimal
        value is found, the calculation is done and returned. The optimal
        lag can be found by using dot notation and finding max_lag. You
        can also find the model by using .model.

        Parameters
        ----------
        y0 : array_like
            The first element in cointegrated system. Must be 1-d.
        y1 : array_like
            The remaining elements in cointegrated system.
        n_lags: int, array, None
            This determines which values the function should search for the
            best vector.

            * int: If an int, the calculation is done for only that lag
            * array: If an array of two integers, the first value is where
                        the search begins and the second is where it ends
            * None: If None is given, the function searches from 2 to
                        ceiling of the cube root of the number of observations
                        divided by two plus two in order to ensure at least
                        one value is searched.
                        I.E last_lag = (n_obs**(1/3) / 2) + 2

        trend : str {'c', 'ct'}
            The trend term included in regression for cointegrating equation.

            * 'c' : constant.
            * 'ct' : constant and linear trend.
            * also available quadratic trend 'ctt', and no constant 'nc'.

        normalize: Boolean
            If true, the first entry in the parameter vector is normalized to
            one and everything else is divided by the first entry. This is
            because any cointegrating vector could be multiplied by a scalar
            and still be a cointegrating vector.
        reverse: Boolean
            The series must be ordered from the latest data points to the last.
            This is in order to calculate the differences. Using this, you can
            reverse the ordering of your data points.

        Returns
        -------
        coefs: array
            A vector that will create a L(0) time series if a
            combination sexists.

        Notes
        -----
        The data must go from the latest observations to the earliest. If not,
        the coef vector will be the opposite sign.

        The series should be checked independently for their integration order.
        The series must be L(1) to get consistent results. You can check this
        by using the int_order function.

        References
        ----------
        .. [1] Stock, J. H., & Watson, M. W. (1993). A simple estimator of
        cointegrating vectors in higher order integrated systems.
        Econometrica: Journal of the Econometric Society, 783-820.
        .. [2] Hamilton, J. D. (1994). Time series analysis
        (Vol. 2, pp. 690-696). Princeton, NJ: Princeton university press.
        """
        self.ic = []
        self.max_val = []
        self.model = ''
        self.coefs = []

        y1 = add_trend(y1, trend=trend, prepend=True)
        y1 = y1.reset_index(drop=True)
        if reverse:
            y0, y1 = y0[::-1], y1[::-1]

        if _is_using_pandas(y0, y1):
            columns = list(y1.columns)

        else:
            # Need to check if NumPy, because I can only support those two
            n_obs, k = y1.shape
            columns = ['Var_{}'.format(x) for x in range(k)]
            y0, y1 = pd.DataFrame(y0), pd.DataFrame(y1)

        # If none or interval, search for it using BIC or AIC
        if n_lags is None or len(n_lags) == 2:
            if len(n_lags) == 2:
                start, end = int(n_lags[0]), int(n_lags[1]) + 1
            elif n_lags is None:
                start = 2
                end = int(np.ceil(n_obs ** (1 / 3) / 2) + 2)

            n_obs, k = y1.shape
            dta = pd.DataFrame(np.diff(a=y1, n=1, axis=0))
            for lag in range(start, end):

                df1 = pd.DataFrame(lagmat(dta, lag + 1, trim='backward'))
                cols = dict(zip(list(df1.columns)[::-1][0:k][::-1], columns))
                df1 = df1.rename(columns=cols)

                df2 = pd.DataFrame(lagmat(dta, lag, trim='forward'))

                lags_leads = pd.concat([df1, df2], axis=1, join='outer')
                lags_leads = lags_leads.drop(list(range(0, lag)))
                lags_leads = lags_leads.reset_index(drop=True)

                lags_leads = lags_leads.drop(
                    list(range(len(lags_leads) - lag, len(lags_leads))))

                lags_leads = lags_leads.reset_index(drop=True)
                data_y = y0.drop(list(range(0, lag))).reset_index(drop=True)
                data_y = data_y.drop(
                    list(range(len(data_y) - lag - 1, len(data_y))))
                data_y = data_y.reset_index(drop=True)

                if auto_lag == 'aic':
                    self.ic.append([OLS(data_y, lags_leads).fit().aic, lag])
                elif auto_lag == 'bic':
                    self.ic.append([OLS(data_y, lags_leads).fit().bic, lag])

            self.max_val = max(self.ic, key=lambda item: item[0])
            self.max_val = self.max_val[1]

        elif len(n_lags) == 1:
            self.max_val = int(n_lags)

        else:
            raise SyntaxError(
                            'Make sure your lags are\
                            in one of the required forms.')

        dta = pd.DataFrame(np.diff(a=y1, n=1, axis=0))
        # Create a matrix of the lags, this also retains the original matrix,
        # which is why max_val + 1
        df1 = pd.DataFrame(lagmat(dta, self.max_val + 1, trim='backward'))

        # Rename the columns, as we need to keep track of them. We know the
        # original will be the final values
        cols = dict(zip(list(df1.columns)[::-1][0:k][::-1], columns))
        df1 = df1.rename(columns=cols)

        # Do the same, but these are leads, this does not keep the
        # original matrix, thus max_val
        df2 = pd.DataFrame(lagmat(dta, self.max_val, trim='forward'))

        # There are missing data due to the lags and leads, we concat
        # the frames and drop the values of which are missing.
        lags_leads = pd.concat([df1, df2], axis=1, join='outer')
        lags_leads = lags_leads.drop(list(range(0, self.max_val)))
        lags_leads = lags_leads.reset_index(drop=True)
        lags_leads = lags_leads.drop(
            list(range(len(lags_leads) - self.max_val, len(lags_leads))))
        lags_leads.reset_index(drop=True)

        # We also need to do this for the endog values, we need to
        # drop 1 extra due to a loss from first differencing.
        # This will be at the end of the matrix.
        data_y = y0.drop(list(range(0, self.max_val))).reset_index(drop=True)
        data_y = data_y.drop(list(range(len(data_y) - self.max_val -
                                        1, len(data_y))))
        data_y = data_y.reset_index(drop=True)

        self.model = OLS(data_y, lags_leads).fit()

        self.coefs = self.model.params[list(y1.columns)]

        if normalize:
            self.coefs = self.coefs / self.coefs[0]

        return(self.coefs)
