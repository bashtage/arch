from typing import Any, NamedTuple, Optional, Union, cast

import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS, RegressionResults

import arch.covariance.kernel as lrcov
from arch.typing import ArrayLike1D, ArrayLike2D, UnitRootTrend
from arch.utility.array import ensure1d, ensure2d
from arch.utility.timeseries import add_trend

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

KERNEL_ESTIMATORS: dict[str, type[lrcov.CovarianceEstimator]] = {
    kernel.lower(): getattr(lrcov, kernel) for kernel in lrcov.KERNELS
}
KERNEL_ESTIMATORS.update({kernel: getattr(lrcov, kernel) for kernel in lrcov.KERNELS})
KNOWN_KERNELS = "\n".join(sorted(k for k in KERNEL_ESTIMATORS))
KERNEL_ERR = f"kernel is not a known estimator. Must be one of:\n {KNOWN_KERNELS}"


class CointegrationSetup(NamedTuple):
    y: pd.Series
    x: pd.DataFrame
    trend: UnitRootTrend


def _check_kernel(kernel: str) -> str:
    kernel = kernel.replace("-", "").replace("_", "").lower()
    if kernel not in KERNEL_ESTIMATORS:
        est = "\n".join(sorted(k for k in KERNEL_ESTIMATORS))
        raise ValueError(
            f"kernel is not a known kernel estimator. Must be one of:\n {est}"
        )
    return kernel


def _check_cointegrating_regression(
    y: ArrayLike1D,
    x: ArrayLike2D,
    trend: UnitRootTrend,
    supported_trends: tuple[str, ...] = ("n", "c", "ct", "ctt"),
) -> CointegrationSetup:
    y = ensure1d(y, "y", True)
    x = ensure2d(x, "x")
    if y.shape[0] != x.shape[0]:
        raise ValueError(
            f"The number of observations in y and x differ. y has "
            f"{y.shape[0]} observtations, and x has {x.shape[0]}."
        )
    if not isinstance(x, pd.DataFrame):
        cols = [f"x{i}" for i in range(1, x.shape[1] + 1)]
        assert isinstance(y, pd.Series)
        x_df = pd.DataFrame(x, columns=cols, index=y.index)
    else:
        x_df = x
    trend_name = trend.lower()
    if trend_name.lower() not in supported_trends:
        trends = ",".join([f'"{st}"' for st in supported_trends])
        raise ValueError(f"Unknown trend. Must be one of {{{trends}}}")
    return CointegrationSetup(y, x_df, cast(UnitRootTrend, trend_name))


def _cross_section(
    y: Union[ArrayLike1D, ArrayLike2D], x: ArrayLike2D, trend: UnitRootTrend
) -> RegressionResults:
    if trend not in ("n", "c", "ct", "ctt"):
        raise ValueError('trend must be one of "n", "c", "ct" or "ctt"')
    y = ensure1d(y, "y", True)
    x = ensure2d(x, "x")

    if not isinstance(x, pd.DataFrame):
        cols = [f"x{i}" for i in range(1, x.shape[1] + 1)]
        assert isinstance(y, pd.Series)
        x = pd.DataFrame(x, columns=cols, index=y.index)
    x = add_trend(x, trend)
    res = OLS(y, x).fit()
    return res


class CointegrationTestResult:
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
        self._additional_info: dict[str, Any] = {}

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


class ResidualCointegrationTestResult(CointegrationTestResult):
    def __init__(
        self,
        stat: float,
        pvalue: float,
        crit_vals: pd.Series,
        null: str = "No Cointegration",
        alternative: str = "Cointegration",
        trend: str = "c",
        order: int = 2,
        xsection: Optional[RegressionResults] = None,
    ) -> None:
        super().__init__(stat, pvalue, crit_vals, null, alternative)
        self.name = "NONE"
        assert xsection is not None
        self._xsection = xsection
        self._order = order
        self._trend = trend
        self._additional_info = {}

    @property
    def trend(self) -> str:
        """The trend used in the cointegrating regression"""
        return self._trend

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
            matplotlib axes instance to hold the figure.
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
        return Summary()

    def _repr_html_(self) -> str:
        """Display as HTML for IPython notebook."""
        return self.summary().as_html()
