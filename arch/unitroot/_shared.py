from collections.abc import Callable, Sequence
import functools
from typing import Any, NamedTuple, Optional, TypeVar, cast
import warnings

import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS, RegressionResults

from arch._typing import ArrayLike1D, ArrayLike2D, UnitRootTrend
from arch.utility._validation import (
    check_array,
    check_instance,
    check_option,
    check_str,
)
from arch.utility.array import ensure1d, ensure2d
from arch.utility.exceptions import InfeasibleTestException
from arch.utility.timeseries import add_trend

try:
    import matplotlib.pyplot as plt

except ImportError:
    pass


class CointegrationSetup(NamedTuple):
    y: pd.Series
    x: pd.DataFrame
    trend: UnitRootTrend


# The number of deterministic terms in the cointegrating regression
_TREND_TERMS = {"n": 0, "c": 1, "ct": 2, "ctt": 3, "t": 1}


def _check_not_constant(y: np.ndarray, name: str = "y") -> None:
    """
    Raise InfeasibleTestException if the 1-d data y, which are finite, are constant.

    The statistics of the tests are ratios of quantities that are 0 for constant
    data. They were nan, raised other errors, or were computed from the rounding
    errors of removing the mean, which gave arbitrary statistics and p-values.
    """
    # A single observation is reported as too few observations by the tests
    if y.shape[0] > 1 and y.max() == y.min():
        raise InfeasibleTestException(
            f"{name} is constant (every value is {float(y[0])!r}), and so the test "
            "statistic cannot be computed. The test requires a series that varies."
        )


_EPS = float(np.finfo(np.float64).eps)
# The largest differences of an exact linear (2) or quadratic (3) trend, and the
# largest residuals of an exact fit, in multiples of the rounding error of the data,
# eps * max|data|. Exact trends computed in float64 have second and third
# differences of at most 1.3 and 7.7, and exact fits residuals of at most 15, while
# random walks and white noise, even with a variation of 1e-14 of their level, have
# at least 88, 164 and 58.
_TREND_DIFFERENCE_TOLERANCE = {2: 16.0, 3: 32.0}
_EXACT_FIT_TOLERANCE = 32.0
_TREND_ORDERS = {"ct": 2, "ctt": 3}
_TREND_SHAPES = {2: "linear", 3: "quadratic"}


def _check_not_deterministic(y: np.ndarray, trend: str, name: str = "y") -> None:
    """
    Raise InfeasibleTestException if y is exactly a trend of the terms in trend.

    A linear trend for "ct" and a quadratic trend for "ctt", whose second or third
    differences are 0 to within the rounding error of y. Removing the trend leaves
    only rounding errors, from which the tests computed arbitrary statistics. A
    constant is checked by _check_not_constant.
    """
    if trend not in _TREND_ORDERS:
        return
    rounding = _EPS * float(np.max(np.abs(y)))
    # A linear trend is reported as linear when the terms also remove a quadratic
    for order in range(2, _TREND_ORDERS[trend] + 1):
        if y.shape[0] <= order:
            return
        tolerance = _TREND_DIFFERENCE_TOLERANCE[order] * rounding
        if np.max(np.abs(np.diff(y, order))) <= tolerance:
            raise InfeasibleTestException(
                f"{name} is exactly a {_TREND_SHAPES[order]} trend, and so the test "
                f"statistic cannot be computed with trend={trend!r}, which removes it. "
                "The test requires a series that is not a deterministic trend."
            )


def _check_regression(
    lhs: Any,
    rhs: Any,
    regression: str,
    *,
    scale: float = 0.0,
    column_scales: Sequence[float] | None = None,
    hint: str,
    result: str = "the test statistic cannot be computed",
) -> None:
    """
    Raise InfeasibleTestException if a regression is exactly singular.

    A regressor is an exact linear combination of the others, which statsmodels
    only warns about, or the regressors fit lhs exactly, so that the residuals are
    rounding errors. Both use the rounding error of the data, and not a tolerance
    on the singular values, which would also reject a series whose variation is
    tiny but real compared with its level.

    scale is the size of the data that lhs is computed from, whose rounding errors
    it contains, for example max|y| for the differences of y. column_scales are the
    same for the columns of rhs. A column computed exactly, such as a trend, has 0,
    and then its own size is used.
    """
    lhs = np.asarray(lhs, dtype=float).ravel()
    rhs = np.asarray(rhs, dtype=float)
    ncols = rhs.shape[1]
    scales = [0.0] * ncols if column_scales is None else list(column_scales)
    collinear = np.any(np.linalg.norm(rhs, axis=0) == 0) or any(
        _is_exact_fit(rhs[:, j], np.delete(rhs, j, axis=1), scales[j])
        for j in range(ncols)
    )
    if collinear:
        raise InfeasibleTestException(
            f"The regressors of the {regression} regression are collinear, and so "
            f"{result}. {hint}"
        )
    if _is_exact_fit(lhs, rhs, scale):
        raise InfeasibleTestException(
            f"The {regression} regression fits exactly, since its residuals are 0 to "
            f"within the rounding error of the data, and so {result}. {hint}"
        )


def _fit_ols(
    lhs: Any,
    rhs: Any,
    regression: str,
    *,
    scale: float = 0.0,
    column_scales: Sequence[float] | None = None,
    hint: str,
    result: str = "the test statistic cannot be computed",
) -> RegressionResults:
    """
    Fit an OLS regression of a test, which is not singular.

    The exact singularity is checked first by _check_regression. statsmodels then
    finds the rank from the singular values of the regressors, and warns with
    SingularMatrixWarning if they are rank-deficient, which happens when a column
    is tiny compared with another one, for example a variation of 1e-13 of the level
    of y and t**2. The estimates are then not unique, and the test raises instead
    of warning.
    """
    _check_regression(
        lhs,
        rhs,
        regression,
        scale=scale,
        column_scales=column_scales,
        hint=hint,
        result=result,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The design matrix is rank-deficient")
        res = OLS(lhs, rhs).fit()
    rank, ncols = int(res.model.rank), int(res.model.wexog.shape[1])
    if rank < ncols:
        raise InfeasibleTestException(
            f"The regressors of the {regression} regression are numerically "
            f"collinear (their rank is {rank} and there are {ncols}), and so {result}. "
            f"{hint}"
        )
    return res


def _is_exact_fit(lhs: np.ndarray, rhs: np.ndarray, scale: float) -> bool:
    """
    Whether rhs, whose columns are not 0, fits lhs to within rounding error.

    The columns are normalized, so that the scale of a column, such as t**2, does
    not reduce the accuracy of the fit. The rounding error is relative to the
    largest of scale, lhs and the terms of the fit, which are the values that lhs
    is computed from.
    """
    if rhs.shape[1] == 0:
        return False
    scaled = rhs / np.linalg.norm(rhs, axis=0)
    beta = np.linalg.lstsq(scaled, lhs, rcond=None)[0]
    bound = max(scale, float(np.max(np.abs(lhs))), float(np.max(np.abs(scaled * beta))))
    resid = lhs - scaled @ beta
    return bool(np.max(np.abs(resid)) <= _EXACT_FIT_TOLERANCE * _EPS * bound)


_COINTEGRATION_SINGULAR = (
    "This happens when y is an exact linear combination of x and the deterministic "
    "terms, or when the columns of x and the deterministic terms are collinear."
)

_DEGENERATE = (
    "This happens when the data are degenerate, for example when their values are "
    "too small or too large in scale (such as 1e-300 or 1e300)."
)


def _check_finite_statistic(stat: float, pvalue: float, test: str) -> None:
    """
    Raise InfeasibleTestException if the statistic or the p-value is nan or inf.

    The data are finite, so a statistic that is not is computed from quantities that
    overflow or underflow, or that are 0, and is not a result.
    """
    if not (np.isfinite(stat) and np.isfinite(pvalue)):
        raise InfeasibleTestException(
            f"The {test} statistic is {float(stat)!r} and its p-value is "
            f"{float(pvalue)!r}, which are not finite, and so the test cannot be "
            f"computed. {_DEGENERATE}"
        )


def _check_finite_data(what: str, *arrays: Any) -> None:
    """
    Raise InfeasibleTestException if data computed during an estimation are not
    finite.

    Used before routines that iterate, such as lstsq, which some LAPACK builds do
    not return from when their input is not finite.
    """
    if not all(np.all(np.isfinite(np.asarray(a, dtype=float))) for a in arrays):
        raise InfeasibleTestException(
            f"The {what} are not finite, and so the parameters cannot be estimated. "
            f"{_DEGENERATE}"
        )


def _check_finite_estimates(params: Any, cov: Any, method: str) -> None:
    """
    Raise InfeasibleTestException if the estimates or their covariance are not finite.
    """
    params = np.asarray(params, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if not (np.all(np.isfinite(params)) and np.all(np.isfinite(cov))):
        raise InfeasibleTestException(
            f"The {method} estimates or their covariance are not finite, and so the "
            f"parameters cannot be estimated. {_DEGENERATE}"
        )


def _infeasible_from(
    err: Exception, test: str, what: str = "statistic"
) -> InfeasibleTestException:
    """The InfeasibleTestException of a test or estimator whose linear algebra failed."""
    return InfeasibleTestException(
        f"The {test} {what} cannot be computed since a matrix that it needs is "
        f"singular or not finite ({type(err).__name__}: {err}). {_DEGENERATE}"
    )


_F = TypeVar("_F", bound=Callable[..., Any])


def _infeasible_on_error(test: str, what: str = "statistic") -> Callable[[_F], _F]:
    """
    Raise InfeasibleTestException for LinAlgError and ZeroDivisionError in func.

    They are raised by degenerate data, such as a singular matrix or a long-run
    variance of 0, and not by invalid arguments, which are checked first.
    """

    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except (np.linalg.LinAlgError, ZeroDivisionError) as err:
                raise _infeasible_from(err, test, what) from err

        return cast("_F", wrapper)

    return decorator


def _check_cointegrating_regression(
    y: ArrayLike1D,
    x: ArrayLike2D,
    trend: UnitRootTrend,
    supported_trends: tuple[str, ...] = ("n", "c", "ct", "ctt"),
) -> CointegrationSetup:
    trend_name = check_option("trend", trend, supported_trends)
    # Checks only, the conversion below keeps the index of a pandas object
    y_values = check_array("y", y, ndim=1)
    x_values = check_array("x", x, ndim=2)
    if y_values.shape[0] != x_values.shape[0]:
        raise ValueError(
            f"The number of observations in y and x differ. y has "
            f"{y_values.shape[0]} observations, and x has {x_values.shape[0]}."
        )
    num_x = x_values.shape[1]
    if num_x < 1:
        raise ValueError(
            f"x must be a 2-d array with at least 1 column; got shape "
            f"{x_values.shape} ({type(x).__name__})"
        )
    # One more row than the regression has parameters is needed to estimate it, and
    # all estimators use the first difference of the data
    num_params = num_x + _TREND_TERMS[trend_name]
    check_array("x", x_values, ndim=2, min_length=num_params + 2)
    # After the checks of the shapes, which report a specification that is not possible
    _check_not_constant(y_values)
    # The constant is added even if x has one, so that the collinearity is found
    rhs_values = (
        x_values
        if trend_name == "n"
        else add_trend(x_values, trend_name, prepend=True, has_constant="add")
    )
    _check_regression(
        y_values,
        rhs_values,
        "cointegrating",
        hint=_COINTEGRATION_SINGULAR,
        result="the test statistic or the estimates cannot be computed",
    )
    y = ensure1d(y, "y", True)
    x = ensure2d(x, "x")
    if not isinstance(x, pd.DataFrame):
        cols = [f"x{i}" for i in range(1, x.shape[1] + 1)]
        assert isinstance(y, pd.Series)
        x_df = pd.DataFrame(x, columns=cols, index=y.index)
    else:
        x_df = x
    if not y.index.equals(x_df.index):
        raise ValueError(
            f"x must be array_like with the same index as y; got a different index "
            f"({type(x_df.index).__name__})"
        )
    return CointegrationSetup(y, x_df, cast("UnitRootTrend", trend_name))


def _cross_section(
    y: ArrayLike1D | ArrayLike2D, x: ArrayLike2D, trend: UnitRootTrend
) -> RegressionResults:
    trend = cast("UnitRootTrend", check_option("trend", trend, ("n", "c", "ct", "ctt")))
    y = ensure1d(y, "y", True)
    x = ensure2d(x, "x")

    if not isinstance(x, pd.DataFrame):
        cols = [f"x{i}" for i in range(1, x.shape[1] + 1)]
        assert isinstance(y, pd.Series)
        x = pd.DataFrame(x, columns=cols, index=y.index)
    x = add_trend(x, trend)
    return _fit_ols(y, x, "cointegrating", hint=_COINTEGRATION_SINGULAR)


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
    null : str, optional
        The null hypothesis.
    alternative : str, optional
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
        """Set or get the name of the cointegration test."""
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
        The critical values of the test.

        Returns
        -------
        Series
            Series with three keys, 1, 5 and 10 containing the critical values
            of the test statistic.
        """
        return self._crit_vals

    @property
    def null_hypothesis(self) -> str:
        """The null hypothesis."""
        return self._null

    @property
    def alternative_hypothesis(self) -> str:
        """The alternative hypothesis."""
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
        xsection: RegressionResults | None = None,
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
        """The trend used in the cointegrating regression."""
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
        self, axes: Optional["plt.Axes"] = None, title: str | None = None
    ) -> "plt.Figure":
        """
        Plot the cointegration residuals.

        Parameters
        ----------
        axes : Axes, optional
            Matplotlib axes instance to hold the figure.
        title : str, optional
            Title for the figure.

        Returns
        -------
        Figure
            The matplotlib Figure instance.

        Raises
        ------
        TypeError
            If axes is not a matplotlib Axes or None, or title is not a string or
            None.
        """
        from matplotlib.axes import Axes  # noqa: PLC0415

        axes = check_instance("axes", axes, Axes, allow_none=True)
        title = check_str("title", title, allow_none=True)
        resid = self.resid
        df = pd.DataFrame(resid)
        title = "Cointegrating Residual" if title is None else title
        ax = df.plot(legend=False, ax=axes, title=title)
        ax.set_xlim(df.index.min(), df.index.max())
        fig = cast("plt.Figure", ax.get_figure())
        fig.tight_layout(pad=1.0)

        return fig

    def summary(self) -> Summary:
        return Summary()

    def _repr_html_(self) -> str:
        """Display as HTML for IPython notebook."""
        return self.summary().as_html()
