"""
Core classes for ARCH models
"""

from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Sequence
from copy import deepcopy
import datetime as dt
from functools import cached_property
from typing import Any, cast, overload
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import OptimizeResult, minimize
from statsmodels.iolib.summary import Summary, fmt_2cols, fmt_params
from statsmodels.iolib.table import SimpleTable
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.tsatools import lagmat

from arch._typing import (
    ArrayLike,
    ArrayLike1D,
    ArrayLike2D,
    DateLike,
    Float64Array,
    Float64Array1D,
    Float64Array2D,
    ForecastingMethod,
    Label,
    Literal,
)
from arch.univariate.distribution import Distribution, Normal
from arch.univariate.volatility import ConstantVariance, VolatilityProcess
from arch.utility.array import ensure1d, to_array_1d
from arch.utility.exceptions import (
    ConvergenceWarning,
    DataScaleWarning,
    StartingValueWarning,
    convergence_warning,
    data_scale_warning,
    starting_value_warning,
)
from arch.utility.testing import WaldTestStatistic
from arch.vendor._decorators import deprecate_kwarg

MPL_LT_310 = False
try:
    from matplotlib import __version__
    from matplotlib.figure import Figure
    from packaging.version import Version

    MPL_LT_310 = Version(__version__) < Version("3.10.0")
except ImportError:
    pass


__all__ = [
    "ARCHModel",
    "ARCHModelForecast",
    "ARCHModelResult",
    "constraint",
    "format_float_fixed",
    "implicit_constant",
]

CONVERGENCE_WARNING: str = """\
WARNING: The optimizer did not indicate successful convergence. The message was {msg}.
See convergence_flag.
"""

# Callback variables
_callback_info = {"iter": 0, "llf": 0.0, "count": 0, "display": 1}


def _callback(parameters: Float64Array1D) -> None:
    """
    Callback for use in optimization

    Parameters
    ----------
    parameters : ndarray
        Parameter value (not used by function).
    *args
        Any other arguments passed to the minimizer.

    Notes
    -----
    Uses global values to track iteration, iteration display frequency,
    log likelihood and function count
    """

    _callback_info["iter"] += 1
    disp = "Iteration: {0:>6},   Func. Count: {1:>6.3g},   Neg. LLF: {2}"
    if _callback_info["iter"] % _callback_info["display"] == 0:
        print(
            disp.format(
                _callback_info["iter"], _callback_info["count"], _callback_info["llf"]
            )
        )


def constraint(a: Float64Array, b: Float64Array) -> list[dict[str, object]]:
    """
    Generate constraints from arrays

    Parameters
    ----------
    a : ndarray
        Parameter loadings
    b : ndarray
        Constraint bounds

    Returns
    -------
    constraints : dict
        Dictionary of inequality constraints, one for each row of a

    Notes
    -----
    Parameter constraints satisfy a.dot(parameters) - b >= 0
    """

    def factory(coeff: Float64Array, val: float) -> Callable[..., float]:
        def f(params: Float64Array, *args: Any) -> float:
            return np.dot(coeff, params) - val

        return f

    constraints = []
    for i in range(a.shape[0]):
        con = {"type": "ineq", "fun": factory(a[i], b[i])}
        constraints.append(con)

    return constraints


def format_float_fixed(x: float, max_digits: int = 10, decimal: int = 4) -> str:
    """Formats a floating point number so that if it can be well expressed
    in using a string with digits len, then it is converted simply, otherwise
    it is expressed in scientific notation"""
    # basic_format = '{:0.' + str(digits) + 'g}'
    if x == 0:
        return ("{:0." + str(decimal) + "f}").format(0.0)
    scale = np.log10(np.abs(x))
    scale = np.sign(scale) * np.ceil(np.abs(scale))
    if scale > (max_digits - 2 - decimal) or scale < -(decimal - 2):
        formatted = ("{0:" + str(max_digits) + "." + str(decimal) + "e}").format(x)
    else:
        formatted = ("{0:" + str(max_digits) + "." + str(decimal) + "f}").format(x)
    return formatted


def implicit_constant(x: Float64Array) -> bool:
    """
    Test a matrix for an implicit constant

    Parameters
    ----------
    x : ndarray
        Array to be tested

    Returns
    -------
    constant : bool
        Flag indicating whether the array has an implicit constant - whether
        the array has a set of columns that adds to a constant value
    """
    nobs = x.shape[0]
    rank = np.linalg.matrix_rank(np.hstack((np.ones((nobs, 1)), x)))
    return rank == x.shape[1]


class ARCHModel(metaclass=ABCMeta):
    """
    Abstract base class for mean models in ARCH processes.  Specifies the
    conditional mean process.

    All public methods that raise NotImplementedError should be overridden by
    any subclass.  Private methods that raise NotImplementedError are optional
    to override but recommended where applicable.
    """

    def __init__(
        self,
        y: ArrayLike | None = None,
        volatility: VolatilityProcess | None = None,
        distribution: Distribution | None = None,
        hold_back: int | None = None,
        rescale: bool | None = None,
    ) -> None:
        self._name = "ARCHModel"
        self._is_pandas = isinstance(y, (pd.DataFrame, pd.Series))
        if y is not None:
            self._y_series = cast("pd.Series", ensure1d(y, "y", series=True))
        else:
            self._y_series = cast(
                "pd.Series", ensure1d(np.empty((0,)), "y", series=True)
            )
        self._y = to_array_1d(
            np.ascontiguousarray(self._y_series.to_numpy()).astype(float)
        )
        if not np.all(np.isfinite(self._y)):
            raise ValueError(
                "NaN or inf values found in y. y must contains only finite values."
            )
        self._y_original = y

        self._fit_indices: list[int] = [0, int(self._y.shape[0])]
        self._fit_y = self._y

        self.hold_back: int | None = hold_back
        self._hold_back = 0 if hold_back is None else hold_back

        self.rescale: bool | None = rescale
        self.scale: float = 1.0

        self._backcast: float | Float64Array1D | None = None
        self._var_bounds: Float64Array | None = None

        if isinstance(volatility, VolatilityProcess):
            self._volatility = volatility
        elif volatility is None:
            self._volatility = ConstantVariance()
        else:
            raise TypeError("volatility must inherit from VolatilityProcess")

        if isinstance(distribution, Distribution):
            self._distribution = distribution
        elif distribution is None:
            self._distribution = Normal()
        else:
            raise TypeError("distribution must inherit from Distribution")

    @property
    def name(self) -> str:
        """The name of the model."""
        return self._name

    def constraints(self) -> tuple[Float64Array, Float64Array1D]:
        """
        Construct linear constraint arrays  for use in non-linear optimization

        Returns
        -------
        a : ndarray
            Number of constraints by number of parameters loading array
        b : ndarray
            Number of constraints array of lower bounds

        Notes
        -----
        Parameters satisfy a.dot(parameters) - b >= 0
        """
        return np.empty((0, self.num_params)), np.empty(0)

    def bounds(self) -> list[tuple[float, float]]:
        """
        Construct bounds for parameters to use in non-linear optimization

        Returns
        -------
        bounds : list (2-tuple of float)
            Bounds for parameters to use in estimation.
        """
        num_params = self.num_params
        return [(-np.inf, np.inf)] * num_params

    @property
    def y(self) -> ArrayLike | None:
        """Returns the dependent variable"""
        return self._y_original

    @property
    def volatility(self) -> VolatilityProcess:
        """Set or gets the volatility process

        Volatility processes must be a subclass of VolatilityProcess
        """
        return self._volatility

    @volatility.setter
    def volatility(self, value: VolatilityProcess) -> None:
        if not isinstance(value, VolatilityProcess):
            raise ValueError("Must subclass VolatilityProcess")
        self._volatility = value

    @property
    def distribution(self) -> Distribution:
        """Set or gets the error distribution

        Distributions must be a subclass of Distribution
        """
        return self._distribution

    @distribution.setter
    def distribution(self, value: Distribution) -> None:
        if not isinstance(value, Distribution):
            raise ValueError("Must subclass Distribution")
        self._distribution = value

    def _check_scale(self, resids: ArrayLike1D) -> None:
        check = self.rescale in (None, True)
        if not check:
            return
        orig_scale = scale = float(np.var(resids))
        rescale = 1.0
        while not 0.1 <= scale < 10000.0 and scale > 0:
            if scale < 1.0:
                rescale *= 10
            else:
                rescale /= 10
            scale = orig_scale * rescale**2
        if rescale == 1.0:
            return
        if self.rescale is None:
            warnings.warn(
                data_scale_warning.format(orig_scale, rescale),
                DataScaleWarning,
                stacklevel=2,
            )
            return
        self.scale = rescale

    @abstractmethod
    def _scale_changed(self) -> None:
        """
        Called when the scale has changed.  This allows the model
        to update any values that are affected by the scale changes,
        e.g., any logged values.
        """

    def _r2(self, params: ArrayLike1D) -> float | None:
        """
        Computes the model r-square.  Optional to over-ride.  Must match
        signature.
        """
        raise NotImplementedError("Subclasses optionally may provide.")

    @abstractmethod
    def _fit_no_arch_normal_errors_params(self) -> Float64Array1D:
        """
        Must be overridden with closed form estimator the return parameters ony
        """

    @abstractmethod
    def _fit_no_arch_normal_errors(
        self, cov_type: Literal["robust", "classic"] = "robust"
    ) -> "ARCHModelResult":
        """
        Must be overridden with closed form estimator
        """

    @staticmethod
    def _static_gaussian_loglikelihood(resids: Float64Array1D) -> float:
        nobs = resids.shape[0]
        sigma2 = resids.dot(resids) / nobs

        loglikelihood = -0.5 * nobs * np.log(2 * np.pi)
        loglikelihood -= 0.5 * nobs * np.log(sigma2)
        loglikelihood -= 0.5 * nobs

        return loglikelihood

    def _fit_parameterless_model(
        self,
        cov_type: Literal["robust", "classic"],
        backcast: float | Float64Array1D,
    ) -> "ARCHModelResult":
        """
        When models have no parameters, fill return values

        Returns
        -------
        results : ARCHModelResult
            Model result from parameterless model
        """
        y = self._fit_y
        # Fake convergence results, see GH #87
        opt = OptimizeResult({"status": 0, "message": ""})

        params = np.empty(0)
        param_cov = np.empty((0, 0))
        first_obs, last_obs = self._fit_indices
        resids_final = np.full(self._y.shape, np.nan)
        resids_final[first_obs:last_obs] = y

        var_bounds = self.volatility.variance_bounds(y)
        vol = np.zeros(y.shape, dtype=float)
        self.volatility.compute_variance(params, y, vol, backcast, var_bounds)
        vol = cast("Float64Array1D", np.sqrt(vol))

        # Reshape resids vol
        vol_final = np.full(self._y.shape, np.nan, dtype=float)
        vol_final[first_obs:last_obs] = vol

        names = self._all_parameter_names()
        r2 = self._r2(params)
        fit_start, fit_stop = self._fit_indices
        loglikelihood = -1.0 * self._loglikelihood(
            params,
            cast("Float64Array1D", vol**2 * np.ones(fit_stop - fit_start)),
            backcast,
            var_bounds,
        )

        assert isinstance(r2, float)
        return ARCHModelResult(
            params,
            param_cov,
            r2,
            resids_final,
            vol_final,
            cov_type,
            self._y_series,
            names,
            loglikelihood,
            self._is_pandas,
            opt,
            fit_start,
            fit_stop,
            deepcopy(self),
        )

    @overload
    def _loglikelihood(
        self,
        parameters: Float64Array1D,
        sigma2: Float64Array1D,
        backcast: float | Float64Array1D,
        var_bounds: Float64Array2D,
    ) -> float:  # pragma: no cover
        ...  # pragma: no cover

    @overload
    def _loglikelihood(
        self,
        parameters: Float64Array1D,
        sigma2: Float64Array1D,
        backcast: float | Float64Array1D,
        var_bounds: Float64Array2D,
        individual: Literal[False] = ...,
    ) -> float:  # pragma: no cover
        ...  # pragma: no cover

    @overload
    def _loglikelihood(
        self,
        parameters: Float64Array1D,
        sigma2: Float64Array1D,
        backcast: float | Float64Array1D,
        var_bounds: Float64Array2D,
        individual: Literal[True] = ...,
    ) -> Float64Array1D:  # pragma: no cover
        ...  # pragma: no cover

    def _loglikelihood(
        self,
        parameters: Float64Array1D,
        sigma2: Float64Array1D,
        backcast: float | Float64Array1D,
        var_bounds: Float64Array2D,
        individual: bool = False,
    ) -> float | Float64Array1D:
        """
        Computes the log-likelihood using the entire model

        Parameters
        ----------
        parameters
        sigma2
        backcast
        individual : bool, optional

        Returns
        -------
        neg_llf : float
            Negative of model loglikelihood
        """
        # Parse parameters
        _callback_info["count"] += 1

        # 1. Resids
        mp, vp, dp = self._parse_parameters(parameters)
        _resids = self.resids(mp)

        # 2. Compute sigma2 using VolatilityModel
        sigma2 = self.volatility.compute_variance(
            vp, _resids, sigma2, backcast, var_bounds
        )
        # 3. Compute log likelihood using Distribution
        llf = self.distribution.loglikelihood(dp, _resids, sigma2, individual)

        if not individual:
            _callback_info["llf"] = llf_f = -float(llf)
            return llf_f

        return cast("np.ndarray", -llf)

    def _all_parameter_names(self) -> list[str]:
        """Returns a list containing all parameter names from the mean model,
        volatility model and distribution"""

        names = self.parameter_names()
        names.extend(self.volatility.parameter_names())
        names.extend(self.distribution.parameter_names())

        return names

    def _parse_parameters(
        self,
        x: ArrayLike1D | Sequence[float],
    ) -> tuple[Float64Array1D, Float64Array1D, Float64Array1D]:
        """Return the parameters of each model in a tuple"""
        _x = to_array_1d(np.asarray(x, dtype=float))
        km, kv = int(self.num_params), int(self.volatility.num_params)
        return (
            to_array_1d(_x[:km]),
            to_array_1d(_x[km : km + kv]),
            to_array_1d(_x[km + kv :]),
        )

    def fix(
        self,
        params: ArrayLike1D | Sequence[float],
        first_obs: int | DateLike | None = None,
        last_obs: int | DateLike | None = None,
    ) -> "ARCHModelFixedResult":
        """
        Allows an ARCHModelFixedResult to be constructed from fixed parameters.

        Parameters
        ----------
        params : {ndarray, Series}
            User specified parameters to use when generating the result. Must
            have the correct number of parameters for a given choice of mean
            model, volatility model and distribution.
        first_obs : {int, str, datetime, Timestamp}
            First observation to use when fixing model
        last_obs : {int, str, datetime, Timestamp}
            Last observation to use when fixing model

        Returns
        -------
        results : ARCHModelFixedResult
            Object containing model results

        Notes
        -----
        Parameters are not checked against model-specific constraints.
        """
        v = self.volatility

        self._adjust_sample(first_obs, last_obs)
        resids = self.resids(self.starting_values())
        sigma2 = np.zeros(resids.shape[0], dtype=float)
        backcast = v.backcast(resids)
        self._backcast = backcast

        var_bounds = v.variance_bounds(resids)

        params = ensure1d(params, "params", False)
        loglikelihood = -1.0 * self._loglikelihood(params, sigma2, backcast, var_bounds)
        assert isinstance(loglikelihood, float)

        mp, vp, _ = self._parse_parameters(params)

        resids = to_array_1d(self.resids(mp))
        vol = np.zeros(resids.shape[0], dtype=float)
        self.volatility.compute_variance(vp, resids, vol, backcast, var_bounds)
        vol = to_array_1d(np.sqrt(vol))

        names = self._all_parameter_names()
        # Reshape resids and vol
        first_obs, last_obs = self._fit_indices
        resids_final = np.full(self._y.shape, np.nan, dtype=float)
        resids_final[first_obs:last_obs] = resids
        vol_final = np.full(self._y.shape, np.nan, dtype=float)
        vol_final[first_obs:last_obs] = vol

        model_copy = deepcopy(self)
        return ARCHModelFixedResult(
            params,
            resids_final,
            vol_final,
            self._y_series,
            names,
            loglikelihood,
            self._is_pandas,
            model_copy,
        )

    @abstractmethod
    def _adjust_sample(
        self,
        first_obs: int | DateLike | None,
        last_obs: int | DateLike | None,
    ) -> None:
        """
        Performs sample adjustment for estimation

        Parameters
        ----------
        first_obs : {int, str, datetime, datetime64, Timestamp}
            First observation to use when estimating model
        last_obs : {int, str, datetime, datetime64, Timestamp}
            Last observation to use when estimating model

        Notes
        -----
        Adjusted sample must follow Python semantics of first_obs:last_obs
        """

    def fit(
        self,
        update_freq: int = 1,
        disp: Literal["off", "final"] | bool = "final",
        starting_values: ArrayLike1D | None = None,
        cov_type: Literal["robust", "classic"] = "robust",
        show_warning: bool = True,
        first_obs: int | DateLike | None = None,
        last_obs: int | DateLike | None = None,
        tol: float | None = None,
        options: dict[str, Any] | None = None,
        backcast: float | Float64Array1D | None = None,
    ) -> "ARCHModelResult":
        r"""
        Estimate model parameters

        Parameters
        ----------
        update_freq : int, optional
            Frequency of iteration updates.  Output is generated every
            `update_freq` iterations. Set to 0 to disable iterative output.
        disp : {bool, "off", "final"}
            Either 'final' to print optimization result or 'off' to display
            nothing. If using a boolean, False is "off" and True is "final"
        starting_values : ndarray, optional
            Array of starting values to use.  If not provided, starting values
            are constructed by the model components.
        cov_type : str, optional
            Estimation method of parameter covariance.  Supported options are
            'robust', which does not assume the Information Matrix Equality
            holds and 'classic' which does.  In the ARCH literature, 'robust'
            corresponds to Bollerslev-Wooldridge covariance estimator.
        show_warning : bool, optional
            Flag indicating whether convergence warnings should be shown.
        first_obs : {int, str, datetime, Timestamp}
            First observation to use when estimating model
        last_obs : {int, str, datetime, Timestamp}
            Last observation to use when estimating model
        tol : float, optional
            Tolerance for termination.
        options : dict, optional
            Options to pass to `scipy.optimize.minimize`.  Valid entries
            include 'ftol', 'eps', 'disp', and 'maxiter'.
        backcast : {float, ndarray}, optional
            Value to use as backcast. Should be measure :math:`\sigma^2_0`
            since model-specific non-linear transformations are applied to
            value before computing the variance recursions.

        Returns
        -------
        results : ARCHModelResult
            Object containing model results

        Notes
        -----
        A ConvergenceWarning is raised if SciPy's optimizer indicates
        difficulty finding the optimum.

        Parameters are optimized using SLSQP.
        """
        if self._y_original is None:
            raise RuntimeError("Cannot estimate model without data.")
        # 1. Check in ARCH or Non-normal dist.  If no ARCH and normal,
        # use closed form
        v, d = self.volatility, self.distribution
        offsets = np.array((self.num_params, v.num_params, d.num_params), dtype=int)
        total_params = sum(offsets)

        # Closed form is applicable when model has no parameters
        # Or when distribution is normal and constant variance
        has_closed_form = (
            v.closed_form and d.num_params == 0 and isinstance(v, ConstantVariance)
        )

        self._adjust_sample(first_obs, last_obs)

        resids = self.resids(self.starting_values())
        self._check_scale(resids)
        if self.scale != 1.0:
            # Scale changed, rescale data and reset model
            self._y = to_array_1d(
                self.scale * ensure1d(self._y_original, "y", series=False)
            )
            self._scale_changed()
            self._adjust_sample(first_obs, last_obs)
            resids = self.resids(self.starting_values())

        if backcast is None:
            backcast = v.backcast(resids)
        else:
            assert backcast is not None
            backcast = v.backcast_transform(backcast)

        if has_closed_form:
            try:
                return self._fit_no_arch_normal_errors(cov_type=cov_type)
            except NotImplementedError:
                pass
        assert backcast is not None
        if total_params == 0:
            return self._fit_parameterless_model(cov_type=cov_type, backcast=backcast)

        sigma2 = np.zeros(resids.shape[0], dtype=float)
        self._backcast = backcast
        sv_volatility = v.starting_values(resids)
        self._var_bounds = var_bounds = v.variance_bounds(resids)
        v.compute_variance(sv_volatility, resids, sigma2, backcast, var_bounds)
        std_resids = resids / np.sqrt(sigma2)

        # 2. Construct constraint matrices from all models and distribution
        constraints = (
            self.constraints(),
            self.volatility.constraints(),
            self.distribution.constraints(),
        )
        num_cons = []
        for c in constraints:
            assert c is not None
            num_cons.append(c[0].shape[0])
        num_constraints = np.array(num_cons, dtype=int)
        num_params = offsets.sum()
        a = np.zeros((int(num_constraints.sum()), int(num_params)))
        b = np.zeros(int(num_constraints.sum()))

        for i, c in enumerate(constraints):
            assert c is not None
            r_en = num_constraints[: i + 1].sum()
            c_en = offsets[: i + 1].sum()
            r_st = r_en - num_constraints[i]
            c_st = c_en - offsets[i]

            if r_en - r_st > 0:
                a[r_st:r_en, c_st:c_en] = c[0]
                b[r_st:r_en] = c[1]

        bounds = self.bounds()
        bounds.extend(v.bounds(resids))
        bounds.extend(d.bounds(std_resids))

        # 3. Construct starting values from all models
        if starting_values is None:
            sv = to_array_1d(
                np.hstack(
                    [
                        self.starting_values(),
                        sv_volatility,
                        d.starting_values(std_resids),
                    ]
                )
            )
        else:
            assert starting_values is not None
            sv = np.asarray(ensure1d(starting_values, "starting_values"), dtype=float)
            assert isinstance(sv, (np.ndarray, pd.Series))
            valid = sv.shape[0] == num_params
            if a.shape[0] > 0:
                satisfies_constraints = a.dot(sv) - b >= 0
                valid = valid and satisfies_constraints.all()
            for i, bound in enumerate(bounds):
                valid = valid and bound[0] <= sv[i] <= bound[1]
            if not valid:
                warnings.warn(
                    starting_value_warning, StartingValueWarning, stacklevel=2
                )
                starting_values = None

        # 4. Estimate models using constrained optimization
        _callback_info["count"], _callback_info["iter"] = 0, 0
        if not isinstance(disp, str):
            disp = bool(disp)
            disp = "off" if not disp else "final"
        if update_freq <= 0 or disp == "off":
            _callback_info["display"] = 2**31

        else:
            _callback_info["display"] = update_freq
        disp_flag = True if disp == "final" else False

        func = self._loglikelihood
        args = (sigma2, backcast, var_bounds)
        ineq_constraints = constraint(a, b)

        options = {} if options is None else options
        options.setdefault("disp", disp_flag)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Values in x were outside bounds during a minimize step",
                RuntimeWarning,
            )
            opt = minimize(  # type: ignore[call-overload]
                func,
                sv,
                args=args,
                method="SLSQP",
                bounds=bounds,
                constraints=ineq_constraints,
                tol=tol,
                callback=_callback,
                options=options,
            )

        if show_warning:
            warnings.filterwarnings("always", "", ConvergenceWarning)
        else:
            warnings.filterwarnings("ignore", "", ConvergenceWarning)

        if opt.status != 0 and show_warning:
            warnings.warn(
                convergence_warning.format(code=opt.status, string_message=opt.message),
                ConvergenceWarning,
                stacklevel=2,
            )

        # 5. Return results
        params = opt.x
        loglikelihood = -1.0 * opt.fun

        mp, vp, _ = self._parse_parameters(params)

        resids = self.resids(mp)
        vol = np.zeros(resids.shape[0], dtype=float)
        self.volatility.compute_variance(vp, resids, vol, backcast, var_bounds)
        vol = cast("Float64Array1D", np.sqrt(vol))

        try:
            r2 = self._r2(mp)
        except NotImplementedError:
            r2 = np.nan

        names = self._all_parameter_names()
        # Reshape resids and vol
        first_obs, last_obs = self._fit_indices
        resids_final = np.full(self._y.shape, np.nan, dtype=float)
        resids_final[first_obs:last_obs] = resids
        vol_final = np.full(self._y.shape, np.nan, dtype=float)
        vol_final[first_obs:last_obs] = vol

        fit_start, fit_stop = self._fit_indices
        model_copy = deepcopy(self)
        assert isinstance(r2, float)
        return ARCHModelResult(
            params,
            None,
            r2,
            resids_final,
            vol_final,
            cov_type,
            self._y_series,
            names,
            loglikelihood,
            self._is_pandas,
            opt,
            fit_start,
            fit_stop,
            model_copy,
        )

    @abstractmethod
    def parameter_names(self) -> list[str]:
        """List of parameters names

        Returns
        -------
        names : list (str)
            List of variable names for the mean model
        """

    def starting_values(self) -> Float64Array1D:
        """
        Returns starting values for the mean model, often the same as the
        values returned from fit

        Returns
        -------
        sv : ndarray
            Starting values
        """
        return self._fit_no_arch_normal_errors_params()

    @cached_property
    @abstractmethod
    def num_params(self) -> int:
        """
        Number of parameters in the model
        """

    @abstractmethod
    def simulate(
        self,
        params: ArrayLike1D | Sequence[float],
        nobs: int,
        burn: int = 500,
        initial_value: float | None = None,
        x: ArrayLike | None = None,
        initial_value_vol: float | None = None,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def resids(
        self,
        params: Float64Array1D,
        y: ArrayLike1D | None = None,
        regressors: ArrayLike2D | None = None,
    ) -> ArrayLike1D:
        """
        Compute model residuals

        Parameters
        ----------
        params : ndarray
            Model parameters
        y : ndarray, optional
            Alternative values to use when computing model residuals
        regressors : ndarray, optional
            Alternative regressor values to use when computing model residuals

        Returns
        -------
        resids : ndarray
            Model residuals
        """

    def compute_param_cov(
        self,
        params: Float64Array1D,
        backcast: float | Float64Array1D | None = None,
        robust: bool = True,
    ) -> Float64Array:
        """
        Computes parameter covariances using numerical derivatives.

        Parameters
        ----------
        params : ndarray
            Model parameters
        backcast : float
            Value to use for pre-sample observations
        robust : bool, optional
            Flag indicating whether to use robust standard errors (True) or
            classic MLE (False)

        """
        resids = self.resids(self.starting_values())
        var_bounds = self.volatility.variance_bounds(resids)
        nobs = resids.shape[0]
        if backcast is None and self._backcast is None:
            backcast = self.volatility.backcast(resids)
            self._backcast = backcast
        elif backcast is None:
            backcast = self._backcast

        kwargs = {
            "sigma2": np.zeros(resids.shape[0], dtype=float),
            "backcast": backcast,
            "var_bounds": var_bounds,
            "individual": False,
        }

        hess = approx_hess(params, self._loglikelihood, kwargs=kwargs)
        hess /= nobs
        inv_hess = np.linalg.inv(hess)
        if robust:
            kwargs["individual"] = True
            scores = approx_fprime(
                params, self._loglikelihood, kwargs=kwargs
            )  # type: np.ndarray
            score_cov = np.cov(scores.T)
            return inv_hess.dot(score_cov).dot(inv_hess) / nobs
        else:
            return inv_hess / nobs

    @abstractmethod
    def forecast(
        self,
        params: ArrayLike1D,
        horizon: int = 1,
        start: int | DateLike | None = None,
        align: Literal["origin", "target"] = "origin",
        method: ForecastingMethod = "analytic",
        simulations: int = 1000,
        rng: Callable[[int | tuple[int, ...]], Float64Array] | None = None,
        random_state: np.random.RandomState | None = None,
        *,
        reindex: bool = False,
        x: dict[Label, ArrayLike] | ArrayLike | None = None,
    ) -> "ARCHModelForecast":
        """
        Construct forecasts from estimated model

        Parameters
        ----------
        params : {ndarray, Series}
            Parameters required to forecast. Must be identical in
            shape to the parameters computed by fitting the model.
        horizon : int, optional
           Number of steps to forecast
        start : {int, datetime, Timestamp, str}, optional
            An integer, datetime or str indicating the first observation to
            produce the forecast for.  Datetimes can only be used with pandas
            inputs that have a datetime index. Strings must be convertible
            to a date time, such as in '1945-01-01'.
        align : str, optional
            Either 'origin' or 'target'.  When set of 'origin', the t-th row
            of forecasts contains the forecasts for t+1, t+2, ..., t+h. When
            set to 'target', the t-th row contains the 1-step ahead forecast
            from time t-1, the 2 step from time t-2, ..., and the h-step from
            time t-h.  'target' simplified computing forecast errors since the
            realization and h-step forecast are aligned.
        method : {'analytic', 'simulation', 'bootstrap'}
            Method to use when producing the forecast. The default is analytic.
            The method only affects the variance forecast generation.  Not all
            volatility models support all methods. In particular, volatility
            models that do not evolve in squares such as EGARCH or TARCH do not
            support the 'analytic' method for horizons > 1.
        simulations : int
            Number of simulations to run when computing the forecast using
            either simulation or bootstrap.
        rng : callable, optional
            Custom random number generator to use in simulation-based forecasts.
            Must produce random samples using the syntax `rng(size)` where size
            the 2-element tuple (simulations, horizon).
        random_state : RandomState, optional
            NumPy RandomState instance to use when method is 'bootstrap'
        reindex : bool, optional
            Whether to reindex the forecasts to have the same dimension as the series
            being forecast. Prior to 4.18 this was the default. As of 4.19 this is
            now optional. If not provided, a warning is raised about the future
            change in the default which will occur after September 2021.

            .. versionadded:: 4.19

        x : {dict[label, array_like], array_like}
            Values to use for exogenous regressors if any are included in the
            model. Three formats are accepted:

            * 2-d array-like: This format can be used when there is a single
              exogenous variable. The input must have shape (nforecast, horizon)
              or (nobs, horizon) where nforecast is the number of forecasting
              periods and nobs is the original shape of y. For example, if a
              single series of forecasts are made from the end of the sample
              with a horizon of 10, then the input can be (1, 10). Alternatively,
              if the original data had 1000 observations, then the input can be
              (1000, 10), and only the final row is used to produce forecasts.
              When using an (nobs, horizon) array, the values much be aligned
              so that all values in row t are all out-of-sample at time-t.
            * A dictionary of 2-d array-like: This format is identical to the
              previous except that the dictionary keys must match the names of
              the exog variables.  Requires that the exog variables were
              passed as a pandas DataFrame.
            * A 3-d NumPy array (or equivalent). In this format, each panel
              (0th axis) is a 2-d array that must have shape (nforecast, horizon)
              or (nobs,horizon). The array x[j] corresponds to the j-th column of
              the exogenous variables.

            Due to the complexity required to accommodate all scenarios, please
            see the example notebook that demonstrates the valid formats for
            x, and discusses alignment.

            .. versionadded:: 4.19

        Returns
        -------
        arch.univariate.base.ARCHModelForecast
            Container for forecasts. Key properties are ``mean``,
            ``variance`` and ``residual_variance``.

        Examples
        --------
        >>> import pandas as pd
        >>> from arch import arch_model
        >>> am = arch_model(None,mean='HAR',lags=[1,5,22],vol='Constant')
        >>> sim_data = am.simulate([0.1,0.4,0.3,0.2,1.0], 250)
        >>> sim_data.index = pd.date_range('2000-01-01',periods=250)
        >>> am = arch_model(sim_data['data'],mean='HAR',lags=[1,5,22],  vol='Constant')
        >>> res = am.fit()
        >>> fig = res.hedgehog_plot()

        Notes
        -----
        The most basic 1-step ahead forecast will return a vector with the same
        length as the original data, where the t-th value will be the time-t
        forecast for time t + 1.  When the horizon is > 1, and when using the
        default value for `align`, the forecast value in position [t, h] is the
        time-t, h+1 step ahead forecast.

        If model contains exogenous variables (model.x is not None), then
        only 1-step ahead forecasts are available.  Using horizon > 1 will
        produce a warning and all columns, except the first, will be
        nan-filled.

        If `align` is 'origin', forecast[t,h] contains the forecast made using
        y[:t] (that is, up to but not including t) for horizon h + 1.  For
        example, y[100,2] contains the 3-step ahead forecast using the first
        100 data points, which will correspond to the realization y[100 + 2].
        If `align` is 'target', then the same forecast is in location
        [102, 2], so that it is aligned with the observation to use when
        evaluating, but still in the same column.
        """


class _SummaryRepr:
    """Base class for returning summary as repr and str"""

    def summary(self) -> Summary:
        raise NotImplementedError("Subclasses must implement")

    def __repr__(self) -> str:
        out = self.__str__() + "\n"
        out += self.__class__.__name__
        out += f", id: {hex(id(self))}"
        return out

    def __str__(self) -> str:
        return self.summary().as_text()


class ARCHModelFixedResult(_SummaryRepr):
    """
    Results for fixed parameters for an ARCHModel model

    Parameters
    ----------
    params : ndarray
        Estimated parameters
    resid : ndarray
        Residuals from model.  Residuals have same shape as original data and
        contain nan-values in locations not used in estimation
    volatility : ndarray
        Conditional volatility from model
    dep_var : Series
        Dependent variable
    names : list (str)
        Model parameter names
    loglikelihood : float
        Loglikelihood at specified parameters
    is_pandas : bool
        Whether the original input was pandas
    model : ARCHModel
        The model object used to estimate the parameters
    """

    def __init__(
        self,
        params: Float64Array1D,
        resid: Float64Array1D,
        volatility: Float64Array1D,
        dep_var: pd.Series,
        names: list[str],
        loglikelihood: float,
        is_pandas: bool,
        model: ARCHModel,
    ) -> None:
        self._params = params
        self._resid = resid
        self._is_pandas = is_pandas
        self._model = model
        self._datetime = dt.datetime.now()
        self._dep_var = dep_var
        self._dep_name = str(dep_var.name)
        self._names = list(names)
        self._loglikelihood = loglikelihood
        self._nobs = self.model._fit_y.shape[0]
        self._index = dep_var.index
        self._volatility = volatility

    def summary(self) -> Summary:
        """
        Constructs a summary of the results from a fit model.

        Returns
        -------
        summary : Summary instance
            Object that contains tables and facilitated export to text, html or
            latex
        """
        # Summary layout
        # 1. Overall information
        # 2. Mean parameters
        # 3. Volatility parameters
        # 4. Distribution parameters
        # 5. Notes

        model = self.model
        model_name = model.name + " - " + model.volatility.name

        # Summary Header
        top_left = [
            ("Dep. Variable:", self._dep_name),
            ("Mean Model:", model.name),
            ("Vol Model:", model.volatility.name),
            ("Distribution:", model.distribution.name),
            ("Method:", "User-specified Parameters"),
            ("", ""),
            ("Date:", self._datetime.strftime("%a, %b %d %Y")),
            ("Time:", self._datetime.strftime("%H:%M:%S")),
        ]

        top_right = [
            ("R-squared:", "--"),
            ("Adj. R-squared:", "--"),
            ("Log-Likelihood:", f"{self.loglikelihood:#10.6g}"),
            ("AIC:", f"{self.aic:#10.6g}"),
            ("BIC:", f"{self.bic:#10.6g}"),
            ("No. Observations:", f"{self._nobs}"),
            ("", ""),
            ("", ""),
        ]

        title = model_name + " Model Results"
        stubs = []
        vals = []
        for stub, val in top_left:
            stubs.append(stub)
            vals.append([val])
        table = SimpleTable(vals, txt_fmt=fmt_2cols, title=title, stubs=stubs)

        # create summary table instance
        smry = Summary()
        # Top Table
        # Parameter table
        fmt = fmt_2cols
        fmt["data_fmts"][1] = "%18s"

        top_right = [("%-21s" % ("  " + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))
        smry.tables.append(table)

        stubs = list(self._names)
        header = ["coef"]
        param_table_data = [[format_float_fixed(param, 10, 4)] for param in self.params]

        mc = self.model.num_params
        vc = self.model.volatility.num_params
        dc = self.model.distribution.num_params
        counts = (mc, vc, dc)
        titles = ("Mean Model", "Volatility Model", "Distribution")
        total = 0
        for title, count in zip(titles, counts, strict=False):
            if count == 0:
                continue

            table_data = param_table_data[total : total + count]
            table_stubs = stubs[total : total + count]
            total += count
            table = SimpleTable(
                table_data,
                stubs=table_stubs,
                txt_fmt=fmt_params,
                headers=header,
                title=title,
            )
            smry.tables.append(table)

        extra_text = [
            "Results generated with user-specified parameters.",
            "Std. errors not available when the model is not estimated, ",
        ]
        smry.add_extra_txt(extra_text)
        return smry

    @cached_property
    def model(self) -> ARCHModel:
        """
        Model instance used to produce the fit
        """
        return self._model

    @cached_property
    def loglikelihood(self) -> float:
        """Model loglikelihood"""
        return self._loglikelihood

    @cached_property
    def aic(self) -> float:
        """Akaike Information Criteria

        -2 * loglikelihood + 2 * num_params"""
        return -2 * self.loglikelihood + 2 * self.num_params

    @cached_property
    def num_params(self) -> int:
        """Number of parameters in model"""
        return len(self.params)

    @cached_property
    def bic(self) -> float:
        """
        Schwarz/Bayesian Information Criteria

        -2 * loglikelihood + log(nobs) * num_params
        """
        return -2 * self.loglikelihood + np.log(self.nobs) * self.num_params

    @cached_property
    def params(self) -> pd.Series:
        """Model Parameters"""
        return pd.Series(self._params, index=self._names, name="params")

    @cached_property
    def conditional_volatility(self) -> Float64Array1D | pd.Series:
        """
        Estimated conditional volatility

        Returns
        -------
        conditional_volatility : {ndarray, Series}
            nobs element array containing the conditional volatility (square
            root of conditional variance).  The values are aligned with the
            input data so that the value in the t-th position is the variance
            of t-th error, which is computed using time-(t-1) information.
        """
        if self._is_pandas:
            return pd.Series(self._volatility, name="cond_vol", index=self._index)
        else:
            return self._volatility

    @cached_property
    def nobs(self) -> int:
        """
        Number of data points used to estimate model
        """
        return self._nobs

    @cached_property
    def resid(self) -> Float64Array1D | pd.Series:
        """
        Model residuals
        """
        if self._is_pandas:
            return pd.Series(self._resid, name="resid", index=self._index)
        else:
            return self._resid

    @cached_property
    def std_resid(self) -> Float64Array1D | pd.Series:
        """
        Residuals standardized by conditional volatility
        """
        std_res = self.resid / self.conditional_volatility
        if isinstance(std_res, pd.Series):
            std_res.name = "std_resid"
        return std_res

    def plot(
        self, annualize: str | None = None, scale: float | None = None
    ) -> "Figure":
        """
        Plot standardized residuals and conditional volatility

        Parameters
        ----------
        annualize : str, optional
            String containing frequency of data that indicates plot should
            contain annualized volatility.  Supported values are 'D' (daily),
            'W' (weekly) and 'M' (monthly), which scale variance by 252, 52,
            and 12, respectively.
        scale : float, optional
            Value to use when scaling returns to annualize.  If scale is
            provided, annualize is ignored and the value in scale is used.

        Returns
        -------
        fig : figure
            Handle to the figure

        Examples
        --------
        >>> from arch import arch_model
        >>> am = arch_model(None)
        >>> sim_data = am.simulate([0.0, 0.01, 0.07, 0.92], 2520)
        >>> am = arch_model(sim_data['data'])
        >>> res = am.fit(update_freq=0, disp='off')
        >>> fig = res.plot()

        Produce a plot with annualized volatility

        >>> fig = res.plot(annualize='D')

        Override the usual scale of 252 to use 360 for an asset that trades
        most days of the year

        >>> fig = res.plot(scale=360)
        """
        from matplotlib.axes import Axes  # noqa: PLC0415
        from matplotlib.pyplot import figure  # noqa: PLC0415

        def _set_tight_x(axis: Axes, index: pd.Index) -> None:
            try:
                axis.set_xlim(index[0], index[-1])
            except ValueError:
                pass

        fig = figure()

        ax = fig.add_subplot(2, 1, 1)
        ax.plot(self._index.values, self.resid / self.conditional_volatility)
        ax.set_title("Standardized Residuals")
        ax.set_xticklabels([])
        _set_tight_x(ax, self._index)

        ax = fig.add_subplot(2, 1, 2)
        vol = self.conditional_volatility
        title = "Annualized Conditional Volatility"
        if scale is not None:
            vol = vol * np.sqrt(scale)
        elif annualize is not None:
            scales = {"D": 252, "W": 52, "M": 12}
            if annualize in scales:
                vol = vol * np.sqrt(scales[annualize])
            else:
                raise ValueError("annualize not recognized")
        else:
            title = "Conditional Volatility"

        ax.plot(self._index.values, vol)
        _set_tight_x(ax, self._index)
        ax.set_title(title)

        return fig

    def forecast(
        self,
        params: ArrayLike1D | None = None,
        horizon: int = 1,
        start: int | DateLike | None = None,
        align: Literal["origin", "target"] = "origin",
        method: ForecastingMethod = "analytic",
        simulations: int = 1000,
        rng: Callable[[int | tuple[int, ...]], Float64Array] | None = None,
        random_state: np.random.RandomState | None = None,
        *,
        reindex: bool = False,
        x: dict[Label, ArrayLike] | ArrayLike | None = None,
    ) -> "ARCHModelForecast":
        """
        Construct forecasts from estimated model

        Parameters
        ----------
        params : ndarray, optional
            Alternative parameters to use.  If not provided, the parameters
            estimated when fitting the model are used.  Must be identical in
            shape to the parameters computed by fitting the model.
        horizon : int, optional
           Number of steps to forecast
        start : {int, datetime, Timestamp, str}, optional
            An integer, datetime or str indicating the first observation to
            produce the forecast for.  Datetimes can only be used with pandas
            inputs that have a datetime index. Strings must be convertible
            to a date time, such as in '1945-01-01'.
        align : str, optional
            Either 'origin' or 'target'.  When set of 'origin', the t-th row
            of forecasts contains the forecasts for t+1, t+2, ..., t+h. When
            set to 'target', the t-th row contains the 1-step ahead forecast
            from time t-1, the 2 step from time t-2, ..., and the h-step from
            time t-h.  'target' simplified computing forecast errors since the
            realization and h-step forecast are aligned.
        method : {'analytic', 'simulation', 'bootstrap'}, optional
            Method to use when producing the forecast. The default is analytic.
            The method only affects the variance forecast generation.  Not all
            volatility models support all methods. In particular, volatility
            models that do not evolve in squares such as EGARCH or TARCH do not
            support the 'analytic' method for horizons > 1.
        simulations : int, optional
            Number of simulations to run when computing the forecast using
            either simulation or bootstrap.
        rng : callable, optional
            Custom random number generator to use in simulation-based forecasts.
            Must produce random samples using the syntax `rng(size)` where size
            the 2-element tuple (simulations, horizon).
        random_state : RandomState, optional
            NumPy RandomState instance to use when method is 'bootstrap'
        reindex : bool, optional
            Whether to reindex the forecasts to have the same dimension as the series
            being forecast.

            .. versionchanged:: 6.2

               The default has been changed to False.

        x : {dict[label, array_like], array_like}
            Values to use for exogenous regressors if any are included in the
            model. Three formats are accepted:

            * 2-d array-like: This format can be used when there is a single
              exogenous variable. The input must have shape (nforecast, horizon)
              or (nobs, horizon) where nforecast is the number of forecasting
              periods and nobs is the original shape of y. For example, if a
              single series of forecasts are made from the end of the sample
              with a horizon of 10, then the input can be (1, 10). Alternatively,
              if the original data had 1000 observations, then the input can be
              (1000, 10), and only the final row is used to produce forecasts.
            * A dictionary of 2-d array-like: This format is identical to the
              previous except that the dictionary keys must match the names of
              the exog variables.  Requires that the exog variables were passed
              as a pandas DataFrame.
            * A 3-d NumPy array (or equivalent). In this format, each panel
              (0th axis) is a 2-d array that must have shape (nforecast, horizon)
              or (nobs,horizon). The array x[j] corresponds to the j-th column of
              the exogenous variables.

            Due to the complexity required to accommodate all scenarios, please
            see the example notebook that demonstrates the valid formats for
            x.

            .. versionadded:: 4.19

        Returns
        -------
        arch.univariate.base.ARCHModelForecast
            Container for forecasts. Key properties are ``mean``,
            ``variance`` and ``residual_variance``.

        Notes
        -----
        The most basic 1-step ahead forecast will return a vector with the same
        length as the original data, where the t-th value will be the time-t
        forecast for time t + 1.  When the horizon is > 1, and when using the
        default value for `align`, the forecast value in position [t, h] is the
        time-t, h+1 step ahead forecast.

        If model contains exogenous variables (`model.x is not None`), then
        only 1-step ahead forecasts are available.  Using horizon > 1 will
        produce a warning and all columns, except the first, will be
        nan-filled.

        If `align` is 'origin', forecast[t,h] contains the forecast made using
        y[:t] (that is, up to but not including t) for horizon h + 1.  For
        example, y[100,2] contains the 3-step ahead forecast using the first
        100 data points, which will correspond to the realization y[100 + 2].
        If `align` is 'target', then the same forecast is in location
        [102, 2], so that it is aligned with the observation to use when
        evaluating, but still in the same column.
        """
        if params is None:
            params = self._params
        elif (
            params.size != np.array(self._params).size
            or params.ndim != self._params.ndim
        ):
            raise ValueError("params have incorrect dimensions")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be an integer >= 1.")
        return self.model.forecast(
            params,
            horizon,
            start,
            align,
            method,
            simulations,
            rng,
            random_state,
            reindex=reindex,
            x=x,
        )

    @deprecate_kwarg("type", "plot_type")
    def hedgehog_plot(
        self,
        params: ArrayLike1D | None = None,
        horizon: int = 10,
        step: int = 10,
        start: int | DateLike | None = None,
        plot_type: Literal["volatility", "mean"] = "volatility",
        method: ForecastingMethod = "analytic",
        simulations: int = 1000,
    ) -> "Figure":
        """
        Plot forecasts from estimated model

        Parameters
        ----------
        params : {ndarray, Series}
            Alternative parameters to use.  If not provided, the parameters
            computed by fitting the model are used.  Must be 1-d and identical
            in shape to the parameters computed by fitting the model.
        horizon : int, optional
            Number of steps to forecast
        step : int, optional
            Non-negative number of forecasts to skip between spines
        start : int, datetime or str, optional
            An integer, datetime or str indicating the first observation to
            produce the forecast for.  Datetimes can only be used with pandas
            inputs that have a datetime index.  Strings must be convertible
            to a date time, such as in '1945-01-01'.  If not provided, the start
            is set to the earliest forecastable date.
        plot_type : {'volatility', 'mean'}
            Quantity to plot, the forecast volatility or the forecast mean
        method : {'analytic', 'simulation', 'bootstrap'}
            Method to use when producing the forecast. The default is analytic.
            The method only affects the variance forecast generation.  Not all
            volatility models support all methods. In particular, volatility
            models that do not evolve in squares such as EGARCH or TARCH do not
            support the 'analytic' method for horizons > 1.
        simulations : int
            Number of simulations to run when computing the forecast using
            either simulation or bootstrap.

        Returns
        -------
        fig : figure
            Handle to the figure

        Examples
        --------
        >>> import pandas as pd
        >>> from arch import arch_model
        >>> am = arch_model(None,mean='HAR',lags=[1,5,22],vol='Constant')
        >>> sim_data = am.simulate([0.1,0.4,0.3,0.2,1.0], 250)
        >>> sim_data.index = pd.date_range('2000-01-01',periods=250)
        >>> am = arch_model(sim_data['data'],mean='HAR',lags=[1,5,22],  vol='Constant')
        >>> res = am.fit()
        >>> fig = res.hedgehog_plot(plot_type='mean')
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        plot_mean = plot_type.lower() == "mean"
        if start is None:
            invalid_start = True
            start = 0
            while invalid_start:
                try:
                    forecasts = self.forecast(
                        params,
                        horizon,
                        start,
                        method=method,
                        simulations=simulations,
                    )
                    invalid_start = False
                except ValueError:  # noqa: PERF203
                    start += 1
        else:
            forecasts = self.forecast(
                params,
                horizon,
                start,
                method=method,
                simulations=simulations,
            )

        fig, ax = plt.subplots(1, 1)
        use_date = isinstance(self._dep_var.index, pd.DatetimeIndex)
        plot_fn = ax.plot
        if MPL_LT_310 and hasattr(ax, "plot_date") and use_date:
            plot_fn = ax.plot_date
        x_values = np.array(self._dep_var.index)
        if plot_mean:
            y_values = np.asarray(self._dep_var)
        else:
            y_values = np.asarray(self.conditional_volatility)

        plot_fn(x_values, y_values, linestyle="-", marker="")
        first_obs = np.min(np.where(np.logical_not(np.isnan(forecasts.mean)))[0])
        spines = []
        t = forecasts.mean.shape[0]
        for i in range(first_obs, t, step):
            if i + horizon + 1 > x_values.shape[0]:
                continue
            temp_x = x_values[i : i + horizon + 1]
            if plot_mean:
                spine_data = np.asarray(forecasts.mean.iloc[i], dtype=float)
            else:
                spine_data = np.asarray(
                    np.sqrt(forecasts.variance.iloc[i]), dtype=float
                )
            temp_y = np.hstack([y_values[i], spine_data])
            line = plot_fn(temp_x, temp_y, linewidth=3, linestyle="-", marker="")
            spines.append(line)
        color = spines[0][0].get_color()
        for spine in spines[1:]:
            spine[0].set_color(color)
        plot_title = "Mean" if plot_mean else "Volatility"
        ax.set_title(self._dep_name + " " + plot_title + " Forecast Hedgehog Plot")

        return fig

    def arch_lm_test(
        self, lags: int | None = None, standardized: bool = False
    ) -> WaldTestStatistic:
        """
        ARCH LM test for conditional heteroskedasticity

        Parameters
        ----------
        lags : int, optional
            Number of lags to include in the model.  If not specified,
        standardized : bool, optional
            Flag indicating to test the model residuals divided by their
            conditional standard deviations.  If False, directly tests the
            estimated residuals.

        Returns
        -------
        result : WaldTestStatistic
            Result of ARCH-LM test
        """
        resids = self.resid
        if standardized:
            resids = resids / np.asarray(self.conditional_volatility)
        # GH 599: drop missing observations
        resids = resids[~np.isnan(resids)]
        nobs = resids.shape[0]
        resid2 = resids**2
        lags = (
            int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
            if lags is None
            else lags
        )
        lags = max(min(resids.shape[0] // 2 - 1, lags), 1)
        if resid2.shape[0] < 3:
            raise ValueError("Test requires at least 3 non-nan observations")
        lag, lead = lagmat(resid2, lags, "both", "sep", False)
        lag = add_constant(lag)
        res = OLS(lead, lag).fit()
        stat = nobs * res.rsquared
        test_type = "R" if not standardized else "Standardized r"
        null = f"{test_type}esiduals are homoskedastic."
        alt = f"{test_type}esiduals are conditionally heteroskedastic."
        assert isinstance(lags, int)
        return WaldTestStatistic(
            stat, df=lags, null=null, alternative=alt, name="ARCH-LM Test"
        )


class ARCHModelResult(ARCHModelFixedResult):
    """
    Results from estimation of an ARCHModel model

    Parameters
    ----------
    params : ndarray
        Estimated parameters
    param_cov : {ndarray, None}
        Estimated variance-covariance matrix of params.  If none, calls method
        to compute variance from model when parameter covariance is first used
        from result
    r2 : float
        Model R-squared
    resid : ndarray
        Residuals from model.  Residuals have same shape as original data and
        contain nan-values in locations not used in estimation
    volatility : ndarray
        Conditional volatility from model
    cov_type : str
        String describing the covariance estimator used
    dep_var : Series
        Dependent variable
    names : list (str)
        Model parameter names
    loglikelihood : float
        Loglikelihood at estimated parameters
    is_pandas : bool
        Whether the original input was pandas
    optim_output : OptimizeResult
        Result of log-likelihood optimization
    fit_start : int
        Integer index of the first observation used to fit the model
    fit_stop : int
        Integer index of the last observation used to fit the model using
        slice notation `fit_start:fit_stop`
    model : ARCHModel
        The model object used to estimate the parameters
    """

    def __init__(
        self,
        params: Float64Array1D,
        param_cov: Float64Array | None,
        r2: float,
        resid: Float64Array1D,
        volatility: Float64Array1D,
        cov_type: str,
        dep_var: pd.Series,
        names: list[str],
        loglikelihood: float,
        is_pandas: bool,
        optim_output: OptimizeResult,
        fit_start: int,
        fit_stop: int,
        model: ARCHModel,
    ) -> None:
        super().__init__(
            params, resid, volatility, dep_var, names, loglikelihood, is_pandas, model
        )

        self._fit_indices = (fit_start, fit_stop)
        self._param_cov = param_cov
        self._r2 = r2
        self.cov_type: str = cov_type
        self._optim_output = optim_output

    @cached_property
    def scale(self) -> float:
        """
        The scale applied to the original data before estimating the model.

        If scale=1.0, the the data have not been rescaled.  Otherwise, the
        model parameters have been estimated on scale * y.
        """
        return self.model.scale

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Parameter confidence intervals

        Parameters
        ----------
        alpha : float, optional
            Size (prob.) to use when constructing the confidence interval.

        Returns
        -------
        ci : DataFrame
            Array where the ith row contains the confidence interval  for the
            ith parameter
        """
        cv = stats.norm.ppf(1.0 - alpha / 2.0)
        se = self.std_err
        params = self.params

        return pd.DataFrame(
            np.vstack((params - cv * se, params + cv * se)).T,
            columns=["lower", "upper"],
            index=self._names,
        )

    def summary(self) -> Summary:
        """
        Constructs a summary of the results from a fit model.

        Returns
        -------
        summary : Summary instance
            Object that contains tables and facilitated export to text, html or
            latex
        """
        # Summary layout
        # 1. Overall information
        # 2. Mean parameters
        # 3. Volatility parameters
        # 4. Distribution parameters
        # 5. Notes

        model = self.model
        model_name = model.name + " - " + model.volatility.name

        # Summary Header
        top_left = [
            ("Dep. Variable:", self._dep_name),
            ("Mean Model:", model.name),
            ("Vol Model:", model.volatility.name),
            ("Distribution:", model.distribution.name),
            ("Method:", "Maximum Likelihood"),
            ("", ""),
            ("Date:", self._datetime.strftime("%a, %b %d %Y")),
            ("Time:", self._datetime.strftime("%H:%M:%S")),
        ]

        top_right = [
            ("R-squared:", f"{self.rsquared:#8.3f}"),
            ("Adj. R-squared:", f"{self.rsquared_adj:#8.3f}"),
            ("Log-Likelihood:", f"{self.loglikelihood:#10.6g}"),
            ("AIC:", f"{self.aic:#10.6g}"),
            ("BIC:", f"{self.bic:#10.6g}"),
            ("No. Observations:", f"{self._nobs}"),
            ("Df Residuals:", f"{self.nobs - self.model.num_params}"),
            ("Df Model:", f"{self.model.num_params}"),
        ]

        title = model_name + " Model Results"
        stubs = []
        vals = []
        for stub, val in top_left:
            stubs.append(stub)
            vals.append([val])
        table = SimpleTable(vals, txt_fmt=fmt_2cols, title=title, stubs=stubs)

        # create summary table instance
        smry = Summary()
        # Top Table
        # Parameter table
        fmt = fmt_2cols
        fmt["data_fmts"][1] = "%18s"

        top_right = [("%-21s" % ("  " + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))
        smry.tables.append(table)

        conf_int = np.asarray(self.conf_int())
        conf_int_str = [
            "["
            + format_float_fixed(c[0], 7, 3)
            + ","
            + format_float_fixed(c[1], 7, 3)
            + "]"
            for c in conf_int
        ]

        stubs = list(self._names)
        header = ["coef", "std err", "t", "P>|t|", "95.0% Conf. Int."]
        table_vals: tuple[np.ndarray | pd.Series, ...] = (
            np.asarray(self.params),
            np.asarray(self.std_err),
            np.asarray(self.tvalues),
            np.asarray(self.pvalues),
            pd.Series(conf_int_str),
        )
        # (0,0) is a dummy format
        formats = [(10, 4), (9, 3), (9, 3), (9, 3), (0, 0)]
        param_table_data = []
        for pos in range(len(table_vals[0])):
            row = []
            for i, table_val in enumerate(table_vals):
                val = table_val[pos]
                if isinstance(val, (np.double, float)):
                    converted = format_float_fixed(val, *formats[i])
                else:
                    converted = val
                row.append(converted)
            param_table_data.append(row)

        mc = self.model.num_params
        vc = self.model.volatility.num_params
        dc = self.model.distribution.num_params
        counts = (mc, vc, dc)
        titles = ("Mean Model", "Volatility Model", "Distribution")
        total = 0
        for title, count in zip(titles, counts, strict=False):
            if count == 0:
                continue

            table_data = param_table_data[total : total + count]
            table_stubs = stubs[total : total + count]
            total += count
            table = SimpleTable(
                table_data,
                stubs=table_stubs,
                txt_fmt=fmt_params,
                headers=header,
                title=title,
            )
            smry.tables.append(table)

        extra_text = ["Covariance estimator: " + self.cov_type]

        if self.convergence_flag:
            string_message = self._optim_output.message
            extra_text.append(CONVERGENCE_WARNING.format(msg=string_message))

        smry.add_extra_txt(extra_text)
        return smry

    @cached_property
    def param_cov(self) -> pd.DataFrame:
        """Parameter covariance"""
        if self._param_cov is not None:
            param_cov = self._param_cov
        else:
            params = to_array_1d(self.params)
            if self.cov_type == "robust":
                param_cov = self.model.compute_param_cov(params)
            else:
                param_cov = self.model.compute_param_cov(params, robust=False)
        return pd.DataFrame(param_cov, columns=self._names, index=self._names)

    @cached_property
    def rsquared(self) -> float:
        """
        R-squared
        """
        return self._r2

    @cached_property
    def fit_start(self) -> int:
        """Start of sample used to estimate parameters"""
        return self._fit_indices[0]

    @cached_property
    def fit_stop(self) -> int:
        """End of sample used to estimate parameters"""
        return self._fit_indices[1]

    @cached_property
    def rsquared_adj(self) -> float:
        """
        Degree of freedom adjusted R-squared
        """
        return 1 - (
            (1 - self.rsquared) * (self.nobs - 1) / (self.nobs - self.model.num_params)
        )

    @cached_property
    def pvalues(self) -> pd.Series:
        """
        Array of p-values for the t-statistics
        """
        pvals = np.asarray(stats.norm.sf(np.abs(self.tvalues)) * 2, dtype=float)
        return pd.Series(pvals, index=self._names, name="pvalues")

    @cached_property
    def std_err(self) -> pd.Series:
        """
        Array of parameter standard errors
        """
        se = np.asarray(np.sqrt(np.diag(self.param_cov)), dtype=float)
        return pd.Series(se, index=self._names, name="std_err")

    @cached_property
    def tvalues(self) -> pd.Series:
        """
        Array of t-statistics testing the null that the coefficient are 0
        """
        tvalues = self.params / self.std_err
        tvalues.name = "tvalues"
        return tvalues

    @cached_property
    def convergence_flag(self) -> int:
        """
        scipy.optimize.minimize result flag
        """
        return self._optim_output.status

    @property
    def optimization_result(self) -> OptimizeResult:
        """
        Information about the convergence of the loglikelihood optimization

        Returns
        -------
        optim_result : OptimizeResult
            Result from numerical optimization of the log-likelihood.
        """
        return self._optim_output


def _align_forecast(
    f: pd.DataFrame, align: Literal["origin", "target"]
) -> pd.DataFrame:
    if align == "origin":
        return f
    elif align in ("target", "horizon"):
        for i, col in enumerate(f):
            f[col] = f[col].shift(i + 1)
        return f
    else:
        raise ValueError("Unknown alignment")


def _format_forecasts(
    values: Float64Array, index: list[Label] | pd.Index, start_index: int
) -> pd.DataFrame:
    horizon = values.shape[1]
    format_str = "{0:>0" + str(int(np.ceil(np.log10(horizon + 0.5)))) + "}"
    columns = ["h." + format_str.format(h + 1) for h in range(horizon)]
    forecasts = pd.DataFrame(
        values, index=index[start_index:], columns=columns, dtype="float"
    )
    return forecasts


class ARCHModelForecastSimulation:
    """
    Container for a simulation or bootstrap-based forecasts from an ARCH Model

    Parameters
    ----------
    index
    values
    residuals
    variances
    residual_variances
    """

    def __init__(
        self,
        index: list[Label] | pd.Index,
        values: Float64Array | None,
        residuals: Float64Array | None,
        variances: Float64Array | None,
        residual_variances: Float64Array | None,
    ) -> None:
        self._index = pd.Index(index)
        self._values = values
        self._residuals = residuals
        self._variances = variances
        self._residual_variances = residual_variances

    @property
    def index(self) -> pd.Index:
        """The index aligned to dimension 0 of the simulation paths"""
        return self._index

    @property
    def values(self) -> Float64Array | None:
        """The values of the process"""
        return self._values

    @property
    def residuals(self) -> Float64Array | None:
        """Simulated residuals used to produce the values"""
        return self._residuals

    @property
    def variances(self) -> Float64Array | None:
        """Simulated variances of the values"""
        return self._variances

    @property
    def residual_variances(self) -> Float64Array | None:
        """Simulated variance of the residuals"""
        return self._residual_variances


def _reindex(
    a: Float64Array | None, idx: list[Label] | pd.Index
) -> Float64Array | None:
    if a is None:
        return a
    assert a is not None
    actual = len(idx)
    obs = a.shape[0]
    if actual > obs:
        addition = np.full((actual - obs,) + a.shape[1:], np.nan)
        a = np.concatenate([addition, a])
    return a


class ARCHModelForecast:
    """
    Container for forecasts from an ARCH Model

    Parameters
    ----------
    index : {list, ndarray}
    mean : ndarray
    variance : ndarray
    residual_variance : ndarray
    simulated_paths : ndarray, optional
    simulated_variances : ndarray, optional
    simulated_residual_variances : ndarray, optional
    simulated_residuals : ndarray, optional
    align : {'origin', 'target'}
    """

    def __init__(
        self,
        index: list[Label] | pd.Index,
        start_index: int,
        mean: Float64Array,
        variance: Float64Array,
        residual_variance: Float64Array,
        simulated_paths: Float64Array | None = None,
        simulated_variances: Float64Array | None = None,
        simulated_residual_variances: Float64Array | None = None,
        simulated_residuals: Float64Array | None = None,
        align: Literal["origin", "target"] = "origin",
        *,
        reindex: bool = False,
    ) -> None:
        mean_df = _format_forecasts(mean, index, start_index)
        variance_df = _format_forecasts(variance, index, start_index)
        residual_variance_df = _format_forecasts(residual_variance, index, start_index)
        if reindex:
            mean_df = mean_df.reindex(index)
            variance_df = variance_df.reindex(index)
            residual_variance_df = residual_variance_df.reindex(index)
        self._mean = _align_forecast(mean_df, align=align)
        self._variance = _align_forecast(variance_df, align=align)
        self._residual_variance = _align_forecast(residual_variance_df, align=align)

        if reindex:
            sim_index = index
            simulated_paths = _reindex(simulated_paths, index)
            simulated_residuals = _reindex(simulated_residuals, index)
            simulated_variances = _reindex(simulated_variances, index)
            simulated_residual_variances = _reindex(simulated_residual_variances, index)
        else:
            sim_index = index[start_index:]

        self._sim = ARCHModelForecastSimulation(
            sim_index,
            simulated_paths,
            simulated_residuals,
            simulated_variances,
            simulated_residual_variances,
        )

    @property
    def mean(self) -> pd.DataFrame:
        """Forecast values for the conditional mean of the process"""
        return self._mean

    @property
    def variance(self) -> pd.DataFrame:
        """Forecast values for the conditional variance of the process"""
        return self._variance

    @property
    def residual_variance(self) -> pd.DataFrame:
        """Forecast values for the conditional variance of the residuals"""
        return self._residual_variance

    @property
    def simulations(self) -> "ARCHModelForecastSimulation":
        """
        Detailed simulation results if using a simulation-based method

        Returns
        -------
        ARCHModelForecastSimulation
            Container for simulation results
        """
        return self._sim
