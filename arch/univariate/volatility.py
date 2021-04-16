"""
Volatility processes for ARCH model estimation.  All volatility processes must
inherit from :class:`VolatilityProcess` and provide the same methods with the
same inputs.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
import itertools
import operator
from typing import Any, List, Optional, Sequence, Tuple, Union, cast
from warnings import warn

import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln
from property_cached import cached_property

from arch.typing import ArrayLike1D, NDArray, RNGType
from arch.univariate.distribution import Normal
from arch.utility.array import AbstractDocStringInheritor, ensure1d
from arch.utility.exceptions import InitialValueWarning, initial_value_warning

try:
    from arch.univariate.recursions import (
        aparch_recursion,
        egarch_recursion,
        figarch_recursion,
        figarch_weights,
        garch_core,
        garch_recursion,
        harch_core,
        harch_recursion,
        midas_core,
        midas_recursion,
    )
except ImportError:  # pragma: no cover
    from arch.univariate.recursions_python import (
        aparch_recursion,
        egarch_recursion,
        figarch_recursion,
        figarch_weights,
        garch_core,
        garch_recursion,
        harch_core,
        harch_recursion,
        midas_core,
        midas_recursion,
    )

__all__ = [
    "GARCH",
    "ARCH",
    "HARCH",
    "ConstantVariance",
    "EWMAVariance",
    "RiskMetrics2006",
    "EGARCH",
    "FIGARCH",
    "FixedVariance",
    "BootstrapRng",
    "MIDASHyperbolic",
    "VolatilityProcess",
]

DOUBLE_EPS = np.finfo(np.float64).eps


def _common_names(p: int, o: int, q: int) -> List[str]:
    names = ["omega"]
    names.extend(["alpha[" + str(i + 1) + "]" for i in range(p)])
    names.extend(["gamma[" + str(i + 1) + "]" for i in range(o)])
    names.extend(["beta[" + str(i + 1) + "]" for i in range(q)])
    return names


class BootstrapRng(object):
    """
    Simple fake RNG used to transform bootstrap-based forecasting into a standard
    simulation forecasting problem

    Parameters
    ----------
    std_resid : ndarray
        Array containing standardized residuals
    start : int
        Location of first forecast
    random_state : RandomState, optional
        NumPy RandomState instance
    """

    def __init__(
        self, std_resid: NDArray, start: int, random_state: Optional[RandomState] = None
    ) -> None:
        if start <= 0 or start > std_resid.shape[0]:
            raise ValueError("start must be > 0 and <= len(std_resid).")

        self.std_resid: np.ndarray[Any, np.dtype[np.float64]] = std_resid
        self.start: int = start
        self._index = start
        if random_state is None:
            self._random_state = RandomState()
        elif isinstance(random_state, RandomState):
            self._random_state = random_state
        else:
            raise TypeError("random_state must be a NumPy RandomState instance.")

    @property
    def random_state(self) -> RandomState:
        return self._random_state

    def rng(self) -> RNGType:
        def _rng(size: Union[int, Tuple[int, ...]]) -> NDArray:
            if self._index >= self.std_resid.shape[0]:
                raise IndexError("not enough data points.")
            index = self._random_state.random_sample(size)
            int_index = np.floor((self._index + 1) * index)
            int_index = int_index.astype(np.int64)
            self._index += 1
            return self.std_resid[int_index]

        return _rng


def ewma_recursion(
    lam: float, resids: NDArray, sigma2: NDArray, nobs: int, backcast: float
) -> NDArray:
    """
    Compute variance recursion for EWMA/RiskMetrics Variance

    Parameters
    ----------
    lam : float
        Smoothing parameter
    resids : ndarray
        Residuals to use in the recursion
    sigma2 : ndarray
        Conditional variances with same shape as resids
    nobs : int
        Length of resids
    backcast : float
        Value to use when initializing the recursion
    """

    # Throw away bounds
    var_bounds = np.ones((nobs, 2)) * np.array([-1.0, 1.7e308])
    garch_recursion(
        np.array([0.0, 1.0 - lam, lam]),
        resids ** 2.0,
        resids,
        sigma2,
        1,
        0,
        1,
        nobs,
        backcast,
        var_bounds,
    )
    return sigma2


class VarianceForecast(object):
    _forecasts = None
    _forecast_paths = None

    def __init__(
        self,
        forecasts: NDArray,
        forecast_paths: Optional[NDArray] = None,
        shocks: Optional[NDArray] = None,
    ) -> None:
        self._forecasts = forecasts
        self._forecast_paths = forecast_paths
        self._shocks = shocks

    @property
    def forecasts(self) -> Optional[NDArray]:
        return self._forecasts

    @property
    def forecast_paths(self) -> Optional[NDArray]:
        return self._forecast_paths

    @property
    def shocks(self) -> Optional[NDArray]:
        return self._shocks


class VolatilityProcess(object, metaclass=ABCMeta):
    """
    Abstract base class for ARCH models.  Allows the conditional mean model to be specified
    separately from the conditional variance, even though parameters are estimated jointly.
    """

    _updatable: bool = False

    def __init__(self) -> None:
        self._num_params = 0
        self._name = ""
        self.closed_form: bool = False
        self._normal = Normal()
        self._min_bootstrap_obs = 100
        self._start = 0
        self._stop = -1

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__() + ", id: " + hex(id(self))

    @property
    def name(self) -> str:
        """The name of the volatilty process"""
        return self._name

    @property
    def start(self) -> int:
        """Index to use to start variance subarray selection"""
        return self._start

    @start.setter
    def start(self, value: int) -> None:
        self._start = value

    @property
    def stop(self) -> int:
        """Index to use to stop variance subarray selection"""
        return self._stop

    @stop.setter
    def stop(self, value: int) -> None:
        self._stop = value

    @property
    def num_params(self) -> int:
        """The number of parameters in the model"""
        return self._num_params

    @property
    def updateable(self) -> bool:
        """Flag indicating that the volatility process supports update"""
        return self._updatable

    def update(
        self,
        index: int,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> float:
        """
        Compute the variance for a single observation

        Parameters
        ----------
        index : int
            The numerical index of the variance to compute
        variance_params : ndarray
            The variance model parameters
        resids :
            The residual array. Only uses ``resids[:index]`` when computing
            ``sigma2[index]``
        sigma2 : ndarray
            The array containing the variances. Only uses ``sigma2[:index]``
            when computing ``sigma2[index]``. The computed value is stored
            in ``sigma2[index]``.
        backcast : {float, ndarray}
            Value to use when initializing the recursion
        var_bounds : ndarray
            Array containing columns of lower and upper bounds

        Returns
        -------
        float
            The variance computed for location ``index``
        """
        raise NotImplementedError("Subclasses may optionally implement")

    @abstractmethod
    def _check_forecasting_method(self, method: str, horizon: int) -> None:
        """
        Verify the requested forecasting method as valid for the specification

        Parameters
        ----------
        method : str
            Forecasting method
        horizon : int
            Forecast horizon

        Raises
        ------
        NotImplementedError
            * If method is not known or not supported
        """

    def _one_step_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        horizon: int,
        start_index: int,
    ) -> Tuple[NDArray, NDArray]:
        """
        One-step ahead forecast

        Parameters
        ----------
        parameters : ndarray
            Parameters required to forecast the volatility model
        resids : ndarray
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : ndarray
            Array containing columns of lower and upper bounds
        horizon : int
            Forecast horizon.  Must be 1 or larger.  Forecasts are produced
            for horizons in [1, horizon].

        Returns
        -------
        sigma2 : ndarray
            t element array containing the one-step ahead forecasts
        forecasts : ndarray
            t by horizon array containing the one-step ahead forecasts in the first location
        """
        t = resids.shape[0]
        _resids = np.concatenate((resids, [0]))
        _var_bounds = np.concatenate((var_bounds, [[0, np.inf]]))
        sigma2 = np.zeros(t + 1)
        self.compute_variance(parameters, _resids, sigma2, backcast, _var_bounds)
        forecasts = np.zeros((t - start_index, horizon))
        forecasts[:, 0] = sigma2[start_index + 1 :]
        sigma2 = sigma2[:-1]

        return sigma2, forecasts

    @abstractmethod
    def _analytic_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
    ) -> VarianceForecast:
        """
        Analytic multi-step volatility forecasts from the model

        Parameters
        ----------
        parameters : ndarray
            Parameters required to forecast the volatility model
        resids : ndarray
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : ndarray
            Array containing columns of lower and upper bounds
        start : int
            Index of the first observation to use as the starting point for
            the forecast.  Default is 0.
        horizon : int
            Forecast horizon.  Must be 1 or larger.  Forecasts are produced
            for horizons in [1, horizon].

        Returns
        -------
        forecasts : VarianceForecast
            Class containing the variance forecasts, and, if using simulation
            or bootstrap, the simulated paths.
        """

    @abstractmethod
    def _simulation_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
        simulations: int,
        rng: RNGType,
    ) -> VarianceForecast:
        """
        Simulation-based volatility forecasts from the model

        Parameters
        ----------
        parameters : ndarray
            Parameters required to forecast the volatility model
        resids : ndarray
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : ndarray
            Array containing columns of lower and upper bounds
        start : int
            Index of the first observation to use as the starting point for
            the forecast.  Default is 0.
        horizon : int
            Forecast horizon.  Must be 1 or larger.  Forecasts are produced
            for horizons in [1, horizon].
        simulations : int
            Number of simulations to run when computing the forecast using
            either simulation or bootstrap.
        rng : callable
            Callable random number generator required if method is
            'simulation'. Must take a single shape input and return random
            samples numbers with that shape.

        Returns
        -------
        forecasts : VarianceForecast
            Class containing the variance forecasts, and, if using simulation
            or bootstrap, the simulated paths.
        """

    def _bootstrap_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
        simulations: int,
        random_state: Optional[RandomState],
    ) -> VarianceForecast:
        """
        Simulation-based volatility forecasts using model residuals

        Parameters
        ----------
        parameters : ndarray
            Parameters required to forecast the volatility model
        resids : ndarray
            Residuals to use in the recursion
        backcast : {float, ndarray}
            Value to use when initializing the recursion
        var_bounds : ndarray
            Array containing columns of lower and upper bounds
        start : int
            Index of the first observation to use as the starting point for
            the forecast.  Default is 0.
        horizon : int
            Forecast horizon.  Must be 1 or larger.  Forecasts are produced
            for horizons in [1, horizon].
        simulations : int
            Number of simulations to run when computing the forecast using
            either simulation or bootstrap.
        random_state : {RandomState, None}
            NumPy RandomState instance to use in the BootstrapRng

        Returns
        -------
        forecasts : VarianceForecast
            Class containing the variance forecasts, and, if using simulation
            or bootstrap, the simulated paths.
        """
        sigma2 = np.empty_like(resids)
        self.compute_variance(parameters, resids, sigma2, backcast, var_bounds)
        std_resid = resids / np.sqrt(sigma2)
        if start < self._min_bootstrap_obs:
            raise ValueError(
                "start must include more than {0} "
                "observations".format(self._min_bootstrap_obs)
            )
        rng = BootstrapRng(std_resid, start, random_state=random_state).rng()
        return self._simulation_forecast(
            parameters, resids, backcast, var_bounds, start, horizon, simulations, rng
        )

    def variance_bounds(self, resids: NDArray, power: float = 2.0) -> NDArray:
        """
        Construct loose bounds for conditional variances.

        These bounds are used in parameter estimation to ensure
        that the log-likelihood does not produce NaN values.

        Parameters
        ----------
        resids : ndarray
            Approximate residuals to use to compute the lower and upper bounds
            on the conditional variance
        power : float, optional
            Power used in the model. 2.0, the default corresponds to standard
            ARCH models that evolve in squares.

        Returns
        -------
        var_bounds : ndarray
            Array containing columns of lower and upper bounds with the same
            number of elements as resids
        """
        nobs = resids.shape[0]

        tau = min(75, nobs)
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)
        var_bound = np.zeros(nobs)
        initial_value = w.dot(resids[:tau] ** 2.0)
        ewma_recursion(0.94, resids, var_bound, resids.shape[0], initial_value)

        var_bounds = np.vstack((var_bound / 1e6, var_bound * 1e6)).T
        var = resids.var()
        min_upper_bound = 1 + (resids ** 2.0).max()
        lower_bound, upper_bound = var / 1e8, 1e7 * (1 + (resids ** 2.0).max())
        var_bounds[var_bounds[:, 0] < lower_bound, 0] = lower_bound
        var_bounds[var_bounds[:, 1] < min_upper_bound, 1] = min_upper_bound
        var_bounds[var_bounds[:, 1] > upper_bound, 1] = upper_bound

        if power != 2.0:
            var_bounds **= power / 2.0

        return np.ascontiguousarray(var_bounds)

    @abstractmethod
    def starting_values(self, resids: NDArray) -> NDArray:
        """
        Returns starting values for the ARCH model

        Parameters
        ----------
        resids : ndarray
            Array of (approximate) residuals to use when computing starting
            values

        Returns
        -------
        sv : ndarray
            Array of starting values
        """

    def backcast(self, resids: NDArray) -> Union[float, NDArray]:
        """
        Construct values for backcasting to start the recursion

        Parameters
        ----------
        resids : ndarray
            Vector of (approximate) residuals

        Returns
        -------
        backcast : float
            Value to use in backcasting in the volatility recursion
        """
        tau = min(75, resids.shape[0])
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)

        return float(np.sum((resids[:tau] ** 2.0) * w))

    def backcast_transform(
        self, backcast: Union[float, NDArray]
    ) -> Union[float, NDArray]:
        """
        Transformation to apply to user-provided backcast values

        Parameters
        ----------
        backcast : {float, ndarray}
            User-provided ``backcast`` that approximates sigma2[0].

        Returns
        -------
        backcast : {float, ndarray}
            Backcast transformed to the model-appropriate scale
        """
        if np.any(backcast < 0):
            raise ValueError("User backcast value must be strictly positive.")
        return backcast

    @abstractmethod
    def bounds(self, resids: NDArray) -> List[Tuple[float, float]]:
        """
        Returns bounds for parameters

        Parameters
        ----------
        resids : ndarray
            Vector of (approximate) residuals

        Returns
        -------
        bounds : list[tuple[float,float]]
            List of bounds where each element is (lower, upper).
        """

    @abstractmethod
    def compute_variance(
        self,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> NDArray:
        """
        Compute the variance for the ARCH model

        Parameters
        ----------
        parameters : ndarray
            Model parameters
        resids : ndarray
            Vector of mean zero residuals
        sigma2 : ndarray
            Array with same size as resids to store the conditional variance
        backcast : {float, ndarray}
            Value to use when initializing ARCH recursion. Can be an ndarray
            when the model contains multiple components.
        var_bounds : ndarray
            Array containing columns of lower and upper bounds
        """

    @abstractmethod
    def constraints(self) -> Tuple[NDArray, NDArray]:
        """
        Construct parameter constraints arrays for parameter estimation

        Returns
        -------
        A : ndarray
            Parameters loadings in constraint. Shape is number of constraints
            by number of parameters
        b : ndarray
            Constraint values, one for each constraint

        Notes
        -----
        Values returned are used in constructing linear inequality
        constraints of the form A.dot(parameters) - b >= 0
        """

    def forecast(
        self,
        parameters: ArrayLike1D,
        resids: NDArray,
        backcast: Union[NDArray, float],
        var_bounds: NDArray,
        start: Optional[int] = None,
        horizon: int = 1,
        method: str = "analytic",
        simulations: int = 1000,
        rng: Optional[RNGType] = None,
        random_state: Optional[RandomState] = None,
    ) -> VarianceForecast:
        """
        Forecast volatility from the model

        Parameters
        ----------
        parameters : {ndarray, Series}
            Parameters required to forecast the volatility model
        resids : ndarray
            Residuals to use in the recursion
        backcast : float
            Value to use when initializing the recursion
        var_bounds : ndarray, 2-d
            Array containing columns of lower and upper bounds
        start : {None, int}
            Index of the first observation to use as the starting point for
            the forecast.  Default is len(resids).
        horizon : int
            Forecast horizon.  Must be 1 or larger.  Forecasts are produced
            for horizons in [1, horizon].
        method : {'analytic', 'simulation', 'bootstrap'}
            Method to use when producing the forecast. The default is analytic.
        simulations : int
            Number of simulations to run when computing the forecast using
            either simulation or bootstrap.
        rng : callable
            Callable random number generator required if method is
            'simulation'. Must take a single shape input and return random
            samples numbers with that shape.
        random_state : RandomState, optional
            NumPy RandomState instance to use when method is 'bootstrap'

        Returns
        -------
        forecasts : VarianceForecast
            Class containing the variance forecasts, and, if using simulation
            or bootstrap, the simulated paths.

        Raises
        ------
        NotImplementedError
            * If method is not supported
        ValueError
            * If the method is not known

        Notes
        -----
        The analytic ``method`` is not supported for all models.  Attempting
        to use this method when not available will raise a ValueError.
        """
        parameters = np.asarray(parameters)
        method = method.lower()
        if method not in ("analytic", "simulation", "bootstrap"):
            raise ValueError("{0} is not a known forecasting method".format(method))
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be an integer >= 1.")
        self._check_forecasting_method(method, horizon)
        start = len(resids) - 1 if start is None else start
        if method == "analytic":
            return self._analytic_forecast(
                parameters, resids, backcast, var_bounds, start, horizon
            )
        elif method == "simulation":
            # TODO: This looks like a design flaw.It is optional above but then must
            #  be present.  This happens because the caller of this function is
            #  expected to know when to provide the rng or not
            assert rng is not None
            return self._simulation_forecast(
                parameters,
                resids,
                backcast,
                var_bounds,
                start,
                horizon,
                simulations,
                rng,
            )
        else:
            if start < 10 or (horizon / start) >= 0.2:
                raise ValueError(
                    "Bootstrap forecasting requires at least 10 initial "
                    "observations, and the ratio of horizon-to-start < 20%."
                )

            return self._bootstrap_forecast(
                parameters,
                resids,
                backcast,
                var_bounds,
                start,
                horizon,
                simulations,
                random_state,
            )

    @abstractmethod
    def simulate(
        self,
        parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
        nobs: int,
        rng: RNGType,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Simulate data from the model

        Parameters
        ----------
        parameters : {ndarray, Series}
            Parameters required to simulate the volatility model
        nobs : int
            Number of data points to simulate
        rng : callable
            Callable function that takes a single integer input and returns
            a vector of random numbers
        burn : int, optional
            Number of additional observations to generate when initializing
            the simulation
        initial_value : {float, ndarray}, optional
            Scalar or array of initial values to use when initializing the
            simulation

        Returns
        -------
        resids : ndarray
            The simulated residuals
        variance : ndarray
            The simulated variance
        """

    def _gaussian_loglikelihood(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> float:
        """
        Private implementation of a Gaussian log-likelihood for use in constructing starting
        values or other quantities that do not depend on the distribution used by the model.
        """
        sigma2 = np.zeros_like(resids)
        self.compute_variance(parameters, resids, sigma2, backcast, var_bounds)
        return float(self._normal.loglikelihood([], resids, sigma2))

    @abstractmethod
    def parameter_names(self) -> List[str]:
        """
        Names of model parameters

        Returns
        -------
        names : list (str)
            Variables names
        """


class ConstantVariance(VolatilityProcess, metaclass=AbstractDocStringInheritor):
    r"""
    Constant volatility process

    Notes
    -----
    Model has the same variance in all periods
    """

    def __init__(self) -> None:
        super().__init__()
        self._num_params = 1
        self._name = "Constant Variance"
        self.closed_form: bool = True

    def compute_variance(
        self,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> NDArray:
        sigma2[:] = parameters[0]
        return sigma2

    def starting_values(self, resids: NDArray) -> NDArray:
        return np.array([resids.var()])

    def simulate(
        self,
        parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
        nobs: int,
        rng: RNGType,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
    ) -> Tuple[NDArray, NDArray]:
        parameters = ensure1d(parameters, "parameters", False)
        errors = rng(nobs + burn)
        sigma2 = np.ones(nobs + burn) * parameters[0]
        data = np.sqrt(sigma2) * errors
        return data[burn:], sigma2[burn:]

    def constraints(self) -> Tuple[NDArray, NDArray]:
        return np.ones((1, 1)), np.zeros(1)

    def backcast_transform(
        self, backcast: Union[float, NDArray]
    ) -> Union[float, NDArray]:
        backcast = super().backcast_transform(backcast)
        return backcast

    def backcast(self, resids: NDArray) -> Union[float, NDArray]:
        return float(resids.var())

    def bounds(self, resids: NDArray) -> List[Tuple[float, float]]:
        v = float(resids.var())
        return [(v / 100000.0, 10.0 * (v + float(resids.mean()) ** 2.0))]

    def parameter_names(self) -> List[str]:
        return ["sigma2"]

    def _check_forecasting_method(self, method: str, horizon: int) -> None:
        return

    def _analytic_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
    ) -> VarianceForecast:
        t = resids.shape[0]
        forecasts = np.full((t - start, horizon), np.nan)

        forecasts[:, :] = parameters[0]
        forecast_paths = None
        return VarianceForecast(forecasts, forecast_paths)

    def _simulation_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
        simulations: int,
        rng: RNGType,
    ) -> VarianceForecast:
        t = resids.shape[0]
        forecasts = np.empty((t - start, horizon))
        forecast_paths = np.empty((t - start, simulations, horizon))
        shocks = np.empty((t - start, simulations, horizon))

        for i in range(t - start):
            shocks[i, :, :] = np.sqrt(parameters[0]) * rng((simulations, horizon))

        forecasts[:, :] = parameters[0]
        forecast_paths[:, :, :] = parameters[0]

        return VarianceForecast(forecasts, forecast_paths, shocks)


class GARCH(VolatilityProcess, metaclass=AbstractDocStringInheritor):
    r"""
    GARCH and related model estimation

    The following models can be specified using GARCH:
        * ARCH(p)
        * GARCH(p,q)
        * GJR-GARCH(p,o,q)
        * AVARCH(p)
        * AVGARCH(p,q)
        * TARCH(p,o,q)
        * Models with arbitrary, pre-specified powers

    Parameters
    ----------
    p : int
        Order of the symmetric innovation
    o : int
        Order of the asymmetric innovation
    q : int
        Order of the lagged (transformed) conditional variance
    power : float, optional
        Power to use with the innovations, abs(e) ** power.  Default is 2.0, which produces ARCH
        and related models. Using 1.0 produces AVARCH and related models.  Other powers can be
        specified, although these should be strictly positive, and usually larger than 0.25.

    Examples
    --------
    >>> from arch.univariate import GARCH

    Standard GARCH(1,1)

    >>> garch = GARCH(p=1, q=1)

    Asymmetric GJR-GARCH process

    >>> gjr = GARCH(p=1, o=1, q=1)

    Asymmetric TARCH process

    >>> tarch = GARCH(p=1, o=1, q=1, power=1.0)

    Notes
    -----
    In this class of processes, the variance dynamics are

    .. math::

        \sigma_{t}^{\lambda}=\omega
        + \sum_{i=1}^{p}\alpha_{i}\left|\epsilon_{t-i}\right|^{\lambda}
        +\sum_{j=1}^{o}\gamma_{j}\left|\epsilon_{t-j}\right|^{\lambda}
        I\left[\epsilon_{t-j}<0\right]+\sum_{k=1}^{q}\beta_{k}\sigma_{t-k}^{\lambda}
    """
    _updatable = True

    def __init__(self, p: int = 1, o: int = 0, q: int = 1, power: float = 2.0) -> None:
        super().__init__()
        self.p: int = int(p)
        self.o: int = int(o)
        self.q: int = int(q)
        self.power: float = power
        self._num_params = 1 + p + o + q
        if p < 0 or o < 0 or q < 0:
            raise ValueError("All lags lengths must be non-negative")
        if p == 0 and o == 0:
            raise ValueError("One of p or o must be strictly positive")
        if power <= 0.0:
            raise ValueError(
                "power must be strictly positive, usually larger than 0.25"
            )
        self._name = self._generate_name()

    def __str__(self) -> str:
        descr = self.name

        if self.power != 1.0 and self.power != 2.0:
            descr = descr[:-1] + ", "
        else:
            descr += "("

        for k, v in (("p", self.p), ("o", self.o), ("q", self.q)):
            if v > 0:
                descr += k + ": " + str(v) + ", "

        descr = descr[:-2] + ")"
        return descr

    def variance_bounds(self, resids: NDArray, power: float = 2.0) -> NDArray:
        return super().variance_bounds(resids, self.power)

    def _generate_name(self) -> str:
        p, o, q, power = self.p, self.o, self.q, self.power  # noqa: F841
        if power == 2.0:
            if o == 0 and q == 0:
                return "ARCH"
            elif o == 0:
                return "GARCH"
            else:
                return "GJR-GARCH"
        elif power == 1.0:
            if o == 0 and q == 0:
                return "AVARCH"
            elif o == 0:
                return "AVGARCH"
            else:
                return "TARCH/ZARCH"
        else:
            if o == 0 and q == 0:
                return "Power ARCH (power: {0:0.1f})".format(self.power)
            elif o == 0:
                return "Power GARCH (power: {0:0.1f})".format(self.power)
            else:
                return "Asym. Power GARCH (power: {0:0.1f})".format(self.power)

    def bounds(self, resids: NDArray) -> List[Tuple[float, float]]:
        v = np.mean(abs(resids) ** self.power)

        bounds = [(0.0, 10.0 * float(v))]
        bounds.extend([(0.0, 1.0)] * self.p)
        for i in range(self.o):
            if i < self.p:
                bounds.append((-1.0, 2.0))
            else:
                bounds.append((0.0, 2.0))

        bounds.extend([(0.0, 1.0)] * self.q)

        return bounds

    def constraints(self) -> Tuple[NDArray, NDArray]:
        p, o, q = self.p, self.o, self.q
        k_arch = p + o + q
        # alpha[i] >0
        # alpha[i] + gamma[i] > 0 for i<=p, otherwise gamma[i]>0
        # beta[i] >0
        # sum(alpha) + 0.5 sum(gamma) + sum(beta) < 1
        a = np.zeros((k_arch + 2, k_arch + 1))
        for i in range(k_arch + 1):
            a[i, i] = 1.0
        for i in range(o):
            if i < p:
                a[i + p + 1, i + 1] = 1.0

        a[k_arch + 1, 1:] = -1.0
        a[k_arch + 1, p + 1 : p + o + 1] = -0.5
        b = np.zeros(k_arch + 2)
        b[k_arch + 1] = -1.0
        return a, b

    def update(
        self,
        index: int,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> float:
        return garch_core(
            index,
            parameters,
            resids,
            sigma2,
            backcast,
            var_bounds,
            self.p,
            self.o,
            self.q,
            self.power,
        )

    def compute_variance(
        self,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> NDArray:
        # fresids is abs(resids) ** power
        # sresids is I(resids<0)
        power = self.power
        fresids = np.abs(resids) ** power
        sresids = np.sign(resids)

        p, o, q = self.p, self.o, self.q
        nobs = resids.shape[0]

        garch_recursion(
            parameters, fresids, sresids, sigma2, p, o, q, nobs, backcast, var_bounds
        )
        inv_power = 2.0 / power
        sigma2 **= inv_power

        return sigma2

    def backcast_transform(
        self, backcast: Union[float, NDArray]
    ) -> Union[float, NDArray]:
        backcast = super().backcast_transform(backcast)
        return np.sqrt(backcast) ** self.power

    def backcast(self, resids: NDArray) -> Union[float, NDArray]:
        power = self.power
        tau = min(75, resids.shape[0])
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)
        backcast = np.sum((abs(resids[:tau]) ** power) * w)

        return float(backcast)

    def simulate(
        self,
        parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
        nobs: int,
        rng: RNGType,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
    ) -> Tuple[NDArray, NDArray]:
        parameters = ensure1d(parameters, "parameters", False)
        p, o, q, power = self.p, self.o, self.q, self.power
        errors = rng(nobs + burn)

        if initial_value is None:
            scale = np.ones_like(parameters)
            scale[p + 1 : p + o + 1] = 0.5

            persistence = np.sum(parameters[1:] * scale[1:])
            if (1.0 - persistence) > 0:
                initial_value = parameters[0] / (1.0 - persistence)
            else:
                warn(initial_value_warning, InitialValueWarning)
                initial_value = parameters[0]

        sigma2 = np.zeros(nobs + burn)
        data = np.zeros(nobs + burn)
        fsigma = np.zeros(nobs + burn)
        fdata = np.zeros(nobs + burn)

        max_lag = np.max([p, o, q])
        fsigma[:max_lag] = initial_value
        sigma2[:max_lag] = initial_value ** (2.0 / power)
        data[:max_lag] = np.sqrt(sigma2[:max_lag]) * errors[:max_lag]
        fdata[:max_lag] = abs(data[:max_lag]) ** power

        for t in range(max_lag, nobs + burn):
            loc = 0
            fsigma[t] = parameters[loc]
            loc += 1
            for j in range(p):
                fsigma[t] += parameters[loc] * fdata[t - 1 - j]
                loc += 1
            for j in range(o):
                fsigma[t] += parameters[loc] * fdata[t - 1 - j] * (data[t - 1 - j] < 0)
                loc += 1
            for j in range(q):
                fsigma[t] += parameters[loc] * fsigma[t - 1 - j]
                loc += 1

            sigma2[t] = fsigma[t] ** (2.0 / power)
            data[t] = errors[t] * np.sqrt(sigma2[t])
            fdata[t] = abs(data[t]) ** power

        return data[burn:], sigma2[burn:]

    def starting_values(self, resids: NDArray) -> NDArray:
        p, o, q = self.p, self.o, self.q
        power = self.power
        alphas = [0.01, 0.05, 0.1, 0.2]
        gammas = alphas
        abg = [0.5, 0.7, 0.9, 0.98]
        abgs = list(itertools.product(*[alphas, gammas, abg]))

        target = np.mean(abs(resids) ** power)
        scale = np.mean(resids ** 2) / (target ** (2.0 / power))
        target *= scale ** (power / 2)

        svs = []
        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        llfs = np.zeros(len(abgs))
        for i, values in enumerate(abgs):
            alpha, gamma, agb = values
            sv = (1.0 - agb) * target * np.ones(p + o + q + 1)
            if p > 0:
                sv[1 : 1 + p] = alpha / p
                agb -= alpha
            if o > 0:
                sv[1 + p : 1 + p + o] = gamma / o
                agb -= gamma / 2.0
            if q > 0:
                sv[1 + p + o : 1 + p + o + q] = agb / q
            svs.append(sv)
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)

        return svs[int(loc)]

    def parameter_names(self) -> List[str]:
        return _common_names(self.p, self.o, self.q)

    def _check_forecasting_method(self, method: str, horizon: int) -> None:
        if horizon == 1:
            return

        if method == "analytic" and self.power != 2.0:
            raise ValueError(
                "Analytic forecasts not available for horizon > 1 when power != 2"
            )
        return

    def _analytic_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
    ) -> VarianceForecast:

        sigma2, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon, start
        )
        if horizon == 1:
            return VarianceForecast(forecasts)

        t = resids.shape[0]
        p, o, q = self.p, self.o, self.q
        omega = parameters[0]
        alpha = parameters[1 : p + 1]
        gamma = parameters[p + 1 : p + o + 1]
        beta = parameters[p + o + 1 :]

        m = np.max([p, o, q])
        _resids = np.zeros(m + horizon)
        _asym_resids = np.zeros(m + horizon)
        _sigma2 = np.zeros(m + horizon)

        for i in range(start, t):
            if i - m + 1 >= 0:
                _resids[:m] = resids[i - m + 1 : i + 1]
                _asym_resids[:m] = _resids[:m] * (_resids[:m] < 0)
                _sigma2[:m] = sigma2[i - m + 1 : i + 1]
            else:  # Back-casting needed
                _resids[: m - i - 1] = np.sqrt(backcast)
                _resids[m - i - 1 : m] = resids[0 : i + 1]
                _asym_resids = cast(np.ndarray, _resids * (_resids < 0))
                _asym_resids[: m - i - 1] = np.sqrt(0.5 * backcast)
                _sigma2[:m] = backcast
                _sigma2[m - i - 1 : m] = sigma2[0 : i + 1]

            for h in range(0, horizon):
                fcast_loc = i - start
                forecasts[fcast_loc, h] = omega
                start_loc = h + m - 1

                for j in range(p):
                    forecasts[fcast_loc, h] += alpha[j] * _resids[start_loc - j] ** 2

                for j in range(o):
                    forecasts[fcast_loc, h] += (
                        gamma[j] * _asym_resids[start_loc - j] ** 2
                    )

                for j in range(q):
                    forecasts[fcast_loc, h] += beta[j] * _sigma2[start_loc - j]

                _resids[h + m] = np.sqrt(forecasts[fcast_loc, h])
                _asym_resids[h + m] = np.sqrt(0.5 * forecasts[fcast_loc, h])
                _sigma2[h + m] = forecasts[fcast_loc, h]

        return VarianceForecast(forecasts)

    def _simulate_paths(
        self,
        m: int,
        parameters: NDArray,
        horizon: int,
        std_shocks: NDArray,
        scaled_forecast_paths: NDArray,
        scaled_shock: NDArray,
        asym_scaled_shock: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray]:

        power = self.power
        p, o, q = self.p, self.o, self.q
        omega = parameters[0]
        alpha = parameters[1 : p + 1]
        gamma = parameters[p + 1 : p + o + 1]
        beta = parameters[p + o + 1 :]
        shock = np.empty_like(scaled_forecast_paths)

        for h in range(horizon):
            loc = h + m - 1

            scaled_forecast_paths[:, h + m] = omega
            for j in range(p):
                scaled_forecast_paths[:, h + m] += alpha[j] * scaled_shock[:, loc - j]

            for j in range(o):
                scaled_forecast_paths[:, h + m] += (
                    gamma[j] * asym_scaled_shock[:, loc - j]
                )

            for j in range(q):
                scaled_forecast_paths[:, h + m] += (
                    beta[j] * scaled_forecast_paths[:, loc - j]
                )

            shock[:, h + m] = std_shocks[:, h] * scaled_forecast_paths[:, h + m] ** (
                1.0 / power
            )
            lt_zero = shock[:, h + m] < 0
            scaled_shock[:, h + m] = np.abs(shock[:, h + m]) ** power
            asym_scaled_shock[:, h + m] = scaled_shock[:, h + m] * lt_zero

        forecast_paths = scaled_forecast_paths[:, m:] ** (2.0 / power)

        return np.asarray(np.mean(forecast_paths, 0)), forecast_paths, shock[:, m:]

    def _simulation_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
        simulations: int,
        rng: RNGType,
    ) -> VarianceForecast:

        sigma2, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon, start
        )
        t = resids.shape[0]
        paths = np.empty((t - start, simulations, horizon))
        shocks = np.empty((t - start, simulations, horizon))
        power = self.power
        m = np.max([self.p, self.o, self.q])
        scaled_forecast_paths = np.zeros((simulations, m + horizon))
        scaled_shock = np.zeros((simulations, m + horizon))
        asym_scaled_shock = np.zeros((simulations, m + horizon))

        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            if i - m < 0:
                scaled_forecast_paths[:, :m] = backcast ** (power / 2.0)
                scaled_shock[:, :m] = backcast ** (power / 2.0)
                asym_scaled_shock[:, :m] = (0.5 * backcast) ** (power / 2.0)

                # Use actual values where available
                count = i + 1
                scaled_forecast_paths[:, m - count : m] = sigma2[:count] ** (
                    power / 2.0
                )
                scaled_shock[:, m - count : m] = np.abs(resids[:count]) ** power
                asym = np.abs(resids[:count]) ** power * (resids[:count] < 0)
                asym_scaled_shock[:, m - count : m] = asym
            else:
                scaled_forecast_paths[:, :m] = sigma2[i - m + 1 : i + 1] ** (
                    power / 2.0
                )
                scaled_shock[:, :m] = np.abs(resids[i - m + 1 : i + 1]) ** power
                asym_scaled_shock[:, :m] = scaled_shock[:, :m] * (
                    resids[i - m + 1 : i + 1] < 0
                )

            f, p, s = self._simulate_paths(
                m,
                parameters,
                horizon,
                std_shocks,
                scaled_forecast_paths,
                scaled_shock,
                asym_scaled_shock,
            )
            loc = i - start
            forecasts[loc, :], paths[loc], shocks[loc] = f, p, s

        return VarianceForecast(forecasts, paths, shocks)


class HARCH(VolatilityProcess, metaclass=AbstractDocStringInheritor):
    r"""
    Heterogeneous ARCH process

    Parameters
    ----------
    lags : {list, array, int}
        List of lags to include in the model, or if scalar, includes all lags up the value

    Examples
    --------
    >>> from arch.univariate import HARCH

    Lag-1 HARCH, which is identical to an ARCH(1)

    >>> harch = HARCH()

    More useful and realistic lag lengths

    >>> harch = HARCH(lags=[1, 5, 22])

    Notes
    -----
    In a Heterogeneous ARCH process, variance dynamics are

    .. math::

        \sigma_{t}^{2}=\omega + \sum_{i=1}^{m}\alpha_{l_{i}}
        \left(l_{i}^{-1}\sum_{j=1}^{l_{i}}\epsilon_{t-j}^{2}\right)

    In the common case where lags=[1,5,22], the model is

    .. math::

        \sigma_{t}^{2}=\omega+\alpha_{1}\epsilon_{t-1}^{2}
        +\alpha_{5} \left(\frac{1}{5}\sum_{j=1}^{5}\epsilon_{t-j}^{2}\right)
        +\alpha_{22} \left(\frac{1}{22}\sum_{j=1}^{22}\epsilon_{t-j}^{2}\right)

    A HARCH process is a special case of an ARCH process where parameters in the more general
    ARCH process have been restricted.
    """
    _updatable = True

    def __init__(self, lags: Union[int, Sequence[int]] = 1) -> None:
        super().__init__()
        if np.isscalar(lags):
            lag_val = operator.index(lags)
            lags = list(range(1, lag_val + 1))
        lags_arr = ensure1d(lags, "lags")
        self.lags: np.ndarray[Any, np.dtype[np.int32]] = np.array(
            lags_arr, dtype=np.int32
        )
        self._num_lags = lags_arr.shape[0]
        self._num_params = self._num_lags + 1
        self._name = "HARCH"

    def __str__(self) -> str:
        descr = self.name + "(lags: "
        descr += ", ".join([str(lag) for lag in self.lags])
        descr += ")"

        return descr

    def bounds(self, resids: NDArray) -> List[Tuple[float, float]]:
        lags = self.lags
        k_arch = lags.shape[0]

        bounds = [(0.0, 10 * float(np.mean(resids ** 2.0)))]
        bounds.extend([(0.0, 1.0)] * k_arch)

        return bounds

    def constraints(self) -> Tuple[NDArray, NDArray]:
        k_arch = self._num_lags
        a = np.zeros((k_arch + 2, k_arch + 1))
        for i in range(k_arch + 1):
            a[i, i] = 1.0
        a[k_arch + 1, 1:] = -1.0
        b = np.zeros(k_arch + 2)
        b[k_arch + 1] = -1.0
        return a, b

    def compute_variance(
        self,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> NDArray:
        lags = self.lags
        nobs = resids.shape[0]

        harch_recursion(parameters, resids, sigma2, lags, nobs, backcast, var_bounds)
        return sigma2

    def update(
        self,
        index: int,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> float:
        """
        Compute the variance for a single observation

        Parameters
        ----------
        index : int
            The numerical index of the variance to compute
        parameters : ndarray
            The variance model parameters
        resids :
            The residual array. Only uses ``resids[:index]`` when computing
            ``sigma2[index]``
        sigma2 : ndarray
            The array containing the variances. Only uses ``sigma2[:index]``
            when computing ``sigma2[index]``. The computed value is stored
            in ``sigma2[index]``.
        backcast : {float, ndarray}
            Value to use when initializing the recursion
        var_bounds : ndarray
            Array containing columns of lower and upper bounds

        Returns
        -------
        float
            The variance computed for location ``index``
        """
        return harch_core(
            index, parameters, resids, sigma2, self.lags, backcast, var_bounds
        )

    def simulate(
        self,
        parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
        nobs: int,
        rng: RNGType,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
    ) -> Tuple[NDArray, NDArray]:
        parameters = ensure1d(parameters, "parameters", False)
        lags = self.lags
        errors = rng(nobs + burn)

        if initial_value is None:
            if (1.0 - np.sum(parameters[1:])) > 0:
                initial_value = parameters[0] / (1.0 - np.sum(parameters[1:]))
            else:
                warn(initial_value_warning, InitialValueWarning)
                initial_value = parameters[0]

        sigma2 = np.empty(nobs + burn)
        data = np.empty(nobs + burn)
        max_lag = np.max(lags)
        sigma2[:max_lag] = initial_value
        data[:max_lag] = np.sqrt(initial_value)
        for t in range(max_lag, nobs + burn):
            sigma2[t] = parameters[0]
            for i in range(lags.shape[0]):
                param = parameters[1 + i] / lags[i]
                for j in range(lags[i]):
                    sigma2[t] += param * data[t - 1 - j] ** 2.0
            data[t] = errors[t] * np.sqrt(sigma2[t])

        return data[burn:], sigma2[burn:]

    def starting_values(self, resids: NDArray) -> NDArray:
        k_arch = self._num_lags

        alpha = 0.9
        sv = (1.0 - alpha) * resids.var() * np.ones((k_arch + 1))
        sv[1:] = alpha / k_arch

        return sv

    def parameter_names(self) -> List[str]:
        names = ["omega"]
        lags = self.lags
        names.extend(["alpha[" + str(lags[i]) + "]" for i in range(self._num_lags)])
        return names

    def _harch_to_arch(self, params: NDArray) -> NDArray:
        arch_params = np.zeros((1 + int(self.lags.max())))
        arch_params[0] = params[0]
        for param, lag in zip(params[1:], self.lags):
            arch_params[1 : lag + 1] += param / lag

        return arch_params

    def _common_forecast_components(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        horizon: int,
    ) -> Tuple[float, NDArray, NDArray]:
        arch_params = self._harch_to_arch(parameters)
        t = resids.shape[0]
        m = int(self.lags.max())
        resids2 = np.empty((t, m + horizon))
        resids2[:m, :m] = backcast
        sq_resids = resids ** 2.0
        for i in range(m):
            resids2[m - i - 1 :, i] = sq_resids[: (t - (m - i - 1))]
        const = arch_params[0]
        arch = arch_params[1:]

        return const, arch, resids2

    def _check_forecasting_method(self, method: str, horizon: int) -> None:
        return

    def _analytic_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
    ) -> VarianceForecast:
        const, arch, resids2 = self._common_forecast_components(
            parameters, resids, backcast, horizon
        )
        m = int(self.lags.max())
        resids2 = resids2[start:]
        arch_rev = arch[::-1]
        for i in range(horizon):
            resids2[:, m + i] = const + resids2[:, i : (m + i)].dot(arch_rev)

        return VarianceForecast(resids2[:, m:].copy())

    def _simulation_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
        simulations: int,
        rng: RNGType,
    ) -> VarianceForecast:
        const, arch, resids2 = self._common_forecast_components(
            parameters, resids, backcast, horizon
        )
        t, m = resids.shape[0], int(self.lags.max())

        paths = np.empty((t - start, simulations, horizon))
        shocks = np.empty((t - start, simulations, horizon))

        temp_resids2 = np.empty((simulations, m + horizon))
        arch_rev = arch[::-1]
        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            temp_resids2[:, :] = resids2[i : (i + 1)]
            path_loc = i - start
            for j in range(horizon):
                paths[path_loc, :, j] = const + temp_resids2[:, j : (m + j)].dot(
                    arch_rev
                )
                shocks[path_loc, :, j] = std_shocks[:, j] * np.sqrt(
                    paths[path_loc, :, j]
                )
                temp_resids2[:, m + j] = shocks[path_loc, :, j] ** 2.0

        return VarianceForecast(np.asarray(paths.mean(1)), paths, shocks)


class MIDASHyperbolic(VolatilityProcess, metaclass=AbstractDocStringInheritor):
    r"""
    MIDAS Hyperbolic ARCH process

    Parameters
    ----------
    m : int
        Length of maximum lag to include in the model
    asym : bool
        Flag indicating whether to include an asymmetric term

    Examples
    --------
    >>> from arch.univariate import MIDASHyperbolic

    22-lag MIDAS Hyperbolic process

    >>> harch = MIDASHyperbolic()

    Longer 66-period lag

    >>> harch = MIDASHyperbolic(m=66)

    Asymmetric MIDAS Hyperbolic process

    >>> harch = MIDASHyperbolic(asym=True)

    Notes
    -----
    In a MIDAS Hyperbolic process, the variance evolves according to

    .. math::

        \sigma_{t}^{2}=\omega+
        \sum_{i=1}^{m}\left(\alpha+\gamma I\left[\epsilon_{t-j}<0\right]\right)
        \phi_{i}(\theta)\epsilon_{t-i}^{2}

    where

    .. math::

        \phi_{i}(\theta) \propto \Gamma(i+\theta)/(\Gamma(i+1)\Gamma(\theta))

    where :math:`\Gamma` is the gamma function. :math:`\{\phi_i(\theta)\}` is
    normalized so that :math:`\sum \phi_i(\theta)=1`

    References
    ----------
    .. [*] Foroni, Claudia, and Massimiliano Marcellino. "A survey of
       Econometric Methods for Mixed-Frequency Data". Norges Bank. (2013).
    .. [*] Sheppard, Kevin. "Direct volatility modeling". Manuscript. (2018).
    """
    _updatable = True

    def __init__(self, m: int = 22, asym: bool = False) -> None:
        super().__init__()
        self.m: int = m
        self._asym = bool(asym)
        self._num_params = 3 + self._asym
        self._name = "MIDAS Hyperbolic"
        self._last_weights = np.empty(m)
        self._last_theta = -9999.99

    def __str__(self) -> str:
        descr = self.name
        descr += "(lags: {0}, asym: {1}".format(self.m, self._asym)

        return descr

    def bounds(self, resids: NDArray) -> List[Tuple[float, float]]:
        bounds = [(0.0, 10 * float(np.mean(resids ** 2.0)))]  # omega
        bounds.extend([(0.0, 1.0)])  # 0 <= alpha < 1
        if self._asym:
            bounds.extend([(-1.0, 2.0)])  # -1 <= gamma < 2
        bounds.extend([(0.0, 1.0)])  # theta

        return bounds

    def constraints(self) -> Tuple[NDArray, NDArray]:
        """
        Constraints

        Notes
        -----
        Parameters are (omega, alpha, gamma, theta)

        A.dot(parameters) - b >= 0

        1. omega >0
        2. alpha>0 or alpha + gamma > 0
        3. alpha<1 or alpha+0.5*gamma<1
        4. theta > 0
        5. theta < 1
        """
        symm = not self._asym
        k = 3 + self._asym
        a = np.zeros((5, k))
        b = np.zeros(5)
        # omega
        a[0, 0] = 1.0
        # alpha >0 or alpha+gamma>0
        # alpha<1 or alpha+0.5*gamma<1
        if symm:
            a[1, 1] = 1.0
            a[2, 1] = -1.0
        else:
            a[1, 1:3] = 1.0
            a[2, 1:3] = [-1, -0.5]
        b[2] = -1.0
        # theta
        a[3, k - 1] = 1.0
        a[4, k - 1] = -1.0
        b[4] = -1.0

        return a, b

    def update(
        self,
        index: int,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> float:
        # TODO: Need a metter way to handle once a function call setup. Probably a new
        #   function `update_setup` that would accept parameters and could create
        #   transformed variables that con be used later
        theta = max(parameters[-1], DOUBLE_EPS)
        if self._last_theta != theta:
            self._last_weights = self._weights(parameters)
        return midas_core(
            index, parameters, self._last_weights, resids, sigma2, backcast, var_bounds
        )

    def compute_variance(
        self,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> NDArray:
        nobs = resids.shape[0]
        weights = self._weights(parameters)
        if not self._asym:
            params = np.zeros(3)
            params[:2] = parameters[:2]
        else:
            params = parameters[:3]

        midas_recursion(params, weights, resids, sigma2, nobs, backcast, var_bounds)
        return sigma2

    def simulate(
        self,
        parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
        nobs: int,
        rng: RNGType,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
    ) -> Tuple[NDArray, NDArray]:
        parameters = ensure1d(parameters, "parameters", False)
        if self._asym:
            omega, alpha, gamma = parameters[:3]
        else:
            omega, alpha = parameters[:2]
            gamma = 0
        weights = self._weights(parameters)
        aw = weights * alpha
        gw = weights * gamma

        errors = rng(nobs + burn)

        if initial_value is None:
            if (1.0 - alpha - 0.5 * gamma) > 0:
                initial_value = parameters[0] / (1.0 - alpha - 0.5 * gamma)
            else:
                warn(initial_value_warning, InitialValueWarning)
                initial_value = parameters[0]

        m = weights.shape[0]
        burn = max(burn, m)
        sigma2 = np.empty(nobs + burn)
        data = np.empty(nobs + burn)

        sigma2[:m] = initial_value
        data[:m] = np.sqrt(initial_value)
        for t in range(m, nobs + burn):
            sigma2[t] = omega
            for i in range(m):
                if t - 1 - i < m:
                    coef = aw[i] + 0.5 * gw[i]
                else:
                    coef = aw[i] + gw[i] * (data[t - 1 - i] < 0)
                sigma2[t] += coef * data[t - 1 - i] ** 2.0
            data[t] = errors[t] * np.sqrt(sigma2[t])

        return data[burn:], sigma2[burn:]

    def starting_values(self, resids: NDArray) -> NDArray:
        theta = [0.1, 0.5, 0.8, 0.9]
        alpha = [0.8, 0.9, 0.95, 0.98]
        var = (resids ** 2).mean()
        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        llfs = []
        svs = []
        for a, t in itertools.product(alpha, theta):
            gamma = [0.0]
            if self._asym:
                gamma.extend([0.5, 0.9])
            for g in gamma:
                total = a + g / 2
                o = (1 - min(total, 0.99)) * var
                if self._asym:
                    sv = np.array([o, a, g, t])
                else:
                    sv = np.array([o, a, t])

                svs.append(sv)

                llf = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
                llfs.append(llf)
        loc = np.argmax(llfs)

        return svs[int(loc)]

    def parameter_names(self) -> List[str]:
        names = ["omega", "alpha", "theta"]
        if self._asym:
            names.insert(2, "gamma")

        return names

    def _weights(self, params: NDArray) -> NDArray:
        m = self.m
        # Prevent 0
        theta = max(params[-1], DOUBLE_EPS)
        j = np.arange(1.0, m + 1)
        w = gammaln(theta + j) - gammaln(j + 1) - gammaln(theta)
        w = np.exp(w)
        return w / w.sum()

    def _common_forecast_components(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        horizon: int,
    ) -> Tuple[int, NDArray, NDArray, NDArray, NDArray]:
        if self._asym:
            omega, alpha, gamma = parameters[:3]
        else:
            omega, alpha = parameters[:2]
            gamma = 0.0
        weights = self._weights(parameters)
        aw = weights * alpha
        gw = weights * gamma

        t = resids.shape[0]
        m = self.m
        resids2 = np.empty((t, m + horizon))
        resids2[:m, :m] = backcast
        indicator = np.empty((t, m + horizon))
        indicator[:m, :m] = 0.5
        sq_resids = resids ** 2.0
        for i in range(m):
            resids2[m - i - 1 :, i] = sq_resids[: (t - (m - i - 1))]
            indicator[m - i - 1 :, i] = resids[: (t - (m - i - 1))] < 0

        return omega, aw, gw, resids2, indicator

    def _check_forecasting_method(self, method: str, horizon: int) -> None:
        return

    def _analytic_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
    ) -> VarianceForecast:
        omega, aw, gw, resids2, indicator = self._common_forecast_components(
            parameters, resids, backcast, horizon
        )
        m = self.m
        resids2 = resids2[start:].copy()
        aw_rev = aw[::-1]
        gw_rev = gw[::-1]

        for i in range(horizon):
            resids2[:, m + i] = omega + resids2[:, i : (m + i)].dot(aw_rev)
            if self._asym:
                resids2_ind = resids2[:, i : (m + i)] * indicator[:, i : (m + i)]
                resids2[:, m + i] += resids2_ind.dot(gw_rev)
                indicator[:, m + i] = 0.5

        return VarianceForecast(resids2[:, m:].copy())

    def _simulation_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
        simulations: int,
        rng: RNGType,
    ) -> VarianceForecast:
        omega, aw, gw, resids2, indicator = self._common_forecast_components(
            parameters, resids, backcast, horizon
        )
        t = resids.shape[0]
        m = self.m

        shocks = np.empty((t - start, simulations, horizon))
        paths = np.empty((t - start, simulations, horizon))

        temp_resids2 = np.empty((simulations, m + horizon))
        temp_indicator = np.empty((simulations, m + horizon))
        aw_rev = aw[::-1]
        gw_rev = gw[::-1]
        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            temp_resids2[:, :] = resids2[i : (i + 1)]
            temp_indicator[:, :] = indicator[i : (i + 1)]
            path_loc = i - start
            for j in range(horizon):
                paths[path_loc, :, j] = omega + temp_resids2[:, j : (m + j)].dot(aw_rev)
                if self._asym:
                    temp_resids2_ind = (
                        temp_resids2[:, j : (m + j)] * temp_indicator[:, j : (m + j)]
                    )
                    paths[path_loc, :, j] += temp_resids2_ind.dot(gw_rev)

                shocks[path_loc, :, j] = std_shocks[:, j] * np.sqrt(
                    paths[path_loc, :, j]
                )
                temp_resids2[:, m + j] = shocks[path_loc, :, j] ** 2.0
                temp_indicator[:, m + j] = (shocks[path_loc, :, j] < 0).astype(
                    np.double
                )

        return VarianceForecast(np.asarray(paths.mean(1)), paths, shocks)


class ARCH(GARCH):
    r"""
    ARCH process

    Parameters
    ----------
    p : int
        Order of the symmetric innovation

    Examples
    --------
    ARCH(1) process

    >>> from arch.univariate import ARCH

    ARCH(5) process

    >>> arch = ARCH(p=5)

    Notes
    -----
    The variance dynamics of the model estimated

    .. math::

        \sigma_t^{2}=\omega+\sum_{i=1}^{p}\alpha_{i}\epsilon_{t-i}^{2}

    """

    def __init__(self, p: int = 1) -> None:
        super().__init__(p, 0, 0, 2.0)
        self._num_params = p + 1

    def starting_values(self, resids: NDArray) -> NDArray:
        p = self.p

        alphas = np.arange(0.1, 0.95, 0.05)
        svs = []
        backcast = self.backcast(resids)
        llfs = alphas.copy()
        var_bounds = self.variance_bounds(resids)
        for i, alpha in enumerate(alphas):
            sv = (1.0 - alpha) * resids.var() * np.ones((p + 1))
            sv[1:] = alpha / p
            svs.append(sv)
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)
        return svs[int(loc)]


class EWMAVariance(VolatilityProcess, metaclass=AbstractDocStringInheritor):
    r"""
    Exponentially Weighted Moving-Average (RiskMetrics) Variance process

    Parameters
    ----------
    lam : {float, None}, optional
        Smoothing parameter. Default is 0.94. Set to None to estimate lam
        jointly with other model parameters

    Examples
    --------
    Daily RiskMetrics EWMA process

    >>> from arch.univariate import EWMAVariance
    >>> rm = EWMAVariance(0.94)

    Notes
    -----
    The variance dynamics of the model

    .. math::

        \sigma_t^{2}=\lambda\sigma_{t-1}^2 + (1-\lambda)\epsilon^2_{t-1}

    When lam is provided, this model has no parameters since the smoothing
    parameter is treated as fixed. Set lam to ``None`` to jointly estimate this
    parameter when fitting the model.
    """
    _updatable = True

    def __init__(self, lam: Optional[float] = 0.94) -> None:
        super().__init__()
        self.lam: Optional[float] = lam
        self._estimate_lam = lam is None
        self._num_params = 1 if self._estimate_lam else 0
        if lam is not None and not 0.0 < lam < 1.0:
            raise ValueError("lam must be strictly between 0 and 1")
        self._name = "EWMA/RiskMetrics"

    def __str__(self) -> str:
        if self._estimate_lam:
            descr = self.name + "(lam: Estimated)"
        else:
            assert self.lam is not None
            descr = self.name + "(lam: " + "{0:0.2f}".format(self.lam) + ")"
        return descr

    def starting_values(self, resids: NDArray) -> NDArray:
        if self._estimate_lam:
            return np.array([0.94])
        return np.array([])

    def parameter_names(self) -> List[str]:
        if self._estimate_lam:
            return ["lam"]
        return []

    def bounds(self, resids: NDArray) -> List[Tuple[float, float]]:
        if self._estimate_lam:
            return [(0, 1)]
        return []

    def compute_variance(
        self,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> NDArray:
        lam = parameters[0] if self._estimate_lam else self.lam
        return ewma_recursion(lam, resids, sigma2, resids.shape[0], float(backcast))

    def update(
        self,
        index: int,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> float:
        lam = parameters[0] if self._estimate_lam else self.lam
        assert lam is not None
        return garch_core(
            index,
            np.array([0.0, 1.0 - lam, lam]),
            resids,
            sigma2,
            backcast,
            var_bounds,
            1,
            0,
            1,
            2,
        )

    def constraints(self) -> Tuple[NDArray, NDArray]:
        if self._estimate_lam:
            a = np.ones((1, 1))
            b = np.zeros((1,))
            return a, b
        return np.empty((0, 0)), np.empty((0,))

    def simulate(
        self,
        parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
        nobs: int,
        rng: RNGType,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
    ) -> Tuple[NDArray, NDArray]:
        parameters = ensure1d(parameters, "parameters", False)
        errors = rng(nobs + burn)

        if initial_value is None:
            initial_value = 1.0

        sigma2 = np.zeros(nobs + burn)
        data = np.zeros(nobs + burn)

        sigma2[0] = initial_value
        data[0] = np.sqrt(sigma2[0])
        if self._estimate_lam:
            lam = parameters[0]
        else:
            lam = self.lam
        one_m_lam = 1.0 - lam
        for t in range(1, nobs + burn):
            sigma2[t] = lam * sigma2[t - 1] + one_m_lam * data[t - 1] ** 2.0
            data[t] = np.sqrt(sigma2[t]) * errors[t]

        return data[burn:], sigma2[burn:]

    def _check_forecasting_method(self, method: str, horizon: int) -> None:
        return

    def _analytic_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
    ) -> VarianceForecast:
        _, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon, start_index=start
        )
        for i in range(1, horizon):
            forecasts[:, i] = forecasts[:, 0]
        return VarianceForecast(forecasts)

    def _simulation_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
        simulations: int,
        rng: RNGType,
    ) -> VarianceForecast:
        one_step = self._analytic_forecast(
            parameters, resids, backcast, var_bounds, start, 1
        )
        t = resids.shape[0]
        paths = np.empty((t - start, simulations, horizon))
        shocks = np.empty((t - start, simulations, horizon))
        if self._estimate_lam:
            lam = parameters[0]
        else:
            lam = self.lam
        assert one_step.forecasts is not None
        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            path_loc = i - start
            paths[path_loc, :, 0] = one_step.forecasts[path_loc]
            shocks[path_loc, :, 0] = (
                np.sqrt(one_step.forecasts[path_loc]) * std_shocks[:, 0]
            )
            for h in range(1, horizon):
                paths[path_loc, :, h] = (1 - lam) * shocks[
                    path_loc, :, h - 1
                ] ** 2.0 + lam * paths[path_loc, :, h - 1]
                shocks[path_loc, :, h] = (
                    np.sqrt(paths[path_loc, :, h]) * std_shocks[:, h]
                )

        return VarianceForecast(np.asarray(paths.mean(1)), paths, shocks)


class RiskMetrics2006(VolatilityProcess, metaclass=AbstractDocStringInheritor):
    """
    RiskMetrics 2006 Variance process

    Parameters
    ----------
    tau0 : {int, float}, optional
        Length of long cycle. Default is 1560.
    tau1 : {int, float}, optional
        Length of short cycle. Default is 4.
    kmax : int, optional
        Number of components. Default is 14.
    rho : float, optional
        Relative scale of adjacent cycles. Default is sqrt(2)

    Examples
    --------
    Daily RiskMetrics 2006 process

    >>> from arch.univariate import RiskMetrics2006
    >>> rm = RiskMetrics2006()

    Notes
    -----
    The variance dynamics of the model are given as a weighted average of kmax EWMA variance
    processes where the smoothing parameters and weights are determined by tau0, tau1 and rho.

    This model has no parameters since the smoothing parameter is fixed.
    """

    def __init__(
        self,
        tau0: float = 1560,
        tau1: float = 4,
        kmax: int = 14,
        rho: float = 1.4142135623730951,
    ) -> None:
        super().__init__()
        self.tau0: float = tau0
        self.tau1: float = tau1
        self.kmax: int = kmax
        self.rho: float = rho
        self._num_params = 0

        if tau0 <= tau1 or tau1 <= 0:
            raise ValueError("tau0 must be greater than tau1 and tau1 > 0")
        if tau1 * rho ** (kmax - 1) > tau0:
            raise ValueError("tau1 * rho ** (kmax-1) smaller than tau0")
        if not kmax >= 1:
            raise ValueError("kmax must be a positive integer")
        if not rho > 1:
            raise ValueError("rho must be a positive number larger than 1")
        self._name = "RiskMetrics2006"

    def __str__(self) -> str:
        descr = self.name
        descr += (
            f"(tau0: {self.tau0:0.1f}, tau1: {self.tau1:0.1f}, "
            f"kmax: {self.kmax:d}, rho: {self.rho:0.3f}"
        )
        descr += ")"
        return descr

    @cached_property
    def _ewma_combination_weights(self) -> NDArray:
        """
        Returns
        -------
        weights : ndarray
            Combination weights for EWMA components
        """
        tau0, tau1, kmax, rho = self.tau0, self.tau1, self.kmax, self.rho
        taus = tau1 * (rho ** np.arange(kmax))
        w = 1 - np.log(taus) / np.log(tau0)
        w = w / w.sum()

        return w

    @cached_property
    def _ewma_smoothing_parameters(self) -> NDArray:
        tau1, kmax, rho = self.tau1, self.kmax, self.rho
        taus = tau1 * (rho ** np.arange(kmax))
        mus = cast(NDArray, np.exp(-1.0 / taus))
        return mus

    def backcast(self, resids: NDArray) -> Union[float, NDArray]:
        """
        Construct values for backcasting to start the recursion

        Parameters
        ----------
        resids : ndarray
            Vector of (approximate) residuals

        Returns
        -------
        backcast : ndarray
            Backcast values for each EWMA component
        """

        nobs = resids.shape[0]
        mus = self._ewma_smoothing_parameters

        resids2 = resids ** 2.0
        backcast = np.zeros(mus.shape[0])
        for k in range(int(self.kmax)):
            mu = mus[k]
            end_point = int(max(min(np.floor(np.log(0.01) / np.log(mu)), nobs), k))
            weights = mu ** np.arange(end_point)
            weights = weights / weights.sum()
            backcast[k] = weights.dot(resids2[:end_point])

        return backcast

    def backcast_transform(
        self, backcast: Union[float, NDArray]
    ) -> Union[float, NDArray]:
        backcast = super().backcast_transform(backcast)
        mus = self._ewma_smoothing_parameters
        backcast_arr = np.asarray(backcast)
        if backcast_arr.ndim == 0:
            backcast_arr = cast(np.ndarray, backcast * np.ones(mus.shape[0]))
        if backcast_arr.shape[0] != mus.shape[0] and backcast_arr.ndim != 0:
            raise ValueError(
                "User backcast mut be either a scalar or an vector containing the "
                "number of\ncomponent EWMAs in the model."
            )

        return backcast_arr

    def starting_values(self, resids: NDArray) -> NDArray:
        return np.empty((0,))

    def parameter_names(self) -> List[str]:
        return []

    def variance_bounds(self, resids: NDArray, power: float = 2.0) -> NDArray:
        return np.ones((resids.shape[0], 1)) * np.array(
            [-1.0, np.finfo(np.float64).max]
        )

    def bounds(self, resids: NDArray) -> List[Tuple[float, float]]:
        return []

    def constraints(self) -> Tuple[NDArray, NDArray]:
        return np.empty((0, 0)), np.empty((0,))

    def compute_variance(
        self,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> NDArray:

        nobs = resids.shape[0]
        kmax = self.kmax
        w = self._ewma_combination_weights
        mus = self._ewma_smoothing_parameters

        sigma2_temp = np.zeros_like(sigma2)
        backcast = cast(NDArray, backcast)
        for k in range(kmax):
            mu = mus[k]
            ewma_recursion(mu, resids, sigma2_temp, nobs, backcast[k])
            if k == 0:
                sigma2[:] = w[k] * sigma2_temp
            else:
                sigma2 += w[k] * sigma2_temp

        return sigma2

    def simulate(
        self,
        parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
        nobs: int,
        rng: RNGType,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
    ) -> Tuple[NDArray, NDArray]:
        errors = rng(nobs + burn)

        kmax = self.kmax
        w = self._ewma_combination_weights
        mus = self._ewma_smoothing_parameters

        if initial_value is None:
            initial_value = 1.0
        sigma2s = np.zeros((nobs + burn, kmax))
        sigma2s[0, :] = initial_value
        sigma2 = np.zeros(nobs + burn)
        data = np.zeros(nobs + burn)
        data[0] = np.sqrt(initial_value)
        sigma2[0] = w.dot(sigma2s[0])
        for t in range(1, nobs + burn):
            sigma2s[t] = mus * sigma2s[t - 1] + (1 - mus) * data[t - 1] ** 2.0
            sigma2[t] = w.dot(sigma2s[t])
            data[t] = np.sqrt(sigma2[t]) * errors[t]

        return data[burn:], sigma2[burn:]

    def _check_forecasting_method(self, method: str, horizon: int) -> None:
        return

    def _analytic_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
    ) -> VarianceForecast:
        _, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon, start_index=start
        )
        for i in range(1, horizon):
            forecasts[:, i] = forecasts[:, 0]
        return VarianceForecast(forecasts)

    def _simulation_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
        simulations: int,
        rng: RNGType,
    ) -> VarianceForecast:
        kmax = self.kmax
        w = self._ewma_combination_weights
        mus = self._ewma_smoothing_parameters
        backcast = cast(NDArray, np.asarray(backcast))

        t = resids.shape[0]
        paths = np.empty((t - start, simulations, horizon))
        shocks = np.empty((t - start, simulations, horizon))

        temp_paths = np.empty((kmax, simulations, horizon))
        # We use the transpose here to get C-contiguous arrays
        component_one_step = np.empty((kmax, t + 1))
        _resids = np.ascontiguousarray(resids)
        for k in range(kmax):
            mu = mus[k]
            ewma_recursion(mu, _resids, component_one_step[k, :], t + 1, backcast[k])
        # Transpose to be (t+1, kmax)
        component_one_step = component_one_step.T

        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            for k in range(kmax):
                temp_paths[k, :, 0] = component_one_step[i, k]
            path_loc = i - start
            paths[path_loc, :, 0] = w.dot(temp_paths[:, :, 0])
            shocks[path_loc, :, 0] = std_shocks[:, 0] * np.sqrt(paths[path_loc, :, 0])
            for j in range(1, horizon):
                for k in range(kmax):
                    mu = mus[k]
                    temp_paths[k, :, j] = (
                        mu * temp_paths[k, :, j - 1]
                        + (1 - mu) * shocks[path_loc, :, j - 1] ** 2.0
                    )
                paths[path_loc, :, j] = w.dot(temp_paths[:, :, j])
                shocks[path_loc, :, j] = std_shocks[:, j] * np.sqrt(
                    paths[path_loc, :, j]
                )

        return VarianceForecast(np.asarray(paths.mean(1)), paths, shocks)


class EGARCH(VolatilityProcess, metaclass=AbstractDocStringInheritor):
    r"""
    EGARCH model estimation

    Parameters
    ----------
    p : int
        Order of the symmetric innovation
    o : int
        Order of the asymmetric innovation
    q : int
        Order of the lagged (transformed) conditional variance

    Examples
    --------
    >>> from arch.univariate import EGARCH

    Symmetric EGARCH(1,1)

    >>> egarch = EGARCH(p=1, q=1)

    Standard EGARCH process

    >>> egarch = EGARCH(p=1, o=1, q=1)

    Exponential ARCH process

    >>> earch = EGARCH(p=5)

    Notes
    -----
    In this class of processes, the variance dynamics are

    .. math::

        \ln\sigma_{t}^{2}=\omega
        +\sum_{i=1}^{p}\alpha_{i}
        \left(\left|e_{t-i}\right|-\sqrt{2/\pi}\right)
        +\sum_{j=1}^{o}\gamma_{j} e_{t-j}
        +\sum_{k=1}^{q}\beta_{k}\ln\sigma_{t-k}^{2}

    where :math:`e_{t}=\epsilon_{t}/\sigma_{t}`.
    """

    def __init__(self, p: int = 1, o: int = 0, q: int = 1) -> None:
        super().__init__()
        self.p: int = int(p)
        self.o: int = int(o)
        self.q: int = int(q)
        self._num_params = 1 + p + o + q
        if p < 0 or o < 0 or q < 0:
            raise ValueError("All lags lengths must be non-negative")
        if p == 0 and o == 0:
            raise ValueError("One of p or o must be strictly positive")
        self._name = "EGARCH" if q > 0 else "EARCH"
        # Helpers for fitting variance
        self._arrays: Optional[Tuple[NDArray, NDArray, NDArray]] = None

    def __str__(self) -> str:
        descr = self.name + "("
        for k, v in (("p", self.p), ("o", self.o), ("q", self.q)):
            if v > 0:
                descr += k + ": " + str(v) + ", "
        descr = descr[:-2] + ")"
        return descr

    def variance_bounds(self, resids: NDArray, power: float = 2.0) -> NDArray:
        return super().variance_bounds(resids, 2.0)

    def bounds(self, resids: NDArray) -> List[Tuple[float, float]]:
        v = np.mean(resids ** 2.0)
        log_const = np.log(10000.0)
        lnv = np.log(v)
        bounds = [(lnv - log_const, lnv + log_const)]
        bounds.extend([(-np.inf, np.inf)] * (self.p + self.o))
        bounds.extend([(0.0, float(self.q))] * self.q)

        return bounds

    def constraints(self) -> Tuple[NDArray, NDArray]:
        p, o, q = self.p, self.o, self.q
        k_arch = p + o + q
        a = np.zeros((1, k_arch + 1))
        a[0, p + o + 1 :] = -1.0
        b = np.zeros((1,))
        b[0] = -1.0
        return a, b

    def compute_variance(
        self,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> NDArray:
        p, o, q = self.p, self.o, self.q
        nobs = resids.shape[0]
        if (self._arrays is not None) and (self._arrays[0].shape[0] == nobs):
            lnsigma2, std_resids, abs_std_resids = self._arrays
        else:
            lnsigma2 = np.empty(nobs)
            abs_std_resids = np.empty(nobs)
            std_resids = np.empty(nobs)
            self._arrays = (lnsigma2, abs_std_resids, std_resids)

        egarch_recursion(
            parameters,
            resids,
            sigma2,
            p,
            o,
            q,
            nobs,
            backcast,
            var_bounds,
            lnsigma2,
            std_resids,
            abs_std_resids,
        )

        return sigma2

    def backcast_transform(
        self, backcast: Union[float, NDArray]
    ) -> Union[float, NDArray]:
        backcast = super().backcast_transform(backcast)
        return float(np.log(backcast))

    def backcast(self, resids: NDArray) -> Union[float, NDArray]:
        return float(np.log(super().backcast(resids)))

    def simulate(
        self,
        parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
        nobs: int,
        rng: RNGType,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
    ) -> Tuple[NDArray, NDArray]:
        parameters = ensure1d(parameters, "parameters", False)
        p, o, q = self.p, self.o, self.q
        errors = rng(nobs + burn)

        if initial_value is None:
            if q > 0:
                beta_sum = np.sum(parameters[p + o + 1 :])
            else:
                beta_sum = 0.0

            if beta_sum < 1:
                initial_value = parameters[0] / (1.0 - beta_sum)
            else:
                warn(initial_value_warning, InitialValueWarning)
                initial_value = parameters[0]

        sigma2 = np.zeros(nobs + burn)
        data = np.zeros(nobs + burn)
        lnsigma2 = np.zeros(nobs + burn)
        abserrors = cast(NDArray, np.abs(errors))

        norm_const = np.sqrt(2 / np.pi)
        max_lag = np.max([p, o, q])
        lnsigma2[:max_lag] = initial_value
        sigma2[:max_lag] = np.exp(initial_value)
        data[:max_lag] = errors[:max_lag] * np.sqrt(sigma2[:max_lag])

        for t in range(max_lag, nobs + burn):
            loc = 0
            lnsigma2[t] = parameters[loc]
            loc += 1
            for j in range(p):
                lnsigma2[t] += parameters[loc] * (abserrors[t - 1 - j] - norm_const)
                loc += 1
            for j in range(o):
                lnsigma2[t] += parameters[loc] * errors[t - 1 - j]
                loc += 1
            for j in range(q):
                lnsigma2[t] += parameters[loc] * lnsigma2[t - 1 - j]
                loc += 1

        sigma2 = cast(NDArray, np.exp(lnsigma2))
        data = errors * np.sqrt(sigma2)

        return data[burn:], sigma2[burn:]

    def starting_values(self, resids: NDArray) -> NDArray:
        p, o, q = self.p, self.o, self.q
        alphas = [0.01, 0.05, 0.1, 0.2]
        gammas = [-0.1, 0.0, 0.1]
        betas = [0.5, 0.7, 0.9, 0.98]
        agbs = list(itertools.product(*[alphas, gammas, betas]))

        target = np.log(np.mean(resids ** 2))

        svs = []
        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        llfs = np.zeros(len(agbs))
        for i, values in enumerate(agbs):
            alpha, gamma, beta = values
            sv = (1.0 - beta) * target * np.ones(p + o + q + 1)
            if p > 0:
                sv[1 : 1 + p] = alpha / p
            if o > 0:
                sv[1 + p : 1 + p + o] = gamma / o
            if q > 0:
                sv[1 + p + o : 1 + p + o + q] = beta / q
            svs.append(sv)
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)

        return svs[int(loc)]

    def parameter_names(self) -> List[str]:
        return _common_names(self.p, self.o, self.q)

    def _check_forecasting_method(self, method: str, horizon: int) -> None:
        if method == "analytic" and horizon > 1:
            raise ValueError("Analytic forecasts not available for horizon > 1")
        return

    def _analytic_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
    ) -> VarianceForecast:

        _, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon, start
        )

        return VarianceForecast(forecasts)

    def _simulation_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
        simulations: int,
        rng: RNGType,
    ) -> VarianceForecast:
        sigma2, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon, start
        )
        t = resids.shape[0]
        p, o, q = self.p, self.o, self.q
        m = np.max([p, o, q])

        lnsigma2 = cast(NDArray, np.log(sigma2))
        e = resids / np.sqrt(sigma2)

        lnsigma2_mat = np.full((t, m), np.log(backcast))
        e_mat = np.zeros((t, m))
        abs_e_mat = np.full((t, m), np.sqrt(2 / np.pi))

        for i in range(m):
            lnsigma2_mat[m - i - 1 :, i] = lnsigma2[: (t - (m - 1) + i)]
            e_mat[m - i - 1 :, i] = e[: (t - (m - 1) + i)]
            abs_e_mat[m - i - 1 :, i] = np.abs(e[: (t - (m - 1) + i)])

        paths = np.empty((t - start, simulations, horizon))
        shocks = np.empty((t - start, simulations, horizon))

        sqrt2pi = np.sqrt(2 / np.pi)
        _lnsigma2 = np.empty((simulations, m + horizon))
        _e = np.empty((simulations, m + horizon))
        _abs_e = np.empty((simulations, m + horizon))
        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            _lnsigma2[:, :m] = lnsigma2_mat[i, :]
            _e[:, :m] = e_mat[i, :]
            _e[:, m:] = std_shocks
            _abs_e[:, :m] = abs_e_mat[i, :]
            _abs_e[:, m:] = np.abs(std_shocks)
            for j in range(horizon):
                loc = 0
                _lnsigma2[:, m + j] = parameters[loc]
                loc += 1
                for k in range(p):
                    _lnsigma2[:, m + j] += parameters[loc] * (
                        _abs_e[:, m + j - 1 - k] - sqrt2pi
                    )
                    loc += 1

                for k in range(o):
                    _lnsigma2[:, m + j] += parameters[loc] * _e[:, m + j - 1 - k]
                    loc += 1

                for k in range(q):
                    _lnsigma2[:, m + j] += parameters[loc] * _lnsigma2[:, m + j - 1 - k]
                    loc += 1
            loc = i - start
            paths[loc, :, :] = np.exp(_lnsigma2[:, m:])
            shocks[loc, :, :] = np.sqrt(paths[loc, :, :]) * std_shocks

        return VarianceForecast(np.asarray(paths.mean(1)), paths, shocks)


class FixedVariance(VolatilityProcess, metaclass=AbstractDocStringInheritor):
    """
    Fixed volatility process

    Parameters
    ----------
    variance : {array, Series}
        Array containing the variances to use.  Should have the same shape as the data used in the
        model.
    unit_scale : bool, optional
        Flag whether to enforce a unit scale.  If False, a scale parameter will be estimated so
        that the model variance will be proportional to ``variance``.  If True, the model variance
        is set of ``variance``

    Notes
    -----
    Allows a fixed set of variances to be used when estimating a mean model, allowing GLS
    estimation.
    """

    def __init__(self, variance: NDArray, unit_scale: bool = False) -> None:
        super().__init__()
        self._num_params = 0 if unit_scale else 1
        self._unit_scale = unit_scale
        self._name = "Fixed Variance"
        self._name += " (Unit Scale)" if unit_scale else ""
        self._variance_series = ensure1d(variance, "variance", True)
        self._variance = np.asarray(variance)

    def compute_variance(
        self,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> NDArray:
        if self._stop - self._start != sigma2.shape[0]:
            raise ValueError("start and stop do not have the correct values.")
        sigma2[:] = self._variance[self._start : self._stop]
        if not self._unit_scale:
            sigma2 *= parameters[0]
        return sigma2

    def starting_values(self, resids: NDArray) -> NDArray:
        if not self._unit_scale:
            _resids = resids / np.sqrt(self._variance[self._start : self._stop])
            return np.array([_resids.var()])
        return np.empty(0)

    def simulate(
        self,
        parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
        nobs: int,
        rng: RNGType,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
    ) -> Tuple[NDArray, NDArray]:
        raise NotImplementedError("Fixed Variance processes do not support simulation")

    def constraints(self) -> Tuple[NDArray, NDArray]:
        if not self._unit_scale:
            return np.ones((1, 1)), np.zeros(1)
        else:
            return np.ones((0, 0)), np.zeros(0)

    def backcast(self, resids: NDArray) -> Union[float, NDArray]:
        return 1.0

    def bounds(self, resids: NDArray) -> List[Tuple[float, float]]:
        if not self._unit_scale:
            v = float(np.squeeze(self.starting_values(resids)))
            _resids = resids / np.sqrt(self._variance[self._start : self._stop])
            mu = _resids.mean()
            return [(v / 100000.0, 10.0 * (v + mu ** 2.0))]
        return []

    def parameter_names(self) -> List[str]:
        if not self._unit_scale:
            return ["scale"]
        return []

    def _check_forecasting_method(self, method: str, horizon: int) -> None:
        return

    def _analytic_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
    ) -> VarianceForecast:
        t = resids.shape[0]
        forecasts = np.full((t, horizon), np.nan)

        return VarianceForecast(forecasts)

    def _simulation_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
        simulations: int,
        rng: RNGType,
    ) -> VarianceForecast:
        t = resids.shape[0]
        forecasts = np.full((t, horizon), np.nan)
        forecast_paths = np.empty((t, simulations, horizon))
        forecast_paths.fill(np.nan)
        shocks = np.full((t, simulations, horizon), np.nan)

        return VarianceForecast(forecasts, forecast_paths, shocks)


class FIGARCH(VolatilityProcess, metaclass=AbstractDocStringInheritor):
    r"""
    FIGARCH model

    Parameters
    ----------
    p : {0, 1}
        Order of the symmetric innovation
    q : {0, 1}
        Order of the lagged (transformed) conditional variance
    power : float, optional
        Power to use with the innovations, abs(e) ** power.  Default is 2.0,
        which produces FIGARCH and related models. Using 1.0 produces
        FIAVARCH and related models.  Other powers can be specified, although
        these should be strictly positive, and usually larger than 0.25.
    truncation : int, optional
        Truncation point to use in ARCH(:math:`\infty`) representation.
        Default is 1000.

    Examples
    --------
    >>> from arch.univariate import FIGARCH

    Standard FIGARCH

    >>> figarch = FIGARCH()

    FIARCH

    >>> fiarch = FIGARCH(p=0)

    FIAVGARCH process

    >>> fiavarch = FIGARCH(power=1.0)

    Notes
    -----
    In this class of processes, the variance dynamics are

    .. math::

        h_t = \omega + [1-\beta L - \phi L  (1-L)^d] \epsilon_t^2 + \beta h_{t-1}

    where ``L`` is the lag operator and ``d`` is the fractional differencing
    parameter. The model is estimated using the ARCH(:math:`\infty`)
    representation,

    .. math::

        h_t = (1-\beta)^{-1}  \omega + \sum_{i=1}^\infty \lambda_i \epsilon_{t-i}^2

    The weights are constructed using

    .. math::

        \delta_1 = d \\
        \lambda_1 = d - \beta + \phi

    and the recursive equations

    .. math::

        \delta_j = \frac{j - 1 - d}{j}  \delta_{j-1} \\
        \lambda_j = \beta \lambda_{j-1} + \delta_j - \phi \delta_{j-1}.

    When `power` is not 2, the ARCH(:math:`\infty`) representation is still used
    where :math:`\epsilon_t^2` is replaced by :math:`|\epsilon_t|^p` and
    ``p`` is the power.
    """

    def __init__(
        self, p: int = 1, q: int = 1, power: float = 2.0, truncation: int = 1000
    ) -> None:
        super().__init__()
        self.p: int = int(p)
        self.q: int = int(q)
        self.power: float = power
        self._num_params = 2 + p + q
        self._truncation = int(truncation)
        if p < 0 or q < 0 or p > 1 or q > 1:
            raise ValueError("p and q must be either 0 or 1.")
        if self._truncation <= 0:
            raise ValueError("truncation must be a positive integer")
        if power <= 0.0:
            raise ValueError(
                "power must be strictly positive, usually larger than 0.25"
            )
        self._name = self._generate_name()

    @property
    def truncation(self) -> int:
        """Truncation lag for the ARCH-infinity approximation"""
        return self._truncation

    def __str__(self) -> str:
        descr = self.name

        if self.power != 1.0 and self.power != 2.0:
            descr = descr[:-1] + ", "
        else:
            descr += "("
        for k, v in (("p", self.p), ("q", self.q)):
            descr += k + ": " + str(v) + ", "
        descr = descr[:-2] + ")"

        return descr

    def variance_bounds(self, resids: NDArray, power: float = 2.0) -> NDArray:
        return super().variance_bounds(resids, self.power)

    def _generate_name(self) -> str:
        q, power = self.q, self.power
        if power == 2.0:
            if q == 0:
                return "FIARCH"
            else:
                return "FIGARCH"
        elif power == 1.0:
            if q == 0:
                return "FIAVARCH"
            else:
                return "FIAVGARCH"
        else:
            if q == 0:
                return "Power FIARCH (power: {0:0.1f})".format(self.power)
            else:
                return "Power FIGARCH (power: {0:0.1f})".format(self.power)

    def bounds(self, resids: NDArray) -> List[Tuple[float, float]]:
        eps_half = np.sqrt(np.finfo(np.float64).eps)
        v = np.mean(abs(resids) ** self.power)

        bounds = [(0.0, 10.0 * float(v))]
        bounds.extend([(0.0, 0.5)] * self.p)  # phi
        bounds.extend([(0.0, 1.0 - eps_half)])  # d
        bounds.extend([(0.0, 1.0 - eps_half)] * self.q)  # beta

        return bounds

    def constraints(self) -> Tuple[NDArray, NDArray]:

        # omega > 0 <- 1
        # 0 <= d <= 1 <- 2
        # 0 <= phi <= (1 - d) / 2 <- 2
        # 0 <= beta <= d + phi <- 2
        a = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, -2, -1, 0],
                [0, 0, 1, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
                [0, 1, 1, -1],
            ]
        )
        b = np.array([0, 0, -1, 0, -1, 0, 0])
        if not self.q:
            a = a[:-2, :-1]
            b = b[:-2]
        if not self.p:
            # Drop column 1 and rows 1 and 2
            a = np.delete(a, (1,), axis=1)
            a = np.delete(a, (1, 2), axis=0)
            b = np.delete(b, (1, 2))

        return a, b

    def compute_variance(
        self,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> NDArray:
        # fresids is abs(resids) ** power
        power = self.power
        fresids = np.abs(resids) ** power

        p, q, truncation = self.p, self.q, self.truncation

        nobs = resids.shape[0]
        figarch_recursion(
            parameters, fresids, sigma2, p, q, nobs, truncation, backcast, var_bounds
        )
        inv_power = 2.0 / power
        sigma2 **= inv_power

        return sigma2

    def backcast_transform(
        self, backcast: Union[float, NDArray]
    ) -> Union[float, NDArray]:
        backcast = super().backcast_transform(backcast)
        return np.sqrt(backcast) ** self.power

    def backcast(self, resids: NDArray) -> Union[float, NDArray]:
        power = self.power
        tau = min(75, resids.shape[0])
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)
        backcast = float(np.sum((abs(resids[:tau]) ** power) * w))

        return backcast

    def simulate(
        self,
        parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
        nobs: int,
        rng: RNGType,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
    ) -> Tuple[NDArray, NDArray]:
        parameters = ensure1d(parameters, "parameters", False)
        truncation = self.truncation
        p, q, power = self.p, self.q, self.power
        lam = figarch_weights(parameters[1:], p, q, truncation)
        lam_rev = lam[::-1]
        errors = rng(truncation + nobs + burn)

        if initial_value is None:
            persistence = np.sum(lam)
            beta = parameters[-1] if q else 0.0

            initial_value = parameters[0]
            if beta < 1:
                initial_value /= 1 - beta
            if persistence < 1:
                initial_value /= 1 - persistence
            if persistence >= 1.0 or beta >= 1.0:
                warn(initial_value_warning, InitialValueWarning)
        assert initial_value is not None
        sigma2 = np.empty(truncation + nobs + burn)
        data = np.empty(truncation + nobs + burn)
        fsigma = np.empty(truncation + nobs + burn)
        fdata = np.empty(truncation + nobs + burn)

        fsigma[:truncation] = initial_value
        sigma2[:truncation] = initial_value ** (2.0 / power)
        data[:truncation] = np.sqrt(sigma2[:truncation]) * errors[:truncation]
        fdata[:truncation] = abs(data[:truncation]) ** power
        omega = parameters[0]
        beta = parameters[-1] if q else 0
        omega_tilde = omega / (1 - beta)
        for t in range(truncation, truncation + nobs + burn):
            fsigma[t] = omega_tilde + lam_rev.dot(fdata[t - truncation : t])
            sigma2[t] = fsigma[t] ** (2.0 / power)
            data[t] = errors[t] * np.sqrt(sigma2[t])
            fdata[t] = abs(data[t]) ** power

        return data[truncation + burn :], sigma2[truncation + burn :]

    def starting_values(self, resids: NDArray) -> NDArray:
        truncation = self.truncation
        ds = [0.2, 0.5, 0.7]
        phi_ratio = [0.2, 0.5, 0.8] if self.p else [0]
        beta_ratio = [0.1, 0.5, 0.9] if self.q else [0]

        power = self.power
        target = np.mean(abs(resids) ** power)
        scale = np.mean(resids ** 2) / (target ** (2.0 / power))
        target *= scale ** (power / 2)

        all_starting_vals = []
        for d in ds:
            for pr in phi_ratio:
                phi = (1 - d) / 2 * pr
                for br in beta_ratio:
                    beta = (d + phi) * br
                    temp = [phi, d, beta]
                    lam = figarch_weights(np.array(temp), 1, 1, truncation)
                    omega = (1 - beta) * target * (1 - np.sum(lam))
                    all_starting_vals.append((omega, phi, d, beta))
        distinct_svs = set(all_starting_vals)
        starting_vals = np.array(list(distinct_svs))
        if not self.q:
            starting_vals = starting_vals[:, :-1]
        if not self.p:
            starting_vals = np.c_[starting_vals[:, [0]], starting_vals[:, 2:]]

        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        llfs = np.zeros(len(starting_vals))
        for i, sv in enumerate(starting_vals):
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)

        return starting_vals[int(loc)]

    def parameter_names(self) -> List[str]:
        names = ["omega"]
        if self.p:
            names += ["phi"]
        names += ["d"]
        if self.q:
            names += ["beta"]
        return names

    def _check_forecasting_method(self, method: str, horizon: int) -> None:
        if horizon == 1:
            return

        if method == "analytic" and self.power != 2.0:
            raise ValueError(
                "Analytic forecasts not available for horizon > 1 when power != 2"
            )
        return

    def _analytic_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
    ) -> VarianceForecast:
        sigma2, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon, start
        )
        if horizon == 1:
            return VarianceForecast(forecasts)

        truncation = self.truncation
        p, q = self.p, self.q
        lam = figarch_weights(parameters[1:], p, q, truncation)
        lam_rev = lam[::-1]
        t = resids.shape[0]
        omega = parameters[0]
        beta = parameters[-1] if q else 0.0
        omega_tilde = omega / (1 - beta)
        temp_forecasts = np.empty(truncation + horizon)
        resids2 = resids ** 2
        for i in range(start, t):
            available = i + 1 - max(0, i - truncation + 1)
            temp_forecasts[truncation - available : truncation] = resids2[
                max(0, i - truncation + 1) : i + 1
            ]
            if available < truncation:
                temp_forecasts[: truncation - available] = backcast
            for h in range(horizon):
                lagged_forecasts = temp_forecasts[h : truncation + h]
                temp_forecasts[truncation + h] = omega_tilde + lam_rev.dot(
                    lagged_forecasts
                )
            forecasts[i, :] = temp_forecasts[truncation:]

        forecasts[:start] = np.nan
        return VarianceForecast(forecasts)

    def _simulation_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
        simulations: int,
        rng: RNGType,
    ) -> VarianceForecast:
        sigma2, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon, start
        )
        t = resids.shape[0]
        paths = np.empty((t - start, simulations, horizon))
        shocks = np.empty((t - start, simulations, horizon))

        power = self.power

        truncation = self.truncation
        p, q = self.p, self.q
        lam = figarch_weights(parameters[1:], p, q, truncation)
        lam_rev = lam[::-1]
        t = resids.shape[0]
        omega = parameters[0]
        beta = parameters[-1] if q else 0.0
        omega_tilde = omega / (1 - beta)
        fpath = np.empty((simulations, truncation + horizon))
        fresids = np.abs(resids) ** power

        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            available = i + 1 - max(0, i - truncation + 1)
            fpath[:, truncation - available : truncation] = fresids[
                max(0, i + 1 - truncation) : i + 1
            ]
            if available < truncation:
                fpath[:, : (truncation - available)] = backcast
            for h in range(horizon):
                # 1. Forecast transformed variance
                lagged_forecasts = fpath[:, h : truncation + h]
                temp = omega_tilde + lagged_forecasts.dot(lam_rev)
                # 2. Transform variance
                sigma2 = temp ** (2.0 / power)
                # 3. Simulate new residual
                path_loc = i - start
                shocks[path_loc, :, h] = std_shocks[:, h] * np.sqrt(sigma2)
                paths[path_loc, :, h] = sigma2
                forecasts[path_loc, h] = sigma2.mean()
                # 4. Transform new residual
                fpath[:, truncation + h] = np.abs(shocks[path_loc, :, h]) ** power

        return VarianceForecast(forecasts, paths, shocks)


class APARCH(VolatilityProcess, metaclass=AbstractDocStringInheritor):
    r"""
    Asymmetric Power ARCH (APARCH) volatility process

    Parameters
    ----------
    p : int
        Order of the symmetric innovation. Must satisfy p>=o.
    o : int
        Order of the asymmetric innovation. Must satisfy o<=p.
    q : int
        Order of the lagged (transformed) conditional variance
    delta : float, optional
        Value to use for a fixed delta in the APARCH model. If
        not provided, the value of delta is jointly estimated
        with other model parameters. User provided delta is restricted
        to lie in (0.05, 4.0).
    common_asym : bool, optional
        Restrict all asymmetry terms to share the same asymmetry
        parameter. If False (default), then there are no restrictions
        on the ``o`` asymmetry parameters.

    Examples
    --------
    >>> from arch.univariate import APARCH

    Symmetric Power ARCH(1,1)

    >>> aparch = APARCH(p=1, q=1)

    Standard APARCH process

    >>> aparch = APARCH(p=1, o=1, q=1)

    Fixed power parameters

    >>> aparch = APARCH(p=1, o=1, q=1, delta=1.3)

    Notes
    -----
    In this class of processes, the variance dynamics are

    .. math::

        \sigma_{t}^{\delta}=\omega
        +\sum_{i=1}^{p}\alpha_{i}
        \left(\left|\epsilon_{t-i}\right|
        -\gamma_{i}I_{[o\geq i]}\epsilon_{t-i}\right)^{\delta}
        +\sum_{k=1}^{q}\beta_{k}\sigma_{t-k}^{\delta}

    If ``common_asym`` is ``True``, then all of :math:`\gamma_i`
    are restricted to have a common value.
    """

    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        delta: Optional[float] = None,
        common_asym: bool = False,
    ) -> None:
        super().__init__()
        self.p: int = int(p)
        self.o: int = int(o)
        self.q: int = int(q)
        self._est_delta = delta is None
        self._common_asym = bool(common_asym) and self.o > 0
        self._delta = float(np.nan)
        if not self._est_delta:
            try:
                assert delta is not None
                self._delta = float(delta)
            except (ValueError, TypeError):
                raise TypeError("delta must be convertible to a float.")
            if not 0.05 < delta < 4:
                raise ValueError("delta must be between 0.05 and 4")
            self._delta = delta
        if p < 1 or o < 0 or q < 0:
            raise ValueError("All lags lengths must be non-negative, and p >= 1")
        if o > p:
            raise ValueError("o must be <= p.")
        self._name = "APARCH" if o > 0 else "Power ARCH"
        self._sigma_delta = np.empty(0)
        self._parameters = np.empty(1 + self.p + self.o + self.q + 1)
        o = 1 if self._common_asym else o
        self._num_params = 1 + p + o + q + int(self._est_delta)
        self._repack = self._common_asym or not self._est_delta

    @property
    def delta(self) -> float:
        """The value of delta in the model. NaN is delta is estimated."""
        return self._delta

    @property
    def common_asym(self) -> bool:
        """The value of delta in the model. NaN is delta is estimated."""
        return self._common_asym

    def _repack_parameters(self, parameters: NDArray) -> NDArray:
        if not self._repack:
            return parameters
        p, o, q = self.p, self.o, self.q
        _parameters = self._parameters
        if not self._common_asym:
            # Must only have fixed delta
            _parameters[: p + o + q + 1] = parameters
        else:
            _parameters[: p + 1] = parameters[: p + 1]
            _parameters[p + 1 : p + o + 1] = parameters[p + 1]
            _parameters[p + o + 1 : p + o + q + 1] = parameters[p + 2 : p + q + 2]
        _parameters[-1] = parameters[-1] if self._est_delta else self.delta
        return _parameters

    def compute_variance(
        self,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> NDArray:
        abs_resids = np.abs(resids)
        if self._sigma_delta.shape[0] != resids.shape[0]:
            self._sigma_delta = np.empty(resids.shape[0])
        sigma_delta = self._sigma_delta
        p, o, q = self.p, self.o, self.q
        nobs = resids.shape[0]
        _parameters = self._repack_parameters(parameters)
        aparch_recursion(
            _parameters,
            resids,
            abs_resids,
            sigma2,
            sigma_delta,
            p,
            o,
            q,
            nobs,
            backcast,
            var_bounds,
        )

        return sigma2

    def bounds(self, resids: NDArray) -> List[Tuple[float, float]]:
        v = max(float(np.mean(abs(resids) ** 0.5)), float(np.mean(resids ** 2)))

        bounds = [(0.0, 10.0 * float(v))]
        bounds.extend([(0.0, 1.0)] * self.p)
        o = 1 if self._common_asym else self.o
        bounds.extend([(-0.999998, 0.999998)] * o)
        bounds.extend([(0.0, 1.0)] * self.q)
        if self._est_delta:
            bounds.append((0.05, 4.0))

        return bounds

    def starting_values(self, resids: NDArray) -> NDArray:
        p, o, q = self.p, self.o, self.q
        alphas = [0.03, 0.05, 0.08, 0.15]
        alpha_beta = [0.8, 0.9, 0.95, 0.975, 0.99]
        gammas = [-0.5, 0, 0.5] if self.o > 0 else [0]
        deltas = [0.5, 1.2, 1.8] if self._est_delta else [self._delta]
        abgs = list(itertools.product(*[alphas, gammas, alpha_beta, deltas]))

        svs = []
        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        llfs = np.zeros(len(abgs))
        est_delta = int(self._est_delta)
        for i, values in enumerate(abgs):
            alpha, gamma, ab, delta = values

            target = np.mean(abs(resids) ** delta)
            scale = np.mean(resids ** 2) / (target ** (2.0 / delta))
            target *= scale ** (delta / 2)

            sv = (1.0 - ab) * target * np.ones(p + o + q + 1 + est_delta)
            sv[1 : 1 + p] = alpha / p
            ab -= alpha
            if o > 0:
                sv[1 + p : 1 + p + o] = gamma
            if q > 0:
                sv[1 + p + o : 1 + p + o + q] = ab / q
            if est_delta:
                sv[-1] = delta
            svs.append(sv)
            llfs[i] = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        loc = np.argmax(llfs)
        sv = svs[int(loc)]
        if self._common_asym:
            sv = np.r_[sv[: p + 1 + (o > 0)], sv[p + o + 1 :]]
        return sv

    def constraints(self) -> Tuple[NDArray, NDArray]:
        p, o, q = self.p, self.o, self.q
        o = 1 if self._common_asym else o
        k_arch = p + o + q
        # alpha[i] > 0, p
        # -1 < gamma[i] < 1, 2*o
        # beta[i] > 0, q
        # sum(alpha) + sum(beta) < 1, 1
        ndelta = 2 * int(self._est_delta)
        a = np.zeros((k_arch + o + 2 + ndelta, self._num_params))
        for i in range(p + 1):
            a[i, i] = 1.0
        for i in range(o):
            a[1 + p + i, 1 + p + i] = 1.0
            a[1 + p + o + i, 1 + p + i] = -1.0
        for i in range(q):
            a[1 + p + 2 * o, 1 + p + o + i] = 1.0
        if self._est_delta:
            a[1 + p + 2 * o + q, 1 + p + o + q] = 1.0
            a[1 + p + 2 * o + q + 1, 1 + p + o + q] = -1.0

        a[-1, 1 : p + o + q + 1] = -1.0
        a[-1, p + 1 : p + o + 1] = 0

        b = np.zeros(k_arch + o + 2 + ndelta)  # omega and alpha, > 0
        b[p + 1 : p + o + 1] = -0.999998  # gamma > -.999998
        b[p + o + 1 : p + 2 * o + 1] = -0.999998  # gamma < .999998
        if self._est_delta:
            b[-3] = 0.05  # delta > 0.05
            b[-2] = -4.0  # delta < 4
        b[-1] = -1.0  # sum < 1
        return a, b

    def parameter_names(self) -> List[str]:
        names = _common_names(self.p, self.o, self.q)
        if self._common_asym:
            names = names[: self.p + 1] + ["gamma"] + names[1 + self.p + self.o :]
        if self._est_delta:
            names += ["delta"]
        return names

    def simulate(
        self,
        parameters: Union[Sequence[Union[int, float]], ArrayLike1D],
        nobs: int,
        rng: RNGType,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
    ) -> Tuple[NDArray, NDArray]:
        params = ensure1d(parameters, "parameters", False)
        params = self._repack_parameters(params)
        p, o, q = self.p, self.o, self.q
        errors = rng(nobs + burn)

        sigma2 = np.zeros(nobs + burn)
        sigma_delta = np.zeros(nobs + burn)
        data = np.zeros(nobs + burn)
        adata = np.zeros(nobs + burn)
        max_lag = np.max([p, q])
        delta = params[-1]

        if initial_value is None:
            persistence = params[1 : p + 1].sum()
            persistence += params[1 + p + o : 1 + p + o + q].sum()
            if (1.0 - persistence) > 0:
                initial_value = params[0] / (1.0 - persistence)
            else:
                warn(initial_value_warning, InitialValueWarning)
                initial_value = params[0]
        sigma_delta[:max_lag] = initial_value
        sigma2[:max_lag] = initial_value ** (2.0 / delta)

        data[:max_lag] = np.sqrt(sigma2[:max_lag]) * errors[:max_lag]
        adata[:max_lag] = np.abs(data[:max_lag])

        for t in range(max_lag, nobs + burn):
            sigma_delta[t] = params[0]
            for j in range(p):
                shock = adata[t - 1 - j]
                if o > j:
                    shock -= params[1 + p + j] * data[t - 1 - j]
                sigma_delta[t] += params[1 + j] * shock ** delta
            for j in range(q):
                sigma_delta[t] += params[1 + p + o + j] * sigma_delta[t - 1 - j]

            sigma2[t] = sigma_delta[t] ** (2.0 / delta)
            data[t] = errors[t] * np.sqrt(sigma2[t])
            adata[t] = np.abs(data[t])

        return data[burn:], sigma2[burn:]

    def _simulate_paths(
        self,
        m: int,
        parameters: NDArray,
        horizon: int,
        std_shocks: NDArray,
        sigma_delta: NDArray,
        shock: NDArray,
        abs_shock: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray]:

        if self._est_delta:
            delta = parameters[-1]
        else:
            delta = self._delta
        p, o, q = self.p, self.o, self.q
        omega = parameters[0]
        alpha = parameters[1 : p + 1]
        gamma = parameters[p + 1 : p + o + 1]
        beta = parameters[p + o + 1 : p + o + q + 1]

        for h in range(horizon):
            loc = h + m - 1

            sigma_delta[:, h + m] = omega
            for j in range(p):
                _shock = abs_shock[:, loc - j]
                if self.o > j:
                    _shock -= gamma[j] * shock[:, loc - j]
                sigma_delta[:, h + m] += alpha[j] * (_shock ** delta)

            for j in range(q):
                sigma_delta[:, h + m] += beta[j] * sigma_delta[:, loc - j]

            shock[:, h + m] = std_shocks[:, h] * sigma_delta[:, h + m] ** (1.0 / delta)
            abs_shock[:, h + m] = np.abs(shock[:, h + m])

        forecast_paths = sigma_delta[:, m:] ** (2.0 / delta)

        return np.asarray(np.mean(forecast_paths, 0)), forecast_paths, shock[:, m:]

    def _simulation_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
        simulations: int,
        rng: RNGType,
    ) -> VarianceForecast:
        sigma2, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon, start
        )
        parameters = self._repack_parameters(parameters)
        t = resids.shape[0]
        paths = np.empty((t - start, simulations, horizon))
        shocks = np.empty((t - start, simulations, horizon))
        delta = parameters[-1]
        m = np.max([self.p, self.q])
        sigma_delta = np.zeros((simulations, m + horizon))
        shock = np.zeros((simulations, m + horizon))
        abs_shock = np.zeros((simulations, m + horizon))

        for i in range(start, t):
            std_shocks = rng((simulations, horizon))
            if i - m < 0:
                sigma_delta[:, :m] = backcast ** (delta / 2.0)
                shock[:, :m] = 0
                abs_shock[:, :m] = backcast ** (1.0 / 2.0)

                # Use actual values where available
                count = i + 1
                sigma_delta[:, m - count : m] = sigma2[:count] ** (delta / 2.0)
                shock[:, m - count : m] = resids[:count]
                abs_shock[:, m - count : m] = np.abs(resids[:count])
            else:
                sigma_delta[:, :m] = sigma2[i - m + 1 : i + 1] ** (delta / 2.0)
                shock[:, :m] = resids[i - m + 1 : i + 1]
                abs_shock[:, :m] = np.abs(resids[i - m + 1 : i + 1])

            f, p, s = self._simulate_paths(
                m,
                parameters,
                horizon,
                std_shocks,
                sigma_delta,
                shock,
                abs_shock,
            )
            loc = i - start
            forecasts[loc, :], paths[loc], shocks[loc] = f, p, s

        return VarianceForecast(forecasts, paths, shocks)

    def _analytic_forecast(
        self,
        parameters: NDArray,
        resids: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        start: int,
        horizon: int,
    ) -> VarianceForecast:
        _, forecasts = self._one_step_forecast(
            parameters, resids, backcast, var_bounds, horizon, start
        )

        return VarianceForecast(forecasts)

    def _check_forecasting_method(self, method: str, horizon: int) -> None:
        if horizon == 1:
            return

        if method == "analytic":
            raise ValueError("Analytic forecasts not available for horizon > 1")
        return

    def __str__(self) -> str:
        descr = self.name + "("
        for k, v in (("p", self.p), ("o", self.o), ("q", self.q)):
            if v > 0:
                descr += f"{k}: {v}, "
        if not self._est_delta:
            descr += f"delta: {self.delta:0.3f}, "
        if self.o > 0:
            descr += f"Common Asym: {self._common_asym}, "
        descr = descr[:-2] + ")"

        return descr
