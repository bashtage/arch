"""
Mean models to use with ARCH processes.  All mean models must inherit from
:class:`ARCHModel` and provide the same methods with the same inputs.
"""
from __future__ import annotations

import copy
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
from statsmodels.tsa.tsatools import lagmat

from arch.__future__._utility import check_reindex
from arch.typing import ArrayLike, ArrayLike1D, DateLike, Label, NDArray
from arch.univariate.base import (
    ARCHModel,
    ARCHModelForecast,
    ARCHModelResult,
    _callback_info,
    implicit_constant,
)
from arch.univariate.distribution import (
    Distribution,
    GeneralizedError,
    Normal,
    SkewStudent,
    StudentsT,
)
from arch.univariate.volatility import (
    ARCH,
    EGARCH,
    FIGARCH,
    GARCH,
    HARCH,
    ConstantVariance,
    VolatilityProcess,
)
from arch.utility.array import (
    AbstractDocStringInheritor,
    cutoff_to_index,
    ensure1d,
    parse_dataframe,
)
from arch.vendor import cached_property

if sys.version_info >= (3, 8):
    from typing import Literal
elif TYPE_CHECKING:
    from typing_extensions import Literal

__all__ = ["HARX", "ConstantMean", "ZeroMean", "ARX", "arch_model", "LS", "ARCHInMean"]

COV_TYPES = {
    "white": "White's Heteroskedasticity Consistent Estimator",
    "classic_ols": "Homoskedastic (Classic)",
    "robust": "Bollerslev-Wooldridge (Robust) Estimator",
    "mle": "ML Estimator",
    "classic": "ML Estimator",
}


def _forecast_pad(count: int, forecasts: NDArray) -> NDArray:
    shape = list(forecasts.shape)
    shape[0] = count
    fill = np.full(tuple(shape), np.nan)
    return np.concatenate((fill, forecasts))


def _ar_forecast(
    y: NDArray,
    horizon: int,
    start_index: int,
    constant: float,
    arp: NDArray,
    x: NDArray,
    exogp: NDArray,
) -> NDArray:
    """
    Generate mean forecasts from an AR-X model

    Parameters
    ----------
    y : ndarray
    horizon : int
    start_index : int
    constant : float
    arp : ndarray
    exogp : ndarray
    x : ndarray

    Returns
    -------
    forecasts : ndarray
    """
    t = y.shape[0]
    p = arp.shape[0]
    fcasts = np.empty((t - start_index, p + horizon))
    for i in range(p):
        first = start_index - p + i + 1
        last = t - p + i + 1
        fcasts[:, i] = y[first:last]
    arp_rev = arp[::-1]
    for i in range(p, horizon + p):
        fcasts[:, i] = constant + fcasts[:, i - p : i].dot(arp_rev)
        if x.shape[0] > 0:
            fcasts[:, i] += x[:, :, i - p].T @ exogp
    fcasts = fcasts[:, p:]

    return fcasts


def _ar_to_impulse(steps: int, params: NDArray) -> NDArray:
    p = params.shape[0]
    impulse = np.zeros(steps)
    impulse[0] = 1
    if p == 0:
        return impulse

    for i in range(1, steps):
        k = min(p - 1, i - 1)
        st = max(i - p, 0)
        impulse[i] = impulse[st:i].dot(params[k::-1])

    return impulse


class HARX(ARCHModel, metaclass=AbstractDocStringInheritor):
    r"""
    Heterogeneous Autoregression (HAR), with optional exogenous regressors,
    model estimation and simulation

    Parameters
    ----------
    y : {ndarray, Series}
        nobs element vector containing the dependent variable
    x : {ndarray, DataFrame}, optional
        nobs by k element array containing exogenous regressors
    lags : {scalar, ndarray}, optional
        Description of lag structure of the HAR.

        * Scalar included all lags between 1 and the value.
        * A 1-d n-element array includes the HAR lags 1:lags[0]+1,
          1:lags[1]+1, ... 1:lags[n]+1.
        * A 2-d (2,n)-element array that includes the HAR lags of the form
          lags[0,j]:lags[1,j]+1 for all columns of lags.

    constant : bool, optional
        Flag whether the model should include a constant
    use_rotated : bool, optional
        Flag indicating to use the alternative rotated form of the HAR where
        HAR lags do not overlap
    hold_back : int
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model
    distribution : Distribution, optional
        Error distribution to use in the model
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        than y is rescaled and the new scale is reported in the estimation results.

    Examples
    --------
    Standard HAR with average lags 1, 5 and 22

    >>> import numpy as np
    >>> from arch.univariate import HARX
    >>> y = np.random.RandomState(1234).randn(100)
    >>> harx = HARX(y, lags=[1, 5, 22])
    >>> res = harx.fit()


    A standard HAR with average lags 1 and 6 but holding back 10 observations

    >>> from pandas import Series, date_range
    >>> index = date_range('2000-01-01', freq='M', periods=y.shape[0])
    >>> y = Series(y, name='y', index=index)
    >>> har = HARX(y, lags=[1, 6], hold_back=10)

    Models with equivalent parametrizations of lags. The first uses
    overlapping lags.

    >>> harx_1 = HARX(y, lags=[1,5,22])

    The next uses rotated lags so that they do not overlap.

    >>> harx_2 = HARX(y, lags=[1,5,22], use_rotated=True)

    The third manually specified overlapping lags.

    >>> harx_3 = HARX(y, lags=[[1, 1, 1], [1, 5, 22]])

    The final manually specified non-overlapping lags

    >>> harx_4 = HARX(y, lags=[[1, 2, 6], [1, 5, 22]])

    It is simple to verify that these are the equivalent by inspecting the R2.

    >>> models = [harx_1, harx_2, harx_3, harx_4]
    >>> print([mod.fit().rsquared for mod in models])
    0.085, 0.085, 0.085, 0.085

    Notes
    -----
    The HAR-X model is described by

    .. math::

        y_t = \mu + \sum_{i=1}^p \phi_{L_{i}} \bar{y}_{t-L_{i,0}:L_{i,1}}
        + \gamma' x_t + \epsilon_t

    where :math:`\bar{y}_{t-L_{i,0}:L_{i,1}}` is the average value of
    :math:`y_t` between :math:`t-L_{i,0}` and :math:`t - L_{i,1}`.
    """
    # To allow for addition mean parameters in simulation
    _extra_simulation_params: int = 0

    def __init__(
        self,
        y: Optional[ArrayLike] = None,
        x: Optional[ArrayLike] = None,
        lags: Optional[
            Union[int, Sequence[int], Sequence[Sequence[int]], NDArray]
        ] = None,
        constant: bool = True,
        use_rotated: bool = False,
        hold_back: Optional[int] = None,
        volatility: Optional[VolatilityProcess] = None,
        distribution: Optional[Distribution] = None,
        rescale: Optional[bool] = None,
    ) -> None:
        super().__init__(
            y,
            hold_back=hold_back,
            volatility=volatility,
            distribution=distribution,
            rescale=rescale,
        )
        self._x = x
        self._x_names: List[str] = []
        self._x_index: Optional[Union[NDArray, pd.Index]] = None
        self.lags: Optional[
            Union[int, Sequence[int], Sequence[Sequence[int]], NDArray]
        ] = lags
        self._lags = np.empty((0, 0))
        self.constant: bool = constant
        self.use_rotated: bool = use_rotated
        self.regressors: np.ndarray[Any, np.dtype[np.float64]] = np.empty(
            (0, 0), dtype=np.float64
        )

        self._name = "HAR"
        if self._x is not None:
            self._name += "-X"
        if lags is not None:
            max_lags = int(np.max(np.asarray(lags, dtype=np.int32)))
        else:
            max_lags = 0
        self._max_lags = max_lags

        self._hold_back = max_lags if hold_back is None else hold_back

        if self._hold_back < max_lags:
            from warnings import warn

            warn(
                "hold_back is less then the minimum number given the lags selected",
                RuntimeWarning,
            )
            self._hold_back = max_lags

        self._init_model()

    @property
    def x(self) -> ArrayLike:
        """Gets the value of the exogenous regressors in the model"""
        return self._x

    def parameter_names(self) -> List[str]:
        return self._generate_variable_names()

    def _model_description(self, include_lags: bool = True) -> Dict[str, str]:
        """Generates the model description for use by __str__ and related
        functions"""
        lagstr = "none"
        if include_lags and self.lags is not None:
            assert self._lags is not None
            lagstr_comp = [f"[{lag[0]}:{lag[1]}]" for lag in self._lags.T]
            lagstr = ", ".join(lagstr_comp)
        xstr = str(self._x.shape[1]) if self._x is not None else "0"
        conststr = "yes" if self.constant else "no"
        od = {"constant": conststr}
        if include_lags:
            od["lags"] = lagstr
        od["no. of exog"] = xstr
        od["volatility"] = self.volatility.__str__()
        od["distribution"] = self.distribution.__str__()
        return od

    def __str__(self) -> str:
        descr = self._model_description()
        descr_str = self.name + "("
        for key, val in descr.items():
            if val and key:
                descr_str += key + ": " + val + ", "
        descr_str = descr_str[:-2]  # Strip final ', '
        descr_str += ")"

        return descr_str

    def __repr__(self) -> str:
        txt = self.__str__()
        txt.replace("\n", "")
        return txt + ", id: " + hex(id(self))

    def _repr_html_(self) -> str:
        """HTML representation for IPython Notebook"""
        descr = self._model_description()
        html = "<strong>" + self.name + "</strong>("
        for key, val in descr.items():
            html += "<strong>" + key + ": </strong>" + val + ",\n"
        html += "<strong>ID: </strong> " + hex(id(self)) + ")"
        return html

    def resids(
        self,
        params: NDArray,
        y: Optional[ArrayLike] = None,
        regressors: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        regressors = self._fit_regressors if y is None else regressors
        y = self._fit_y if y is None else y
        assert regressors is not None
        return y - regressors.dot(params)

    @cached_property
    def num_params(self) -> int:
        """
        Returns the number of parameters
        """
        assert self.regressors is not None
        return int(self.regressors.shape[1])

    def _simulate_mean(
        self,
        parameters: NDArray,
        x: NDArray,
        errors: NDArray,
        initial_value: Optional[Union[float, NDArray]],
        conditional_variance: NDArray,
    ) -> NDArray:
        max_lag = 0 if not self._lags.size else np.max(self._lags)
        nobs_and_burn = errors.shape[0]
        y = np.zeros(nobs_and_burn)
        if initial_value is None:
            initial_value = 0.0
        elif not np.isscalar(initial_value):
            initial_value = ensure1d(initial_value, "initial_value")
            if initial_value.shape[0] != max_lag:
                raise ValueError(
                    f"initial_value has the wrong shape. Expected {max_lag} values"
                )
        y[:max_lag] = initial_value
        k_x = x.shape[1]

        for t in range(max_lag, nobs_and_burn):
            ind = 0
            if self.constant:
                y[t] = parameters[ind]
                ind += 1
            for lag in self._lags.T:
                y[t] += parameters[ind] * y[t - lag[1] : t - lag[0]].mean()
                ind += 1
            for i in range(k_x):
                y[t] += parameters[ind] * x[t, i]
                ind += 1
            y[t] += errors[t]

        return y

    def simulate(
        self,
        params: Union[NDArray, Sequence[float]],
        nobs: int,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
        x: Optional[ArrayLike] = None,
        initial_value_vol: Optional[Union[float, NDArray]] = None,
    ) -> pd.DataFrame:
        """
        Simulates data from a linear regression, AR or HAR models

        Parameters
        ----------
        params : ndarray
            Parameters to use when simulating the model.  Parameter order is
            [mean volatility distribution] where the parameters of the mean
            model are ordered [constant lag[0] lag[1] ... lag[p] ex[0] ...
            ex[k-1]] where lag[j] indicates the coefficient on the jth lag in
            the model and ex[j] is the coefficient on the jth exogenous
            variable.
        nobs : int
            Length of series to simulate
        burn : int, optional
            Number of values to simulate to initialize the model and remove
            dependence on initial values.
        initial_value : {ndarray, float}, optional
            Either a scalar value or `max(lags)` array set of initial values to
            use when initializing the model.  If omitted, 0.0 is used.
        x : {ndarray, DataFrame}, optional
            nobs + burn by k array of exogenous variables to include in the
            simulation.
        initial_value_vol : {ndarray, float}, optional
            An array or scalar to use when initializing the volatility process.

        Returns
        -------
        simulated_data : DataFrame
            DataFrame with columns data containing the simulated values,
            volatility, containing the conditional volatility and errors
            containing the errors used in the simulation

        Examples
        --------
        >>> import numpy as np
        >>> from arch.univariate import HARX, GARCH
        >>> harx = HARX(lags=[1, 5, 22])
        >>> harx.volatility = GARCH()
        >>> harx_params = np.array([1, 0.2, 0.3, 0.4])
        >>> garch_params = np.array([0.01, 0.07, 0.92])
        >>> params = np.concatenate((harx_params, garch_params))
        >>> sim_data = harx.simulate(params, 1000)

        Simulating models with exogenous regressors requires the regressors
        to have nobs plus burn data points

        >>> nobs = 100
        >>> burn = 200
        >>> x = np.random.randn(nobs + burn, 2)
        >>> x_params = np.array([1.0, 2.0])
        >>> params = np.concatenate((harx_params, x_params, garch_params))
        >>> sim_data = harx.simulate(params, nobs=nobs, burn=burn, x=x)
        """

        if x is None:
            x = np.empty((nobs + burn, 0))
        else:
            x = np.asarray(x)
        k_x = x.shape[1]
        if x.shape[0] != nobs + burn:
            raise ValueError("x must have nobs + burn rows")
        assert self._lags is not None
        mc = (
            int(self.constant)
            + self._lags.shape[1]
            + k_x
            + self._extra_simulation_params
        )
        vc = self.volatility.num_params
        dc = self.distribution.num_params
        num_params = mc + vc + dc
        params = cast(NDArray, ensure1d(params, "params", series=False))
        if params.shape[0] != num_params:
            raise ValueError(
                "params has the wrong number of elements. "
                "Expected " + str(num_params) + ", got " + str(params.shape[0])
            )

        dist_params = [] if dc == 0 else params[-dc:]
        vol_params = params[mc : mc + vc]
        simulator = self.distribution.simulate(dist_params)
        sim_data = self.volatility.simulate(
            vol_params, nobs + burn, simulator, burn, initial_value_vol
        )
        errors = sim_data[0]
        vol = cast(NDArray, np.sqrt(sim_data[1]))

        y = self._simulate_mean(params[:mc], x, errors, initial_value, sim_data[1])

        df = dict(data=y[burn:], volatility=vol[burn:], errors=errors[burn:])
        return pd.DataFrame(df)

    def _generate_variable_names(self) -> List[str]:
        """Generates variable names or use in summaries"""
        variable_names = []
        lags = self._lags
        if self.constant:
            variable_names.append("Const")
        if lags is not None and lags.size:
            variable_names.extend(self._generate_lag_names())
        if self._x is not None:
            variable_names.extend(self._x_names)
        return variable_names

    def _generate_lag_names(self) -> List[str]:
        """Generates lag names.  Overridden by other models"""
        lags = self._lags
        names = []
        var_name = self._y_series.name
        if len(var_name) > 10:
            var_name = var_name[:4] + "..." + var_name[-3:]
        for i in range(lags.shape[1]):
            names.append(var_name + "[" + str(lags[0, i]) + ":" + str(lags[1, i]) + "]")
        return names

    def _check_specification(self) -> None:
        """Checks the specification for obvious errors """
        if self._x is not None:
            if isinstance(self._x, pd.Series):
                self._x = pd.DataFrame(self._x)
            elif self._x.ndim == 1:
                self._x = self._x[:, None]
            if self._x.ndim != 2 or self._x.shape[0] != self._y.shape[0]:
                raise ValueError(
                    "x must be nobs by n, where nobs is the same as "
                    "the number of elements in y"
                )
            def_names = ["x" + str(i) for i in range(self._x.shape[1])]
            names, self._x_index = parse_dataframe(self._x, def_names)
            self._x_names = [str(name) for name in names]
            self._x = np.asarray(self._x)

    def _reformat_lags(self) -> None:
        """
        Reformat input lags to be a 2 by m array, which simplifies other
        operations.  Output is stored in _lags
        """

        if self.lags is None:
            return
        lags = np.asarray(self.lags)
        if np.any(lags < 0):
            raise ValueError("Input to lags must be non-negative")

        if lags.ndim == 0:
            lags = np.arange(1, int(lags) + 1)

        if lags.ndim == 1:
            if np.any(lags <= 0):
                raise ValueError(
                    "When using the 1-d format of lags, values must be positive"
                )
            lags = np.unique(lags)
            temp = np.array([lags, lags])
            if self.use_rotated:
                temp[0, 1:] = temp[0, 0:-1]
                temp[0, 0] = 0
            else:
                temp[0, :] = 0
            self._lags = temp
        elif lags.ndim == 2:
            if lags.shape[0] != 2:
                raise ValueError("When using a 2-d array, lags must by k by 2")
            if np.any(lags[0] <= 0) or np.any(lags[1] < lags[0]):
                raise ValueError(
                    "When using a 2-d array, all values must be larger than 0 and "
                    "lags[0,j] <= lags[1,j] for all lags values."
                )
            ind = np.lexsort(np.flipud(lags))
            lags = lags[:, ind]
            test_mat = np.zeros((lags.shape[1], np.max(lags)))
            # Subtract 1 so first is 0 indexed
            lags = lags - np.array([[1], [0]])
            for i in range(lags.shape[1]):
                test_mat[i, lags[0, i] : lags[1, i]] = 1.0
            rank = np.linalg.matrix_rank(test_mat)
            if rank != lags.shape[1]:
                raise ValueError("lags contains redundant entries")

            self._lags = lags
            if self.use_rotated:
                from warnings import warn

                warn("Rotation is not available when using the 2-d lags input format")
        else:
            raise ValueError("Incorrect format for lags")

    def _har_to_ar(self, params: NDArray) -> NDArray:
        if self._max_lags == 0:
            return params
        har = params[int(self.constant) :]
        ar = np.zeros(self._max_lags)
        for value, lag in zip(har, self._lags.T):
            ar[lag[0] : lag[1]] += value / (lag[1] - lag[0])
        if self.constant:
            ar = np.concatenate((params[:1], ar))
        return ar

    def _init_model(self) -> None:
        """Should be called whenever the model is initialized or changed"""
        self._reformat_lags()
        self._check_specification()

        nobs_orig = self._y.shape[0]
        if self.constant:
            reg_constant = np.ones((nobs_orig, 1), dtype=np.float64)
        else:
            reg_constant = np.ones((nobs_orig, 0), dtype=np.float64)

        if self.lags is not None and nobs_orig > 0:
            maxlag = np.max(self.lags)
            lag_array = lagmat(self._y, maxlag)
            reg_lags = np.empty((nobs_orig, self._lags.shape[1]), dtype=np.float64)
            for i, lags in enumerate(self._lags.T):
                reg_lags[:, i] = np.mean(lag_array[:, lags[0] : lags[1]], 1)
        else:
            reg_lags = np.empty((nobs_orig, 0), dtype=np.float64)

        if self._x is not None:
            reg_x = self._x
        else:
            reg_x = np.empty((nobs_orig, 0), dtype=np.float64)

        self.regressors = np.hstack((reg_constant, reg_lags, reg_x))

    def _r2(self, params: NDArray) -> float:
        y = self._fit_y
        constant = False
        x = self._fit_regressors
        if x is not None and x.shape[1] > 0:
            constant = self.constant or implicit_constant(x)
        if constant:
            if x.shape[1] == 1:
                # Shortcut for constant only
                return 0.0
            y = y - np.mean(y)
        tss = float(y.dot(y))
        if tss <= 0.0:
            return np.nan
        e = self.resids(params)

        return 1.0 - float(e.T.dot(e)) / tss

    def _adjust_sample(
        self,
        first_obs: Optional[Union[int, DateLike]],
        last_obs: Optional[Union[int, DateLike]],
    ) -> None:
        index = self._y_series.index
        _first_obs_index = cutoff_to_index(first_obs, index, 0)
        _first_obs_index += self._hold_back
        _last_obs_index = cutoff_to_index(last_obs, index, self._y.shape[0])
        if _last_obs_index <= _first_obs_index:
            raise ValueError("first_obs and last_obs produce in an empty array.")
        self._fit_indices = [_first_obs_index, _last_obs_index]
        self._fit_y = self._y[_first_obs_index:_last_obs_index]
        reg = self.regressors
        self._fit_regressors = reg[_first_obs_index:_last_obs_index]
        self.volatility.start, self.volatility.stop = self._fit_indices

    def _fit_no_arch_normal_errors_params(self) -> NDArray:
        """
        Estimates model parameters excluding sigma2

        Returns
        -------
        params : ndarray
            Array of estiamted parameters
        """
        assert self._fit_y is not None
        nobs = self._fit_y.shape[0]

        if nobs < self.num_params:
            raise ValueError(
                "Insufficient data, "
                + str(self.num_params)
                + " regressors, "
                + str(nobs)
                + " data points available"
            )
        x = self._fit_regressors
        y = self._fit_y

        return np.empty(0) if x.shape[1] == 0 else np.linalg.pinv(x).dot(y)

    def _fit_no_arch_normal_errors(
        self, cov_type: Literal["robust", "classic"] = "robust"
    ) -> ARCHModelResult:
        """
        Estimates model parameters

        Parameters
        ----------
        cov_type : str, optional
            Covariance estimator to use when estimating parameter variances and
            covariances.  'robust' for Whites's covariance estimator, or 'classic' for
            the classic estimator appropriate for homoskedastic data.  'robust' is the
            the default.

        Returns
        -------
        result : ARCHModelResult
            Results class containing parameter estimates, estimated parameter
            covariance and related estimates

        Notes
        -----
        See :class:`ARCHModelResult` for details on computed results
        """
        assert self._fit_y is not None
        nobs = self._fit_y.shape[0]

        if nobs < self.num_params:
            raise ValueError(
                "Insufficient data, "
                + str(self.num_params)
                + " regressors, "
                + str(nobs)
                + " data points available"
            )
        x = self._fit_regressors
        y = self._fit_y

        # Fake convergence results, see GH #87
        opt = OptimizeResult({"status": 0, "message": ""})

        if x.shape[1] > 0:
            regression_params = np.linalg.pinv(x).dot(y)
            xpxi = np.linalg.inv(x.T.dot(x) / nobs)
            fitted = x.dot(regression_params)
        else:
            regression_params = np.empty(0)
            xpxi = np.empty((0, 0))
            fitted = 0.0

        e = y - fitted
        sigma2 = e.T.dot(e) / nobs

        params = np.hstack((regression_params, sigma2))
        hessian = np.zeros((self.num_params + 1, self.num_params + 1))
        hessian[: self.num_params, : self.num_params] = -xpxi
        hessian[-1, -1] = -1
        if cov_type in ("classic",):
            param_cov = sigma2 * -hessian
            param_cov[self.num_params, self.num_params] = 2 * sigma2 ** 2.0
            param_cov /= nobs
            cov_type_name = COV_TYPES["classic_ols"]
        elif cov_type in ("robust",):
            scores = np.zeros((nobs, self.num_params + 1))
            scores[:, : self.num_params] = x * e[:, None]
            scores[:, -1] = e ** 2.0 - sigma2
            score_cov = np.asarray(scores.T.dot(scores) / nobs)
            param_cov = (hessian @ score_cov @ hessian) / nobs
            cov_type_name = COV_TYPES["white"]
        else:
            raise ValueError("Unknown cov_type")

        r2 = self._r2(regression_params)

        first_obs, last_obs = self._fit_indices
        resids = np.empty_like(self._y, dtype=np.float64)
        resids.fill(np.nan)
        resids[first_obs:last_obs] = e
        vol = np.zeros_like(resids)
        vol.fill(np.nan)
        vol[first_obs:last_obs] = np.sqrt(sigma2)
        names = self._all_parameter_names()
        loglikelihood = self._static_gaussian_loglikelihood(e)

        # Throw away names in the case of starting values
        num_params = params.shape[0]
        if len(names) != num_params:
            names = ["p" + str(i) for i in range(num_params)]

        fit_start, fit_stop = self._fit_indices
        return ARCHModelResult(
            params,
            param_cov,
            r2,
            resids,
            vol,
            cov_type_name,
            self._y_series,
            names,
            loglikelihood,
            self._is_pandas,
            opt,
            fit_start,
            fit_stop,
            copy.deepcopy(self),
        )

    def _reformat_forecast_x(
        self,
        x: Union[None, Dict[Label, ArrayLike], ArrayLike],
        horizon: int,
        start: int,
    ) -> NDArray:
        """
        Always return a correctly formatted 3-d array

        Parameters
        ----------
        x: Union[None, Dict[Label, ArrayLike], ArrayLike] = None
            The input data to reformat

        Returns
        -------
        ndarray
            The 3-d array suitable for computing forecasts. Always has shape
            (nx, nobs - start, horizon).
        """
        if x is None:
            if self._x is None:
                return np.empty(0)
            else:
                raise TypeError(
                    "x is None but the model contains exogenous variables. You must "
                    "provide expected values to use for the exogenous variables to "
                    "construct forecasts."
                )
        elif self._x is None:
            raise TypeError(
                "x is not None but the model does not contain any exogenous "
                "variables."
            )
        assert self._x is not None
        nx = self._x.shape[1]
        if isinstance(x, Mapping):
            if len(x) != nx:
                raise ValueError(
                    "x must have the same number of entries as the number of x "
                    f"variables included in the model ({nx})"
                )
            collected: List[NDArray] = []
            for key in self._x_names:
                if key not in x:
                    keys = ", ".join([str(k) for k in x.keys()])
                    raise KeyError(
                        "The keys of x must exactly match the variable names of "
                        f"the included exogenous regressors. {key} not found in: "
                        f"{keys}"
                    )
                temp = np.asarray(x[key])
                if temp.ndim == 1:
                    temp = temp.reshape((1, -1))
                collected.append(temp)
            base_shape = collected[0].shape
            shapes = [v.shape == base_shape for v in collected]
            if not all(shapes):
                raise ValueError(
                    "The shapes of the arrays contained in the dictionary differ. "
                    "These must all be the same and satisfy the requirement for "
                    "expected x values."
                )
            if len(base_shape) != 2:
                raise ValueError(
                    "The arrays contained in the dictionary must be 1 or 2-dimensional."
                )
            arr: NDArray = np.array(collected)
        else:
            arr = np.asarray(x)

            if arr.ndim == 1:
                arr = arr.reshape((1, -1))
            if arr.ndim == 2:
                if nx != 1:
                    raise ValueError(
                        "1- and 2-dimensional x values can only be used when the "
                        f"model has a single exogenous regressor.  The model has {nx} "
                        "exogenous regressors, and so you must use either a "
                        "dictionary or a 3-dimensional NumPy array."
                    )
                arr = arr.reshape((1,) + arr.shape)
            if arr.shape[0] != self._x.shape[1]:
                raise ValueError(
                    "The leading dimension of x must match the number of x variables "
                    f"included in the model ({nx})"
                )
        assert isinstance(arr, np.ndarray)
        if arr.shape[2] != horizon:
            raise ValueError(
                f"The number of values passed {arr.shape[2]} does not match the "
                f"horizon of the forecasts ({horizon})."
            )
        possible_nobs = (self._y.shape[0], self._y.shape[0] - start)
        if arr.shape[1] not in possible_nobs:
            raise ValueError(
                "The shape of x does not satisfy the requirements for expected "
                "exogenous regressors.  The number of values must either match "
                f"the original sample size ({self._y.shape[0]}) or the number of "
                f"forecasts ({self._y.shape[0] - start})."
            )
        if arr.shape[1] > (self._y.shape[0] - start):
            arr = arr[:, start:]
        return arr

    def forecast(
        self,
        params: ArrayLike1D,
        horizon: int = 1,
        start: Optional[Union[int, DateLike]] = None,
        align: str = "origin",
        method: str = "analytic",
        simulations: int = 1000,
        rng: Optional[Callable[[Union[int, Tuple[int, ...]]], NDArray]] = None,
        random_state: Optional[np.random.RandomState] = None,
        *,
        reindex: Optional[bool] = None,
        x: Union[None, Dict[Label, ArrayLike], ArrayLike] = None,
    ) -> ARCHModelForecast:
        reindex = check_reindex(reindex)
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be an integer >= 1.")
        # Check start
        earliest, default_start = self._fit_indices
        default_start = max(0, default_start - 1)
        start_index = cutoff_to_index(start, self._y_series.index, default_start)
        if start_index < (earliest - 1):
            raise ValueError(
                "Due to backcasting and/or data availability start cannot be less "
                "than the index of the largest value in the right-hand-side "
                "variables used to fit the first observation.  In this model, "
                "this value is {0}.".format(max(0, earliest - 1))
            )
        # Parse params
        params = np.asarray(params)
        mp, vp, dp = self._parse_parameters(params)

        #####################################
        # Compute residual variance forecasts
        #####################################
        # Back cast should use only the sample used in fitting
        resids = self.resids(mp)
        backcast = self._volatility.backcast(resids)
        full_resids = self.resids(mp, self._y[earliest:], self.regressors[earliest:])
        vb = self._volatility.variance_bounds(full_resids, 2.0)
        if rng is None:
            rng = self._distribution.simulate(dp)
        variance_start = max(0, start_index - earliest)
        vfcast = self._volatility.forecast(
            vp,
            full_resids,
            backcast,
            vb,
            start=variance_start,
            horizon=horizon,
            method=method,
            simulations=simulations,
            rng=rng,
            random_state=random_state,
        )
        var_fcasts = vfcast.forecasts
        assert var_fcasts is not None
        if start_index < earliest:
            # Pad if asking for variance forecast before earliest available
            var_fcasts = _forecast_pad(earliest - start_index, var_fcasts)

        arp = self._har_to_ar(mp)
        nexog = 0 if self._x is None else self._x.shape[1]
        exog_p = np.empty([]) if self._x is None else mp[-nexog:]
        constant = arp[0] if self.constant else 0.0
        dynp = arp[int(self.constant) :]
        expected_x = self._reformat_forecast_x(x, horizon, start_index)
        mean_fcast = _ar_forecast(
            self._y, horizon, start_index, constant, dynp, expected_x, exog_p
        )
        # Compute total variance forecasts, which depend on model
        impulse = _ar_to_impulse(horizon, dynp)
        longrun_var_fcasts = var_fcasts.copy()
        for i in range(horizon):
            lrf = var_fcasts[:, : (i + 1)].dot(impulse[i::-1] ** 2)
            longrun_var_fcasts[:, i] = lrf
        variance_paths: Optional[NDArray] = None
        mean_paths: Optional[NDArray] = None
        shocks: Optional[NDArray] = None
        long_run_variance_paths: Optional[NDArray] = None
        if method.lower() in ("simulation", "bootstrap"):
            # TODO: This is not tested, but probably right
            assert isinstance(vfcast.forecast_paths, np.ndarray)
            variance_paths = vfcast.forecast_paths
            assert isinstance(vfcast.shocks, np.ndarray)
            shocks = vfcast.shocks
            if start_index < earliest:
                # Pad if asking for variance forecast before earliest available
                variance_paths = _forecast_pad(earliest - start_index, variance_paths)
                shocks = _forecast_pad(earliest - start_index, shocks)

            long_run_variance_paths = variance_paths.copy()
            for i in range(horizon):
                _impulses = impulse[i::-1][:, None]
                lrvp = variance_paths[:, :, : (i + 1)].dot(_impulses ** 2)
                long_run_variance_paths[:, :, i] = np.squeeze(lrvp)
            t, m = self._y.shape[0], self._max_lags
            mean_paths = np.empty(shocks.shape[:2] + (m + horizon,))
            dynp_rev = dynp[::-1]
            for i in range(start_index, t):
                path_loc = i - start_index
                mean_paths[path_loc, :, :m] = self._y[i - m + 1 : i + 1]

                for j in range(horizon):
                    mean_paths[path_loc, :, m + j] = (
                        constant
                        + mean_paths[path_loc, :, j : m + j].dot(dynp_rev)
                        + shocks[path_loc, :, j]
                    )
                    if expected_x.shape[0] > 0:
                        mean_paths[path_loc, :, m + j] += expected_x[:, :, j].T @ exog_p

            mean_paths = mean_paths[:, :, m:]

        index = self._y_series.index
        reindex = True if reindex is None else reindex
        return ARCHModelForecast(
            index,
            start_index,
            mean_fcast,
            longrun_var_fcasts,
            var_fcasts,
            align=align,
            simulated_paths=mean_paths,
            simulated_residuals=shocks,
            simulated_variances=long_run_variance_paths,
            simulated_residual_variances=variance_paths,
            reindex=reindex,
        )


class ConstantMean(HARX):
    r"""
    Constant mean model estimation and simulation.

    Parameters
    ----------
    y : {ndarray, Series}
        nobs element vector containing the dependent variable
    hold_back : int
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model
    distribution : Distribution, optional
        Error distribution to use in the model
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        than y is rescaled and the new scale is reported in the estimation results.

    Examples
    --------
    >>> import numpy as np
    >>> from arch.univariate import ConstantMean
    >>> y = np.random.randn(100)
    >>> cm = ConstantMean(y)
    >>> res = cm.fit()

    Notes
    -----
    The constant mean model is described by

    .. math::

        y_t = \mu + \epsilon_t
    """

    def __init__(
        self,
        y: Optional[ArrayLike] = None,
        hold_back: Optional[int] = None,
        volatility: Optional[VolatilityProcess] = None,
        distribution: Optional[Distribution] = None,
        rescale: Optional[bool] = None,
    ) -> None:
        super().__init__(
            y,
            hold_back=hold_back,
            volatility=volatility,
            distribution=distribution,
            rescale=rescale,
        )
        self._name = "Constant Mean"

    def parameter_names(self) -> List[str]:
        return ["mu"]

    @cached_property
    def num_params(self) -> int:
        return 1

    def _model_description(self, include_lags: bool = False) -> Dict[str, str]:
        return super()._model_description(include_lags)

    def simulate(
        self,
        params: ArrayLike,
        nobs: int,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
        x: Optional[ArrayLike] = None,
        initial_value_vol: Optional[Union[float, NDArray]] = None,
    ) -> pd.DataFrame:
        """
        Simulated data from a constant mean model

        Parameters
        ----------
        params : ndarray
            Parameters to use when simulating the model.  Parameter order is
            [mean volatility distribution]. There is one parameter in the mean
            model, mu.
        nobs : int
            Length of series to simulate
        burn : int, optional
            Number of values to simulate to initialize the model and remove
            dependence on initial values.
        initial_value : None
            This value is not used.
        x : None
            This value is not used.
        initial_value_vol : {ndarray, float}, optional
            An array or scalar to use when initializing the volatility process.

        Returns
        -------
        simulated_data : DataFrame
            DataFrame with columns data containing the simulated values,
            volatility, containing the conditional volatility and errors
            containing the errors used in the simulation

        Examples
        --------
        Basic data simulation with a constant mean and volatility

        >>> import numpy as np
        >>> from arch.univariate import ConstantMean, GARCH
        >>> cm = ConstantMean()
        >>> cm.volatility = GARCH()
        >>> cm_params = np.array([1])
        >>> garch_params = np.array([0.01, 0.07, 0.92])
        >>> params = np.concatenate((cm_params, garch_params))
        >>> sim_data = cm.simulate(params, 1000)
        """
        if initial_value is not None or x is not None:
            raise ValueError(
                "Both initial value and x must be none when "
                "simulating a constant mean process."
            )

        mp, vp, dp = self._parse_parameters(params)

        sim_values = self.volatility.simulate(
            vp, nobs + burn, self.distribution.simulate(dp), burn, initial_value_vol
        )
        errors = sim_values[0]
        y = errors + mp
        vol = np.sqrt(sim_values[1])
        assert isinstance(vol, np.ndarray)
        df = dict(data=y[burn:], volatility=vol[burn:], errors=errors[burn:])
        df = pd.DataFrame(df)
        return df

    def resids(
        self,
        params: NDArray,
        y: Optional[ArrayLike] = None,
        regressors: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        y = self._fit_y if y is None else y
        return y - params


class ZeroMean(HARX):
    r"""
    Model with zero conditional mean estimation and simulation

    Parameters
    ----------
    y : {ndarray, Series}
        nobs element vector containing the dependent variable
    hold_back : int
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model
    distribution : Distribution, optional
        Error distribution to use in the model
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        than y is rescaled and the new scale is reported in the estimation results.

    Examples
    --------
    >>> import numpy as np
    >>> from arch.univariate import ZeroMean
    >>> y = np.random.randn(100)
    >>> zm = ZeroMean(y)
    >>> res = zm.fit()

    Notes
    -----
    The zero mean model is described by

    .. math::

        y_t = \epsilon_t

    """

    def __init__(
        self,
        y: Optional[ArrayLike] = None,
        hold_back: Optional[int] = None,
        volatility: Optional[VolatilityProcess] = None,
        distribution: Optional[Distribution] = None,
        rescale: Optional[bool] = None,
    ) -> None:
        super().__init__(
            y,
            x=None,
            constant=False,
            hold_back=hold_back,
            volatility=volatility,
            distribution=distribution,
            rescale=rescale,
        )
        self._name = "Zero Mean"

    def parameter_names(self) -> List[str]:
        return []

    @cached_property
    def num_params(self) -> int:
        return 0

    def _model_description(self, include_lags: bool = False) -> Dict[str, str]:
        return super()._model_description(include_lags)

    def simulate(
        self,
        params: Union[Sequence[float], ArrayLike1D],
        nobs: int,
        burn: int = 500,
        initial_value: Optional[Union[float, NDArray]] = None,
        x: Optional[ArrayLike] = None,
        initial_value_vol: Optional[Union[float, NDArray]] = None,
    ) -> pd.DataFrame:
        """
        Simulated data from a zero mean model

        Parameters
        ----------
        params : {ndarray, DataFrame}
            Parameters to use when simulating the model.  Parameter order is
            [volatility distribution]. There are no mean parameters.
        nobs : int
            Length of series to simulate
        burn : int, optional
            Number of values to simulate to initialize the model and remove
            dependence on initial values.
        initial_value : None
            This value is not used.
        x : None
            This value is not used.
        initial_value_vol : {ndarray, float}, optional
            An array or scalar to use when initializing the volatility process.

        Returns
        -------
        simulated_data : DataFrame
            DataFrame with columns data containing the simulated values,
            volatility, containing the conditional volatility and errors
            containing the errors used in the simulation

        Examples
        --------
        Basic data simulation with no mean and constant volatility

        >>> from arch.univariate import ZeroMean
        >>> import numpy as np
        >>> zm = ZeroMean()
        >>> params = np.array([1.0])
        >>> sim_data = zm.simulate(params, 1000)

        Simulating data with a non-trivial volatility process

        >>> from arch.univariate import GARCH
        >>> zm.volatility = GARCH(p=1, o=1, q=1)
        >>> sim_data = zm.simulate([0.05, 0.1, 0.1, 0.8], 300)
        """
        params = ensure1d(params, "params", False)
        if initial_value is not None or x is not None:
            raise ValueError(
                "Both initial value and x must be none when "
                "simulating a constant mean process."
            )

        _, vp, dp = self._parse_parameters(params)

        sim_values = self.volatility.simulate(
            vp, nobs + burn, self.distribution.simulate(dp), burn, initial_value_vol
        )
        errors = sim_values[0]
        y = errors
        vol = np.sqrt(sim_values[1])
        assert isinstance(vol, np.ndarray)
        df = dict(data=y[burn:], volatility=vol[burn:], errors=errors[burn:])
        df = pd.DataFrame(df)

        return df

    def resids(
        self,
        params: NDArray,
        y: Optional[ArrayLike] = None,
        regressors: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        if y is not None:
            return y
        assert self._fit_y is not None
        return self._fit_y


class ARX(HARX):
    r"""
    Autoregressive model with optional exogenous regressors estimation and
    simulation

    Parameters
    ----------
    y : {ndarray, Series}
        nobs element vector containing the dependent variable
    x : {ndarray, DataFrame}, optional
        nobs by k element array containing exogenous regressors
    lags : scalar, 1-d array, optional
        Description of lag structure of the HAR.  Scalar included all lags
        between 1 and the value.  A 1-d array includes the AR lags lags[0],
        lags[1], ...
    constant : bool, optional
        Flag whether the model should include a constant
    hold_back : int
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model
    distribution : Distribution, optional
        Error distribution to use in the model
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        than y is rescaled and the new scale is reported in the estimation results.

    Examples
    --------
    >>> import numpy as np
    >>> from arch.univariate import ARX
    >>> y = np.random.randn(100)
    >>> arx = ARX(y, lags=[1, 5, 22])
    >>> res = arx.fit()

    Estimating an AR with GARCH(1,1) errors

    >>> from arch.univariate import GARCH
    >>> arx.volatility = GARCH()
    >>> res = arx.fit(update_freq=0, disp='off')

    Notes
    -----
    The AR-X model is described by

    .. math::

        y_t = \mu + \sum_{i=1}^p \phi_{L_{i}} y_{t-L_{i}} + \gamma' x_t
        + \epsilon_t

    """

    def __init__(
        self,
        y: Optional[ArrayLike] = None,
        x: Optional[ArrayLike] = None,
        lags: Optional[Union[int, List[int], NDArray]] = None,
        constant: bool = True,
        hold_back: Optional[int] = None,
        volatility: Optional[VolatilityProcess] = None,
        distribution: Optional[Distribution] = None,
        rescale: Optional[bool] = None,
    ) -> None:
        # Convert lags to 2-d format

        if lags is not None:
            lags_arr = np.asarray(lags)
            assert lags_arr is not None
            if lags_arr.ndim == 0:
                if lags_arr < 0:
                    raise ValueError("lags must be a positive integer.")
                elif lags_arr == 0:
                    lags = None
                else:
                    lags_arr = np.arange(1, int(lags_arr) + 1)
            if lags is not None:
                if lags_arr.ndim != 1:
                    raise ValueError("lags does not follow a supported format")
                else:
                    lags_arr = np.vstack((lags_arr, lags_arr))
                    assert lags_arr is not None

        super().__init__(
            y,
            x,
            None if lags is None else lags_arr,
            constant,
            False,
            hold_back,
            volatility=volatility,
            distribution=distribution,
            rescale=rescale,
        )
        self._name = "AR"
        if self._x is not None:
            self._name += "-X"

    def _model_description(self, include_lags: bool = True) -> Dict[str, str]:
        """Generates the model description for use by __str__ and related
        functions"""
        lagstr = "none"
        if include_lags and self.lags is not None:
            assert self._lags is not None
            lagstr_comp = [str(lag[1]) for lag in self._lags.T]
            lagstr = ", ".join(lagstr_comp)

        xstr = str(self._x.shape[1]) if self._x is not None else "0"
        conststr = "yes" if self.constant else "no"
        od = {"constant": conststr}
        if include_lags:
            od["lags"] = lagstr
        od["no. of exog"] = xstr
        od["volatility"] = self.volatility.__str__()
        od["distribution"] = self.distribution.__str__()
        return od

    def _generate_lag_names(self) -> List[str]:
        lags = self._lags
        names = []
        var_name = self._y_series.name
        if len(var_name) > 10:
            var_name = var_name[:4] + "..." + var_name[-3:]
        for i in range(lags.shape[1]):
            names.append(var_name + "[" + str(lags[1, i]) + "]")
        return names


class LS(HARX):
    r"""
    Least squares model estimation and simulation

    Parameters
    ----------
    y : {ndarray, Series}
        nobs element vector containing the dependent variable
    y : {ndarray, DataFrame}, optional
        nobs by k element array containing exogenous regressors
    constant : bool, optional
        Flag whether the model should include a constant
    hold_back : int
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model
    distribution : Distribution, optional
        Error distribution to use in the model
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        than y is rescaled and the new scale is reported in the estimation results.

    Examples
    --------
    >>> import numpy as np
    >>> from arch.univariate import LS
    >>> y = np.random.randn(100)
    >>> x = np.random.randn(100,2)
    >>> ls = LS(y, x)
    >>> res = ls.fit()

    Notes
    -----
    The LS model is described by

    .. math::

        y_t = \mu + \gamma' x_t + \epsilon_t

    """

    def __init__(
        self,
        y: Optional[ArrayLike] = None,
        x: Optional[ArrayLike] = None,
        constant: bool = True,
        hold_back: Optional[int] = None,
        volatility: Optional[VolatilityProcess] = None,
        distribution: Optional[Distribution] = None,
        rescale: Optional[bool] = None,
    ) -> None:
        # Convert lags to 2-d format
        super().__init__(
            y,
            x,
            None,
            constant,
            False,
            hold_back=hold_back,
            volatility=volatility,
            distribution=distribution,
            rescale=rescale,
        )
        self._name = "Least Squares"


class ARCHInMean(ARX):
    r"""
    (G)ARCH-in-mean model and simulation

    Parameters
    ----------
    y : {ndarray, Series}
        nobs element vector containing the dependent variable
    x : {ndarray, DataFrame}, optional
        nobs by k element array containing exogenous regressors
    lags : {scalar, 1-d array}, optional
        Description of lag structure of the HAR.  Scalar included all lags
        between 1 and the value.  A 1-d array includes the AR lags lags[0],
        lags[1], ...
    constant : bool, optional
        Flag whether the model should include a constant
    hold_back : int, optional
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model. ``volatility.updateable`` must
        return ``True``.
    distribution : Distribution, optional
        Error distribution to use in the model
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        than y is rescaled and the new scale is reported in the estimation results.
    form : {"log", "vol", "var", int, float}
        The form of the conditional variance that appears in the mean equation. The
        string names use the log of the conditional variance ("log"), the square-root
        of the conditional variance ("vol") or the conditional variance. When
        specified using a float, interpreted as :math:`\sigma_t^{form}` so that
        1 is equivalent to "vol" and 2 is equivalent to "var". When using a number,
        must be different from 0.

    Examples
    --------
    >>> import numpy as np
    >>> from arch.univariate import ARCHInMean, GARCH
    >>> from arch.data.sp500 import load
    >>> sp500 = load()
    >>> rets = 100 * sp500["Adj Close"].pct_change().dropna()
    >>> gim = ARCHInMean(rets, lags=[1, 2], volatility=GARCH())
    >>> res = gim.fit()

    Notes
    -----
    The (G)arch-in-mean model with exogenous regressors (-X)  is described by

    .. math::

        y_t = \mu + \kappa f(\sigma^2_t)+ \sum_{i=1}^p \phi_{L_{i}} y_{t-L_{i}}
              + \gamma' x_t +  \epsilon_t

    where :math:`f(\cdot)` is the function specified by ``form``.

    **Performance**

    Estimating ARCH-in-mean models is slower than estimating other models
    because the critical loop is compsed on code from multiple functions
    and so is written in Python.  It is possible to accelerate this, and
    performance may improve in the future, depending on whether there is
    and expressed interest.
    """
    # To account for GIM param
    _extra_simulation_params = 1

    def __init__(
        self,
        y: Optional[ArrayLike] = None,
        x: Optional[ArrayLike] = None,
        lags: Optional[Union[int, List[int], NDArray]] = None,
        constant: bool = True,
        hold_back: Optional[int] = None,
        volatility: Optional[VolatilityProcess] = None,
        distribution: Optional[Distribution] = None,
        rescale: Optional[bool] = None,
        form: Union[int, float, Literal["log", "vol", "var"]] = "vol",
    ) -> None:
        super().__init__(
            y, x, lags, constant, hold_back, volatility, distribution, rescale
        )
        self._name = "ARCH-in-mean"
        form_err = (
            "form must be a floating point number of one of 'log', 'vol' or 'var', got "
        )
        if not isinstance(form, (str, int, float, np.floating, np.integer)):
            raise TypeError(form_err + f"{type(form)}")
        if isinstance(form, str):
            lower_form = form.lower()
            if lower_form not in ("log", "vol", "var"):
                raise ValueError(form_err + f"{form}")
            self._form_id = {"log": 0, "vol": 1, "var": 2}[lower_form]
            self._form_power = float(self._form_id)
        else:
            float_form = float(form)
            if float_form == 0:
                raise ValueError(
                    "When using a floating point number for form, its value "
                    "must be different from 0"
                )
            self._form_id = 3
            self._form_power = float_form
        self._form = form

        if not self.volatility.updateable:
            raise ValueError(
                f"The volatility process {self.volatility.__class__} is not "
                "updateable. Only processes that have implemented the "
                "``update`` function and ``updateable`` returns True can be "
                "used with ``ARCHInMean``."
            )

    @property
    def form(self) -> Union[int, float, Literal["log", "vol", "var"]]:
        """The form of the conditional variance in the mean"""
        return self._form

    def _model_description(self, include_lags: bool = False) -> Dict[str, str]:
        base_descr = super()._model_description(include_lags)
        keys = list(base_descr.keys())
        descr = {key: base_descr[key] for key in keys[:-2]}
        if not isinstance(self._form, str):
            form = f"{float(self._form)} (numeric)"
        else:
            form = self._form
        descr["form"] = str(form)
        for key in keys[-2:]:
            descr[key] = base_descr[key]
        return descr

    def _generate_variable_names(self) -> List[str]:
        names = super()._generate_variable_names()
        return names + ["kappa"]

    @cached_property
    def num_params(self) -> int:
        assert self.regressors is not None
        return int(self.regressors.shape[1]) + 1

    def forecast(
        self,
        params: ArrayLike1D,
        horizon: int = 1,
        start: Optional[Union[int, DateLike]] = None,
        align: str = "origin",
        method: str = "analytic",
        simulations: int = 1000,
        rng: Optional[Callable[[Union[int, Tuple[int, ...]]], NDArray]] = None,
        random_state: Optional[np.random.RandomState] = None,
        *,
        reindex: Optional[bool] = None,
        x: Union[None, Dict[Label, ArrayLike], ArrayLike] = None,
    ) -> ARCHModelForecast:
        raise NotImplementedError(
            "forecasts are not implemented for (G)ARCH-in-mean models"
        )

    def resids(
        self,
        params: NDArray,
        y: Optional[ArrayLike] = None,
        regressors: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        return super().resids(params[:-1], y=y, regressors=regressors)

    def starting_values(self) -> NDArray:
        return np.r_[super().starting_values(), 0.0]

    def _resids_in_mean(
        self,
        mean_params: NDArray,
        variance_params: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
    ) -> NDArray:
        """
        TODO: Finish

        Parameters
        ----------
        mean_params
        variance_params
        sigma2
        backcast
        var_bounds

        Returns
        -------

        """
        resids = self._fit_y - self._fit_regressors @ mean_params[:-1]
        nobs = self._fit_y.shape[0]
        form_id = self._form_id
        gamma = mean_params[-1]
        # Map from natural power to power of sigma2
        form_power = self._form_power / 2.0
        for t in range(nobs):
            sigma2[t] = self.volatility.update(
                t, variance_params, resids, sigma2, backcast, var_bounds
            )
            if form_id == 0:
                trans_var = np.log(sigma2[t])
            else:
                trans_var = sigma2[t] ** form_power
            resids[t] -= gamma * trans_var

        return resids

    def _loglikelihood(
        self,
        parameters: NDArray,
        sigma2: NDArray,
        backcast: Union[float, NDArray],
        var_bounds: NDArray,
        individual: bool = False,
    ) -> Union[float, NDArray]:
        # Parse parameters
        _callback_info["count"] += 1

        # 1. Resids
        mp, vp, dp = self._parse_parameters(parameters)
        resids = self._resids_in_mean(mp, vp, sigma2, backcast, var_bounds)

        # 3. Compute log likelihood using Distribution
        llf = self.distribution.loglikelihood(dp, resids, sigma2, individual)

        if not individual:
            _callback_info["llf"] = llf_f = -float(llf)
            return llf_f

        return cast(np.ndarray, -llf)

    def _simulate_mean(
        self,
        parameters: NDArray,
        x: NDArray,
        errors: NDArray,
        initial_value: Optional[Union[float, NDArray]],
        conditional_variance: NDArray,
    ) -> NDArray:
        """

        Parameters
        ----------
        parameters
        x
        errors
        initial_value
        conditional_variance

        Returns
        -------

        """
        if self._form_id == 0:
            trans_vol = np.log(conditional_variance)
        else:
            trans_vol = conditional_variance ** (self._form_power / 2)

        max_lag = 0 if not self._lags.size else np.max(self._lags)
        nobs_and_burn = errors.shape[0]
        y = np.zeros(nobs_and_burn)
        if initial_value is None:
            initial_value = 0.0
        elif not np.isscalar(initial_value):
            initial_value = ensure1d(initial_value, "initial_value")
            if initial_value.shape[0] != max_lag:
                raise ValueError(
                    f"initial_value has the wrong shape. Expected {max_lag} values"
                )
        y[:max_lag] = initial_value
        k_x = x.shape[1]

        for t in range(max_lag, nobs_and_burn):
            ind = 0
            if self.constant:
                y[t] = parameters[ind]
                ind += 1
            for lag in self._lags.T:
                y[t] += parameters[ind] * y[t - lag[1] : t - lag[0]].mean()
                ind += 1
            for i in range(k_x):
                y[t] += parameters[ind] * x[t, i]
                ind += 1
            y[t] += parameters[ind] * trans_vol[t] + errors[t]

        return y


def arch_model(
    y: Optional[ArrayLike],
    x: Optional[ArrayLike] = None,
    mean: str = "Constant",
    lags: Optional[Union[int, List[int], NDArray]] = 0,
    vol: str = "Garch",
    p: Union[int, List[int]] = 1,
    o: int = 0,
    q: int = 1,
    power: float = 2.0,
    dist: str = "Normal",
    hold_back: Optional[int] = None,
    rescale: Optional[bool] = None,
) -> HARX:
    """
    Initialization of common ARCH model specifications

    Parameters
    ----------
    y : {ndarray, Series, None}
        The dependent variable
    x : {np.array, DataFrame}, optional
        Exogenous regressors.  Ignored if model does not permit exogenous
        regressors.
    mean : str, optional
        Name of the mean model.  Currently supported options are: 'Constant',
        'Zero', 'LS', 'AR', 'ARX', 'HAR' and  'HARX'
    lags : int or list (int), optional
        Either a scalar integer value indicating lag length or a list of
        integers specifying lag locations.
    vol : str, optional
        Name of the volatility model.  Currently supported options are:
        'GARCH' (default), 'ARCH', 'EGARCH', 'FIARCH' and 'HARCH'
    p : int, optional
        Lag order of the symmetric innovation
    o : int, optional
        Lag order of the asymmetric innovation
    q : int, optional
        Lag order of lagged volatility or equivalent
    power : float, optional
        Power to use with GARCH and related models
    dist : int, optional
        Name of the error distribution.  Currently supported options are:

            * Normal: 'normal', 'gaussian' (default)
            * Students's t: 't', 'studentst'
            * Skewed Student's t: 'skewstudent', 'skewt'
            * Generalized Error Distribution: 'ged', 'generalized error"

    hold_back : int
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    rescale : bool
        Flag indicating whether to automatically rescale data if the scale
        of the data is likely to produce convergence issues when estimating
        model parameters. If False, the model is estimated on the data without
        transformation.  If True, than y is rescaled and the new scale is
        reported in the estimation results.

    Returns
    -------
    model : ARCHModel
        Configured ARCH model

    Examples
    --------
    >>> import datetime as dt
    >>> import pandas_datareader.data as web
    >>> djia = web.get_data_fred('DJIA')
    >>> returns = 100 * djia['DJIA'].pct_change().dropna()

    A basic GARCH(1,1) with a constant mean can be constructed using only
    the return data

    >>> from arch.univariate import arch_model
    >>> am = arch_model(returns)

    Alternative mean and volatility processes can be directly specified

    >>> am = arch_model(returns, mean='AR', lags=2, vol='harch', p=[1, 5, 22])

    This example demonstrates the construction of a zero mean process
    with a TARCH volatility process and Student t error distribution

    >>> am = arch_model(returns, mean='zero', p=1, o=1, q=1,
    ...                 power=1.0, dist='StudentsT')

    Notes
    -----
    Input that are not relevant for a particular specification, such as `lags`
    when `mean='zero'`, are silently ignored.
    """
    known_mean = ("zero", "constant", "harx", "har", "ar", "arx", "ls")
    known_vol = ("arch", "figarch", "garch", "harch", "constant", "egarch")
    known_dist = (
        "normal",
        "gaussian",
        "studentst",
        "t",
        "skewstudent",
        "skewt",
        "ged",
        "generalized error",
    )
    mean = mean.lower()
    vol = vol.lower()
    dist = dist.lower()
    if mean not in known_mean:
        raise ValueError("Unknown model type in mean")
    if vol.lower() not in known_vol:
        raise ValueError("Unknown model type in vol")
    if dist.lower() not in known_dist:
        raise ValueError("Unknown model type in dist")

    if mean == "harx":
        am = HARX(y, x, lags, hold_back=hold_back, rescale=rescale)
    elif mean == "har":
        am = HARX(y, None, lags, hold_back=hold_back, rescale=rescale)
    elif mean == "arx":
        am = ARX(y, x, lags, hold_back=hold_back, rescale=rescale)
    elif mean == "ar":
        am = ARX(y, None, lags, hold_back=hold_back, rescale=rescale)
    elif mean == "ls":
        am = LS(y, x, hold_back=hold_back, rescale=rescale)
    elif mean == "constant":
        am = ConstantMean(y, hold_back=hold_back, rescale=rescale)
    else:  # mean == "zero"
        am = ZeroMean(y, hold_back=hold_back, rescale=rescale)

    if vol in ("arch", "garch", "figarch", "egarch") and not isinstance(p, int):
        raise TypeError(
            "p must be a scalar int for all volatility processes except HARCH."
        )

    if vol == "constant":
        v: VolatilityProcess = ConstantVariance()
    elif vol == "arch":
        assert isinstance(p, int)
        v = ARCH(p=p)
    elif vol == "figarch":
        assert isinstance(p, int)
        v = FIGARCH(p=p, q=q)
    elif vol == "garch":
        assert isinstance(p, int)
        v = GARCH(p=p, o=o, q=q, power=power)
    elif vol == "egarch":
        assert isinstance(p, int)
        v = EGARCH(p=p, o=o, q=q)
    else:  # vol == 'harch'
        v = HARCH(lags=p)

    if dist in ("skewstudent", "skewt"):
        d: Distribution = SkewStudent()
    elif dist in ("studentst", "t"):
        d = StudentsT()
    elif dist in ("ged", "generalized error"):
        d = GeneralizedError()
    else:  # ('gaussian', 'normal')
        d = Normal()

    am.volatility = v
    am.distribution = d

    return am
