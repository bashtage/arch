"""
Mean models to use with ARCH processes.

All mean models must inherit from :class:`ARCHModel` and provide the same methods
with the same inputs.
"""

from collections.abc import Callable, Mapping, Sequence
import copy
from typing import TYPE_CHECKING, Any, cast, overload
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from scipy.optimize import OptimizeResult
from scipy.signal import lfilter
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
    Int32Array,
    Int64Array,
    Int64Array2D,
    Label,
    NDArray,
)
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
from arch.utility._validation import (
    _describe,
    _fail,
    check_array,
    check_bool,
    check_callable,
    check_float,
    check_index,
    check_instance,
    check_int,
    check_option,
)
from arch.utility.array import append_same_type, to_array_1d

if TYPE_CHECKING:
    # Fake path to satisfy mypy
    from arch.univariate.recursions_python import ARCHInMeanRecursion, VolatilityUpdater
else:
    try:
        from arch.univariate.recursions import ARCHInMeanRecursion, VolatilityUpdater
    except ImportError:  # pragma: no cover
        from arch.univariate.recursions_python import (
            ARCHInMeanRecursion,
            VolatilityUpdater,
        )

from functools import cached_property

from arch._typing import Literal
from arch.univariate.volatility import (
    APARCH,
    ARCH,
    EGARCH,
    FIAPARCH,
    FIGARCH,
    GARCH,
    HARCH,
    ConstantVariance,
    VolatilityProcess,
)
from arch.utility.array import (
    AbstractDocStringInheritor,
    as_float64,
    parse_dataframe,
)

__all__ = [
    "ARX",
    "HARX",
    "LS",
    "ARCHInMean",
    "ConstantMean",
    "MAMean",
    "ZeroMean",
    "arch_model",
]

COV_TYPES = {
    "white": "White's Heteroskedasticity Consistent Estimator",
    "classic_ols": "Homoskedastic (Classic)",
    "robust": "Bollerslev-Wooldridge (Robust) Estimator",
    "mle": "ML Estimator",
    "classic": "ML Estimator",
}


def _forecast_pad(count: int, forecasts: Float64Array) -> Float64Array:
    shape = list(forecasts.shape)
    shape[0] = count
    fill = np.full(tuple(shape), np.nan)
    return np.concatenate((fill, forecasts))


def _ar_forecast(
    y: Float64Array,
    horizon: int,
    start_index: int,
    constant: float,
    arp: Float64Array,
    x: Float64Array,
    exogp: Float64Array,
) -> Float64Array:
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
    fcasts = cast("Float64Array2D", fcasts[:, p:])

    return fcasts


def _ar_to_impulse(steps: int, params: Float64Array) -> Float64Array:
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


def _check_simulation_params(
    model: ARCHModel, params: ArrayLike1D | Sequence[float]
) -> Float64Array1D:
    """Convert params to a 1-d array and check the number of parameters"""
    num_params = (
        int(model.num_params)
        + int(model.volatility.num_params)
        + int(model.distribution.num_params)
    )
    return cast(
        "Float64Array1D", check_array("params", params, ndim=1, shape=(num_params,))
    )


def _check_model_arguments(
    constant: Any,
    hold_back: Any,
    volatility: Any,
    distribution: Any,
    rescale: Any,
) -> tuple[bool, int | None, bool | None]:
    """
    Check the arguments that all of the mean models share

    Returns
    -------
    constant : bool
        The flag for the constant.
    hold_back : int or None
        The number of observations to hold back.
    rescale : bool or None
        The flag for rescaling.
    """
    constant = check_bool("constant", constant)
    hold_back = check_int("hold_back", hold_back, lower=0, allow_none=True)
    check_instance("volatility", volatility, VolatilityProcess, allow_none=True)
    check_instance("distribution", distribution, Distribution, allow_none=True)
    rescale = check_bool("rescale", rescale, allow_none=True)
    return constant, hold_back, rescale


def _check_lags(lags: Any, name: str = "lags", *, allow_2d: bool = True) -> None:
    """
    Check the format and the values of lags

    Parameters
    ----------
    lags : int or array_like
        The lags. A scalar is a non-negative integer, a 1-d array_like contains
        integers that are at least 1 and, if allow_2d, a (2, n) array_like
        contains the first and last lag of each entry.
    name : str
        The name of the argument for the error message.
    allow_2d : bool
        Whether the (2, n) format is allowed.
    """
    rule = "an integer >= 0 or a non-empty 1-d array_like of integers >= 1"
    if allow_2d:
        rule += (
            ", or a (2, n) array_like of integers with 1 <= lags[0, j] <= lags[1, j]"
        )
    try:
        arr = np.asarray(lags)
    except ValueError:  # Ragged sequences
        _fail(ValueError, name, rule, _describe(lags))
    if arr.ndim == 0:
        check_int(name, lags, lower=0)
        return
    if (
        arr.size == 0
        or arr.ndim > 2
        or (arr.ndim == 2 and (not allow_2d or arr.shape[0] != 2))
    ):
        _fail(ValueError, name, rule, _describe(lags))
    if arr.dtype.kind not in "iu":
        # Floats, including 5.0, Booleans and strings are rejected
        for value in arr.ravel().tolist():
            check_int(name, value)
    if arr.ndim == 1:
        check_int(name, min(arr.tolist()), lower=1)
    elif np.any(arr[0] < 1) or np.any(arr[1] < arr[0]):
        _fail(ValueError, name, rule, _describe(lags))


def _fit_sample(
    model: ARCHModel,
    first_obs: int | DateLike | None,
    last_obs: int | DateLike | None,
) -> tuple[int, int]:
    """
    Check first_obs, last_obs and the observations that are held back

    Returns
    -------
    first : int
        The position of the first observation used in estimation, which is after the
        observations that are held back.
    last : int
        The position after the last observation used in estimation.
    """
    index = model._y_series.index
    nobs = int(model._y.shape[0])
    if nobs == 0:
        check_array("y", model._y, ndim=1, min_length=1)
    raw_first = check_index("first_obs", first_obs, nobs, index=index, default=0)
    last = check_index("last_obs", last_obs, nobs, index=index, default=nobs)
    assert raw_first is not None
    assert last is not None
    if last <= raw_first:
        if last_obs is None:
            rule = f"smaller than the number of observations ({nobs})"
            _fail(ValueError, "first_obs", rule, _describe(first_obs))
        rule = f"larger than first_obs ({raw_first})"
        _fail(ValueError, "last_obs", rule, _describe(last_obs))
    hold_back = model._hold_back
    if last - raw_first <= hold_back:
        rule = (
            "smaller than the number of observations in the estimation sample "
            f"({last - raw_first}), where hold_back is at least the largest lag"
        )
        _fail(ValueError, "hold_back", rule, _describe(hold_back))
    return raw_first + hold_back, last


def _check_resid_params(params: Any, size: int) -> Float64Array1D:
    """
    Check the parameters of the mean in resids

    Notes
    -----
    resids is called with any parameters by the likelihood while the model is fit,
    and so NaN and inf are not rejected here. The check of the type and the shape
    is skipped for an array that is already valid.
    """
    if (
        type(params) is np.ndarray
        and params.dtype == np.float64
        and params.shape == (size,)
    ):
        return cast("Float64Array1D", params)
    return cast(
        "Float64Array1D",
        check_array(
            "params", params, ndim=1, shape=(size,), allow_nan=True, allow_inf=True
        ),
    )


def _check_resid_data(
    y: Any, regressors: Any, num_regressors: int | None
) -> tuple[Float64Array1D | None, Float64Array2D | None]:
    """
    Check the alternative data in resids

    Parameters
    ----------
    y : array_like or None
        The alternative values of the dependent variable.
    regressors : array_like or None
        The alternative values of the regressors.
    num_regressors : int or None
        The number of columns of regressors. None for models that do not use the
        regressors, which are only checked to be a 2-d array_like if provided.
    """
    _y = None
    nobs = None
    if y is not None:
        _y = cast("Float64Array1D", check_array("y", y, ndim=1, min_length=1))
        nobs = _y.shape[0]
    if num_regressors is None:
        _regressors = check_array("regressors", regressors, ndim=2, allow_none=True)
    elif regressors is None and y is None:
        _regressors = None
    else:
        _regressors = check_array(
            "regressors", regressors, ndim=2, shape=(nobs, num_regressors)
        )
    return _y, cast("Float64Array2D | None", _regressors)


def _check_initial_value_vol(initial_value_vol: Any) -> Any:
    """Check the initial value of the volatility process in simulate"""
    if initial_value_vol is None:
        return None
    if np.ndim(initial_value_vol) == 0:
        return check_float("initial_value_vol", initial_value_vol)
    return check_array("initial_value_vol", initial_value_vol)


def _check_simulation_no_dynamics(
    model: str,
    nobs: Any,
    burn: Any,
    initial_value: Any,
    x: Any,
    initial_value_vol: Any,
) -> tuple[int, int, Any]:
    """
    Check the arguments of simulate for a model without mean dynamics or regressors

    Parameters
    ----------
    model : str
        The description of the model for the error message.

    Returns
    -------
    nobs : int
        The number of observations.
    burn : int
        The number of burn-in observations.
    initial_value_vol : float, ndarray or None
        The initial value of the volatility process.
    """
    nobs = check_int("nobs", nobs, lower=1)
    burn = check_int("burn", burn, lower=0)
    if initial_value is not None:
        rule = f"None, since a {model} has no mean dynamics"
        _fail(ValueError, "initial_value", rule, _describe(initial_value))
    if x is not None:
        rule = f"None, since a {model} has no exogenous variables"
        _fail(ValueError, "x", rule, _describe(x))
    return nobs, burn, _check_initial_value_vol(initial_value_vol)


def _check_forecast_arguments(
    model: ARCHModel,
    params: Any,
    horizon: Any,
    start: Any,
    align: Any,
    method: Any,
    simulations: Any,
    rng: Any,
    random_state: Any,
    reindex: Any,
) -> tuple[
    Float64Array1D, int, int, Literal["origin", "target"], ForecastingMethod, int, bool
]:
    """
    Check the arguments of forecast

    Returns
    -------
    params : ndarray
        The parameters of the model as a float64 array.
    horizon : int
        The forecast horizon.
    start_index : int
        The position of the first forecast, which is the default start if start is
        None.
    align : str
        The alignment, in lower case.
    method : str
        The forecasting method, in lower case.
    simulations : int
        The number of simulations.
    reindex : bool
        Whether to reindex the forecasts.
    """
    num_params = (
        int(model.num_params)
        + int(model.volatility.num_params)
        + int(model.distribution.num_params)
    )
    params = check_array("params", params, ndim=1, shape=(num_params,))
    horizon = check_int("horizon", horizon, lower=1)
    align = cast(
        "Literal['origin', 'target']",
        check_option("align", align, ("origin", "target")),
    )
    method = cast(
        "ForecastingMethod",
        check_option("method", method, ("analytic", "simulation", "bootstrap")),
    )
    simulations = check_int("simulations", simulations, lower=1)
    check_callable("rng", rng, allow_none=True)
    check_instance("random_state", random_state, np.random.RandomState, allow_none=True)
    reindex = check_bool("reindex", reindex)
    default_start = max(0, model._fit_indices[1] - 1)
    start_index = check_index(
        "start",
        start,
        int(model._y.shape[0]),
        index=model._y_series.index,
        default=default_start,
    )
    assert start_index is not None
    return params, horizon, start_index, align, method, simulations, reindex


class HARX(ARCHModel, metaclass=AbstractDocStringInheritor):
    r"""
    Heterogeneous Autoregression (HAR) model with optional exogenous regressors.

    Provides model estimation and simulation.

    Parameters
    ----------
    y : ndarray or Series, optional
        Vector of nobs elements containing the dependent variable.
    x : ndarray or DataFrame, optional
        Array with nobs rows and k columns containing exogenous regressors.  Must
        contain only finite values.
    lags : int or array_like, optional
        Description of lag structure of the HAR.

        * Scalar included all lags between 1 and the value.
        * A 1-d n-element array includes the HAR lags 1:lags[0]+1,
          1:lags[1]+1, ... 1:lags[n]+1.
        * A 2-d (2,n)-element array that includes the HAR lags of the form
          lags[0,j]:lags[1,j]+1 for all columns of lags.

        The lags must be integers (2.5, 5.0 and True are not accepted). A scalar
        must be at least 0, and the values of an array at least 1.

    constant : bool, optional
        Flag whether the model should include a constant.
    use_rotated : bool, optional
        Flag indicating to use the alternative rotated form of the HAR where
        HAR lags do not overlap.
    hold_back : int, optional
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.  Must be a non-negative
        integer.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model.
    distribution : Distribution, optional
        Error distribution to use in the model.
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        then y is rescaled and the new scale is reported in the estimation results.
        If None, the data is not rescaled and a warning is issued when its scale is
        likely to produce convergence issues.

    Raises
    ------
    TypeError
        If ``lags`` or ``hold_back`` is not an integer (a float, including 5.0, and a
        bool are not accepted), if ``constant``, ``use_rotated`` or ``rescale`` is not
        a bool (``rescale`` can also be None), if ``volatility`` or ``distribution`` is
        not an instance of :class:`~arch.univariate.volatility.VolatilityProcess` or
        :class:`~arch.univariate.distribution.Distribution`, or if ``x`` cannot be
        converted to numbers.
    ValueError
        If ``lags`` is negative, empty or in an unsupported format, if the largest lag
        is not smaller than the number of observations, if ``hold_back`` is negative,
        or if ``x`` contains NaN or inf or does not have a row for each observation of
        ``y``.

    Notes
    -----
    The HAR-X model is described by

    .. math::

        y_t = \mu + \sum_{i=1}^p \phi_{L_{i}} \bar{y}_{t-L_{i,0}:L_{i,1}}
        + \gamma' x_t + \epsilon_t

    where :math:`\bar{y}_{t-L_{i,0}:L_{i,1}}` is the average value of
    :math:`y_t` between :math:`t-L_{i,0}` and :math:`t - L_{i,1}`.

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
    >>> index = date_range('2000-01-01', freq='ME', periods=y.shape[0])
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
    >>> print([round(mod.fit().rsquared, 3) for mod in models])
    [0.085, 0.085, 0.085, 0.085]
    """

    # To allow for addition mean parameters in simulation
    _extra_simulation_params: int = 0

    def __init__(
        self,
        y: ArrayLike | None = None,
        x: ArrayLike | ArrayLike2D | None = None,
        lags: (
            int
            | Sequence[int]
            | Sequence[Sequence[int]]
            | Int32Array
            | Int64Array
            | None
        ) = None,
        constant: bool = True,
        use_rotated: bool = False,
        hold_back: int | None = None,
        volatility: VolatilityProcess | None = None,
        distribution: Distribution | None = None,
        rescale: bool | None = None,
    ) -> None:
        constant, hold_back, rescale = _check_model_arguments(
            constant, hold_back, volatility, distribution, rescale
        )
        use_rotated = check_bool("use_rotated", use_rotated)
        if lags is not None:
            _check_lags(lags)
        super().__init__(
            y,
            hold_back=hold_back,
            volatility=volatility,
            distribution=distribution,
            rescale=rescale,
        )
        self._x: pd.DataFrame | Float64Array2D | None = None
        self._x_original = x
        self._x_names: list[str] = []
        self._x_index: NDArray | pd.Index | None = None
        self.lags: (
            int
            | Sequence[int]
            | Sequence[Sequence[int]]
            | Int32Array
            | Int64Array
            | None
        ) = lags
        self._lags: Int64Array2D = np.empty((0, 0), dtype=int)
        self.constant: bool = constant
        self.use_rotated: bool = use_rotated
        self.regressors: Float64Array2D = np.empty((0, 0), dtype=np.double)

        self._name = "HAR"
        if self._x_original is not None:
            self._name += "-X"
        if lags is not None:
            max_lags = int(np.max(np.asarray(lags, dtype=np.int64)))
        else:
            max_lags = 0
        self._max_lags = max_lags

        self._hold_back = max_lags if hold_back is None else hold_back

        if self._hold_back < max_lags:
            warnings.warn(
                "hold_back is less then the minimum number given the lags selected",
                RuntimeWarning,
                stacklevel=2,
            )
            self._hold_back = max_lags

        self._init_model()
        if self._y.shape[0] > self._hold_back:
            # Use the complete sample, subject to hold_back, as fit does by
            # default, so that the model can be used before it is fit
            self._adjust_sample(None, None)

    def _prepare_append_x(
        self, x: ArrayLike | ArrayLike2D | None, nobs_new: int
    ) -> ArrayLike | ArrayLike2D | None:
        """
        Validate new exogenous regressors and construct the extended regressors

        Parameters
        ----------
        x : {ndarray, Series, DataFrame, None}
            The new exogenous regressors.
        nobs_new : int
            The number of new observations of y.

        Returns
        -------
        {ndarray, Series, DataFrame, None}
            The original regressors extended with the new regressors. None if the
            model does not include exogenous regressors.

        Notes
        -----
        The model is not modified so that a failure leaves it unchanged.
        """
        if self._x_original is None:
            if x is not None:
                raise ValueError(
                    "x was not provided in the original model, and so x cannot be "
                    "appended."
                )
            return None
        if x is None:
            raise ValueError(
                "x must be provided when appending to a model that includes "
                "exogenous regressors."
            )
        # Finite numbers, as when constructing the model
        check_array("x", x)
        x_original = append_same_type(self._x_original, x)
        nobs_x_new = np.shape(x_original)[0] - np.shape(self._x_original)[0]
        if nobs_x_new != nobs_new:
            raise ValueError(
                "x must have the same number of observations as y. Got "
                f"{nobs_x_new} new observations of x and {nobs_new} new "
                "observations of y."
            )
        return x_original

    def append(
        self,
        y: ArrayLike | float,
        x: ArrayLike | ArrayLike2D | None = None,
    ) -> None:
        """
        Append observations to the model in-place.

        Parameters
        ----------
        y : ndarray, Series, DataFrame, list or float
            The observations to append. Must have the same type as the data
            used to construct the model. If the model was constructed using a
            pandas object, then the index of the new data must be unique and
            must not overlap the existing index. If the existing index is
            increasing, the index of the new data must also be increasing and
            must follow the existing observations. When the model was
            constructed using an ndarray or a list, a scalar can be appended
            to add a single observation.
        x : ndarray, Series or DataFrame, optional
            The values of the exogenous regressors for the new observations.
            Must be provided if, and only if, the model includes exogenous
            regressors, must have the same type and number of columns as the
            exogenous regressors used to construct the model, and must have the
            same number of observations as ``y``. When the exogenous
            regressors are stored in an ndarray, a 1-d array with one value
            for each regressor is treated as a single observation.

        Raises
        ------
        TypeError
            If the type of ``y`` or ``x`` differs from the type of the data in
            the model, or if ``x`` cannot be converted to numbers.
        ValueError
            If ``y`` is empty, contains non-finite values or, when using
            pandas, has an index that is not unique, overlaps the existing
            index, or does not follow an increasing index of the existing data.
            Also raised if ``x`` is provided when the model does not have
            exogenous regressors, is not provided when the model does,
            contains NaN or inf, or does not have the same number of
            observations as ``y``.
        RuntimeError
            If the model was created without data.

        Notes
        -----
        Parameters are not re-estimated. Use ``fit`` to re-estimate the model
        using the extended sample, or ``fix`` or ``forecast`` to use parameters
        estimated previously. Appending resets the estimation sample to the
        complete extended sample, subject to ``hold_back``, and clears values
        that were computed using the previous sample, such as the backcast.
        Results objects returned before appending are not affected.

        If the data was rescaled when fitting the model, the new data is
        rescaled using the same factor. Exogenous regressors are never
        rescaled.

        If the new data is not valid, an exception is raised and the model is
        not modified.

        Examples
        --------
        Estimate the model using the first part of a sample, and then use the
        estimated parameters to forecast after observing more data.

        >>> import numpy as np
        >>> from arch.univariate import ARX, GARCH
        >>> y = np.random.RandomState(1234).standard_normal(1000)
        >>> mod = ARX(y[:800], lags=1, volatility=GARCH())
        >>> res = mod.fit(disp="off")
        >>> mod.append(y[800:])
        >>> forecasts = mod.forecast(res.params, horizon=3, reindex=False)

        Adding a single observation to a model that uses an ndarray.

        >>> mod.append(0.1)
        """
        y_original, y_series, y_new = self._prepare_append(y)
        x_original = self._prepare_append_x(x, y_new.shape[0])
        self._commit_append(y_original, y_series, y_new)
        self._x_original = x_original
        self._init_model()
        if self._y.shape[0] > self._hold_back:
            # Match the default sample used by fit
            self._adjust_sample(None, None)

    def _scale_changed(self) -> None:
        """
        Called when the scale has changed.  This allows the model
        to update any values that are affected by the scale changes,
        e.g., any logged values.
        """
        # Reinitialize the model
        self._init_model()

    @property
    def x(self) -> ArrayLike2D | None:
        """The exogenous regressors in the model."""
        return self._x

    def parameter_names(self) -> list[str]:
        return self._generate_variable_names()

    def _model_description(self, include_lags: bool = True) -> dict[str, str]:
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
        params: Float64Array1D,
        y: ArrayLike1D | None = None,
        regressors: ArrayLike2D | None = None,
    ) -> ArrayLike1D:
        k = int(self.regressors.shape[1])
        params = _check_resid_params(params, k)
        if y is None and regressors is None:
            # The path used by the likelihood
            return self._fit_y - self._fit_regressors.dot(params)
        _y, _x = _check_resid_data(y, regressors, k)
        if _y is None:
            _y, _x = self._fit_y, self._fit_regressors
        assert _x is not None
        return _y - _x.dot(params)

    @cached_property
    def num_params(self) -> int:
        """
        Number of parameters in the model.
        """
        assert self.regressors is not None
        return int(self.regressors.shape[1])

    def _check_initial_value(
        self, initial_value: float | Float64Array | None
    ) -> float | Float64Array | None:
        """Check the initial value of the mean, a scalar or one value for each lag"""
        if initial_value is None:
            return None
        if np.ndim(initial_value) == 0:
            return check_float("initial_value", initial_value)
        max_lag = 0 if not self._lags.size else int(np.max(self._lags))
        return cast(
            "Float64Array",
            check_array("initial_value", initial_value, ndim=1, shape=(max_lag,)),
        )

    def _simulate_mean(
        self,
        parameters: Float64Array,
        x: Float64Array,
        errors: Float64Array,
        initial_value: float | Float64Array | None,
        conditional_variance: Float64Array,
    ) -> Float64Array:
        max_lag = 0 if not self._lags.size else int(np.max(self._lags))
        nobs_and_burn = errors.shape[0]
        y = np.zeros(nobs_and_burn)
        if initial_value is None:
            initial_value = 0.0
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
        params: ArrayLike1D | Sequence[float],
        nobs: int,
        burn: int = 500,
        initial_value: float | Float64Array | None = None,
        x: ArrayLike | ArrayLike2D | None = None,
        initial_value_vol: float | Float64Array | None = None,
    ) -> pd.DataFrame:
        """
        Simulate data from a linear regression, AR or HAR models.

        Parameters
        ----------
        params : array_like
            Parameters to use when simulating the model.  Parameter order is
            [mean volatility distribution] where the parameters of the mean
            model are ordered [constant lag[0] lag[1] ... lag[p] ex[0] ...
            ex[k-1]] where lag[j] indicates the coefficient on the jth lag in
            the model and ex[j] is the coefficient on the jth exogenous
            variable.
        nobs : int
            Length of series to simulate.
        burn : int, optional
            Number of values to simulate to initialize the model and remove
            dependence on initial values.
        initial_value : float or array_like, optional
            Either a scalar value or `max(lags)` array set of initial values to
            use when initializing the model.  If omitted, 0.0 is used.
        x : array_like, optional
            Array of exogenous variables to include in the simulation, with nobs +
            burn rows and k columns.
        initial_value_vol : float or ndarray, optional
            An array or scalar to use when initializing the volatility process.
            It is not always a variance. The units depend on the volatility
            process, see
            :meth:`~arch.univariate.volatility.VolatilityProcess.simulate`.

        Returns
        -------
        simulated_data : DataFrame
            DataFrame with columns data containing the simulated values,
            volatility, containing the conditional volatility and errors
            containing the errors used in the simulation.

        Raises
        ------
        TypeError
            If ``nobs`` or ``burn`` is not an integer, or if ``params``, ``x``,
            ``initial_value`` or ``initial_value_vol`` cannot be converted to
            numbers.
        ValueError
            If ``nobs`` is less than 1, ``burn`` is negative, ``params`` does not
            have one finite value for each parameter of the model, ``x`` does not
            have ``nobs + burn`` rows or is not finite, ``initial_value`` is not
            finite or does not have ``max(lags)`` values, or ``initial_value_vol``
            is not finite.

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

        nobs = check_int("nobs", nobs, lower=1)
        burn = check_int("burn", burn, lower=0)
        if x is None:
            x_arr: Float64Array2D = np.empty((nobs + burn, 0))
        else:
            # The shape in the message has nobs + burn rows
            x_arr = check_array("x", x, shape=(nobs + burn, None))
        k_x = x_arr.shape[1]
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
        params = cast(
            "Float64Array1D", check_array("params", params, ndim=1, shape=(num_params,))
        )
        initial_value = self._check_initial_value(initial_value)
        initial_value_vol = _check_initial_value_vol(initial_value_vol)

        dist_params = np.empty(0) if dc == 0 else params[-dc:]
        vol_params = cast("Float64Array1D", params[mc : mc + vc])
        simulator = self.distribution.simulate(cast("Float64Array1D", dist_params))
        sim_data = self.volatility.simulate(
            vol_params, nobs + burn, simulator, burn, initial_value_vol
        )
        errors = sim_data[0]
        vol = cast("Float64Array", np.sqrt(sim_data[1]))

        y = self._simulate_mean(params[:mc], x_arr, errors, initial_value, sim_data[1])

        df = {"data": y[burn:], "volatility": vol[burn:], "errors": errors[burn:]}
        return pd.DataFrame(df)

    def _generate_variable_names(self) -> list[str]:
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

    def _generate_lag_names(self) -> list[str]:
        """Generates lag names.  Overridden by other models"""
        lags = self._lags
        var_name = str(self._y_series.name) if self._y_series.name else ""
        if len(var_name) > 10:
            var_name = var_name[:4] + "..." + var_name[-3:]
        names = [
            var_name + "[" + str(lags[0, i]) + ":" + str(lags[1, i]) + "]"
            for i in range(lags.shape[1])
        ]
        return names

    def _check_specification(self) -> None:
        """Checks the specification for obvious errors"""
        err_msg = (
            "x must be nobs by n, where nobs is the same as the number of elements in y"
        )
        if self._x_original is not None:
            # Numbers that are finite
            check_array("x", as_float64(self._x_original, "x"))
            if isinstance(self._x_original, pd.Series):
                self._x = pd.DataFrame(self._x_original)
            elif isinstance(self._x_original, pd.DataFrame):
                self._x = self._x_original
            else:
                x_original = as_float64(self._x_original, "x")
                if x_original.ndim == 1:
                    x_original = x_original[:, None]
                self._x = cast("Float64Array2D", x_original)
            assert isinstance(self._x, (np.ndarray, pd.DataFrame))
            if self._x.ndim != 2 or self._x.shape[0] != self._y.shape[0]:
                raise ValueError(f"{err_msg}; got {_describe(self._x_original)}")
            def_names = ["x" + str(i) for i in range(self._x.shape[1])]
            names, self._x_index = parse_dataframe(self._x, def_names)
            self._x_names = [str(name) for name in names]
            self._x = cast("Float64Array2D", as_float64(self._x, "x"))

    def _reformat_lags(self) -> None:
        """
        Reformat input lags to be a 2 by m array, which simplifies other
        operations.  Output is stored in _lags
        """

        if self.lags is None:
            return
        # The format and the values of lags are checked when the model is created
        lags = np.asarray(self.lags)

        if lags.ndim == 0:
            lags = np.arange(1, int(lags) + 1)

        if lags.ndim == 1:
            lags = np.unique(lags)
            temp = cast("Int64Array2D", np.array([lags, lags], dtype=int))
            if self.use_rotated:
                temp[0, 1:] = temp[0, 0:-1]
                temp[0, 0] = 0
            else:
                temp[0, :] = 0
            self._lags = temp
        else:
            ind = np.lexsort(np.flipud(lags))
            lags = cast("Int64Array2D", lags[:, ind])
            test_mat = np.zeros((lags.shape[1], np.max(lags)), dtype=int)
            # Subtract 1 so first is 0 indexed
            lags = lags - np.array([[1], [0]])
            for i in range(lags.shape[1]):
                test_mat[i, lags[0, i] : lags[1, i]] = 1
            rank = np.linalg.matrix_rank(test_mat.astype(float))
            if rank != lags.shape[1]:
                raise ValueError("lags contains redundant entries")
            self._lags = cast("Int64Array2D", lags)

            if self.use_rotated:
                warnings.warn(
                    "Rotation is not available when using the 2-d lags input format",
                    stacklevel=2,
                )

    def _har_to_ar(self, params: Float64Array) -> Float64Array:
        if self._max_lags == 0:
            return params[: int(self.constant)]
        har = params[int(self.constant) :]
        ar: Float64Array1D
        ar = np.zeros(self._max_lags, dtype=float)
        for value, lag in zip(har, self._lags.T, strict=False):
            ar[lag[0] : lag[1]] += value / (lag[1] - lag[0])
        if self.constant:
            ar = cast("Float64Array1D", np.concatenate((params[:1], ar)))
        return ar

    def _init_model(self) -> None:
        """Should be called whenever the model is initialized or changed"""
        self._reformat_lags()
        self._check_specification()

        nobs_orig = self._y.shape[0]
        if self.constant:
            reg_constant = np.ones((nobs_orig, 1), dtype=np.double)
        else:
            reg_constant = np.ones((nobs_orig, 0), dtype=np.double)

        if self.lags is not None and nobs_orig > 0:
            maxlag = np.max(self.lags)
            if maxlag >= nobs_orig:
                _fail(
                    ValueError,
                    "lags",
                    f"smaller than the number of observations ({nobs_orig})",
                    f"a largest lag of {maxlag} ({type(self.lags).__name__})",
                )
            lag_array = lagmat(self._y, maxlag)
            reg_lags = np.empty((nobs_orig, self._lags.shape[1]), dtype=np.double)
            for i, lags in enumerate(self._lags.T):
                reg_lags[:, i] = np.mean(lag_array[:, lags[0] : lags[1]], 1)
        else:
            reg_lags = np.empty((nobs_orig, 0), dtype=np.double)

        if self._x is not None:
            reg_x = self._x
        else:
            reg_x = np.empty((nobs_orig, 0), dtype=np.double)

        # hstack preserves F-ordering if x is F-ordered, which is the case when
        # x is a DataFrame. Compiled recursions require C-ordered regressors.
        self.regressors = cast(
            "Float64Array2D",
            np.ascontiguousarray(np.hstack((reg_constant, reg_lags, reg_x))),
        )

    def _r2(self, params: ArrayLike1D, resids: Float64Array1D | None = None) -> float:
        y = self._fit_y
        constant = False
        x = self._fit_regressors
        if x is not None and x.shape[1] > 0:
            constant = self.constant or implicit_constant(x)
        if constant:
            if x.shape[1] == 1 and len(params) == 1:
                # Shortcut for constant only, which is the only mean parameter
                return 0.0
            y = y - np.mean(y)
        tss = float(y.dot(y))
        if tss <= 0.0:
            return np.nan
        e = to_array_1d(self.resids(to_array_1d(params)) if resids is None else resids)

        return 1.0 - float(e.T.dot(e)) / tss

    def _adjust_sample(
        self,
        first_obs: int | DateLike | None,
        last_obs: int | DateLike | None,
    ) -> None:
        _first_obs_index, _last_obs_index = _fit_sample(self, first_obs, last_obs)
        self._fit_indices = [_first_obs_index, _last_obs_index]
        self._fit_y = cast("Float64Array1D", self._y[_first_obs_index:_last_obs_index])
        reg = self.regressors
        self._fit_regressors = reg[_first_obs_index:_last_obs_index]
        self.volatility.start, self.volatility.stop = self._fit_indices

    def _fit_no_arch_normal_errors_params(self) -> Float64Array1D:
        """
        Estimates model parameters excluding sigma2

        Returns
        -------
        params : ndarray
            Array of estimated parameters
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
        opt = cast("Any", OptimizeResult())
        opt.status = 0
        opt.message = ""

        if x.shape[1] > 0:
            regression_params: Float64Array1D = cast(
                "Float64Array1D", np.linalg.pinv(x).dot(y)
            )
            xpxi = np.linalg.inv(x.T.dot(x) / nobs)
            fitted = x.dot(regression_params)
        else:
            regression_params = np.empty(0)
            xpxi = np.empty((0, 0))
            fitted = 0.0

        e = y - fitted
        sigma2 = e.T.dot(e) / nobs

        params = to_array_1d(np.hstack((regression_params, sigma2)))
        hessian = np.zeros((self.num_params + 1, self.num_params + 1))
        hessian[: self.num_params, : self.num_params] = -xpxi
        hessian[-1, -1] = -1
        if cov_type in ("classic",):
            param_cov = sigma2 * -hessian
            param_cov[self.num_params, self.num_params] = 2 * sigma2**2.0
            param_cov /= nobs
            cov_type_name = COV_TYPES["classic_ols"]
        elif cov_type in ("robust",):
            scores = np.zeros((nobs, self.num_params + 1))
            scores[:, : self.num_params] = x * e[:, None]
            scores[:, -1] = e**2.0 - sigma2
            score_cov = np.asarray(scores.T.dot(scores) / nobs)
            param_cov = (hessian @ score_cov @ hessian) / nobs
            cov_type_name = COV_TYPES["white"]
        else:
            raise ValueError("Unknown cov_type")

        r2 = self._r2(regression_params)

        first_obs, last_obs = self._fit_indices
        resids = np.empty(self._y.shape, dtype=float)
        resids.fill(np.nan)
        resids[first_obs:last_obs] = e
        vol = np.zeros(resids.shape, dtype=float)
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
        x: dict[Label, ArrayLike] | ArrayLike | None,
        horizon: int,
        start: int,
    ) -> Float64Array:
        """
        Always return a correctly formatted 3-d array

        Parameters
        ----------
        x: dict[label, array_like], array_like, None
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
                "x is not None but the model does not contain any exogenous variables."
            )
        assert self._x is not None
        nx = self._x.shape[1]
        if isinstance(x, Mapping):
            if len(x) != nx:
                raise ValueError(
                    "x must have the same number of entries as the number of x "
                    f"variables included in the model ({nx})"
                )
            collected: list[Float64Array] = []
            for key in self._x_names:
                if key not in x:
                    keys = ", ".join([str(k) for k in x.keys()])
                    raise KeyError(
                        "The keys of x must exactly match the variable names of "
                        f"the included exogenous regressors. {key} not found in: "
                        f"{keys}"
                    )
                temp = check_array("x", x[key], allow_nan=True)
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
                _fail(
                    ValueError,
                    "x",
                    "a dict of 1-d or 2-d array_like",
                    f"arrays with {len(base_shape)} dimensions (dict)",
                )
            arr: Float64Array = np.array(collected)
        else:
            arr = check_array("x", x, allow_nan=True)
            if arr.ndim not in (1, 2, 3):
                _fail(
                    ValueError,
                    "x",
                    "a 1-d, 2-d or 3-d array_like or a dict of array_like",
                    _describe(x),
                )

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
        start: int | DateLike | None = None,
        align: Literal["origin", "target"] = "origin",
        method: ForecastingMethod = "analytic",
        simulations: int = 1000,
        rng: Callable[[int | tuple[int, ...]], Float64Array] | None = None,
        random_state: np.random.RandomState | None = None,
        *,
        reindex: bool = False,
        x: dict[Label, ArrayLike] | ArrayLike | None = None,
    ) -> ARCHModelForecast:
        (
            params,
            horizon,
            start_index,
            align,
            method,
            simulations,
            reindex,
        ) = _check_forecast_arguments(
            self,
            params,
            horizon,
            start,
            align,
            method,
            simulations,
            rng,
            random_state,
            reindex,
        )
        earliest = self._fit_indices[0]
        if start_index < (earliest - 1):
            raise ValueError(
                "Due to backcasting and/or data availability start cannot be less "
                "than the index of the largest value in the right-hand-side "
                "variables used to fit the first observation.  In this model, "
                f"this value is {max(0, earliest - 1)}."
            )
        mp, vp, dp = self._parse_parameters(params)

        #####################################
        # Compute residual variance forecasts
        #####################################
        # Back cast should use only the sample used in fitting
        resids = self.resids(mp)
        backcast = self._volatility.backcast(resids)
        full_resids = to_array_1d(
            self.resids(
                mp,
                cast("Float64Array1D", self._y[earliest:]),
                cast("Float64Array2D", self.regressors[earliest:]),
            )
        )
        vb = self._volatility.variance_bounds(full_resids, 2.0)
        if rng is None:
            rng = self._distribution.simulate(dp)
        variance_start = max(0, start_index - earliest)
        asym_weight = self._analytic_asym_weight(method, dp)
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
            asym_weight=asym_weight,
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
        variance_paths: Float64Array | None = None
        mean_paths: Float64Array | None = None
        shocks: Float64Array | None = None
        long_run_variance_paths: Float64Array | None = None
        if method.lower() in ("simulation", "bootstrap"):
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
                lrvp = variance_paths[:, :, : (i + 1)].dot(_impulses**2)
                lrvp = lrvp[:, :, 0]
                long_run_variance_paths[:, :, i] = lrvp
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
                        mean_paths[path_loc, :, m + j] += (
                            expected_x[:, path_loc, j].T @ exog_p
                        )

            mean_paths = mean_paths[:, :, m:]

        index = self._y_series.index
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
    y : ndarray or Series, optional
        Vector of nobs elements containing the dependent variable.
    hold_back : int, optional
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.  Must be a non-negative
        integer.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model.
    distribution : Distribution, optional
        Error distribution to use in the model.
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        then y is rescaled and the new scale is reported in the estimation results.
        If None, the data is not rescaled and a warning is issued when its scale is
        likely to produce convergence issues.

    Raises
    ------
    TypeError
        If ``hold_back`` is not an integer (a float, including 5.0, and a bool are not
        accepted), if ``rescale`` is not a bool or None, or if ``volatility`` or
        ``distribution`` is not an instance of
        :class:`~arch.univariate.volatility.VolatilityProcess` or
        :class:`~arch.univariate.distribution.Distribution`.
    ValueError
        If ``hold_back`` is negative.

    Notes
    -----
    The constant mean model is described by

    .. math::

        y_t = \mu + \epsilon_t

    Examples
    --------
    >>> import numpy as np
    >>> from arch.univariate import ConstantMean
    >>> y = np.random.randn(100)
    >>> cm = ConstantMean(y)
    >>> res = cm.fit()
    """

    def __init__(
        self,
        y: ArrayLike | None = None,
        hold_back: int | None = None,
        volatility: VolatilityProcess | None = None,
        distribution: Distribution | None = None,
        rescale: bool | None = None,
    ) -> None:
        super().__init__(
            y,
            hold_back=hold_back,
            volatility=volatility,
            distribution=distribution,
            rescale=rescale,
        )
        self._name = "Constant Mean"

    def parameter_names(self) -> list[str]:
        return ["mu"]

    @cached_property
    def num_params(self) -> int:
        return 1

    def _model_description(self, include_lags: bool = False) -> dict[str, str]:
        return super()._model_description(include_lags)

    def simulate(
        self,
        params: ArrayLike1D | Sequence[float],
        nobs: int,
        burn: int = 500,
        initial_value: float | Float64Array | None = None,
        x: ArrayLike | ArrayLike2D | None = None,
        initial_value_vol: float | Float64Array | None = None,
    ) -> pd.DataFrame:
        """
        Simulate data from a constant mean model.

        Parameters
        ----------
        params : array_like
            Parameters to use when simulating the model.  Parameter order is
            [mean volatility distribution]. There is one parameter in the mean
            model, mu.
        nobs : int
            Length of series to simulate.
        burn : int, optional
            Number of values to simulate to initialize the model and remove
            dependence on initial values.
        initial_value : None, optional
            Must be None, since there are no mean dynamics. A ValueError is
            raised if it is not None.
        x : None, optional
            Must be None, since there are no exogenous variables. A ValueError
            is raised if it is not None.
        initial_value_vol : float or ndarray, optional
            An array or scalar to use when initializing the volatility process.
            It is not always a variance. The units depend on the volatility
            process, see
            :meth:`~arch.univariate.volatility.VolatilityProcess.simulate`.

        Returns
        -------
        simulated_data : DataFrame
            DataFrame with columns data containing the simulated values,
            volatility, containing the conditional volatility and errors
            containing the errors used in the simulation.

        Raises
        ------
        TypeError
            If ``nobs`` or ``burn`` is not an integer, or if ``params`` or
            ``initial_value_vol`` cannot be converted to numbers.
        ValueError
            If ``nobs`` is less than 1, ``burn`` is negative, ``params`` does not
            have one finite value for each parameter of the model, ``initial_value``
            or ``x`` is not None, since a constant mean model has no mean dynamics or
            exogenous variables, or ``initial_value_vol`` is not finite.

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
        nobs, burn, initial_value_vol = _check_simulation_no_dynamics(
            "constant mean model", nobs, burn, initial_value, x, initial_value_vol
        )

        mp, vp, dp = self._parse_parameters(_check_simulation_params(self, params))

        sim_values = self.volatility.simulate(
            vp, nobs + burn, self.distribution.simulate(dp), burn, initial_value_vol
        )
        errors = sim_values[0]
        y = errors + mp
        vol = np.sqrt(sim_values[1])
        assert isinstance(vol, np.ndarray)
        df = {"data": y[burn:], "volatility": vol[burn:], "errors": errors[burn:]}

        return pd.DataFrame(df)

    def resids(
        self,
        params: Float64Array1D,
        y: ArrayLike1D | None = None,
        regressors: ArrayLike2D | None = None,
    ) -> ArrayLike1D:
        params = _check_resid_params(params, 1)
        if y is None and regressors is None:
            return self._fit_y - params[0]
        _y, _ = _check_resid_data(y, regressors, None)
        return (self._fit_y if _y is None else _y) - params[0]


class ZeroMean(HARX):
    r"""
    Model with zero conditional mean estimation and simulation.

    Parameters
    ----------
    y : ndarray or Series, optional
        Vector of nobs elements containing the dependent variable.
    hold_back : int, optional
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.  Must be a non-negative
        integer.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model.
    distribution : Distribution, optional
        Error distribution to use in the model.
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        then y is rescaled and the new scale is reported in the estimation results.
        If None, the data is not rescaled and a warning is issued when its scale is
        likely to produce convergence issues.

    Raises
    ------
    TypeError
        If ``hold_back`` is not an integer (a float, including 5.0, and a bool are not
        accepted), if ``rescale`` is not a bool or None, or if ``volatility`` or
        ``distribution`` is not an instance of
        :class:`~arch.univariate.volatility.VolatilityProcess` or
        :class:`~arch.univariate.distribution.Distribution`.
    ValueError
        If ``hold_back`` is negative.

    Notes
    -----
    The zero mean model is described by

    .. math::

        y_t = \epsilon_t

    Examples
    --------
    >>> import numpy as np
    >>> from arch.univariate import ZeroMean
    >>> y = np.random.randn(100)
    >>> zm = ZeroMean(y)
    >>> res = zm.fit()
    """

    def __init__(
        self,
        y: ArrayLike | None = None,
        hold_back: int | None = None,
        volatility: VolatilityProcess | None = None,
        distribution: Distribution | None = None,
        rescale: bool | None = None,
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

    def parameter_names(self) -> list[str]:
        return []

    @cached_property
    def num_params(self) -> int:
        return 0

    def _model_description(self, include_lags: bool = False) -> dict[str, str]:
        return super()._model_description(include_lags)

    def simulate(
        self,
        params: ArrayLike1D | Sequence[float],
        nobs: int,
        burn: int = 500,
        initial_value: float | Float64Array | None = None,
        x: ArrayLike | ArrayLike2D | None = None,
        initial_value_vol: float | Float64Array | None = None,
    ) -> pd.DataFrame:
        """
        Simulate data from a zero mean model.

        Parameters
        ----------
        params : array_like
            Parameters to use when simulating the model.  Parameter order is
            [volatility distribution]. There are no mean parameters.
        nobs : int
            Length of series to simulate.
        burn : int, optional
            Number of values to simulate to initialize the model and remove
            dependence on initial values.
        initial_value : None, optional
            Must be None, since there are no mean dynamics. A ValueError is
            raised if it is not None.
        x : None, optional
            Must be None, since there are no exogenous variables. A ValueError
            is raised if it is not None.
        initial_value_vol : float or ndarray, optional
            An array or scalar to use when initializing the volatility process.
            It is not always a variance. The units depend on the volatility
            process, see
            :meth:`~arch.univariate.volatility.VolatilityProcess.simulate`.

        Returns
        -------
        simulated_data : DataFrame
            DataFrame with columns data containing the simulated values,
            volatility, containing the conditional volatility and errors
            containing the errors used in the simulation.

        Raises
        ------
        TypeError
            If ``nobs`` or ``burn`` is not an integer, or if ``params`` or
            ``initial_value_vol`` cannot be converted to numbers.
        ValueError
            If ``nobs`` is less than 1, ``burn`` is negative, ``params`` does not
            have one finite value for each parameter of the model, ``initial_value``
            or ``x`` is not None, since a zero mean model has no mean dynamics or
            exogenous variables, or ``initial_value_vol`` is not finite.

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
        nobs, burn, initial_value_vol = _check_simulation_no_dynamics(
            "zero mean model", nobs, burn, initial_value, x, initial_value_vol
        )
        _params = _check_simulation_params(self, params)

        _, vp, dp = self._parse_parameters(_params)

        sim_values = self.volatility.simulate(
            vp, nobs + burn, self.distribution.simulate(dp), burn, initial_value_vol
        )
        errors = sim_values[0]
        y = errors
        vol = np.sqrt(sim_values[1])
        assert isinstance(vol, np.ndarray)
        df = {"data": y[burn:], "volatility": vol[burn:], "errors": errors[burn:]}

        return pd.DataFrame(df)

    def resids(
        self,
        params: Float64Array1D,
        y: ArrayLike1D | None = None,
        regressors: ArrayLike2D | None = None,
    ) -> ArrayLike1D:
        _check_resid_params(params, 0)
        _y, _ = _check_resid_data(y, regressors, None)
        if _y is not None:
            return _y
        assert self._fit_y is not None
        return self._fit_y


class ARX(HARX):
    r"""
    Autoregressive model with optional exogenous regressors.

    Provides model estimation and simulation.

    Parameters
    ----------
    y : ndarray or Series, optional
        Vector of nobs elements containing the dependent variable.
    x : ndarray or DataFrame, optional
        Array with nobs rows and k columns containing exogenous regressors.  Must
        contain only finite values.
    lags : int or array_like, optional
        Description of lag structure of the AR.  Scalar included all lags
        between 1 and the value.  A 1-d array includes the AR lags lags[0],
        lags[1], ...  The lags must be integers (2.5, 5.0 and True are not
        accepted). A scalar must be at least 0, and the values of an array at
        least 1.
    constant : bool, optional
        Flag whether the model should include a constant.
    hold_back : int, optional
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.  Must be a non-negative
        integer.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model.
    distribution : Distribution, optional
        Error distribution to use in the model.
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        then y is rescaled and the new scale is reported in the estimation results.
        If None, the data is not rescaled and a warning is issued when its scale is
        likely to produce convergence issues.

    Raises
    ------
    TypeError
        If ``lags`` or ``hold_back`` is not an integer (a float, including 5.0, and a
        bool are not accepted), if ``constant`` or ``rescale`` is not a bool
        (``rescale`` can also be None), if ``volatility`` or ``distribution`` is not
        an instance of :class:`~arch.univariate.volatility.VolatilityProcess` or
        :class:`~arch.univariate.distribution.Distribution`, or if ``x`` cannot be
        converted to numbers.
    ValueError
        If ``lags`` is negative, empty or in an unsupported format, if the largest lag
        is not smaller than the number of observations, if ``hold_back`` is negative,
        or if ``x`` contains NaN or inf or does not have a row for each observation of
        ``y``.

    Notes
    -----
    The AR-X model is described by

    .. math::

        y_t = \mu + \sum_{i=1}^p \phi_{L_{i}} y_{t-L_{i}} + \gamma' x_t
        + \epsilon_t

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
    """

    def __init__(
        self,
        y: ArrayLike | None = None,
        x: ArrayLike | ArrayLike2D | None = None,
        lags: int | list[int] | Int32Array | Int64Array | None = None,
        constant: bool = True,
        hold_back: int | None = None,
        volatility: VolatilityProcess | None = None,
        distribution: Distribution | None = None,
        rescale: bool | None = None,
    ) -> None:
        # Convert lags to 2-d format

        if lags is not None:
            _check_lags(lags, allow_2d=False)
            lags_arr = np.asarray(lags)
            assert lags_arr is not None
            if lags_arr.ndim == 0:
                if lags_arr == 0:
                    lags = None
                else:
                    lags_arr = np.arange(1, int(lags_arr) + 1)
            if lags is not None:
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
        if self._x_original is not None:
            self._name += "-X"

    def _model_description(self, include_lags: bool = True) -> dict[str, str]:
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

    def _generate_lag_names(self) -> list[str]:
        lags = self._lags

        var_name = str(self._y_series.name) if self._y_series.name else ""
        if len(var_name) > 10:
            var_name = var_name[:4] + "..." + var_name[-3:]
        names = [var_name + "[" + str(lags[1, i]) + "]" for i in range(lags.shape[1])]
        return names


class LS(HARX):
    r"""
    Least squares model estimation and simulation.

    Parameters
    ----------
    y : ndarray or Series, optional
        Vector of nobs elements containing the dependent variable.
    x : ndarray or DataFrame, optional
        Array with nobs rows and k columns containing exogenous regressors.  Must
        contain only finite values.
    constant : bool, optional
        Flag whether the model should include a constant.
    hold_back : int, optional
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.  Must be a non-negative
        integer.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model.
    distribution : Distribution, optional
        Error distribution to use in the model.
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        then y is rescaled and the new scale is reported in the estimation results.
        If None, the data is not rescaled and a warning is issued when its scale is
        likely to produce convergence issues.

    Raises
    ------
    TypeError
        If ``hold_back`` is not an integer (a float, including 5.0, and a bool are not
        accepted), if ``constant`` or ``rescale`` is not a bool (``rescale`` can also be
        None), if ``volatility`` or ``distribution`` is not an instance of
        :class:`~arch.univariate.volatility.VolatilityProcess` or
        :class:`~arch.univariate.distribution.Distribution`, or if ``x`` cannot be
        converted to numbers.
    ValueError
        If ``hold_back`` is negative, or if ``x`` contains NaN or inf or does not have
        a row for each observation of ``y``.

    Notes
    -----
    The LS model is described by

    .. math::

        y_t = \mu + \gamma' x_t + \epsilon_t

    Examples
    --------
    >>> import numpy as np
    >>> from arch.univariate import LS
    >>> y = np.random.randn(100)
    >>> x = np.random.randn(100,2)
    >>> ls = LS(y, x)
    >>> res = ls.fit()
    """

    def __init__(
        self,
        y: ArrayLike | None = None,
        x: ArrayLike | ArrayLike2D | None = None,
        constant: bool = True,
        hold_back: int | None = None,
        volatility: VolatilityProcess | None = None,
        distribution: Distribution | None = None,
        rescale: bool | None = None,
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
    (G)ARCH-in-mean model and simulation.

    Parameters
    ----------
    y : ndarray or Series, optional
        Vector of nobs elements containing the dependent variable.
    x : ndarray or DataFrame, optional
        Array with nobs rows and k columns containing exogenous regressors.  Must
        contain only finite values.
    lags : int or array_like, optional
        Description of lag structure of the AR.  Scalar included all lags
        between 1 and the value.  A 1-d array includes the AR lags lags[0],
        lags[1], ...  The lags must be integers (2.5, 5.0 and True are not
        accepted). A scalar must be at least 0, and the values of an array at
        least 1.
    constant : bool, optional
        Flag whether the model should include a constant.
    hold_back : int, optional
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.  Must be a non-negative
        integer.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model. ``volatility.updateable`` must
        return ``True``.
    distribution : Distribution, optional
        Error distribution to use in the model.
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        then y is rescaled and the new scale is reported in the estimation results.
        If None, the data is not rescaled and a warning is issued when its scale is
        likely to produce convergence issues.
    form : {"log", "vol", "var"}, int or float, optional
        The form of the conditional variance that appears in the mean equation. The
        string names use the log of the conditional variance ("log"), the square-root
        of the conditional variance ("vol") or the conditional variance. When
        specified using a float, interpreted as :math:`\sigma_t^{form}` so that
        1 is equivalent to "vol" and 2 is equivalent to "var". When using a number,
        must be finite and different from 0.  The strings are not case sensitive.

    Raises
    ------
    TypeError
        If ``lags`` or ``hold_back`` is not an integer (a float, including 5.0, and a
        bool are not accepted), if ``constant`` or ``rescale`` is not a bool
        (``rescale`` can also be None), if ``volatility`` or ``distribution`` is not
        an instance of :class:`~arch.univariate.volatility.VolatilityProcess` or
        :class:`~arch.univariate.distribution.Distribution`, or if ``x`` cannot be
        converted to numbers, or if ``form`` is not a string or a number.
    ValueError
        If ``lags`` is negative, empty or in an unsupported format, if the largest lag
        is not smaller than the number of observations, if ``hold_back`` is negative,
        if ``form`` is a string other than "log", "vol" or "var" or is not finite or
        is 0, or if ``x`` contains NaN or inf or does not have a row for each
        observation of ``y``.

    Notes
    -----
    The (G)arch-in-mean model with exogenous regressors (-X)  is described by

    .. math::

        y_t = \mu + \kappa f(\sigma^2_t)+ \sum_{i=1}^p \phi_{L_{i}} y_{t-L_{i}}
              + \gamma' x_t +  \epsilon_t

    where :math:`f(\cdot)` is the function specified by ``form``.

    Examples
    --------
    >>> import numpy as np
    >>> from arch.univariate import ARCHInMean, GARCH
    >>> from arch.data.sp500 import load
    >>> sp500 = load()
    >>> rets = 100 * sp500["Adj Close"].pct_change().dropna()
    >>> gim = ARCHInMean(rets, lags=[1, 2], volatility=GARCH())
    >>> res = gim.fit(disp="off")
    """

    # To account for GIM param
    _extra_simulation_params = 1

    def __init__(
        self,
        y: ArrayLike | None = None,
        x: ArrayLike | ArrayLike2D | None = None,
        lags: int | list[int] | Int32Array | Int64Array | None = None,
        constant: bool = True,
        hold_back: int | None = None,
        volatility: VolatilityProcess | None = None,
        distribution: Distribution | None = None,
        rescale: bool | None = None,
        form: float | Literal["log", "vol", "var"] = "vol",
    ) -> None:
        form_rule = "'log', 'vol', 'var' or a finite number that is not 0"
        if isinstance(form, str):
            if form.lower() not in ("log", "vol", "var"):
                _fail(ValueError, "form", form_rule, _describe(form))
            form = cast("Literal['log', 'vol', 'var']", form.lower())
        elif not isinstance(form, (bool, np.bool_)) and isinstance(
            form, (int, float, np.floating, np.integer)
        ):
            if not np.isfinite(form) or form == 0:
                _fail(ValueError, "form", form_rule, _describe(form))
        else:
            _fail(TypeError, "form", form_rule, _describe(form))
        super().__init__(
            y, x, lags, constant, hold_back, volatility, distribution, rescale
        )
        self._name = "ARCH-in-mean"
        if isinstance(form, str):
            self._form_id = {"log": 0, "vol": 1, "var": 2}[form]
            self._form_power = float(self._form_id)
        else:
            self._form_id = 3
            self._form_power = float(form)
        self._form = form

        if not self.volatility.updateable:
            raise ValueError(
                f"The volatility process {self.volatility.__class__} is not "
                "updateable. Only processes that have implemented the "
                "``update`` function and ``updateable`` returns True can be "
                "used with ``ARCHInMean``."
            )
        self._volatility_updater: VolatilityUpdater = self.volatility.volatility_updater
        self._recursion = ARCHInMeanRecursion(self._volatility_updater)

    @property
    def form(self) -> int | float | Literal["log", "vol", "var"]:
        """The form of the conditional variance in the mean."""
        return self._form

    def _model_description(self, include_lags: bool = False) -> dict[str, str]:
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

    def _generate_variable_names(self) -> list[str]:
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
        start: int | DateLike | None = None,
        align: Literal["origin", "target"] = "origin",
        method: ForecastingMethod = "analytic",
        simulations: int = 1000,
        rng: Callable[[int | tuple[int, ...]], Float64Array] | None = None,
        random_state: np.random.RandomState | None = None,
        *,
        reindex: bool | None = None,
        x: dict[Label, ArrayLike] | ArrayLike | None = None,
    ) -> ARCHModelForecast:
        raise NotImplementedError(
            "forecasts are not implemented for (G)ARCH-in-mean models"
        )

    def resids(
        self,
        params: Float64Array1D,
        y: ArrayLike1D | None = None,
        regressors: ArrayLike2D | None = None,
    ) -> ArrayLike1D:
        params = _check_resid_params(params, self.num_params)
        return super().resids(
            cast("Float64Array1D", params[:-1]), y=y, regressors=regressors
        )

    def _fitted_resids_variance(
        self,
        params: Float64Array1D,
        backcast: float | Float64Array1D,
        var_bounds: Float64Array2D,
    ) -> tuple[Float64Array1D, Float64Array1D]:
        # The residuals depend on the conditional variance, so both come from
        # the recursion that is used to compute the log-likelihood
        mp, vp, _ = self._parse_parameters(params)
        nobs = self._fit_y.shape[0]
        variance = np.zeros(nobs, dtype=float)
        self._volatility_updater.initialize_update(vp, backcast, nobs)
        resids = self._recursion.recursion(
            self._fit_y,
            self._fit_regressors,
            mp,
            vp,
            variance,
            var_bounds,
            self._form_power / 2.0,
        )
        return to_array_1d(resids), variance

    def starting_values(self) -> Float64Array1D:
        return np.r_[super().starting_values(), 0.0]

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
        # Parse parameters
        _callback_info["count"] += 1

        nobs = sigma2.shape[0]
        # 1. Resids
        mp, vp, dp = self._parse_parameters(parameters)
        # 2. Initialize volatility update
        self._volatility_updater.initialize_update(vp, backcast, nobs)
        # 3. Compute recursions
        power = self._form_power / 2.0
        resids = self._recursion.recursion(
            self._fit_y,
            self._fit_regressors,
            mp,
            vp,
            sigma2,
            var_bounds,
            power,
        )

        # 3. Compute log likelihood using Distribution
        llf = self.distribution.loglikelihood(dp, resids, sigma2, individual)

        if not individual:
            _callback_info["llf"] = llf_f = -float(llf)
            return llf_f

        return cast("Float64Array1D", -llf)

    def _simulate_mean(
        self,
        parameters: Float64Array,
        x: Float64Array,
        errors: Float64Array,
        initial_value: float | Float64Array | None,
        conditional_variance: Float64Array,
    ) -> Float64Array:
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

        max_lag = 0 if not self._lags.size else int(np.max(self._lags))
        nobs_and_burn = errors.shape[0]
        y = np.zeros(nobs_and_burn)
        if initial_value is None:
            initial_value = 0.0
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


# The largest residual that is used when computing the likelihood of an MA
_MAX_MA_RESID = 1e100


def _make_invertible(
    theta: Float64Array1D, min_modulus: float = 1.01
) -> Float64Array1D:
    """
    Make the moving average polynomial invertible

    Parameters
    ----------
    theta : ndarray
        The coefficients of the moving average polynomial
        1 + theta[0] z + ... + theta[q-1] z^q.
    min_modulus : float
        The smallest modulus allowed for any root of the polynomial.

    Returns
    -------
    ndarray
        The coefficients of the polynomial where roots inside the unit circle
        are reflected to the outside, and roots closer to the unit circle than
        min_modulus are moved to min_modulus. This is theta if all roots
        are already at least min_modulus away from the origin.

    Notes
    -----
    Reflecting a root, r -> 1 / conj(r), keeps the autocovariance of the
    process unchanged up to a scale for the innovation variance.
    """
    theta = np.asarray(theta, dtype=float)
    q = theta.shape[0]
    if q == 0:
        return theta.copy()
    if not np.all(np.isfinite(theta)):
        return np.zeros(q)
    roots = np.roots(np.r_[theta[::-1], 1.0])
    if roots.shape[0] == 0:
        return theta.copy()
    modulus = np.abs(roots)
    if np.all(modulus >= min_modulus):
        return theta.copy()
    roots = np.where(modulus < 1.0, 1.0 / np.conj(roots), roots)
    modulus = np.abs(roots)
    roots = np.where(modulus < min_modulus, roots / modulus * min_modulus, roots)
    coefficients = np.poly(1.0 / roots).real
    out = np.zeros(q)
    out[: coefficients.shape[0] - 1] = coefficients[1:]
    return out


def _ma_starting_values(y: Float64Array1D, lags: int, constant: bool) -> Float64Array1D:
    """
    Starting values for a moving average using the Hannan-Rissanen method

    Parameters
    ----------
    y : ndarray
        The data
    lags : int
        The order of the moving average
    constant : bool
        Whether the mean is estimated

    Returns
    -------
    ndarray
        The sample mean, if constant, and the moving average coefficients

    Notes
    -----
    A long autoregression is estimated by OLS and its residuals are used in
    place of the unobserved innovations in an OLS regression of y on lags
    of the residuals. The coefficients are then made invertible. If the sample
    is too small to estimate the regressions, the coefficients are 0.
    """
    nobs = y.shape[0]
    mu = float(np.mean(y)) if constant else 0.0
    demeaned = y - mu
    long_ar = int(min(max(2 * lags, round(10 * np.log10(max(nobs, 1)))), nobs // 4))
    theta = np.zeros(lags)
    if long_ar >= 1 and nobs - long_ar - lags > lags:
        ar_lags = sliding_window_view(demeaned, long_ar)[:-1, ::-1]
        ar_target = demeaned[long_ar:]
        phi = np.linalg.lstsq(ar_lags, ar_target, rcond=None)[0]
        innovations = ar_target - ar_lags @ phi
        ma_lags = sliding_window_view(innovations, lags)[:-1, ::-1]
        theta = np.linalg.lstsq(ma_lags, demeaned[long_ar + lags :], rcond=None)[0]
        theta = _make_invertible(theta)
    return np.r_[mu, theta] if constant else theta


class MAMean(ARCHModel, metaclass=AbstractDocStringInheritor):
    r"""
    Moving average (MA) mean model, with optional constant.

    Provides model estimation, simulation and forecasting.

    Parameters
    ----------
    y : ndarray or Series, optional
        Vector of nobs elements containing the dependent variable.
    lags : int, optional
        The order of the moving average, q. Must be at least 1.
    constant : bool, optional
        Flag whether the model should include a constant.
    hold_back : int, optional
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Must be a non-negative integer.
    volatility : VolatilityProcess, optional
        Volatility process to use in the model.
    distribution : Distribution, optional
        Error distribution to use in the model.
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        then y is rescaled and the new scale is reported in the estimation results.
        If None, the data is not rescaled and a warning is issued when its scale is
        likely to produce convergence issues.

    Attributes
    ----------
    lags : int
        The order of the moving average
    constant : bool
        Whether the model includes a constant

    Raises
    ------
    TypeError
        If ``lags`` or ``hold_back`` is not an integer (a float, including 5.0, and a
        bool are not accepted), if ``constant`` or ``rescale`` is not a bool
        (``rescale`` can also be None), or if ``volatility`` or ``distribution`` is not
        an instance of :class:`~arch.univariate.volatility.VolatilityProcess` or
        :class:`~arch.univariate.distribution.Distribution`.
    ValueError
        If ``lags`` is smaller than 1 or ``hold_back`` is negative.

    Notes
    -----
    The MA(q) model is

    .. math::

       y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \ldots +
             \theta_q \epsilon_{t-q}

    where the innovations :math:`\epsilon_t=\sigma_t e_t` have the conditional
    variance :math:`\sigma^2_t` of the volatility process and :math:`e_t` has
    the distribution of the model. The parameters are ordered
    :math:`[\mu, \theta_1, \ldots, \theta_q]`, and :math:`\mu` is not included
    when ``constant`` is False.

    The innovations are not observed. They are computed recursively using

    .. math::

       \epsilon_t = y_t - \mu - \theta_1 \epsilon_{t-1} - \ldots -
                    \theta_q \epsilon_{t-q}

    where the innovations before the first observation in the estimation sample
    are 0. This is the conditional likelihood, and its effect vanishes when the
    process is invertible, that is when all roots of
    :math:`1+\theta_1 z+\ldots+\theta_q z^q` are outside of the unit circle.
    Starting values are invertible. Invertibility is imposed when q is 1 or 2
    (by bounds and linear constraints, respectively), but is not imposed for
    larger q since the restrictions are nonlinear. The sample used to estimate
    the parameters is also the sample used to compute the innovations, and so
    observations excluded using ``hold_back`` or when fitting are not used.

    The mean forecast of :math:`y_{T+h}` is
    :math:`\mu + \sum_{j=h}^{q}\theta_j\epsilon_{T+h-j}`, which uses the last
    :math:`q-h+1` innovations if :math:`h \leq q`, and is :math:`\mu` after
    that. The forecast variance is
    :math:`\sum_{j=0}^{h-1}\psi_j^2\sigma^2_{T+h-j}` with :math:`\psi_0=1`,
    :math:`\psi_j=\theta_j` for :math:`j\leq q` and 0 otherwise.

    Exogenous regressors are not supported.

    Examples
    --------
    Simulate an MA(2) with GARCH errors and estimate the parameters

    >>> import numpy as np
    >>> from arch.univariate import GARCH, MAMean
    >>> model = MAMean(lags=2, volatility=GARCH())
    >>> params = np.array([0.1, 0.6, 0.2, 0.05, 0.1, 0.85])
    >>> sim_data = model.simulate(params, 1000)
    >>> model = MAMean(sim_data["data"], lags=2, volatility=GARCH())
    >>> res = model.fit(disp="off")

    Forecasts from an MA(q) are not constant for the first q steps

    >>> forecasts = res.forecast(horizon=4, reindex=False)
    """

    def __init__(
        self,
        y: ArrayLike | None = None,
        lags: int = 1,
        constant: bool = True,
        hold_back: int | None = None,
        volatility: VolatilityProcess | None = None,
        distribution: Distribution | None = None,
        rescale: bool | None = None,
    ) -> None:
        lags = check_int("lags", lags, lower=1)
        constant, hold_back, rescale = _check_model_arguments(
            constant, hold_back, volatility, distribution, rescale
        )
        super().__init__(
            y,
            hold_back=hold_back,
            volatility=volatility,
            distribution=distribution,
            rescale=rescale,
        )
        self.lags: int = lags
        self.constant: bool = constant
        self._name = "MA"

    def _model_description(self, include_lags: bool = True) -> dict[str, str]:
        """Generates the model description for use by __str__ and related
        functions"""
        od = {"constant": "yes" if self.constant else "no"}
        if include_lags:
            od["lags"] = str(self.lags)
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
        return self.__str__() + ", id: " + hex(id(self))

    def _repr_html_(self) -> str:
        """HTML representation for IPython Notebook"""
        descr = self._model_description()
        html = "<strong>" + self.name + "</strong>("
        for key, val in descr.items():
            html += "<strong>" + key + ": </strong>" + val + ",\n"
        html += "<strong>ID: </strong> " + hex(id(self)) + ")"
        return html

    def parameter_names(self) -> list[str]:
        names = ["mu"] if self.constant else []
        return names + [f"theta[{i}]" for i in range(1, self.lags + 1)]

    @cached_property
    def num_params(self) -> int:
        """
        Number of parameters in the model.
        """
        return int(self.constant) + self.lags

    def bounds(self) -> list[tuple[float, float]]:
        """
        Construct bounds for parameters to use in non-linear optimization.

        Returns
        -------
        bounds : list of tuple of float
            Bounds for parameters to use in estimation. The coefficient of
            an MA(1) is restricted to [-1, 1], which makes it invertible.
        """
        bounds = super().bounds()
        if self.lags == 1:
            bounds[-1] = (-1.0, 1.0)
        return bounds

    def constraints(self) -> tuple[Float64Array, Float64Array1D]:
        """
        Construct linear constraint arrays for use in non-linear optimization.

        Returns
        -------
        a : ndarray
            Number of constraints by number of parameters loading array.
        b : ndarray
            Number of constraints array of lower bounds.

        Notes
        -----
        Parameters satisfy a.dot(parameters) - b >= 0. The constraints make an
        MA(2) invertible. There are no constraints for other orders.
        """
        if self.lags != 2:
            return super().constraints()
        a = np.zeros((3, self.num_params))
        # theta_1 + theta_2 > -1, theta_2 - theta_1 > -1, theta_2 < 1
        a[:, -2:] = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, -1.0]])
        return a, np.array([-1.0, -1.0, -1.0])

    def resids(
        self,
        params: Float64Array1D,
        y: ArrayLike1D | None = None,
        regressors: ArrayLike2D | None = None,
    ) -> ArrayLike1D:
        params = _check_resid_params(params, self.num_params)
        _y, _ = _check_resid_data(y, regressors, None)
        if _y is None:
            _y = self._fit_y
        mu = params[0] if self.constant else 0.0
        theta = params[int(self.constant) :]
        resids = lfilter([1.0], np.r_[1.0, theta], _y - mu)
        if not np.all(np.abs(resids) < _MAX_MA_RESID):
            # Parameters that are not invertible produce explosive residuals,
            # which can be reached when the numerical optimizer takes a large
            # step. Keep them finite so that the likelihood is very small
            # instead of overflowing.
            resids = np.nan_to_num(
                resids, nan=_MAX_MA_RESID, posinf=_MAX_MA_RESID, neginf=-_MAX_MA_RESID
            )
            resids = np.clip(resids, -_MAX_MA_RESID, _MAX_MA_RESID)
        return to_array_1d(resids)

    def _r2(self, params: ArrayLike1D, resids: Float64Array1D | None = None) -> float:
        y = self._fit_y
        if self.constant:
            y = y - np.mean(y)
        tss = float(y.dot(y))
        if tss <= 0.0:
            return np.nan
        e = to_array_1d(self.resids(to_array_1d(params)) if resids is None else resids)

        return 1.0 - float(e.T.dot(e)) / tss

    def _adjust_sample(
        self,
        first_obs: int | DateLike | None,
        last_obs: int | DateLike | None,
    ) -> None:
        _first_obs_index, _last_obs_index = _fit_sample(self, first_obs, last_obs)
        self._fit_indices = [_first_obs_index, _last_obs_index]
        self._fit_y = cast("Float64Array1D", self._y[_first_obs_index:_last_obs_index])
        self.volatility.start, self.volatility.stop = self._fit_indices

    def _scale_changed(self) -> None:
        # Nothing is computed from the data when the model is created
        pass

    def append(
        self,
        y: ArrayLike | float,
        x: ArrayLike | ArrayLike2D | None = None,
    ) -> None:
        super().append(y, x)
        if self._y.shape[0] > self._hold_back:
            # Match the default sample used by fit
            self._adjust_sample(None, None)

    def _fit_no_arch_normal_errors_params(self) -> Float64Array1D:
        """
        Starting values of the model parameters

        Returns
        -------
        params : ndarray
            Array of parameters estimated by the Hannan-Rissanen method

        Notes
        -----
        There is no closed form estimator of an MA model, and so these are
        used as starting values for the numerical optimizer.
        """
        nobs = self._fit_y.shape[0]
        if nobs < self.num_params:
            raise ValueError(
                "Insufficient data, "
                + str(self.num_params)
                + " parameters, "
                + str(nobs)
                + " data points available"
            )
        return _ma_starting_values(self._fit_y, self.lags, self.constant)

    def _fit_no_arch_normal_errors(
        self, cov_type: Literal["robust", "classic"] = "robust"
    ) -> ARCHModelResult:
        raise NotImplementedError(
            "MA models do not have a closed form estimator. The parameters are "
            "estimated using numerical optimization."
        )

    def simulate(
        self,
        params: ArrayLike1D | Sequence[float],
        nobs: int,
        burn: int = 500,
        initial_value: float | Float64Array | None = None,
        x: ArrayLike | ArrayLike2D | None = None,
        initial_value_vol: float | Float64Array | None = None,
    ) -> pd.DataFrame:
        """
        Simulate data from a moving average model.

        Parameters
        ----------
        params : array_like
            Parameters to use when simulating the model.  Parameter order is
            [mean volatility distribution] where the parameters of the mean
            model are ordered [mu theta[1] ... theta[q]], and mu is not
            included when the model does not have a constant.
        nobs : int
            Length of series to simulate.
        burn : int, optional
            Number of values to simulate to initialize the model and remove
            dependence on initial values.
        initial_value : None, optional
            Must be None. A ValueError is raised if it is not None. The
            innovations before the first observation are 0.
        x : None, optional
            Must be None, since there are no exogenous variables. A ValueError
            is raised if it is not None.
        initial_value_vol : float or ndarray, optional
            An array or scalar to use when initializing the volatility process.
            It is not always a variance. The units depend on the volatility
            process, see
            :meth:`~arch.univariate.volatility.VolatilityProcess.simulate`.

        Returns
        -------
        simulated_data : DataFrame
            DataFrame with columns data containing the simulated values,
            volatility, containing the conditional volatility and errors
            containing the errors used in the simulation.

        Raises
        ------
        TypeError
            If ``nobs`` or ``burn`` is not an integer, or if ``params`` or
            ``initial_value_vol`` cannot be converted to numbers.
        ValueError
            If ``nobs`` is less than 1, ``burn`` is negative, ``params`` does not
            have one finite value for each parameter of the model, ``initial_value``
            or ``x`` is not None, since a moving average model has no mean dynamics or
            exogenous variables, or ``initial_value_vol`` is not finite.

        Notes
        -----
        The innovations are the errors of the volatility process, and so the
        conditional variance of the simulated data is a function of the
        simulated innovations. With ``burn`` equal to 0, ``resids`` applied to
        the simulated data returns the errors.

        Examples
        --------
        >>> import numpy as np
        >>> from arch.univariate import GARCH, MAMean
        >>> ma = MAMean(lags=2, volatility=GARCH())
        >>> ma_params = np.array([1.0, 0.5, 0.2])
        >>> garch_params = np.array([0.01, 0.07, 0.92])
        >>> params = np.concatenate((ma_params, garch_params))
        >>> sim_data = ma.simulate(params, 1000)
        """
        nobs, burn, initial_value_vol = _check_simulation_no_dynamics(
            "moving average model", nobs, burn, initial_value, x, initial_value_vol
        )
        mp, vp, dp = self._parse_parameters(_check_simulation_params(self, params))

        sim_values = self.volatility.simulate(
            vp, nobs + burn, self.distribution.simulate(dp), burn, initial_value_vol
        )
        errors = sim_values[0]
        vol = np.sqrt(sim_values[1])
        mu = mp[0] if self.constant else 0.0
        theta = mp[int(self.constant) :]
        y = mu + lfilter(np.r_[1.0, theta], [1.0], errors)
        df = {"data": y[burn:], "volatility": vol[burn:], "errors": errors[burn:]}

        return pd.DataFrame(df)

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
    ) -> ARCHModelForecast:
        (
            params,
            horizon,
            start_index,
            align,
            method,
            simulations,
            reindex,
        ) = _check_forecast_arguments(
            self,
            params,
            horizon,
            start,
            align,
            method,
            simulations,
            rng,
            random_state,
            reindex,
        )
        if x is not None:
            raise TypeError(
                "x is not None but the model does not contain any exogenous variables."
            )
        earliest = self._fit_indices[0]
        if start_index < (earliest - 1):
            raise ValueError(
                "Due to backcasting and/or data availability start cannot be less "
                "than the index of the largest value in the right-hand-side "
                "variables used to fit the first observation.  In this model, "
                f"this value is {max(0, earliest - 1)}."
            )
        mp, vp, dp = self._parse_parameters(params)
        mu = mp[0] if self.constant else 0.0
        theta = mp[int(self.constant) :]
        q = self.lags

        #####################################
        # Compute residual variance forecasts
        #####################################
        # Back cast should use only the sample used in fitting
        resids = self.resids(mp)
        backcast = self._volatility.backcast(resids)
        # The innovations start at 0 in the first observation of the fit sample
        full_resids = to_array_1d(
            self.resids(mp, cast("Float64Array1D", self._y[earliest:]))
        )
        vb = self._volatility.variance_bounds(full_resids, 2.0)
        if rng is None:
            rng = self._distribution.simulate(dp)
        variance_start = max(0, start_index - earliest)
        asym_weight = self._analytic_asym_weight(method, dp)
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
            asym_weight=asym_weight,
        )
        var_fcasts = vfcast.forecasts
        assert var_fcasts is not None
        if start_index < earliest:
            # Pad if asking for variance forecast before earliest available
            var_fcasts = _forecast_pad(earliest - start_index, var_fcasts)

        ######################
        # Compute mean forecasts
        ######################
        # padded[s + q] is the innovation at observation s, which is 0 before
        # the first observation in the fit sample
        nobs = self._y.shape[0]
        padded = np.zeros(nobs + q)
        padded[earliest + q :] = full_resids
        norigins = nobs - start_index
        mean_fcast = np.full((norigins, horizon), mu)
        for h in range(1, min(q, horizon) + 1):
            for j in range(h, q + 1):
                first = start_index + h - j + q
                mean_fcast[:, h - 1] += theta[j - 1] * padded[first : first + norigins]

        # Compute total variance forecasts, which depend on model
        impulse = np.zeros(horizon)
        impulse[0] = 1.0
        nma = min(q, horizon - 1)
        impulse[1 : nma + 1] = theta[:nma]
        longrun_var_fcasts = var_fcasts.copy()
        for i in range(horizon):
            lrf = var_fcasts[:, : (i + 1)].dot(impulse[i::-1] ** 2)
            longrun_var_fcasts[:, i] = lrf
        variance_paths: Float64Array | None = None
        mean_paths: Float64Array | None = None
        shocks: Float64Array | None = None
        long_run_variance_paths: Float64Array | None = None
        if method.lower() in ("simulation", "bootstrap"):
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
                lrvp = variance_paths[:, :, : (i + 1)].dot(_impulses**2)
                lrvp = lrvp[:, :, 0]
                long_run_variance_paths[:, :, i] = lrvp

            # The last q innovations at each origin followed by the shocks
            nsim = shocks.shape[1]
            history = sliding_window_view(padded, q)[start_index + 1 :][:norigins]
            all_shocks = np.empty((norigins, nsim, q + horizon))
            all_shocks[:, :, :q] = history[:, None, :]
            all_shocks[:, :, q:] = shocks[:, :, :horizon]
            mean_paths = mu + all_shocks[:, :, q:]
            for j in range(1, q + 1):
                mean_paths = (
                    mean_paths
                    + theta[j - 1] * all_shocks[:, :, q - j : q - j + horizon]
                )

        index = self._y_series.index
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


def _check_order(name: str, value: Any) -> int:
    """Check the order of a volatility process, which is an integer, not a list"""
    if isinstance(value, (Sequence, np.ndarray, pd.Series)) and not isinstance(
        value, (str, bytes)
    ):
        # A sequence of lags is only used by HARCH
        rule = "an integer (a list of lags can only be used with vol='harch')"
        _fail(TypeError, name, rule, _describe(value))
    return check_int(name, value, lower=0)


def arch_model(
    y: ArrayLike | None,
    x: ArrayLike | ArrayLike2D | None = None,
    mean: Literal[
        "Constant", "Zero", "LS", "AR", "ARX", "HAR", "HARX", "constant", "zero"
    ] = "Constant",
    lags: int | list[int] | Int32Array | Int64Array | None = 0,
    vol: Literal[
        "GARCH", "ARCH", "EGARCH", "FIGARCH", "FIAPARCH", "APARCH", "HARCH"
    ] = "GARCH",
    p: int | list[int] = 1,
    o: int = 0,
    q: int = 1,
    power: float = 2.0,
    dist: Literal[
        "normal",
        "gaussian",
        "t",
        "studentst",
        "skewstudent",
        "skewt",
        "ged",
        "generalized error",
    ] = "normal",
    hold_back: int | None = None,
    rescale: bool | None = None,
) -> HARX:
    """
    Initialize common ARCH model specifications.

    Parameters
    ----------
    y : ndarray, Series or None
        The dependent variable.
    x : ndarray or DataFrame, optional
        Exogenous regressors.  Ignored if model does not permit exogenous
        regressors.
    mean : str, optional
        Name of the mean model.  Currently supported options are: 'Constant',
        'Zero', 'LS', 'AR', 'ARX', 'HAR' and  'HARX'.
    lags : int or list of int, optional
        Either a scalar integer value indicating lag length or a list of
        integers specifying lag locations.
    vol : str, optional
        Name of the volatility model.  Currently supported options are:
        'GARCH' (default), 'ARCH', 'EGARCH', 'FIGARCH', 'FIAPARCH', 'APARCH'
        and 'HARCH'.
    p : int or list of int, optional
        Lag order of the symmetric innovation.  A list of lags can only be used
        with the 'HARCH' volatility model.  Must be at least 0, and integers
        only (2.0 and True are not accepted).
    o : int, optional
        Lag order of the asymmetric innovation.  Must be at least 0.
    q : int, optional
        Lag order of lagged volatility or equivalent.  Must be at least 0.
    power : float, optional
        Power to use with GARCH and related models.  Must be finite and positive.
    dist : str, optional
        Name of the error distribution.  Currently supported options are:

            * Normal: 'normal', 'gaussian' (default)
            * Student's t: 't', 'studentst'
            * Skewed Student's t: 'skewstudent', 'skewt'
            * Generalized Error Distribution: 'ged', 'generalized error'

    hold_back : int, optional
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    rescale : bool, optional
        Flag indicating whether to automatically rescale data if the scale
        of the data is likely to produce convergence issues when estimating
        model parameters. If False, the model is estimated on the data without
        transformation.  If True, then y is rescaled and the new scale is
        reported in the estimation results.

    Returns
    -------
    model : ARCHModel
        Configured ARCH model.

    Raises
    ------
    TypeError
        If ``mean``, ``vol`` or ``dist`` is not a string, ``lags``, ``p``, ``o``,
        ``q`` or ``hold_back`` is not an integer (a float, including 5.0, and a
        bool are not accepted), ``rescale`` is not a bool or None, ``power`` is not
        a real number, or ``x`` cannot be converted to numbers.
    ValueError
        If ``mean``, ``vol`` or ``dist`` is not one of the options, ``lags``, ``p``,
        ``o``, ``q`` or ``hold_back`` is negative, ``power`` is not finite and
        positive, or ``x`` contains NaN or inf or does not have a row for each
        observation of ``y``.

    Notes
    -----
    Input that are not relevant for a particular specification, such as `lags`
    when `mean='zero'`, are silently ignored, and so are not checked.

    Examples
    --------
    >>> from arch.data.sp500 import load
    >>> sp500 = load()
    >>> returns = 100 * sp500['Adj Close'].pct_change().dropna()

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
    """
    am: ARCHModel

    known_mean = ("zero", "constant", "harx", "har", "ar", "arx", "ls")
    known_vol = (
        "arch",
        "figarch",
        "fiaparch",
        "aparch",
        "garch",
        "harch",
        "constant",
        "egarch",
    )
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
    mean_model = check_option("mean", mean, known_mean)
    vol_model = check_option("vol", vol, known_vol)
    dist_name = check_option("dist", dist, known_dist)

    if mean_model == "harx":
        am = HARX(y, x, lags, hold_back=hold_back, rescale=rescale)
    elif mean_model == "har":
        am = HARX(y, None, lags, hold_back=hold_back, rescale=rescale)
    elif mean_model == "arx":
        am = ARX(y, x, lags, hold_back=hold_back, rescale=rescale)
    elif mean_model == "ar":
        am = ARX(y, None, lags, hold_back=hold_back, rescale=rescale)
    elif mean_model == "ls":
        am = LS(y, x, hold_back=hold_back, rescale=rescale)
    elif mean_model == "constant":
        am = ConstantMean(y, hold_back=hold_back, rescale=rescale)
    else:  # mean == "zero"
        am = ZeroMean(y, hold_back=hold_back, rescale=rescale)

    if vol_model == "constant":
        v: VolatilityProcess = ConstantVariance()
    elif vol_model == "harch":
        # An integer, or a list of lags
        _check_lags(p, "p", allow_2d=False)
        v = HARCH(lags=p)
    else:
        # p, o and q must be integers, which includes NumPy integers
        p = _check_order("p", p)
        if vol_model in ("garch", "egarch", "aparch", "fiaparch"):
            o = _check_order("o", o)
        if vol_model != "arch":
            q = _check_order("q", q)
        if vol_model in ("garch", "figarch"):
            power = check_float("power", power, lower=0.0, inclusive="neither")
        if vol_model == "arch":
            v = ARCH(p=p)
        elif vol_model == "figarch":
            v = FIGARCH(p=p, q=q, power=power)
        elif vol_model == "garch":
            v = GARCH(p=p, o=o, q=q, power=power)
        elif vol_model == "egarch":
            v = EGARCH(p=p, o=o, q=q)
        elif vol_model == "fiaparch":
            v = FIAPARCH(p=p, o=o, q=q)
        else:  # vol == 'aparch'
            v = APARCH(p=p, o=o, q=q)

    if dist_name in ("skewstudent", "skewt"):
        d: Distribution = SkewStudent()
    elif dist_name in ("studentst", "t"):
        d = StudentsT()
    elif dist_name in ("ged", "generalized error"):
        d = GeneralizedError()
    else:  # ('gaussian', 'normal')
        d = Normal()

    am.volatility = v
    am.distribution = d

    return am
