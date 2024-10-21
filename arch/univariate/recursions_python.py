"""
Pure Python implementations of the core recursions in the models. Only used for
testing and if it is not possible to install the Cython version using
export ARCH_NO_BINARY=1
python -m pip install .
"""

from arch.compat.numba import jit

from abc import ABCMeta, abstractmethod
from typing import Optional, Union, cast

import numpy as np
from scipy.special import gammaln

from arch.typing import Float64Array, Int32Array
from arch.utility.array import AbstractDocStringInheritor

__all__ = [
    "bounds_check",
    "harch_recursion",
    "arch_recursion",
    "garch_recursion",
    "egarch_recursion",
    "midas_recursion",
    "figarch_weights",
    "figarch_recursion",
    "aparch_recursion",
    "GARCHUpdater",
    "FIGARCHUpdater",
    "EWMAUpdater",
    "HARCHUpdater",
    "MIDASUpdater",
    "EGARCHUpdater",
    "VolatilityUpdater",
    "ARCHInMeanRecursion",
]

LNSIGMA_MAX = float(np.log(np.finfo(np.double).max) - 0.1)
SQRT2_OV_PI = 0.79788456080286541  # E[abs(e)], e~N(0,1)


def bounds_check_python(sigma2: float, var_bounds: Float64Array) -> float:
    if sigma2 < var_bounds[0]:
        sigma2 = var_bounds[0]
    elif sigma2 > var_bounds[1]:
        if not np.isinf(sigma2):
            sigma2 = var_bounds[1] + np.log(sigma2 / var_bounds[1])
        else:
            sigma2 = var_bounds[1] + 1000
    return sigma2


bounds_check = jit(bounds_check_python, nopython=True, inline="always")


def harch_core_python(
    t: int,
    parameters: Float64Array,
    resids: Float64Array,
    sigma2: Float64Array,
    lags: Int32Array,
    backcast: float,
    var_bounds: Float64Array,
) -> float:
    sigma2[t] = parameters[0]
    for i in range(lags.shape[0]):
        param = parameters[i + 1] / lags[i]
        for j in range(lags[i]):
            if (t - j - 1) >= 0:
                sigma2[t] += param * resids[t - j - 1] * resids[t - j - 1]
            else:
                sigma2[t] += param * backcast

    sigma2[t] = bounds_check(sigma2[t], var_bounds[t])
    return sigma2[t]


harch_core = jit(harch_core_python, nopython=True, inline="always")


def harch_recursion_python(
    parameters: Float64Array,
    resids: Float64Array,
    sigma2: Float64Array,
    lags: Int32Array,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array:
    """
    Parameters
    ----------
    parameters : ndarray
        Model parameters
    resids : ndarray
        Residuals to use in the recursion
    sigma2 : ndarray
        Conditional variances with same shape as resids
    lags : ndarray
        Lag lengths in the HARCH
    nobs : int
        Length of resids
    backcast : float
        Value to use when initializing the recursion
    var_bounds : ndarray
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period
    """

    for t in range(nobs):
        sigma2[t] = parameters[0]
        for i in range(lags.shape[0]):
            param = parameters[i + 1] / lags[i]
            for j in range(lags[i]):
                if (t - j - 1) >= 0:
                    sigma2[t] += param * resids[t - j - 1] * resids[t - j - 1]
                else:
                    sigma2[t] += param * backcast

        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

    return sigma2


harch_recursion = jit(harch_recursion_python, nopython=True)


def arch_recursion_python(
    parameters: Float64Array,
    resids: Float64Array,
    sigma2: Float64Array,
    p: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array:
    """
    Parameters
    ----------
    parameters : ndarray
        Model parameters
    resids : ndarray
        Residuals to use in the recursion
    sigma2 : ndarray
        Conditional variances with same shape as resids
    p : int
        Number of lags in ARCH model
    nobs : int
        Length of resids
    backcast : float
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period
    """

    for t in range(nobs):
        sigma2[t] = parameters[0]
        for i in range(p):
            if (t - i - 1) < 0:
                sigma2[t] += parameters[i + 1] * backcast
            else:
                sigma2[t] += parameters[i + 1] * resids[t - i - 1] ** 2
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

    return sigma2


arch_recursion = jit(arch_recursion_python, nopython=True)


def garch_core_python(
    t: int,
    parameters: Float64Array,
    resids: Float64Array,
    sigma2: Float64Array,
    backcast: float,
    var_bounds: Float64Array,
    p: int,
    o: int,
    q: int,
    power: float,
) -> float:
    """
    Compute variance recursion for GARCH and related models

    Parameters
    ----------
    t : int
        The time period to update
    parameters : ndarray
        Model parameters
    resids : ndarray
        Residuals
    sigma2 : ndarray
        Conditional variances with same shape as resids
    backcast : float
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        transformed variances for each time period
    p : int
        Number of symmetric innovations in model
    o : int
        Number of asymmetric innovations in model
    q : int
        Number of lags of the (transformed) variance in the model
    power : float
        The power used in the model
    """

    loc = 0
    sigma2[t] = parameters[loc]
    loc += 1
    for j in range(p):
        if (t - 1 - j) < 0:
            sigma2[t] += parameters[loc] * backcast
        else:
            sigma2[t] += parameters[loc] * (np.abs(resids[t - 1 - j]) ** power)
        loc += 1
    for j in range(o):
        if (t - 1 - j) < 0:
            sigma2[t] += parameters[loc] * 0.5 * backcast
        else:
            sigma2[t] += (
                parameters[loc]
                * (np.abs(resids[t - 1 - j]) ** power)
                * (resids[t - 1 - j] < 0)
            )
        loc += 1
    for j in range(q):
        if (t - 1 - j) < 0:
            sigma2[t] += parameters[loc] * backcast
        else:
            sigma2[t] += parameters[loc] * sigma2[t - 1 - j]
        loc += 1
    sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

    return sigma2[t]


garch_core = jit(garch_core_python, nopython=True, inline="always")


def garch_recursion_python(
    parameters: Float64Array,
    fresids: Float64Array,
    sresids: Float64Array,
    sigma2: Float64Array,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array:
    """
    Compute variance recursion for GARCH and related models

    Parameters
    ----------
    parameters : ndarray
        Model parameters
    fresids : ndarray
        Absolute value of residuals raised to the power in the model.  For
        example, in a standard GARCH model, the power is 2.0.
    sresids : ndarray
        Variable containing the sign of the residuals (-1.0, 0.0, 1.0)
    sigma2 : ndarray
        Conditional variances with same shape as resids
    p : int
        Number of symmetric innovations in model
    o : int
        Number of asymmetric innovations in model
    q : int
        Number of lags of the (transformed) variance in the model
    nobs : int
        Length of resids
    backcast : float
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        transformed variances for each time period
    """

    for t in range(nobs):
        loc = 0
        sigma2[t] = parameters[loc]
        loc += 1
        for j in range(p):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * backcast
            else:
                sigma2[t] += parameters[loc] * fresids[t - 1 - j]
            loc += 1
        for j in range(o):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * 0.5 * backcast
            else:
                sigma2[t] += (
                    parameters[loc] * fresids[t - 1 - j] * (sresids[t - 1 - j] < 0)
                )
            loc += 1
        for j in range(q):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * backcast
            else:
                sigma2[t] += parameters[loc] * sigma2[t - 1 - j]
            loc += 1
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

    return sigma2


garch_recursion = jit(garch_recursion_python, nopython=True)


def egarch_recursion_python(
    parameters: Float64Array,
    resids: Float64Array,
    sigma2: Float64Array,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array,
    lnsigma2: Float64Array,
    std_resids: Float64Array,
    abs_std_resids: Float64Array,
) -> Float64Array:
    """
    Compute variance recursion for EGARCH models

    Parameters
    ----------
    parameters : ndarray
        Model parameters
    resids : ndarray
        Residuals to use in the recursion
    sigma2 : ndarray
        Conditional variances with same shape as resids
    p : int
        Number of symmetric innovations in model
    o : int
        Number of asymmetric innovations in model
    q : int
        Number of lags of the (transformed) variance in the model
    nobs : int
        Length of resids
    backcast : float
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period
    lnsigma2 : ndarray
        Temporary array (overwritten) with same shape as resids
    std_resids : ndarray
        Temporary array (overwritten) with same shape as resids
    abs_std_resids : ndarray
        Temporary array (overwritten) with same shape as resids
    """

    for t in range(nobs):
        loc = 0
        lnsigma2[t] = parameters[loc]
        loc += 1
        for j in range(p):
            if (t - 1 - j) >= 0:
                lnsigma2[t] += parameters[loc] * (
                    abs_std_resids[t - 1 - j] - SQRT2_OV_PI
                )
            loc += 1
        for j in range(o):
            if (t - 1 - j) >= 0:
                lnsigma2[t] += parameters[loc] * std_resids[t - 1 - j]
            loc += 1
        for j in range(q):
            if (t - 1 - j) < 0:
                lnsigma2[t] += parameters[loc] * backcast
            else:
                lnsigma2[t] += parameters[loc] * lnsigma2[t - 1 - j]
            loc += 1
        if lnsigma2[t] > LNSIGMA_MAX:
            lnsigma2[t] = LNSIGMA_MAX
        sigma2[t] = np.exp(lnsigma2[t])
        if sigma2[t] < var_bounds[t, 0]:
            sigma2[t] = var_bounds[t, 0]
            lnsigma2[t] = np.log(sigma2[t])
        elif sigma2[t] > var_bounds[t, 1]:
            sigma2[t] = var_bounds[t, 1] + np.log(sigma2[t]) - np.log(var_bounds[t, 1])
            lnsigma2[t] = np.log(sigma2[t])
        std_resids[t] = resids[t] / np.sqrt(sigma2[t])
        abs_std_resids[t] = np.abs(std_resids[t])

    return sigma2


egarch_recursion = jit(egarch_recursion_python, nopython=True)


def midas_recursion_python(
    parameters: Float64Array,
    weights: Float64Array,
    resids: Float64Array,
    sigma2: Float64Array,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array:
    """
    Parameters
    ----------
    parameters : ndarray
        Model parameters of the form (omega, alpha, gamma) where omega is the
        intercept, alpha is the scale for all shocks and gamma is the shock
        to negative returns (can be 0.0) for a symmetric model.
    weights : ndarray
        The weights on the lagged squared returns. Should sum to 1
    resids : ndarray
        Residuals to use in the recursion
    sigma2 : ndarray
        Conditional variances with same shape as resids
    nobs : int
        Length of resids
    backcast : float
        Value to use when initializing the recursion
    var_bounds : ndarray
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period
    """
    omega, alpha, gamma = parameters

    m = weights.shape[0]
    aw = np.zeros(m)
    gw = np.zeros(m)
    for i in range(m):
        aw[i] = alpha * weights[i]
        gw[i] = gamma * weights[i]

    resids2 = np.zeros(nobs)

    for t in range(nobs):
        resids2[t] = resids[t] * resids[t]
        sigma2[t] = omega
        for i in range(m):
            if (t - i - 1) >= 0:
                sigma2[t] += (aw[i] + gw[i] * (resids[t - i - 1] < 0)) * resids2[
                    t - i - 1
                ]
            else:
                sigma2[t] += (aw[i] + 0.5 * gw[i]) * backcast
        if sigma2[t] < var_bounds[t, 0]:
            sigma2[t] = var_bounds[t, 0]
        elif sigma2[t] > var_bounds[t, 1]:
            if np.isinf(sigma2[t]):
                sigma2[t] = var_bounds[t, 1] + 1000
            else:
                sigma2[t] = var_bounds[t, 1] + np.log(sigma2[t] / var_bounds[t, 1])

    return sigma2


midas_recursion = jit(midas_recursion_python, nopython=True)


def figarch_weights_python(
    parameters: Float64Array, p: int, q: int, trunc_lag: int
) -> Float64Array:
    r"""
    Parameters
    ----------
    parameters : ndarray
        Model parameters of the form (phi, d, beta) where omega is the
        intercept, d is the fractional integration coefficient and phi and beta
        are parameters of the volatility process.
    p : int
        0 or 1 to indicate whether the model contains phi
    q : int
        0 or 1 to indicate whether the model contains beta
    trunc_lag : int
        Truncation lag for the ARCH approximations

    Returns
    -------
    lam : ndarray
        ARCH(:math:`\infty`) coefficients used to approximate model dynamics
    """
    phi = parameters[0] if p else 0.0
    d = parameters[1] if p else parameters[0]
    beta = parameters[p + q] if q else 0.0

    # Recursive weight computation
    lam = np.empty(trunc_lag)
    delta = np.empty(trunc_lag)
    lam[0] = phi - beta + d
    delta[0] = d
    for i in range(1, trunc_lag):
        delta[i] = (i - d) / (i + 1) * delta[i - 1]
        lam[i] = beta * lam[i - 1] + (delta[i] - phi * delta[i - 1])

    return lam


figarch_weights = jit(figarch_weights_python, nopython=True)


def figarch_recursion_python(
    parameters: Float64Array,
    fresids: Float64Array,
    sigma2: Float64Array,
    p: int,
    q: int,
    nobs: int,
    trunc_lag: int,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array:
    """
    Parameters
    ----------
    parameters : ndarray
        Model parameters of the form (omega, phi, d, beta) where omega is the
        intercept, d is the fractional integration coefficient and phi and beta
        are parameters of the volatility process.
    fresids : ndarray
        Absolute value of residuals raised to the power in the model.  For
        example, in a standard GARCH model, the power is 2.0.
    sigma2 : ndarray
        Conditional variances with same shape as resids
    p : int
        0 or 1 to indicate whether the model contains phi
    q : int
        0 or 1 to indicate whether the model contains beta
    nobs : int
        Length of resids
    trunc_lag : int
        Truncation lag for the ARCH approximations
    backcast : float
        Value to use when initializing the recursion
    var_bounds : ndarray
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period

    Returns
    -------
    sigma2 : ndarray
        Conditional variances
    """

    omega = parameters[0]
    beta = parameters[1 + p + q] if q else 0.0
    omega_tilde = omega / (1 - beta)
    lam = figarch_weights(parameters[1:], p, q, trunc_lag)
    for t in range(nobs):
        bc_weight = 0.0
        for i in range(t, trunc_lag):
            bc_weight += lam[i]
        sigma2[t] = omega_tilde + bc_weight * backcast
        for i in range(min(t, trunc_lag)):
            sigma2[t] += lam[i] * fresids[t - i - 1]
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

    return sigma2


figarch_recursion = jit(figarch_recursion_python, nopython=True)


def aparch_recursion_python(
    parameters: Float64Array,
    resids: Float64Array,
    abs_resids: Float64Array,
    sigma2: Float64Array,
    sigma_delta: Float64Array,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array:
    """
    Compute variance recursion for GARCH and related models

    Parameters
    ----------
    parameters : ndarray
        Model parameters
    resids : ndarray
        Residuals.
    aresids : ndarray
        Absolute value of residuals.
    sigma2 : ndarray
        Conditional variances with same shape as resids
    sigma_delta : ndarray
        Conditional variance to the power delta with same shape as resids
    p : int
        Number of symmetric innovations in model
    o : int
        Number of asymmetric innovations in model
    q : int
        Number of lags of the (transformed) variance in the model
    nobs : int
        Length of resids
    backcast : float
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        transformed variances for each time period
    """
    delta = parameters[1 + p + o + q]
    for t in range(nobs):
        sigma_delta[t] = parameters[0]
        for j in range(p):
            if (t - 1 - j) < 0:
                shock = backcast**0.5
            else:
                shock = abs_resids[t - 1 - j]
                if o > j:
                    shock -= parameters[1 + p + j] * resids[t - 1 - j]
            sigma_delta[t] += parameters[1 + j] * (shock**delta)
        for j in range(q):
            if (t - 1 - j) < 0:
                sigma_delta[t] += parameters[1 + p + o + j] * backcast ** (delta / 2.0)
            else:
                sigma_delta[t] += parameters[1 + p + o + j] * sigma_delta[t - 1 - j]
        sigma2[t] = sigma_delta[t] ** (2.0 / delta)
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])
        sigma_delta[t] = sigma2[t] ** (delta / 2.0)
    return sigma2


aparch_recursion = jit(aparch_recursion_python, nopython=True)


class VolatilityUpdater(metaclass=ABCMeta):
    """
    Base class that all volatility updaters must inherit from.

    Notes
    -----
    See the implementation available for information on modifying ``__init__``
    to capture model-specific parameters and how ``initialize_update`` is
    used to precompute values that change in each likelihood but not
    each iteration of the recursion.

    When writing a volatility updater, it is recommended to follow the
    examples in recursions.pyx which use Cython to produce a C-callable
    update function that can then be used to improve performance. The
    subclasses of this abstract metaclass are all pure Python and
    model estimation performance is poor since loops are written
    in Python.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def initialize_update(
        self, parameters: Float64Array, backcast: Union[float, Float64Array], nobs: int
    ) -> None:
        """
        Initialize the recursion prior to calling update

        Parameters
        ----------
        parameters : ndarray
            The model parameters.
        backcast : {float, ndarray}
            The backcast value(s).
        nobs : int
            The number of observations in the sample.

        Notes
        -----
        This function is called once per likelihood evaluation and can be used
        to pre-compute expensive parameter transformations that do not change
        with each call to ``update``.
        """
        pass

    @abstractmethod
    def update(
        self,
        t: int,
        parameters: Float64Array,
        resids: Float64Array,
        sigma2: Float64Array,
        var_bounds: Float64Array,
    ) -> None:
        """
        Update the current variance at location t

        Parameters
        ----------
        t : int
            The index of the value of sigma2 to update. Assumes but does not check
            that update has been called recursively for 0,1,...,t-1.
        parameters : ndarray
            Model parameters
        resids : ndarray
            Residuals to use in the recursion
        sigma2 : ndarray
            Conditional variances with same shape as resids
        var_bounds : ndarray
            nobs by 2-element array of upper and lower bounds for conditional
            variances for each time period

        Notes
        -----
        The update to sigma2 occurs inplace.
        """
        pass

    def _update_tester(
        self,
        t: int,
        parameters: Float64Array,
        resids: Float64Array,
        sigma2: Float64Array,
        var_bounds: Float64Array,
    ) -> None:
        """
        Testing shim for update with compatibility with the Cythonized version

        Parameters
        ----------
        t : int
            The index of the value of sigma2 to update. Assumes but does not check
            that update has been called recursively for 0,1,...,t-1.
        parameters : ndarray
            Model parameters
        resids : ndarray
            Residuals to use in the recursion
        sigma2 : ndarray
            Conditional variances with same shape as resids
        var_bounds : ndarray
            nobs by 2-element array of upper and lower bounds for conditional
            variances for each time period
        """
        self.update(t, parameters, resids, sigma2, var_bounds)


class GARCHUpdater(VolatilityUpdater, metaclass=AbstractDocStringInheritor):
    def __init__(self, p: int, o: int, q: int, power: float) -> None:
        super().__init__()
        self.p = p
        self.o = o
        self.q = q
        self.power = power
        self.backcast = -1.0

    def initialize_update(
        self, parameters: Float64Array, backcast: Union[float, Float64Array], nobs: int
    ) -> None:
        self.backcast = cast(float, backcast)

    def update(
        self,
        t: int,
        parameters: Float64Array,
        resids: Float64Array,
        sigma2: Float64Array,
        var_bounds: Float64Array,
    ) -> None:
        loc = 0
        sigma2[t] = parameters[loc]
        loc += 1
        for j in range(self.p):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * self.backcast
            else:
                sigma2[t] += parameters[loc] * (np.abs(resids[t - 1 - j]) ** self.power)
            loc += 1
        for j in range(self.o):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * 0.5 * self.backcast
            else:
                sigma2[t] += (
                    parameters[loc]
                    * (np.abs(resids[t - 1 - j]) ** self.power)
                    * (resids[t - 1 - j] < 0)
                )
            loc += 1
        for j in range(self.q):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * self.backcast
            else:
                sigma2[t] += parameters[loc] * sigma2[t - 1 - j]
            loc += 1
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])


class HARCHUpdater(VolatilityUpdater, metaclass=AbstractDocStringInheritor):
    def __init__(self, lags: Int32Array) -> None:
        super().__init__()
        self.lags = lags
        self.backcast = -1.0

    def initialize_update(
        self, parameters: Float64Array, backcast: Union[float, Float64Array], nobs: int
    ) -> None:
        self.backcast = cast(float, backcast)

    def update(
        self,
        t: int,
        parameters: Float64Array,
        resids: Float64Array,
        sigma2: Float64Array,
        var_bounds: Float64Array,
    ) -> None:
        backcast = self.backcast

        sigma2[t] = parameters[0]
        for i in range(self.lags.shape[0]):
            param = parameters[i + 1] / self.lags[i]
            for j in range(self.lags[i]):
                if (t - j - 1) >= 0:
                    sigma2[t] += param * resids[t - j - 1] * resids[t - j - 1]
                else:
                    sigma2[t] += param * backcast

        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])


class EWMAUpdater(VolatilityUpdater, metaclass=AbstractDocStringInheritor):
    def __init__(self, lam: Optional[float]) -> None:
        super().__init__()
        self.estimate_lam = lam is None
        self.params = np.zeros(3)
        if lam is not None:
            self.params[1] = 1.0 - lam
            self.params[2] = lam

    def initialize_update(
        self, parameters: Float64Array, backcast: Union[float, Float64Array], nobs: int
    ) -> None:
        if self.estimate_lam:
            self.params[1] = 1.0 - parameters[0]
            self.params[2] = parameters[0]
        self.backcast = backcast

    def update(
        self,
        t: int,
        parameters: Float64Array,
        resids: Float64Array,
        sigma2: Float64Array,
        var_bounds: Float64Array,
    ) -> None:
        sigma2[t] = self.params[0]
        if t == 0:
            sigma2[t] += self.backcast
        else:
            sigma2[t] += (
                self.params[1] * resids[t - 1] * resids[t - 1]
                + self.params[2] * sigma2[t - 1]
            )
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])


class MIDASUpdater(VolatilityUpdater, metaclass=AbstractDocStringInheritor):
    def __init__(self, m: int, asym: bool) -> None:
        super().__init__()
        self.m = m
        self.asym = asym
        self.aw = np.empty(m)
        self.gw = np.empty(m)
        self.weights = np.empty(m)
        self.resids2 = np.empty(0)
        self.DOUBLE_EPS = float(np.finfo(np.double).eps)

    def update_weights(self, theta: float) -> None:
        sum_w = 0.0
        m = self.m
        # Prevent 0
        theta = theta if theta > self.DOUBLE_EPS else self.DOUBLE_EPS
        j = 1.0
        for i in range(m):
            self.weights[i] = np.exp(
                gammaln(theta + j) - gammaln(j + 1) - gammaln(theta)
            )
            j += 1.0
        for i in range(m):
            sum_w += self.weights[i]
        for i in range(m):
            self.weights[i] /= sum_w

    def initialize_update(
        self, parameters: Float64Array, backcast: Union[float, Float64Array], nobs: int
    ) -> None:
        self.update_weights(parameters[2 + self.asym])
        alpha = parameters[1]
        if self.asym:
            gamma = parameters[2]
        else:
            gamma = 0.0

        for i in range(self.m):
            self.aw[i] = alpha * self.weights[i]
            self.gw[i] = gamma * self.weights[i]
        self.backcast = backcast
        if self.resids2.shape[0] < nobs:
            self.resids2 = np.empty(nobs)

    def update(
        self,
        t: int,
        parameters: Float64Array,
        resids: Float64Array,
        sigma2: Float64Array,
        var_bounds: Float64Array,
    ) -> None:
        omega = parameters[0]
        if t > 0:
            self.resids2[t - 1] = resids[t - 1] * resids[t - 1]

        sigma2[t] = omega
        for i in range(self.m):
            if (t - i - 1) >= 0:
                sigma2[t] += (
                    self.aw[i] + self.gw[i] * (resids[t - i - 1] < 0)
                ) * self.resids2[t - i - 1]
            else:
                sigma2[t] += (self.aw[i] + 0.5 * self.gw[i]) * self.backcast

        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])


class FIGARCHUpdater(VolatilityUpdater, metaclass=AbstractDocStringInheritor):
    def __init__(self, p: int, q: int, power: float, truncation: int) -> None:
        super().__init__()
        self.p = p
        self.q = q
        self.truncation = truncation
        self.power = power
        self.lam = np.empty(0)
        self.fresids = np.empty(0)

    def initialize_update(
        self, parameters: Float64Array, backcast: Union[float, Float64Array], nobs: int
    ) -> None:
        self.lam = figarch_weights(parameters[1:], self.p, self.q, self.truncation)
        self.backcast = backcast
        if self.fresids.shape[0] < nobs:
            self.fresids = np.empty(nobs)

    def update(
        self,
        t: int,
        parameters: Float64Array,
        resids: Float64Array,
        sigma2: Float64Array,
        var_bounds: Float64Array,
    ) -> None:
        p = self.p
        q = self.q
        trunc_lag = self.truncation

        omega = parameters[0]
        beta = parameters[1 + p + q] if q else 0.0
        omega_tilde = omega / (1 - beta)

        if t > 0:
            self.fresids[t - 1] = np.abs(resids[t - 1]) ** self.power

        bc_weight = 0.0
        for i in range(t, trunc_lag):
            bc_weight += self.lam[i]
        sigma2[t] = omega_tilde + bc_weight * self.backcast
        for i in range(min(t, trunc_lag)):
            sigma2[t] += self.lam[i] * self.fresids[t - i - 1]
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])


class RiskMetrics2006Updater(VolatilityUpdater, metaclass=AbstractDocStringInheritor):
    def __init__(
        self,
        kmax: int,
        combination_weights: Float64Array,
        smoothing_parameters: Float64Array,
    ) -> None:
        super().__init__()
        self.kmax = kmax
        self.combination_weights = combination_weights
        self.smoothing_parameters = smoothing_parameters
        self.backcast = np.empty(kmax)
        self.last_sigma2s = np.empty((1, kmax))

    def initialize_update(
        self, parameters: Float64Array, backcast: Union[float, Float64Array], nobs: int
    ) -> None:
        self.backcast = cast(Float64Array, backcast)

    def update(
        self,
        t: int,
        parameters: Float64Array,
        resids: Float64Array,
        sigma2: Float64Array,
        var_bounds: Float64Array,
    ) -> None:
        w = self.combination_weights
        mus = self.smoothing_parameters
        if t == 0:
            self.last_sigma2s = self.backcast
        else:
            self.last_sigma2s = (1 - mus) * resids[t - 1] ** 2 + mus * self.last_sigma2s
        sigma2[t] = self.last_sigma2s @ w
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])


class EGARCHUpdater(VolatilityUpdater, metaclass=AbstractDocStringInheritor):
    def __init__(self, p: int, o: int, q: int) -> None:
        super().__init__()
        self.p = p
        self.o = o
        self.q = q
        self.backcast = 9999.99
        self.lnsigma2 = np.empty(0)
        self.std_resids = np.empty(0)
        self.abs_std_resids = np.empty(0)

    def _resize(self, nobs: int) -> None:
        if self.lnsigma2.shape[0] < nobs:
            self.lnsigma2 = np.empty(nobs)
            self.abs_std_resids = np.empty(nobs)
            self.std_resids = np.empty(nobs)

    def initialize_update(
        self, parameters: Float64Array, backcast: Union[float, Float64Array], nobs: int
    ) -> None:
        self.backcast = cast(float, backcast)
        self._resize(nobs)

    def update(
        self,
        t: int,
        parameters: Float64Array,
        resids: Float64Array,
        sigma2: Float64Array,
        var_bounds: Float64Array,
    ) -> None:
        if t > 0:
            self.std_resids[t - 1] = resids[t - 1] / np.sqrt(sigma2[t - 1])
            self.abs_std_resids[t - 1] = abs(self.std_resids[t - 1])

        self.lnsigma2[t] = parameters[0]
        loc = 1
        for j in range(self.p):
            if (t - 1 - j) >= 0:
                self.lnsigma2[t] += parameters[loc] * (
                    self.abs_std_resids[t - 1 - j] - SQRT2_OV_PI
                )
            loc += 1
        for j in range(self.o):
            if (t - 1 - j) >= 0:
                self.lnsigma2[t] += parameters[loc] * self.std_resids[t - 1 - j]
            loc += 1
        for j in range(self.q):
            if (t - 1 - j) < 0:
                self.lnsigma2[t] += parameters[loc] * self.backcast
            else:
                self.lnsigma2[t] += parameters[loc] * self.lnsigma2[t - 1 - j]
            loc += 1
        if self.lnsigma2[t] > LNSIGMA_MAX:
            self.lnsigma2[t] = LNSIGMA_MAX
        sigma2[t] = np.exp(self.lnsigma2[t])
        if sigma2[t] < var_bounds[t, 0]:
            sigma2[t] = var_bounds[t, 0]
            self.lnsigma2[t] = np.log(sigma2[t])
        elif sigma2[t] > var_bounds[t, 1]:
            sigma2[t] = var_bounds[t, 1] + np.log(sigma2[t]) - np.log(var_bounds[t, 1])
            self.lnsigma2[t] = np.log(sigma2[t])


class ARCHInMeanRecursion:
    def __init__(self, updater: VolatilityUpdater):
        if not isinstance(updater, VolatilityUpdater):
            raise TypeError("updater must be a VolatilityUpdater")
        self.volatility_updater = updater

    def recursion(
        self,
        y: Float64Array,
        x: Float64Array,
        mean_parameters: Float64Array,
        variance_params: Float64Array,
        sigma2: Float64Array,
        var_bounds: Float64Array,
        power: float,
    ) -> Float64Array:
        nobs = y.shape[0]
        k = x.shape[1]
        resids = np.empty(nobs)
        gamma = mean_parameters[k]

        for t in range(nobs):
            self.volatility_updater.update(
                t, variance_params, resids, sigma2, var_bounds
            )
            resids[t] = y[t]
            for i in range(k):
                resids[t] -= x[t, i] * mean_parameters[i]
            if power == 0.0:
                resids[t] -= gamma * np.log(sigma2[t])
            else:
                resids[t] -= gamma * sigma2[t] ** power

        return resids
