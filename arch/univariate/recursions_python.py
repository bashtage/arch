"""
Pure Python implementations of the core recursions in the models. Only used for
testing and if it is not possible to install the Cython version using
export ARCH_NO_BINARY=1
python -m pip install .
"""

from arch.compat.numba import jit

from abc import ABCMeta, abstractmethod
from typing import cast

import numpy as np
from scipy.special import gammaln

from arch._typing import Float64Array, Float64Array1D, Float64Array2D, Int32Array
from arch.utility._validation import _describe, check_int
from arch.utility.array import AbstractDocStringInheritor

__all__ = [
    "ARCHInMeanRecursion",
    "EGARCHUpdater",
    "EWMAUpdater",
    "FIGARCHUpdater",
    "GARCHUpdater",
    "HARCHUpdater",
    "MIDASUpdater",
    "NGARCHUpdater",
    "VolatilityUpdater",
    "aparch_recursion",
    "arch_recursion",
    "bounds_check",
    "egarch_recursion",
    "figarch_recursion",
    "figarch_weights",
    "garch_recursion",
    "harch_recursion",
    "midas_recursion",
    "ngarch_recursion",
]

LNSIGMA_MAX = float(np.log(np.finfo(np.double).max) - 0.1)
SQRT2_OV_PI = 0.79788456080286541  # E[abs(e)], e~N(0,1)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _check_vector(name: str, arr: object, min_length: int) -> None:
    """
    Check that an argument is not None and has at least min_length elements.

    This is the check of the compiled recursions, which read and write the arrays
    without bounds checks. It is O(1) and is made once per likelihood evaluation,
    never in the per-observation update. The messages are the same as theirs, and
    are only built when a check fails.
    """
    shape = None if arr is None else np.shape(arr)
    if shape is None or len(shape) != 1 or shape[0] < min_length:
        rule = "a 1-d C contiguous float64 ndarray"
        if min_length > 0:
            rule += f" with at least {_plural(min_length, 'element')}"
        exc = TypeError if arr is None else ValueError
        raise exc(f"{name} must be {rule}; got {_describe(arr)}")


def _check_matrix(name: str, arr: object, min_rows: int, min_cols: int) -> None:
    """Check that an argument is not None and has at least min_rows by min_cols."""
    shape = None if arr is None else np.shape(arr)
    if shape is None or len(shape) != 2 or shape[0] < min_rows or shape[1] < min_cols:
        rule = (
            "a 2-d C contiguous float64 ndarray with at least "
            f"{_plural(min_rows, 'row')} and {_plural(min_cols, 'column')}"
        )
        exc = TypeError if arr is None else ValueError
        raise exc(f"{name} must be {rule}; got {_describe(arr)}")


def bounds_check_python(sigma2: float, var_bounds: Float64Array2D) -> float:
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
    parameters: Float64Array1D,
    resids: Float64Array1D,
    sigma2: Float64Array1D,
    lags: Int32Array,
    backcast: float,
    var_bounds: Float64Array2D,
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


def harch_step_python(
    t: int,
    parameters: Float64Array1D,
    resids: Float64Array1D,
    lags: Int32Array,
    backcast: float,
    var_bounds: Float64Array1D,
) -> float:
    """One observation of the HARCH recursion, which can use the backcast"""
    sigma2 = parameters[0]
    for i in range(lags.shape[0]):
        param = parameters[i + 1] / lags[i]
        for j in range(lags[i]):
            if (t - j - 1) >= 0:
                sigma2 += param * resids[t - j - 1] * resids[t - j - 1]
            else:
                sigma2 += param * backcast

    return bounds_check(sigma2, var_bounds)


harch_step = jit(harch_step_python, nopython=True)


def harch_recursion_python(
    parameters: Float64Array1D,
    resids: Float64Array1D,
    sigma2: Float64Array1D,
    lags: Int32Array,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
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

    num_lags = lags.shape[0]
    max_lag = 0
    for i in range(num_lags):
        max_lag = max(max_lag, lags[i])

    # The first max_lag observations use the backcast
    t = 0
    while t < min(max_lag, nobs):
        sigma2[t] = harch_step(t, parameters, resids, lags, backcast, var_bounds[t])
        t += 1

    # Later observations have a full history. The sums for different observations
    # are independent, so four are accumulated at a time so that the floating point
    # adds of different sums overlap, rather than each add waiting for the previous
    # one. Every sum still adds its terms one at a time in the original order.
    while t + 3 < nobs:
        a0 = parameters[0]
        a1 = parameters[0]
        a2 = parameters[0]
        a3 = parameters[0]
        for i in range(num_lags):
            param = parameters[i + 1] / lags[i]
            for j in range(lags[i]):
                a0 += param * resids[t - j - 1] * resids[t - j - 1]
                a1 += param * resids[t - j] * resids[t - j]
                a2 += param * resids[t - j + 1] * resids[t - j + 1]
                a3 += param * resids[t - j + 2] * resids[t - j + 2]
        sigma2[t] = bounds_check(a0, var_bounds[t])
        sigma2[t + 1] = bounds_check(a1, var_bounds[t + 1])
        sigma2[t + 2] = bounds_check(a2, var_bounds[t + 2])
        sigma2[t + 3] = bounds_check(a3, var_bounds[t + 3])
        t += 4
    while t < nobs:
        sigma2[t] = harch_step(t, parameters, resids, lags, backcast, var_bounds[t])
        t += 1

    return sigma2


harch_recursion = jit(harch_recursion_python, nopython=True)


def arch_recursion_python(
    parameters: Float64Array1D,
    resids: Float64Array1D,
    sigma2: Float64Array1D,
    p: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
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
    parameters: Float64Array1D,
    resids: Float64Array1D,
    sigma2: Float64Array1D,
    backcast: float,
    var_bounds: Float64Array2D,
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
    parameters: Float64Array1D,
    fresids: Float64Array1D,
    sresids: Float64Array1D,
    sigma2: Float64Array1D,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
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
    parameters: Float64Array1D,
    resids: Float64Array1D,
    sigma2: Float64Array1D,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
    lnsigma2: Float64Array1D,
    std_resids: Float64Array1D,
    abs_std_resids: Float64Array1D,
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
        lnsigma2[t] = min(lnsigma2[t], LNSIGMA_MAX)
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


def ngarch_recursion_python(
    parameters: Float64Array1D,
    resids: Float64Array1D,
    sigma2: Float64Array1D,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
) -> Float64Array:
    """
    Compute variance recursion for NGARCH (NAGARCH) models

    Parameters
    ----------
    parameters : ndarray
        Model parameters, ordered omega, alpha (p), theta (o) and beta (q)
    resids : ndarray
        Residuals to use in the recursion
    sigma2 : ndarray
        Conditional variances with same shape as resids
    p : int
        Number of symmetric innovations in model
    o : int
        Number of asymmetric shifts in model, which cannot exceed p. The first
        o innovations are shifted by theta times the lagged volatility.
    q : int
        Number of lags of the conditional variance in the model
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
        for j in range(p):
            theta = parameters[1 + p + j] if j < o else 0.0
            if (t - 1 - j) < 0:
                # E[(z - theta) ** 2] = 1 + theta ** 2 when sigma2 = backcast
                sigma2[t] += parameters[1 + j] * (1.0 + theta * theta) * backcast
            else:
                shock = resids[t - 1 - j] - theta * np.sqrt(sigma2[t - 1 - j])
                sigma2[t] += parameters[1 + j] * shock * shock
        for j in range(q):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[1 + p + o + j] * backcast
            else:
                sigma2[t] += parameters[1 + p + o + j] * sigma2[t - 1 - j]
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

    return sigma2


ngarch_recursion = jit(ngarch_recursion_python, nopython=True)


def midas_step_python(
    t: int,
    omega: float,
    aw: Float64Array1D,
    gw: Float64Array1D,
    resids: Float64Array1D,
    resids2: Float64Array1D,
    backcast: float,
    var_bounds: Float64Array1D,
) -> float:
    """One observation of the MIDAS recursion, which can use the backcast"""
    sigma2 = omega
    for i in range(aw.shape[0]):
        if (t - i - 1) >= 0:
            sigma2 += (aw[i] + gw[i] * (resids[t - i - 1] < 0)) * resids2[t - i - 1]
        else:
            sigma2 += (aw[i] + 0.5 * gw[i]) * backcast

    return bounds_check(sigma2, var_bounds)


midas_step = jit(midas_step_python, nopython=True)


def midas_recursion_python(
    parameters: Float64Array1D,
    weights: Float64Array,
    resids: Float64Array1D,
    sigma2: Float64Array1D,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
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

    # The first m observations use the backcast
    t = 0
    while t < min(m, nobs):
        sigma2[t] = midas_step(
            t, omega, aw, gw, resids, resids2, backcast, var_bounds[t]
        )
        t += 1

    # Later observations have a full history. The sums for different observations
    # are independent, so four are accumulated at a time so that the floating point
    # adds of different sums overlap, rather than each add waiting for the previous
    # one. Every sum still adds its terms one at a time in the original order.
    while t + 3 < nobs:
        a0 = omega
        a1 = omega
        a2 = omega
        a3 = omega
        for i in range(m):
            a0 += (aw[i] + gw[i] * (resids[t - i - 1] < 0)) * resids2[t - i - 1]
            a1 += (aw[i] + gw[i] * (resids[t - i] < 0)) * resids2[t - i]
            a2 += (aw[i] + gw[i] * (resids[t - i + 1] < 0)) * resids2[t - i + 1]
            a3 += (aw[i] + gw[i] * (resids[t - i + 2] < 0)) * resids2[t - i + 2]
        sigma2[t] = bounds_check(a0, var_bounds[t])
        sigma2[t + 1] = bounds_check(a1, var_bounds[t + 1])
        sigma2[t + 2] = bounds_check(a2, var_bounds[t + 2])
        sigma2[t + 3] = bounds_check(a3, var_bounds[t + 3])
        t += 4
    while t < nobs:
        sigma2[t] = midas_step(
            t, omega, aw, gw, resids, resids2, backcast, var_bounds[t]
        )
        t += 1

    return sigma2


midas_recursion = jit(midas_recursion_python, nopython=True)


def figarch_weights_python(
    parameters: Float64Array1D, p: int, q: int, trunc_lag: int
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


def backcast_weights_python(
    lam: Float64Array1D, nobs: int, trunc_lag: int
) -> Float64Array1D:
    """
    Weights on the backcast in the ARCH(inf) representation

    Element t is lam[t] + ... + lam[trunc_lag - 1], the weight on the lags that
    are before the start of the sample, which is 0 for t >= trunc_lag. Each sum
    adds the terms in increasing order of the lag.
    """
    # The sums for different t are independent. Four are accumulated at a time so
    # that the floating point adds of different sums overlap, rather than each
    # add waiting for the previous one. Every sum still adds its terms one at a
    # time in the same order as a single loop would, so the results do not change.
    n_bc = min(nobs, trunc_lag)
    bc_weights = np.zeros(nobs)
    t = 0
    while t + 3 < n_bc:
        a0 = 0.0
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        a0 += lam[t]
        a0 += lam[t + 1]
        a1 += lam[t + 1]
        a0 += lam[t + 2]
        a1 += lam[t + 2]
        a2 += lam[t + 2]
        for i in range(t + 3, trunc_lag):
            w = lam[i]
            a0 += w
            a1 += w
            a2 += w
            a3 += w
        bc_weights[t] = a0
        bc_weights[t + 1] = a1
        bc_weights[t + 2] = a2
        bc_weights[t + 3] = a3
        t += 4
    while t < n_bc:
        a0 = 0.0
        for i in range(t, trunc_lag):
            a0 += lam[i]
        bc_weights[t] = a0
        t += 1
    return bc_weights


backcast_weights = jit(backcast_weights_python, nopython=True)


def arch_inf_sums_python(
    lam: Float64Array1D,
    shocks: Float64Array1D,
    bc_weights: Float64Array1D,
    omega_tilde: float,
    backcast: float,
    nobs: int,
    trunc_lag: int,
    sums: Float64Array1D,
) -> None:
    """
    Truncated ARCH(inf) sums of FIGARCH-type models

    Sets sums[t] to

        omega_tilde + bc_weights[t] * backcast + sum_i lam[i] * shocks[t - i - 1]

    where the sum is over i < min(t, trunc_lag) and is accumulated in increasing
    order of i, after the first two terms.
    """
    # As in backcast_weights, four sums are accumulated at a time
    t = 0
    while t + 3 < nobs:
        a0 = omega_tilde + bc_weights[t] * backcast
        a1 = omega_tilde + bc_weights[t + 1] * backcast
        a2 = omega_tilde + bc_weights[t + 2] * backcast
        a3 = omega_tilde + bc_weights[t + 3] * backcast
        # Lags available to all four, t, t + 1, t + 2 and t + 3
        m = min(t, trunc_lag)
        for i in range(m):
            w = lam[i]
            a0 += w * shocks[t - i - 1]
            a1 += w * shocks[t - i]
            a2 += w * shocks[t - i + 1]
            a3 += w * shocks[t - i + 2]
        # Additional lags only available to the later observations
        for i in range(m, min(t + 1, trunc_lag)):
            a1 += lam[i] * shocks[t - i]
        for i in range(m, min(t + 2, trunc_lag)):
            a2 += lam[i] * shocks[t - i + 1]
        for i in range(m, min(t + 3, trunc_lag)):
            a3 += lam[i] * shocks[t - i + 2]
        sums[t] = a0
        sums[t + 1] = a1
        sums[t + 2] = a2
        sums[t + 3] = a3
        t += 4
    while t < nobs:
        a0 = omega_tilde + bc_weights[t] * backcast
        for i in range(min(t, trunc_lag)):
            a0 += lam[i] * shocks[t - i - 1]
        sums[t] = a0
        t += 1


arch_inf_sums = jit(arch_inf_sums_python, nopython=True)


def figarch_recursion_python(
    parameters: Float64Array1D,
    fresids: Float64Array1D,
    sigma2: Float64Array1D,
    p: int,
    q: int,
    nobs: int,
    trunc_lag: int,
    backcast: float,
    var_bounds: Float64Array2D,
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
    bc_weights = backcast_weights(lam, nobs, trunc_lag)
    arch_inf_sums(
        lam, fresids, bc_weights, omega_tilde, backcast, nobs, trunc_lag, sigma2
    )
    for t in range(nobs):
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

    return sigma2


figarch_recursion = jit(figarch_recursion_python, nopython=True)


def fiaparch_recursion_python(
    parameters: Float64Array1D,
    resids: Float64Array1D,
    abs_resids: Float64Array1D,
    sigma2: Float64Array1D,
    sigma_delta: Float64Array1D,
    p: int,
    q: int,
    nobs: int,
    trunc_lag: int,
    backcast: float,
    var_bounds: Float64Array2D,
    gamma: float,
    delta: float,
) -> Float64Array:
    """
    Parameters
    ----------
    parameters : ndarray
        Model parameters of the form (omega, phi, d, beta) where omega is the
        intercept, d is the fractional integration coefficient and phi and beta
        are parameters of the volatility process.
    resids : ndarray
        Residuals to use in the recursion.
    abs_resids : ndarray
        Absolute value of residuals.
    sigma2 : ndarray
        Conditional variances with same shape as resids
    sigma_delta : ndarray
        Work array with same shape as resids. On return it contains the
        conditional variance to the power delta before the variance bounds
        are applied
    p : int
        0 or 1 to indicate whether the model contains phi
    q : int
        0 or 1 to indicate whether the model contains beta
    nobs : int
        Length of resids
    trunc_lag : int
        Truncation lag for the ARCH approximations
    backcast : float
        Value to use when initializing the recursion. This is a variance and is
        converted to the sigma^delta scale using delta
    var_bounds : ndarray
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period
    gamma : float
        Asymmetry parameter
    delta : float
        Power parameter

    Returns
    -------
    sigma2 : ndarray
        Conditional variances
    """
    omega = parameters[0]
    beta = parameters[1 + p + q] if q else 0.0
    omega_tilde = omega / (1 - beta)
    lam = figarch_weights(parameters[1:], p, q, trunc_lag)
    backcast_delta = backcast ** (delta / 2.0)
    # The transformed shock depends only on the observation, not on the lag, so
    # raise it to delta once (nobs powers) instead of once per (t, lag) pair
    fshocks = np.empty(nobs)
    for t in range(nobs - 1):
        fshocks[t] = (abs_resids[t] - gamma * resids[t]) ** delta
    bc_weights = backcast_weights(lam, nobs, trunc_lag)
    arch_inf_sums(
        lam,
        fshocks,
        bc_weights,
        omega_tilde,
        backcast_delta,
        nobs,
        trunc_lag,
        sigma_delta,
    )
    # sigma_delta is a work array that holds sigma ** delta before the variance
    # bounds are applied. It is not read by any caller, so it is not recomputed from
    # the bounded variance, which would cost a call to pow for each observation.
    for t in range(nobs):
        sigma2[t] = sigma_delta[t] ** (2.0 / delta)
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

    return sigma2


fiaparch_recursion = jit(fiaparch_recursion_python, nopython=True)


def aparch_recursion_python(
    parameters: Float64Array1D,
    resids: Float64Array1D,
    abs_resids: Float64Array1D,
    sigma2: Float64Array1D,
    sigma_delta: Float64Array,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
) -> Float64Array:
    """
    Compute variance recursion for GARCH and related models

    Parameters
    ----------
    parameters : ndarray
        Model parameters
    resids : ndarray
        Residuals.
    abs_resids : ndarray
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

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def initialize_update(
        self,
        parameters: Float64Array1D,
        backcast: float | Float64Array1D,
        nobs: int,
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

    @abstractmethod
    def update(
        self,
        t: int,
        parameters: Float64Array1D,
        resids: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
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

    def _update_tester(
        self,
        t: int,
        parameters: Float64Array1D,
        resids: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
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
        self.p = p
        self.o = o
        self.q = q
        self.power = power
        self.backcast = -1.0
        self.sigma_power = np.empty(0)

    def initialize_update(
        self,
        parameters: Float64Array1D,
        backcast: float | Float64Array1D,
        nobs: int,
    ) -> None:
        _check_vector("parameters", parameters, 1 + self.p + self.o + self.q)
        self.backcast = cast("float", backcast)
        if self.sigma_power.shape[0] < nobs:
            self.sigma_power = np.empty(nobs)

    def update(
        self,
        t: int,
        parameters: Float64Array1D,
        resids: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
    ) -> None:
        # The recursion runs in sigma ** power, which is what the backcast and
        # var_bounds use. sigma2 is the variance, sigma ** 2
        loc = 0
        sigma_power = parameters[loc]
        loc += 1
        for j in range(self.p):
            if (t - 1 - j) < 0:
                sigma_power += parameters[loc] * self.backcast
            else:
                sigma_power += parameters[loc] * (
                    np.abs(resids[t - 1 - j]) ** self.power
                )
            loc += 1
        for j in range(self.o):
            if (t - 1 - j) < 0:
                sigma_power += parameters[loc] * 0.5 * self.backcast
            else:
                sigma_power += (
                    parameters[loc]
                    * (np.abs(resids[t - 1 - j]) ** self.power)
                    * (resids[t - 1 - j] < 0)
                )
            loc += 1
        for j in range(self.q):
            if (t - 1 - j) < 0:
                sigma_power += parameters[loc] * self.backcast
            else:
                sigma_power += parameters[loc] * self.sigma_power[t - 1 - j]
            loc += 1
        sigma_power = bounds_check(sigma_power, var_bounds[t])
        self.sigma_power[t] = sigma_power
        sigma2[t] = sigma_power ** (2.0 / self.power)


class HARCHUpdater(VolatilityUpdater, metaclass=AbstractDocStringInheritor):
    def __init__(self, lags: Int32Array) -> None:
        if lags is None:
            raise TypeError(
                f"lags must be a 1-d C contiguous int32 ndarray; got {_describe(lags)}"
            )
        self.lags = lags
        self.backcast = -1.0

    def initialize_update(
        self,
        parameters: Float64Array1D,
        backcast: float | Float64Array1D,
        nobs: int,
    ) -> None:
        _check_vector("parameters", parameters, 1 + len(self.lags))
        self.backcast = cast("float", backcast)

    def update(
        self,
        t: int,
        parameters: Float64Array1D,
        resids: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
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
    def __init__(self, lam: float | None) -> None:
        self.estimate_lam = lam is None
        self.params = np.zeros(3)
        if lam is not None:
            self.params[1] = 1.0 - lam
            self.params[2] = lam

    def initialize_update(
        self,
        parameters: Float64Array1D,
        backcast: float | Float64Array1D,
        nobs: int,
    ) -> None:
        if self.estimate_lam:
            _check_vector("parameters", parameters, 1)
            self.params[1] = 1.0 - parameters[0]
            self.params[2] = parameters[0]
        self.backcast = backcast

    def update(
        self,
        t: int,
        parameters: Float64Array1D,
        resids: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
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
        self,
        parameters: Float64Array1D,
        backcast: float | Float64Array1D,
        nobs: int,
    ) -> None:
        # omega, alpha, gamma (if asym) and theta
        _check_vector("parameters", parameters, 3 + int(self.asym))
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
        parameters: Float64Array1D,
        resids: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
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
        check_int("truncation", truncation, lower=1)
        self.p = p
        self.q = q
        self.truncation = truncation
        self.power = power
        self.lam = np.empty(0)
        self.fresids = np.empty(0)

    def initialize_update(
        self,
        parameters: Float64Array1D,
        backcast: float | Float64Array1D,
        nobs: int,
    ) -> None:
        # omega, phi (if p), d and beta (if q), which the weights and update read
        _check_vector("parameters", parameters, 2 + self.p + self.q)
        self.lam = figarch_weights(parameters[1:], self.p, self.q, self.truncation)
        self.backcast = backcast
        if self.fresids.shape[0] < nobs:
            self.fresids = np.empty(nobs)

    def update(
        self,
        t: int,
        parameters: Float64Array1D,
        resids: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
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
        sigma_power = omega_tilde + bc_weight * self.backcast
        for i in range(min(t, trunc_lag)):
            sigma_power += self.lam[i] * self.fresids[t - i - 1]
        # The recursion runs in sigma ** power, which is what the backcast and
        # var_bounds use. sigma2 is the variance, sigma ** 2
        sigma_power = bounds_check(sigma_power, var_bounds[t])
        sigma2[t] = sigma_power ** (2.0 / self.power)


class FIAPARCHUpdater(VolatilityUpdater, metaclass=AbstractDocStringInheritor):
    def __init__(
        self,
        p: int,
        q: int,
        o: int,
        truncation: int,
        est_delta: bool,
        delta: float,
    ) -> None:
        check_int("truncation", truncation, lower=1)
        self.p = p
        self.q = q
        self.o = o
        self.truncation = truncation
        self.est_delta = est_delta
        self.delta = delta
        self.backcast: float | Float64Array1D = -1.0
        self.lam = np.empty(truncation)
        self.fshocks = np.empty(0)

    def __setstate__(self, state: tuple) -> None:
        self.backcast = state[0]
        self.delta = state[1]
        lam = np.asarray(state[2], dtype=np.float64)
        assert lam.shape[0] == self.truncation
        self.lam = lam.copy()
        self.fshocks = np.asarray(state[3], dtype=np.float64).copy()

    def __reduce__(self) -> tuple:
        return (
            FIAPARCHUpdater,
            (
                self.p,
                self.q,
                self.o,
                self.truncation,
                bool(self.est_delta),
                float(self.delta),
            ),
            (
                self.backcast,
                self.delta,
                np.asarray(self.lam),
                np.asarray(self.fshocks),
            ),
        )

    def initialize_update(
        self,
        parameters: Float64Array1D,
        backcast: float | Float64Array1D,
        nobs: int,
    ) -> None:
        # omega, phi (if p), d, beta (if q), gamma (if o) and delta (if estimated)
        _check_vector(
            "parameters", parameters, 2 + self.p + self.q + self.o + int(self.est_delta)
        )
        self.lam = figarch_weights(parameters[1:], self.p, self.q, self.truncation)
        # The backcast is a variance, so convert it to sigma ** delta. It is
        # stored on this scale
        delta = parameters[-1] if self.est_delta else self.delta
        self.backcast = cast("float", backcast) ** (delta / 2.0)
        if self.fshocks.shape[0] < nobs:
            self.fshocks = np.empty(nobs)

    def update(
        self,
        t: int,
        parameters: Float64Array1D,
        resids: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
    ) -> None:
        p = self.p
        q = self.q
        o = self.o
        trunc_lag = self.truncation
        gamma = parameters[2 + p + q] if o else 0.0
        delta = parameters[-1] if self.est_delta else self.delta

        omega = parameters[0]
        beta = parameters[1 + p + q] if q else 0.0
        omega_tilde = omega / (1 - beta)

        # The transformed shock depends only on the observation, so it is raised to
        # delta once, when it becomes available, rather than once for each lag
        if t > 0:
            self.fshocks[t - 1] = (
                np.abs(resids[t - 1]) - gamma * resids[t - 1]
            ) ** delta

        bc_weight = 0.0
        for i in range(t, trunc_lag):
            bc_weight += self.lam[i]
        sigma_delta = omega_tilde + bc_weight * self.backcast
        for i in range(min(t, trunc_lag)):
            sigma_delta += self.lam[i] * self.fshocks[t - i - 1]
        sigma2[t] = sigma_delta ** (2.0 / delta)
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])


class RiskMetrics2006Updater(VolatilityUpdater, metaclass=AbstractDocStringInheritor):
    def __init__(
        self,
        kmax: int,
        combination_weights: Float64Array,
        smoothing_parameters: Float64Array1D,
    ) -> None:
        self.kmax = kmax
        self.combination_weights = combination_weights
        self.smoothing_parameters = smoothing_parameters
        self.backcast: Float64Array1D = np.empty(kmax)
        self.last_sigma2s: Float64Array = np.empty((1, kmax))

    def initialize_update(
        self,
        parameters: Float64Array1D,
        backcast: float | Float64Array1D,
        nobs: int,
    ) -> None:
        self.backcast = cast("Float64Array1D", backcast)

    def update(
        self,
        t: int,
        parameters: Float64Array1D,
        resids: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
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
        self,
        parameters: Float64Array1D,
        backcast: float | Float64Array1D,
        nobs: int,
    ) -> None:
        _check_vector("parameters", parameters, 1 + self.p + self.o + self.q)
        self.backcast = cast("float", backcast)
        self._resize(nobs)

    def update(
        self,
        t: int,
        parameters: Float64Array1D,
        resids: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
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
        self.lnsigma2[t] = min(self.lnsigma2[t], LNSIGMA_MAX)
        sigma2[t] = np.exp(self.lnsigma2[t])
        if sigma2[t] < var_bounds[t, 0]:
            sigma2[t] = var_bounds[t, 0]
            self.lnsigma2[t] = np.log(sigma2[t])
        elif sigma2[t] > var_bounds[t, 1]:
            sigma2[t] = var_bounds[t, 1] + np.log(sigma2[t]) - np.log(var_bounds[t, 1])
            self.lnsigma2[t] = np.log(sigma2[t])


class NGARCHUpdater(VolatilityUpdater, metaclass=AbstractDocStringInheritor):
    def __init__(self, p: int, o: int, q: int) -> None:
        self.p = p
        self.o = o
        self.q = q
        self.backcast = 9999.99

    def initialize_update(
        self,
        parameters: Float64Array1D,
        backcast: float | Float64Array1D,
        nobs: int,
    ) -> None:
        _check_vector("parameters", parameters, 1 + self.p + self.o + self.q)
        self.backcast = cast("float", backcast)

    def update(
        self,
        t: int,
        parameters: Float64Array1D,
        resids: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
    ) -> None:
        p, o, q = self.p, self.o, self.q
        # The lagged resids and sigma2 are final: sigma2 is stored after the
        # variance bounds are applied
        sigma2[t] = parameters[0]
        for j in range(p):
            theta = parameters[1 + p + j] if j < o else 0.0
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[1 + j] * (1.0 + theta * theta) * self.backcast
            else:
                shock = resids[t - 1 - j] - theta * np.sqrt(sigma2[t - 1 - j])
                sigma2[t] += parameters[1 + j] * shock * shock
        for j in range(q):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[1 + p + o + j] * self.backcast
            else:
                sigma2[t] += parameters[1 + p + o + j] * sigma2[t - 1 - j]
        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])


class ARCHInMeanRecursion:
    def __init__(self, updater: VolatilityUpdater):
        if not isinstance(updater, VolatilityUpdater):
            raise TypeError(
                f"updater must be a VolatilityUpdater; got {_describe(updater)}"
            )
        self.volatility_updater = updater

    def recursion(
        self,
        y: Float64Array,
        x: Float64Array,
        mean_parameters: Float64Array1D,
        variance_params: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
        power: float,
    ) -> Float64Array:
        # Checked once per call, as in the compiled version
        _check_vector("y", y, 0)
        nobs = y.shape[0]
        _check_matrix("x", x, nobs, 0)
        k = x.shape[1]
        _check_vector("mean_parameters", mean_parameters, k + 1)
        # The length that variance_params needs depends on the updater
        _check_vector("variance_params", variance_params, 0)
        _check_vector("sigma2", sigma2, nobs)
        _check_matrix("var_bounds", var_bounds, nobs, 2)
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
