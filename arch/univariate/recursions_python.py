"""
Pure Python implementations of the core recursions in the models. Only used for
testing and if it is not possible to install the Cython version using
python setup.py install --no-binary
"""
from arch.compat.numba import jit

import numpy as np

from arch.typing import NDArray

__all__ = [
    "harch_recursion",
    "arch_recursion",
    "garch_recursion",
    "egarch_recursion",
    "midas_recursion",
    "figarch_weights",
    "figarch_recursion",
    "aparch_recursion",
]

LNSIGMA_MAX = np.log(np.finfo(np.double).max) - 0.1


def bounds_check_python(sigma2: float, var_bounds: NDArray) -> float:
    if sigma2 < var_bounds[0]:
        sigma2 = var_bounds[0]
    elif sigma2 > var_bounds[1]:
        if not np.isinf(sigma2):
            sigma2 = var_bounds[1] + np.log(sigma2 / var_bounds[1])
        else:
            sigma2 = var_bounds[1] + 1000
    return sigma2


bounds_check = jit(bounds_check_python, nopython=True)


def harch_recursion_python(
    parameters: NDArray,
    resids: NDArray,
    sigma2: NDArray,
    lags: NDArray,
    nobs: int,
    backcast: float,
    var_bounds: NDArray,
) -> NDArray:
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
    parameters: NDArray,
    resids: NDArray,
    sigma2: NDArray,
    p: int,
    nobs: int,
    backcast: float,
    var_bounds: NDArray,
) -> NDArray:
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


def garch_recursion_python(
    parameters: NDArray,
    fresids: NDArray,
    sresids: NDArray,
    sigma2: NDArray,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: NDArray,
) -> NDArray:
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
    parameters: NDArray,
    resids: NDArray,
    sigma2: NDArray,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: NDArray,
    lnsigma2: NDArray,
    std_resids: NDArray,
    abs_std_resids: NDArray,
) -> NDArray:
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
    norm_const = 0.79788456080286541  # E[abs(e)], e~N(0,1)

    for t in range(nobs):
        loc = 0
        lnsigma2[t] = parameters[loc]
        loc += 1
        for j in range(p):
            if (t - 1 - j) >= 0:
                lnsigma2[t] += parameters[loc] * (
                    abs_std_resids[t - 1 - j] - norm_const
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
    parameters: NDArray,
    weights: NDArray,
    resids: NDArray,
    sigma2: NDArray,
    nobs: int,
    backcast: float,
    var_bounds: NDArray,
) -> NDArray:
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
    parameters: NDArray, p: int, q: int, trunc_lag: int
) -> NDArray:
    r"""
    Parameters
    ----------
    parameters : ndarray
        Model parameters of the form (omega, phi, d, beta) where omega is the
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
    parameters: NDArray,
    fresids: NDArray,
    sigma2: NDArray,
    p: int,
    q: int,
    nobs: int,
    trunc_lag: int,
    backcast: float,
    var_bounds: NDArray,
) -> NDArray:
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
    parameters: NDArray,
    resids: NDArray,
    abs_resids: NDArray,
    sigma2: NDArray,
    sigma_delta: NDArray,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: NDArray,
) -> NDArray:
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
                shock = backcast ** 0.5
            else:
                shock = abs_resids[t - 1 - j]
                if o > j:
                    shock -= parameters[1 + p + j] * resids[t - 1 - j]
            sigma_delta[t] += parameters[1 + j] * (shock ** delta)
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
