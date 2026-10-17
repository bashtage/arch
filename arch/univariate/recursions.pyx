#!python

import numpy as np

cimport numpy as np
from libc.float cimport DBL_MAX
from libc.math cimport exp, fabs, lgamma, log, sqrt

from arch.utility._validation import _describe, check_int

cdef:
    double SQRT2_OV_PI = 0.79788456080286535587989211


__all__ = [
    "harch_recursion",
    "arch_recursion",
    "garch_recursion",
    "egarch_recursion",
    "midas_recursion",
    "figarch_recursion",
    "figarch_weights",
    "aparch_recursion",
    "harch_core",
    "garch_core",
    "GARCHUpdater",
    "HARCHUpdater",
    "EGARCHUpdater",
    "EWMAUpdater",
    "ARCHInMeanRecursion",
    "VolatilityUpdater",
    "FIGARCHUpdater",
    "MIDASUpdater",
]

cdef double LNSIGMA_MAX = log(DBL_MAX)

np.import_array()

cdef int _bad_array(str name, object value, str rule, bint wrong_type) except -1:
    """Raise TypeError (wrong_type) or ValueError: <name> must be <rule>; got <value>"""
    exc = TypeError if wrong_type else ValueError
    raise exc(f"{name} must be {rule}; got {_describe(value)}")


cdef str _vector_rule(Py_ssize_t min_length):
    """The rule in the message, which is only built when a check fails."""
    rule = "a 1-d C contiguous float64 ndarray"
    if min_length > 0:
        rule += f" with at least {min_length} element{'' if min_length == 1 else 's'}"
    return rule


cdef int _check_vector(
    str name, const double[::1] arr, Py_ssize_t min_length
) except -1:
    """
    Check that an argument is not None and has at least min_length elements.

    The checks are O(1) and are made once per likelihood evaluation, never in the
    per-observation update. A memoryview argument accepts None unless it is
    declared "not None", and the recursions read and write memoryviews without
    bounds checks.
    """
    if arr is None:
        _bad_array(name, None, _vector_rule(min_length), True)
    if arr.shape[0] < min_length:
        _bad_array(name, np.asarray(arr), _vector_rule(min_length), False)
    return 0


cdef int _check_matrix(
    str name, const double[:, ::1] arr, Py_ssize_t min_rows, Py_ssize_t min_cols
) except -1:
    """Check that an argument is not None and has at least min_rows by min_cols."""
    cdef str rule
    if arr is None or arr.shape[0] < min_rows or arr.shape[1] < min_cols:
        rule = (
            "a 2-d C contiguous float64 ndarray with at least "
            f"{min_rows} row{'' if min_rows == 1 else 's'} and "
            f"{min_cols} column{'' if min_cols == 1 else 's'}"
        )
        _bad_array(name, None if arr is None else np.asarray(arr), rule, arr is None)
    return 0


cdef inline void bounds_check(double* sigma2, const double* var_bounds):
    if sigma2[0] < var_bounds[0]:
        sigma2[0] = var_bounds[0]
    elif sigma2[0] > var_bounds[1]:
        if sigma2[0] > DBL_MAX:
            sigma2[0] = var_bounds[1] + 1000
        else:
            sigma2[0] = var_bounds[1] + log(sigma2[0] / var_bounds[1])


def harch_core(
    Py_ssize_t t,
    const double[::1] parameters,
    const double[::1] resids,
    double[::1] sigma2,
    const np.int32_t[::1] lags,
    double backcast,
    const double[:, ::1] var_bounds,
):
    """
    Compute variance recursion step for HARCH model

    Parameters
    ----------
    t: int
        Location of variance to compute. Assumes variance has been computed
        at times t-1, t-2, ...
    parameters : 1-d array, double
        Model parameters
    resids : 1-d array, double
        Residuals to use in the recursion
    sigma2 : 1-d array, double
        Conditional variances with same shape as resids
    lags : 1-d array, int
        Lag lengths in the HARCH
    backcast : double
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period

    Returns
    -------
    float
        Conditional variance at time t
    """
    cdef:
        Py_ssize_t i, j
        double param

    sigma2[t] = parameters[0]
    for i in range(lags.shape[0]):
        param = parameters[i + 1] / lags[i]
        for j in range(lags[i]):
            if (t - j - 1) >= 0:
                sigma2[t] += param * resids[t - j - 1] * resids[t - j - 1]
            else:
                sigma2[t] += param * backcast

    bounds_check(&sigma2[t], &var_bounds[t, 0])
    return sigma2[t]


cdef inline void _harch_step(Py_ssize_t t,
                             const double[::1] parameters,
                             const double[::1] resids,
                             double[::1] sigma2,
                             const np.int32_t[::1] lags,
                             double backcast,
                             const double[:, ::1] var_bounds):
    cdef Py_ssize_t i, j
    cdef double param

    sigma2[t] = parameters[0]
    for i in range(lags.shape[0]):
        param = parameters[i + 1] / lags[i]
        for j in range(lags[i]):
            if (t - j - 1) >= 0:
                sigma2[t] += param * resids[t - j - 1] * resids[t - j - 1]
            else:
                sigma2[t] += param * backcast
    bounds_check(&sigma2[t], &var_bounds[t, 0])


def harch_recursion(const double[::1] parameters,
                    const double[::1] resids,
                    double[::1] sigma2,
                    np.int32_t[::1] lags,
                    int nobs,
                    double backcast,
                    const double[:, ::1] var_bounds):
    """
    Parameters
    ----------
    parameters : 1-d array, double
        Model parameters
    resids : 1-d array, double
        Residuals to use in the recursion
    sigma2 : 1-d array, double
        Conditional variances with same shape as resids
    lags : 1-d array, int
        Lag lengths in the HARCH
    nobs : int
        Length of resids
    backcast : double
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period
    """
    cdef Py_ssize_t t, i, j, num_lags, max_lag
    cdef double param, a0, a1, a2, a3
    num_lags = lags.shape[0]
    max_lag = 0
    for i in range(num_lags):
        if lags[i] > max_lag:
            max_lag = lags[i]

    # The first max_lag observations use the backcast
    t = 0
    while t < min(max_lag, nobs):
        _harch_step(t, parameters, resids, sigma2, lags, backcast, var_bounds)
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
        sigma2[t] = a0
        sigma2[t + 1] = a1
        sigma2[t + 2] = a2
        sigma2[t + 3] = a3
        bounds_check(&sigma2[t], &var_bounds[t, 0])
        bounds_check(&sigma2[t + 1], &var_bounds[t + 1, 0])
        bounds_check(&sigma2[t + 2], &var_bounds[t + 2, 0])
        bounds_check(&sigma2[t + 3], &var_bounds[t + 3, 0])
        t += 4
    while t < nobs:
        _harch_step(t, parameters, resids, sigma2, lags, backcast, var_bounds)
        t += 1

    return np.asarray(sigma2)


def arch_recursion(const double[::1] parameters,
                   const double[::1] resids,
                   double[::1] sigma2,
                   int p,
                   int nobs,
                   double backcast,
                   const double[:, ::1] var_bounds):
    """
    Parameters
    ----------
    parameters : 1-d array, double
        Model parameters
    resids : 1-d array, double
        Residuals to use in the recursion
    sigma2 : 1-d array, double
        Conditional variances with same shape as resids
    p : int
        Number of lags in ARCH model
    nobs : int
        Length of resids
    backcast : double
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period
    """

    cdef Py_ssize_t t
    cdef int i

    for t in range(nobs):
        sigma2[t] = parameters[0]
        for i in range(p):
            if (t - i - 1) < 0:
                sigma2[t] += parameters[i + 1] * backcast
            else:
                sigma2[t] += parameters[i + 1] * resids[t - i - 1] * \
                             resids[t - i - 1]
        bounds_check(&sigma2[t], &var_bounds[t, 0])

    return np.asarray(sigma2)


def garch_core(
    Py_ssize_t t,
    const double[::1] parameters,
    const double[::1] resids,
    double[::1] sigma2,
    double backcast,
    const double[:, ::1] var_bounds,
    int p,
    int o,
    int q,
    double power,
):
    """
    Compute variance recursion step for GARCH and related models

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
    cdef:
        Py_ssize_t j, loc

    loc = 0
    sigma2[t] = parameters[loc]
    loc += 1
    for j in range(p):
        if (t - 1 - j) < 0:
            sigma2[t] += parameters[loc] * backcast
        else:
            sigma2[t] += parameters[loc] * (fabs(resids[t - 1 - j]) ** power)
        loc += 1
    for j in range(o):
        if (t - 1 - j) < 0:
            sigma2[t] += parameters[loc] * 0.5 * backcast
        else:
            sigma2[t] += (
                parameters[loc]
                * (fabs(resids[t - 1 - j]) ** power)
                * (resids[t - 1 - j] < 0)
            )
        loc += 1
    for j in range(q):
        if (t - 1 - j) < 0:
            sigma2[t] += parameters[loc] * backcast
        else:
            sigma2[t] += parameters[loc] * sigma2[t - 1 - j]
        loc += 1
    bounds_check(&sigma2[t], &var_bounds[t, 0])

    return sigma2[t]


def garch_recursion(const double[::1] parameters,
                    const double[::1] fresids,
                    const double[::1] sresids,
                    double[::1] sigma2,
                    int p,
                    int o,
                    int q,
                    int nobs,
                    double backcast,
                    const double[:, ::1] var_bounds):
    """
    Compute variance recursion for GARCH and related models

    Parameters
    ----------
    parameters : 1-d array, double
        Model parameters
    fresids : 1-d array, double
        Absolute value of residuals raised to the power in the model.  For
        example, in a standard GARCH model, the power is 2.0.
    sresids : 1-d array, double
        Variable containing the sign of the residuals (-1.0, 0.0, 1.0)
    sigma2 : 1-d array, double
        Conditional variances with same shape as resids
    p : int
        Number of symmetric innovations in model
    o : int
        Number of asymmetric innovations in model
    q : int
        Number of lags of the (transformed) variance in the model
    nobs : int
        Length of resids
    backcast : double
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        transformed variances for each time period
    """

    cdef Py_ssize_t t
    cdef int j, loc

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
                sigma2[t] += parameters[loc] * fresids[t - 1 - j] * (sresids[t-1-j] < 0)
            loc += 1
        for j in range(q):
            if (t - 1 - j) < 0:
                sigma2[t] += parameters[loc] * backcast
            else:
                sigma2[t] += parameters[loc] * sigma2[t - 1 - j]
            loc += 1
        bounds_check(&sigma2[t], &var_bounds[t, 0])

    return np.asarray(sigma2)


def egarch_recursion(const double[::1] parameters,
                     const double[::1] resids,
                     double[::1] sigma2,
                     int p,
                     int o,
                     int q,
                     int nobs,
                     double backcast,
                     const double[:, ::1] var_bounds,
                     double[::1] lnsigma2,
                     double[::1] std_resids,
                     double[::1] abs_std_resids):
    """
    Compute variance recursion for EGARCH models

    Parameters
    ----------
    parameters : 1-d array, double
        Model parameters
    resids : 1-d array, double
        Residuals to use in the recursion
    sigma2 : 1-d array, double
        Conditional variances with same shape as resids
    p : int
        Number of symmetric innovations in model
    o : int
        Number of asymmetric innovations in model
    q : int
        Number of lags of the (transformed) variance in the model
    nobs : int
        Length of resids
    backcast : double
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period
    lnsigma2 : 1-d array, double
        Temporary array (overwritten) with same shape as resids
    std_resids : 1-d array, double
        Temporary array (overwritten) with same shape as resids
    abs_std_resids : 1-d array, double
        Temporary array (overwritten) with same shape as resids
    """

    cdef Py_ssize_t t
    cdef int j, loc

    for t in range(nobs):
        loc = 0
        lnsigma2[t] = parameters[loc]
        loc += 1
        for j in range(p):
            if (t - 1 - j) >= 0:
                lnsigma2[t] += (
                        parameters[loc] * (abs_std_resids[t - 1 - j] - SQRT2_OV_PI)
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
        sigma2[t] = exp(lnsigma2[t])
        if sigma2[t] < var_bounds[t, 0]:
            sigma2[t] = var_bounds[t, 0]
            lnsigma2[t] = log(sigma2[t])
        elif sigma2[t] > var_bounds[t, 1]:
            sigma2[t] = var_bounds[t, 1] + log(sigma2[t]) - log(var_bounds[t, 1])
            lnsigma2[t] = log(sigma2[t])
        std_resids[t] = resids[t] / sqrt(sigma2[t])
        abs_std_resids[t] = fabs(std_resids[t])

    return np.asarray(sigma2)


cdef inline void _midas_step(Py_ssize_t t,
                             double omega,
                             const double[::1] aw,
                             const double[::1] gw,
                             const double[::1] resids,
                             const double[::1] resids2,
                             double[::1] sigma2,
                             double backcast,
                             const double[:, ::1] var_bounds):
    cdef Py_ssize_t i

    sigma2[t] = omega
    for i in range(aw.shape[0]):
        if (t - i - 1) >= 0:
            sigma2[t] += (
                    (aw[i] + gw[i] * (resids[t - i - 1] < 0)) * resids2[t - i - 1]
            )
        else:
            sigma2[t] += (aw[i] + 0.5 * gw[i]) * backcast
    bounds_check(&sigma2[t], &var_bounds[t, 0])


def midas_recursion(const double[::1] parameters,
                    const double[::1] weights,
                    const double[::1] resids,
                    double[::1] sigma2,
                    int nobs,
                    double backcast,
                    double[:, ::1] var_bounds):
    """
    Parameters
    ----------
    parameters : 1-d array, double
        Model parameters
    weights : 1-d array, double
        Weights for MIDAS recursions
    resids : 1-d array, double
        Residuals to use in the recursion
    sigma2 : 1-d array, double
        Conditional variances with same shape as resids
    nobs : int
        Length of resids
    backcast : double
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period
    """
    cdef Py_ssize_t m, t, i
    cdef double omega, alpha, gamma, a0, a1, a2, a3
    cdef double [::1] aw, gw, resids2

    m = weights.shape[0]
    omega = parameters[0]
    alpha = parameters[1]
    gamma = parameters[2]

    aw = np.zeros(m, dtype=np.double)
    gw = np.zeros(m, dtype=np.double)
    for i in range(m):
        aw[i] = alpha * weights[i]
        gw[i] = gamma * weights[i]

    resids2 = np.zeros(nobs, dtype=np.double)
    for t in range(nobs):
        resids2[t] = resids[t] * resids[t]

    # The first m observations use the backcast
    t = 0
    while t < min(m, nobs):
        _midas_step(t, omega, aw, gw, resids, resids2, sigma2, backcast, var_bounds)
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
        sigma2[t] = a0
        sigma2[t + 1] = a1
        sigma2[t + 2] = a2
        sigma2[t + 3] = a3
        bounds_check(&sigma2[t], &var_bounds[t, 0])
        bounds_check(&sigma2[t + 1], &var_bounds[t + 1, 0])
        bounds_check(&sigma2[t + 2], &var_bounds[t + 2, 0])
        bounds_check(&sigma2[t + 3], &var_bounds[t + 3, 0])
        t += 4
    while t < nobs:
        _midas_step(t, omega, aw, gw, resids, resids2, sigma2, backcast, var_bounds)
        t += 1

    return np.asarray(sigma2)


cdef double[::1] _figarch_weights(const double[::1] parameters,
                                  int p,
                                  int q,
                                  int trunc_lag):

    cdef double phi, d, beta
    cdef double [::1] lam, delta
    cdef Py_ssize_t i

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


def figarch_weights(const double[::1] parameters, int p, int q, int trunc_lag):
    return np.asarray(_figarch_weights(parameters, p, q, trunc_lag))


cdef double[::1] _backcast_weights(const double[::1] lam,
                                   Py_ssize_t nobs,
                                   Py_ssize_t trunc_lag):
    """
    Weights on the backcast in the ARCH(inf) representation

    Element t is lam[t] + ... + lam[trunc_lag - 1], the weight on the lags that
    are before the start of the sample, which is 0 for t >= trunc_lag. Each sum
    adds the terms in increasing order of the lag.
    """
    cdef Py_ssize_t t, i, n_bc
    cdef double w, a0, a1, a2, a3
    cdef double[::1] bc_weights

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


cdef void _arch_inf_sums(const double[::1] lam,
                         const double[::1] shocks,
                         const double[::1] bc_weights,
                         double omega_tilde,
                         double backcast,
                         Py_ssize_t nobs,
                         Py_ssize_t trunc_lag,
                         double[::1] sums):
    """
    Truncated ARCH(inf) sums of FIGARCH-type models

    Sets sums[t] to

        omega_tilde + bc_weights[t] * backcast + sum_i lam[i] * shocks[t - i - 1]

    where the sum is over i < min(t, trunc_lag) and is accumulated in increasing
    order of i, after the first two terms.
    """
    cdef Py_ssize_t t, i, m
    cdef double w, a0, a1, a2, a3

    # As in _backcast_weights, four sums are accumulated at a time
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


def figarch_recursion(const double[::1] parameters,
                      const double[::1] fresids,
                      double[::1] sigma2,
                      int p,
                      int q,
                      int nobs,
                      int trunc_lag,
                      double backcast,
                      double[:, ::1] var_bounds):
    cdef Py_ssize_t t
    cdef double omega, beta, omega_tilde
    cdef double [::1] lam, bc_weights

    omega = parameters[0]
    beta = parameters[1 + p + q] if q else 0.0
    omega_tilde = omega / (1 - beta)
    lam = _figarch_weights(parameters[1:], p, q, trunc_lag)
    bc_weights = _backcast_weights(lam, nobs, trunc_lag)
    _arch_inf_sums(
        lam, fresids, bc_weights, omega_tilde, backcast, nobs, trunc_lag, sigma2
    )
    for t in range(nobs):
        bounds_check(&sigma2[t], &var_bounds[t, 0])

    return np.asarray(sigma2)


def fiaparch_recursion(const double[::1] parameters,
                       const double[::1] resids,
                       const double[::1] abs_resids,
                       double[::1] sigma2,
                       double[::1] sigma_delta,
                       int p,
                       int q,
                       int nobs,
                       int trunc_lag,
                       double backcast,
                       double[:, ::1] var_bounds,
                       double gamma,
                       double delta):
    cdef Py_ssize_t t
    cdef double omega, beta, omega_tilde, backcast_delta
    cdef double [::1] lam, fshocks, bc_weights

    omega = parameters[0]
    beta = parameters[1 + p + q] if q else 0.0
    omega_tilde = omega / (1 - beta)
    lam = _figarch_weights(parameters[1:], p, q, trunc_lag)
    backcast_delta = backcast ** (delta / 2.0)
    # The transformed shock depends only on the observation, not on the lag, so
    # raise it to delta once (nobs powers) instead of once per (t, lag) pair
    fshocks = np.empty(nobs)
    for t in range(nobs - 1):
        fshocks[t] = (abs_resids[t] - gamma * resids[t]) ** delta
    bc_weights = _backcast_weights(lam, nobs, trunc_lag)
    _arch_inf_sums(
        lam, fshocks, bc_weights, omega_tilde, backcast_delta, nobs, trunc_lag,
        sigma_delta
    )
    # sigma_delta is a work array that holds sigma ** delta before the variance
    # bounds are applied. It is not read by any caller, so it is not recomputed from
    # the bounded variance, which would cost a call to pow for each observation.
    for t in range(nobs):
        sigma2[t] = sigma_delta[t] ** (2.0 / delta)
        bounds_check(&sigma2[t], &var_bounds[t, 0])

    return np.asarray(sigma2)


def aparch_recursion(const double[::1] parameters,
                     const double[::1] resids,
                     const double[::1] abs_resids,
                     double[::1] sigma2,
                     double[::1] sigma_delta,
                     int p,
                     int o,
                     int q,
                     int nobs,
                     double backcast,
                     const double[:, ::1] var_bounds):
    """
    Compute variance recursion for Asymmetric Power ARCH

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
    cdef double delta, shock
    cdef Py_ssize_t t, j

    delta = parameters[1 + p + o + q]
    for t in range(nobs):
        sigma_delta[t] = parameters[0]
        for j in range(p):
            if (t - 1 - j) < 0:
                shock = sqrt(backcast)
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
        bounds_check(&sigma2[t], &var_bounds[t, 0])
        sigma_delta[t] = sigma2[t] ** (delta / 2.0)
    return np.asarray(sigma2)


cdef class VolatilityUpdater:
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
    def __init__(self):
        pass

    def initialize_update(
            self,
            const double[::1] parameters,
            object backcast,
            Py_ssize_t nobs
    ):
        """
        Initialize the recursion prior to calling update.

        Parameters
        ----------
        parameters : ndarray
            The model parameters.
        backcast : float or ndarray
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

    cdef void update(self,
                     Py_ssize_t t,
                     const double[::1] parameters,
                     const double[::1] resids,
                     double[::1] sigma2,
                     const double[:, ::1] var_bounds
                     ):
        pass

    def _update_tester(self,
                       Py_ssize_t t,
                       const double[::1] parameters,
                       const double[::1] resids,
                       double[::1] sigma2,
                       const double[:, ::1] var_bounds):
        self.update(t, parameters, resids, sigma2, var_bounds)

cdef class GARCHUpdater(VolatilityUpdater):
    cdef:
        int p, o, q
        double power, backcast
        double[::1] sigma_power

    def __init__(self, int p, int o, int q, double power):
        self.p = p
        self.o = o
        self.q = q
        self.power = power
        self.backcast = -1.0
        self.sigma_power = np.empty(0)

    def __setstate__(self, state):
        cdef Py_ssize_t i
        cdef double[::1] temp
        self.backcast = state[0]
        temp = state[1]
        self.sigma_power = np.empty(temp.shape[0])
        for i in range(temp.shape[0]):
            self.sigma_power[i] = temp[i]

    def __reduce__(self):
        return (
            GARCHUpdater,
            (self.p, self.o, self.q, self.power),
            (self.backcast, np.asarray(self.sigma_power))
        )

    def initialize_update(
            self,
            const double[::1] parameters,
            object backcast,
            Py_ssize_t nobs
    ):
        _check_vector("parameters", parameters, 1 + self.p + self.o + self.q)
        self.backcast = backcast
        if self.sigma_power.shape[0] < nobs:
            self.sigma_power = np.empty(nobs)

    cdef void update(self,
                     Py_ssize_t t,
                     const double[::1] parameters,
                     const double[::1] resids,
                     double[::1] sigma2,
                     const double[:, ::1] var_bounds
                     ):
        cdef:
            int p = self.p, o = self.o, q = self.q
            double power = self.power, backcast = self.backcast
            double sigma_power
            Py_ssize_t j, loc

        # The recursion runs in sigma ** power, which is what the backcast and
        # var_bounds use. sigma2 is the variance, sigma ** 2
        loc = 0
        sigma_power = parameters[loc]
        loc += 1
        for j in range(p):
            if (t - 1 - j) < 0:
                sigma_power += parameters[loc] * backcast
            else:
                sigma_power += parameters[loc] * (fabs(resids[t - 1 - j]) ** power)
            loc += 1
        for j in range(o):
            if (t - 1 - j) < 0:
                sigma_power += parameters[loc] * 0.5 * backcast
            else:
                sigma_power += (
                        parameters[loc]
                        * (fabs(resids[t - 1 - j]) ** power)
                        * (resids[t - 1 - j] < 0)
                )
            loc += 1
        for j in range(q):
            if (t - 1 - j) < 0:
                sigma_power += parameters[loc] * backcast
            else:
                sigma_power += parameters[loc] * self.sigma_power[t - 1 - j]
            loc += 1
        bounds_check(&sigma_power, &var_bounds[t, 0])
        self.sigma_power[t] = sigma_power
        sigma2[t] = sigma_power ** (2.0 / power)

cdef class HARCHUpdater(VolatilityUpdater):
    cdef:
        const np.int32_t[::1] lags
        double backcast

    def __init__(self, const np.int32_t[::1] lags):
        self.lags = lags
        self.backcast = -1.0

    def __setstate__(self, state):
        self.backcast = state[0]

    def __reduce__(self):
        return HARCHUpdater, (np.asarray(self.lags),), (self.backcast,)

    def initialize_update(
            self,
            const double[::1] parameters,
            object backcast,
            Py_ssize_t nobs
    ):
        _check_vector("parameters", parameters, 1 + self.lags.shape[0])
        self.backcast = backcast

    cdef void update(self,
                     Py_ssize_t t,
                     const double[::1] parameters,
                     const double[::1] resids,
                     double[::1] sigma2,
                     const double[:, ::1] var_bounds
                     ):
        cdef:
            double backcast = self.backcast
            Py_ssize_t i, j
            double param

        sigma2[t] = parameters[0]
        for i in range(self.lags.shape[0]):
            param = parameters[i + 1] / self.lags[i]
            for j in range(self.lags[i]):
                if (t - j - 1) >= 0:
                    sigma2[t] += param * resids[t - j - 1] * resids[t - j - 1]
                else:
                    sigma2[t] += param * backcast

        bounds_check(&sigma2[t], &var_bounds[t, 0])

cdef class EWMAUpdater(VolatilityUpdater):
    cdef:
        bint estimate_lam
        double[::1] params
        double backcast

    def __init__(self, object lam):
        self.estimate_lam = lam is None
        self.params = np.zeros(3)
        if lam is not None:
            self.params[1] = 1.0 - lam
            self.params[2] = lam

    def __setstate__(self, state):
        cdef Py_ssize_t i
        self.backcast = state[0]
        params = state[1]
        for i in range(3):
            self.params[i] = params[i]

    def __reduce__(self):
        lam = None if self.estimate_lam else self.params[2]
        return EWMAUpdater, (lam,), (self.backcast, np.asarray(self.params))

    def initialize_update(
            self,
            const double[::1] parameters,
            object backcast,
            Py_ssize_t nobs
    ):
        if self.estimate_lam:
            _check_vector("parameters", parameters, 1)
            self.params[1] = 1.0 - parameters[0]
            self.params[2] = parameters[0]
        self.backcast = backcast

    cdef void update(self,
                     Py_ssize_t t,
                     const double[::1] parameters,
                     const double[::1] resids,
                     double[::1] sigma2,
                     const double[:, ::1] var_bounds
                     ):

        sigma2[t] = self.params[0]
        if t == 0:
            sigma2[t] += self.backcast
        else:
            sigma2[t] += (self.params[1] * resids[t-1] * resids[t-1] +
                          self.params[2] * sigma2[t-1])
        bounds_check(&sigma2[t], &var_bounds[t, 0])

cdef class MIDASUpdater(VolatilityUpdater):
    cdef:
        int m
        bint asym
        double backcast
        double[::1] aw
        double[::1] gw
        double[::1] weights
        double[::1] resids2
        double DOUBLE_EPS

    def __init__(self, int m, bint asym):
        self.m = m
        self.asym = asym
        self.aw = np.empty(m)
        self.gw = np.empty(m)
        self.weights = np.empty(m)
        self.resids2 = np.empty(0)
        self.DOUBLE_EPS = np.finfo(np.double).eps

    def __setstate__(self, state):
        cdef Py_ssize_t i

        self.backcast = state[0]
        aw, gw, weights = state[1:4]
        for i in range(self.m):
            self.aw[i] = aw[i]
            self.gw[i] = gw[i]
            self.weights[i] = weights[i]
        resids2 = state[4]
        self.resids2 = np.empty_like(resids2)
        for i in range(self.resids2.shape[0]):
            self.resids2[i] = resids2[i]

    def __reduce__(self):
        return (MIDASUpdater,
                (self.m, self.asym),
                (
                    self.backcast,
                    np.asarray(self.aw),
                    np.asarray(self.gw),
                    np.asarray(self.weights),
                    np.asarray(self.resids2)
                ))

    cdef update_weights(self, double theta):
        cdef:
            double j, sum_w = 0.0
            Py_ssize_t i
        m = self.m
        # Prevent 0
        theta = theta if theta > self.DOUBLE_EPS else self.DOUBLE_EPS
        j = 1.0
        for i in range(m):
            self.weights[i] = exp(lgamma(theta + j) - lgamma(j + 1) - lgamma(theta))
            sum_w += self.weights[i]
            j += 1.0
        for i in range(m):
            self.weights[i] /= sum_w

    def initialize_update(
            self,
            const double[::1] parameters,
            object backcast,
            Py_ssize_t nobs
    ):
        cdef double alpha, gamma

        # omega, alpha, gamma (if asym) and theta
        _check_vector("parameters", parameters, 3 + <int>self.asym)
        self.update_weights(parameters[2 + <int>self.asym])
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

    cdef void update(self,
                     Py_ssize_t t,
                     const double[::1] parameters,
                     const double[::1] resids,
                     double[::1] sigma2,
                     const double[:, ::1] var_bounds
                     ):
        cdef Py_ssize_t i
        cdef double omega

        omega = parameters[0]
        if t > 0:
            self.resids2[t-1] = resids[t-1] * resids[t-1]

        sigma2[t] = omega
        for i in range(self.m):
            if (t - i - 1) >= 0:
                sigma2[t] += (
                        (self.aw[i] + self.gw[i] * (resids[t - i - 1] < 0))
                        * self.resids2[t - i - 1]
                )
            else:
                sigma2[t] += (self.aw[i] + 0.5 * self.gw[i]) * self.backcast

        bounds_check(&sigma2[t], &var_bounds[t, 0])

cdef class FIGARCHUpdater(VolatilityUpdater):
    cdef:
        int p, q, truncation
        double power
        double[::1] lam
        double[::1] fresids
        double backcast

    def __init__(self, int p, int q, double power, object truncation):
        check_int("truncation", truncation, lower=1)
        self.p = p
        self.q = q
        self.truncation = truncation
        self.power = power
        self.lam = np.empty(truncation)
        self.fresids = np.empty(0)

    def __setstate__(self, state):
        cdef Py_ssize_t i
        cdef double[::1] temp
        self.backcast = state[0]
        temp = state[1]
        assert self.lam.shape[0] == temp.shape[0], f"lam.shape[0]: {self.lam.shape[0]}"
        for i in range(self.truncation):
            self.lam[i] = temp[i]
        temp = state[2]
        self.fresids = np.empty(temp.shape[0])
        assert self.fresids.shape[0] == temp.shape[0]
        for i in range(temp.shape[0]):
            self.fresids[i] = temp[i]

    def __reduce__(self):
        return (
            FIGARCHUpdater,
            (
                self.p,
                self.q,
                self.power,
                self.truncation
            ),
            (
                self.backcast,
                np.asarray(self.lam),
                np.asarray(self.fresids)
            )
        )

    def initialize_update(
            self,
            const double[::1] parameters,
            object backcast,
            Py_ssize_t nobs
    ):
        # omega, phi (if p), d and beta (if q), which the weights and update read
        _check_vector("parameters", parameters, 2 + self.p + self.q)
        self.lam = _figarch_weights(parameters[1:], self.p, self.q, self.truncation)
        self.backcast = backcast
        if self.fresids.shape[0] < nobs:
            self.fresids = np.empty(nobs)

    cdef void update(self,
                     Py_ssize_t t,
                     const double[::1] parameters,
                     const double[::1] resids,
                     double[::1] sigma2,
                     const double[:, ::1] var_bounds
                     ):
        cdef Py_ssize_t i
        cdef double bc_weight, omega, beta, omega_tilde, sigma_power
        cdef int p = self.p, q = self.q, trunc_lag = self.truncation

        omega = parameters[0]
        beta = parameters[1 + p + q] if q else 0.0
        omega_tilde = omega / (1 - beta)

        if t > 0:
            self.fresids[t-1] = fabs(resids[t-1]) ** self.power

        bc_weight = 0.0
        for i in range(t, trunc_lag):
            bc_weight += self.lam[i]
        sigma_power = omega_tilde + bc_weight * self.backcast
        for i in range(min(t, trunc_lag)):
            sigma_power += self.lam[i] * self.fresids[t - i - 1]
        # The recursion runs in sigma ** power, which is what the backcast and
        # var_bounds use. sigma2 is the variance, sigma ** 2
        bounds_check(&sigma_power, &var_bounds[t, 0])
        sigma2[t] = sigma_power ** (2.0 / self.power)


cdef class FIAPARCHUpdater(VolatilityUpdater):
    cdef:
        int p, q, o, truncation, est_delta
        double delta
        double[::1] lam
        double[::1] _fshocks
        double backcast

    def __init__(
        self,
        int p,
        int q,
        int o,
        object truncation,
        bint est_delta,
        double delta
    ):
        check_int("truncation", truncation, lower=1)
        self.p = p
        self.q = q
        self.o = o
        self.truncation = truncation
        self.est_delta = est_delta
        self.delta = delta
        self.lam = np.empty(truncation)
        self._fshocks = np.empty(0)

    def __setstate__(self, state):
        cdef Py_ssize_t i
        cdef double[::1] temp
        self.backcast = state[0]
        self.delta = state[1]
        temp = state[2]
        assert self.lam.shape[0] == temp.shape[0]
        for i in range(self.truncation):
            self.lam[i] = temp[i]
        temp = state[3]
        self._fshocks = np.empty(temp.shape[0])
        for i in range(temp.shape[0]):
            self._fshocks[i] = temp[i]

    def __reduce__(self):
        return (
            FIAPARCHUpdater,
            (
                self.p,
                self.q,
                self.o,
                self.truncation,
                bool(self.est_delta),
                self.delta,
            ),
            (
                self.backcast,
                self.delta,
                np.asarray(self.lam),
                np.asarray(self._fshocks),
            )
        )

    def initialize_update(
            self,
            const double[::1] parameters,
            object backcast,
            Py_ssize_t nobs
    ):
        cdef double delta
        # omega, phi (if p), d, beta (if q), gamma (if o) and delta (if estimated)
        _check_vector(
            "parameters", parameters, 2 + self.p + self.q + self.o + self.est_delta
        )
        self.lam = _figarch_weights(parameters[1:], self.p, self.q, self.truncation)
        # The backcast is a variance, so convert it to sigma ** delta. It is
        # stored on this scale
        delta = parameters[parameters.shape[0] - 1] if self.est_delta else self.delta
        self.backcast = backcast ** (delta / 2.0)
        if self._fshocks.shape[0] < nobs:
            self._fshocks = np.empty(nobs)

    cdef void update(self,
                     Py_ssize_t t,
                     const double[::1] parameters,
                     const double[::1] resids,
                     double[::1] sigma2,
                     const double[:, ::1] var_bounds
                     ):
        cdef Py_ssize_t i
        cdef double bc_weight, omega, beta, omega_tilde, gamma, delta, sigma_delta
        cdef int p = self.p, q = self.q, o = self.o, trunc_lag = self.truncation

        omega = parameters[0]
        beta = parameters[1 + p + q] if q else 0.0
        omega_tilde = omega / (1 - beta)
        gamma = parameters[2 + p + q] if o else 0.0
        delta = parameters[parameters.shape[0] - 1] if self.est_delta else self.delta

        # The transformed shock depends only on the observation, so it is raised to
        # delta once, when it becomes available, rather than once for each lag
        if t > 0:
            self._fshocks[t-1] = (fabs(resids[t-1]) - gamma * resids[t-1]) ** delta

        bc_weight = 0.0
        for i in range(t, trunc_lag):
            bc_weight += self.lam[i]
        sigma_delta = omega_tilde + bc_weight * self.backcast
        for i in range(min(t, trunc_lag)):
            sigma_delta += self.lam[i] * self._fshocks[t - i - 1]
        sigma2[t] = sigma_delta ** (2.0 / delta)
        bounds_check(&sigma2[t], &var_bounds[t, 0])


cdef class RiskMetrics2006Updater(VolatilityUpdater):
    cdef:
        int kmax
        double[::1] backcast
        double[::1] combination_weights
        double[::1] smoothing_parameters
        double[::1] last_sigma2s

    def __init__(self, int kmax, combination_weights, smoothing_parameters):
        super().__init__()
        self.kmax = kmax
        self.combination_weights = combination_weights
        self.smoothing_parameters = smoothing_parameters
        self.backcast = np.empty(kmax)
        self.last_sigma2s = np.empty(kmax)

    def __setstate__(self, state):
        cdef Py_ssize_t i
        for i in range(self.kmax):
            self.backcast[i] = state[0][i]
            self.last_sigma2s[i] = state[1][i]

    def __reduce__(self):
        return (
            RiskMetrics2006Updater,
            (
                self.kmax,
                np.asarray(self.combination_weights),
                np.asarray(self.smoothing_parameters)
            ),
            (
                np.asarray(self.backcast),
                np.asarray(self.last_sigma2s)
            )
        )

    def initialize_update(self, parameters, backcast, nobs) -> None:
        if isinstance(backcast, (float, np.floating)):
            for i in range(self.kmax):
                self.backcast[i] = backcast
        else:
            for i in range(self.kmax):
                self.backcast[i] = backcast[i]

    cdef void update(self,
                     Py_ssize_t t,
                     const double[::1] parameters,
                     const double[::1] resids,
                     double[::1] sigma2,
                     const double[:, ::1] var_bounds
                     ):
        cdef:
            Py_ssize_t i

        sigma2[t] = 0.0
        if t > 0:
            for i in range(self.kmax):
                self.last_sigma2s[i] = (
                        (1 - self.smoothing_parameters[i]) * resids[t - 1] ** 2 +
                        self.smoothing_parameters[i] * self.last_sigma2s[i]
                )
                sigma2[t] += self.last_sigma2s[i] * self.combination_weights[i]
        else:
            for i in range(self.kmax):
                self.last_sigma2s[i] = self.backcast[i]
                sigma2[t] += self.last_sigma2s[i] * self.combination_weights[i]
        bounds_check(&sigma2[t], &var_bounds[t, 0])

cdef class EGARCHUpdater(VolatilityUpdater):
    cdef:
        double[::1] std_resids
        double[::1] abs_std_resids
        double[::1] lnsigma2
        double backcast
        int p, o, q

    def __init__(self, int p, int o, int q):
        super().__init__()
        self.p = p
        self.o = o
        self.q = q
        self.backcast = 9999.99
        self.lnsigma2 = np.empty(0)
        self.std_resids = np.empty(0)
        self.abs_std_resids = np.empty(0)

    def __setstate__(self, state):
        cdef Py_ssize_t i
        cdef double[::1] s1
        cdef double[::1] s2
        cdef double[::1] s3

        self.backcast = state[0]

        s1 = state[1]
        s2 = state[2]
        s3 = state[3]

        self._resize(s1.shape[0])
        for i in range(s1.shape[0]):
            self.lnsigma2[i] = s1[i]
            self.std_resids[i] = s2[i]
            self.abs_std_resids[i] = s3[i]

    def __reduce__(self):
        return (
            EGARCHUpdater,
            (
                self.p,
                self.o,
                self.q
            ),
            (
                self.backcast,
                np.asarray(self.lnsigma2),
                np.asarray(self.std_resids),
                np.asarray(self.abs_std_resids)
            )
        )

    cdef void _resize(self, Py_ssize_t nobs):
        if self.lnsigma2.shape[0] < nobs:
            self.lnsigma2 = np.empty(nobs)
            self.abs_std_resids = np.empty(nobs)
            self.std_resids = np.empty(nobs)

    def initialize_update(
            self,
            const double[::1] parameters,
            object backcast,
            Py_ssize_t nobs
    ):
        _check_vector("parameters", parameters, 1 + self.p + self.o + self.q)
        self.backcast = backcast
        self._resize(nobs)

    cdef void update(self,
                     Py_ssize_t t,
                     const double[::1] parameters,
                     const double[::1] resids,
                     double[::1] sigma2,
                     const double[:, ::1] var_bounds
                     ):
        cdef Py_ssize_t j, loc

        if t > 0:
            self.std_resids[t-1] = resids[t-1] / sqrt(sigma2[t-1])
            self.abs_std_resids[t-1] = fabs(self.std_resids[t-1])

        self.lnsigma2[t] = parameters[0]
        loc = 1
        for j in range(self.p):
            if (t - 1 - j) >= 0:
                self.lnsigma2[t] += (
                        parameters[loc] * (self.abs_std_resids[t - 1 - j] - SQRT2_OV_PI)
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
        sigma2[t] = exp(self.lnsigma2[t])
        if sigma2[t] < var_bounds[t, 0]:
            sigma2[t] = var_bounds[t, 0]
            self.lnsigma2[t] = log(sigma2[t])
        elif sigma2[t] > var_bounds[t, 1]:
            sigma2[t] = var_bounds[t, 1] + log(sigma2[t]) - log(var_bounds[t, 1])
            self.lnsigma2[t] = log(sigma2[t])


cdef class ARCHInMeanRecursion:
    cdef:
        VolatilityUpdater volatility_updater

    def __init__(self, object updater):
        if not isinstance(updater, VolatilityUpdater):
            raise TypeError(
                f"updater must be a VolatilityUpdater; got {_describe(updater)}"
            )
        self.volatility_updater = updater

    def recursion(self,
                  const double[::1] y,
                  const double[:, ::1] x,
                  const double[::1] mean_parameters,
                  const double[::1] variance_params,
                  double[::1] sigma2,
                  const double[:, ::1] var_bounds,
                  double power):
        cdef:
            Py_ssize_t t, i, nobs, k
            double[::1] resids
            double gamma

        # Checked once per call. The loop below reads and writes these arrays
        # without bounds checks, and None would be dereferenced.
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
                resids[t] -= gamma * log(sigma2[t])
            else:
                resids[t] -= gamma * sigma2[t] ** power

        return np.asarray(resids)
