#!python
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np

cimport numpy as np
from libc.float cimport DBL_MAX
from libc.math cimport exp, fabs, log, sqrt

__all__ = ['harch_recursion', 'arch_recursion', 'garch_recursion', 'egarch_recursion',
           'midas_recursion', 'figarch_recursion', 'figarch_weights', 'aparch_recursion']

cdef double LNSIGMA_MAX = log(DBL_MAX)

np.import_array()

cdef inline void bounds_check(double* sigma2, double* var_bounds):
    if sigma2[0] < var_bounds[0]:
        sigma2[0] = var_bounds[0]
    elif sigma2[0] > var_bounds[1]:
        if sigma2[0] > DBL_MAX:
            sigma2[0] = var_bounds[1] + 1000
        else:
            sigma2[0] = var_bounds[1] + log(sigma2[0] / var_bounds[1])


def harch_recursion(double[::1] parameters,
                    double[::1] resids,
                    double[::1] sigma2,
                    int[::1] lags,
                    int nobs,
                    double backcast,
                    double[:, ::1] var_bounds):
    """
    Parameters
    ----------
    parameters : 1-d array, float64
        Model parameters
    resids : 1-d array, float64
        Residuals to use in the recursion
    sigma2 : 1-d array, float64
        Conditional variances with same shape as resids
    lags : 1-d array, int
        Lag lengths in the HARCH
    nobs : int
        Length of resids
    backcast : float64
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period
    """
    cdef Py_ssize_t t, i, num_lags
    cdef int j
    cdef double param
    num_lags = lags.shape[0]

    for t in range(nobs):
        sigma2[t] = parameters[0]
        for i in range(num_lags):
            param = parameters[i + 1] / lags[i]
            for j in range(lags[i]):
                if (t - j - 1) >= 0:
                    sigma2[t] += param * resids[t - j - 1] * resids[t - j - 1]
                else:
                    sigma2[t] += param * backcast
        bounds_check(&sigma2[t], &var_bounds[t, 0])

    return np.asarray(sigma2)

def arch_recursion(double[::1] parameters,
                   double[::1] resids,
                   double[::1] sigma2,
                   int p,
                   int nobs,
                   double backcast,
                   double[:, ::1] var_bounds):
    """
    Parameters
    ----------
    parameters : 1-d array, float64
        Model parameters
    resids : 1-d array, float64
        Residuals to use in the recursion
    sigma2 : 1-d array, float64
        Conditional variances with same shape as resids
    p : int
        Number of lags in ARCH model
    nobs : int
        Length of resids
    backcast : float64
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period
    """

    cdef Py_ssize_t t
    cdef int i
    cdef double param

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

def garch_recursion(double[::1] parameters,
                    double[::1] fresids,
                    double[::1] sresids,
                    double[::1] sigma2,
                    int p,
                    int o,
                    int q,
                    int nobs,
                    double backcast,
                    double[:, ::1] var_bounds):
    """
    Compute variance recursion for GARCH and related models

    Parameters
    ----------
    parameters : 1-d array, float64
        Model parameters
    fresids : 1-d array, float64
        Absolute value of residuals raised to the power in the model.  For
        example, in a standard GARCH model, the power is 2.0.
    sresids : 1-d array, float64
        Variable containing the sign of the residuals (-1.0, 0.0, 1.0)
    sigma2 : 1-d array, float64
        Conditional variances with same shape as resids
    p : int
        Number of symmetric innovations in model
    o : int
        Number of asymmetric innovations in model
    q : int
        Number of lags of the (transformed) variance in the model
    nobs : int
        Length of resids
    backcast : float64
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

def egarch_recursion(double[::1] parameters,
                     double[::1] resids,
                     double[::1] sigma2,
                     int p,
                     int o,
                     int q,
                     int nobs,
                     double backcast,
                     double[:, ::1] var_bounds,
                     double[::1] lnsigma2,
                     double[::1] std_resids,
                     double[::1] abs_std_resids):
    """
    Compute variance recursion for EGARCH models

    Parameters
    ----------
    parameters : 1-d array, float64
        Model parameters
    resids : 1-d array, float64
        Residuals to use in the recursion
    sigma2 : 1-d array, float64
        Conditional variances with same shape as resids
    p : int
        Number of symmetric innovations in model
    o : int
        Number of asymmetric innovations in model
    q : int
        Number of lags of the (transformed) variance in the model
    nobs : int
        Length of resids
    backcast : float64
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period
    lnsigma2 : 1-d array, float64
        Temporary array (overwritten) with same shape as resids
    std_resids : 1-d array, float64
        Temporary array (overwritten) with same shape as resids
    abs_std_resids : 1-d array, float64
        Temporary array (overwritten) with same shape as resids
    """

    cdef double norm_const = 0.79788456080286541  # E[abs(e)], e~N(0,1)
    cdef Py_ssize_t t
    cdef int j, loc

    for t in range(nobs):
        loc = 0
        lnsigma2[t] = parameters[loc]
        loc += 1
        for j in range(p):
            if (t - 1 - j) >= 0:
                lnsigma2[t] += parameters[loc] * (abs_std_resids[t - 1 - j] - norm_const)
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
        elif sigma2[t] > var_bounds[t, 1]:
            sigma2[t] = var_bounds[t, 1] + log(sigma2[t]) - log(var_bounds[t, 1])
        std_resids[t] = resids[t] / sqrt(sigma2[t])
        abs_std_resids[t] = fabs(std_resids[t])

    return np.asarray(sigma2)



def midas_recursion(double[::1] parameters,
                    double[::1] weights,
                    double[::1] resids,
                    double[::1] sigma2,
                    int nobs,
                    double backcast,
                    double[:, ::1] var_bounds):
    """
    Parameters
    ----------
    parameters : 1-d array, float64
        Model parameters
    weights : 1-d array, float64
        Weights for MIDAS recursions
    resids : 1-d array, float64
        Residuals to use in the recursion
    sigma2 : 1-d array, float64
        Conditional variances with same shape as resids
    nobs : int
        Length of resids
    backcast : float64
        Value to use when initializing the recursion
    var_bounds : 2-d array
        nobs by 2-element array of upper and lower bounds for conditional
        variances for each time period
    """
    cdef Py_ssize_t m, t, i, num_lags
    cdef int j
    cdef double param, omega, alpha, gamma
    cdef double [::1] aw, gw, resids2

    m = weights.shape[0]
    omega = parameters[0]
    alpha = parameters[1]
    gamma = parameters[2]

    aw = np.zeros(m, dtype=np.float64)
    gw = np.zeros(m, dtype=np.float64)
    for i in range(m):
        aw[i] = alpha * weights[i]
        gw[i] = gamma * weights[i]

    resids2 = np.zeros(nobs, dtype=np.float64)

    for t in range(nobs):
        resids2[t] = resids[t] * resids[t]
        sigma2[t] = omega
        for i in range(m):
            if (t - i - 1) >= 0:
                sigma2[t] += (aw[i] + gw[i] * (resids[t - i - 1] < 0)) * resids2[t - i - 1]
            else:
                sigma2[t] +=  (aw[i] + 0.5 * gw[i]) * backcast

        bounds_check(&sigma2[t], &var_bounds[t, 0])

    return np.asarray(sigma2)

cdef double[::1] _figarch_weights(double[::1] parameters,
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


def figarch_weights(double[::1] parameters, int p, int q, int trunc_lag):
    return np.asarray(_figarch_weights(parameters, p, q, trunc_lag))


def figarch_recursion(double[::1] parameters,
                      double[::1] fresids,
                      double[::1] sigma2,
                      int p,
                      int q,
                      int nobs,
                      int trunc_lag,
                      double backcast,
                      double[:, ::1] var_bounds):
    cdef Py_ssize_t t, i
    cdef double bc1, bc2, bc_weight, omega, beta, omega_tilde
    cdef double [::1] lam

    omega = parameters[0]
    beta = parameters[1 + p + q] if q else 0.0
    omega_tilde = omega / (1-beta)
    lam = _figarch_weights(parameters[1:], p, q, trunc_lag)
    for t in range(nobs):
        bc_weight = 0.0
        for i in range(t, trunc_lag):
            bc_weight += lam[i]
        sigma2[t] = omega_tilde + bc_weight * backcast
        for i in range(min(t, trunc_lag)):
            sigma2[t] += lam[i] * fresids[t - i - 1]
        bounds_check(&sigma2[t], &var_bounds[t, 0])

    return np.asarray(sigma2)


def aparch_recursion(double[::1] parameters,
                    double[::1] resids,
                    double[::1] abs_resids,
                    double[::1] sigma2,
                    double[::1] sigma_delta,
                    int p,
                    int o,
                    int q,
                    int nobs,
                    double backcast,
                    double[:, ::1] var_bounds):
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
