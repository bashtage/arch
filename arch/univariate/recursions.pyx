import numpy as np
cimport numpy as np
cimport cython

__all__ = ['harch_recursion','arch_recursion','garch_recursion',
           'egarch_recursion']

cdef extern from 'math.h':
    double log(double x)
    double exp(double x)
    double sqrt(double x)
    double fabs(double x)

cdef extern from 'float.h':
    double DBL_MAX

cdef double LNSIGMA_MAX = log(DBL_MAX)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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
        if sigma2[t] < var_bounds[t, 0]:
            sigma2[t] = var_bounds[t, 0]
        elif sigma2[t] > var_bounds[t, 1]:
            if sigma2[t] > DBL_MAX:
                sigma2[t] = var_bounds[t, 1] + 1000
            else:
                sigma2[t] = var_bounds[t, 1] + log(sigma2[t] / var_bounds[t, 1])

    return np.asarray(sigma2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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
        if sigma2[t] < var_bounds[t, 0]:
            sigma2[t] = var_bounds[t, 0]
        elif sigma2[t] > var_bounds[t, 1]:
            if sigma2[t] > DBL_MAX:
                sigma2[t] = var_bounds[t, 1] + 1000
            else:
                sigma2[t] = var_bounds[t, 1] + log(sigma2[t] / var_bounds[t, 1])

    return np.asarray(sigma2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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

        if sigma2[t] < var_bounds[t, 0]:
            sigma2[t] = var_bounds[t, 0]
        elif sigma2[t] > var_bounds[t, 1]:
            if sigma2[t] > DBL_MAX:
                sigma2[t] = var_bounds[t, 1] + 1000
            else:
                sigma2[t] = var_bounds[t, 1] + log(sigma2[t] / var_bounds[t, 1])

    return np.asarray(sigma2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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

        if sigma2[t] < var_bounds[t, 0]:
            sigma2[t] = var_bounds[t, 0]
        elif sigma2[t] > var_bounds[t, 1]:
            if sigma2[t] > DBL_MAX:
                sigma2[t] = var_bounds[t, 1] + 1000
            else:
                sigma2[t] = var_bounds[t, 1] + log(sigma2[t] / var_bounds[t, 1])

    return np.asarray(sigma2)
