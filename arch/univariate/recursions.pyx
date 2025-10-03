#!python

import numpy as np

cimport numpy as np
from libc.float cimport DBL_MAX
from libc.math cimport exp, fabs, lgamma, log, sqrt

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
    cdef double omega, alpha, gamma
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
        sigma2[t] = omega
        for i in range(m):
            if (t - i - 1) >= 0:
                sigma2[t] += (
                        (aw[i] + gw[i] * (resids[t - i - 1] < 0)) * resids2[t - i - 1]
                )
            else:
                sigma2[t] += (aw[i] + 0.5 * gw[i]) * backcast

        bounds_check(&sigma2[t], &var_bounds[t, 0])

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


def figarch_recursion(const double[::1] parameters,
                      const double[::1] fresids,
                      double[::1] sigma2,
                      int p,
                      int q,
                      int nobs,
                      int trunc_lag,
                      double backcast,
                      double[:, ::1] var_bounds):
    cdef Py_ssize_t t, i
    cdef double bc_weight, omega, beta, omega_tilde
    cdef double [::1] lam

    omega = parameters[0]
    beta = parameters[1 + p + q] if q else 0.0
    omega_tilde = omega / (1 - beta)
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

    def __init__(self, int p, int o, int q, double power):
        self.p = p
        self.o = o
        self.q = q
        self.power = power
        self.backcast = -1.0

    def initialize_update(
            self,
            const double[::1] parameters,
            object backcast,
            Py_ssize_t nobs
    ):
        self.backcast = backcast

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

    def __init__(self, int p, int q, double power, int truncation):
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
        cdef double bc_weight, omega, beta, omega_tilde
        cdef int p = self.p, q = self.q, trunc_lag = self.truncation

        omega = parameters[0]
        beta = parameters[1 + p + q] if q else 0.0
        omega_tilde = omega / (1 - beta)

        if t > 0:
            self.fresids[t-1] = fabs(resids[t-1]) ** self.power

        bc_weight = 0.0
        for i in range(t, trunc_lag):
            bc_weight += self.lam[i]
        sigma2[t] = omega_tilde + bc_weight * self.backcast
        for i in range(min(t, trunc_lag)):
            sigma2[t] += self.lam[i] * self.fresids[t - i - 1]
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

    def __init__(self, VolatilityUpdater updater):
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
            Py_ssize_t t, i, nobs = y.shape[0], k = x.shape[1]
            double[::1] resids = np.empty(nobs)
            double gamma = mean_parameters[k]

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
