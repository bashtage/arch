from numpy import arange, cumsum, dot, ones, vstack
from numpy.linalg import pinv
from numpy.random import RandomState


def adf_simulation(n, trend, b, rng=None):
    """
    Simulates the empirical distribution of the ADF z-test statistic
    """
    if rng is None:
        rng = RandomState(0)
    standard_normal = rng.standard_normal

    nobs = n - 1
    z = None
    if trend == "c":
        z = ones((nobs, 1))
    elif trend == "ct":
        z = vstack((ones(nobs), arange(1, nobs + 1))).T
    elif trend == "ctt":
        tau = arange(1, nobs + 1)
        z = vstack((ones(nobs), tau, tau ** 2.0)).T

    y = standard_normal((n + 50, b))
    y = cumsum(y, axis=0)
    y = y[50:, :]
    lhs = y[1:, :]
    rhs = y[:-1, :]
    if z is not None:
        z_inv = pinv(z)
        beta = dot(z_inv, lhs)
        lhs = lhs - dot(z, beta)
        beta = dot(z_inv, rhs)
        rhs = rhs - dot(z, beta)

    xpy = sum(rhs * lhs, 0)
    xpx = sum(rhs ** 2.0, 0)
    gamma = xpy / xpx
    nobs = lhs.shape[0]
    stat = nobs * (gamma - 1.0)
    return stat
