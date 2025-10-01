import os
import platform

from numpy import arange, array, cumsum, dot, ones, vstack
from numpy.linalg import pinv
from numpy.random import Generator, RandomState

from arch._typing import UnitRootTrend

# Storage Location
if platform.system() == "Linux":
    BASE_PATH = os.path.join("/mnt", "c")
else:
    BASE_PATH = "C:\\\\"
OUTPUT_PATH = os.path.join(BASE_PATH, "Users", "kevin", "Dropbox", "adf-z")

_PERCENTILES = (
    list(arange(1, 10))
    + list(arange(10, 50, 5))
    + list(arange(50, 950, 10))
    + list(arange(950, 990, 5))
    + list(arange(990, 999))
)
PERCENTILES = array(_PERCENTILES) / 10.0

TRENDS = ("n", "c", "ct", "ctt")
TIME_SERIES_LENGTHS = array(
    (
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        60,
        70,
        80,
        90,
        100,
        120,
        140,
        160,
        180,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        600,
        700,
        800,
        900,
        1000,
        1200,
        1400,
        2000,
    )
)


def adf_simulation(
    n: int,
    trend: UnitRootTrend,
    b: int,
    rng: None | RandomState | Generator = None,
) -> float:
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
        z = vstack((ones(nobs), tau, tau**2.0)).T

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
    xpx = sum(rhs**2.0, 0)
    gamma = xpy / xpx
    nobs = lhs.shape[0]
    stat = nobs * (gamma - 1.0)
    return stat
