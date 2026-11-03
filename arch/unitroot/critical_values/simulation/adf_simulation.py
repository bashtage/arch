import pathlib
import platform

from numpy import arange, array, cumsum, dot, ndarray, ones, sqrt, vstack
from numpy.linalg import pinv
from numpy.random import Generator, RandomState

from arch._typing import UnitRootTrend

# Storage Location
if platform.system() == "Linux":
    BASE_PATH = pathlib.Path("/mnt", "c")
else:
    BASE_PATH = pathlib.Path(r"C:\\")
# Created by the scripts that write to them, and not when this module is imported
OUTPUT_PATH = BASE_PATH / "Users" / "kevin" / "Dropbox" / "adf-z"
T_OUTPUT_PATH = BASE_PATH / "Users" / "kevin" / "Dropbox" / "adf-t"

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
    rng: RandomState | Generator | None = None,
) -> ndarray:
    """
    Simulates the empirical distribution of the ADF z-test statistic
    """
    return adf_statistics(n, trend, b, rng)[0]


def adf_statistics(
    n: int,
    trend: UnitRootTrend,
    b: int,
    rng: RandomState | Generator | None = None,
) -> tuple[ndarray, ndarray]:
    """
    Simulates the empirical distributions of the ADF z- and t-test statistics

    Both statistics are computed from the same b random walks of length n, and
    are those of ADF with lags=0 and the same trend.

    Returns
    -------
    z_stat : ndarray
        The z-test statistics, nobs * (gamma - 1).
    t_stat : ndarray
        The t-test statistics, (gamma - 1) / se(gamma).
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

    y = standard_normal((n, b))
    y = cumsum(y, axis=0)
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
    z_stat = nobs * (gamma - 1.0)
    # The residual variance is corrected for the lagged level and the trend terms,
    # as in the OLS regression of ADF
    num_params = 1 if z is None else 1 + z.shape[1]
    sigma2 = sum((lhs - gamma * rhs) ** 2.0, 0) / (nobs - num_params)
    t_stat = (gamma - 1.0) / sqrt(sigma2 / xpx)
    return z_stat, t_stat
