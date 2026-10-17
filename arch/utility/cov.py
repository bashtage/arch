from numpy import asarray, squeeze

from arch._typing import Float64Array
from arch.utility._validation import (
    _describe,
    _fail,
    check_array,
    check_bool,
    check_int,
)


def cov_nw(
    y: Float64Array, lags: int = 0, demean: bool = True, axis: int = 0, ddof: int = 0
) -> Float64Array | float:
    """
    Compute the Newey-West covariance for 1-d and 2-d arrays.

    Parameters
    ----------
    y : array_like
        Values to use when computing the Newey-West covariance estimator, either
        1-d or 2-d. When y is 2d, default behavior is to treat columns as variables
        and rows as observations. Integer and Boolean data are converted to float64.
    lags : int, optional
        Number of lags to include in the Newey-West covariance estimator.
    demean : bool, optional
        Indicates whether to subtract the mean.
    axis : int, optional
        The axis to use when y is 2d.
    ddof : int, optional
        Degree of freedom correction for compatibility with simple covariance
        estimators.

    Returns
    -------
    float or ndarray
        The estimated covariance. A float if y is 1-d and a 2-d ndarray
        otherwise.
    """
    z = check_array("y", y)
    if z.ndim not in (1, 2) or z.size == 0:
        _fail(
            ValueError,
            "y",
            "a non-empty 1-d or 2-d array of finite numbers",
            _describe(y),
        )
    axis = check_int("axis", axis, lower=-z.ndim, upper=z.ndim - 1)
    is_1d = False
    axis %= z.ndim
    if z.ndim == 1:
        is_1d = True
        z = asarray(z)[:, None]
    if axis == 1:
        z = z.T
    n = z.shape[0]
    lags = check_int("lags", lags, lower=0, upper=n)
    demean = check_bool("demean", demean)
    ddof = check_int("ddof", ddof, lower=0, upper=n - 1)

    if demean:
        z = z - z.mean(0)
    cov = z.T.dot(z)
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)
        gamma = z[j:].T.dot(z[:-j])
        cov += w * (gamma + gamma.T)
    cov = cov / (n - ddof)
    if is_1d:
        return float(squeeze(cov))
    return asarray(cov)
