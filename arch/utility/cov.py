from typing import Union

from numpy import asarray, squeeze

from arch.typing import Float64Array


def cov_nw(
    y: Float64Array, lags: int = 0, demean: bool = True, axis: int = 0, ddof: int = 0
) -> Union[Float64Array, float]:
    """
    Computes Newey-West covariance for 1-d and 2-d arrays

    Parameters
    ----------
    y : ndarray
        Values to use when computing the Newey-West covariance estimator, either
        1-d or 2-d. When y is 2d, default behavior is to treat columns as variables
        and rows as observations.
    lags : int
        Number of lags to include in the Newey-West covariance estimator
    demean : bool
        Indicates whether to subtract the mean.  Default is True
    axis : int
        The axis to use when y is 2d
    ddof : int
        Degree of freedom correction for compatibility with simple covariance
        estimators.  Default is 0.

    Returns
    -------
    ndarray
        The estimated covariance
    """
    z = y
    is_1d = False
    if axis > z.ndim:
        raise ValueError("axis must be less than the dimension of y")
    if z.ndim == 1:
        is_1d = True
        z = asarray(z)[:, None]
    if axis == 1:
        z = z.T
    n = z.shape[0]
    if ddof > n:
        raise ValueError(
            "ddof must be strictly smaller than the number of observations"
        )
    if lags > n:
        error = "lags must be weakly smaller than the number of observations"
        raise ValueError(error)

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
