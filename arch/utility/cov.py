from numpy import asarray, iscomplexobj, real, squeeze

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
        and rows as observations. Integer, Boolean and float32 data are converted to
        float64, and complex data to complex128. float16 data are not accepted.
        Must not contain NaN or infinite values.
    lags : int, optional
        Number of lags to include in the Newey-West covariance estimator. Must be
        non-negative and at most the number of observations.
    demean : bool, optional
        Indicates whether to subtract the mean.
    axis : int, optional
        The axis to use when y is 2d.
    ddof : int, optional
        Degree of freedom correction for compatibility with simple covariance
        estimators. Must be non-negative and smaller than the number of
        observations.

    Returns
    -------
    float or ndarray
        The estimated covariance. A float if y is 1-d and a 2-d ndarray
        otherwise. The covariance of complex data is Hermitian, so that it is
        real if y is 1-d and a complex128 matrix if y is 2-d.

    Raises
    ------
    TypeError
        If y cannot be converted to float64 or complex128 (it is float16, or
        contains strings or objects that are not numbers) or if lags, demean, axis
        or ddof have the wrong type. Booleans and floats are not accepted for the
        integers.
    ValueError
        If y is not 1-d or 2-d, is empty or contains NaN or infinite values, or if
        lags, axis or ddof are outside their range.

    Notes
    -----
    The covariance of complex data is the Hermitian covariance, which uses the
    conjugate transpose: the term at lag j is the sum of z_t times the conjugate
    of z_{t-j}, and a term and its conjugate transpose are added for each lag. It
    is the covariance of the Bartlett kernel estimator in
    :mod:`arch.covariance.kernel`. The pseudo-covariance is not estimated.
    """
    z = check_array("y", y, allow_complex=True)
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
    # The Hermitian covariance has the conjugate in the second term of the products
    is_complex = iscomplexobj(z)
    z_conj = z.conj() if is_complex else z
    cov = z.T.dot(z_conj)
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)
        gamma = z[j:].T.dot(z_conj[:-j])
        cov += w * (gamma + (gamma.conj().T if is_complex else gamma.T))
    cov = cov / (n - ddof)
    if is_complex:
        # Each element of the matrix product is rounded on its own
        cov = (cov + cov.conj().T) / 2
    if is_1d:
        # The variance of a complex series is real
        return float(real(squeeze(cov)))
    return asarray(cov)
