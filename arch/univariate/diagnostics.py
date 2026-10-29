"""
Residual diagnostics for univariate ARCH models.
"""

from collections.abc import Sequence

import numpy as np
from scipy import stats

from arch._typing import ArrayLike1D
from arch.univariate.distribution import Distribution, Normal
from arch.utility._validation import (
    check_array,
    check_float,
    check_instance,
    check_int,
)

__all__ = ["excess_kurtosis", "hill_estimator", "var_ratio"]


def _clean_std_resid(std_resid: Sequence[float] | ArrayLike1D) -> np.ndarray:
    # NaN and inf are removed by design, and so they are accepted
    resids = check_array("std_resid", std_resid, ndim=1, allow_nan=True, allow_inf=True)
    resids = resids[np.isfinite(resids)]
    if resids.shape[0] == 0:
        raise ValueError("std_resid must contain at least one finite observation")
    return resids


def excess_kurtosis(std_resid: Sequence[float] | ArrayLike1D) -> float:
    """
    Sample excess kurtosis of standardized residuals.

    Parameters
    ----------
    std_resid : array_like
        Standardized residuals. Non-finite observations are removed before
        computing the statistic, since the residuals of a fitted model are NaN
        outside the estimation sample.

    Returns
    -------
    float
        Sample excess kurtosis computed using sample central moments.

    Raises
    ------
    TypeError
        If ``std_resid`` cannot be converted to numbers.
    ValueError
        If ``std_resid`` is not 1-d or has no finite observation.
    """
    resids = _clean_std_resid(std_resid)
    demeaned = resids - resids.mean()
    variance = np.mean(demeaned**2.0)
    if variance == 0.0:
        return np.nan
    return float(np.mean(demeaned**4.0) / variance**2.0 - 3.0)


def hill_estimator(
    std_resid: Sequence[float] | ArrayLike1D, k: int | None = None
) -> float:
    r"""
    Hill tail-index estimator of standardized residuals.

    Parameters
    ----------
    std_resid : array_like
        Standardized residuals. The estimator is computed from the largest
        absolute standardized residuals after removing non-finite observations,
        since the residuals of a fitted model are NaN outside the estimation
        sample.
    k : int, optional
        Number of upper order statistics to use. If not provided, uses
        ``int(sqrt(n))`` where ``n`` is the number of finite residuals. Must be at
        least 1 and less than the number of non-zero finite residuals.

    Returns
    -------
    float
        Inverse Hill estimate of the tail index.

    Raises
    ------
    TypeError
        If ``std_resid`` cannot be converted to numbers or ``k`` is not an
        integer.
    ValueError
        If ``std_resid`` is not 1-d, has fewer than two non-zero finite
        observations, or ``k`` is outside the allowed range.

    Notes
    -----
    The estimator is

    .. math::

        \hat{\nu} = \left(k^{-1}\sum_{i=1}^k
        \log X_{n-i+1:n} - \log X_{n-k:n}\right)^{-1}

    where ``X`` contains the positive absolute standardized residuals.
    """
    k = check_int("k", k, lower=1, allow_none=True)
    abs_resids = np.abs(_clean_std_resid(std_resid))
    nobs = abs_resids.shape[0]
    resids = np.sort(abs_resids[abs_resids > 0.0])
    positive = resids.shape[0]
    if positive < 2:
        raise ValueError(
            "std_resid must contain at least two non-zero finite observations"
        )

    if k is None:
        k = int(np.sqrt(nobs))
    # k is at most the number of non-zero finite observations less one
    k = check_int("k", k, lower=1, upper=positive - 1)

    threshold = resids[-k - 1]
    hill = np.mean(np.log(resids[-k:]) - np.log(threshold))
    if hill == 0.0:
        return np.inf
    return float(1.0 / hill)


def _tail_probability(p: float | None) -> float:
    """
    The probability of an exception, 0.001 if ``p`` is None.
    """
    if p is None:
        return 0.001
    return check_float("p", p, lower=0.0, upper=0.5, inclusive="neither")


def var_ratio(
    p: float | None = None,
    distribution: Distribution | None = None,
    parameters: Sequence[float] | ArrayLike1D | None = None,
) -> float:
    r"""
    Ratio of Gaussian VaR to model-distribution VaR.

    Parameters
    ----------
    p : float, optional
        The probability of an exception that the VaR is designed for, which is
        the size of the tail, in the open interval (0, 0.5). It is 0.05 for a 95%
        VaR and 0.01 for a 99% VaR. The default is 0.001, a 99.9% VaR.
    distribution : Distribution, optional
        Standardized distribution used to compute the model VaR. If not
        provided, uses the standard normal distribution.
    parameters : array_like, optional
        Distribution parameters. Must have one finite element for each parameter
        of ``distribution``. Use ``None`` for parameterless distributions.

    Returns
    -------
    float
        Ratio of the Gaussian left-tail VaR to the VaR implied by the supplied
        model distribution.

    Raises
    ------
    TypeError
        If ``p`` is not a real number, ``distribution`` is not a Distribution or
        ``parameters`` cannot be converted to numbers.
    ValueError
        If ``p`` is not in (0, 0.5), ``parameters`` does not have one finite
        element for each parameter of the distribution, or the model distribution
        does not produce a positive finite VaR.

    Notes
    -----
    VaR is named by its confidence level, and ``p`` is the probability of an
    exception, as in the VaR backtests such as :func:`~arch.univariate.kupiec_pof`.
    The ratio is :math:`\Phi^{-1}(p)/F^{-1}(p)` where :math:`F` is the
    standardized distribution.
    """
    p = _tail_probability(p)
    distribution = check_instance(
        "distribution", distribution, Distribution, allow_none=True
    )
    if distribution is None:
        distribution = Normal()
    nparams = distribution.num_params
    parameters = check_array(
        "parameters",
        parameters,
        ndim=1,
        shape=(nparams,),
        allow_none=nparams == 0,
    )

    gaussian_var = -stats.norm.ppf(p)
    model_var = -float(distribution.ppf(p, parameters))
    if not np.isfinite(model_var) or model_var <= 0.0:
        raise ValueError("model distribution must produce a positive finite VaR")
    return float(gaussian_var / model_var)
