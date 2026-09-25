"""
Residual diagnostics for univariate ARCH models.
"""

from collections.abc import Sequence

import numpy as np
from scipy import stats

from arch._typing import ArrayLike1D
from arch.univariate.distribution import Distribution, Normal
from arch.utility.array import ensure1d

__all__ = ["excess_kurtosis", "hill_estimator", "var_ratio"]


def _clean_std_resid(std_resid: Sequence[float] | ArrayLike1D) -> np.ndarray:
    resids = np.asarray(ensure1d(std_resid, "std_resid", series=False), dtype=float)
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
        computing the statistic.

    Returns
    -------
    float
        Sample excess kurtosis computed using sample central moments.
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
        absolute standardized residuals after removing non-finite observations.
    k : int, optional
        Number of upper order statistics to use. If not provided, uses
        ``int(sqrt(n))`` where ``n`` is the number of finite residuals.

    Returns
    -------
    float
        Inverse Hill estimate of the tail index.

    Notes
    -----
    The estimator is

    .. math::

        \hat{\nu} = \left(k^{-1}\sum_{i=1}^k
        \log X_{n-i+1:n} - \log X_{n-k:n}\right)^{-1}

    where ``X`` contains the positive absolute standardized residuals.
    """
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
    elif not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    if not 1 <= int(k) < positive:
        raise ValueError(
            "k must satisfy 1 <= k < the number of non-zero finite observations"
        )
    k = int(k)

    threshold = resids[-k - 1]
    hill = np.mean(np.log(resids[-k:]) - np.log(threshold))
    if hill == 0.0:
        return np.inf
    return float(1.0 / hill)


def var_ratio(
    level: float = 0.999,
    distribution: Distribution | None = None,
    parameters: Sequence[float] | ArrayLike1D | None = None,
) -> float:
    """
    Ratio of Gaussian VaR to model-distribution VaR.

    Parameters
    ----------
    level : float, optional
        VaR confidence level. Must be between 0.5 and 1.0.
    distribution : Distribution, optional
        Standardized distribution used to compute the model VaR. If not
        provided, uses the standard normal distribution.
    parameters : array_like, optional
        Distribution parameters. Use ``None`` for parameterless distributions.

    Returns
    -------
    float
        Ratio of the Gaussian left-tail VaR to the VaR implied by the supplied
        model distribution.
    """
    level = float(level)
    if not 0.5 < level < 1.0:
        raise ValueError("level must be larger than 0.5 and smaller than 1.0")

    if distribution is None:
        distribution = Normal()
    elif not isinstance(distribution, Distribution):
        raise TypeError("distribution must inherit from Distribution")

    tail_probability = 1.0 - level
    gaussian_var = -stats.norm.ppf(tail_probability)
    model_var = -float(distribution.ppf(tail_probability, parameters))
    if not np.isfinite(model_var) or model_var <= 0.0:
        raise ValueError("model distribution must produce a positive finite VaR")
    return float(gaussian_var / model_var)
