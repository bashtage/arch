"""
Backtests of Value-at-Risk (VaR) forecasts.

The tests compare a series of VaR forecasts with the returns that were realized.
VaR is reported as a positive loss, and an observation is an exception if the loss
exceeds the VaR, ``-returns > var``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from arch._typing import ArrayLike1D, BoolArray, Float64Array1D
from arch.utility._validation import check_array, check_float, check_int
from arch.utility.testing import WaldTestStatistic

__all__ = [
    "TrafficLightResult",
    "christoffersen_conditional_coverage",
    "christoffersen_independence",
    "kupiec_pof",
    "traffic_light",
    "var_exceptions",
]

# The cumulative probabilities that begin the yellow and the red zone (BCBS, 1996)
_YELLOW_PROBABILITY = 0.95
_RED_PROBABILITY = 0.9999


def _aligned(
    returns: ArrayLike1D | list[float],
    var: ArrayLike1D | list[float],
) -> tuple[Float64Array1D, Float64Array1D]:
    """Validate returns and var, and align them (on the index if both are Series)."""
    ret = check_array("returns", returns, ndim=1, min_length=1)
    var_arr = check_array("var", var, ndim=1, min_length=1)
    if isinstance(returns, pd.Series) and isinstance(var, pd.Series):
        if not returns.index.equals(var.index):
            if not (returns.index.is_unique and var.index.is_unique):
                raise ValueError(
                    "returns and var must have an index without duplicates to be "
                    "aligned on the index when the indices differ; got a duplicated "
                    f"index ({type(returns).__name__})"
                )
            # Keep the order of returns, which is the order of time
            pos = np.sort(
                returns.index.get_indexer(var.index.intersection(returns.index))
            )
            ret = ret[pos]
            var_arr = var_arr[var.index.get_indexer(returns.index[pos])]
    elif ret.shape[0] != var_arr.shape[0]:
        raise ValueError(
            f"var must be an array with the same length as returns ({ret.shape[0]}) "
            f"or aligned with returns on the index; got length {var_arr.shape[0]} "
            f"({type(var).__name__})"
        )
    return ret, var_arr


def _exceptions(
    returns: ArrayLike1D | list[float],
    var: ArrayLike1D | list[float],
    min_obs: int,
) -> BoolArray:
    ret, var_arr = _aligned(returns, var)
    nobs = ret.shape[0]
    if nobs < min_obs:
        raise ValueError(
            f"returns must be aligned with var on at least {min_obs} observation"
            f"{'' if min_obs == 1 else 's'}; got {nobs} (int)"
        )
    return -ret > var_arr


def _lr_pof(exceptions: int, nobs: int, p: float) -> float:
    """
    Kupiec proportion-of-failures statistic.

    ``2 * [x log(x / (T p)) + (T - x) log((T - x) / (T (1 - p)))]`` where
    ``xlogy`` gives 0 log 0 = 0, so that no exceptions and all exceptions are
    handled by the limit of the likelihood.
    """
    failures = float(exceptions)
    successes = float(nobs - exceptions)
    total = float(nobs)
    stat = 2.0 * (
        xlogy(failures, failures / (total * p))
        + xlogy(successes, successes / (total * (1.0 - p)))
    )
    return max(float(stat), 0.0)


def _lr_independence(exceptions: BoolArray) -> float:
    """Christoffersen (1998) first-order Markov independence statistic."""
    previous = exceptions[:-1]
    current = exceptions[1:]
    n11 = float(np.sum(previous & current))
    n10 = float(np.sum(previous & ~current))
    n01 = float(np.sum(~previous & current))
    n00 = float(np.sum(~previous & ~current))
    from_clear = n00 + n01
    from_exception = n10 + n11
    pooled = (n01 + n11) / (from_clear + from_exception)
    # A conditional probability is only used when the state was visited, and then its
    # counts are multiplied by 0 otherwise, which xlogy treats as 0 log 0 = 0.
    pi01 = n01 / from_clear if from_clear > 0.0 else 0.0
    pi11 = n11 / from_exception if from_exception > 0.0 else 0.0
    markov = (
        xlogy(n00, 1.0 - pi01)
        + xlogy(n01, pi01)
        + xlogy(n10, 1.0 - pi11)
        + xlogy(n11, pi11)
    )
    iid = xlogy(n00 + n10, 1.0 - pooled) + xlogy(n01 + n11, pooled)
    return max(float(2.0 * (markov - iid)), 0.0)


def var_exceptions(
    returns: ArrayLike1D | list[float],
    var: ArrayLike1D | list[float],
) -> BoolArray:
    r"""
    Identify the exceptions of a VaR forecast.

    Parameters
    ----------
    returns : array_like
        Realized returns, a 1-d array of finite numbers.
    var : array_like
        VaR forecasts, a 1-d array of finite numbers. VaR is a positive loss,
        so that a VaR of 2.5 forecasts that the return is below -2.5 with the
        probability that the VaR is designed for. If both ``returns`` and ``var``
        are Series, they are aligned on the index and the observations that are
        in both are used, in the order of ``returns``. Otherwise, they must have
        the same length and are matched by position.

    Returns
    -------
    ndarray
        A bool array that is True where the return is an exception, that is
        where the loss exceeds the VaR, ``-returns > var``.

    Raises
    ------
    TypeError
        If ``returns`` or ``var`` cannot be converted to numbers.
    ValueError
        If ``returns`` or ``var`` is not a 1-d array of finite numbers (NaN and
        inf raise), has no observations, has an index with duplicates that
        differs from the index of the other Series, or does not have the same
        length as the other input.

    Notes
    -----
    The sign convention is the same for all the functions of this module. VaR is
    reported as a positive loss, so that a 1% VaR of 2.5 is the loss that is
    exceeded with probability 1%, and an observation is an exception if
    :math:`-r_t > \mathrm{VaR}_t`. A quantile of the distribution of returns,
    which is negative for the lower tail, must be negated first. A loss that
    equals the VaR is not an exception.
    """
    return _exceptions(returns, var, 1)


def kupiec_pof(
    returns: ArrayLike1D | list[float],
    var: ArrayLike1D | list[float],
    p: float,
) -> WaldTestStatistic:
    r"""
    Kupiec proportion-of-failures test of the coverage of VaR.

    Parameters
    ----------
    returns : array_like
        Realized returns, a 1-d array of finite numbers.
    var : array_like
        VaR forecasts as positive losses, a 1-d array of finite numbers. See
        `var_exceptions` for the sign convention and for how the inputs are
        aligned.
    p : float
        The probability of an exception that the VaR is designed for, in the open
        interval (0, 1). It is 0.01 for a 99% VaR.

    Returns
    -------
    arch.utility.testing.WaldTestStatistic
        The likelihood-ratio statistic, which is asymptotically
        :math:`\chi^2_1` under the null that the probability of an exception is
        ``p``, with its p-value.

    Raises
    ------
    TypeError
        If ``returns`` or ``var`` cannot be converted to numbers, or ``p`` is
        not a real number.
    ValueError
        If ``returns`` or ``var`` is not valid, as in `var_exceptions`, or ``p``
        is not in (0, 1).

    See Also
    --------
    christoffersen_independence : Test of the independence of the exceptions.
    christoffersen_conditional_coverage : Joint test of coverage and independence.
    traffic_light : The Basel zone of the number of exceptions.

    Notes
    -----
    With :math:`x` exceptions in :math:`T` observations and
    :math:`\hat{\pi}=x/T`, the statistic of Kupiec (1995) [1]_ is

    .. math::

        LR_{uc} = -2\ln\left[\frac{(1-p)^{T-x}p^x}
                  {(1-\hat{\pi})^{T-x}\hat{\pi}^x}\right].

    The test is two-sided, so too few exceptions reject as well as too many.
    The terms :math:`0\ln 0` of a series with no exception, or only exceptions,
    are 0, which is the limit of the likelihood, so that the statistic is
    :math:`-2T\ln(1-p)` and :math:`-2T\ln p`.

    References
    ----------
    .. [1] Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk
       measurement models. The Journal of Derivatives, 3(2), 73-84.
    """
    p = check_float("p", p, lower=0.0, upper=1.0, inclusive="neither")
    exceptions = _exceptions(returns, var, 1)
    stat = _lr_pof(int(exceptions.sum()), exceptions.shape[0], p)
    return WaldTestStatistic(
        stat,
        df=1,
        null=f"The probability of an exception is {p:g}.",
        alternative=f"The probability of an exception is not {p:g}.",
        name="Kupiec Proportion of Failures Test",
    )


def christoffersen_independence(
    returns: ArrayLike1D | list[float],
    var: ArrayLike1D | list[float],
) -> WaldTestStatistic:
    r"""
    Christoffersen test of the independence of the exceptions of VaR.

    Parameters
    ----------
    returns : array_like
        Realized returns, a 1-d array of finite numbers with at least 2
        observations.
    var : array_like
        VaR forecasts as positive losses, a 1-d array of finite numbers. See
        `var_exceptions` for the sign convention and for how the inputs are
        aligned.

    Returns
    -------
    arch.utility.testing.WaldTestStatistic
        The likelihood-ratio statistic, which is asymptotically
        :math:`\chi^2_1` under the null that exceptions are independent, with its
        p-value.

    Raises
    ------
    TypeError
        If ``returns`` or ``var`` cannot be converted to numbers.
    ValueError
        If ``returns`` or ``var`` is not valid, as in `var_exceptions`, or there
        are fewer than 2 observations.

    See Also
    --------
    kupiec_pof : Test of the proportion of exceptions.
    christoffersen_conditional_coverage : Joint test of coverage and independence.

    Notes
    -----
    The null is that the probability of an exception does not depend on whether
    the previous day was an exception, against a first-order Markov chain. With
    :math:`n_{ij}` the number of days in state :math:`j` that follow a day in
    state :math:`i` (1 is an exception), :math:`\hat{\pi}_{i}=n_{i1}/(n_{i0}+n_{i1})`
    and :math:`\hat{\pi}=(n_{01}+n_{11})/\sum n_{ij}`, the statistic of
    Christoffersen (1998) [1]_ is

    .. math::

        LR_{ind} = -2\ln\left[\frac{(1-\hat{\pi})^{n_{00}+n_{10}}
                   \hat{\pi}^{n_{01}+n_{11}}}
                   {(1-\hat{\pi}_0)^{n_{00}}\hat{\pi}_0^{n_{01}}
                   (1-\hat{\pi}_1)^{n_{10}}\hat{\pi}_1^{n_{11}}}\right].

    The first observation is conditioned on, so the :math:`T-1` transitions are
    used. A term :math:`0\ln 0` is 0. If no day follows a state, for example when
    there is no exception before the last day, :math:`\hat{\pi}_i` is not defined
    and does not enter the statistic. The statistic is 0 for a series with no
    exceptions or only exceptions. A probability that is 0, because no exception
    follows an exception (:math:`n_{11}=0`), is evaluated at the limit.

    References
    ----------
    .. [1] Christoffersen, P. F. (1998). Evaluating interval forecasts.
       International Economic Review, 39(4), 841-862.
    """
    exceptions = _exceptions(returns, var, 2)
    return WaldTestStatistic(
        _lr_independence(exceptions),
        df=1,
        null="Exceptions are independent over time.",
        alternative="Exceptions follow a first-order Markov chain.",
        name="Christoffersen Independence Test",
    )


def christoffersen_conditional_coverage(
    returns: ArrayLike1D | list[float],
    var: ArrayLike1D | list[float],
    p: float,
) -> WaldTestStatistic:
    r"""
    Christoffersen test of the conditional coverage of VaR.

    Parameters
    ----------
    returns : array_like
        Realized returns, a 1-d array of finite numbers with at least 2
        observations.
    var : array_like
        VaR forecasts as positive losses, a 1-d array of finite numbers. See
        `var_exceptions` for the sign convention and for how the inputs are
        aligned.
    p : float
        The probability of an exception that the VaR is designed for, in the open
        interval (0, 1). It is 0.01 for a 99% VaR.

    Returns
    -------
    arch.utility.testing.WaldTestStatistic
        The likelihood-ratio statistic, which is asymptotically
        :math:`\chi^2_2` under the null that exceptions are independent and have
        probability ``p``, with its p-value.

    Raises
    ------
    TypeError
        If ``returns`` or ``var`` cannot be converted to numbers, or ``p`` is
        not a real number.
    ValueError
        If ``returns`` or ``var`` is not valid, as in `var_exceptions`, there
        are fewer than 2 observations, or ``p`` is not in (0, 1).

    See Also
    --------
    kupiec_pof : Test of the proportion of exceptions.
    christoffersen_independence : Test of the independence of the exceptions.

    Notes
    -----
    The statistic of Christoffersen (1998) [1]_ is the sum of the Kupiec statistic,
    computed with all the observations, and the independence statistic,
    :math:`LR_{cc}=LR_{uc}+LR_{ind}`. See `kupiec_pof` and
    `christoffersen_independence` for the statistics and for the handling of
    series with no exceptions or with only exceptions.

    References
    ----------
    .. [1] Christoffersen, P. F. (1998). Evaluating interval forecasts.
       International Economic Review, 39(4), 841-862.
    """
    p = check_float("p", p, lower=0.0, upper=1.0, inclusive="neither")
    exceptions = _exceptions(returns, var, 2)
    stat = _lr_pof(int(exceptions.sum()), exceptions.shape[0], p)
    stat += _lr_independence(exceptions)
    return WaldTestStatistic(
        stat,
        df=2,
        null=f"Exceptions are independent and have probability {p:g}.",
        alternative=(
            f"The probability of an exception is not {p:g}, or exceptions follow "
            "a first-order Markov chain."
        ),
        name="Christoffersen Conditional Coverage Test",
    )


@dataclass(frozen=True)
class TrafficLightResult:  # numpydoc ignore=PR02
    """
    Result of the Basel traffic light backtest of VaR.

    Parameters
    ----------
    zone : {"green", "yellow", "red"}
        The Basel zone of the number of exceptions.
    exceptions : int
        The number of exceptions in the window.
    probability : float
        The cumulative binomial probability of ``exceptions`` or fewer exceptions
        if the probability of an exception is ``p``.
    nobs : int
        The number of observations in the window.
    p : float
        The probability of an exception that the VaR is designed for.

    Attributes
    ----------
    zone : {"green", "yellow", "red"}
        The Basel zone of the number of exceptions.
    exceptions : int
        The number of exceptions in the window.
    probability : float
        The cumulative binomial probability of ``exceptions`` or fewer exceptions
        if the probability of an exception is ``p``.
    nobs : int
        The number of observations in the window.
    p : float
        The probability of an exception that the VaR is designed for.
    """

    zone: str
    exceptions: int
    probability: float
    nobs: int
    p: float


def traffic_light(
    returns: ArrayLike1D | list[float],
    var: ArrayLike1D | list[float],
    p: float = 0.01,
    window: int = 250,
) -> TrafficLightResult:
    r"""
    Basel traffic light zone of the exceptions of VaR.

    Parameters
    ----------
    returns : array_like
        Realized returns, a 1-d array of finite numbers with at least ``window``
        observations.
    var : array_like
        VaR forecasts as positive losses, a 1-d array of finite numbers. See
        `var_exceptions` for the sign convention and for how the inputs are
        aligned.
    p : float, optional
        The probability of an exception that the VaR is designed for, in the open
        interval (0, 1). The default is 0.01, which is a 99% VaR.
    window : int, optional
        The number of the most recent observations that are counted. It must be
        positive and no larger than the number of observations. The default is
        250, about one year of daily data.

    Returns
    -------
    TrafficLightResult
        The zone, the number of exceptions in the window, the cumulative binomial
        probability of that number of exceptions or fewer, the number of
        observations in the window and ``p``.

    Raises
    ------
    TypeError
        If ``returns`` or ``var`` cannot be converted to numbers, ``p`` is not a
        real number or ``window`` is not an integer.
    ValueError
        If ``returns`` or ``var`` is not valid, as in `var_exceptions`, ``p`` is
        not in (0, 1), ``window`` is not positive, or there are fewer than
        ``window`` observations.

    See Also
    --------
    kupiec_pof : Test of the proportion of exceptions.

    Notes
    -----
    The framework of the Basel Committee on Banking Supervision (1996) [1]_ counts the
    exceptions in the last 250 days. If the probability of an exception is ``p``,
    the number of exceptions has a binomial distribution, and the cumulative
    probability :math:`P(X\leq x)` of the observed number :math:`x` sets the zone.
    The zone is green if the probability is below 95%, yellow if it is at least 95%
    and below 99.99%, and red if it is at least 99.99%. For 250 observations and
    ``p=0.01``, green is 0 to 4 exceptions, yellow is 5 to 9 and red is 10 or more,
    which are the boundaries of Table 2 of the framework. The framework
    deduces the boundaries for other sample sizes from the binomial distribution
    in the same way, as this function does.

    A series with no exception is always green. Its cumulative probability is the
    probability of the smallest number of exceptions, which is above 95% when
    ``window * p`` is small, but no exception is never a sign of understated risk.

    A series with fewer than ``window`` observations raises an error rather than
    being tested with the observations that there are, because the zones of the
    framework are defined for a sample of the specified size. Set ``window`` to
    the number of observations to test a shorter series.

    References
    ----------
    .. [1] Basel Committee on Banking Supervision (1996). Supervisory framework
       for the use of "backtesting" in conjunction with the internal models
       approach to market risk capital requirements.
    """
    p = check_float("p", p, lower=0.0, upper=1.0, inclusive="neither")
    window = check_int("window", window, lower=1)
    exceptions = _exceptions(returns, var, 1)
    nobs = exceptions.shape[0]
    if window > nobs:
        raise ValueError(
            f"window must be an integer no larger than the number of aligned "
            f"observations ({nobs}); got {window} (int)"
        )
    count = int(exceptions[-window:].sum())
    probability = float(stats.binom.cdf(count, window, p))
    if count == 0 or probability < _YELLOW_PROBABILITY:
        zone = "green"
    elif probability < _RED_PROBABILITY:
        zone = "yellow"
    else:
        zone = "red"
    return TrafficLightResult(
        zone=zone, exceptions=count, probability=probability, nobs=window, p=p
    )
