"""
Calculates quantiles of the KPSS test statistic for both the constant
and constant plus trend scenarios.
"""
import os

import numpy as np
from numpy.random import RandomState
import pandas as pd

from arch.utility.timeseries import add_trend


def simulate_kpss(nobs, b, trend="c", rng=None):
    """
    Simulated the KPSS test statistic for nobs observations,
    performing b replications.
    """
    if rng is None:
        rng = RandomState()
        rng.seed(0)

    standard_normal = rng.standard_normal

    e = standard_normal((nobs, b))
    z = np.ones((nobs, 1))
    if trend == "ct":
        z = add_trend(z, trend="t")
    zinv = np.linalg.pinv(z)
    trend_coef = zinv.dot(e)
    resid = e - z.dot(trend_coef)
    s = np.cumsum(resid, axis=0)
    lam = np.mean(resid ** 2.0, axis=0)
    kpss = 1 / (nobs ** 2.0) * np.sum(s ** 2.0, axis=0) / lam
    return kpss


def wrapper(nobs, b, trend="c", max_memory=1024):
    """
    A wrapper around the main simulation that runs it in blocks so that large
    simulations can be run without constructing very large arrays and running
    out of memory.
    """
    rng = RandomState()
    rng.seed(0)
    memory = max_memory * 2 ** 20
    b_max_memory = memory // 8 // nobs
    b_max_memory = max(b_max_memory, 1)
    remaining = b
    results = np.zeros(b)
    now = dt.datetime.now()
    time_fmt = "{0:d}:{1:0>2d}:{2:0>2d}"
    msg = "trend {0}, {1} reps remaining, " + "elapsed {2}, remaining {3}"
    while remaining > 0:
        b_eff = min(remaining, b_max_memory)
        completed = b - remaining
        results[completed : completed + b_eff] = simulate_kpss(
            nobs, b_eff, trend=trend, rng=rng
        )
        remaining -= b_max_memory
        elapsed = (dt.datetime.now() - now).total_seconds()
        expected_remaining = max(0, remaining) * (elapsed / (b - remaining))

        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        elapsed = time_fmt.format(h, m, s)

        m, s = divmod(int(expected_remaining), 60)
        h, m = divmod(m, 60)
        expected_remaining = time_fmt.format(h, m, s)

        print(msg.format(trend, max(0, remaining), elapsed, expected_remaining))

    return results


if __name__ == "__main__":
    import datetime as dt

    nobs = 2000
    B = 100000000

    percentiles = np.concatenate(
        (
            np.arange(0.0, 99.0, 0.5),
            np.arange(99.0, 99.9, 0.1),
            np.arange(99.9, 100.0, 0.01),
        )
    )

    critical_values = 100 - percentiles
    critical_values_string = map("{0:0.1f}".format, critical_values)

    hdf_filename = "kpss_critical_values.h5"
    try:
        os.remove(hdf_filename)
    except OSError:
        pass

    for tr in ("c", "ct"):
        now = dt.datetime.now()
        kpss = wrapper(nobs, B, trend=tr)
        quantiles = np.percentile(kpss, list(percentiles))
        df = pd.DataFrame(quantiles, index=critical_values, columns=[tr])
        df.to_hdf(hdf_filename, key=tr, mode="a")
