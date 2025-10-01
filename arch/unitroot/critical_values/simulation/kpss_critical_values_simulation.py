"""
Calculates quantiles of the KPSS test statistic for both the constant
and constant plus trend scenarios.
"""

import os
from typing import cast

import numpy as np
from numpy.random import RandomState
import pandas as pd

from arch._typing import Float64Array
from arch.utility.timeseries import add_trend


def simulate_kpss(
    nobs: int,
    b: int,
    trend: str = "c",
    rng: RandomState | None = None,
) -> float:
    """
    Simulated the KPSS test statistic for nobs observations,
    performing b replications.
    """
    if rng is None:
        rng = RandomState()
        rng.seed(0)

    standard_normal = rng.standard_normal

    e = standard_normal((nobs, b))
    z: Float64Array = np.ones((nobs, 1))
    if trend == "ct":
        z = add_trend(z, trend="t")
    zinv = np.linalg.pinv(z)
    trend_coef = zinv.dot(e)
    resid = e - cast("np.ndarray", z.dot(trend_coef))
    s = np.cumsum(resid, axis=0)
    lam = (resid**2.0).mean(axis=0)
    kpss = 1 / (nobs**2.0) * (s**2.0).sum(axis=0) / lam
    return kpss


def wrapper(nobs: int, b: int, trend: str = "c", max_memory: int = 1024) -> np.ndarray:
    """
    A wrapper around the main simulation that runs it in blocks so that large
    simulations can be run without constructing very large arrays and running
    out of memory.
    """
    rng = RandomState()
    rng.seed(0)
    memory = max_memory * 2**20
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
        elapsed_fmt = time_fmt.format(h, m, s)

        m, s = divmod(int(expected_remaining), 60)
        h, m = divmod(m, 60)
        expected_remaining_fmt = time_fmt.format(h, m, s)

        print(msg.format(trend, max(0, remaining), elapsed_fmt, expected_remaining_fmt))

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
    critical_values_string = [f"{cv:0.1f}" for cv in critical_values]

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
