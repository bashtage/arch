"""
Critical value simulation for the Dickey-Fuller GLS model.  Similar in design
to MacKinnon (2010).  Makes use of parallel_fun which works
best when joblib is installed.
"""

import argparse
import datetime as dt
import os
from typing import cast

import numpy as np
from numpy.linalg import pinv
from numpy.random import RandomState
from statsmodels.tools.parallel import parallel_func

from arch._typing import Literal

# Controls memory use, in MiB
MAX_MEMORY_SIZE = 100
NUM_JOBS = 4
EX_NUM = 500
EX_SIZE = 200000


def wrapper(n: int, trend: Literal["c", "ct"], b: int, seed: int = 0) -> np.ndarray:
    """
    Wraps and blocks the main simulation so that the maximum amount of memory
    can be controlled on multi processor systems when executing in parallel
    """
    rng = RandomState()
    rng.seed(seed)
    remaining = b
    res = np.zeros(b)
    finished = 0
    block_size = int(2**20.0 * MAX_MEMORY_SIZE / (8.0 * n))
    for _ in range(0, b, block_size):
        if block_size < remaining:
            count = block_size
        else:
            count = remaining
        st = finished
        en = finished + count
        res[st:en] = dfgsl_simulation(n, trend, count, rng)
        finished += count
        remaining -= count

    return res


def dfgsl_simulation(
    n: int, trend: Literal["c", "ct"], b: int, rng: RandomState | None = None
) -> float:
    """
    Simulates the empirical distribution of the DFGLS test statistic
    """
    if rng is None:
        rng = RandomState(0)
    standard_normal = rng.standard_normal

    nobs = n
    if trend == "c":
        c = -7.0
        z = np.ones((nobs, 1))
    else:
        c = -13.5
        z = np.vstack((np.ones(nobs), np.arange(1, nobs + 1))).T

    ct = c / nobs

    delta_z = np.copy(z)
    delta_z[1:, :] = delta_z[1:, :] - (1 + ct) * delta_z[:-1, :]
    delta_z_inv = pinv(delta_z)
    y = standard_normal((n + 50, b))
    y = np.cumsum(y, axis=0)
    y = y[50:, :]
    delta_y = y.copy()
    delta_y[1:, :] = delta_y[1:, :] - (1 + ct) * delta_y[:-1, :]
    detrend_coef = delta_z_inv.dot(delta_y)
    y_detrended = y - cast("np.ndarray", z.dot(detrend_coef))

    delta_y_detrended = np.diff(y_detrended, axis=0)
    rhs = y_detrended[:-1, :]
    lhs = delta_y_detrended

    xpy = np.sum(rhs * lhs, 0)
    xpx = np.sum(rhs**2.0, 0)
    gamma = xpy / xpx
    e = lhs - rhs * gamma
    sigma2 = np.sum(e**2.0, axis=0) / (n - 1)  # DOF correction?
    gamma_var = sigma2 / xpx

    stat = gamma / np.sqrt(gamma_var)
    return stat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate DFGLS critical values")
    parser.add_argument(
        "--ex-num", type=int, default=EX_NUM, help="Number of experiments"
    )
    parser.add_argument(
        "--ex-size",
        type=int,
        default=EX_SIZE,
        help="Number of simulations per experiment",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=NUM_JOBS, help="Number of parallel jobs"
    )
    parser.add_argument(
        "--output-path",
        default=".",
        help="Directory to write dfgls_c.npz and dfgls_ct.npz to",
    )
    args = parser.parse_args()
    trends = ("c", "ct")
    T = np.array(
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
    T = T[::-1]
    percentiles = list(np.arange(0.5, 100.0, 0.5))
    seeds = np.arange(0, 2**32, step=2**23)
    if args.ex_num > seeds.shape[0]:
        raise ValueError(f"ex-num must be at most {seeds.shape[0]}")
    for tr in trends:
        results = np.zeros((len(percentiles), len(T), args.ex_num))
        file_name = os.path.join(args.output_path, "dfgls_" + tr + ".npz")

        for i in range(args.ex_num):
            print(f"Experiment Number {i + 1} of {args.ex_num} (trend {tr})")
            now = dt.datetime.now()
            parallel, p_func, n_jobs = parallel_func(
                wrapper, n_jobs=args.n_jobs, verbose=2
            )
            out = parallel(p_func(t, tr, args.ex_size, seed=seeds[i]) for t in T)
            quantiles = [np.percentile(x, percentiles) for x in out]
            results[:, :, i] = np.array(quantiles).T
            print(f"Elapsed time {dt.datetime.now() - now} seconds")

            if i % 50 == 0:
                # Only the experiments that have finished, so that a partial file
                # can be processed
                np.savez(
                    file_name,
                    trend=tr,
                    results=results[:, :, : i + 1],
                    percentiles=percentiles,
                    T=T,
                )

        np.savez(
            file_name,
            trend=tr,
            results=results,
            percentiles=percentiles,
            T=T,
        )
