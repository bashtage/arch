"""
Simulation of ADF z-test critical values.  Closely follows MacKinnon (2010).
"""

import argparse
import os
import random
from typing import cast

from adf_simulation import (
    OUTPUT_PATH,
    PERCENTILES,
    TIME_SERIES_LENGTHS,
    TRENDS,
    adf_simulation,
)
import colorama
from joblib import Parallel, delayed
import numpy as np
from numpy.random import PCG64, Generator, SeedSequence
import psutil

from arch._typing import UnitRootTrend

GREEN = colorama.Fore.GREEN
BLUE = colorama.Fore.BLUE
RED = colorama.Fore.RED
RESET = colorama.Fore.RESET

# Number of repetitions
EX_NUM = 500
# Number of simulations per exercise
EX_SIZE = 250000
# Approximately controls memory use, in MiB
MAX_MEMORY_SIZE = 100


# raw entropy (16 bits each) from random.org
RAW = [
    64303,
    60269,
    6936,
    46495,
    33811,
    56090,
    36001,
    55726,
    32840,
    17611,
    32276,
    58287,
    10615,
    53045,
    52978,
    10484,
    25209,
    35367,
    52618,
    24147,
]
ENTROPY = [(RAW[i] << 16) + RAW[i + 1] for i in range(0, len(RAW), 2)]


def single_experiment(trend: str, gen: Generator, file_name: str) -> None:
    """
    Wraps and blocks the main simulation so that the maximum amount of memory
    can be controlled on multi processor systems when executing in parallel
    """

    res = np.zeros(EX_SIZE)
    output: np.ndarray = np.zeros(
        (len(cast("np.ndarray", PERCENTILES)), len(TIME_SERIES_LENGTHS))
    )
    for col, nobs in enumerate(TIME_SERIES_LENGTHS):
        remaining = EX_SIZE
        finished = 0
        block_size = int(2**20.0 * MAX_MEMORY_SIZE / (8.0 * nobs))
        for _ in range(0, EX_SIZE, block_size):
            if block_size < remaining:
                count = block_size
            else:
                count = remaining
            st = finished
            en = finished + count
            _trend: UnitRootTrend
            if trend == "n":
                _trend = "n"
            elif trend == "c":
                _trend = "c"
            elif trend == "ct":
                _trend = "ct"
            else:
                _trend = "ctt"
            res[st:en] = adf_simulation(nobs, _trend, count, gen)
            finished += count
            remaining -= count
        output[:, col] = np.percentile(res, PERCENTILES)
    np.savez(file_name, results=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simulations for ADF-Z critical values")
    parser.add_argument(
        "--ncpu",
        type=int,
        action="store",
        help="Number of CPUs to use. If not specified, uses cpu_count() - 1",
    )
    parser.add_argument(
        "--z_only",
        action="store_true",
        help="Only execute Z-type tests",
    )
    args = parser.parse_args()
    njobs = getattr(args, "ncpu", None)
    njobs = psutil.cpu_count(logical=False) - 1 if njobs is None else njobs
    njobs = max(njobs, 1)

    ss = SeedSequence(ENTROPY)
    children = ss.spawn(len(TRENDS) * EX_NUM)
    generators = [Generator(PCG64(child)) for child in children]
    jobs = []
    count = 0
    for tr in TRENDS:
        for i in range(EX_NUM):
            file_name = os.path.join(OUTPUT_PATH, f"adf_z_{tr}-{i:04d}.npz")
            jobs.append((tr, generators[count], file_name))
            count += 1
    jobs = [job for job in jobs if not os.path.exists(job[-1])]
    random.shuffle(jobs)
    nremconfig = len(jobs)
    nconfig = len(children)
    print(
        f"Total configurations: {BLUE}{nconfig}{RESET}, "
        f"Remaining: {RED}{nremconfig}{RESET}"
    )
    print(f"Running on {BLUE}{njobs}{RESET} CPUs")
    if njobs > 1:
        Parallel(verbose=50, n_jobs=njobs)(
            delayed(single_experiment)(t, g, f) for t, g, f in jobs
        )
    else:
        for job in jobs:
            single_experiment(*job)
