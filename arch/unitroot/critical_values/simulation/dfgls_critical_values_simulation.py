"""
Critical value simulation for the Dickey-Fuller GLS model.  Similar in design
to MacKinnon (2010).  Makes use of parallel_func which works
best when joblib is installed.
"""

import argparse
import pathlib
import platform
import random
from typing import cast

import colorama
from joblib import Parallel, delayed
import numpy as np
from numpy.linalg import pinv
from numpy.random import PCG64, Generator, SeedSequence
import psutil

from arch._typing import Literal

# Storage Location
if platform.system() == "Linux":
    BASE_PATH = pathlib.Path("/mnt", "c")
else:
    BASE_PATH = pathlib.Path(r"C:\\")
# Created by the scripts that write to it, and not when this module is imported
OUTPUT_PATH = BASE_PATH / "Users" / "kevin" / "Dropbox" / "dfgls-simulation"
if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True)


GREEN = colorama.Fore.GREEN
BLUE = colorama.Fore.BLUE
RED = colorama.Fore.RED
RESET = colorama.Fore.RESET

# Number of repetitions
EX_NUM = 2000
# Number of simulations per exercise
EX_SIZE = 250_000
# Approximately controls memory use, in MiB
MAX_MEMORY_SIZE = 100
PERCENTILES = list(np.arange(0, 500, 25) / 100.0)
PERCENTILES += list(np.arange(50, 950, 5) / 10.0)
PERCENTILES += list(np.arange(9500, 10000, 25) / 100.0)
PERCENTILES = sorted(float(v) for v in PERCENTILES)

TRENDS = ("c", "ct")
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

# raw entropy (16 bits each) from random.org
RAW = [
    13991,
    62560,
    42376,
    1909,
    45867,
    59806,
    25767,
    65094,
    54534,
    46023,
    16925,
    34786,
    20589,
    8564,
    55417,
    41999,
]
ENTROPY = [(RAW[i] << 16) + RAW[i + 1] for i in range(0, len(RAW), 2)]


def wrapper(
    trend: Literal["c", "ct"],
    gen: Generator,
    file_name: pathlib.Path,
    max_memory_size: int = MAX_MEMORY_SIZE,
) -> pathlib.Path:
    """
    Wraps and blocks the main simulation so that the maximum amount of memory
    can be controlled on multi-processor systems when executing in parallel
    """
    if file_name.exists():
        return file_name
    results = np.zeros((len(T), len(PERCENTILES)))

    for i, nobs in enumerate(T):
        output = np.zeros(EX_SIZE)
        remaining = EX_SIZE
        finished = 0
        block_size = int(2**20.0 * max_memory_size / (8.0 * nobs))
        for _ in range(0, EX_SIZE, block_size):
            if block_size < remaining:
                nsim = block_size
            else:
                nsim = remaining
            st = finished
            en = finished + nsim
            output[st:en] = dfgls_simulation(nobs, trend, nsim, gen)
            finished += nsim
            remaining -= nsim

        results[i] = np.percentile(output, PERCENTILES)

    np.save(file_name, results)

    return file_name


def dfgls_simulation(
    n: int, trend: Literal["c", "ct"], block_size: int, gen: Generator
) -> float:
    """
    Simulates the empirical distribution of the DFGLS test statistic
    """
    standard_normal = gen.standard_normal
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
    y = standard_normal((n, block_size))
    y = np.cumsum(y, axis=0)
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
    parser = argparse.ArgumentParser("Simulations for DFGLS critical values")
    parser.add_argument(
        "--ncpu",
        type=int,
        action="store",
        help="Number of CPUs to use. If not specified, uses cpu_count() - 1",
    )
    parser.add_argument(
        "--max-memory-size",
        type=int,
        action="store",
        help="Maximum memory size to use in each thread. Default is 250",
    )

    args = parser.parse_args()
    ncpu = getattr(args, "ncpu", None)
    ncpu = psutil.cpu_count(logical=False) - 1 if ncpu is None else ncpu
    ncpu = max(ncpu, 1)

    # argparse sets max_memory_size to None when the option is not given
    max_memory_size = getattr(args, "max_memory_size", None)
    max_memory_size = MAX_MEMORY_SIZE if max_memory_size is None else max_memory_size

    jobs = []
    ss = SeedSequence(ENTROPY)
    children = ss.spawn(len(TRENDS) * EX_NUM)
    generators = [Generator(PCG64(child)) for child in children]

    count = 0
    for tr in TRENDS:
        for i in range(EX_NUM):
            jobs.append(
                (
                    tr,
                    generators[count],
                    pathlib.Path(f"dfgls_{tr}_{i}.npy"),
                )
            )
            count += 1
    jobs = [job for job in jobs if not job[-1].exists()]
    random.shuffle(jobs)
    nremconfig = len(jobs)
    nconfig = len(children)
    print(
        f"Total configurations: {BLUE}{nconfig}{RESET}, "
        f"Remaining: {RED}{nremconfig}{RESET}"
    )
    print(f"Running on {BLUE}{ncpu}{RESET} CPUs")
    if ncpu > 1:
        Parallel(verbose=50, n_jobs=ncpu)(
            delayed(wrapper)(tr, gen, file_name, max_memory_size)
            for tr, gen, file_name in jobs
        )
    else:
        for job in jobs:
            wrapper(*job)
