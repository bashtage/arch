"""
Simulation for critical value production for Engle-Granger
"""

import argparse
import datetime as dt
from functools import partial
from itertools import product
import os
from random import shuffle
import sys
from typing import Sequence

import colorama
from joblib import Parallel, cpu_count, delayed
import numpy as np
from numpy.random import PCG64, Generator, SeedSequence

ROOT = os.path.join(os.path.split(os.path.abspath(__file__))[0], "engle-granger")
if not os.path.exists(ROOT):
    os.mkdir(ROOT)
# Number of simulations per exercise
EX_SIZE = 250000
# Number of experiments
EX_NUM = 500
# percentiles to save
PERCENTILES = list(np.arange(0.1, 1.0, 0.1)) + list(np.arange(1.0, 100.0, 0.5))
PERCENTILES = PERCENTILES[::-1]
# Maximum memory of the main simulated data
MAX_MEMORY = 2**21

if sys.platform.lower() == "win32":
    os.system("color")


def block(
    rg: Generator,
    trend: str,
    sample_sizes: Sequence[int],
    cross_section_size: int,
    simulations: int,
    idx: int,
) -> None:
    filename = f"engle-granger-results-trend-{trend}-{idx:04d}.npz"
    fullfile = os.path.join(ROOT, filename)
    if os.path.exists(fullfile):
        return
    max_k = cross_section_size
    max_t = max(sample_sizes) + 1
    max_blocks_size = max(MAX_MEMORY // (max_t * max_k * 8), 1)
    print(f"Max block size: {max_blocks_size}")
    remaining = simulations
    completed = 0
    tstats = np.empty((len(sample_sizes), simulations, max_k))
    start = dt.datetime.now()
    last_report = remaining + 1001
    while remaining:
        this_block = min(remaining, max_blocks_size)
        remaining -= this_block
        e = rg.standard_normal((this_block, max_t + 50, max_k))
        data = np.cumsum(e, 1)
        data = data[:, 50:, :]
        del e
        for j, sub_size in enumerate(sample_sizes):
            # Copy for needed for 'nc' trend
            sub = data[:, : sub_size + 1].copy()
            if trend not in ("c", "ct", "ctt", "nc"):
                raise NotImplementedError
            if trend in ("c", "ct", "ctt"):
                # Mean 0
                mu = sub.mean(1)
                mu.shape = (this_block, 1, cross_section_size)
                sub -= mu
            if trend in ("ct", "ctt"):
                # Orthogonalize to trend
                tau = np.arange(float(sub_size + 1))
                tau -= tau.mean()
                tau.shape = (sub_size + 1, 1)
                coefs = (tau * sub).sum(1) / (tau**2).sum()
                coefs.shape = (this_block, 1, cross_section_size)
                sub -= coefs * tau
            if trend == "ctt":
                tau = np.arange(float(sub_size + 1))
                tau -= tau.mean()
                tau2 = np.arange(float(sub_size + 1)) ** 2
                tau2 -= tau2.mean()
                tau2 -= (tau * tau2).sum() / (tau**2).sum() * tau
                tau2.shape = (sub_size + 1, 1)
                coefs = (tau2 * sub).sum(1) / (tau2**2).sum()
                coefs.shape = (this_block, 1, cross_section_size)
                sub -= coefs * tau2
            errors = np.empty((this_block, sub_size + 1, max_k))
            errors[:, :, :1] = sub[:, :, :1]
            rhs = sub[:, :, 1:]
            ip = np.matmul(np.transpose(rhs, (0, 2, 1)), rhs) / sub_size
            ip_chol_inv = np.linalg.inv(np.linalg.cholesky(ip))
            orth = np.matmul(rhs, np.transpose(ip_chol_inv, (0, 2, 1)))
            del ip_chol_inv, ip, rhs, sub
            for i in range(1, max_k):
                y = errors[:, :, (i - 1) : i]
                x = orth[:, :, (i - 1) : i]
                xp = np.transpose(x, (0, 2, 1))
                xpy = np.matmul(xp, y)
                xpx = np.matmul(xp, x)
                coefs = xpy / xpx
                errors[:, :, i : (i + 1)] = y - coefs * x
            del orth, x, y
            # Compute ADF statistics for each of the error series
            err_lag = errors[:, :-1]
            derrors = errors[:, 1:] - err_lag

            ypy = (derrors * derrors).sum(1)
            xpy = (derrors * err_lag).sum(1)
            xpx = (err_lag * err_lag).sum(1)
            del errors, err_lag, derrors

            coefs = xpy / xpx
            sse = ypy - coefs**2 * xpx
            sigma2 = sse / sub_size
            se = np.sqrt(sigma2 / xpx)
            tstats[j, completed : (completed + this_block)] = coefs / se
        completed += this_block
        elapsed = dt.datetime.now() - start
        time_per_iter = elapsed.total_seconds() / completed
        remaining_time = int(time_per_iter * remaining)
        if last_report - remaining > 1000:
            last_report = remaining
            print(f"Index: {idx}, Trend: {trend}, Remaining: {remaining}")
            print(f"Est. time remaining: {str(dt.timedelta(seconds=remaining_time))}")

    out = np.percentile(tstats, PERCENTILES, 1)
    np.savez(
        fullfile,
        quantiles=out,
        trend=np.array([trend]),
        sample_sizes=np.array(sample_sizes),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simulations for Engle-Granger critical values")
    parser.add_argument(
        "--ncpu",
        type=int,
        action="store",
        help="Number of CPUs to use. If not specified, uses cpu_count() - 1",
    )
    args = parser.parse_args()
    njobs = getattr(args, "ncpu", None)
    njobs = cpu_count() - 1 if njobs is None else njobs
    print(f"Running on {njobs} CPUs")
    entropy_bits = [
        41526,
        18062,
        18883,
        25265,
        56208,
        23325,
        29606,
        40099,
        9776,
        46303,
        6333,
        15881,
        63110,
        6022,
        61267,
        56526,
    ]
    entropy = sum(bits << (16 * i) for i, bits in enumerate(entropy_bits))
    seq = SeedSequence(entropy)
    gen = [Generator(PCG64(child)) for child in seq.spawn(EX_NUM)]
    sample_sizes = (
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
    trends = ("nc", "c", "ct", "ctt")
    cross_section_size = 12
    simulations = EX_SIZE

    partial_block = partial(
        block,
        sample_sizes=sample_sizes,
        cross_section_size=cross_section_size,
        simulations=simulations,
    )

    remaining_configs = []
    configs = list(product(enumerate(gen), trends))
    for config in configs:
        trend = config[-1]
        idx = config[0][0]
        filename = f"engle-granger-results-trend-{trend}-{idx:04d}.npz"
        fullfile = os.path.join(ROOT, filename)
        if not os.path.exists(fullfile):
            remaining_configs.append(config)
    nconfig = colorama.Fore.GREEN + f"{len(configs)}" + colorama.Fore.RESET
    nremconfig = colorama.Fore.RED + f"{len(remaining_configs)}" + colorama.Fore.RESET
    print(f"Total configuration: {nconfig}, Remaining: {nremconfig}")

    shuffle(remaining_configs)
    if njobs == 1:
        for (idx, rg), trend in remaining_configs:
            partial_block(rg, trend=trend, idx=idx)
    else:
        Parallel(verbose=50, n_jobs=njobs)(
            delayed(partial_block)(rg, trend=trend, idx=idx)
            for ((idx, rg), trend) in remaining_configs
        )
