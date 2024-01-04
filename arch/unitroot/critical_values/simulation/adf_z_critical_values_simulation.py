"""
Simulation of ADF z-test critical values.  Closely follows MacKinnon (2010).
Running this files requires an IPython cluster, which is assumed to be
on the local machine.  This can be started using a command similar to

    ipcluster start -n 4

Remote clusters can be used by modifying the call to Client.
"""

import datetime
from typing import Literal, cast

from ipyparallel import Client
from numpy import array, nan, ndarray, percentile, savez

from .adf_simulation import adf_simulation

# Number of repetitions
EX_NUM = 500
# Number of simulations per exercise
EX_SIZE = 200000
# Approximately controls memory use, in MiB
MAX_MEMORY_SIZE = 100

rc = Client()
dview = rc.direct_view()
with dview.sync_imports():
    from numpy import arange, zeros
    from numpy.random import RandomState


def wrapper(
    n: int, trend: Literal["n", "c", "ct", "ctt"], b: int, seed: int = 0
) -> ndarray:
    """
    Wraps and blocks the main simulation so that the maximum amount of memory
    can be controlled on multi processor systems when executing in parallel
    """
    rng = RandomState()
    rng.seed(seed)
    remaining = b
    res = zeros(b)
    finished = 0
    block_size = int(2**20.0 * MAX_MEMORY_SIZE / (8.0 * n))
    for _ in range(0, b, block_size):
        if block_size < remaining:
            count = block_size
        else:
            count = remaining
        st = finished
        en = finished + count
        res[st:en] = adf_simulation(n, trend, count, rng)
        finished += count
        remaining -= count

    return res


# Push variables and functions to all engines
dview.execute("import numpy as np")
dview["MAX_MEMORY_SIZE"] = MAX_MEMORY_SIZE
dview["wrapper"] = wrapper
dview["adf_simulation"] = adf_simulation
lview = rc.load_balanced_view()

trends = ("n", "c", "ct", "ctt")
T = array(
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
m = T.shape[0]
percentiles = list(arange(0.5, 100.0, 0.5))
rng = RandomState(0)
seeds = rng.random_integers(0, 2**31 - 2, size=EX_NUM)

for tr in trends:
    results = cast(ndarray, zeros((len(percentiles), len(T), EX_NUM)) * nan)
    filename = "adf_z_" + tr + ".npz"

    for i in range(EX_NUM):
        print(f"Experiment Number {i + 1} for Trend {tr}")
        # Non parallel version
        # args = (T, [tr] * m, [EX_SIZE] * m, [seeds[i]] * m)
        # out = [wrapper(a, b, c, d) for a, b, c, d in args]
        now = datetime.datetime.now()
        out = lview.map_sync(wrapper, T, [tr] * m, [EX_SIZE] * m, [seeds[i]] * m)
        # Prevent unnecessary results from accumulating
        lview.purge_results("all")
        rc.purge_everything()
        print(datetime.datetime.now() - now)
        quantiles = [percentile(x, percentiles) for x in out]
        results[:, :, i] = cast(ndarray, array(quantiles).T)

        if i % 50 == 0:
            savez(filename, trend=tr, results=results, percentiles=percentiles, T=T)

    savez(filename, trend=tr, results=results, percentiles=percentiles, T=T)
