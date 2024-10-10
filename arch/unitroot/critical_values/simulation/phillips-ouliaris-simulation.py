import argparse
import datetime as dt
import gzip
import os
import pickle
from random import shuffle
import sys
from typing import IO, Optional, cast

import colorama
from joblib import Parallel, delayed
import numpy as np
from numpy.linalg import inv, lstsq, solve
import pandas as pd
from phillips_ouliaris import QUANTILES, ROOT, SAMPLE_SIZES, TRENDS
import psutil

from arch.typing import Float64Array, Literal, UnitRootTrend
from arch.utility.timeseries import add_trend

GREEN = colorama.Fore.GREEN
BLUE = colorama.Fore.BLUE
RED = colorama.Fore.RED
RESET = colorama.Fore.RESET
MAX_STOCHASTIC_TRENDS = 13
if not os.path.exists(ROOT):
    os.mkdir(ROOT)
# Number of simulations per exercise
EX_SIZE = 250000
# Number of experiments
EX_NUM = 100
# Maximum memory of the main simulated data
MAX_MEMORY = 2**21
# Number of iterations between display
DISP_ITERATIONS = 25000

if sys.platform.lower() == "win32":
    os.system("color")

STATISTICS = ("z", "p")
Z_STATISTICS = ("z_a", "z_t")
P_STATISTICS = ("p_u", "p_z")
INDEX_NAMES = ["sample_size", "stochastic_trends", "statistic"]
DF_Z_COLUMNS = pd.MultiIndex.from_product(
    [SAMPLE_SIZES, list(range(1, MAX_STOCHASTIC_TRENDS + 1)), Z_STATISTICS]
)
DF_P_COLUMNS = pd.MultiIndex.from_product(
    [SAMPLE_SIZES, list(range(1, MAX_STOCHASTIC_TRENDS + 1)), P_STATISTICS]
)
DF_Z_COLUMNS = DF_Z_COLUMNS.set_names(INDEX_NAMES)
DF_P_COLUMNS = DF_P_COLUMNS.set_names(INDEX_NAMES)


def demean(w: Float64Array) -> Float64Array:
    return w - w.mean(1).reshape((w.shape[0], 1, w.shape[2]))


def inner_prod(a: Float64Array, b: Optional[Float64Array] = None) -> Float64Array:
    if b is None:
        b = a
    return a.transpose((0, 2, 1)) @ b


def z_tests_vec(
    z: Float64Array, lag: int, trend: UnitRootTrend
) -> tuple[np.ndarray, np.ndarray]:
    assert z.ndim == 3
    nobs = int(z.shape[1])
    if trend == "c":
        z = demean(z)
    elif trend in ("ct", "ctt"):
        tr = add_trend(nobs=nobs, trend=trend)
        tr /= np.sqrt((tr**2).mean(0) * nobs)
        trptr = tr.T @ tr
        trpz = tr.T @ z
        z = z - tr @ solve(trptr, trpz)
    y = z[..., :1]
    x = z[..., 1:]
    u = y
    if z.shape[-1] > 1:
        xpx = inner_prod(x)
        xpx_inv = inv(xpx)
        b = xpx_inv @ inner_prod(x, y)
        u = y - x @ b
    nseries = u.shape[0]
    u = u.reshape((nseries, -1)).T
    ulag = u[:-1]
    ulead = u[1:]
    alpha = (ulead * ulag).mean(0) / (ulag**2).mean(0)
    one_sided_strict = np.zeros_like(alpha)
    k = ulead - ulag * alpha
    for i in range(1, lag + 1):
        w = 1 - i / (lag + 1)
        one_sided_strict += 1 / nobs * w * (k[i:] * k[:-i]).sum(0)

    u2 = (u[:-1] * u[:-1]).sum(0)
    z = (alpha - 1) - nobs * one_sided_strict / u2
    z_a = nobs * z
    long_run = (k**2).sum(0) / nobs + 2 * one_sided_strict
    z_t = np.sqrt(u2) * z / np.sqrt(long_run)
    assert isinstance(z_a, np.ndarray)
    assert isinstance(z_t, np.ndarray)
    return z_a, z_t


def z_tests(z: Float64Array, lag: int, trend: UnitRootTrend) -> tuple[float, float]:
    z = add_trend(z, trend=trend)
    u = z
    if z.shape[1] > 1:
        u = z[:, 0] - z[:, 1:] @ lstsq(z[:, 1:], z[:, 0], rcond=None)[0]
    alpha = float((u[:-1].T @ u[1:]) / (u[:-1].T @ u[:-1]))
    k = u[1:] - alpha * u[:-1]
    nobs = int(u.shape[0])
    one_sided_strict = 0.0
    for i in range(1, lag + 1):
        w = 1 - i / (lag + 1)
        one_sided_strict += 1 / nobs * w * k[i:].T @ k[:-i]
    u2 = float(u[:-1].T @ u[:-1])

    z_ = (alpha - 1) - nobs * one_sided_strict / u2
    z_a = nobs * z_
    long_run = k.T @ k / nobs + 2 * one_sided_strict
    z_t = np.sqrt(u2) * z_ / long_run
    return z_a, z_t


def p_tests_vec(
    z: Float64Array, lag: int, trend: UnitRootTrend
) -> tuple[np.ndarray, np.ndarray]:
    assert z.ndim == 3
    z_lag, z_lead = z[:, :-1], z[:, 1:]
    nobs = z.shape[1]
    if trend == "c":
        z = demean(z)
        z_lag = demean(z_lag)
        z_lead = demean(z_lead)
    elif trend in ("ct", "ctt"):
        post = []
        for v in (z, z_lag, z_lead):
            tr = add_trend(nobs=v.shape[1], trend=trend)
            tr /= np.sqrt((tr**2).mean(0) * nobs)
            trptr = tr.T @ tr
            trpv = tr.T @ v
            post.append(v - tr @ solve(trptr, trpv))
        z, z_lag, z_lead = post
    else:
        z = z - z[:, :1]

    x, y = z[..., 1:], z[..., :1]
    u = y
    if x.shape[-1]:
        beta = solve(inner_prod(x), inner_prod(x, y))
        u = y - x @ beta
    phi = solve(inner_prod(z_lag), inner_prod(z_lag, z_lead))
    xi = z_lead - z_lag @ phi

    omega = inner_prod(xi) / nobs
    for i in range(1, lag + 1):
        w = 1 - i / (lag + 1)
        gamma = inner_prod(xi[:, i:], xi[:, :-i]) / nobs
        omega += w * (gamma + cast(np.ndarray, gamma).transpose((0, 2, 1)))
    omega21 = omega[:, :1, 1:]
    omega22 = omega[:, 1:, 1:]
    omega112 = omega[:, :1, :1] - omega21 @ inv(omega22) @ omega21.transpose((0, 2, 1))
    denom = inner_prod(u) / nobs
    p_u = nobs * np.squeeze(omega112 / denom)

    # z detrended above
    m_zz = inner_prod(z) / nobs
    # ufunc trace using einsum
    p_z = nobs * np.einsum("...ii", omega @ inv(m_zz))

    return p_u, p_z


def p_tests(z: Float64Array, lag: int, trend: UnitRootTrend) -> tuple[float, float]:
    x, y = z[:, 1:], z[:, 0]
    nobs = int(x.shape[0])
    x = add_trend(x, trend=trend)
    beta = lstsq(x, y, rcond=None)[0]
    u = y - x @ beta
    z_lead = z[1:]
    z_lag = add_trend(z[:-1], trend=trend)
    phi = lstsq(z_lag, z_lead, rcond=None)[0]
    xi = z_lead - z_lag @ phi

    omega = xi.T @ xi / nobs
    for i in range(1, lag + 1):
        w = 1 - i / (lag + 1)
        gamma = xi[i:].T @ xi[:-i] / nobs
        omega += w * (gamma + gamma.T)
    omega21 = omega[0, 1:]
    omega22 = omega[1:, 1:]
    omega112 = float(omega[0, 0] - np.squeeze(omega21.T @ inv(omega22) @ omega21))
    denom = u.T @ u / nobs
    p_u = nobs * omega112 / denom

    tr = add_trend(nobs=z.shape[0], trend=trend)
    if tr.shape[1]:
        z = z - tr @ lstsq(tr, z, rcond=None)[0]
    else:
        z = z - z[:1]  # Recenter on first
    m_zz = z.T @ z / nobs
    p_z = nobs * float((omega @ inv(m_zz)).trace())
    return p_u, p_z


def block(
    gen: np.random.Generator,
    statistic: str,
    num: int,
    trend: UnitRootTrend,
) -> Float64Array:
    max_sample = max(SAMPLE_SIZES)
    e = gen.standard_normal((num, max_sample, MAX_STOCHASTIC_TRENDS))
    z = e.cumsum(axis=1)
    columns = DF_Z_COLUMNS if statistic == "a" else DF_P_COLUMNS
    results = np.empty((num, len(columns)))
    loc = 0
    for ss in SAMPLE_SIZES:
        for ns in range(1, MAX_STOCHASTIC_TRENDS + 1):
            omega_dof = ss - 2 * ns - len(trend)
            z_a = z_t = p_u = p_z = np.full(z.shape[0], np.nan)
            if omega_dof >= 20:
                if statistic == "z":
                    z_a, z_t = z_tests_vec(z[:, :ss, :ns], 0, trend=trend)
                elif statistic == "p":
                    p_u, p_z = p_tests_vec(z[:, :ss, :ns], 0, trend=trend)
                else:
                    raise ValueError(f"statistic must be a or p, saw {statistic}")
            if statistic == "z":
                stats = np.column_stack([z_a, z_t])
            else:  # p
                stats = np.column_stack([p_u, p_z])
            stride = stats.shape[1]
            results[:, loc : loc + stride] = stats
            loc += stride
    return results


def temp_file_name(full_path: str, gzip: bool = True) -> str:
    base, file_name = list(os.path.split(full_path))
    extension = ".pkl" if not gzip else ".pkl.gz"
    temp_file = "partial-" + file_name.replace(".hdf", extension)
    return os.path.join(base, temp_file)


def save_partial(
    gen: np.random.Generator, results: pd.DataFrame, remaining: int, full_path: str
) -> None:
    temp_file = temp_file_name(full_path)
    info = {"results": results, "remaining": remaining, "gen": gen}
    with gzip.open(temp_file, "wb", 4) as pkl:
        pickle.dump(info, cast(IO[bytes], pkl))


def load_partial(
    gen: np.random.Generator, results: pd.DataFrame, remaining: int, full_path: str
) -> tuple[np.random.Generator, pd.DataFrame, int]:
    temp_file = temp_file_name(full_path)
    if os.path.exists(temp_file):
        try:
            with gzip.open(temp_file, "rb") as pkl:
                info = pickle.load(cast(IO[bytes], pkl))
            gen = info["gen"]
            results = info["results"]
            remaining = info["remaining"]
        except EOFError:
            print(f"{RED}{temp_file} is corrupt, deleting.{RESET}")
            os.unlink(temp_file)

    return gen, results, remaining


def worker(
    gen: np.random.Generator,
    statistic: str,
    trend: UnitRootTrend,
    idx: int,
    full_path: str,
) -> None:
    print(
        f"Starting Index: {BLUE}{idx}{RESET}, Statistic: {RED}{statistic}{RESET},"
        f"Trend: {GREEN}{trend}{RESET}"
    )
    remaining = EX_SIZE
    ncols = len(SAMPLE_SIZES) * MAX_STOCHASTIC_TRENDS * 4
    block_size = int(MAX_MEMORY / (ncols * 8))
    columns = DF_Z_COLUMNS if statistic == "z" else DF_P_COLUMNS
    results = pd.DataFrame(
        index=pd.RangeIndex(EX_SIZE), columns=columns, dtype="double"
    )
    gen, results, remaining = load_partial(gen, results, remaining, full_path)
    start = dt.datetime.now()
    last_print_remaining = remaining
    while remaining > 0:
        nsim = min(remaining, block_size)
        res_block = block(gen, statistic, nsim, trend)
        loc = EX_SIZE - remaining
        results.iloc[loc : loc + nsim] = res_block
        remaining -= block_size
        remaining = max(0, remaining)
        elapsed = dt.datetime.now() - start
        time_per_iter = elapsed.total_seconds() / (EX_SIZE - remaining)
        remaining_time = int(time_per_iter * remaining)
        rem = str(dt.timedelta(seconds=remaining_time))
        if last_print_remaining - remaining >= DISP_ITERATIONS:
            print(f"Index: {idx} {RED}Saving{RESET}")
            save_partial(gen, results, remaining, full_path)
            print(
                f"Index: {idx}, Statistic: {statistic}, Trend: {trend}, "
                f"Remaining: {GREEN}{remaining}{RESET}, "
                f"Est. time remaining: {RED}{rem}{RESET}"
            )
            last_print_remaining = remaining
    results = results.quantile(QUANTILES)
    results.to_hdf(full_path, key="results")
    if os.path.exists(temp_file_name(full_path)):
        try:
            os.unlink(temp_file_name(full_path))
        except OSError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simulations for Engle-Granger critical values")
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
    # random.org seeds
    entropy = [
        387520566,
        658404341,
        801610112,
        45811674,
        150145835,
        848151192,
        904081896,
        322265304,
        96932831,
        931388087,
    ]

    ss = np.random.SeedSequence(entropy)
    children = ss.spawn(len(TRENDS) * EX_NUM * len(STATISTICS))
    jobs: list[
        tuple[np.random.Generator, Literal["z", "p"], UnitRootTrend, int, str]
    ] = []
    loc = 0
    from itertools import product

    for statistic, trend, idx in product(STATISTICS, TRENDS, range(EX_NUM)):
        child = children[loc]
        gen = np.random.Generator(np.random.PCG64(child))
        filename = (
            "phillips-ouliaris-results-statistic-"
            + f"{statistic}-trend-{trend}-{idx:04d}.hdf"
        )

        full_file = os.path.join(ROOT, filename)
        if os.path.exists(full_file):
            continue
        jobs.append(
            (
                gen,
                cast(Literal["z", "p"], statistic),
                cast(UnitRootTrend, trend),
                idx,
                full_file,
            )
        )
        loc += 1
    shuffle(jobs)
    if args.z_only:
        print(f"{BLUE}Note{RESET}: Only running Z-type tests")
        jobs = [job for job in jobs if job[1] == "z"]
    # Reorder jobs to prefer those with partial results first
    first = []
    remaining = []
    for job in jobs:
        if os.path.exists(temp_file_name(job[4])):
            first.append(job)
        else:
            remaining.append(job)
    jobs = first + remaining
    nremconfig = len(jobs)
    nconfig = len(children)
    print(
        f"Total configurations: {BLUE}{nconfig}{RESET}, "
        f"Remaining: {RED}{nremconfig}{RESET}"
    )
    print(f"Running on {BLUE}{njobs}{RESET} CPUs")
    if njobs == 1:
        for job in jobs:
            worker(*job)
    else:
        Parallel(verbose=50, n_jobs=njobs)(
            delayed(worker)(gen, statistic, trend, idx, fullfile)
            for (gen, statistic, trend, idx, fullfile) in jobs
        )
