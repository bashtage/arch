"""
Tests of the local block bootstrap.

There is no other implementation of the local block bootstrap of Paparoditis and
Politis (2002) in Python, R (CRAN) or any package that is available, so the checks
are against an independent loop implementation of the algorithm of the paper
(``reference_indices``) that is driven by the same uniform draws, the exact support
of the block starts, the distribution of the starts and a simulation.
"""

from __future__ import annotations

import math
import pickle
from typing import Any

import numpy as np
from numpy.random import Generator, RandomState
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest
from scipy import stats

from arch.bootstrap import (
    MCS,
    SPA,
    CircularBlockBootstrap,
    IIDBootstrap,
    LocalBlockBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
)
import arch.bootstrap.base as base_module

NOBS = 23
BLOCK = 4
LOCALITY = 3

# (n, b, K) cases: the example, no locality, locality of the whole sample and more,
# block as large as the sample, b dividing and not dividing n, one block, b of 1.
CASES = [
    (23, 4, 3),
    (23, 4, 0),
    (23, 4, 1),
    (23, 4, 22),
    (23, 4, 23),
    (23, 4, 500),
    (23, 23, 0),
    (23, 23, 7),
    (23, 22, 3),
    (23, 5, 2),
    (23, 1, 0),
    (23, 1, 4),
    (20, 4, 0),
    (20, 4, 6),
    (24, 6, 5),
    (10, 4, 2),
    (10, 4, 100),
    (1, 1, 0),
    (2, 2, 3),
    (60, 7, 10),
]
CASE_IDS = [f"n{n}-b{b}-K{k}" for n, b, k in CASES]


def reference_indices(n: int, b: int, k: int, u: Any) -> list[int]:
    """
    The algorithm of Paparoditis and Politis (2002), Section 2, with loops.

    Observations are numbered 1, ..., n as in the paper, and ``u`` has one uniform
    draw for each block. The indices that are returned are zero-based. The block m
    starts at position m * b + 1 of the data, and its start is drawn uniformly from
    the integers J1, ..., J2 with J1 = max(1, m * b + 1 - K) and
    J2 = min(n - b + 1, m * b + 1 + K).
    """
    out: list[int] = []
    num_blocks = math.ceil(n / b)
    assert len(u) == num_blocks
    for m in range(num_blocks):
        j1 = max(1, m * b + 1 - k)
        j2 = min(n - b + 1, m * b + 1 + k)
        j1 = min(j1, j2)
        candidates = list(range(j1, j2 + 1))
        start = candidates[int(u[m] * len(candidates))]  # I_m, one-based
        # X*_{mb+j} = X_{I_m + j - 1} for j = 1, ..., b, zero-based here
        out.extend(start + j - 1 - 1 for j in range(1, b + 1))
    return out[:n]


def reference_support(n: int, b: int, k: int) -> list[list[int]]:
    """The zero-based starts that the reference can draw for each block"""
    support = []
    for m in range(math.ceil(n / b)):
        j1 = max(1, m * b + 1 - k)
        j2 = min(n - b + 1, m * b + 1 + k)
        j1 = min(j1, j2)
        support.append([j - 1 for j in range(j1, j2 + 1)])
    return support


class FixedDraws:
    """A stand-in for a RandomState that returns chosen uniform draws"""

    def __init__(self, draws: Any) -> None:
        self.draws = np.asarray(draws, dtype=float)

    def random_sample(self, size: int) -> Any:
        assert size == self.draws.shape[0]
        return self.draws.copy()


def with_draws(n: int, b: int, k: int, draws: Any) -> Any:
    bs = LocalBlockBootstrap(b, k, np.arange(n), seed=0)
    bs._generator = FixedDraws(draws)  # type: ignore[assignment]
    return bs.update_indices()


def make_generator(kind: str, seed: int) -> Generator | RandomState:
    return np.random.default_rng(seed) if kind == "generator" else RandomState(seed)


def draw(prng: Generator | RandomState, size: int) -> Any:
    if isinstance(prng, Generator):
        return prng.random(size)
    return prng.random_sample(size)


GENERATORS = ["generator", "randomstate"]


# ---- indices against the loop reference with the same uniform draws
@pytest.mark.parametrize(("n", "b", "k"), CASES, ids=CASE_IDS)
@pytest.mark.parametrize("kind", GENERATORS)
def test_indices_match_reference(n, b, k, kind):
    bs = LocalBlockBootstrap(b, k, np.arange(n), seed=make_generator(kind, 20261018))
    prng = make_generator(kind, 20261018)
    num_blocks = math.ceil(n / b)
    for _ in range(50):
        indices = bs.update_indices()
        expected = reference_indices(n, b, k, draw(prng, num_blocks))
        assert indices.dtype == np.int64
        assert_array_equal(indices, expected)


@pytest.mark.parametrize(("n", "b", "k"), CASES, ids=CASE_IDS)
def test_indices_match_reference_edge_draws(n, b, k):
    # The smallest and largest draws, and draws that are exact multiples of 1 / width
    num_blocks = math.ceil(n / b)
    top = np.nextafter(1.0, 0.0)
    for value in (0.0, top, 0.5):
        draws = np.full(num_blocks, value)
        assert_array_equal(
            with_draws(n, b, k, draws), reference_indices(n, b, k, draws)
        )
    rng = np.random.default_rng(5)
    for _ in range(20):
        draws = rng.integers(0, 8, size=num_blocks) / 8
        assert_array_equal(
            with_draws(n, b, k, draws), reference_indices(n, b, k, draws)
        )


@pytest.mark.parametrize(("n", "b", "k"), CASES, ids=CASE_IDS)
def test_exact_support(n, b, k):
    # Choose the draw of one block at a time to hit every start in its support
    support = reference_support(n, b, k)
    num_blocks = len(support)
    for m, starts in enumerate(support):
        seen = []
        for i in range(len(starts)):
            draws = np.zeros(num_blocks)
            draws[m] = (i + 0.5) / len(starts)
            indices = with_draws(n, b, k, draws)
            seen.append(int(indices[m * b]))
        assert seen == starts
        # and the limits of the draws do not leave the support
        for value in (0.0, np.nextafter(1.0, 0.0)):
            draws = np.full(num_blocks, value)
            first = int(with_draws(n, b, k, draws)[m * b])
            assert first in starts


@pytest.mark.parametrize(("n", "b", "k"), CASES, ids=CASE_IDS)
def test_blocks_are_consecutive_and_in_range(n, b, k):
    bs = LocalBlockBootstrap(b, k, np.arange(n), seed=1)
    support = reference_support(n, b, k)
    for _ in range(100):
        indices = bs.update_indices()
        assert indices.shape == (n,)
        assert indices.min() >= 0
        assert indices.max() <= n - 1
        for m, starts in enumerate(support):
            block = indices[m * b : (m + 1) * b]
            assert block[0] in starts
            assert_array_equal(np.diff(block), 1)


# Zero-based indices from an independent implementation of the algorithm of the paper
# in R (R 4.6.1, base R only, function lbb_indices in lbb_ref.R): (n, b, K, draws,
# indices). No R package implements the local block bootstrap (tseries 0.10-63
# tsbootstrap has the stationary and the moving block bootstraps only).
R_CASES = [
    (
        (23, 4, 3),
        [0, 0.17, 0.5, 0.83, 0.999, 0.31],
        [
            0,
            1,
            2,
            3,
            2,
            3,
            4,
            5,
            8,
            9,
            10,
            11,
            14,
            15,
            16,
            17,
            19,
            20,
            21,
            22,
            17,
            18,
            19,
        ],
    ),
    (
        (23, 4, 3),
        [0.99, 0.01, 0.62, 0.38, 0, 0.75],
        [
            3,
            4,
            5,
            6,
            1,
            2,
            3,
            4,
            9,
            10,
            11,
            12,
            11,
            12,
            13,
            14,
            13,
            14,
            15,
            16,
            19,
            20,
            21,
        ],
    ),
    (
        (23, 5, 2),
        [0.4, 0.9, 0.1, 0.55, 0.7],
        [
            1,
            2,
            3,
            4,
            5,
            7,
            8,
            9,
            10,
            11,
            8,
            9,
            10,
            11,
            12,
            15,
            16,
            17,
            18,
            19,
            18,
            19,
            20,
        ],
    ),
    (
        (23, 4, 0),
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [*range(20), 19, 20, 21],
    ),
    (
        (23, 4, 100),
        [0.05, 0.95, 0.5, 0.25, 0.75, 0.6],
        [
            1,
            2,
            3,
            4,
            19,
            20,
            21,
            22,
            10,
            11,
            12,
            13,
            5,
            6,
            7,
            8,
            15,
            16,
            17,
            18,
            12,
            13,
            14,
        ],
    ),
    (
        (20, 4, 6),
        [0.3, 0.6, 0.9, 0.1, 0.45],
        [2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 7, 8, 9, 10, 13, 14, 15, 16],
    ),
    ((23, 23, 7), [0.7], list(range(23))),
    ((10, 4, 2), [0.5, 0.5, 0.99], [1, 2, 3, 4, 4, 5, 6, 7, 6, 7]),
]


@pytest.mark.parametrize(
    ("params", "draws", "expected"),
    R_CASES,
    ids=[f"n{p[0]}-b{p[1]}-K{p[2]}-{i}" for i, (p, _, _) in enumerate(R_CASES)],
)
def test_indices_match_r_implementation(params, draws, expected):
    n, b, k = params
    assert_array_equal(with_draws(n, b, k, draws), expected)
    assert_array_equal(reference_indices(n, b, k, draws), expected)


def test_example_support_n23_b4_k3():
    # Block m starts in [max(0, 4m - 3), min(19, 4m + 3)]
    expected = [
        (0, 3),  # block 0
        (1, 7),  # block 1
        (5, 11),  # block 2
        (9, 15),  # block 3
        (13, 19),  # block 4
        (17, 19),  # block 5: 20 - 3 = 17 up to the last start, 19 (3 observations kept)
    ]
    support = reference_support(23, 4, 3)
    assert [(s[0], s[-1]) for s in support] == expected
    bs = LocalBlockBootstrap(4, 3, np.arange(23), seed=0)
    starts = [set() for _ in expected]
    for _ in range(3000):
        indices = bs.update_indices()
        for m, s in enumerate(starts):
            s.add(int(indices[4 * m]))
    for m, (lo, hi) in enumerate(expected):
        assert starts[m] == set(range(lo, hi + 1))
    # The last block is truncated to 3 observations
    assert bs.update_indices().shape == (23,)


def test_zero_locality():
    # b divides n: the data. Otherwise the last block ends at the last observation
    bs = LocalBlockBootstrap(4, 0, np.arange(20), seed=0)
    assert_array_equal(bs.update_indices(), np.arange(20))
    bs = LocalBlockBootstrap(4, 0, np.arange(23), seed=0)
    indices = bs.update_indices()
    assert_array_equal(indices[:20], np.arange(20))
    assert_array_equal(indices[20:], [19, 20, 21])
    x = np.arange(20.0)
    bs = LocalBlockBootstrap(5, 0, x, seed=0)
    assert_allclose(bs.cov(np.mean, 10), 0.0, atol=1e-12)


def test_large_locality_is_moving_block_support():
    n, b = 23, 4
    bs = LocalBlockBootstrap(b, n, np.arange(n), seed=0)
    support = reference_support(n, b, n)
    assert all(s == list(range(n - b + 1)) for s in support)
    starts = set()
    for _ in range(3000):
        starts.update(int(i) for i in bs.update_indices()[::b])
    assert starts == set(range(n - b + 1))


@pytest.mark.parametrize("locality", [10**12, 2**62, 2**100, np.int64(2**62)])
def test_huge_locality_does_not_overflow(locality):
    bs = LocalBlockBootstrap(4, locality, np.arange(23), seed=0)
    assert bs.locality == int(locality)
    draws = np.array([0.0, 0.5, 0.9, 0.2, 0.7, 0.99])
    assert_array_equal(
        with_draws(23, 4, int(locality), draws), reference_indices(23, 4, 23, draws)
    )
    assert bs.update_indices().shape == (23,)


def test_block_size_one_is_local_bootstrap():
    # b = 1: observation t is replaced by one of t - K, ..., t + K (clipped)
    n, k = 40, 5
    bs = LocalBlockBootstrap(1, k, np.arange(n), seed=0)
    for _ in range(200):
        indices = bs.update_indices()
        assert np.all(np.abs(indices - np.arange(n)) <= k)
        assert indices.min() >= 0
        assert indices.max() <= n - 1


# ---- the distribution of the starts
@pytest.mark.parametrize("kind", GENERATORS)
@pytest.mark.parametrize(("n", "b", "k"), [(23, 4, 3), (60, 7, 10), (23, 5, 2)])
def test_starts_are_uniform(n, b, k, kind):
    reps = 20000
    bs = LocalBlockBootstrap(b, k, np.arange(n), seed=make_generator(kind, 12345))
    starts = np.array([bs.update_indices()[::b] for _ in range(reps)])
    support = reference_support(n, b, k)
    for m, values in enumerate(support):
        counts = np.array([(starts[:, m] == v).sum() for v in values])
        assert counts.sum() == reps
        if len(values) > 1:
            result = stats.chisquare(counts)
            # Seeded, so not random. 1 / 1000 is far below any ordinary fit
            assert result.pvalue > 0.001, (m, counts)
        else:
            assert counts[0] == reps


@pytest.mark.parametrize("kind", GENERATORS)
def test_blocks_are_independent(kind):
    # The starts of two blocks with a common support have a flat joint distribution
    bs = LocalBlockBootstrap(4, 12, np.arange(40), seed=make_generator(kind, 7))
    reps = 20000
    starts = np.array([bs.update_indices()[::4] for _ in range(reps)])
    a = starts[:, 4]
    c = starts[:, 5]
    table = np.zeros((4, 4))
    qa = np.minimum((a - a.min()) * 4 // (a.max() - a.min() + 1), 3)
    qc = np.minimum((c - c.min()) * 4 // (c.max() - c.min() + 1), 3)
    for i in range(4):
        for j in range(4):
            table[i, j] = np.sum((qa == i) & (qc == j))
    assert stats.chi2_contingency(table).pvalue > 0.001


# ---- a statistical check: the mean of a trend plus noise
def trend_series(seed: int, nobs: int = 600, slope: float = 3.0) -> Any:
    t = np.arange(nobs) / nobs
    return slope * t + np.random.default_rng(seed).standard_normal(nobs)


def bootstrap_se_of_mean(bs: IIDBootstrap, reps: int) -> float:
    return float(np.sqrt(np.ravel(bs.cov(np.mean, reps))[0]))


def test_trend_plus_noise_standard_error():
    # The noise is N(0, 1), so the standard deviation of the mean is 1 / sqrt(600).
    # The circular block bootstrap resamples the trend as if it were noise and
    # inflates the standard error by a factor of about 3.
    nobs = 600
    truth = 1 / np.sqrt(nobs)
    y = trend_series(20261018, nobs)
    local = bootstrap_se_of_mean(LocalBlockBootstrap(10, 40, y, seed=1), 1000)
    circular = bootstrap_se_of_mean(CircularBlockBootstrap(10, y, seed=1), 1000)
    moving = bootstrap_se_of_mean(MovingBlockBootstrap(10, y, seed=1), 1000)
    assert abs(local / truth - 1) < 0.25
    assert circular > 2 * truth
    assert moving > 2 * truth
    assert local < 0.6 * circular


def test_trend_plus_noise_standard_error_average():
    # The average over data sets is closer to the truth than a single data set
    nobs = 600
    truth = 1 / np.sqrt(nobs)
    local = []
    for seed in range(12):
        y = trend_series(100 + seed, nobs)
        local.append(
            bootstrap_se_of_mean(LocalBlockBootstrap(10, 40, y, seed=seed), 300)
        )
    assert abs(np.mean(local) / truth - 1) < 0.15


def test_heteroskedastic_noise_standard_error():
    # The variance of the noise rises over the sample
    nobs = 600
    t = np.arange(nobs) / nobs
    scale = 0.5 + 2 * t
    truth = np.sqrt(np.sum(scale**2)) / nobs
    y = scale * np.random.default_rng(3).standard_normal(nobs)
    local = bootstrap_se_of_mean(LocalBlockBootstrap(10, 40, y, seed=1), 1000)
    assert abs(local / truth - 1) < 0.2


def test_large_locality_matches_moving_block_variance():
    y = trend_series(4, 300)
    local = bootstrap_se_of_mean(LocalBlockBootstrap(10, 300, y, seed=1), 4000)
    moving = bootstrap_se_of_mean(MovingBlockBootstrap(10, y, seed=2), 4000)
    assert abs(local / moving - 1) < 0.1


# ---- the framework: seed, reset, state, clone, properties
@pytest.mark.parametrize("kind", GENERATORS)
def test_seed_reproducibility_prng(kind):
    y = np.arange(40.0)
    one = LocalBlockBootstrap(4, 6, y, seed=make_generator(kind, 5))
    other = LocalBlockBootstrap(4, 6, y, seed=make_generator(kind, 5))
    for _ in range(5):
        assert_array_equal(one.update_indices(), other.update_indices())


def test_seed_reproducibility_values():
    y = np.arange(40.0)
    a = [d[0][0].copy() for d in LocalBlockBootstrap(4, 6, y, seed=5).bootstrap(5)]
    b = [d[0][0].copy() for d in LocalBlockBootstrap(4, 6, y, seed=5).bootstrap(5)]
    c = [d[0][0].copy() for d in LocalBlockBootstrap(4, 6, y, seed=6).bootstrap(5)]
    for x, z in zip(a, b, strict=True):
        assert_array_equal(x, z)
    assert any(not np.array_equal(x, z) for x, z in zip(a, c, strict=True))


@pytest.mark.parametrize("kind", GENERATORS)
def test_state_and_reset(kind):
    y = np.arange(50.0)
    bs = LocalBlockBootstrap(5, 8, y, seed=make_generator(kind, 23456))
    state = bs.state
    first = [d[0][0].copy() for d in bs.bootstrap(6)]
    after = bs.state
    bs.state = state
    again = [d[0][0].copy() for d in bs.bootstrap(6)]
    for x, z in zip(first, again, strict=True):
        assert_array_equal(x, z)
    bs.reset()
    reset = [d[0][0].copy() for d in bs.bootstrap(6)]
    for x, z in zip(first, reset, strict=True):
        assert_array_equal(x, z)
    if isinstance(state, dict):
        assert state != after


def test_reset_with_integer_seed_replays_reference():
    y = np.arange(46.0)
    bs = LocalBlockBootstrap(5, 8, y, seed=11)
    prng = np.random.default_rng(11)
    nb = math.ceil(46 / 5)
    for _ in range(2):
        for pos, _kw in bs.bootstrap(4):
            expected = reference_indices(46, 5, 8, draw(prng, nb))
            assert_array_equal(pos[0], y[expected])
        bs.reset()
        prng = np.random.default_rng(11)


def test_properties_and_read_only():
    bs = LocalBlockBootstrap(5, 8, np.arange(100.0), seed=0)
    assert bs.block_size == 5
    assert bs.locality == 8
    assert type(bs.locality) is int
    with pytest.raises(AttributeError, match="locality cannot be changed"):
        bs.locality = 20
    with pytest.raises(AttributeError, match="block_size cannot be changed"):
        bs.block_size = 20
    assert bs.block_size == 5
    assert bs.locality == 8
    assert bs._parameters == [5, 8]
    numpy_int = LocalBlockBootstrap(np.int32(5), np.int16(8), np.arange(20.0))
    assert type(numpy_int.locality) is int
    assert type(numpy_int.block_size) is int


def test_clone_keeps_block_size_and_locality():
    bs = LocalBlockBootstrap(7, 11, np.arange(100.0), seed=0)
    clone = bs.clone(np.arange(60.0), seed=1)
    assert isinstance(clone, LocalBlockBootstrap)
    assert clone.block_size == 7
    assert clone.locality == 11
    assert clone._parameters == [7, 11]
    assert clone.update_indices().shape == (60,)
    # Data of a different length and a keyword
    clone = bs.clone(np.arange(30.0), z=np.arange(30.0), seed=2)
    assert clone.locality == 11
    assert clone.pos_data[0].shape == (30,)
    assert "z" in clone.kw_data
    # The seed of a clone is reproducible, and the clone does not share the prng
    a = bs.clone(np.arange(60.0), seed=3).update_indices()
    b = bs.clone(np.arange(60.0), seed=3).update_indices()
    assert_array_equal(a, b)
    # A clone follows the reference for its own data length
    prng = np.random.default_rng(9)
    clone = bs.clone(np.arange(61.0), seed=9)
    expected = reference_indices(61, 7, 11, draw(prng, math.ceil(61 / 7)))
    assert_array_equal(clone.update_indices(), expected)


def test_is_subclass_of_circular_but_not_a_multiple_comparison_option():
    assert issubclass(LocalBlockBootstrap, CircularBlockBootstrap)
    assert not issubclass(LocalBlockBootstrap, MovingBlockBootstrap)
    assert not issubclass(LocalBlockBootstrap, StationaryBootstrap)
    losses = pd.DataFrame(np.random.default_rng(0).standard_normal((50, 3)) ** 2)
    for name in ("local", "lbb", "local block", "local_block"):
        with pytest.raises(ValueError, match="bootstrap must be one of"):
            SPA(losses[0], losses[[1, 2]], bootstrap=name)
        with pytest.raises(ValueError, match="bootstrap must be one of"):
            MCS(losses, 0.1, bootstrap=name)
    assert "local" not in base_module.__all__
    assert "LocalBlockBootstrap" in base_module.__all__


def test_block_larger_than_sample():
    bs = LocalBlockBootstrap(24, 3, np.arange(23.0), seed=0)
    with pytest.raises(ValueError, match=r"block_size must be an integer in \[1, 23\]"):
        bs.bootstrap(1)
    with pytest.raises(ValueError, match=r"block_size must be an integer in \[1, 23\]"):
        bs.cov(np.mean, 10)
    # b = n is allowed
    ok = LocalBlockBootstrap(23, 3, np.arange(23.0), seed=0)
    assert ok.update_indices().shape == (23,)
    assert next(ok.bootstrap(1))[0][0].shape == (23,)


def test_block_size_checked_for_clone_data():
    bs = LocalBlockBootstrap(10, 3, np.arange(100.0), seed=0)
    clone = bs.clone(np.arange(5.0))
    with pytest.raises(ValueError, match=r"block_size must be an integer in \[1, 5\]"):
        clone.cov(np.mean, 5)


# ---- string output
def test_str_and_html():
    y = pd.Series(np.arange(30.0))
    x = pd.DataFrame(np.ones((30, 2)))
    bs = LocalBlockBootstrap(5, 9, y, x, seed=0)
    expected = (
        "Local Block Bootstrap(block size: 5, locality: 9, no. pos. inputs: 2, "
        "no. keyword inputs: 0)"
    )
    assert str(bs) == expected
    assert repr(bs) == expected[:-1] + ", ID: " + hex(id(bs)) + ")"
    html = (
        "<strong>Local Block Bootstrap</strong>("
        "<strong>block size</strong>: 5, <strong>locality</strong>: 9, "
        "<strong>no. pos. inputs</strong>: 2, "
        "<strong>no. keyword inputs</strong>: 0, "
        "<strong>ID</strong>: " + hex(id(bs)) + ")"
    )
    assert bs._repr_html_() == html
    bs = LocalBlockBootstrap(block_size=5, locality=0, y=y, x=x, seed=0)
    assert str(bs) == (
        "Local Block Bootstrap(block size: 5, locality: 0, no. pos. inputs: 0, "
        "no. keyword inputs: 2)"
    )


# ---- apply, cov, var and conf_int against a replay with the reference
REPS = 200
SEED = 31337


def make_data() -> Any:
    rng = np.random.default_rng(99)
    t = np.arange(60) / 60
    return 2 * t[:, None] + rng.standard_normal((60, 2))


def replay_means(data: Any, b: int, k: int, reps: int, seed: int) -> Any:
    """Statistic (column means) of the data resampled with the reference"""
    n = data.shape[0]
    prng = np.random.default_rng(seed)
    nb = math.ceil(n / b)
    out = []
    for _ in range(reps):
        index = reference_indices(n, b, k, draw(prng, nb))
        out.append(data[index].mean(0))
    return np.array(out)


def col_mean(x: Any) -> Any:
    return np.asarray(x).mean(0)


def col_se(params: Any, x: Any) -> Any:
    x = np.asarray(x)
    return x.std(0) / np.sqrt(x.shape[0])


@pytest.fixture(scope="module")
def data() -> Any:
    return make_data()


@pytest.fixture(scope="module")
def replay(data) -> Any:
    return replay_means(data, 5, 9, REPS, SEED)


def new_bs(data: Any) -> LocalBlockBootstrap:
    return LocalBlockBootstrap(5, 9, data, seed=SEED)


def test_apply(data, replay):
    results = new_bs(data).apply(col_mean, REPS)
    assert_allclose(results, replay, rtol=1e-12, atol=1e-12)


def test_cov_and_var(data, replay):
    mean = replay.mean(0)
    errors = replay - mean
    expected_cov = errors.T @ errors / REPS
    assert_allclose(new_bs(data).cov(col_mean, REPS), expected_cov, rtol=1e-10)
    assert_allclose(new_bs(data).var(col_mean, REPS), np.diag(expected_cov), rtol=1e-10)


def test_bootstrap_iteration_applies_same_index_to_all_inputs():
    n = 46
    x = np.arange(float(n))
    y = pd.Series(np.arange(float(n)) * 2, index=pd.date_range("2020-01-01", periods=n))
    z = pd.DataFrame({"a": np.arange(float(n)) * 3, "b": -np.arange(float(n))})
    w = np.arange(float(n))[:, None] * np.ones((1, 3))
    bs = LocalBlockBootstrap(5, 8, x, y, z=z, w=w, seed=4)
    prng = np.random.default_rng(4)
    nb = math.ceil(n / 5)
    for pos, kw in bs.bootstrap(10):
        index = np.array(reference_indices(n, 5, 8, draw(prng, nb)))
        assert_array_equal(pos[0], x[index])
        assert isinstance(pos[1], pd.Series)
        assert_array_equal(pos[1].to_numpy(), 2 * index)
        assert_array_equal(pos[1].index, y.index[index])
        assert isinstance(kw["z"], pd.DataFrame)
        assert_array_equal(kw["z"]["a"].to_numpy(), 3 * index)
        assert_array_equal(kw["z"]["b"].to_numpy(), -index)
        assert_array_equal(kw["w"], w[index])
        assert_array_equal(bs.index, index)


def test_series_dataframe_and_keyword_results_agree(data):
    frame = pd.DataFrame(data, columns=["a", "b"])
    series = frame["a"]

    def mean_kw(a=None, b=None):
        return np.array([a.mean(), b.mean()])

    kw = LocalBlockBootstrap(5, 9, a=frame["a"], b=frame["b"], seed=SEED)
    expected = new_bs(data).apply(col_mean, 50)
    assert_allclose(kw.apply(mean_kw, 50), expected)
    df = LocalBlockBootstrap(5, 9, frame, seed=SEED)
    assert_allclose(df.apply(lambda f: f.mean().to_numpy(), 50), expected)
    ser = LocalBlockBootstrap(5, 9, series, seed=SEED)
    assert_allclose(ser.apply(lambda s: float(s.mean()), 50).ravel(), expected[:, 0])
    assert kw.a.shape == (60,)
    assert isinstance(kw.a, pd.Series)


def percentile_bounds(values: Any, lo: float, hi: float) -> Any:
    return np.array(
        [np.percentile(values[:, i], [lo, hi]) for i in range(values.shape[1])]
    ).T


def test_conf_int_percentile_basic_norm(data, replay):
    base = data.mean(0)
    sd = replay.std(0)
    pct = percentile_bounds(replay, 2.5, 97.5)
    ci = new_bs(data).conf_int(col_mean, REPS, method="percentile")
    assert_allclose(ci, pct, rtol=1e-10)
    ci = new_bs(data).conf_int(col_mean, REPS, method="basic")
    assert_allclose(ci, np.vstack((2 * base - pct[1], 2 * base - pct[0])), rtol=1e-10)
    z = stats.norm.ppf(0.975)
    expected = np.vstack((base - z * sd, base + z * sd))
    for method in ("norm", "var", "cov"):
        ci = new_bs(data).conf_int(col_mean, REPS, method=method)
        assert_allclose(ci, expected, rtol=1e-10)


def test_conf_int_tails_and_size(data, replay):
    pct = percentile_bounds(replay, 5.0, 95.0)
    ci = new_bs(data).conf_int(col_mean, REPS, method="percentile", size=0.9)
    assert_allclose(ci, pct, rtol=1e-10)
    low, high = np.percentile(replay, 10, axis=0), np.percentile(replay, 90, axis=0)
    ci = new_bs(data).conf_int(
        col_mean, REPS, method="percentile", size=0.9, tail="lower"
    )
    assert_allclose(ci[0], low, rtol=1e-10)
    assert np.all(np.isposinf(ci[1]))
    ci = new_bs(data).conf_int(
        col_mean, REPS, method="percentile", size=0.9, tail="upper"
    )
    assert_allclose(ci[1], high, rtol=1e-10)
    assert np.all(np.isneginf(ci[0]))


def bias_corrected_percentiles(replay: Any, base: Any, accel: Any) -> Any:
    out = []
    for i in range(replay.shape[1]):
        b = stats.norm.ppf((replay[:, i] < base[i]).mean())
        a = 0.0 if accel is None else accel[i]
        bounds = [
            stats.norm.cdf(b + (b + q) / (1 - a * (b + q)))
            for q in stats.norm.ppf([0.025, 0.975])
        ]
        out.append(np.percentile(replay[:, i], [100 * p for p in bounds]))
    return np.array(out).T


def jackknife_acceleration(data: Any) -> Any:
    n = data.shape[0]
    jk = np.array([np.delete(data, i, axis=0).mean(0) for i in range(n)])
    u = jk.mean(0) - jk
    return (u**3).sum(0) / (6 * ((u**2).sum(0)) ** 1.5)


@pytest.mark.parametrize("method", ["bc", "debiased", "bias-corrected"])
def test_conf_int_bias_corrected(data, replay, method):
    expected = bias_corrected_percentiles(replay, data.mean(0), None)
    ci = new_bs(data).conf_int(col_mean, REPS, method=method)
    assert_allclose(ci, expected, rtol=1e-10)


def test_conf_int_bca(data, replay):
    # The acceleration is the iid leave-one-out jackknife
    accel = jackknife_acceleration(data)
    expected = bias_corrected_percentiles(replay, data.mean(0), accel)
    ci = new_bs(data).conf_int(col_mean, REPS, method="bca")
    assert_allclose(ci, expected, rtol=1e-10)


def test_conf_int_studentized_std_err_func(data):
    base = data.mean(0)
    n = data.shape[0]
    prng = np.random.default_rng(SEED)
    nb = math.ceil(n / 5)
    tstats = []
    results = []
    for _ in range(REPS):
        sample = data[reference_indices(n, 5, 9, draw(prng, nb))]
        results.append(sample.mean(0))
        tstats.append((sample.mean(0) - base) / (sample.std(0) / np.sqrt(n)))
    results = np.array(results)
    tstats = np.array(tstats)
    errors = results - results.mean(0)
    std_err = np.sqrt(np.diag(errors.T @ errors / REPS))
    q = percentile_bounds(tstats, 2.5, 97.5)
    expected = np.vstack((base - q[1] * std_err, base - q[0] * std_err))
    ci = new_bs(data).conf_int(
        col_mean, REPS, method="studentized", std_err_func=col_se
    )
    assert_allclose(ci, expected, rtol=1e-10)


def test_conf_int_studentized_nested_bootstrap(data):
    bs = new_bs(data)
    ci = bs.conf_int(col_mean, 100, method="studentized", studentize_reps=40)
    assert ci.shape == (2, 2)
    assert np.all(np.isfinite(ci))
    assert np.all(ci[0] < ci[1])
    base = data.mean(0)
    assert np.all((ci[0] < base) & (base < ci[1]))
    # Reproducible
    again = new_bs(data).conf_int(
        col_mean, 100, method="studentized", studentize_reps=40
    )
    assert_allclose(ci, again)
    # The nested bootstrap is a local block bootstrap with the same b and K
    nested = []
    original = LocalBlockBootstrap.clone

    def spy(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        nested.append((type(out), out.block_size, out.locality))
        return out

    LocalBlockBootstrap.clone = spy  # type: ignore[method-assign]
    try:
        new_bs(data).conf_int(col_mean, 5, method="studentized", studentize_reps=5)
    finally:
        LocalBlockBootstrap.clone = original  # type: ignore[method-assign]
    assert nested
    assert set(nested) == {(LocalBlockBootstrap, 5, 9)}


def test_studentized_nested_is_close_to_std_err_func():
    # A loose agreement with the nested bootstrap and an analytic standard error
    y = trend_series(8, 300)[:, None]
    func = col_mean
    nested = LocalBlockBootstrap(6, 30, y, seed=1).conf_int(
        func, 200, method="studentized", studentize_reps=100
    )
    se = LocalBlockBootstrap(6, 30, y, seed=1).conf_int(
        func, 200, method="studentized", std_err_func=col_se
    )
    width_nested = float(nested[1, 0] - nested[0, 0])
    width_se = float(se[1, 0] - se[0, 0])
    assert 0.3 < width_nested / width_se < 3.0


@pytest.mark.parametrize(
    "method",
    [
        "percentile",
        "basic",
        "norm",
        "var",
        "cov",
        "bc",
        "debiased",
        "bca",
        "studentized",
    ],
)
@pytest.mark.parametrize("kind", ["array", "series", "frame"])
def test_conf_int_every_method_runs(method, kind):
    values = trend_series(21, 120)
    if kind == "array":
        inputs = values
    elif kind == "series":
        inputs = pd.Series(values, index=pd.RangeIndex(120))
    else:
        inputs = pd.DataFrame({"y": values})
    bs = LocalBlockBootstrap(6, 24, inputs, seed=2)

    def func(x):
        return np.array([np.asarray(x).mean()])

    ci = bs.conf_int(func, 200, method=method, studentize_reps=30)
    assert ci.shape == (2, 1)
    assert ci[0, 0] < ci[1, 0]
    assert ci[0, 0] < values.mean() + 0.3
    assert ci[1, 0] > values.mean() - 0.3


def test_conf_int_reuse(data):
    bs = new_bs(data)
    first = bs.conf_int(col_mean, 50, method="percentile")
    reused = bs.conf_int(col_mean, 50, method="basic", reuse=True)
    assert reused.shape == first.shape


def test_apply_extra_kwargs(data):
    def shifted(x, shift=0.0):
        return np.asarray(x).mean(0) + shift

    plain = new_bs(data).apply(shifted, 20)
    moved = new_bs(data).apply(shifted, 20, extra_kwargs={"shift": 10.0})
    assert_allclose(moved - plain, 10.0)


def test_multidimensional_data_alignment():
    n = 30
    x = np.arange(n)[:, None, None] * np.ones((1, 2, 3))
    bs = LocalBlockBootstrap(5, 4, x, seed=0)
    for pos, _ in bs.bootstrap(5):
        assert pos[0].shape == (n, 2, 3)
        assert_array_equal(pos[0][:, 0, 0], bs.index)


def test_data_not_modified_and_nan_allowed():
    y = trend_series(1, 50)
    y[3] = np.nan
    copy = y.copy()
    bs = LocalBlockBootstrap(4, 6, y, seed=0)
    result = bs.apply(np.nanmean, 20)
    assert np.all(np.isfinite(result))
    assert_array_equal(y, copy)


def test_pickle_roundtrip_keeps_parameters():
    bs = LocalBlockBootstrap(5, 7, np.arange(40.0), seed=0)
    other = pickle.loads(pickle.dumps(bs))
    assert other.block_size == 5
    assert other.locality == 7
    assert_array_equal(other.update_indices(), bs.update_indices())


def test_wrap_around_not_used():
    # Unlike the circular block bootstrap, no index is reduced modulo n
    n = 23
    bs = LocalBlockBootstrap(4, 50, np.arange(n), seed=0)
    for _ in range(500):
        idx = bs.update_indices()
        for m in range(math.ceil(n / 4)):
            block = idx[4 * m : 4 * m + 4]
            assert np.all(np.diff(block) == 1)
