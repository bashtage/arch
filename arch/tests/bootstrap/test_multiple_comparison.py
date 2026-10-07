import itertools
from typing import NamedTuple

import numpy as np
from numpy import linspace
from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
from scipy import stats

from arch.bootstrap import (
    CircularBlockBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
)
from arch.bootstrap.base import _get_random_integers
from arch.bootstrap.multiple_comparison import MCS, SPA, StepM, _KernelVariance
from arch.covariance.kernel import Bartlett

BOOTSTRAPS = {
    "sb": StationaryBootstrap,
    "cbb": CircularBlockBootstrap,
    "mbb": MovingBlockBootstrap,
}


def direct_long_run_variance(x, weights):
    """Weighted sum of the sample autocovariances computed lag by lag"""
    t = x.shape[0]
    x = x - x.mean(0)
    variances = (x**2).sum(0) / t
    for i in range(1, t):
        variances = variances + 2 * weights[i] * (x[: t - i] * x[i:]).sum(0) / t
    return variances


def stationary_bootstrap_variance(x, block_size):
    """
    Exact t * Var(mean) of the stationary bootstrap

    The indices of the stationary bootstrap are a Markov chain that moves to
    the next observation (circularly) with probability 1 - p and otherwise
    draws an index uniformly.
    """
    t = x.shape[0]
    p = 1.0 / block_size
    x = x - x.mean()
    transition = (1 - p) * np.roll(np.eye(t), 1, axis=1) + p / t
    power = np.eye(t)
    variance = 0.0
    for lag in range(t):
        covariance = np.mean(x * (power @ x))
        variance += (t if lag == 0 else 2.0 * (t - lag)) * covariance
        power = power @ transition
    return variance / t


def circular_block_bootstrap_variance(x, block_size):
    """Exact t * E[(mean* - mean)**2] of the CBB by enumerating all samples"""
    t = x.shape[0]
    num_blocks = -(-t // block_size)
    offsets = np.arange(block_size)
    means = []
    for starts in itertools.product(range(t), repeat=num_blocks):
        indices = (np.array(starts)[:, None] + offsets).ravel()[:t] % t
        means.append(x[indices].mean())
    return t * np.mean((np.array(means) - x.mean()) ** 2)


def moving_block_bootstrap_variance(x, block_size):
    """Exact t * Var(mean) of the MBB using independent, non-wrapping blocks"""
    t = x.shape[0]
    num_blocks, remainder = divmod(t, block_size)
    cumsum = np.concatenate([[0.0], np.cumsum(x)])

    def window_variance(length):
        if length == 0:
            return 0.0
        return (cumsum[length:] - cumsum[:-length]).var()

    return (num_blocks * window_variance(block_size) + window_variance(remainder)) / t


class SPAData(NamedTuple):
    rng: RandomState
    k: int
    t: int
    benchmark: np.ndarray
    models: np.ndarray
    data_index: pd.DatetimeIndex
    benchmark_series: pd.Series
    benchmark_df: pd.DataFrame
    models_df: pd.DataFrame


@pytest.fixture
def spa_data():
    rng = RandomState(23456)
    fixed_rng = stats.chi2(10)
    t = 1000
    k = 500
    benchmark = fixed_rng.rvs(t)
    models = fixed_rng.rvs((t, k))
    index = pd.date_range("2000-01-01", periods=t)
    benchmark_series = pd.Series(benchmark, index=index)
    benchmark_df = pd.DataFrame(benchmark, index=index)
    models_df = pd.DataFrame(models, index=index)
    return SPAData(
        rng, k, t, benchmark, models, index, benchmark_series, benchmark_df, models_df
    )


def test_equivalence(spa_data):
    spa = SPA(spa_data.benchmark, spa_data.models, block_size=10, reps=100, seed=23456)
    spa.compute()
    numpy_pvalues = spa.pvalues
    spa = SPA(
        spa_data.benchmark_df, spa_data.models_df, block_size=10, reps=100, seed=23456
    )
    spa.compute()
    pandas_pvalues = spa.pvalues
    assert_series_equal(numpy_pvalues, pandas_pvalues)


def test_variances_and_selection(spa_data):
    adj_models = spa_data.models + linspace(-2, 0.5, spa_data.k)
    spa = SPA(spa_data.benchmark, adj_models, block_size=10, reps=10, seed=23456)
    spa.compute()
    variances = spa._loss_diff_var
    loss_diffs = spa._loss_diff
    demeaned = spa._loss_diff - loss_diffs.mean(0)
    t = loss_diffs.shape[0]
    kernel_weights = np.zeros(t)
    p = 1 / 10.0
    for i in range(1, t):
        kernel_weights[i] = ((1.0 - (i / t)) * ((1 - p) ** i)) + (
            (i / t) * ((1 - p) ** (t - i))
        )
    direct_vars = (demeaned**2).sum(0) / t
    for i in range(1, t):
        direct_vars += (
            2 * kernel_weights[i] * (demeaned[: t - i, :] * demeaned[i:, :]).sum(0) / t
        )
    assert_allclose(direct_vars, variances)

    selection_criteria = -1.0 * np.sqrt((direct_vars / t) * 2 * np.log(np.log(t)))
    valid = loss_diffs.mean(0) >= selection_criteria
    assert_equal(valid, spa._valid_columns)

    # Bootstrap variances
    spa = SPA(
        spa_data.benchmark,
        spa_data.models[:, :20],
        block_size=10,
        reps=100,
        nested=True,
        seed=23456,
    )
    spa._compute_variance()
    demeaned = spa._loss_diff - spa._loss_diff.mean(0)
    bs = spa.bootstrap.clone(demeaned, seed=23456)
    variances = spa._loss_diff_var
    bootstrap_variances = t * bs.var(lambda x: x.mean(0), reps=100, recenter=True)
    assert_allclose(bootstrap_variances, variances)


def test_pvalues_and_critvals(spa_data):
    spa = SPA(spa_data.benchmark, spa_data.models, reps=100, seed=23456)
    spa.compute()
    simulated_vals = spa._simulated_vals
    max_stats = np.max(simulated_vals, 0)
    std_err = np.sqrt(spa._loss_diff_var / spa_data.t)
    max_loss_diff = np.max(spa._loss_diff.mean(0) / std_err, 0)
    pvalues = np.mean(max_loss_diff <= max_stats, 0)
    pvalues = pd.Series(pvalues, index=["lower", "consistent", "upper"])
    assert_series_equal(pvalues, spa.pvalues)

    crit_vals = np.percentile(max_stats, 90.0, axis=0)
    crit_vals = pd.Series(crit_vals, index=["lower", "consistent", "upper"])
    assert_series_equal(spa.critical_values(0.10), crit_vals)


def test_studentization():
    rng = RandomState(0)
    t = 500
    benchmark = rng.standard_normal(t) ** 2
    # Small but precisely estimated improvement and a large noisy improvement
    precise = benchmark - 0.05 + 0.01 * rng.standard_normal(t)
    noisy = benchmark - 0.20 + 10.0 * rng.standard_normal(t)
    models = np.column_stack([precise, noisy])

    spa = SPA(benchmark, models, block_size=10, reps=500, seed=23456)
    spa.compute()
    raw = SPA(benchmark, models, block_size=10, reps=500, studentize=False, seed=23456)
    raw.compute()

    assert np.all(spa.pvalues < 0.05)
    assert np.all(raw.pvalues > 0.05)
    assert_equal(spa.better_models(), np.array([0]))


@pytest.mark.parametrize("bootstrap", ["sb", "cbb", "mbb"])
def test_studentization_inside_bootstrap(bootstrap):
    rng = np.random.default_rng(12345)
    t, k, reps, block_size = 60, 3, 25, 4
    benchmark = rng.standard_normal(t)
    models = rng.standard_normal((t, k)) * np.array([0.5, 1.0, 3.0]) + 0.1
    spa = SPA(
        benchmark,
        models,
        block_size=block_size,
        reps=reps,
        bootstrap=bootstrap,
        seed=23456,
    )
    spa.compute()

    # Recompute using the same bootstrap draws, a standard error computed
    # from each bootstrap sample and direct sums
    loss_diff = benchmark[:, None] - models
    bs = BOOTSTRAPS[bootstrap](block_size, loss_diff, seed=23456)
    weights = _KernelVariance._implied_kernel(bs, t)
    mean = loss_diff.mean(0)
    means = [np.maximum(mean, 0.0), np.where(spa._valid_columns, mean, 0.0), mean]
    std_errs = []
    for i, bs_data in enumerate(bs.bootstrap(reps)):
        sample = bs_data[0][0]
        std_err = np.sqrt(direct_long_run_variance(sample, weights) / t)
        std_errs.append(std_err)
        for j, center in enumerate(means):
            expected = (sample.mean(0) - center) / std_err
            assert_allclose(spa._simulated_vals[:, i, j], expected)
    # The scale changes across bootstrap samples
    assert np.all(np.ptp(np.array(std_errs), axis=0) > 0)

    # The observed statistic uses the same estimator in the full sample
    full_sample_std_err = np.sqrt(direct_long_run_variance(loss_diff, weights) / t)
    assert_allclose(spa._scale(), full_sample_std_err)
    max_stats = np.max(spa._simulated_vals, 0)
    pvalues = np.mean(max_stats > np.max(mean / full_sample_std_err), 0)
    assert_allclose(spa.pvalues.to_numpy(), pvalues)


class CountingKernel:
    """Wraps a kernel variance estimator and counts the calls"""

    def __init__(self, kernel):
        self.kernel = kernel
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.kernel(x)


def forbidden(*args, **kwargs):
    raise AssertionError("This must not be used")


def small_problem(t=40, k=3, seed=12345):
    rng = np.random.default_rng(seed)
    benchmark = rng.standard_normal(t)
    models = rng.standard_normal((t, k)) * np.linspace(0.5, 2.0, k) + 0.1
    return benchmark, models


@pytest.mark.parametrize("bootstrap", ["sb", "cbb", "mbb"])
def test_kernel_used_if_studentized_and_not_nested(bootstrap, monkeypatch):
    benchmark, models = small_problem()
    reps = 12
    spa = SPA(
        benchmark, models, block_size=4, reps=reps, bootstrap=bootstrap, seed=23456
    )
    assert spa.studentize
    assert not spa.nested
    assert isinstance(spa._kernel_variance, _KernelVariance)
    counter = CountingKernel(spa._kernel_variance)
    spa._kernel_variance = counter
    # A nested bootstrap clones the bootstrap
    monkeypatch.setattr(CircularBlockBootstrap, "clone", forbidden)
    spa.compute()
    # Once for the original data, and once in each bootstrap replication
    assert counter.calls == reps + 1


@pytest.mark.parametrize("bootstrap", ["sb", "cbb", "mbb"])
def test_nested_bootstrap_replaces_kernel(bootstrap, monkeypatch):
    benchmark, models = small_problem()
    t = benchmark.shape[0]
    reps, block_size = 12, 4
    with monkeypatch.context() as patch:
        # The kernel must not even be constructed
        patch.setattr(_KernelVariance, "__init__", forbidden)
        spa = SPA(
            benchmark,
            models,
            block_size=block_size,
            reps=reps,
            bootstrap=bootstrap,
            nested=True,
            seed=23456,
        )
        assert spa.studentize
        assert spa._kernel_variance is None
        spa.compute()

    # Full sample variance: bootstrap of the demeaned loss differentials
    loss_diff = benchmark[:, None] - models
    demeaned = loss_diff - loss_diff.mean(0)
    full_sample = BOOTSTRAPS[bootstrap](block_size, demeaned, seed=23456)
    expected_var = t * full_sample.var(lambda x: x.mean(0), reps=reps, recenter=True)
    assert_allclose(spa._loss_diff_var, expected_var)
    assert_allclose(spa._scale(), np.sqrt(expected_var / t))

    # In each replication, the variance is from a bootstrap of the sample.
    # This is seeded from the generator of the bootstrap, as in confidence
    # intervals that are studentized using a nested bootstrap
    bs = BOOTSTRAPS[bootstrap](block_size, loss_diff, seed=23456)
    mean = loss_diff.mean(0)
    means = [np.maximum(mean, 0.0), np.where(spa._valid_columns, mean, 0.0), mean]
    std_errs = []
    for i, bs_data in enumerate(bs.bootstrap(reps)):
        sample = bs_data[0][0]
        seed = int(_get_random_integers(bs.generator, 2**31 - 1)[0])
        nested_bs = BOOTSTRAPS[bootstrap](block_size, sample, seed=seed)
        variances = t * nested_bs.var(lambda x: x.mean(0), reps=reps, recenter=True)
        std_err = np.sqrt(variances / t)
        std_errs.append(std_err)
        for j, center in enumerate(means):
            expected = (sample.mean(0) - center) / std_err
            assert_allclose(spa._simulated_vals[:, i, j], expected)
    assert np.all(np.ptp(np.array(std_errs), axis=0) > 0)

    # And it differs from using the kernel
    kernel = SPA(
        benchmark,
        models,
        block_size=block_size,
        reps=reps,
        bootstrap=bootstrap,
        seed=23456,
    )
    kernel.compute()
    assert not np.allclose(kernel._loss_diff_var, spa._loss_diff_var)
    assert not np.allclose(kernel._simulated_vals, spa._simulated_vals)


def test_nested_is_reproducible():
    benchmark, models = small_problem()
    spa = SPA(benchmark, models, block_size=4, reps=10, nested=True, seed=1)
    spa.compute()
    first = spa._simulated_vals.copy()
    pvalues = spa.pvalues.copy()
    again = SPA(benchmark, models, block_size=4, reps=10, nested=True, seed=1)
    again.compute()
    assert_allclose(again._simulated_vals, first)
    assert_series_equal(again.pvalues, pvalues)


@pytest.mark.parametrize("bootstrap", ["sb", "cbb", "mbb"])
def test_no_studentization_if_not_studentized(bootstrap, monkeypatch):
    benchmark, models = small_problem()
    k = models.shape[1]
    reps, block_size = 25, 4
    spa = SPA(
        benchmark,
        models,
        block_size=block_size,
        reps=reps,
        bootstrap=bootstrap,
        studentize=False,
        seed=1,
    )
    assert not spa.studentize
    assert not spa.nested
    counter = CountingKernel(spa._kernel_variance)
    spa._kernel_variance = counter
    monkeypatch.setattr(CircularBlockBootstrap, "clone", forbidden)
    spa.compute()
    # The variance is only used to select the models to re-center, once
    assert counter.calls == 1

    # Nothing is studentized in the original data
    loss_diff = benchmark[:, None] - models
    mean = loss_diff.mean(0)
    assert_equal(spa._scale(), np.ones(k))
    assert_equal(spa._studentized_mean(), mean)

    # Or in the bootstrap samples
    bs = BOOTSTRAPS[bootstrap](block_size, loss_diff, seed=1)
    means = [np.maximum(mean, 0.0), np.where(spa._valid_columns, mean, 0.0), mean]
    for i, bs_data in enumerate(bs.bootstrap(reps)):
        for j, center in enumerate(means):
            expected = bs_data[0][0].mean(0) - center
            assert_allclose(spa._simulated_vals[:, i, j], expected)

    # So the results are in the units of the loss differentials
    max_stats = np.max(spa._simulated_vals, 0)
    assert_allclose(spa.pvalues.to_numpy(), np.mean(max_stats > np.max(mean), 0))
    crit_vals = np.percentile(max_stats, 95.0, axis=0)
    assert_allclose(spa.critical_values(0.05).to_numpy(), crit_vals)
    expected_better = np.argwhere(mean > crit_vals[1]).flatten()
    assert_equal(spa.better_models(0.05), expected_better)


@pytest.mark.parametrize("procedure", [SPA, StepM])
def test_nested_requires_studentize(procedure):
    benchmark, models = small_problem()
    with pytest.raises(
        ValueError, match=r"nested can only be True when studentize is True"
    ):
        procedure(benchmark, models, studentize=False, nested=True)
    for studentize, nested in ((True, False), (True, True), (False, False)):
        procedure(benchmark, models, studentize=studentize, nested=nested)


def test_stepm_studentization_options():
    benchmark, models = small_problem()
    stepm = StepM(benchmark, models, reps=10, studentize=False, seed=1)
    assert not stepm.spa.studentize
    assert_equal(stepm.spa._scale(), np.ones(models.shape[1]))
    stepm = StepM(benchmark, models, reps=10, nested=True, seed=1)
    assert stepm.spa.nested
    assert stepm.spa._kernel_variance is None
    stepm.compute()
    stepm = StepM(benchmark, models, reps=10, seed=1)
    assert isinstance(stepm.spa._kernel_variance, _KernelVariance)
    stepm.compute()


@pytest.mark.parametrize("bootstrap", ["sb", "cbb", "mbb"])
def test_variance_uses_kernel_of_bootstrap(spa_data, bootstrap):
    spa = SPA(
        spa_data.benchmark,
        spa_data.models[:, :10],
        block_size=12,
        reps=10,
        bootstrap=bootstrap,
        seed=1,
    )
    spa.compute()
    weights = _KernelVariance._implied_kernel(spa.bootstrap, spa_data.t)
    assert_allclose(
        spa._loss_diff_var, direct_long_run_variance(spa._loss_diff, weights)
    )


def test_bootstrap_kernels_differ(spa_data):
    variances = {}
    for bootstrap in BOOTSTRAPS:
        spa = SPA(
            spa_data.benchmark,
            spa_data.models[:, :10],
            block_size=12,
            reps=10,
            bootstrap=bootstrap,
        )
        spa.compute()
        variances[bootstrap] = spa._loss_diff_var
    assert not np.allclose(variances["sb"], variances["cbb"])
    assert not np.allclose(variances["cbb"], variances["mbb"])


def test_zero_loss_differential():
    rng = np.random.default_rng(0)
    benchmark = rng.standard_normal(100)
    models = np.column_stack([benchmark, benchmark - 0.1 + rng.standard_normal(100)])
    spa = SPA(benchmark, models, block_size=5, reps=50, seed=1)
    spa.compute()
    assert np.all(np.isfinite(spa._simulated_vals))
    assert_equal(spa._simulated_vals[0], np.zeros((50, 3)))
    assert np.all(np.isfinite(spa.pvalues))


@pytest.mark.parametrize("bootstrap", ["sb", "cbb", "mbb"])
def test_sparse_loss_differential(bootstrap):
    rng = np.random.default_rng(0)
    t, reps = 200, 300
    benchmark = rng.standard_normal(t)
    models = benchmark[:, None] + rng.standard_normal((t, 3))
    models[:, 0] -= 0.5
    # This model differs from the benchmark in a single period, so its loss
    # differential is constant in the bootstrap samples that omit this period
    sparse = benchmark.copy()
    sparse[17] += 1.0
    models = np.column_stack([models, sparse])
    spa = SPA(benchmark, models, block_size=10, reps=reps, bootstrap=bootstrap, seed=1)
    spa.compute()

    loss_diff = benchmark[:, None] - models
    bs = BOOTSTRAPS[bootstrap](10, loss_diff, seed=1)
    mean = loss_diff.mean(0)
    full_sample_std_err = spa._scale()
    num_constant = 0
    for i, bs_data in enumerate(bs.bootstrap(reps)):
        sample = bs_data[0][0]
        if np.ptp(sample[:, 3]) == 0.0:
            num_constant += 1
            # Lower and upper
            for j, center in ((0, np.maximum(mean, 0.0)), (2, mean)):
                expected = (sample[:, 3].mean() - center[3]) / full_sample_std_err[3]
                assert_allclose(spa._simulated_vals[3, i, j], expected)
    assert num_constant > 25
    assert np.max(np.abs(spa._simulated_vals)) < 100
    # The sparse model does not affect the ability to detect the better model
    assert np.all(spa.pvalues < 0.05)


@pytest.mark.parametrize("bootstrap", ["sb", "cbb", "mbb"])
@pytest.mark.parametrize(
    ("t", "block_size"), [(50, 5), (53, 7), (40, 40), (30, 1), (25, 40)]
)
def test_kernel_variance_matches_direct_sum(bootstrap, t, block_size):
    rng = np.random.default_rng(t * block_size)
    x = rng.standard_normal((t, 4)).cumsum(0) * 0.1 + rng.standard_normal((t, 4))
    bs = BOOTSTRAPS[bootstrap](block_size, x)
    weights = _KernelVariance._implied_kernel(bs, t)
    assert_equal(weights.shape, (t,))
    assert_allclose(weights[0], 1.0)
    expected = direct_long_run_variance(x, weights)
    assert_allclose(_KernelVariance(bs, t)(x), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("block_size", [1, 3, 12, 20])
def test_stationary_kernel_is_bootstrap_variance(block_size):
    rng = np.random.default_rng(block_size)
    t = 12
    x = rng.standard_normal(t) + np.sin(np.arange(t))
    bs = StationaryBootstrap(block_size, x)
    estimate = _KernelVariance(bs, t)(x[:, None])[0]
    assert_allclose(estimate, stationary_bootstrap_variance(x, block_size))


@pytest.mark.parametrize(("t", "block_size"), [(6, 3), (7, 3), (8, 4), (5, 7), (6, 1)])
def test_circular_kernel_is_bootstrap_variance(t, block_size):
    rng = np.random.default_rng(t * block_size)
    x = rng.standard_normal(t) + np.sin(np.arange(t))
    bs = CircularBlockBootstrap(block_size, x)
    estimate = _KernelVariance(bs, t)(x[:, None])[0]
    expected = circular_block_bootstrap_variance(x, block_size)
    # The mean is constant when block_size > t, so the variance is 0
    assert_allclose(estimate, expected, atol=1e-12)


def test_moving_block_kernel_is_bartlett():
    rng = np.random.default_rng(0)
    t, block_size = 60, 5
    x = rng.standard_normal((t, 3)).cumsum(0) * 0.2 + rng.standard_normal((t, 3))
    bs = MovingBlockBootstrap(block_size, x)
    bartlett = Bartlett(x, bandwidth=block_size - 1, center=True, force_int=True)
    expected = np.diag(bartlett.cov.long_run)
    assert_allclose(_KernelVariance(bs, t)(x), expected)


@pytest.mark.parametrize(("t", "block_size"), [(250, 16), (100, 10)])
def test_moving_block_kernel_approximates_bootstrap_variance(t, block_size):
    rng = np.random.default_rng(t)
    ratios = []
    for _ in range(100):
        shocks = rng.standard_normal(t + 50)
        x = np.zeros(t + 50)
        for i in range(1, t + 50):
            x[i] = 0.5 * x[i - 1] + shocks[i]
        x = x[50:]
        bs = MovingBlockBootstrap(block_size, x)
        ratios.append(
            _KernelVariance(bs, t)(x[:, None])[0]
            / moving_block_bootstrap_variance(x, block_size)
        )
    # Equal up to edge effects that vanish as block_size / t -> 0
    assert abs(np.mean(ratios) - 1.0) < 0.04


def test_errors(spa_data):
    spa = SPA(spa_data.benchmark, spa_data.models, reps=100)

    with pytest.raises(
        RuntimeError, match=r"compute must be called before pvalues are available"
    ):
        _ = spa.pvalues

    with pytest.raises(
        RuntimeError, match=r"compute must be called before pvalues are available"
    ):
        _ = spa.critical_values()
    with pytest.raises(
        RuntimeError, match=r"compute must be called before pvalues are available"
    ):
        _ = spa.better_models()

    with pytest.raises(ValueError, match=r"Unknown bootstrap: unknown"):
        _ = SPA(spa_data.benchmark, spa_data.models, bootstrap="unknown")
    spa.compute()
    with pytest.raises(ValueError, match=r"Unknown pvalue type"):
        _ = spa.better_models(pvalue_type="unknown")
    with pytest.raises(ValueError, match=r"pvalue must be in \(0,1\)"):
        _ = spa.critical_values(pvalue=1.0)


def test_str_repr(spa_data):
    spa = SPA(spa_data.benchmark, spa_data.models)
    expected = "SPA(studentization: asymptotic, bootstrap: " + str(spa.bootstrap) + ")"
    assert_equal(str(spa), expected)
    expected = expected[:-1] + ", ID: " + hex(id(spa)) + ")"
    assert_equal(spa.__repr__(), expected)

    expected = (
        "<strong>SPA</strong>("
        "<strong>studentization</strong>: asymptotic, "
        "<strong>bootstrap</strong>: "
        + str(spa.bootstrap)
        + ", <strong>ID</strong>: "
        + hex(id(spa))
        + ")"
    )

    assert_equal(spa._repr_html_(), expected)
    spa = SPA(spa_data.benchmark, spa_data.models, studentize=False, bootstrap="cbb")
    expected = "SPA(studentization: none, bootstrap: " + str(spa.bootstrap) + ")"
    assert_equal(str(spa), expected)

    spa = SPA(
        spa_data.benchmark, spa_data.models, nested=True, bootstrap="moving_block"
    )
    expected = "SPA(studentization: bootstrap, bootstrap: " + str(spa.bootstrap) + ")"
    assert_equal(str(spa), expected)


@pytest.mark.parametrize("bootstrap", ["sb", "cbb", "mbb"])
def test_bootstrap_cannot_be_replaced(spa_data, bootstrap):
    models = spa_data.models[:, :5]
    spa = SPA(spa_data.benchmark, models, bootstrap=bootstrap, reps=10)
    stepm = StepM(spa_data.benchmark, models, bootstrap=bootstrap, reps=10)
    mcs = MCS(models, 0.05, bootstrap=bootstrap, reps=10)
    replacement = CircularBlockBootstrap(10, np.ones(100))
    for procedure in (spa, stepm, mcs):
        assert isinstance(procedure.bootstrap, BOOTSTRAPS[bootstrap])
        with pytest.raises(AttributeError):
            procedure.bootstrap = replacement
        assert procedure.bootstrap is not replacement
    assert stepm.bootstrap is stepm.spa.bootstrap


def test_seed_reset(spa_data):
    spa = SPA(spa_data.benchmark, spa_data.models, reps=10, seed=23456)
    initial_state = spa.bootstrap.state
    spa.compute()
    spa.reset()
    assert spa._pvalues == {}
    assert_equal(spa.bootstrap.state["state"]["state"], initial_state["state"]["state"])
    assert_equal(spa.bootstrap.state["state"]["inc"], initial_state["state"]["inc"])
    assert_equal(spa.bootstrap.state["has_uint32"], initial_state["has_uint32"])
    assert_equal(spa.bootstrap.state["uinteger"], initial_state["uinteger"])


def test_spa_nested(spa_data):
    spa = SPA(spa_data.benchmark, spa_data.models[:, :10], nested=True, reps=20)
    spa.compute()
    assert np.all((spa.pvalues >= 0.0) & (spa.pvalues <= 1.0))


def test_bootstrap_selection(spa_data):
    spa = SPA(spa_data.benchmark, spa_data.models, bootstrap="sb")
    assert isinstance(spa.bootstrap, StationaryBootstrap)
    spa = SPA(spa_data.benchmark, spa_data.models, bootstrap="cbb")
    assert isinstance(spa.bootstrap, CircularBlockBootstrap)
    spa = SPA(spa_data.benchmark, spa_data.models, bootstrap="circular")
    assert isinstance(spa.bootstrap, CircularBlockBootstrap)
    spa = SPA(spa_data.benchmark, spa_data.models, bootstrap="mbb")
    assert isinstance(spa.bootstrap, MovingBlockBootstrap)
    spa = SPA(spa_data.benchmark, spa_data.models, bootstrap="moving block")
    assert isinstance(spa.bootstrap, MovingBlockBootstrap)


def test_single_model(spa_data):
    spa = SPA(spa_data.benchmark, spa_data.models[:, 0])
    spa.compute()

    spa = SPA(spa_data.benchmark_series, spa_data.models_df.iloc[:, 0])
    spa.compute()


class TestStepM:
    @classmethod
    def setup_class(cls):
        cls.rng = RandomState(23456)
        fixed_rng = stats.chi2(10)
        cls.t = t = 1000
        cls.k = k = 500
        cls.benchmark = fixed_rng.rvs(t)
        cls.models = fixed_rng.rvs((t, k))
        index = pd.date_range("2000-01-01", periods=t)
        cls.benchmark_series = pd.Series(cls.benchmark, index=index)
        cls.benchmark_df = pd.DataFrame(cls.benchmark, index=index)
        cols = ["col_" + str(i) for i in range(cls.k)]
        cls.models_df = pd.DataFrame(cls.models, index=index, columns=cols)

    def test_equivalence(self):
        adj_models = self.models - linspace(-2.0, 2.0, self.k)
        stepm = StepM(self.benchmark, adj_models, size=0.20, reps=200, seed=23456)
        stepm.compute()

        adj_models = self.models_df - linspace(-2.0, 2.0, self.k)
        stepm_pandas = StepM(
            self.benchmark_series, adj_models, size=0.20, reps=200, seed=23456
        )
        stepm_pandas.compute()
        assert isinstance(stepm_pandas.superior_models, list)
        members = adj_models.columns.isin(stepm_pandas.superior_models)
        numeric_locs = np.argwhere(members).squeeze()
        numeric_locs.sort()
        assert_equal(np.array(stepm.superior_models), numeric_locs)

    def test_superior_models(self):
        adj_models = self.models - linspace(-1.0, 1.0, self.k)
        stepm = StepM(self.benchmark, adj_models, reps=120)
        stepm.compute()
        superior_models = stepm.superior_models
        assert len(superior_models) > 0
        spa = SPA(self.benchmark, adj_models, reps=120)
        spa.compute()
        assert isinstance(spa.pvalues, pd.Series)
        spa.critical_values(0.05)
        spa.better_models(0.05)
        adj_models = self.models_df - linspace(-3.0, 3.0, self.k)
        stepm = StepM(self.benchmark_series, adj_models, reps=120)
        stepm.compute()
        superior_models = stepm.superior_models
        assert len(superior_models) > 0

    def test_str_repr(self):
        stepm = StepM(self.benchmark_series, self.models, size=0.10)
        expected = (
            "StepM(FWER (size): 0.10, studentization: "
            "asymptotic, bootstrap: " + str(stepm.spa.bootstrap) + ")"
        )
        assert_equal(str(stepm), expected)
        expected = expected[:-1] + ", ID: " + hex(id(stepm)) + ")"
        assert_equal(stepm.__repr__(), expected)

        expected = (
            "<strong>StepM</strong>("
            "<strong>FWER (size)</strong>: 0.10, "
            "<strong>studentization</strong>: asymptotic, "
            "<strong>bootstrap</strong>: "
            + str(stepm.spa.bootstrap)
            + ", "
            + "<strong>ID</strong>: "
            + hex(id(stepm))
            + ")"
        )

        assert_equal(stepm._repr_html_(), expected)

        stepm = StepM(self.benchmark_series, self.models, size=0.05, studentize=False)
        expected = (
            "StepM(FWER (size): 0.05, studentization: none, "
            "bootstrap: " + str(stepm.spa.bootstrap) + ")"
        )
        assert_equal(expected, str(stepm))

    def test_single_model(self):
        stepm = StepM(self.benchmark, self.models[:, 0], size=0.10)
        stepm.compute()

        stepm = StepM(self.benchmark_series, self.models_df.iloc[:, 0])
        stepm.compute()

    def test_all_superior(self):
        adj_models = self.models - 100.0
        stepm = StepM(self.benchmark, adj_models, size=0.10)
        stepm.compute()
        assert_equal(len(stepm.superior_models), self.models.shape[1])

    def test_all_superior_multiple_rounds(self):
        adj_models = self.models - self.models.mean(0) - 2.0
        adj_models /= linspace(1.0, 1000.0, self.k)
        adj_models += self.benchmark[:, None]
        stepm = StepM(
            self.benchmark, adj_models, reps=200, studentize=False, seed=23456
        )
        stepm.spa.compute()
        assert 0 < len(stepm.spa.better_models()) < self.k
        stepm.compute()
        assert_equal(len(stepm.superior_models), self.models.shape[1])

    def test_errors(self):
        stepm = StepM(self.benchmark, self.models, size=0.10)
        with pytest.raises(RuntimeError):
            _ = stepm.superior_models

    def test_exact_ties(self):
        adj_models = self.models_df - 100.0
        adj_models.iloc[:, :2] -= adj_models.iloc[:, :2].mean()
        adj_models.iloc[:, :2] += self.benchmark_df.mean().iloc[0]
        stepm = StepM(self.benchmark_df, adj_models, size=0.10)
        stepm.compute()
        assert_equal(len(stepm.superior_models), self.models.shape[1] - 2)


class TestMCS:
    @classmethod
    def setup_class(cls):
        cls.rng = RandomState(23456)
        fixed_rng = stats.chi2(10)
        cls.t = t = 1000
        cls.k = k = 50
        cls.losses = fixed_rng.rvs((t, k))
        index = pd.date_range("2000-01-01", periods=t)
        cls.losses_df = pd.DataFrame(cls.losses, index=index)

    def test_r_method(self):
        def r_step(losses, indices):
            # A basic but direct implementation of the r method
            k = losses.shape[1]
            b = len(indices)
            mean_diffs = losses.mean(0)
            loss_diffs = np.zeros((k, k))
            variances = np.zeros((k, k))
            bs_diffs = np.zeros(b)
            stat_candidates = []
            for i in range(k):
                for j in range(i, k):
                    if i == j:
                        variances[i, i] = 1.0
                        loss_diffs[i, j] = 0.0
                        continue
                    loss_diffs_vec = losses[:, i] - losses[:, j]
                    loss_diffs_vec = loss_diffs_vec - loss_diffs_vec.mean()
                    loss_diffs[i, j] = mean_diffs[i] - mean_diffs[j]
                    loss_diffs[j, i] = mean_diffs[j] - mean_diffs[i]
                    for n in range(b):
                        # Compute bootstrapped versions
                        bs_diffs[n] = loss_diffs_vec[indices[n]].mean()
                    variances[j, i] = variances[i, j] = (bs_diffs**2).mean()
                    std_diffs = np.abs(bs_diffs) / np.sqrt(variances[i, j])
                    stat_candidates.append(std_diffs)
            stat_candidates = np.array(stat_candidates).T
            stat_distn = np.max(stat_candidates, 1)
            std_loss_diffs = loss_diffs / np.sqrt(variances)
            stat = np.max(std_loss_diffs)
            pval = np.mean(stat <= stat_distn)
            loc = np.argwhere(std_loss_diffs == stat)
            drop_index = loc.flat[0]
            return pval, drop_index

        losses = self.losses[:, :10]  # Limit size
        mcs = MCS(losses, 0.05, reps=200, seed=23456)
        mcs.compute()
        m = 5  # Number of direct
        pvals = np.zeros(m) * np.nan
        indices = np.zeros(m) * np.nan
        for i in range(m):
            removed = list(indices[np.isfinite(indices)])
            include = list(set(range(10)).difference(removed))
            include.sort()
            pval, drop_index = r_step(
                losses[:, np.array(include)], mcs._bootstrap_indices
            )
            pvals[i] = pval if i == 0 else np.max([pvals[i - 1], pval])
            indices[i] = include[drop_index]
        direct = pd.DataFrame(
            pvals, index=np.array(indices, dtype=np.int64), columns=["Pvalue"]
        )
        direct.index.name = "Model index"
        assert_frame_equal(mcs.pvalues.iloc[:m], direct)

    def test_max_method(self):
        def max_step(losses, indices):
            # A basic but direct implementation of the max method
            k = losses.shape[1]
            b = len(indices)
            loss_errors = losses - losses.mean(0)
            stats = np.zeros((b, k))
            for n in range(b):
                # Compute bootstrapped versions
                bs_loss_errors = loss_errors[indices[n]]
                stats[n] = bs_loss_errors.mean(0) - bs_loss_errors.mean()
            variances = (stats**2).mean(0)
            std_devs = np.sqrt(variances)
            stat_dist = np.max(stats / std_devs, 1)

            test_stat = losses.mean(0) - losses.mean()
            std_test_stat = test_stat / std_devs
            test_stat = np.max(std_test_stat)
            pval = (test_stat < stat_dist).mean()
            drop_index = np.argwhere(std_test_stat == test_stat).squeeze()
            return pval, drop_index, std_devs

        losses = self.losses[:, :10]  # Limit size
        mcs = MCS(losses, 0.05, reps=200, method="max", seed=23456)
        mcs.compute()
        m = 8  # Number of direct
        pvals = np.zeros(m) * np.nan
        indices = np.zeros(m) * np.nan
        for i in range(m):
            removed = list(indices[np.isfinite(indices)])
            include = list(set(range(10)).difference(removed))
            include.sort()
            pval, drop_index, _ = max_step(
                losses[:, np.array(include)], mcs._bootstrap_indices
            )
            pvals[i] = pval if i == 0 else np.max([pvals[i - 1], pval])
            indices[i] = include[drop_index]
        direct = pd.DataFrame(
            pvals, index=np.array(indices, dtype=np.int64), columns=["Pvalue"]
        )
        direct.index.name = "Model index"
        assert_frame_equal(mcs.pvalues.iloc[:m], direct)

    def test_output_types(self):
        mcs = MCS(self.losses_df, 0.05, reps=100, block_size=10, method="r")
        mcs.compute()
        assert isinstance(mcs.included, list)
        assert isinstance(mcs.excluded, list)
        assert isinstance(mcs.pvalues, pd.DataFrame)

    def test_mcs_error(self):
        mcs = MCS(self.losses_df, 0.05, reps=100, block_size=10, method="r")
        with pytest.raises(
            RuntimeError, match=r"Must call compute before accessing results"
        ):
            _ = mcs.included

    def test_errors(self):
        with pytest.raises(ValueError, match=r"losses must have at least two columns"):
            MCS(self.losses[:, 1], 0.05)
        mcs = MCS(
            self.losses,
            0.05,
            reps=100,
            block_size=10,
            method="max",
            bootstrap="circular",
        )
        mcs.compute()
        mcs = MCS(
            self.losses,
            0.05,
            reps=100,
            block_size=10,
            method="max",
            bootstrap="moving block",
        )
        mcs.compute()
        with pytest.raises(ValueError, match=r"Unknown bootstrap: unknown"):
            MCS(self.losses, 0.05, bootstrap="unknown")

    def test_str_repr(self):
        mcs = MCS(self.losses, 0.05)
        expected = "MCS(size: 0.05, bootstrap: " + str(mcs.bootstrap) + ")"
        assert_equal(str(mcs), expected)
        expected = expected[:-1] + ", ID: " + hex(id(mcs)) + ")"
        assert_equal(mcs.__repr__(), expected)
        expected = (
            "<strong>MCS</strong>("
            "<strong>size</strong>: 0.05, "
            "<strong>bootstrap</strong>: "
            + str(mcs.bootstrap)
            + ", "
            + "<strong>ID</strong>: "
            + hex(id(mcs))
            + ")"
        )
        assert_equal(mcs._repr_html_(), expected)

    def test_all_models_have_pval(self):
        losses = self.losses_df.iloc[:, :20]
        mcs = MCS(losses, 0.05, reps=200, seed=23456)
        mcs.compute()
        nan_locs = np.isnan(mcs.pvalues.iloc[:, 0])
        assert not nan_locs.any()

    def test_exact_ties(self):
        losses = self.losses_df.iloc[:, :20].copy()
        tied_mean = losses.mean().median()
        losses.iloc[:, 10:] -= losses.iloc[:, 10:].mean()
        losses.iloc[:, 10:] += tied_mean
        mcs = MCS(losses, 0.05, reps=200, seed=23456)
        mcs.compute()

    def test_missing_included_max(self):
        losses = self.losses_df.iloc[:, :20].copy()
        losses = losses.values + 5 * np.arange(20)[None, :]
        mcs = MCS(losses, 0.05, reps=200, method="max", seed=23456)
        mcs.compute()
        assert len(mcs.included) > 0
        assert (len(mcs.included) + len(mcs.excluded)) == 20


def test_bad_values():
    # GH 654
    qlike = np.array([[0.38443391, 0.39939706, 0.2619653]])
    q = MCS(qlike, size=0.05, method="max")
    with pytest.warns(RuntimeWarning, match=r"During computation of a step"):
        q.compute()
