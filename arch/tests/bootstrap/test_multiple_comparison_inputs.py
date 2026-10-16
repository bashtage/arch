import threading

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.bootstrap import StationaryBootstrap
from arch.bootstrap.multiple_comparison import MCS, SPA, StepM


def reference_mcs_r(losses, block_size, reps, seed):
    """
    Hansen, Lunde and Nason (2011) T_R elimination with explicit resampling

    Returns the models in the order of elimination and the (non-monotone)
    p-values. A tie for the largest statistic eliminates the first model.
    """
    t, k = losses.shape
    bs = StationaryBootstrap(block_size, np.arange(t), seed=seed)
    indices = [bs.update_indices() for _ in range(reps)]
    mean = losses.mean(0)
    boot_mean = np.array([losses[idx].mean(0) for idx in indices])
    dbar = mean[:, None] - mean[None, :]
    dstar = boot_mean[:, :, None] - boot_mean[:, None, :]
    var_ij = ((dstar - dbar) ** 2).mean(0) + np.eye(k)
    alive = list(range(k))
    order, pvals = [], []
    while len(alive) > 1:
        a = np.array(alive)
        sd = np.sqrt(var_ij[np.ix_(a, a)])
        t_ij = dbar[np.ix_(a, a)] / sd
        t_star = ((dstar[:, a][:, :, a] - dbar[np.ix_(a, a)]) / sd).max(axis=(1, 2))
        i, _ = np.unravel_index(np.argmax(t_ij), t_ij.shape)
        order.append(alive[i])
        pvals.append(float((t_star > t_ij.max()).mean()))
        alive.pop(i)
    order.append(alive[0])
    pvals.append(1.0)
    return order, np.maximum.accumulate(pvals)


def tied_losses(design):
    if design == "two-models":
        # Equal mean losses, different patterns
        a = np.zeros(32)
        a[:8] = 1.0
        b = np.zeros(32)
        b[11:19] = 1.0
        return np.column_stack([a, b])
    # b = -a so both have a zero mean and the same variance, and c = -1:
    # the statistics of a and b are exactly tied. The values are dyadic.
    a = np.tile([-1.0, 0.5, 0.5, 1.0, -0.5, -0.5, 0.25, -0.25], 8)
    c = -np.ones(a.shape[0])
    return np.column_stack([a, -a, c])


@pytest.mark.parametrize("design", ["two-models", "three-models"])
def test_mcs_r_tied_worst_models(design):
    # The worst models have exactly equal statistics
    losses = tied_losses(design)
    if design == "two-models":
        assert losses[:, 0].mean() == losses[:, 1].mean()
    reps, block_size, seed = 128, 4, 12345
    mcs = MCS(losses, 0.1, reps=reps, block_size=block_size, method="R", seed=seed)
    mcs.compute()
    order, pvals = reference_mcs_r(losses, block_size, reps, seed)
    pvalues = mcs.pvalues
    assert list(pvalues.index) == order
    assert np.all((pvalues["Pvalue"] >= 0) & (pvalues["Pvalue"] <= 1))
    assert_allclose(pvalues["Pvalue"].to_numpy(), pvals, rtol=0, atol=1e-12)


def run_with_timeout(func, timeout=10.0):
    """Run func in a thread so that a hang fails the test rather than blocking"""
    outcome = {}

    def target():
        try:
            outcome["value"] = func()
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail("The computation did not finish. It would loop forever.")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


@pytest.fixture(scope="module")
def finite_losses():
    rng = np.random.default_rng(0)
    benchmark = rng.standard_normal(60) ** 2
    models = benchmark[:, None] + rng.standard_normal((60, 3))
    return benchmark, models


@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"]
)
@pytest.mark.parametrize("method", ["R", "max"])
@pytest.mark.parametrize("frame", [False, True], ids=["array", "frame"])
def test_mcs_non_finite_losses(finite_losses, bad_value, method, frame):
    losses = np.column_stack(finite_losses)
    losses[3, 2] = bad_value
    if frame:
        losses = pd.DataFrame(losses, columns=list("abcd"))
    with pytest.raises(ValueError, match="losses must be finite"):
        run_with_timeout(lambda: MCS(losses, 0.1, reps=5, method=method, seed=1))


@pytest.mark.parametrize("method", ["R", "max"])
def test_mcs_non_finite_statistic_does_not_hang(finite_losses, method):
    # If non-finite losses get past the constructor, the elimination loop must
    # stop rather than run forever
    losses = np.column_stack(finite_losses)
    mcs = MCS(losses, 0.1, reps=5, method=method, seed=1)
    mcs._losses_arr[3, 2] = np.nan
    with pytest.raises(RuntimeError, match="statistic is not finite"):
        run_with_timeout(mcs.compute)


@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"]
)
@pytest.mark.parametrize("procedure", [SPA, StepM])
@pytest.mark.parametrize("where", ["benchmark", "models"])
@pytest.mark.parametrize("frame", [False, True], ids=["array", "frame"])
def test_spa_stepm_non_finite_losses(finite_losses, bad_value, procedure, where, frame):
    benchmark, models = (np.array(v, copy=True) for v in finite_losses)
    if where == "benchmark":
        benchmark[4] = bad_value
    else:
        models[4, 1] = bad_value
    if frame:
        benchmark = pd.Series(benchmark, name="bench")
        models = pd.DataFrame(models, columns=list("abc"))
    with pytest.raises(ValueError, match=f"{where} must be finite"):
        run_with_timeout(lambda: procedure(benchmark, models, reps=5, seed=1))


@pytest.mark.parametrize(
    ("procedure", "better_rate"), [(SPA, 0.3), (StepM, 0.1)], ids=["SPA", "StepM"]
)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint64, np.bool_])
def test_unsigned_integer_and_bool_losses(procedure, better_rate, dtype):
    # Unsigned integer differences wrap around and bool cannot be subtracted
    rng = np.random.default_rng(0)
    t = 150
    benchmark = (rng.random(t) < 0.4).astype(float)
    # The second model has a smaller rate of 1s, the loss, than the benchmark
    models = np.column_stack(
        [
            (rng.random(t) < 0.4),
            (rng.random(t) < better_rate),
            (rng.random(t) < 0.4),
        ]
    ).astype(float)
    kwargs = {"reps": 200, "seed": 3, "block_size": 5}
    expected = procedure(benchmark, models, **kwargs)
    expected.compute()
    actual = procedure(benchmark.astype(dtype), models.astype(dtype), **kwargs)
    actual.compute()
    if procedure is SPA:
        assert_allclose(actual.pvalues.to_numpy(), expected.pvalues.to_numpy())
        assert_allclose(actual.critical_values(), expected.critical_values())
        # The p-values are informative, neither 0 nor 1
        assert np.all((expected.pvalues > 0.05) & (expected.pvalues < 0.95))
        loss_diff = actual._loss_diff
    else:
        assert actual.superior_models == expected.superior_models
        assert expected.superior_models == [1]
        loss_diff = actual.spa._loss_diff
    assert_allclose(loss_diff, benchmark[:, None] - models)
