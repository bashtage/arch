import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from arch.bootstrap import (
    IIDBootstrap,
)


def _shrunk_mean(y, params=None):
    # Semi-parametric statistic: the sample mean when params is None,
    # otherwise params[0] plus half of the deviation of the mean from params[0]
    if params is None:
        return np.array([y.mean()])
    return np.array([params[0] + 0.5 * (y.mean() - params[0])])


def _simulated_mean(y, params=None, state=None):
    if state is None:
        return np.array([y.mean()])
    return np.array([params[0] + state.standard_normal(y.shape[0]).mean()])


@pytest.mark.parametrize("sampling", ["semi", "parametric"])
@pytest.mark.parametrize("method", ["percentile", "studentized"])
@pytest.mark.parametrize("use_extra", [False, True], ids=["no-extra", "extra"])
def test_simulation_keywords_not_in_kw_data(sampling, method, use_extra):
    # sampling="semi" and "parametric" add params (and state) to the keywords
    # used to call func. These must not be added to the bootstrap's data
    y = np.random.default_rng(0).standard_normal(50)

    def func(y, params=None, state=None, flag=0):
        if state is not None:
            return np.array([params[0] + state.standard_normal(y.shape[0]).mean()])
        if params is not None:
            return np.array([params[0] + 0.5 * (y.mean() - params[0])])
        return np.array([y.mean()])

    bs = IIDBootstrap(y=y, seed=0)
    extra_kwargs = {"flag": 1} if use_extra else None
    bs.conf_int(
        func,
        20,
        method=method,
        sampling=sampling,
        studentize_reps=10,
        extra_kwargs=extra_kwargs,
    )
    assert sorted(bs.kw_data) == ["y"]
    assert sorted(bs._kwargs) == ["y"]
    assert_equal(bs._kwargs["y"], y)


@pytest.mark.parametrize("sampling", ["semi", "parametric"])
@pytest.mark.parametrize("use_extra", [False, True], ids=["no-extra", "extra"])
def test_studentized_nested_simulation_keywords(sampling, use_extra):
    # Every call of func that is not on the original data, including the
    # nested bootstrap draws, must receive params (and state)
    y = np.random.default_rng(0).standard_normal(50)
    calls = []

    def func(y, params=None, state=None, flag=0):
        calls.append((params is not None, state is not None))
        if state is not None:
            return np.array([params[0] + state.standard_normal(y.shape[0]).mean()])
        if params is not None:
            return np.array([params[0] + 0.5 * (y.mean() - params[0])])
        return np.array([y.mean()])

    reps, nested_reps = 6, 4
    bs = IIDBootstrap(y, seed=0)
    extra_kwargs = {"flag": 1} if use_extra else None
    ci = bs.conf_int(
        func,
        reps,
        method="studentized",
        sampling=sampling,
        studentize_reps=nested_reps,
        extra_kwargs=extra_kwargs,
    )
    assert ci.shape == (2, 1)
    assert np.all(np.isfinite(ci))
    simulated = sampling == "parametric"
    # One call on the data, then per outer draw one call for the draw, one on
    # the bootstrap sample (nested base) and nested_reps nested draws
    assert len(calls) == 1 + reps * (2 + nested_reps)
    assert calls[0] == (False, False)
    for i in range(reps):
        start = 1 + i * (2 + nested_reps)
        assert calls[start] == (True, simulated)
        assert calls[start + 1] == (False, False)
        for call in calls[start + 2 : start + 2 + nested_reps]:
            assert call == (True, simulated)


def test_studentized_semiparametric_hand_computation():
    # Independent computation of the studentized semi-parametric interval
    # from the outer and nested replications using a fixed seed
    x = np.random.default_rng(12345).standard_normal(40)
    reps, nested_reps, seed, size = 50, 30, 20240601, 0.9
    ci = IIDBootstrap(x, seed=seed).conf_int(
        _shrunk_mean,
        reps,
        method="studentized",
        sampling="semi",
        studentize_reps=nested_reps,
        size=size,
    )

    base = x.mean()
    theta = np.zeros(reps)
    err = np.zeros(reps)
    nested_se = np.zeros(reps)
    bs = IIDBootstrap(x, seed=seed)
    for i, (pos, _) in enumerate(bs.bootstrap(reps)):
        y_star = pos[0]
        theta[i] = base + 0.5 * (y_star.mean() - base)
        # The bootstrap draws the seed of the nested bootstrap
        nested_seed = int(bs.generator.integers(2**31 - 1, size=1, dtype=np.int64)[0])
        nested = IIDBootstrap(y_star, seed=nested_seed)
        # Nested "base" is the statistic in the bootstrap sample
        nested_base = y_star.mean()
        nested_theta = np.zeros(nested_reps)
        for j, (nested_pos, _) in enumerate(nested.bootstrap(nested_reps)):
            nested_theta[j] = nested_base + 0.5 * (nested_pos[0].mean() - nested_base)
        nested_se[i] = nested_theta.std(ddof=0)
        err[i] = theta[i] - base
    t = err / nested_se
    outer_se = theta.std(ddof=0)
    alpha = (1 - size) / 2
    lower = base - outer_se * np.percentile(t, 100 * (1 - alpha))
    upper = base - outer_se * np.percentile(t, 100 * alpha)
    assert_allclose(ci.ravel(), [lower, upper], rtol=1e-10)


def test_studentized_semiparametric_std_err_func():
    # The documented signature std_err_func(params, *args, **kwargs) must work
    # with params as the first argument name, and the simulation keywords must
    # not be passed
    x = np.random.default_rng(12345).standard_normal(40)
    seen = []

    def std_err_func(params, y, **kwargs):
        seen.append(sorted(kwargs))
        return np.array([y.std() / np.sqrt(y.shape[0])])

    reps = 25
    ci = IIDBootstrap(x, seed=1).conf_int(
        _shrunk_mean,
        reps,
        method="studentized",
        sampling="semi",
        std_err_func=std_err_func,
        extra_kwargs=None,
    )
    assert len(seen) == reps
    assert all(keys == [] for keys in seen)
    # Hand computation
    base = x.mean()
    bs = IIDBootstrap(x, seed=1)
    theta = np.zeros(reps)
    t = np.zeros(reps)
    for i, (pos, _) in enumerate(bs.bootstrap(reps)):
        y_star = pos[0]
        theta[i] = base + 0.5 * (y_star.mean() - base)
        t[i] = (theta[i] - base) / (y_star.std() / np.sqrt(y_star.shape[0]))
    outer_se = theta.std(ddof=0)
    lower = base - outer_se * np.percentile(t, 97.5)
    upper = base - outer_se * np.percentile(t, 2.5)
    assert_allclose(ci.ravel(), [lower, upper], rtol=1e-10)


@pytest.mark.parametrize("sampling", ["semi", "parametric"])
def test_studentized_std_err_func_extra_kwargs_no_simulation_keywords(sampling):
    y = np.random.default_rng(0).standard_normal(30)
    seen = []

    def std_err_func(params, y, scale=1.0, **kwargs):
        seen.append((scale, sorted(kwargs)))
        return np.array([scale])

    def func(y, params=None, state=None, scale=1.0):
        if state is not None:
            return np.array([params[0] + state.standard_normal(y.shape[0]).mean()])
        return _shrunk_mean(y, params)

    IIDBootstrap(y, seed=3).conf_int(
        func,
        10,
        method="studentized",
        sampling=sampling,
        std_err_func=std_err_func,
        extra_kwargs={"scale": 2.0},
    )
    assert seen == [(2.0, [])] * 10
