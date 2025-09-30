from collections.abc import Callable
import copy
from typing import NamedTuple
import warnings

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
from scipy import stats

from arch.bootstrap import (
    CircularBlockBootstrap,
    IIDBootstrap,
    IndependentSamplesBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
)
from arch.bootstrap._samplers_python import (
    stationary_bootstrap_sample,
    stationary_bootstrap_sample_python,
)
from arch.bootstrap.base import _loo_jackknife
from arch.utility.exceptions import StudentizationError

try:
    from arch.bootstrap._samplers import (
        stationary_bootstrap_sample as stationary_bootstrap_sample_cython,
    )

    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False


class BSData(NamedTuple):
    rng: RandomState
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    x_df: pd.DataFrame
    y_series: pd.Series
    z_df: pd.DataFrame

    func: Callable[[np.ndarray, int], float | np.ndarray]


@pytest.fixture(params=[1234, "gen", "rs"])
def seed(request):
    if request.param == "gen":
        return np.random.default_rng(1234)
    elif request.param == "rs":
        return RandomState(1234)
    return request.param


@pytest.fixture
def bs_setup():
    rng = RandomState(1234)
    y = rng.standard_normal(1000)
    x = rng.standard_normal((1000, 2))
    z = rng.standard_normal((1000, 1))

    y_series = pd.Series(y)
    x_df = pd.DataFrame(x)
    z_df = pd.DataFrame(z)

    def func(y, axis=0):
        return y.mean(axis=axis)

    return BSData(rng, x, y, z, x_df, y_series, z_df, func)


def test_numpy(bs_setup, seed):
    x, y, z = bs_setup.x, bs_setup.y, bs_setup.z
    bs = IIDBootstrap(y, seed=seed)
    for data, kwdata in bs.bootstrap(10):
        index = bs.index
        assert_equal(len(kwdata.keys()), 0)
        assert_equal(y[index], data[0])
    # Ensure no changes to original data
    assert_equal(bs._args[0], y)

    bs = IIDBootstrap(y=y, seed=seed)
    for data, kwdata in bs.bootstrap(10):
        index = bs.index
        assert_equal(len(data), 0)
        assert_equal(y[index], kwdata["y"])
        assert_equal(y[index], bs.y)
    # Ensure no changes to original data
    assert_equal(bs._kwargs["y"], y)

    bs = IIDBootstrap(x, y, z, seed=seed)
    for data, kwdata in bs.bootstrap(10):
        index = bs.index
        assert_equal(len(data), 3)
        assert_equal(len(kwdata.keys()), 0)
        assert_equal(x[index], data[0])
        assert_equal(y[index], data[1])
        assert_equal(z[index], data[2])

    bs = IIDBootstrap(x, y=y, z=z, seed=seed)
    for data, kwdata in bs.bootstrap(10):
        index = bs.index
        assert_equal(len(data), 1)
        assert_equal(len(kwdata.keys()), 2)
        assert_equal(x[index], data[0])
        assert_equal(y[index], kwdata["y"])
        assert_equal(z[index], kwdata["z"])
        assert_equal(y[index], bs.y)
        assert_equal(z[index], bs.z)


def test_pandas(bs_setup, seed):
    x, y, z = bs_setup.x_df, bs_setup.y_series, bs_setup.z_df
    bs = IIDBootstrap(y, seed=seed)
    for data, kwdata in bs.bootstrap(10):
        index = bs.index
        assert_equal(len(kwdata.keys()), 0)
        assert_series_equal(y.iloc[index], data[0])
    # Ensure no changes to original data
    assert_series_equal(bs._args[0], y)

    bs = IIDBootstrap(y=y, seed=seed)
    for data, kwdata in bs.bootstrap(10):
        index = bs.index
        assert_equal(len(data), 0)
        assert_series_equal(y.iloc[index], kwdata["y"])
        assert_series_equal(y.iloc[index], bs.y)
    # Ensure no changes to original data
    assert_series_equal(bs._kwargs["y"], y)

    bs = IIDBootstrap(x, y, z, seed=seed)
    for data, kwdata in bs.bootstrap(10):
        index = bs.index
        assert_equal(len(data), 3)
        assert_equal(len(kwdata.keys()), 0)
        assert_frame_equal(x.iloc[index], data[0])
        assert_series_equal(y.iloc[index], data[1])
        assert_frame_equal(z.iloc[index], data[2])

    bs = IIDBootstrap(x, y=y, z=z, seed=seed)
    for data, kwdata in bs.bootstrap(10):
        index = bs.index
        assert_equal(len(data), 1)
        assert_equal(len(kwdata.keys()), 2)
        assert_frame_equal(x.iloc[index], data[0])
        assert_series_equal(y.iloc[index], kwdata["y"])
        assert_frame_equal(z.iloc[index], kwdata["z"])
        assert_series_equal(y.iloc[index], bs.y)
        assert_frame_equal(z.iloc[index], bs.z)


def test_mixed_types(bs_setup, seed):
    x, y, z = bs_setup.x_df, bs_setup.y_series, bs_setup.z
    bs = IIDBootstrap(y, x=x, z=z, seed=seed)
    for data, kwdata in bs.bootstrap(10):
        index = bs.index
        assert_equal(len(data), 1)
        assert_equal(len(kwdata.keys()), 2)
        assert_frame_equal(x.iloc[index], kwdata["x"])
        assert_frame_equal(x.iloc[index], bs.x)
        assert_series_equal(y.iloc[index], data[0])
        assert_equal(z[index], kwdata["z"])
        assert_equal(z[index], bs.z)


def test_errors(bs_setup):
    x = np.arange(10)
    y = np.arange(100)
    with pytest.raises(ValueError, match=r"All inputs must hav"):
        IIDBootstrap(x, y)
    with pytest.raises(ValueError, match=r"index is a reserved name"):
        IIDBootstrap(index=x)
    bs = IIDBootstrap(y)

    with pytest.raises(ValueError, match=r"Unknown method"):
        bs.conf_int(bs_setup.func, method="unknown")
    with pytest.raises(ValueError, match=r"tail must be one of two-sided"):
        bs.conf_int(bs_setup.func, tail="dragon")
    with pytest.raises(ValueError, match=r"size must be strictly between 0 and 1"):
        bs.conf_int(bs_setup.func, size=95)


def test_cov(bs_setup):
    bs = IIDBootstrap(bs_setup.x)
    num_bootstrap = 10
    cov = bs.cov(func=bs_setup.func, reps=num_bootstrap, recenter=False)
    bs.reset()

    results = np.zeros((num_bootstrap, 2))
    count = 0
    for data, _ in bs.bootstrap(num_bootstrap):
        results[count] = data[0].mean(axis=0)
        count += 1
    errors = results - bs_setup.x.mean(axis=0)
    direct_cov = errors.T.dot(errors) / num_bootstrap
    assert_allclose(cov, direct_cov)

    bs.reset()
    cov = bs.cov(func=bs_setup.func, recenter=True, reps=num_bootstrap)
    errors = results - results.mean(axis=0)
    direct_cov = errors.T.dot(errors) / num_bootstrap
    assert_allclose(cov, direct_cov)

    bs = IIDBootstrap(bs_setup.x_df)
    cov = bs.cov(func=bs_setup.func, reps=num_bootstrap, recenter=False)
    bs.reset()
    var = bs.var(func=bs_setup.func, reps=num_bootstrap, recenter=False)
    bs.reset()
    results = np.zeros((num_bootstrap, 2))
    count = 0
    for data, _ in bs.bootstrap(num_bootstrap):
        results[count] = data[0].mean(axis=0)
        count += 1
    errors = results - bs_setup.x.mean(axis=0)
    direct_cov = errors.T.dot(errors) / num_bootstrap
    assert_allclose(cov, direct_cov)
    assert_allclose(var, np.diag(direct_cov))

    bs.reset()
    cov = bs.cov(func=bs_setup.func, recenter=True, reps=num_bootstrap)
    errors = results - results.mean(axis=0)
    direct_cov = errors.T.dot(errors) / num_bootstrap
    assert_allclose(cov, direct_cov)


def test_conf_int_basic(bs_setup):
    num_bootstrap = 200
    bs = IIDBootstrap(bs_setup.x)

    ci = bs.conf_int(bs_setup.func, reps=num_bootstrap, size=0.90, method="basic")
    bs.reset()
    ci_u = bs.conf_int(
        bs_setup.func, tail="upper", reps=num_bootstrap, size=0.95, method="basic"
    )
    bs.reset()
    ci_l = bs.conf_int(
        bs_setup.func, tail="lower", reps=num_bootstrap, size=0.95, method="basic"
    )
    bs.reset()
    results = np.zeros((num_bootstrap, 2))
    count = 0
    for pos, _ in bs.bootstrap(num_bootstrap):
        results[count] = bs_setup.func(*pos)
        count += 1
    mu = bs_setup.func(bs_setup.x)
    upper = mu + (mu - np.percentile(results, 5, axis=0))
    lower = mu + (mu - np.percentile(results, 95, axis=0))

    assert_allclose(lower, ci[0, :])
    assert_allclose(upper, ci[1, :])

    assert_allclose(ci[1, :], ci_u[1, :])
    assert_allclose(ci[0, :], ci_l[0, :])
    inf = np.empty_like(ci_l[0, :])
    inf.fill(np.inf)
    assert_equal(inf, ci_l[1, :])
    assert_equal(-1 * inf, ci_u[0, :])


def test_conf_int_percentile(bs_setup):
    num_bootstrap = 200
    bs = IIDBootstrap(bs_setup.x)

    ci = bs.conf_int(bs_setup.func, reps=num_bootstrap, size=0.90, method="percentile")
    bs.reset()
    ci_u = bs.conf_int(
        bs_setup.func, tail="upper", reps=num_bootstrap, size=0.95, method="percentile"
    )
    bs.reset()
    ci_l = bs.conf_int(
        bs_setup.func, tail="lower", reps=num_bootstrap, size=0.95, method="percentile"
    )
    bs.reset()
    results = np.zeros((num_bootstrap, 2))
    count = 0
    for pos, _ in bs.bootstrap(num_bootstrap):
        results[count] = bs_setup.func(*pos)
        count += 1

    upper = np.percentile(results, 95, axis=0)
    lower = np.percentile(results, 5, axis=0)

    assert_allclose(lower, ci[0, :])
    assert_allclose(upper, ci[1, :])

    assert_allclose(ci[1, :], ci_u[1, :])
    assert_allclose(ci[0, :], ci_l[0, :])
    inf = np.empty_like(ci_l[0, :])
    inf.fill(np.inf)
    assert_equal(inf, ci_l[1, :])
    assert_equal(-1 * inf, ci_u[0, :])


def test_conf_int_norm(bs_setup):
    num_bootstrap = 200
    bs = IIDBootstrap(bs_setup.x)

    ci = bs.conf_int(bs_setup.func, reps=num_bootstrap, size=0.90, method="norm")
    bs.reset()
    ci_u = bs.conf_int(
        bs_setup.func, tail="upper", reps=num_bootstrap, size=0.95, method="var"
    )
    bs.reset()
    ci_l = bs.conf_int(
        bs_setup.func, tail="lower", reps=num_bootstrap, size=0.95, method="cov"
    )
    bs.reset()
    cov = bs.cov(bs_setup.func, reps=num_bootstrap)
    mu = bs_setup.func(bs_setup.x)
    std_err = np.sqrt(np.diag(cov))
    upper = mu + stats.norm.ppf(0.95) * std_err
    lower = mu + stats.norm.ppf(0.05) * std_err
    assert_allclose(lower, ci[0, :])
    assert_allclose(upper, ci[1, :])

    assert_allclose(ci[1, :], ci_u[1, :])
    assert_allclose(ci[0, :], ci_l[0, :])
    inf = np.empty_like(ci_l[0, :])
    inf.fill(np.inf)
    assert_equal(inf, ci_l[1, :])
    assert_equal(-1 * inf, ci_u[0, :])


def test_reuse(bs_setup, seed):
    num_bootstrap = 100
    bs = IIDBootstrap(bs_setup.x, seed=seed)

    ci = bs.conf_int(bs_setup.func, reps=num_bootstrap)
    old_results = bs._results.copy()
    ci_reuse = bs.conf_int(bs_setup.func, reps=num_bootstrap, reuse=True)
    results = bs._results
    assert_equal(results, old_results)
    assert_equal(ci, ci_reuse)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        warnings.simplefilter("always")
        bs.conf_int(bs_setup.func, tail="lower", reps=num_bootstrap // 2, reuse=True)
        assert_equal(len(w), 1)


def test_studentized(bs_setup, seed):
    num_bootstrap = 20
    bs = IIDBootstrap(bs_setup.x, seed=seed)

    def std_err_func(mu, y):
        errors = y - mu
        var = (errors**2.0).mean(axis=0)
        return np.sqrt(var / y.shape[0])

    ci = bs.conf_int(
        bs_setup.func,
        reps=num_bootstrap,
        method="studentized",
        std_err_func=std_err_func,
    )
    bs.reset()
    base = bs_setup.func(bs_setup.x)
    results = np.zeros((num_bootstrap, 2))
    stud_results = np.zeros((num_bootstrap, 2))
    count = 0
    for pos, _ in bs.bootstrap(reps=num_bootstrap):
        results[count] = bs_setup.func(*pos)
        std_err = std_err_func(results[count], *pos)
        stud_results[count] = (results[count] - base) / std_err
        count += 1

    assert_allclose(results, bs._results)
    assert_allclose(stud_results, bs._studentized_results)
    errors = results - results.mean(0)
    std_err = np.sqrt(np.mean(errors**2.0, axis=0))
    ci_direct = np.zeros((2, 2))
    for i in range(2):
        ci_direct[0, i] = base[i] - std_err[i] * np.percentile(stud_results[:, i], 97.5)
        ci_direct[1, i] = base[i] - std_err[i] * np.percentile(stud_results[:, i], 2.5)
    assert_allclose(ci, ci_direct)

    bs.reset()
    ci = bs.conf_int(
        bs_setup.func, reps=num_bootstrap, method="studentized", studentize_reps=50
    )

    bs.reset()
    base = bs_setup.func(bs_setup.x)
    results = np.zeros((num_bootstrap, 2))
    stud_results = np.zeros((num_bootstrap, 2))
    count = 0
    for pos, _ in bs.bootstrap(reps=num_bootstrap):
        results[count] = bs_setup.func(*pos)
        if isinstance(bs._generator, RandomState):
            seed = bs._generator.randint(2**31 - 1)
        else:
            seed = bs._generator.integers(2**31 - 1)
        inner_bs = IIDBootstrap(*pos, seed=seed)
        cov = inner_bs.cov(bs_setup.func, reps=50)
        std_err = np.sqrt(np.diag(cov))
        stud_results[count] = (results[count] - base) / std_err
        count += 1

    assert_allclose(results, bs._results)
    assert_allclose(stud_results, bs._studentized_results)
    errors = results - results.mean(0)
    std_err = np.sqrt(np.mean(errors**2.0, axis=0))

    ci_direct = np.zeros((2, 2))
    for i in range(2):
        ci_direct[0, i] = base[i] - std_err[i] * np.percentile(stud_results[:, i], 97.5)
        ci_direct[1, i] = base[i] - std_err[i] * np.percentile(stud_results[:, i], 2.5)
    assert_allclose(ci, ci_direct)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bs.conf_int(
            bs_setup.func,
            reps=num_bootstrap,
            method="studentized",
            std_err_func=std_err_func,
            reuse=True,
        )
        assert_equal(len(w), 1)


def test_conf_int_bias_corrected(bs_setup):
    num_bootstrap = 20
    bs = IIDBootstrap(bs_setup.x)

    ci = bs.conf_int(bs_setup.func, reps=num_bootstrap, method="bc")
    bs.reset()
    ci_db = bs.conf_int(bs_setup.func, reps=num_bootstrap, method="debiased")
    assert_equal(ci, ci_db)
    base, results = bs._base, bs._results
    p = np.zeros(2)
    p[0] = np.mean(results[:, 0] < base[0])
    p[1] = np.mean(results[:, 1] < base[1])
    b = stats.norm.ppf(p)
    q = stats.norm.ppf(np.array([0.025, 0.975]))
    q = q[:, None]
    percentiles = 100 * stats.norm.cdf(2 * b + q)

    ci = np.zeros((2, 2))
    for i in range(2):
        ci[i] = np.percentile(results[:, i], list(percentiles[:, i]))
    ci = ci.T
    assert_allclose(ci_db, ci)


def test_conf_int_bca_scaler(bs_setup):
    num_bootstrap = 100
    bs = IIDBootstrap(bs_setup.y)

    ci = bs.conf_int(np.mean, reps=num_bootstrap, method="bca")
    msg = (
        "conf_int(method='bca') scalar input regression. Ensure "
        "output is at least 1D with numpy.atleast_1d()."
    )
    assert ci.shape == (2, 1), msg


def test_conf_int_parametric(bs_setup):
    def param_func(x, params=None, state=None):
        if state is not None:
            mu = params
            e = state.standard_normal(x.shape)
            return (mu + e).mean(0)
        else:
            return x.mean(0)

    def semi_func(x, params=None):
        if params is not None:
            mu = params
            e = x - mu
            return (mu + e).mean(0)
        else:
            return x.mean(0)

    reps = 100
    bs = IIDBootstrap(bs_setup.x)

    ci = bs.conf_int(func=param_func, reps=reps, sampling="parametric")
    assert len(ci) == 2
    assert np.all(ci[0] < ci[1])
    bs.reset()
    results = np.zeros((reps, 2))
    count = 0
    mu = bs_setup.x.mean(0)
    for pos, _ in bs.bootstrap(100):
        results[count] = param_func(*pos, params=mu, state=bs.generator)
        count += 1
    assert_equal(bs._results, results)

    bs.reset()
    ci = bs.conf_int(func=semi_func, reps=100, sampling="semi")
    assert len(ci) == 2
    assert np.all(ci[0] < ci[1])
    bs.reset()
    results = np.zeros((reps, 2))
    count = 0
    for pos, _ in bs.bootstrap(100):
        results[count] = semi_func(*pos, params=mu)
        count += 1
    assert_allclose(bs._results, results)

    with pytest.raises(ValueError, match=r"sampling must be one"):
        bs.conf_int(func=semi_func, reps=100, sampling="other")


def test_extra_kwargs(bs_setup):
    extra_kwargs = {"axis": 0}
    bs = IIDBootstrap(bs_setup.x)
    num_bootstrap = 100

    bs.cov(bs_setup.func, reps=num_bootstrap, extra_kwargs=extra_kwargs)

    bs = IIDBootstrap(axis=bs_setup.x)
    with pytest.raises(ValueError, match=r"extra_kwargs contains keys used"):
        bs.cov(bs_setup.func, reps=num_bootstrap, extra_kwargs=extra_kwargs)


def test_jackknife(bs_setup):
    x = bs_setup.x
    results = _loo_jackknife(bs_setup.func, len(x), (x,), {})

    direct_results = np.zeros_like(x)
    for i in range(len(x)):
        if i == 0:
            y = x[1:]
        elif i == (len(x) - 1):
            y = x[:-1]
        else:
            temp = list(x[:i])
            temp.extend(list(x[i + 1 :]))
            y = np.array(temp)
        direct_results[i] = bs_setup.func(y)
    assert_allclose(direct_results, results)

    x = bs_setup.x_df
    results_df = _loo_jackknife(bs_setup.func, len(x), (x,), {})
    assert_allclose(results, results_df)

    y = bs_setup.y
    results = _loo_jackknife(bs_setup.func, len(y), (y,), {})

    direct_results = np.zeros_like(y)
    for i in range(len(y)):
        if i == 0:
            z = y[1:]
        elif i == (len(y) - 1):
            z = y[:-1]
        else:
            temp = list(y[:i])
            temp.extend(list(y[i + 1 :]))
            z = np.array(temp)
        direct_results[i] = bs_setup.func(z)
    assert_allclose(direct_results, results)

    y = bs_setup.y_series
    results_series = _loo_jackknife(bs_setup.func, len(y), (y,), {})
    assert_allclose(results, results_series)


def test_bca(bs_setup):
    num_bootstrap = 20
    bs = IIDBootstrap(bs_setup.x, seed=23456)

    ci_direct = bs.conf_int(bs_setup.func, reps=num_bootstrap, method="bca")
    bs.reset()
    base, results = bs._base, bs._results
    p = np.zeros(2)
    p[0] = np.mean(results[:, 0] < base[0])
    p[1] = np.mean(results[:, 1] < base[1])
    b = stats.norm.ppf(p)
    b = b[:, None]
    q = stats.norm.ppf(np.array([0.025, 0.975]))

    bs_setup.func(bs_setup.x)
    nobs = bs_setup.x.shape[0]
    jk = _loo_jackknife(bs_setup.func, nobs, [bs_setup.x], {})
    u = jk.mean() - jk
    u2 = np.sum(u * u, 0)
    u3 = np.sum(u * u * u, 0)
    a = u3 / (6.0 * (u2**1.5))
    a = a[:, None]
    percentiles = 100 * stats.norm.cdf(b + (b + q) / (1 - a * (b + q)))

    ci = np.zeros((2, 2))
    for i in range(2):
        ci[i] = np.percentile(results[:, i], list(percentiles[i]))
    ci = ci.T
    assert_allclose(ci_direct, ci)

    bs = IIDBootstrap(y=bs_setup.x, seed=23456)
    ci_kwarg = bs.conf_int(bs_setup.func, reps=num_bootstrap, method="bca")
    assert_allclose(ci_kwarg, ci)

    bs = IIDBootstrap(y=bs_setup.x_df, seed=23456)
    ci_kwarg_pandas = bs.conf_int(bs_setup.func, reps=num_bootstrap, method="bca")
    assert_allclose(ci_kwarg_pandas, ci)


def test_pandas_integer_index(bs_setup):
    x = bs_setup.x
    x_int = bs_setup.x_df.copy()
    x_int.index = 10 + np.arange(x.shape[0])
    bs = IIDBootstrap(x, x_int, seed=23456)
    for pdata, _ in bs.bootstrap(10):
        assert_equal(pdata[0], np.asarray(pdata[1]))


def test_apply(bs_setup, seed):
    bs = IIDBootstrap(bs_setup.x, seed=seed)
    results = bs.apply(bs_setup.func, 1000)
    bs.reset(True)
    direct_results = []
    for pos, _ in bs.bootstrap(1000):
        direct_results.append(bs_setup.func(*pos))
    direct_results = np.array(direct_results)
    assert_equal(results, direct_results)


def test_apply_series(bs_setup, seed):
    bs = IIDBootstrap(bs_setup.y_series, seed=seed)
    results = bs.apply(bs_setup.func, 1000)
    bs.reset(True)
    direct_results = []
    for pos, _ in bs.bootstrap(1000):
        direct_results.append(bs_setup.func(*pos))
    direct_results = np.array(direct_results)
    direct_results = direct_results[:, None]
    assert_equal(results, direct_results)


def test_str(bs_setup, seed):
    bs = IIDBootstrap(bs_setup.y_series, seed=seed)
    expected = "IID Bootstrap(no. pos. inputs: 1, no. keyword inputs: 0)"
    assert_equal(str(bs), expected)
    expected = expected[:-1] + ", ID: " + hex(id(bs)) + ")"
    assert_equal(bs.__repr__(), expected)
    expected = (
        "<strong>IID Bootstrap</strong>("
        "<strong>no. pos. inputs</strong>: 1, "
        "<strong>no. keyword inputs</strong>: 0, "
        "<strong>ID</strong>: " + hex(id(bs)) + ")"
    )
    assert_equal(bs._repr_html(), expected)

    bs = StationaryBootstrap(10, bs_setup.y_series, bs_setup.x_df, seed=seed)
    expected = (
        "Stationary Bootstrap(block size: 10, no. pos. "
        "inputs: 2, no. keyword inputs: 0)"
    )
    assert_equal(str(bs), expected)
    expected = expected[:-1] + ", ID: " + hex(id(bs)) + ")"
    assert_equal(bs.__repr__(), expected)

    bs = CircularBlockBootstrap(
        block_size=20, y=bs_setup.y_series, x=bs_setup.x_df, seed=seed
    )
    expected = (
        "Circular Block Bootstrap(block size: 20, no. pos. "
        "inputs: 0, no. keyword inputs: 2)"
    )
    assert_equal(str(bs), expected)
    expected = expected[:-1] + ", ID: " + hex(id(bs)) + ")"
    assert_equal(bs.__repr__(), expected)
    expected = (
        "<strong>Circular Block Bootstrap</strong>"
        "(<strong>block size</strong>: 20, "
        "<strong>no. pos. inputs</strong>: 0, "
        "<strong>no. keyword inputs</strong>: 2,"
        " <strong>ID</strong>: " + hex(id(bs)) + ")"
    )
    assert_equal(bs._repr_html(), expected)

    bs = MovingBlockBootstrap(
        block_size=20, y=bs_setup.y_series, x=bs_setup.x_df, seed=seed
    )
    expected = (
        "Moving Block Bootstrap(block size: 20, no. pos. "
        "inputs: 0, no. keyword inputs: 2)"
    )
    assert_equal(str(bs), expected)
    expected = expected[:-1] + ", ID: " + hex(id(bs)) + ")"
    assert_equal(bs.__repr__(), expected)
    expected = (
        "<strong>Moving Block Bootstrap</strong>"
        "(<strong>block size</strong>: 20, "
        "<strong>no. pos. inputs</strong>: 0, "
        "<strong>no. keyword inputs</strong>: 2,"
        " <strong>ID</strong>: " + hex(id(bs)) + ")"
    )
    assert_equal(bs._repr_html(), expected)


def test_uneven_sampling(bs_setup, seed):
    bs = MovingBlockBootstrap(
        block_size=31, y=bs_setup.y_series, x=bs_setup.x_df, seed=seed
    )
    for _, kw in bs.bootstrap(10):
        assert kw["y"].shape == bs_setup.y_series.shape
        assert kw["x"].shape == bs_setup.x_df.shape
    bs = CircularBlockBootstrap(
        block_size=31, y=bs_setup.y_series, x=bs_setup.x_df, seed=seed
    )
    for _, kw in bs.bootstrap(10):
        assert kw["y"].shape == bs_setup.y_series.shape
        assert kw["x"].shape == bs_setup.x_df.shape


@pytest.mark.skipif(not HAS_EXTENSION, reason="Extension not built.")
@pytest.mark.filterwarnings("ignore::arch.compat.numba.PerformanceWarning")
def test_samplers(bs_setup):
    """
    Test all three implementations are identical
    """
    indices = np.array(bs_setup.rng.randint(0, 1000, 1000), dtype=np.int64)
    u = bs_setup.rng.random_sample(1000)
    p = 0.1
    indices_orig = indices.copy()

    numba = stationary_bootstrap_sample(indices, u, p)
    indices = indices_orig.copy()
    python = stationary_bootstrap_sample_python(indices, u, p)
    indices = indices_orig.copy()
    cython = stationary_bootstrap_sample_cython(indices, u, p)
    assert_equal(numba, cython)
    assert_equal(numba, python)


def test_bca_against_bcajack():
    # import rpy2.rinterface as ri
    # import rpy2.robjects as robjects
    # import rpy2.robjects.numpy2ri
    # from rpy2.robjects.packages import importr
    # rpy2.robjects.numpy2ri.activate()
    # utils = importr('utils')
    # try:
    #     bcaboot = importr('bcaboot')
    # except Exception:
    #     utils.install_packages('bcaboot',
    #                            repos='https://cran.us.r-project.org')
    #     bcaboot = importr('bcaboot')

    rng_seed_obs = 42
    rs = np.random.RandomState(rng_seed_obs)
    observations = rs.multivariate_normal(mean=[8, 4], cov=np.identity(2), size=20)
    b = 2000
    rng_seed = 123
    rs = np.random.RandomState(rng_seed)
    arch_bs = IIDBootstrap(observations, seed=rs)
    confidence_interval_size = 0.90

    def func(x):
        sample = x.mean(axis=0)
        return sample[1] / sample[0]

    arch_ci = arch_bs.conf_int(
        func=func, reps=b, size=confidence_interval_size, method="bca"
    )

    # # callable from R
    # @ri.rternalize
    # def func_r(x):
    #     x = np.asarray(x)
    #     _mean = x.mean(axis=0)
    #     return float(_mean[1] / _mean[0])
    # output = bcaboot.bcajack(x=observations, B=float(B), func=func_r)
    a = arch_bs._bca_acceleration(func, None)
    b = arch_bs._bca_bias()
    # bca_lims = np.array(output[1])[:, 0]
    # # bca confidence intervals for: 0.025, 0.05, 0.1, 0.16, 0.5,
    #                                 0.84, 0.9, 0.95, 0.975
    # bcajack_ci_90 = [bca_lims[1], bca_lims[-2]]
    # bcajack should estimate similar "a" using jackknife on
    # the same observations
    assert_allclose(a, -0.0004068984)
    # bcajack returns b (or z0) = -0.03635412, but based on
    # different bootstrap samples
    assert_allclose(b, 0.04764396)
    # bcajack_ci_90 = [0.42696, 0.53188]
    arch_ci = list(arch_ci[:, -1])
    saved_arch_ci_90 = [0.42719805360154717, 0.5336561953393736]
    assert_allclose(arch_ci, saved_arch_ci_90)


def test_state():
    final = 0
    final_seed = 1
    final_state = 2
    bs = IIDBootstrap(np.arange(100), seed=23456)
    state = bs.state
    for data, _ in bs.bootstrap(10):
        final = data[0]
    bs.state = state
    for data, _ in bs.bootstrap(10):
        final_seed = data[0]
    bs.state = state
    for data, _ in bs.bootstrap(10):
        final_state = data[0]
    assert_equal(final, final_seed)
    assert_equal(final, final_state)


def test_reset():
    final = 0
    final_reset = 1
    bs = IIDBootstrap(np.arange(100))
    state = bs.state
    for data, _ in bs.bootstrap(10):
        final = data[0]
    bs.reset()
    state_reset = bs.state
    for data, _ in bs.bootstrap(10):
        final_reset = data[0]
    assert_equal(final, final_reset)
    assert_equal(state, state_reset)


def test_iid_unequal_equiv():
    rs = RandomState(0)
    x = rs.standard_normal(500)
    rs1 = RandomState(0)
    bs1 = IIDBootstrap(x, seed=rs1)

    rs2 = RandomState(0)
    bs2 = IndependentSamplesBootstrap(x, seed=rs2)

    v1 = bs1.var(np.mean)
    v2 = bs2.var(np.mean)
    assert_allclose(v1, v2)
    assert isinstance(bs2.index, tuple)
    assert isinstance(bs2.index[0], list)
    assert isinstance(bs2.index[0][0], np.ndarray)
    assert bs2.index[0][0].shape == x.shape


def test_unequal_bs():
    def mean_diff(*args):
        return args[0].mean() - args[1].mean()

    rs = RandomState(0)
    x = rs.standard_normal(800)
    y = rs.standard_normal(200)

    bs = IndependentSamplesBootstrap(x, y, seed=rs)
    variance = bs.var(mean_diff)
    assert variance > 0
    ci = bs.conf_int(mean_diff)
    assert ci[0] < ci[1]
    applied = bs.apply(mean_diff, 1000)
    assert len(applied) == 1000

    x = pd.Series(x)
    y = pd.Series(y)
    bs = IndependentSamplesBootstrap(x, y)
    variance = bs.var(mean_diff)
    assert variance > 0

    with pytest.raises(ValueError, match=r"BCa cannot be applied"):
        bs.conf_int(mean_diff, method="bca")


def test_unequal_bs_kwargs():
    def mean_diff(x, y):
        return x.mean() - y.mean()

    rs = RandomState(0)
    x = rs.standard_normal(800)
    y = rs.standard_normal(200)

    bs = IndependentSamplesBootstrap(x=x, y=y, seed=rs)
    variance = bs.var(mean_diff)
    assert variance > 0
    ci = bs.conf_int(mean_diff)
    assert ci[0] < ci[1]
    applied = bs.apply(mean_diff, 1000)

    x = pd.Series(x)
    y = pd.Series(y)
    bs = IndependentSamplesBootstrap(x=x, y=y, seed=rs)
    variance = bs.var(mean_diff)
    assert variance > 0

    assert len(applied) == 1000


def test_unequal_reset():
    def mean_diff(*args):
        return args[0].mean() - args[1].mean()

    rs = RandomState(0)
    x = rs.standard_normal(800)
    y = rs.standard_normal(200)
    orig_state = rs.get_state()
    bs = IndependentSamplesBootstrap(x, y, seed=rs)
    variance = bs.var(mean_diff)
    assert variance > 0
    bs.reset()
    state = bs.state
    assert_equal(state[1], orig_state[1])

    rs = RandomState(1234)
    bs = IndependentSamplesBootstrap(x, y, seed=rs)
    bs.seed = RandomState(1234)
    orig_state = bs.state
    bs.var(mean_diff)
    bs.reset(use_seed=True)
    state = bs.state
    assert_equal(state[1], orig_state[1])


def test_studentization_error():
    def f(x):
        return np.array([x.mean(), 3])

    x = np.random.standard_normal(100)
    bs = IIDBootstrap(x)
    with pytest.raises(StudentizationError):
        bs.conf_int(f, 100, method="studentized")


def test_list_input():
    # GH 315
    vals = np.random.standard_normal(25).tolist()
    with pytest.raises(TypeError, match=r"Positional input 0 "):
        IIDBootstrap(vals)
    vals = np.random.standard_normal(25).tolist()
    with pytest.raises(TypeError, match=r"Input `data` "):
        IIDBootstrap(data=vals)


def test_bca_extra_kwarg():
    # GH 366
    def f(a, b):
        return a.mean(0)

    x = np.random.standard_normal(1000)
    bs = IIDBootstrap(x)
    ci = bs.conf_int(f, extra_kwargs={"b": "anything"}, reps=100, method="bca")
    assert isinstance(ci, np.ndarray)
    assert ci.shape == (2, 1)


def test_set_randomstate(bs_setup):
    rs = np.random.RandomState([12345])
    bs = IIDBootstrap(bs_setup.x, seed=rs)
    bs.generator = rs
    assert bs.generator is rs


def test_iid_args_kwargs(bs_setup):
    bs1 = IIDBootstrap(bs_setup.y, seed=0)
    bs2 = IIDBootstrap(y=bs_setup.y, seed=0)
    for a, b in zip(bs1.bootstrap(1), bs2.bootstrap(1), strict=False):
        assert np.all(a[0][0] == b[1]["y"])


def test_iid_semiparametric(bs_setup, seed):
    bs = IIDBootstrap(bs_setup.y, seed=seed)

    def func(y, axis=0, params=None):
        if params is not None:
            return (y - params).mean(axis=axis)
        return y.mean(axis=axis)

    ci = bs.conf_int(func, reps=10, sampling="semiparametric")
    assert ci.shape == (2, 1)


def test_bc_extremum_error():
    # GH 496

    def profile_function(scores):
        tau = np.linspace(-0.1, 1.0, 10)
        comparisons = np.expand_dims(scores.flatten(), axis=0) >= tau[:, np.newaxis]
        return np.mean(comparisons, axis=-1)

    val = np.array(
        [
            0.14333333,
            0.6576,
            0.35882353,
            0.48982389,
            0.35660377,
            0.7,
            -0.00457143,
            0.87817109,
            -0.01538462,
            0.54444444,
        ]
    )
    bs = IIDBootstrap(val, seed=np.random.RandomState(0))
    with pytest.raises(RuntimeError, match=r"Empirical probability used"):
        bs.conf_int(profile_function, 100, method="bc")


def test_invalid_random_state_generator():
    with pytest.raises(TypeError, match=r"generator keyword argument"):
        IIDBootstrap(np.empty(100), seed="123")


def test_generator(seed):
    bs = IIDBootstrap(np.empty(100), seed=seed)
    typ = RandomState if isinstance(seed, RandomState) else np.random.Generator
    assert isinstance(bs.generator, typ)
    gen_copy = copy.deepcopy(bs.generator)
    bs.generator = gen_copy
    with pytest.raises(TypeError, match=r"Only a Generator or RandomState"):
        bs.generator = 3
    assert isinstance(bs.generator, typ)
    assert bs.generator is gen_copy
    state = bs.state
    if isinstance(bs.generator, np.random.Generator):
        assert isinstance(state, dict)
    else:
        assert isinstance(state, tuple)
    bs.state = state
    if isinstance(bs.generator, np.random.Generator):
        assert isinstance(state, dict)
    else:
        assert isinstance(state, tuple)


def test_staionary_seed(bs_setup, seed):
    sb = StationaryBootstrap(10, bs_setup.y, seed=seed)
    for pos, _ in sb.bootstrap(10):
        assert pos[0].shape[0] == bs_setup.y.shape[0]
