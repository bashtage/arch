import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import stats

from arch.bootstrap import (
    CircularBlockBootstrap,
    IIDBootstrap,
    IndependentSamplesBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
)
from arch.bootstrap.base import _get_acceleration, _loo_jackknife


@pytest.mark.parametrize("shift", [0.0, 0.1, 0.3])
def test_bca_all_jackknife_estimates_equal(shift):
    # With a tie for the maximum, the maximum is the same when any single
    # observation is left out, so the acceleration is 0/0 or rounding error
    x = np.random.default_rng(0).standard_normal(50) + shift
    x[10] = x.max()

    def func(y):
        return np.atleast_1d(y.max())

    jk = _loo_jackknife(func, x.shape[0], [x], {})
    assert np.unique(jk).shape[0] == 1
    bs = IIDBootstrap(x, seed=1)
    with pytest.raises(RuntimeError, match="too small to use BCa"):
        bs.conf_int(func, 200, method="bca")


def test_bca_acceleration_unchanged():
    # Acceleration of the mean from the definition
    x = np.random.default_rng(1).standard_exponential(40)
    jk = _loo_jackknife(lambda y: np.atleast_1d(y.mean()), x.shape[0], [x], {})
    u = jk.mean(0) - jk
    expected = (u**3).sum(0) / (6 * ((u**2).sum(0)) ** 1.5)
    assert_allclose(_get_acceleration(jk).ravel(), expected, rtol=1e-12)
    # The acceleration of the mean is the scaled skewness of the data
    skew = stats.skew(x)
    assert_allclose(_get_acceleration(jk).ravel(), skew / (6 * np.sqrt(40)), rtol=0.1)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_bca_acceleration_non_finite(bad):
    jk = np.random.default_rng(1).standard_normal((20, 2))
    jk[3, 1] = bad
    with pytest.raises(RuntimeError, match="not finite"):
        _get_acceleration(jk)


BLOCK_BOOTSTRAPS = [CircularBlockBootstrap, MovingBlockBootstrap, StationaryBootstrap]


@pytest.mark.parametrize("cls", BLOCK_BOOTSTRAPS)
@pytest.mark.parametrize("block_size", [0, -3, np.int64(0)], ids=["0", "-3", "np-0"])
def test_block_size_too_small(cls, block_size):
    with pytest.raises(
        ValueError, match=r"block_size must be (an integer|a finite float) >= 1"
    ):
        cls(block_size, np.arange(20.0))


@pytest.mark.parametrize("cls", BLOCK_BOOTSTRAPS)
@pytest.mark.parametrize("block_size", [0.5, np.nan, np.inf])
def test_block_size_not_valid_number(cls, block_size):
    if cls is StationaryBootstrap:
        # Any number >= 1 is valid for the stationary bootstrap
        expected_error = ValueError
    else:
        expected_error = TypeError
    with pytest.raises(expected_error, match="block_size must be"):
        cls(block_size, np.arange(20.0))


@pytest.mark.parametrize("cls", BLOCK_BOOTSTRAPS)
@pytest.mark.parametrize("block_size", ["5", None, [5]], ids=["str", "None", "list"])
def test_block_size_wrong_type(cls, block_size):
    with pytest.raises(TypeError, match="block_size must be"):
        cls(block_size, np.arange(20.0))


@pytest.mark.parametrize("cls", [CircularBlockBootstrap, MovingBlockBootstrap])
@pytest.mark.parametrize(
    "block_size", [4.0, 4.5, np.float64(4.0)], ids=["4.0", "4.5", "np-4.0"]
)
def test_block_size_not_integer(cls, block_size):
    # These raised a TypeError from NumPy when sampling
    with pytest.raises(TypeError, match="block_size must be an integer"):
        cls(block_size, np.arange(20.0))


@pytest.mark.parametrize("cls", BLOCK_BOOTSTRAPS)
@pytest.mark.parametrize(
    "block_size",
    [1, 4, np.int64(4), np.int32(4)],
    ids=["1", "4", "np-int64", "np-int32"],
)
def test_block_size_valid(cls, block_size):
    bs = cls(block_size, np.arange(20.0), seed=0)
    assert bs.update_indices().shape == (20,)


@pytest.mark.parametrize(
    "block_size",
    [1.0, 1.5, 12.25, np.float64(7.5)],
    ids=["1.0", "1.5", "12.25", "np-7.5"],
)
def test_stationary_block_size_not_integer(block_size):
    # The stationary bootstrap block size is the mean block length, which is
    # an estimate from optimal_block_length
    bs = StationaryBootstrap(block_size, np.arange(20.0), seed=0)
    assert bs.update_indices().shape == (20,)


@pytest.mark.parametrize("studentize_reps", [0, -5])
def test_studentize_reps_positive(studentize_reps):
    x = np.random.default_rng(0).standard_normal(40)
    bs = IIDBootstrap(x, seed=1)
    with pytest.raises(ValueError, match="studentize_reps must be an integer >= 1"):
        bs.conf_int(np.mean, 20, method="studentized", studentize_reps=studentize_reps)


def test_studentize_reps_not_used():
    x = np.random.default_rng(0).standard_normal(40)
    bs = IIDBootstrap(x, seed=1)

    def std_err_func(params, y):
        return np.atleast_1d(y.std() / np.sqrt(y.shape[0]))

    # Ignored if std_err_func is provided or the method is not studentized
    ci = bs.conf_int(
        lambda y: np.atleast_1d(y.mean()),
        20,
        method="studentized",
        studentize_reps=0,
        std_err_func=std_err_func,
    )
    assert ci.shape == (2, 1)
    ci = bs.conf_int(np.mean, 20, method="percentile", studentize_reps=0)
    assert ci.shape == (2, 1)


def _stat_array(y):
    return np.array([y.mean(), y.std()])


def _stat_list(y):
    return [y.mean(), y.std()]


def _stat_tuple(y):
    return (y.mean(), y.std())


def _stat_scalar_array(y):
    return np.array(y.mean())


def _stat_scalar_vector(y):
    return np.array([y.mean()])


def _stat_scalar_float(y):
    return float(y.mean())


@pytest.mark.parametrize(
    ("func", "reference"),
    [
        (_stat_list, _stat_array),
        (_stat_tuple, _stat_array),
        (_stat_scalar_array, _stat_scalar_vector),
        (_stat_scalar_float, _stat_scalar_vector),
    ],
    ids=["list", "tuple", "0-d-array", "float"],
)
@pytest.mark.parametrize(
    "method", ["percentile", "basic", "norm", "bc", "bca", "studentized"]
)
def test_statistic_return_types(func, reference, method):
    x = np.random.default_rng(0).standard_normal(50)
    kwargs = {"studentize_reps": 10} if method == "studentized" else {}
    ci = IIDBootstrap(x, seed=1).conf_int(func, 30, method=method, **kwargs)
    expected = IIDBootstrap(x, seed=1).conf_int(reference, 30, method=method, **kwargs)
    assert ci.shape == expected.shape
    assert_allclose(ci, expected)


@pytest.mark.parametrize(
    ("func", "reference"),
    [
        (_stat_list, _stat_array),
        (_stat_tuple, _stat_array),
        (_stat_scalar_array, _stat_scalar_vector),
        (_stat_scalar_float, _stat_scalar_vector),
    ],
    ids=["list", "tuple", "0-d-array", "float"],
)
def test_statistic_return_types_cov_var_apply(func, reference):
    x = np.random.default_rng(0).standard_normal(50)
    reps = 30
    expected = IIDBootstrap(x, seed=1).apply(reference, reps)
    assert_allclose(IIDBootstrap(x, seed=1).apply(func, reps), expected)
    cov = IIDBootstrap(x, seed=1).cov(func, reps)
    # Hand computation of the covariance using the recentered replications
    centered = expected - expected.mean(0)
    assert_allclose(cov, centered.T @ centered / reps)
    var = IIDBootstrap(x, seed=1).var(func, reps)
    assert_allclose(var, (centered**2).mean(0))


@pytest.mark.parametrize("method", ["apply", "cov", "var", "conf_int"])
def test_extra_kwargs_conflict(method):
    x = np.random.default_rng(0).standard_normal(50)
    bs = IIDBootstrap(x, y=x, seed=1)

    def func(a, y):
        return np.array([y.mean()])

    with pytest.raises(
        ValueError, match="extra_kwargs must be a dict with str keys that are not names"
    ):
        getattr(bs, method)(func, 3, extra_kwargs={"y": np.zeros(50)})


@pytest.mark.parametrize("cls", [IIDBootstrap, IndependentSamplesBootstrap])
def test_repr_html_is_jupyter_hook(cls):
    # Jupyter uses _repr_html_, the name that was misspelled
    bs = cls(np.arange(10.0))
    assert callable(bs._repr_html_)
    assert bs._repr_html_().startswith("<strong>")
    assert not hasattr(bs, "_repr_html")
    for block_cls in BLOCK_BOOTSTRAPS:
        assert block_cls(3, np.arange(10.0))._repr_html_().startswith("<strong>")


def test_bca_bias_message():
    bs = IIDBootstrap(np.arange(10.0))
    bs._base = np.array([1.0])
    bs._results = np.zeros((5, 1))
    with pytest.raises(RuntimeError, match=r"is 0 or 1, and so bias cannot be"):
        bs._bca_bias()
