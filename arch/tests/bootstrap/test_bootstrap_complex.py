"""
Bootstraps of complex data and of statistics that are complex.

The data of the bootstraps are not converted, so complex data are resampled like any
other data. A statistic that returns complex values is stored as complex128 (its
imaginary part used to be discarded with a ComplexWarning). Its covariance is the
Hermitian covariance and its variance is real. A confidence interval is not defined for
a complex statistic.
"""

import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from arch.bootstrap import (
    CircularBlockBootstrap,
    IIDBootstrap,
    LocalBlockBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
)

NOBS = 150
NVAR = 2
REPS = 400


def _data():
    rng = np.random.default_rng(2024)
    a = rng.standard_normal((NOBS, NVAR))
    b = rng.standard_normal((NOBS, NVAR)) + 0.5 * a
    return a, b


def _mean(x):
    return x.mean(0)


def _hermitian(stacked):
    s_aa, s_bb = stacked[:NVAR, :NVAR], stacked[NVAR:, NVAR:]
    s_ab, s_ba = stacked[:NVAR, NVAR:], stacked[NVAR:, :NVAR]
    return s_aa + s_bb + 1j * (s_ba - s_ab)


def _bootstraps():
    return [
        (IIDBootstrap, (), "iid"),
        (StationaryBootstrap, (10,), "stationary"),
        (CircularBlockBootstrap, (10,), "circular"),
        (MovingBlockBootstrap, (10,), "moving"),
        (LocalBlockBootstrap, (10, 20), "local"),
    ]


@pytest.mark.parametrize(
    ("cls", "args"),
    [(c, a) for c, a, _ in _bootstraps()],
    ids=[i for _, _, i in _bootstraps()],
)
def test_complex_statistic(cls, args):
    a, b = _data()
    z = a + 1j * b
    # The same seed gives the same observations in each bootstrap sample
    complex_bs = cls(*args, z, seed=7)
    real_bs = cls(*args, np.column_stack([a, b]), seed=7)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        applied = complex_bs.apply(_mean, 20)
    assert applied.dtype == np.complex128
    reference = cls(*args, np.column_stack([a, b]), seed=7).apply(_mean, 20)
    assert_allclose(applied, reference[:, :NVAR] + 1j * reference[:, NVAR:], rtol=1e-12)

    cov = cls(*args, z, seed=7).cov(_mean, REPS)
    expected = _hermitian(real_bs.cov(_mean, REPS))
    assert cov.dtype == np.complex128
    assert_allclose(cov, expected, rtol=1e-10, atol=1e-14)
    assert_allclose(cov, np.asarray(cov).conj().T, rtol=1e-14)

    var = cls(*args, z, seed=7).var(_mean, REPS)
    assert var.dtype == np.float64
    assert_allclose(var, np.diag(expected).real, rtol=1e-10)


@pytest.mark.parametrize("recenter", [True, False])
def test_complex_statistic_recenter(recenter):
    a, b = _data()
    cov = IIDBootstrap(a + 1j * b, seed=3).cov(_mean, REPS, recenter=recenter)
    reference = IIDBootstrap(np.column_stack([a, b]), seed=3).cov(
        _mean, REPS, recenter=recenter
    )
    assert_allclose(cov, _hermitian(reference), rtol=1e-10, atol=1e-14)


def test_scalar_complex_statistic():
    a, b = _data()
    z = a[:, 0] + 1j * b[:, 0]
    applied = IIDBootstrap(z, seed=1).apply(np.mean, 10)
    assert applied.dtype == np.complex128
    assert applied.shape == (10, 1)
    var = IIDBootstrap(z, seed=1).var(np.mean, 100)
    assert np.isreal(var).all()


@pytest.mark.parametrize(
    "method", ["basic", "percentile", "norm", "bc", "bca", "studentized"]
)
def test_complex_statistic_has_no_confidence_interval(method):
    a, b = _data()
    bs = IIDBootstrap(a + 1j * b, seed=1)
    with pytest.raises(TypeError, match=r"^the value returned by func must be real"):
        bs.conf_int(_mean, 20, method=method, studentize_reps=5)


def test_real_statistics_are_unchanged():
    a, _ = _data()
    bs = IIDBootstrap(a, seed=5)
    assert bs.apply(_mean, 10).dtype == np.float64
    assert IIDBootstrap(a, seed=5).cov(_mean, 50).dtype == np.float64
    assert IIDBootstrap(a, seed=5).var(_mean, 50).dtype == np.float64
    assert IIDBootstrap(a, seed=5).conf_int(_mean, 50).shape == (2, NVAR)


def test_complex_data_are_resampled_as_given():
    a, b = _data()
    z = a + 1j * b
    original = z.copy()
    bs = IIDBootstrap(z, seed=11)
    for pos, _ in bs.bootstrap(3):
        sample = pos[0]
        assert sample.dtype == np.complex128
        # Every row of the sample is a row of the data
        for row in sample:
            assert any(np.array_equal(row, value) for value in z)
    assert_array_equal(z, original)
