from fractions import Fraction

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest

from arch.covariance.kernel import (
    Bartlett,
    Parzen,
    ParzenCauchy,
    ParzenGeometric,
    ParzenRiesz,
    QuadraticSpectral,
    TukeyHamming,
    TukeyHanning,
    TukeyParzen,
    ZeroLag,
)
from arch.covariance.var import PreWhitenedRecolored
from arch.utility.cov import cov_nw


@pytest.mark.parametrize("dtype", [np.int16, np.uint8, np.int64, np.bool_])
@pytest.mark.parametrize("lags", [0, 2])
def test_uncentered_discrete_covariance(dtype, lags):
    data = np.arange(1, 13).reshape(6, 2) * 20
    if dtype == np.bool_:
        data = data % 60 == 0
    data = data.astype(dtype)
    original = data.copy()
    nobs, nvar = data.shape
    short_run = np.array(
        [
            [sum(int(row[i]) * int(row[j]) for row in data) / nobs for j in range(nvar)]
            for i in range(nvar)
        ]
    )
    strict = np.zeros((nvar, nvar))
    for lag in range(1, lags + 1):
        weight = Fraction(lags + 1 - lag, lags + 1)
        for i in range(nvar):
            for j in range(nvar):
                products = sum(
                    int(data[t, i]) * int(data[t - lag, j]) for t in range(lag, nobs)
                )
                strict[i, j] += float(weight * products / nobs)
    expected = short_run + strict + strict.T

    cov = Bartlett(data, bandwidth=lags, center=False).cov

    assert_allclose(cov.short_run, short_run)
    assert_allclose(cov.one_sided_strict, strict)
    assert_allclose(cov.long_run, expected)
    assert_allclose(cov_nw(data, lags=lags, demean=False), expected)
    assert_array_equal(data, original)


@pytest.mark.parametrize("lags", [0, 2])
def test_uncentered_integer_newey_west(lags):
    data = np.array([2**32 + 1, 2**32 + 2, 2**32 + 3], dtype=np.int64)
    nobs = len(data)
    expected = Fraction(sum(int(x) ** 2 for x in data), nobs)
    for lag in range(1, lags + 1):
        products = sum(int(data[t]) * int(data[t - lag]) for t in range(lag, nobs))
        expected += Fraction(2 * (lags + 1 - lag) * products, (lags + 1) * nobs)

    assert_allclose(cov_nw(data, lags=lags, demean=False), float(expected))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_floating_kernel_dtype_is_float64(dtype):
    # float32 data are promoted to float64 on entry, and so are all of the estimates
    # (an earlier version preserved float32 in short_run only, and long_run was float64)
    data = np.arange(12).reshape(6, 2).astype(dtype) / 10
    result = Bartlett(data, bandwidth=2, center=False).cov
    data64 = data.astype(np.float64)
    expected = data64.T @ data64 / len(data)
    for name in ("short_run", "one_sided_strict", "one_sided", "long_run"):
        assert getattr(result, name).dtype == np.float64
    assert_allclose(result.short_run, expected, rtol=1e-14)
    float64_result = Bartlett(data64, bandwidth=2, center=False).cov
    assert_allclose(result.long_run, float64_result.long_run, rtol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("lags", [0, 2])
def test_inexact_newey_west(dtype, lags):
    data = np.arange(12).reshape(6, 2).astype(dtype) / 10
    is_complex = np.issubdtype(dtype, np.complexfloating)
    if is_complex:
        data = data + 1j * data[::-1]
    # float32 and complex64 data are promoted to float64 and complex128
    data64 = data.astype(np.complex128 if is_complex else np.float64)
    # The Hermitian covariance has the conjugate transpose, so that a covariance matrix
    # is Hermitian with a real diagonal: data.T @ data is the pseudo-covariance
    expected = data64.T @ data64.conj()
    for lag in range(1, lags + 1):
        gamma = data64[lag:].T @ data64[:-lag].conj()
        expected += (1 - lag / (lags + 1)) * (gamma + gamma.conj().T)
    expected /= len(data)
    result = cov_nw(data, lags=lags, demean=False)
    assert result.dtype == expected.dtype
    assert_allclose(result, expected, rtol=1e-14)
    if is_complex:
        assert_array_equal(result, result.conj().T)
        assert_array_equal(np.diag(result).imag, 0)


# ---------------------------------------------------------------------------------------
# Complex data: the estimators are Hermitian
# ---------------------------------------------------------------------------------------
KERNELS = [
    Bartlett,
    Parzen,
    ParzenCauchy,
    ParzenGeometric,
    ParzenRiesz,
    TukeyHamming,
    TukeyHanning,
    TukeyParzen,
    QuadraticSpectral,
    ZeroLag,
]
ESTIMATES = ("long_run", "short_run", "one_sided", "one_sided_strict")
NVAR = 3


def _real_and_imaginary(seed=12345, nobs=300, mean=3.0):
    """Serially and cross correlated real and imaginary parts with non-zero means."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((nobs + 1, 2 * NVAR))
    ab = e[1:] + 0.5 * e[:-1]
    ab[:, :NVAR] += 0.4 * ab[:, NVAR:]
    ab += mean
    return ab[:, :NVAR], ab[:, NVAR:]


def _hermitian_from_real(stacked, nvar=NVAR):
    """
    The Hermitian matrix from the covariance of the real data [a, b].

    For z = a + ib, E[z z^H] = S_aa + S_bb + i (S_ba - S_ab) where S_uv is the
    covariance of u and v.
    """
    s = np.asarray(stacked)
    s_aa, s_bb = s[:nvar, :nvar], s[nvar:, nvar:]
    s_ab, s_ba = s[:nvar, nvar:], s[nvar:, :nvar]
    return s_aa + s_bb + 1j * (s_ba - s_ab)


@pytest.mark.parametrize("kernel", KERNELS, ids=[k.__name__ for k in KERNELS])
@pytest.mark.parametrize("center", [True, False], ids=["center", "no-center"])
@pytest.mark.parametrize("df_adjust", [0, 2])
def test_hermitian_identity(kernel, center, df_adjust):
    """
    The complex estimates are the Hermitian combination of the real estimates of [a, b].

    This is the reference: the real estimator, applied to the real data of the real and
    imaginary parts with the same kernel, bandwidth, centering and degrees of freedom,
    does not use a conjugate or a complex number.
    """
    a, b = _real_and_imaginary()
    bandwidth = None if kernel is ZeroLag else 6.5
    options = {"bandwidth": bandwidth, "center": center, "df_adjust": df_adjust}
    result = kernel(a + 1j * b, **options).cov
    reference = kernel(np.column_stack([a, b]), **options).cov
    for name in ESTIMATES:
        actual = getattr(result, name)
        assert actual.dtype == np.complex128
        expected = _hermitian_from_real(getattr(reference, name))
        assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    long_run = np.asarray(result.long_run)
    assert_array_equal(long_run, long_run.conj().T)
    assert_array_equal(np.diag(long_run).imag, 0)


def _automatic_bandwidth(kernel, z, weights, center):
    """
    The automatic bandwidth from the real and imaginary parts of the combined series.

    The sums use the lag-j autocovariance of v = w'z, which is the sum of the
    autocovariances of the real and imaginary parts of v. The sums over j and -j are
    added, which is twice the real part.
    """
    constants = kernel(np.random.default_rng(0).standard_normal((20, 1)), bandwidth=1.0)
    x = z - z.mean(0) if center else z
    p, q = x.real @ weights, x.imag @ weights
    nobs = p.shape[0]
    n = int(np.ceil(4 * ((nobs / 100) ** constants.rate)))
    f_0 = f_q = 0.0
    for j in range(n + 1):
        sigma = (p[j:] @ p[: nobs - j] + q[j:] @ q[: nobs - j]) / nobs
        scale = 1 + (j != 0)
        f_0 += scale * sigma
        f_q += scale * j**constants.bandwidth_scale * sigma
    alpha = (f_q / f_0) ** 2
    bandwidth = constants.kernel_const * (alpha * nobs) ** (
        1 / (2 * constants.bandwidth_scale + 1)
    )
    return min(bandwidth, nobs - 1.0)


@pytest.mark.parametrize(
    "kernel", [Bartlett, Parzen, QuadraticSpectral], ids=lambda k: k.__name__
)
@pytest.mark.parametrize("center", [True, False], ids=["center", "no-center"])
@pytest.mark.parametrize("weights", [None, [1.0, 2.0, 0.5]], ids=["ones", "weights"])
def test_hermitian_automatic_bandwidth(kernel, center, weights):
    a, b = _real_and_imaginary()
    z = a + 1j * b
    w = np.ones(NVAR) if weights is None else np.asarray(weights)
    estimator = kernel(z, center=center, weights=weights)
    expected = _automatic_bandwidth(kernel, z, w, center)
    assert_allclose(estimator.bandwidth, expected, rtol=1e-10)
    # With that bandwidth the estimates are those of the real data
    reference = kernel(
        np.column_stack([a, b]), bandwidth=estimator.bandwidth, center=center
    ).cov
    assert_allclose(
        estimator.cov.long_run,
        _hermitian_from_real(reference.long_run),
        rtol=1e-11,
        atol=1e-12,
    )


@pytest.mark.parametrize("kernel", KERNELS, ids=[k.__name__ for k in KERNELS])
def test_complex_with_zero_imaginary_part_is_real(kernel):
    a, _ = _real_and_imaginary()
    result = kernel(a.astype(complex)).cov
    reference = kernel(a).cov
    for name in ESTIMATES:
        actual = getattr(result, name)
        assert actual.dtype == np.complex128
        assert_array_equal(actual.imag, 0)
        assert_allclose(actual.real, getattr(reference, name), rtol=1e-12)


@pytest.mark.parametrize("kernel", KERNELS, ids=[k.__name__ for k in KERNELS])
def test_complex_phase_does_not_change_the_estimates(kernel):
    # A Hermitian covariance does not depend on a rotation of the data, while the
    # pseudo-covariance, which has no conjugate, is multiplied by exp(2i theta)
    a, b = _real_and_imaginary()
    z = a + 1j * b
    rotated = z * np.exp(1j * 0.7)
    result = kernel(z).cov
    result_rotated = kernel(rotated).cov
    assert_allclose(kernel(rotated).bandwidth, kernel(z).bandwidth, rtol=1e-10)
    for name in ESTIMATES:
        assert_allclose(
            getattr(result_rotated, name), getattr(result, name), rtol=1e-10
        )


def test_complex_dataframe_keeps_labels():
    a, b = _real_and_imaginary()
    columns = ["u", "v", "w"]
    frame = pd.DataFrame(a + 1j * b, columns=columns)
    result = Bartlett(frame, bandwidth=4).cov
    for name in ESTIMATES:
        estimate = getattr(result, name)
        assert isinstance(estimate, pd.DataFrame)
        assert list(estimate.columns) == columns
        assert list(estimate.index) == columns
        assert all(dtype == np.complex128 for dtype in estimate.dtypes)
    reference = Bartlett(a + 1j * b, bandwidth=4).cov
    assert_allclose(result.long_run, reference.long_run, rtol=1e-14)


def test_complex_input_is_not_modified():
    a, b = _real_and_imaginary()
    z = a + 1j * b
    original = z.copy()
    _ = Bartlett(z, bandwidth=3).cov.long_run
    cov_nw(z, lags=3)
    assert_array_equal(z, original)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_cov_nw_complex_matches_the_real_reference(dtype):
    a, b = _real_and_imaginary()
    z = (a + 1j * b).astype(dtype)
    z64 = z.astype(np.complex128)
    for lags in (0, 4):
        for demean in (True, False):
            actual = cov_nw(z, lags=lags, demean=demean)
            reference = cov_nw(
                np.column_stack([z64.real, z64.imag]), lags=lags, demean=demean
            )
            assert actual.dtype == np.complex128
            assert_allclose(
                actual, _hermitian_from_real(reference), rtol=1e-10, atol=1e-12
            )


def test_cov_nw_complex_1d_is_a_real_float():
    a, b = _real_and_imaginary()
    z = a[:, 0] + 1j * b[:, 0]
    actual = cov_nw(z, lags=3)
    reference = cov_nw(np.column_stack([a[:, 0], b[:, 0]]), lags=3)
    assert isinstance(actual, float)
    assert_allclose(actual, reference[0, 0] + reference[1, 1], rtol=1e-12)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("kernel", [None, "bartlett"])
def test_prewhitened_recolored_rejects_complex(dtype, kernel):
    a, b = _real_and_imaginary()
    z = (a + 1j * b).astype(dtype)
    text = np.dtype(dtype).name
    with pytest.raises(TypeError, match=rf"^x must be .*; got {text} data with shape"):
        PreWhitenedRecolored(z, kernel=kernel)
    # The real and imaginary parts, stacked, are the way to use a VAR
    stacked = np.column_stack([z.real, z.imag])
    result = PreWhitenedRecolored(stacked, kernel=kernel, lags=1).cov.long_run
    assert np.asarray(result).dtype == np.float64


def test_kernel_weights_must_be_real():
    a, b = _real_and_imaginary()
    with pytest.raises(TypeError, match=r"^weights must be .*; got complex128 data"):
        Bartlett(a + 1j * b, weights=np.ones(NVAR) * (1 + 1j))


@pytest.mark.parametrize("kernel", [Bartlett, QuadraticSpectral])
def test_float16_is_rejected(kernel):
    a, _ = _real_and_imaginary()
    with pytest.raises(TypeError, match=r"^x must be .*float32.*; got float16 data"):
        kernel(a.astype(np.float16))
    with pytest.raises(TypeError, match=r"^y must be .*float32.*; got float16 data"):
        cov_nw(a.astype(np.float16))


# ---------------------------------------------------------------------------------------
# The precision of float32 data
# ---------------------------------------------------------------------------------------
def _exact_bartlett_long_run(x, bandwidth):
    """Long-run covariance of the columns of x using exact rational arithmetic."""
    nobs, nvar = x.shape
    values = [[Fraction(float(v)) for v in row] for row in x]
    mean = [sum(values[t][i] for t in range(nobs)) / nobs for i in range(nvar)]
    centered = [[values[t][i] - mean[i] for i in range(nvar)] for t in range(nobs)]
    result = np.zeros((nvar, nvar))
    for i in range(nvar):
        for j in range(nvar):
            total = Fraction(0)
            for lag in range(bandwidth + 1):
                weight = Fraction(bandwidth + 1 - lag, bandwidth + 1)
                gamma_ij = sum(
                    centered[t][i] * centered[t - lag][j] for t in range(lag, nobs)
                )
                gamma_ji = sum(
                    centered[t][j] * centered[t - lag][i] for t in range(lag, nobs)
                )
                total += weight * (gamma_ij if lag == 0 else gamma_ij + gamma_ji)
            result[i, j] = float(total / nobs)
    return result


def test_float32_data_with_a_large_mean_are_computed_in_float64():
    # The mean is 1e4 and the standard deviation is 1. Centering in float32 loses digits
    # when the mean is much larger than the standard deviation, and the products and
    # sums lose more, so float32 data are converted to float64 and the estimates are
    # exact up to the rounding in float64 of the values of the float32 data
    rng = np.random.default_rng(31)
    e = rng.standard_normal((41, 2))
    x32 = (1e4 + e[1:] + 0.5 * e[:-1]).astype(np.float32)
    expected = _exact_bartlett_long_run(x32, 3)
    result = Bartlett(x32, bandwidth=3).cov
    assert result.long_run.dtype == np.float64
    assert_allclose(result.long_run, expected, rtol=1e-11)
    assert_allclose(cov_nw(x32, lags=3), expected, rtol=1e-11)
