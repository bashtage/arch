"""
Tests of the Andrews (1991) AR(1) plug-in bandwidth, ``bandwidth_method="andrews"``
"""

from collections.abc import Callable
import re
from typing import Any
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.signal import lfilter

from arch.covariance.kernel import (
    KERNELS,
    Bartlett,
    CovarianceEstimator,
    Parzen,
    QuadraticSpectral,
    TukeyHanning,
    ZeroLag,
    get_kernel_estimator,
)
from arch.covariance.var import PreWhitenedRecolored


def make_data(
    nobs: int,
    rhos: list[float],
    seed: int,
    means: list[float] | None = None,
) -> np.ndarray:
    """AR(1) columns with coefficients rhos and scales 1, 2, ..."""
    e = np.random.default_rng(seed).standard_normal((nobs, len(rhos)))
    x = np.column_stack(
        [lfilter([1.0], [1.0, -r], e[:, i]) * (1 + i) for i, r in enumerate(rhos)]
    )
    if means is not None:
        x = x + np.asarray(means)
    return x


# name: (nobs, AR(1) coefficients, seed, means, weights)
CASES: dict[str, tuple[Any, ...]] = {
    "uni_short": (60, [0.5], 101, None, None),
    "uni_long": (500, [0.8], 102, None, None),
    "uni_mean": (200, [0.3], 103, [5.0], None),
    "multi_short": (50, [0.2, 0.6, -0.3], 104, None, None),
    "multi_long": (400, [0.7, 0.4, 0.9, 0.0], 105, None, [1.0, 2.0, 0.5, 3.0]),
    "multi_zero_weight": (150, [0.5, 0.9], 106, [1.0, -2.0], [1.0, 0.0]),
}

# R 4.6.1, sandwich 3.1.3. The estimating functions are the columns of the data made
# from CASES, passed as a matrix so that bwAndrews uses them as they are. Each value is
#   sandwich::bwAndrews(x, kernel=<name>, approx="AR(1)", weights=w, prewhite=FALSE)
# where w is the weights in CASES (NULL if None), printed with 17 digits. bwAndrews
# fits each AR(1) with ar(x, order.max=1, aic=FALSE, method="ols"), which demeans
# the column and includes an intercept, and sets sigma^2 to the sum of squared
# residuals over nobs - 1.
R_BANDWIDTHS: dict[str, dict[str, float]] = {
    "uni_short": {
        "Bartlett": 7.0037911418620755,
        "Parzen": 13.939904263857645,
        "Quadratic Spectral": 6.924906976495902,
        "Tukey-Hanning": 9.146261676391456,
    },
    "uni_long": {
        "Bartlett": 22.69445759929277,
        "Parzen": 36.73869411194425,
        "Quadratic Spectral": 18.25063030187176,
        "Tukey-Hanning": 24.10502279186783,
    },
    "uni_mean": {
        "Bartlett": 4.302682385722105,
        "Parzen": 7.179646589249792,
        "Quadratic Spectral": 3.5666231140178666,
        "Tukey-Hanning": 4.71071574139475,
    },
    "multi_short": {
        "Bartlett": 8.235119161799497,
        "Parzen": 17.42464412350561,
        "Quadratic Spectral": 8.65601638073449,
        "Tukey-Hanning": 11.432672115602879,
    },
    "multi_long": {
        "Bartlett": 40.85079524385522,
        "Parzen": 77.37996547400593,
        "Quadratic Spectral": 38.43993851100294,
        "Tukey-Hanning": 50.77060784200389,
    },
    "multi_zero_weight": {
        "Bartlett": 7.317015202270406,
        "Parzen": 12.528527575394566,
        "Quadratic Spectral": 6.22377932946162,
        "Tukey-Hanning": 8.220228019897043,
    },
}

R_KERNELS: dict[str, type[CovarianceEstimator]] = {
    "Bartlett": Bartlett,
    "Parzen": Parzen,
    "Quadratic Spectral": QuadraticSpectral,
    "Tukey-Hanning": TukeyHanning,
}

R_PARAMS = [
    pytest.param(case, kern, id=f"{case}-{kern.replace(' ', '-')}")
    for case in R_BANDWIDTHS
    for kern in R_KERNELS
]


def _case(case: str) -> tuple[np.ndarray, list[float] | None]:
    nobs, rhos, seed, means, weights = CASES[case]
    return make_data(nobs, rhos, seed, means), weights


def _opt(est: CovarianceEstimator) -> float:
    return est.opt_bandwidth


def _short_series() -> np.ndarray:
    return np.sin(np.arange(10.0))[:, None] + np.arange(10.0)[:, None] ** 0.5


@pytest.mark.parametrize("center", [True, False], ids=["center", "no-center"])
@pytest.mark.parametrize(("case", "kern"), R_PARAMS)
def test_andrews_matches_r(case: str, kern: str, center: bool) -> None:
    x, weights = _case(case)
    est = R_KERNELS[kern](x, weights=weights, center=center, bandwidth_method="andrews")
    assert_allclose(est.opt_bandwidth, R_BANDWIDTHS[case][kern], rtol=1e-10)
    # No bandwidth was provided, so this is the bandwidth that is used
    assert_allclose(est.bandwidth, R_BANDWIDTHS[case][kern], rtol=1e-10)


def andrews_alpha(x: np.ndarray, weights: list[float] | None, q: float) -> float:
    """
    Andrews (1991) eqs. (6.3) and (6.4). An AR(1) with an intercept is fit to each
    column by least squares, and sigma^2 is the residual sum of squares over nobs - 1.
    """
    nobs, nvar = x.shape
    w = np.ones(nvar) if weights is None else np.asarray(weights, dtype=float)
    num = 0.0
    den = 0.0
    for a in range(nvar):
        design = np.column_stack([np.ones(nobs - 1), x[:-1, a]])
        coef, *_ = np.linalg.lstsq(design, x[1:, a], rcond=None)
        rho = coef[1]
        resid = x[1:, a] - design @ coef
        sigma2 = resid @ resid / (nobs - 1)
        if q == 1:
            term = 4 * rho**2 * sigma2**2 / ((1 - rho) ** 6 * (1 + rho) ** 2)
        else:
            term = 4 * rho**2 * sigma2**2 / (1 - rho) ** 8
        num += w[a] * term
        den += w[a] * sigma2**2 / (1 - rho) ** 4
    return num / den


def test_andrews_by_hand() -> None:
    # One value from the formulas of the paper, using the closed form of the AR(1)
    # for a single series, where sigma^4 cancels
    x = make_data(60, [0.5], 101)[:, 0]
    nobs = x.shape[0]
    y, z = x[1:], x[:-1]
    rho = np.sum((z - z.mean()) * (y - y.mean())) / np.sum((z - z.mean()) ** 2)
    alpha1 = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    alpha2 = 4 * rho**2 / (1 - rho) ** 4
    bartlett = 1.1447 * (alpha1 * nobs) ** (1 / 3)
    assert_allclose(
        Bartlett(x[:, None], bandwidth_method="andrews").opt_bandwidth,
        bartlett,
        rtol=1e-12,
    )
    assert_allclose(
        QuadraticSpectral(x[:, None], bandwidth_method="andrews").opt_bandwidth,
        1.3221 * (alpha2 * nobs) ** (1 / 5),
        rtol=1e-12,
    )
    # This is the value of R for this series
    assert_allclose(bartlett, R_BANDWIDTHS["uni_short"]["Bartlett"], rtol=1e-10)


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("case", ["uni_short", "multi_long", "multi_zero_weight"])
def test_every_kernel_follows_the_formula(name: str, case: str) -> None:
    x, weights = _case(case)
    est = get_kernel_estimator(name)(x, weights=weights, bandwidth_method="andrews")
    q = est.bandwidth_scale
    assert q in (1, 2)
    alpha = andrews_alpha(x, weights, q)
    expected = est.kernel_const * (alpha * x.shape[0]) ** (1 / (2 * q + 1))
    assert_allclose(est.opt_bandwidth, min(expected, x.shape[0] - 1.0), rtol=1e-10)
    assert est.bandwidth_method == "andrews"
    # The covariance can be computed with the bandwidth
    assert np.all(np.isfinite(est.cov.long_run))


@pytest.mark.parametrize(
    ("alias", "base"),
    [
        ("Andrews", "QuadraticSpectral"),
        ("Gallant", "Parzen"),
        ("NeweyWest", "Bartlett"),
    ],
)
def test_aliases_share_the_constants(alias: str, base: str) -> None:
    x, _ = _case("multi_short")
    a = get_kernel_estimator(alias)(x, bandwidth_method="andrews")
    b = get_kernel_estimator(base)(x, bandwidth_method="andrews")
    assert_allclose(a.opt_bandwidth, b.opt_bandwidth, rtol=1e-14)


def _qs(u: float) -> float:
    if u == 0:
        return 1.0
    z = 6 * np.pi * u / 5
    return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))


def _parzen(u: float) -> float:
    return 1 - 6 * u**2 * (1 - u) if u <= 0.5 else 2 * (1 - u) ** 3


PI = np.pi
# kernel: (function on [0, 1], or [0, inf) for the quadratic spectral, q, k_q) in the
# notation of Andrews (1991), where k_q = lim (1 - k(x)) / |x|^q
KERNEL_THEORY: dict[str, tuple[Callable[[float], float], int, float]] = {
    "Bartlett": (lambda u: 1 - u, 1, 1.0),
    "Parzen": (_parzen, 2, 6.0),
    "ParzenRiesz": (lambda u: 1 - u**2, 2, 1.0),
    "ParzenGeometric": (lambda u: 1 / (1 + u), 1, 1.0),
    "ParzenCauchy": (lambda u: 1 / (1 + u**2), 2, 1.0),
    "TukeyHamming": (lambda u: 0.54 + 0.46 * np.cos(PI * u), 2, 0.23 * PI**2),
    "TukeyHanning": (lambda u: 0.5 + 0.5 * np.cos(PI * u), 2, 0.25 * PI**2),
    "TukeyParzen": (lambda u: 0.436 + 0.564 * np.cos(PI * u), 2, 0.282 * PI**2),
    "QuadraticSpectral": (_qs, 2, (6 * PI / 5) ** 2 / 10),
}


@pytest.mark.parametrize("name", list(KERNEL_THEORY))
def test_kernel_constants(name: str) -> None:
    # c_k = (q k_q^2 / int k^2)^(1/(2q+1)), where the integral is over the real line,
    # is the constant of Andrews (1991, Table 1, eqs. (5.3) and (6.2)). The constant
    # of arch is the formula rounded to 4 decimals for every kernel.
    func, q, kq = KERNEL_THEORY[name]
    upper = np.inf if name == "QuadraticSpectral" else 1.0
    integral = 2 * quad(lambda u: func(u) ** 2, 0, upper, limit=500)[0]
    const = (q * kq**2 / integral) ** (1 / (2 * q + 1))
    est = get_kernel_estimator(name)(_short_series())
    assert est.bandwidth_scale == q
    assert_allclose(est.kernel_const, const, rtol=0, atol=5e-5)
    assert_allclose(est.kernel_const, round(const, 4), rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("cls", "const", "q"),
    [
        (Bartlett, 1.1447, 1),
        (Parzen, 2.6614, 2),
        (TukeyHanning, 1.7462, 2),
        (QuadraticSpectral, 1.3221, 2),
    ],
)
def test_paper_constants(cls: type[CovarianceEstimator], const: float, q: int) -> None:
    # The four constants of Andrews (1991) that sandwich::bwAndrews uses
    est = cls(_short_series())
    assert_allclose(est.kernel_const, const, rtol=1e-14)
    assert est.bandwidth_scale == q


def test_zero_lag_ignores_method() -> None:
    x, _ = _case("multi_short")
    est = ZeroLag(x, bandwidth_method="andrews")
    assert est.bandwidth == 0.0
    assert est.opt_bandwidth == 0.0
    assert est.bandwidth_method == "andrews"
    assert_allclose(est.cov.long_run, ZeroLag(x).cov.long_run, rtol=1e-14)


def test_explicit_bandwidth_overrides_method() -> None:
    x, _ = _case("multi_short")
    est = Bartlett(x, bandwidth=3.0, bandwidth_method="andrews")
    assert est.bandwidth == 3.0
    assert_allclose(est.cov.long_run, Bartlett(x, bandwidth=3.0).cov.long_run)
    assert "Automatic Bandwidth: False" in str(est)


def test_force_int() -> None:
    x, weights = _case("multi_long")
    est = Parzen(x, weights=weights, bandwidth_method="andrews")
    est_int = Parzen(x, weights=weights, bandwidth_method="andrews", force_int=True)
    assert est.bandwidth != np.ceil(est.bandwidth)
    assert est_int.bandwidth == np.ceil(est.bandwidth)
    assert est_int.opt_bandwidth == np.ceil(est.opt_bandwidth)


def test_bandwidth_limited_to_nobs() -> None:
    x = np.cumsum(np.random.default_rng(7).standard_normal((30, 1)), axis=0)
    unrestricted = 1.1447 * (andrews_alpha(x, None, 1) * 30) ** (1 / 3)
    assert unrestricted > 29
    assert Bartlett(x, bandwidth_method="andrews").opt_bandwidth == 29.0
    assert Bartlett(x, bandwidth_method="andrews", force_int=True).bandwidth == 29.0


def test_constant_column_is_ignored() -> None:
    x, _ = _case("uni_short")
    x2 = np.column_stack([x, np.full(x.shape[0], 0.1), np.zeros(x.shape[0])])
    ref = Parzen(x, bandwidth_method="andrews")
    for w in (None, [1.0, 2.0, 3.0]):
        est = Parzen(x2, weights=w, bandwidth_method="andrews")
        assert_allclose(est.opt_bandwidth, ref.opt_bandwidth, rtol=1e-12)


def test_zero_weight_column_is_not_used() -> None:
    # The second column is constant and has no weight
    x, _ = _case("uni_short")
    x2 = np.column_stack([x, np.full(x.shape[0], 3.0)])
    est = Bartlett(x2, weights=[1.0, 0.0], bandwidth_method="andrews")
    assert_allclose(
        est.opt_bandwidth,
        Bartlett(x, bandwidth_method="andrews").opt_bandwidth,
        rtol=1e-12,
    )


def test_pandas_input() -> None:
    x, _ = _case("multi_short")
    df = pd.DataFrame(x, columns=["a", "b", "c"])
    assert_allclose(
        Bartlett(df, bandwidth_method="andrews").opt_bandwidth,
        Bartlett(x, bandwidth_method="andrews").opt_bandwidth,
        rtol=1e-12,
    )


def test_complex_data_uses_real_and_imaginary_parts() -> None:
    x, _ = _case("multi_short")
    z = x[:, :2] + 1j * x[:, 1:]
    stacked = np.column_stack([z.real, z.imag])
    est = Bartlett(z, weights=[1.0, 2.0], bandwidth_method="andrews")
    ref = Bartlett(stacked, weights=[1.0, 2.0, 1.0, 2.0], bandwidth_method="andrews")
    assert_allclose(est.opt_bandwidth, ref.opt_bandwidth, rtol=1e-12)
    # Real data stored as complex numbers have the bandwidth of the real data
    real = Bartlett(x[:, :2], bandwidth_method="andrews")
    cplx = Bartlett(x[:, :2] + 0j, bandwidth_method="andrews")
    assert_allclose(cplx.opt_bandwidth, real.opt_bandwidth, rtol=1e-12)


def test_all_constant_raises() -> None:
    match = r"every column of x with a positive weight is constant"
    with pytest.raises(ValueError, match=match):
        _opt(Bartlett(np.ones((20, 2)), bandwidth_method="andrews"))
    x = np.column_stack([np.ones(20), np.random.default_rng(3).standard_normal(20)])
    with pytest.raises(ValueError, match=match):
        _opt(Bartlett(x, weights=[1.0, 0.0], bandwidth_method="andrews"))


def test_constant_except_final_value_raises() -> None:
    x = np.zeros((20, 1))
    x[-1] = 1.0
    with pytest.raises(ValueError, match="constant except for its final value"):
        _opt(Bartlett(x, bandwidth_method="andrews"))


def test_unit_coefficient_raises() -> None:
    # The AR(1) coefficient of a line is exactly 1
    x = np.arange(20.0)[:, None]
    with pytest.raises(ValueError, match="coefficient of x is -1 or 1"):
        _opt(Bartlett(x, bandwidth_method="andrews"))


class _Cubic(Bartlett):
    @property
    def bandwidth_scale(self) -> float:
        return 3.0


def test_unsupported_exponent_raises() -> None:
    x, _ = _case("uni_short")
    with pytest.raises(ValueError, match="exponent of 1 or 2"):
        _opt(_Cubic(x, bandwidth_method="andrews"))
    # The default method does not restrict the exponent
    assert _Cubic(x).opt_bandwidth > 0


def _baseline_data(nobs: int, nvar: int, seed: int) -> np.ndarray:
    e = np.random.default_rng(seed).standard_normal((nobs, nvar))
    return np.column_stack(
        [lfilter([1.0], [1.0, -0.3 * (i + 1)], e[:, i]) for i in range(nvar)]
    )


# The bandwidth and the sum of the long-run covariance of every kernel with the default
# method, from the code before bandwidth_method was added (origin/main, d2c90b43c9).
# A has 200 rows and 3 columns, and B has 40 rows and 2 columns. The Tukey-Parzen
# constant was 1.8576 then and is now 1.8587, so its bandwidth is scaled by the ratio.
BASELINE: dict[tuple[str, str], tuple[float, float]] = {
    ("A", "Bartlett"): (11.272999809578199, 67.79495170599591),
    ("A", "Parzen"): (16.951578711405514, 73.56762377158034),
    ("A", "ParzenCauchy"): (6.957956182587879, 66.92788827410591),
    ("A", "ParzenGeometric"): (9.847994941537694, 72.97698452862986),
    ("A", "ParzenRiesz"): (7.222924122166472, 66.56590007494339),
    ("A", "TukeyHamming"): (10.63311245991597, 70.81081202535776),
    ("A", "TukeyHanning"): (11.122284040676451, 70.07504762843789),
    ("A", "TukeyParzen"): (11.831837609644127, 68.4912499330181),
    ("A", "QuadraticSpectral"): (8.421012329732182, 65.15275977553551),
    ("A", "Andrews"): (8.421012329732182, 65.15275977553551),
    ("A", "Gallant"): (16.951578711405514, 73.56762377158034),
    ("A", "NeweyWest"): (11.272999809578199, 67.79495170599591),
    ("B", "Bartlett"): (5.022996879508972, 9.404459858424694),
    ("B", "Parzen"): (9.372543954848854, 10.187548531018887),
    ("B", "ParzenCauchy"): (3.8470605757409215, 10.43437204791627),
    ("B", "ParzenGeometric"): (4.388046544517317, 10.130793300951574),
    ("B", "ParzenRiesz"): (3.9935616009613732, 10.25138679669899),
    ("B", "TukeyHamming"): (5.8790579688226785, 10.536883438965251),
    ("B", "TukeyHanning"): (6.149521399998898, 10.40382167954531),
    ("B", "TukeyParzen"): (6.541834241574821, 10.495206711730397),
    ("B", "QuadraticSpectral"): (4.655985707787506, 10.295574535000805),
    ("B", "Andrews"): (4.655985707787506, 10.295574535000805),
    ("B", "Gallant"): (9.372543954848854, 10.187548531018887),
    ("B", "NeweyWest"): (5.022996879508972, 9.404459858424694),
}


@pytest.mark.parametrize("key", list(BASELINE), ids=["-".join(k) for k in BASELINE])
def test_default_method_is_unchanged(key: tuple[str, str]) -> None:
    which, name = key
    x = _baseline_data(200, 3, 2025) if which == "A" else _baseline_data(40, 2, 2026)
    bandwidth, total = BASELINE[key]
    cls = get_kernel_estimator(name)
    if name == "TukeyParzen":
        bandwidth *= 1.8587 / 1.8576
        total = float(cls(x, bandwidth=bandwidth).cov.long_run.sum())
    for est in (cls(x), cls(x, bandwidth_method="newey-west")):
        assert est.bandwidth_method == "newey-west"
        assert_allclose(est.opt_bandwidth, bandwidth, rtol=1e-12)
        assert_allclose(est.cov.long_run.sum(), total, rtol=1e-12)


def test_methods_differ() -> None:
    x, _ = _case("uni_long")
    nw = Bartlett(x).opt_bandwidth
    andrews = Bartlett(x, bandwidth_method="andrews").opt_bandwidth
    assert abs(nw - andrews) > 1.0


@pytest.mark.parametrize("spelling", ["andrews", "Andrews", "ANDREWS", "aNdReWs"])
def test_method_case_insensitive(spelling: str) -> None:
    x, _ = _case("multi_short")
    est = Parzen(x, bandwidth_method=spelling)
    assert est.bandwidth_method == "andrews"
    assert_allclose(
        est.opt_bandwidth,
        Parzen(x, bandwidth_method="andrews").opt_bandwidth,
        rtol=1e-14,
    )


def test_prewhitened_recolored() -> None:
    x, _ = _case("multi_long")
    pw = PreWhitenedRecolored(x, lags=1, bandwidth_method="andrews")
    assert pw.bandwidth_method == "andrews"
    resid = pw._setup()[0].resids
    expected = Bartlett(resid, center=False, bandwidth_method="andrews").opt_bandwidth
    assert_allclose(pw.opt_bandwidth, expected, rtol=1e-12)
    default = PreWhitenedRecolored(x, lags=1)
    assert default.bandwidth_method == "newey-west"
    assert_allclose(
        default.opt_bandwidth, Bartlett(resid, center=False).opt_bandwidth, rtol=1e-12
    )
    assert abs(default.opt_bandwidth - pw.opt_bandwidth) > 0.1
    # A provided bandwidth is used, and kernel=None has no bandwidth
    provided = PreWhitenedRecolored(
        x, lags=1, bandwidth=2.0, bandwidth_method="andrews"
    )
    assert provided.bandwidth == 2.0
    none = PreWhitenedRecolored(x, lags=1, kernel=None, bandwidth_method="andrews")
    assert none.bandwidth == 0.0


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [
        ("newey-west", "newey-west"),
        ("Newey-West", "newey-west"),
        ("newey_west", "newey-west"),
        ("Newey_West", "newey-west"),
        ("neweywest", "newey-west"),
        ("NeweyWest", "newey-west"),
        ("NEWEYWEST", "newey-west"),
        ("andrews", "andrews"),
        ("Andrews", "andrews"),
        ("ANDREWS", "andrews"),
    ],
)
def test_method_spellings(spelling: str, canonical: str) -> None:
    # The spellings of kernel names are accepted: case, hyphens and underscores
    # are ignored
    x, _ = _case("multi_short")
    expected = Parzen(x, bandwidth_method=canonical)
    est = Parzen(x, bandwidth_method=spelling)
    assert est.bandwidth_method == canonical
    assert_allclose(est.opt_bandwidth, expected.opt_bandwidth, rtol=1e-14)
    pw = PreWhitenedRecolored(x, lags=1, bandwidth_method=spelling)
    assert pw.bandwidth_method == canonical
    expected_pw = PreWhitenedRecolored(x, lags=1, bandwidth_method=canonical)
    assert_allclose(pw.opt_bandwidth, expected_pw.opt_bandwidth, rtol=1e-14)


def _scale_cases() -> list[tuple[str, np.ndarray, list[float] | None]]:
    uni, _ = _case("uni_short")
    multi, weights = _case("multi_long")
    cplx = multi[:, :2] + 1j * multi[:, -2:]
    return [("uni", uni, None), ("multi", multi, weights), ("complex", cplx, None)]


@pytest.mark.parametrize("method", ["newey-west", "andrews"])
@pytest.mark.parametrize("cls", [Bartlett, Parzen, QuadraticSpectral])
@pytest.mark.parametrize("case", ["uni", "multi", "complex"])
def test_extreme_scale(method: str, cls: type[CovarianceEstimator], case: str) -> None:
    # The estimates of alpha(q) do not depend on the scale of the data, and the data
    # are scaled by a power of 2 before they are computed, so that a very large or
    # very small scale neither overflows nor underflows
    x, weights = {name: (x, w) for name, x, w in _scale_cases()}[case]
    expected = cls(x, weights=weights, bandwidth_method=method).opt_bandwidth
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for scale in (1e-200, 1e-150, 1e-100, 1e100, 1e150, 1e200):
            est = cls(scale * x, weights=weights, bandwidth_method=method)
            assert_allclose(est.opt_bandwidth, expected, rtol=1e-12)
        # A power of 2 is exact, and so is the bandwidth
        for power in (-600, -300, 300, 600):
            est = cls(x * 2.0**power, weights=weights, bandwidth_method=method)
            assert est.opt_bandwidth == expected


@pytest.mark.parametrize("bad", ["bogus", "", "newey west", "newey", "andrew"])
def test_invalid_method_value(bad: str) -> None:
    x, _ = _case("uni_short")
    match = (
        r"^bandwidth_method must be one of 'newey-west' or 'andrews'; got "
        + re.escape(repr(bad))
    )
    with pytest.raises(ValueError, match=match):
        Bartlett(x, bandwidth_method=bad)


@pytest.mark.parametrize("bad", [None, 1, 1.0, b"andrews", ["andrews"]], ids=str)
def test_invalid_method_type(bad: object) -> None:
    x, _ = _case("uni_short")
    with pytest.raises(TypeError, match=r"^bandwidth_method must be one of"):
        Bartlett(x, bandwidth_method=bad)  # type: ignore[arg-type]
