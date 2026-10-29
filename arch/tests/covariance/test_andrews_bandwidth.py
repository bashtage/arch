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


# R 4.6.1, sandwich 3.1.3: the diagonal and the sum of the long-run covariance
#   nobs * kernHAC(lm(x ~ 1), bw=bw, kernel=<name>, prewhite=FALSE, adjust=FALSE)
# with bw from R_BANDWIDTHS, the covariance of the Andrews (1991) estimator.
R_ANDREWS_LONG_RUN: dict[str, dict[str, tuple[list[float], float]]] = {
    "uni_short": {
        "Bartlett": ([2.6950887103082284], 2.6950887103082284),
        "Parzen": ([1.927801221779428], 1.927801221779428),
        "Quadratic Spectral": ([2.206398258492332], 2.206398258492332),
        "Tukey-Hanning": ([2.2974635965820576], 2.2974635965820576),
    },
    "uni_long": {
        "Bartlett": ([16.504064337618196], 16.504064337618196),
        "Parzen": ([18.115528554593638], 18.115528554593638),
        "Quadratic Spectral": ([17.65047557413648], 17.65047557413648),
        "Tukey-Hanning": ([17.92820969313286], 17.92820969313286),
    },
    "uni_mean": {
        "Bartlett": ([1.3627574146805939], 1.3627574146805939),
        "Parzen": ([1.4240600851216039], 1.4240600851216039),
        "Quadratic Spectral": ([1.4527735533509207], 1.4527735533509207),
        "Tukey-Hanning": ([1.435123422415427], 1.435123422415427),
    },
    "multi_short": {
        "Bartlett": (
            [1.1327545703278654, 18.22365065924197, 4.733268156452423],
            25.76576937369616,
        ),
        "Parzen": (
            [0.7557788667918434, 20.633989003651408, 2.888901042499894],
            27.78304052864491,
        ),
        "Quadratic Spectral": (
            [0.9664786935159582, 19.88363435452576, 3.1009193574072493],
            26.857907525889644,
        ),
        "Tukey-Hanning": (
            [0.9619239803731144, 20.253872392057996, 3.379993761002658],
            27.5458560262804,
        ),
    },
    "multi_long": {
        "Bartlett": (
            [
                6.676433679548986,
                2.5645079165835893,
                872.8784616009508,
                11.693051330496703,
            ],
            711.5105903463049,
        ),
        "Parzen": (
            [
                5.8998602175418755,
                1.5825689897477098,
                956.9894612594361,
                11.571982155951371,
            ],
            773.6832571372834,
        ),
        "Quadratic Spectral": (
            [
                7.05773864574444,
                1.3393398179413185,
                1027.0517006575913,
                11.716468721489935,
            ],
            816.5715188748555,
        ),
        "Tukey-Hanning": (
            [
                6.447865541731161,
                1.6839113575399551,
                979.3421648399217,
                11.536674670937069,
            ],
            783.8118849372423,
        ),
    },
    "multi_zero_weight": {
        "Bartlett": ([3.250622006727823, 95.19136944807488], 87.64006240258722),
        "Parzen": ([3.518138733278852, 114.70026037174542], 104.8765004656144),
        "Quadratic Spectral": (
            [3.5185186137498374, 104.66368039348463],
            96.70104233813232,
        ),
        "Tukey-Hanning": ([3.545362498970807, 106.29207514959447], 97.7832838759039),
    },
}

# R 4.6.1, sandwich 3.1.3: bw = bwNeweyWest(xc, kernel=<name>, weights=w,
# prewhite=FALSE), where xc is the data minus its mean and w the weights in CASES
# (1s if None), and the diagonal and the sum of
#   nobs * kernHAC(lm(x ~ 1), bw=bw, kernel=<name>, prewhite=FALSE, adjust=FALSE)
R_NEWEY_WEST: dict[str, dict[str, tuple[float, list[float], float]]] = {
    "uni_short": {
        "Bartlett": (2.390258945185522, [3.238241429125315], 3.238241429125315),
        "Parzen": (3.4259799723855435, [3.3500520639358933], 3.3500520639358933),
        "Quadratic Spectral": (
            1.701919336248188,
            [3.159268940461116],
            3.159268940461116,
        ),
    },
    "uni_long": {
        "Bartlett": (14.622218886182115, [15.069663309872407], 15.069663309872407),
        "Parzen": (19.53402804450467, [15.770538184098939], 15.770538184098939),
        "Quadratic Spectral": (
            8.55039379321342,
            [14.480455952681691],
            14.480455952681691,
        ),
    },
    "uni_mean": {
        "Bartlett": (2.1225090945099767, [1.2124700432270492], 1.2124700432270492),
        "Parzen": (5.760818135062306, [1.4082841797991594], 1.4082841797991594),
        "Quadratic Spectral": (
            2.8617936636228585,
            [1.4172649358554488],
            1.4172649358554488,
        ),
    },
    "multi_short": {
        "Bartlett": (
            4.6406724808262005,
            [1.241696360889315, 16.258418642253638, 5.661411058734253],
            23.415267478242068,
        ),
        "Parzen": (
            9.070180548822714,
            [1.2930539944035742, 18.94939874830143, 4.722285262819893],
            25.811812531615832,
        ),
        "Quadratic Spectral": (
            4.505781056435903,
            [1.3803645270343394, 19.031259183738403, 5.07279016948057],
            25.713186037845755,
        ),
    },
    "multi_long": {
        "Bartlett": (
            6.8136567530857155,
            [
                5.983181431431297,
                6.661834178251041,
                324.4355785216234,
                14.24473483186312,
            ],
            337.68768330943936,
        ),
        "Parzen": (
            11.38641125519595,
            [
                6.77384495214792,
                6.995616253080248,
                392.13535447505797,
                13.448582494615174,
            ],
            398.2564068243523,
        ),
        "Quadratic Spectral": (
            5.656411783457791,
            [
                6.470501820708087,
                7.21846922650694,
                344.68237268337634,
                14.49557285409189,
            ],
            361.4288815224729,
        ),
    },
    "multi_zero_weight": {
        "Bartlett": (
            7.034096694193144,
            [3.240416536783542, 92.42322737965894],
            85.23473548524504,
        ),
        "Parzen": (
            11.058025518808522,
            [3.4750481152844825, 105.58736392554779],
            97.02125334439084,
        ),
        "Quadratic Spectral": (
            5.493280055014935,
            [3.539991249885647, 95.14786161203646],
            88.27943243821,
        ),
    },
}

# R 4.6.1, sandwich 3.1.3: lag = floor(bw) for the Bartlett bw in R_NEWEY_WEST, and
# the diagonal and the sum of
#   nobs * NeweyWest(lm(x ~ 1), lag=lag, prewhite=FALSE, adjust=FALSE)
# which is the estimator of Newey and West (1994, eq. 2-2) with m = lag.
R_NEWEY_WEST_LAG: dict[str, tuple[int, list[float], float]] = {
    "uni_short": (2, [3.5354757660896334], 3.5354757660896334),
    "uni_long": (14, [15.179731255578883], 15.179731255578883),
    "uni_mean": (2, [1.318445379967149], 1.318445379967149),
    "multi_short": (
        4,
        [1.2659691907274546, 16.680874019399916, 5.6130253545876645],
        23.82843360268973,
    ),
    "multi_long": (
        6,
        [6.036805072951339, 6.672202250294113, 330.89147121276596, 14.104265609518741],
        343.3206046123978,
    ),
    "multi_zero_weight": (
        7,
        [3.2722841304947967, 101.06702597520932],
        92.74560772955134,
    ),
}


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
    optimal = R_BANDWIDTHS[case][kern]
    assert_allclose(est.opt_bandwidth, optimal, rtol=1e-10)
    # A kernel that is 0 beyond 1 weights lag h by k(h / (H + 1)), so the bandwidth H
    # that gives the weights k(h / S) of the optimal bandwidth S is S - 1
    bounded = kern != "Quadratic Spectral"
    assert_allclose(est.bandwidth, optimal - 1.0 if bounded else optimal, rtol=1e-10)
    if center:
        # The long-run covariance is that of the Andrews (1991) estimator in R
        diag, total = R_ANDREWS_LONG_RUN[case][kern]
        long_run = np.asarray(est.cov.long_run)
        assert_allclose(np.diag(long_run), diag, rtol=1e-10)
        assert_allclose(long_run.sum(), total, rtol=1e-10)


@pytest.mark.parametrize("case", list(R_NEWEY_WEST))
@pytest.mark.parametrize("kern", ["Bartlett", "Parzen", "Quadratic Spectral"])
def test_newey_west_matches_r(case: str, kern: str) -> None:
    x, weights = _case(case)
    est = R_KERNELS[kern](x, weights=weights)
    bandwidth, diag, total = R_NEWEY_WEST[case][kern]
    assert_allclose(est.opt_bandwidth, bandwidth, rtol=1e-10)
    long_run = np.asarray(est.cov.long_run)
    assert_allclose(np.diag(long_run), diag, rtol=1e-10)
    assert_allclose(long_run.sum(), total, rtol=1e-10)


@pytest.mark.parametrize("case", list(R_NEWEY_WEST_LAG))
def test_newey_west_integer_lag_matches_r(case: str) -> None:
    # With force_int, the number of lags is the integer part of the optimal bandwidth,
    # m = [gamma T^(1/3)] in Newey and West (1994, eq. 2-2), with weights 1 - h/(m+1)
    x, weights = _case(case)
    est = Bartlett(x, weights=weights, force_int=True)
    lags, diag, total = R_NEWEY_WEST_LAG[case]
    assert est.bandwidth == lags
    assert est.kernel_weights.shape[0] == lags + 1
    long_run = np.asarray(est.cov.long_run)
    assert_allclose(np.diag(long_run), diag, rtol=1e-10)
    assert_allclose(long_run.sum(), total, rtol=1e-10)


@pytest.mark.parametrize("name", KERNELS)
def test_fractional_bandwidth_uses_every_positive_lag(name: str) -> None:
    # A fractional bandwidth H uses every lag smaller than H + 1
    x, _ = _case("uni_long")
    est = get_kernel_estimator(name)(x, bandwidth=5.4)
    weights = est.kernel_weights
    if name in ("QuadraticSpectral", "Andrews"):
        assert weights.shape[0] == x.shape[0]
        expected = np.array([_qs(h / 5.4) for h in range(weights.shape[0])])
    elif name == "ZeroLag":
        assert_allclose(weights, [1.0])
        return
    else:
        assert weights.shape[0] == 7
        func = KERNEL_THEORY[_BASE_KERNEL.get(name, name)][0]
        expected = np.array([func(h / 6.4) for h in range(7)])
    assert_allclose(weights, expected, rtol=1e-12, atol=1e-15)
    # An integer bandwidth uses the lags 0, ..., H
    if name not in ("QuadraticSpectral", "Andrews"):
        assert get_kernel_estimator(name)(x, bandwidth=5.0).kernel_weights.shape[0] == 6


def test_fractional_bandwidth_matches_r() -> None:
    # R 4.6.1, sandwich 3.1.3, for the data of "uni_long":
    #   nobs * NeweyWest(lm(x ~ 1), lag=5.4, prewhite=FALSE, adjust=FALSE)
    #   nobs * kernHAC(lm(x ~ 1), bw=6.4, kernel="Parzen", prewhite=FALSE, adjust=FALSE)
    x, _ = _case("uni_long")
    bartlett = Bartlett(x, bandwidth=5.4).cov.long_run.item()
    assert_allclose(bartlett, 10.350529373033602, rtol=1e-10)
    parzen = Parzen(x, bandwidth=5.4).cov.long_run.item()
    assert_allclose(parzen, 8.8512947144139691, rtol=1e-10)


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
# The kernels that the aliases of KERNELS use
_BASE_KERNEL = {
    "Andrews": "QuadraticSpectral",
    "Gallant": "Parzen",
    "NeweyWest": "Bartlett",
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
    est = Bartlett(x, bandwidth_method="andrews")
    assert est.opt_bandwidth == 29.0
    # The weights of the bandwidth 28 are k(h / 29)
    assert est.bandwidth == 28.0
    assert Bartlett(x, bandwidth_method="andrews", force_int=True).bandwidth == 28.0


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


def test_all_constant_newey_west_raises() -> None:
    # The default method raised a ZeroDivisionError for constant data
    match = r'bandwidth_method="newey-west" cannot estimate the bandwidth'
    with pytest.raises(ValueError, match=match):
        _opt(Bartlett(np.ones((50, 1))))
    with pytest.raises(ValueError, match=match):
        _opt(Bartlett(np.ones((50, 2)), bandwidth_method="newey-west"))
    with pytest.raises(ValueError, match=match):
        _ = PreWhitenedRecolored(np.ones((30, 1))).bandwidth
    # An explicit bandwidth does not need the estimate
    assert_allclose(Bartlett(np.ones((50, 1)), bandwidth=3).cov.long_run, 0.0)


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


def _kernel_long_run(
    x: np.ndarray, func: Callable[[float], float], bandwidth: float, bounded: bool
) -> np.ndarray:
    """
    Gamma_0 + sum_h k(h / S) (Gamma_h + Gamma_h'), over h / S < 1 if k is 0 beyond 1
    """
    u = x - x.mean(0)
    nobs = u.shape[0]
    total = u.T @ u / nobs
    for h in range(1, nobs):
        z = h / bandwidth
        if bounded and z >= 1:
            break
        gamma = u[h:].T @ u[:-h] / nobs
        total += func(z) * (gamma + gamma.T)
    return total


@pytest.mark.parametrize("method", ["newey-west", "andrews"])
@pytest.mark.parametrize("which", ["A", "B"])
@pytest.mark.parametrize("name", [name for name in KERNELS if name != "ZeroLag"])
def test_automatic_weights_are_those_of_opt_bandwidth(
    name: str, which: str, method: str
) -> None:
    # The automatic estimate is the estimator of Andrews (1991) and Newey and West
    # (1994) with the weights k(h / S) of the optimal bandwidth S. The kernels that are
    # 0 beyond 1 use the bandwidth S - 1, since they weight lag h by k(h / (H + 1))
    x = _baseline_data(200, 3, 2025) if which == "A" else _baseline_data(40, 2, 2026)
    est = get_kernel_estimator(name)(x, bandwidth_method=method)
    base = _BASE_KERNEL.get(name, name)
    bounded = base != "QuadraticSpectral"
    optimal = est.opt_bandwidth
    assert_allclose(est.bandwidth, max(optimal - 1.0, 0.0) if bounded else optimal)
    expected = _kernel_long_run(x, KERNEL_THEORY[base][0], optimal, bounded)
    assert_allclose(est.cov.long_run, expected, rtol=1e-10, atol=1e-12)


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
    assert abs(default.opt_bandwidth - pw.opt_bandwidth) > 0.05
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
    # The largest differences seen are 2.2e-15 (Windows) and, for powers of 2, which
    # are exact, 5e-16 (macOS Intel), where the sums round differently in the last bits
    scales = (1e-200, 1e-150, 1e-100, 1e100, 1e150, 1e200, 2.0**-600, 2.0**600)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for scale in scales:
            est = cls(scale * x, weights=weights, bandwidth_method=method)
            assert_allclose(est.opt_bandwidth, expected, rtol=1e-13)


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
