"""
Tests of the NGARCH volatility process of Engle and Ng (1993).

The recursion is checked against an independent NumPy implementation and against the
values of the fGARCH model with the NAGARCH submodel of the rugarch package. The
variance forecasts are checked against closed forms, an independent forecast
recursion and the mean of simulated paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
import types
import warnings

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.univariate import (
    GARCH,
    NGARCH,
    ARCHInMean,
    ConstantMean,
    Normal,
    SkewStudent,
    StudentsT,
    ZeroMean,
    arch_model,
)
import arch.univariate.recursions_python as recpy
from arch.utility.exceptions import InitialValueWarning, ValueWarning

try:
    import arch.univariate.recursions as rec_cython

    MISSING_EXTENSION = False
except ImportError:
    MISSING_EXTENSION = True

pytestmark = pytest.mark.filterwarnings("ignore::arch.compat.numba.PerformanceWarning")

MODULES = [pytest.param(recpy, id="python")]
if not MISSING_EXTENSION:
    MODULES.append(pytest.param(rec_cython, id="compiled"))

# (p, o, q)
ORDERS = [
    pytest.param((1, 1, 1), id="111"),
    pytest.param((1, 0, 1), id="101"),
    pytest.param((1, 1, 0), id="110"),
    pytest.param((2, 1, 1), id="211"),
    pytest.param((2, 2, 2), id="222"),
    pytest.param((3, 1, 2), id="312"),
    pytest.param((3, 3, 1), id="331"),
]


@dataclass
class RugarchCase:
    """
    The output of rugarch for NAGARCH with fixed parameters on RUGARCH_Y.

    ``params`` are the parameters of the volatility process in the order of arch
    (omega, alpha, theta, beta). rugarch calls theta ``eta21``. ``init`` is the
    initial variance of rugarch, the mean of the squared residuals.
    """

    order: tuple[int, int, int]
    mu: float
    params: list[float]
    dist: str
    shape: float | None
    init: float
    loglikelihood: float
    sigma2: np.ndarray
    forecast: np.ndarray


# Reference values: rugarch 1.5-6 (CRAN) on R 4.6.1, using
#   spec <- ugarchspec(
#       variance.model = list(model = "fGARCH", submodel = "NAGARCH",
#                             garchOrder = c(p, q)),
#       mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
#       distribution.model = dist, fixed.pars = <the parameters>)
#   sigma(ugarchfilter(spec, data = RUGARCH_Y))^2
#   likelihood(ugarchfilter(spec, data = RUGARCH_Y))
#   sigma(ugarchforecast(spec, data = RUGARCH_Y, n.ahead = 10))^2
# with dist "norm" or "std" (a unit variance Student's t) and the data below, which
# are the first 60 observations of a seeded NGARCH(1,1,1) simulation.
# rugarch sets the first max(p, q) variances to init, and its fGARCH filter adds 1e-6
# to (z - eta21)**2, which is the same as adding alpha * 1e-6 to beta in the
# recursion. Its forecasts do not include the 1e-6.

# fmt: off
RUGARCH_Y = np.array([
    2.677519, -2.379285, 1.739935,
    6.57373, -0.252421, 1.408194,
    0.7075, -0.070338, 2.094999,
    1.450817, -2.107544, -1.271095,
    3.076108, 1.617279, 1.03541,
    0.513262, -3.825167, -0.71068,
    0.977933, 2.488432, -2.4809,
    -0.320755, -0.236325, 2.187222,
    0.362754, -0.479313, -0.258346,
    1.049698, -1.046813, -3.490709,
    -3.428229, -0.801431, -4.544892,
    -0.503777, -0.908317, -4.126741,
    -2.309132, -3.326128, -2.291956,
    -4.308347, 1.278016, 2.149189,
    -0.407156, 0.788726, 0.574681,
    -4.486222, 1.575152, 2.682013,
    -1.040923, 1.233603, 0.191771,
    -1.006626, 0.408802, -3.71525,
    -2.55803, -1.182729, -3.287748,
    0.055482, -0.479473, 1.802926,
])
RUGARCH_CASES = {
    "s1": RugarchCase(
        order=(1, 1, 1),
        mu=0.02,
        params=[0.05, 0.08, 0.6, 0.88],
        dist="norm",
        shape=None,
        init=5.0840975000813495,
        loglikelihood=-135.93320823976583,
        sigma2=np.array([
            5.08409750008135, 4.66017354974831, 5.24291790468773,
            4.67335043131802, 6.37314165370248, 5.91387068948674,
            5.25460897296297, 4.71191019144281, 4.35166246161945,
            3.93369713899769, 3.51629272184347, 3.9907168704976,
            4.05772033835052, 3.89384909417268, 3.49025344082637,
            3.1223141188053, 2.82335055684899, 4.41893890752732,
            4.25609779708587, 3.80163328364689, 3.53033935090181,
            4.20983582313507, 3.95230733892611, 3.69603355073396,
            3.38471971280987, 3.0748957750997, 2.9484647823432,
            2.7816464016073, 2.497916344716, 2.57301710443584,
            3.91497850696549, 5.21414144957307, 5.02265891039633,
            7.26378104645799, 6.80879082209498, 6.53931404475112,
            8.38655770994659, 8.75321877850126, 9.85103257927477,
            10.126843600832, 12.0743443817882, 10.7301215967967,
            9.49465369974831, 8.81969544059068, 7.89345090369236,
            7.09857684936722, 9.27824585026273, 8.22079588100924,
            7.35524451940537, 7.10071048442039, 6.31049775093426,
            5.74591720172998, 5.59245218622098, 5.05624684634614,
            6.56760050486157, 7.18458800772242, 7.00456396629364,
            8.13145982541306, 7.43025882959764, 6.95328058011377,
        ]),
        forecast=np.array([
            6.1721119694919, 6.15298431543359, 6.13407089110073,
            6.1153692971204, 6.09687716099265, 6.07859213678953,
            6.06051190485749, 6.04263417152309, 6.02495666880203,
            6.00747715411144,
        ]),
    ),
    "s2": RugarchCase(
        order=(1, 1, 1),
        mu=0.0,
        params=[0.1, 0.05, -0.3, 0.9],
        dist="norm",
        shape=None,
        init=5.0697364034146837,
        loglikelihood=-137.44602248993945,
        sigma2=np.array([
            5.06973640341468, 5.2248938872344, 4.94580935699772,
            4.84093766140731, 7.0732328672149, 6.48078547404141,
            6.16856812082543, 5.75721367452301, 5.30258431723063,
            5.26036558942072, 5.06307001987939, 4.75936694301013,
            4.40244123884078, 4.74875899219715, 4.63176183553183,
            4.40988332231283, 4.13424665779675, 4.33769183188235,
            4.00429151592989, 3.82840697167248, 4.01847732409741,
            3.89325877424036, 3.60761016415989, 3.35240998365588,
            3.49159340018119, 3.28506098070813, 3.05676260243714,
            2.85462860534472, 2.79031098890223, 2.62616870536361,
            2.914916576675, 3.14858812824078, 2.93735029832401,
            3.55595503784914, 3.30055154129737, 3.0770956931086,
            3.517563203688, 3.41831614867339, 3.56053626414988,
            3.45341495538166, 3.91151592469721, 3.79546068081327,
            3.88955617045004, 3.60230281079156, 3.43429693531817,
            3.2547843792249, 3.80745394977274, 3.76010387635895,
            4.01669447067183, 3.72469080742153, 3.61649544764733,
            3.38389989561233, 3.15585061790203, 2.98460974722392,
            3.29717955906862, 3.27012776900425, 3.06360950449318,
            3.23886126179102, 3.03269958730736, 2.82952208593685,
        ]),
        forecast=np.array([
            2.91281188112848, 2.88027894053713, 2.8492262487427,
            2.8195864544249, 2.79129527074857, 2.76429133592951,
            2.73851608014472, 2.71391359849813, 2.69043052976647,
            2.66801594066209,
        ]),
    ),
    "s3": RugarchCase(
        order=(1, 1, 1),
        mu=0.01,
        params=[0.05, 0.1, 0.5, 0.8],
        dist="std",
        shape=7.0,
        init=5.0768169517480164,
        loglikelihood=-139.58393916822246,
        sigma2=np.array([
            5.07681695174802, 4.34890039546127, 4.70697345065864,
            3.85720198351442, 6.25134631176059, 5.27986012609031,
            4.2801035127664, 3.48543478405268, 2.94112802244902,
            2.55358156217018, 2.13405886183931, 2.56833771951646,
            2.53830803808074, 2.59571178794169, 2.1908449225749,
            1.81081775469135, 1.5015298220976, 3.22956241985259,
            2.89584056826497, 2.36804338866963, 2.23650632513209,
            2.88808915822204, 2.49982348943347, 2.15736825660963,
            1.98406860797887, 1.64961247874271, 1.49771919265818,
            1.32565998385284, 1.13205878285714, 1.20807703565712,
            2.65693190570775, 3.98454564554006, 3.56506497584529,
            5.92590952602985, 5.0903423490417, 4.54105228422946,
            6.38916015367506, 6.44509716985024, 7.32713016457665,
            7.24789218567027, 9.0569053364117, 7.30112874622956,
            5.95302318069407, 5.08042776527691, 4.12647151650461,
            3.3715181692394, 5.67868659200516, 4.60691144206176,
            3.99115444110869, 3.66309879956682, 2.9875891919844,
            2.48664698149276, 2.36514968044878, 1.9558210947639,
            3.57228007739032, 4.14197913298282, 3.85213589630106,
            4.96277103310353, 4.13436130836019, 3.58433205062352,
        ]),
        forecast=np.array([
            2.98908959829329, 2.81490787842129, 2.65378978753969,
            2.50475555347422, 2.36689888696365, 2.23938147044138,
            2.12142786015827, 2.0123207706464, 1.91139671284792,
            1.81804195938433,
        ]),
    ),
    "s5": RugarchCase(
        order=(1, 1, 2),
        mu=0.02,
        params=[0.5, 0.1, 0.5, 0.25, 0.15],
        dist="norm",
        shape=None,
        init=5.0840975000813495,
        loglikelihood=-163.96855797085291,
        sigma2=np.array([
            5.08409750008135, 5.08409750008135, 3.77738822004075,
            2.262936365188, 4.99817215331453, 2.28226366895073,
            1.86034009123472, 1.30742781570084, 1.14973938992343,
            1.22036106668678, 1.05472176489261, 1.64424511739675,
            1.44262315302684, 1.71027105507784, 1.23296031851909,
            1.08596076255567, 0.956511548778817, 2.78052855992513,
            1.5833519739134, 1.32372682969065, 1.42684214380554,
            2.0151245877725, 1.32816908637678, 1.20362605723947,
            1.26214207555179, 1.00087440474961, 1.03944634523619,
            0.972104936127193, 0.927750235846816, 1.11751135216431,
            2.55011154467394, 3.10858601347663, 1.94968108492803,
            4.2236746405364, 2.08904179800558, 1.92838960848573,
            3.63905297560279, 2.77679621484445, 3.4867239005782,
            2.84158916584855, 4.40753471919001, 2.03246178132542,
            1.86985535630996, 1.39573634733409, 1.13258162657219,
            0.992556898536103, 3.42238621607635, 1.54419152325765,
            1.81584569254081, 1.48650513194741, 1.18048384861434,
            1.03189654206684, 1.17052736308739, 0.949731479420709,
            2.69598029227712, 2.47177654417197, 1.91788321047471,
            2.95038652141677, 1.59307033436716, 1.46864173598671,
        ]),
        forecast=np.array([
            1.24465114208376, 1.18704043867941, 1.13183783581734,
            1.10249525423342, 1.08321139571013, 1.07157856152631,
            1.06432366992889, 1.05985816045228, 1.05709536065894,
            1.05538948431494,
        ]),
    ),
}
# fmt: on


NO_SHIFT = "NGARCH with o=0 has no shift, and so is the same model as GARCH"


def make_ngarch(p, o, q):
    """NGARCH(p, o, q), checking the warning that a model with o = 0 is a GARCH."""
    if o == 0:
        with pytest.warns(ValueWarning, match=NO_SHIFT):
            return NGARCH(p, o, q)
    return NGARCH(p, o, q)


def make_parameters(order, seed=0, theta_scale=0.6):
    """Parameters with a persistence of about 0.8, omega, alpha, theta and beta."""
    p, o, q = order
    rs = np.random.default_rng(seed)
    omega = 0.1 + 0.1 * rs.uniform()
    alpha = 0.08 + 0.04 * rs.uniform(size=p)
    alpha = 0.3 * alpha / alpha.sum() if p > 1 else alpha
    theta = theta_scale * rs.uniform(-1.0, 1.0, size=o)
    beta = 0.3 * rs.uniform(0.5, 1.0, size=q)
    beta = 0.3 * beta / beta.sum() if q > 0 else beta
    return np.r_[omega, alpha, theta, beta]


def reference_variance(params, order, resids, backcast):
    """
    NGARCH variance in terms of the standardized shocks, with one-based lags.

    sigma2[t] = omega + sum_i alpha_i sigma2[t-i] (z[t-i] - theta_i)**2
                + sum_j beta_j sigma2[t-j]

    where z = resids / sqrt(sigma2). The terms before the start of the sample are
    omega + alpha_i (1 + theta_i**2) backcast and beta_j backcast.
    """
    p, o, q = order
    omega = params[0]
    theta = dict(enumerate(params[1 + p : 1 + p + o], 1))
    sigma2 = {}
    for t in range(resids.shape[0]):
        value = omega
        for i in range(1, p + 1):
            th = theta.get(i, 0.0)
            if t - i >= 0:
                z = resids[t - i] / np.sqrt(sigma2[t - i])
                value += params[i] * sigma2[t - i] * (z - th) ** 2
            else:
                value += params[i] * (1 + th**2) * backcast
        for j in range(1, q + 1):
            value += params[p + o + j] * (sigma2[t - j] if t - j >= 0 else backcast)
        sigma2[t] = value
    return np.array([sigma2[t] for t in range(resids.shape[0])])


def simulate_data(order, params, nobs, seed, rng=None, burn=500):
    process = make_ngarch(*order)
    if rng is None:
        rng = RandomState(seed).standard_normal
    return process.simulate(params, nobs, rng, burn=burn)


def var_inputs(process, resids):
    return process.backcast(resids), process.variance_bounds(resids)


@pytest.fixture(params=MODULES)
def rec(request) -> types.ModuleType:
    return request.param


class TestRecursion:
    @pytest.mark.parametrize("order", ORDERS)
    def test_matches_reference(self, rec, order):
        p, o, q = order
        params = make_parameters(order, seed=sum(order))
        resids = np.random.default_rng(1).standard_normal(120) * 1.3
        process = make_ngarch(*order)
        backcast, var_bounds = var_inputs(process, resids)
        sigma2 = np.empty_like(resids)
        out = rec.ngarch_recursion(
            params, resids, sigma2, p, o, q, resids.shape[0], backcast, var_bounds
        )
        expected = reference_variance(params, order, resids, backcast)
        assert_allclose(sigma2, expected, rtol=1e-10)
        assert_allclose(out, expected, rtol=1e-10)

    @pytest.mark.parametrize("order", ORDERS)
    def test_python_numba_compiled_agree(self, order):
        p, o, q = order
        params = make_parameters(order, seed=7 + sum(order))
        resids = np.random.default_rng(2).standard_normal(150)
        process = make_ngarch(*order)
        backcast, var_bounds = var_inputs(process, resids)
        args = (p, o, q, resids.shape[0], backcast, var_bounds)
        python = np.empty_like(resids)
        recpy.ngarch_recursion_python(params, resids, python, *args)
        jitted = np.empty_like(resids)
        recpy.ngarch_recursion(params, resids, jitted, *args)
        assert_allclose(jitted, python, rtol=1e-12)
        if not MISSING_EXTENSION:
            compiled = np.empty_like(resids)
            rec_cython.ngarch_recursion(params, resids, compiled, *args)
            assert_allclose(compiled, python, rtol=1e-12)

    def test_first_variance_uses_backcast(self, rec):
        # The first variance is omega + (alpha (1 + theta ** 2) + beta) * backcast
        params = np.array([0.1, 0.2, 0.5, 0.4])
        resids = np.array([1.0, 2.0, 3.0])
        var_bounds = np.tile([1e-8, 1e8], (3, 1))
        sigma2 = np.empty(3)
        rec.ngarch_recursion(params, resids, sigma2, 1, 1, 1, 3, 0.7, var_bounds)
        assert_allclose(sigma2[0], 0.1 + (0.2 * 1.25 + 0.4) * 0.7, rtol=1e-14)
        shock = 1.0 - 0.5 * np.sqrt(sigma2[0])
        assert_allclose(sigma2[1], 0.1 + 0.2 * shock**2 + 0.4 * sigma2[0], rtol=1e-14)

    def test_shift_only_for_first_o_lags(self, rec):
        # p = 2, o = 1: the second lag has no shift
        params = np.array([0.1, 0.2, 0.3, 0.9, 0.0])
        resids = np.array([1.0, -2.0, 0.5, 1.5])
        var_bounds = np.tile([1e-8, 1e8], (4, 1))
        sigma2 = np.empty(4)
        rec.ngarch_recursion(params, resids, sigma2, 2, 1, 0, 4, 0.7, var_bounds)
        assert_allclose(sigma2[0], 0.1 + (0.2 * (1 + 0.81) + 0.3) * 0.7, rtol=1e-14)
        assert_allclose(
            sigma2[1],
            0.1 + 0.2 * (1.0 - 0.9 * np.sqrt(sigma2[0])) ** 2 + 0.3 * 0.7,
            rtol=1e-14,
        )
        assert_allclose(
            sigma2[2],
            0.1 + 0.2 * (-2.0 - 0.9 * np.sqrt(sigma2[1])) ** 2 + 0.3 * resids[0] ** 2,
            rtol=1e-14,
        )

    def test_negative_theta_reverses_the_asymmetry(self, rec):
        # With theta > 0 a negative shock raises the variance more than a positive
        # shock of the same size, and the reverse for theta < 0
        var_bounds = np.tile([1e-8, 1e8], (2, 1))
        out = {}
        for theta in (0.8, -0.8):
            params = np.array([0.1, 0.2, theta, 0.5])
            for sign in (1.0, -1.0):
                sigma2 = np.empty(2)
                resids = np.array([sign * 2.0, 0.0])
                rec.ngarch_recursion(
                    params, resids, sigma2, 1, 1, 1, 2, 1.0, var_bounds
                )
                out[theta, sign] = sigma2[1]
        assert out[0.8, -1.0] > out[0.8, 1.0]
        assert out[-0.8, 1.0] > out[-0.8, -1.0]
        assert_allclose(out[0.8, -1.0], out[-0.8, 1.0], rtol=1e-12)

    def test_variance_bounds_are_applied(self):
        params = np.array([1e100, 0.4, 0.3, 0.9])
        resids = np.random.default_rng(3).standard_normal(100)
        process = NGARCH()
        backcast, var_bounds = var_inputs(process, resids)
        results = []
        for module in (recpy,) if MISSING_EXTENSION else (recpy, rec_cython):
            sigma2 = np.empty_like(resids)
            module.ngarch_recursion(
                params, resids, sigma2, 1, 1, 1, 100, backcast, var_bounds
            )
            assert np.all(sigma2 >= var_bounds[:, 0])
            assert np.all(sigma2 <= 1.1 * var_bounds[:, 1])
            results.append(sigma2)
        assert_allclose(results[0], results[-1], rtol=1e-12)

        params = np.array([-1e100, 0.4, 0.3, 0.2])
        for module in (recpy,) if MISSING_EXTENSION else (recpy, rec_cython):
            sigma2 = np.empty_like(resids)
            module.ngarch_recursion(
                params, resids, sigma2, 1, 1, 1, 100, backcast, var_bounds
            )
            assert_allclose(sigma2, var_bounds[:, 0])

    def test_extreme_residuals_compiled_matches_python(self):
        params = np.array([0.1, 0.2, 0.5, 0.7])
        resids = np.random.default_rng(4).standard_normal(100)
        resids[10] = np.finfo(float).max
        resids[50] = np.inf
        var_bounds = np.tile([1e-6, 1e6], (100, 1))
        results = []
        for module in (recpy,) if MISSING_EXTENSION else (recpy, rec_cython):
            sigma2 = np.empty_like(resids)
            with np.errstate(over="ignore"):
                module.ngarch_recursion(
                    params, resids, sigma2, 1, 1, 1, 100, 1.0, var_bounds
                )
            assert np.all(np.isfinite(sigma2))
            assert np.all(sigma2 >= var_bounds[:, 0])
            assert np.all(sigma2 <= 1.1 * var_bounds[:, 1])
            results.append(sigma2)
        assert_allclose(results[0], results[-1], rtol=1e-12)


class TestUpdater:
    @pytest.mark.parametrize("order", ORDERS)
    def test_matches_recursion(self, rec, order):
        p, o, q = order
        params = make_parameters(order, seed=3 + sum(order))
        resids = np.random.default_rng(5).standard_normal(80)
        process = make_ngarch(*order)
        backcast, var_bounds = var_inputs(process, resids)
        expected = np.empty_like(resids)
        recpy.ngarch_recursion_python(
            params, resids, expected, p, o, q, resids.shape[0], backcast, var_bounds
        )
        updater = rec.NGARCHUpdater(p, o, q)
        update = updater.update if rec is recpy else updater._update_tester
        sigma2 = np.full_like(resids, np.nan)
        updater.initialize_update(params, backcast, resids.shape[0])
        for t in range(resids.shape[0]):
            update(t, params, resids, sigma2, var_bounds)
        assert_allclose(sigma2, expected, rtol=1e-12)

    def test_pickle(self, rec):
        order = (2, 1, 2)
        p, o, q = order
        params = make_parameters(order, seed=11)
        resids = np.random.default_rng(6).standard_normal(60)
        process = make_ngarch(*order)
        backcast, var_bounds = var_inputs(process, resids)
        expected = reference_variance(params, order, resids, backcast)
        updater = rec.NGARCHUpdater(p, o, q)
        updater.initialize_update(params, backcast, resids.shape[0])
        sigma2 = np.full_like(resids, np.nan)
        for t in range(resids.shape[0]):
            update = updater.update if rec is recpy else updater._update_tester
            update(t, params, resids, sigma2, var_bounds)
            if t == 30:
                updater = pickle.loads(pickle.dumps(updater))
        assert_allclose(sigma2, expected, rtol=1e-10)
        # The unpickled updater uses the backcast that it was initialized with
        restored = pickle.loads(pickle.dumps(updater))
        sigma2 = np.full_like(resids, np.nan)
        update = restored.update if rec is recpy else restored._update_tester
        update(0, params, resids, sigma2, var_bounds)
        assert_allclose(sigma2[0], expected[0], rtol=1e-12)

    def test_process_updater(self):
        process = NGARCH(2, 1, 1)
        assert process.updateable
        updater = process.volatility_updater
        assert isinstance(updater, type(rec_updater()))


def rec_updater():
    module = recpy if MISSING_EXTENSION else rec_cython
    return module.NGARCHUpdater(1, 1, 1)


class TestProcess:
    @pytest.mark.parametrize("order", ORDERS)
    def test_compute_variance_matches_reference(self, order):
        params = make_parameters(order, seed=2 * sum(order))
        resids = np.random.default_rng(8).standard_normal(100)
        process = make_ngarch(*order)
        backcast, var_bounds = var_inputs(process, resids)
        sigma2 = np.empty_like(resids)
        out = process.compute_variance(params, resids, sigma2, backcast, var_bounds)
        assert out is sigma2
        assert_allclose(
            sigma2, reference_variance(params, order, resids, backcast), rtol=1e-10
        )

    @pytest.mark.parametrize("order", [(1, 0, 1), (2, 0, 1), (2, 0, 0), (3, 0, 2)])
    def test_no_shift_is_garch(self, order):
        # With o = 0 the process is the GARCH with the same parameters
        p, _, q = order
        params = make_parameters(order, seed=5)
        resids = np.random.default_rng(9).standard_normal(200) * 2
        ngarch, garch = make_ngarch(*order), GARCH(p, 0, q)
        backcast, var_bounds = var_inputs(ngarch, resids)
        assert_allclose(garch.backcast(resids), backcast, rtol=1e-14)
        assert_allclose(garch.variance_bounds(resids), var_bounds, rtol=1e-14)
        expected = np.empty_like(resids)
        garch.compute_variance(params, resids, expected, backcast, var_bounds)
        sigma2 = np.empty_like(resids)
        ngarch.compute_variance(params, resids, sigma2, backcast, var_bounds)
        assert_allclose(sigma2, expected, rtol=1e-12)

    @pytest.mark.parametrize("order", [(1, 1, 1), (2, 1, 1), (2, 2, 2)])
    def test_zero_theta_is_garch(self, order):
        p, o, q = order
        params = make_parameters(order, seed=6)
        params[1 + p : 1 + p + o] = 0.0
        resids = np.random.default_rng(10).standard_normal(200) * 2
        ngarch, garch = make_ngarch(*order), GARCH(p, 0, q)
        backcast, var_bounds = var_inputs(ngarch, resids)
        expected = np.empty_like(resids)
        garch.compute_variance(
            np.r_[params[: 1 + p], params[1 + p + o :]],
            resids,
            expected,
            backcast,
            var_bounds,
        )
        sigma2 = np.empty_like(resids)
        ngarch.compute_variance(params, resids, sigma2, backcast, var_bounds)
        assert_allclose(sigma2, expected, rtol=1e-12)

    def test_fit_no_shift_is_garch(self):
        y, _ = simulate_data((1, 0, 1), [0.1, 0.08, 0.88], 2000, 12)
        garch = ConstantMean(y, volatility=GARCH(1, 0, 1)).fit(disp="off")
        ngarch = ConstantMean(y, volatility=make_ngarch(1, 0, 1)).fit(disp="off")
        assert_allclose(ngarch.loglikelihood, garch.loglikelihood, rtol=1e-8)
        assert_allclose(ngarch.params.to_numpy(), garch.params.to_numpy(), rtol=1e-3)
        assert list(ngarch.params.index) == list(garch.params.index)

    @pytest.mark.parametrize(
        ("order", "names"),
        [
            ((1, 1, 1), ["omega", "alpha[1]", "theta[1]", "beta[1]"]),
            ((1, 0, 1), ["omega", "alpha[1]", "beta[1]"]),
            ((2, 1, 0), ["omega", "alpha[1]", "alpha[2]", "theta[1]"]),
            (
                (2, 2, 2),
                [
                    "omega",
                    "alpha[1]",
                    "alpha[2]",
                    "theta[1]",
                    "theta[2]",
                    "beta[1]",
                    "beta[2]",
                ],
            ),
        ],
    )
    def test_names_and_counts(self, order, names):
        process = make_ngarch(*order)
        assert process.parameter_names() == names
        assert process.num_params == len(names)
        assert process.name == "NGARCH"
        assert process.updateable
        assert (process.p, process.o, process.q) == order

    def test_defaults_and_str(self):
        process = NGARCH()
        assert (process.p, process.o, process.q) == (1, 1, 1)
        assert str(process) == "NGARCH(p: 1, o: 1, q: 1)"
        assert str(make_ngarch(2, 0, 0)) == "NGARCH(p: 2)"
        assert str(make_ngarch(1, 0, 1)) == "NGARCH(p: 1, q: 1)"
        assert str(NGARCH(3, 2, 1)) == "NGARCH(p: 3, o: 2, q: 1)"
        assert repr(process).startswith("NGARCH(p: 1, o: 1, q: 1), id: 0x")

    @pytest.mark.parametrize("order", ORDERS)
    def test_bounds(self, order):
        p, o, q = order
        resids = np.random.default_rng(13).standard_normal(300) * 3
        process = make_ngarch(*order)
        bounds = process.bounds(resids)
        assert len(bounds) == process.num_params
        v = float(np.mean(resids**2))
        assert bounds[0] == pytest.approx((1e-8 * v, 10 * v))
        assert bounds[1 : 1 + p] == [(0.0, 1.0)] * p
        assert bounds[1 + p : 1 + p + o] == [(-5.0, 5.0)] * o
        assert bounds[1 + p + o :] == [(0.0, 1.0)] * q

    @pytest.mark.parametrize("order", ORDERS)
    def test_constraints(self, order):
        p, o, q = order
        process = make_ngarch(*order)
        a, b = process.constraints()
        assert a.shape[1] == process.num_params
        assert a.shape[0] == b.shape[0]
        rs = np.random.default_rng(14)

        def feasible(x):
            alpha, beta = x[1 : 1 + p], x[1 + p + o :]
            return (
                x[0] >= 0
                and np.all(alpha >= 0)
                and np.all(beta >= 0)
                and (alpha.sum() + beta.sum() <= 1)
            )

        counts = {True: 0, False: 0}
        for _ in range(400):
            x = rs.uniform(-0.3, 0.8, size=process.num_params)
            x[1 + p : 1 + p + o] = rs.uniform(-3, 3, size=o)
            ok = bool(np.all(a @ x - b >= 0))
            assert ok == feasible(x)
            counts[ok] += 1
        assert counts[True] > 0
        assert counts[False] > 0
        # Theta is only restricted by its bounds, not by the constraints
        x = np.r_[
            0.1, np.full(p, 0.5 / p), np.full(o, 4.0), np.full(q, 0.4 / max(q, 1))
        ]
        assert np.all(a @ x - b > 0)

    @pytest.mark.parametrize("order", ORDERS)
    def test_starting_values(self, order):
        p, o, q = order
        true = make_parameters(order, seed=15, theta_scale=0.9)
        resids, _ = simulate_data(order, true, 1500, 16)
        process = make_ngarch(*order)
        sv = process.starting_values(resids)
        assert sv.shape == (process.num_params,)
        assert np.all(np.isfinite(sv))
        a, b = process.constraints()
        assert np.all(a @ sv - b > 0)
        for value, (lower, upper) in zip(sv, process.bounds(resids), strict=True):
            assert lower <= value <= upper
        # The grid contains a symmetric start, so the Gaussian log-likelihood of the
        # starting values is at least that of the symmetric start
        backcast, var_bounds = var_inputs(process, resids)
        total = 0.9
        alpha = 0.05
        symmetric = np.zeros(process.num_params)
        symmetric[0] = (1 - total) * np.mean(resids**2)
        symmetric[1 : 1 + p] = alpha / p
        if q > 0:
            symmetric[1 + p + o :] = (total - alpha) / q
        else:
            symmetric[0] = (1 - alpha) * np.mean(resids**2)
        ll_sv = process._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
        ll_sym = process._gaussian_loglikelihood(
            symmetric, resids, backcast, var_bounds
        )
        assert ll_sv >= ll_sym - 1e-8

    @pytest.mark.parametrize(
        "order",
        [(1, 1, 1), (2, 1, 1), (3, 1, 2), (3, 2, 1), (2, 1, 0)],
        ids=["111", "211", "312", "321", "210"],
    )
    @pytest.mark.parametrize("seed", [16, 26])
    def test_starting_values_persistence(self, order, seed):
        # The starting values are chosen from a grid of persistences, computed with
        # the shift on the first o of the p lags only
        p, o, q = order
        true = make_parameters(order, seed=15, theta_scale=0.9)
        resids, _ = simulate_data(order, true, 1500, seed)
        sv = make_ngarch(*order).starting_values(resids)
        alpha = sv[1 : 1 + p]
        theta = np.r_[sv[1 + p : 1 + p + o], np.zeros(p - o)]
        beta = sv[1 + p + o :]
        persistence = np.sum(alpha * (1 + theta**2)) + np.sum(beta)
        if q > 0:
            grid = np.array([0.8, 0.9, 0.95, 0.975, 0.99])
            assert np.min(np.abs(grid - persistence)) < 1e-12
        assert_allclose(sv[0], (1 - persistence) * np.mean(resids**2), rtol=1e-12)

    def test_variance_bounds_and_backcast_use_the_variance(self):
        resids = np.random.default_rng(17).standard_t(5, size=400)
        process = NGARCH()
        garch = GARCH()
        assert_allclose(process.variance_bounds(resids), garch.variance_bounds(resids))
        assert_allclose(
            process.variance_bounds(resids, 1.0), garch.variance_bounds(resids)
        )
        assert_allclose(process.backcast(resids), garch.backcast(resids))
        assert process.backcast_transform(0.5) == 0.5
        assert_allclose(process.backcast_transform([0.5, 0.25]), [0.5, 0.25])
        with pytest.raises(ValueError, match="User backcast value must be strictly"):
            process.backcast_transform(-0.5)

    def test_check_forecasting_method(self):
        process = NGARCH()
        for method in ("analytic", "simulation", "bootstrap"):
            process._check_forecasting_method(method, 1)
            process._check_forecasting_method(method, 10)


class TestSimulate:
    @pytest.mark.parametrize("order", ORDERS)
    def test_follows_the_recursion(self, order):
        p, o, q = order
        params = make_parameters(order, seed=21)
        process = make_ngarch(*order)
        data, sigma2 = process.simulate(
            params, 300, RandomState(0).standard_normal, burn=0, initial_value=1.5
        )
        assert data.shape == sigma2.shape == (300,)
        m = max(p, q)
        assert_allclose(sigma2[:m], 1.5)
        omega = params[0]
        theta = np.r_[params[1 + p : 1 + p + o], np.zeros(p - o)]
        for t in range(m, 300):
            expected = omega
            for i in range(1, p + 1):
                shock = data[t - i] - theta[i - 1] * np.sqrt(sigma2[t - i])
                expected += params[i] * shock**2
            for j in range(1, q + 1):
                expected += params[p + o + j] * sigma2[t - j]
            assert_allclose(sigma2[t], expected, rtol=1e-12)
        # data = sqrt(sigma2) * z with the draws of the rng
        z = RandomState(0).standard_normal(300)
        assert_allclose(data, np.sqrt(sigma2) * z, rtol=1e-12)

    def test_burn_discards_initial_values(self):
        params = make_parameters((1, 1, 1), seed=22)
        process = NGARCH()
        full, full_s2 = process.simulate(params, 100, RandomState(1).standard_normal, 0)
        burned, burned_s2 = process.simulate(
            params, 80, RandomState(1).standard_normal, burn=20
        )
        assert burned.shape == (80,)
        assert_allclose(burned, full[20:], rtol=1e-12)
        assert_allclose(burned_s2, full_s2[20:], rtol=1e-12)

    def test_default_initial_value_is_unconditional_variance(self):
        params = np.array([0.2, 0.1, 0.8, 0.5])
        phi = 0.1 * (1 + 0.8**2) + 0.5
        _, sigma2 = NGARCH().simulate(params, 5, RandomState(2).standard_normal, 0)
        assert_allclose(sigma2[0], 0.2 / (1 - phi), rtol=1e-12)

    def test_unconditional_variance(self):
        # omega / (1 - persistence) with the persistence alpha (1 + theta ** 2) + beta
        # The linear sum alpha + beta is 0.55, the persistence 0.775
        params = np.array([0.3, 0.15, 0.9, 0.4])
        phi = 0.15 * (1 + 0.81) + 0.4
        data, sigma2 = NGARCH().simulate(
            params, 100000, RandomState(3).standard_normal, burn=1000
        )
        target = 0.3 / (1 - phi)
        assert_allclose(sigma2.mean(), target, rtol=0.05)
        assert_allclose(data.var(), target, rtol=0.05)
        # Leverage: negative returns are followed by larger variances for theta > 0
        lag = data[:-1]
        assert sigma2[1:][lag < 0].mean() > sigma2[1:][lag > 0].mean()
        corr = np.corrcoef(data[:-1], sigma2[1:])[0, 1]
        assert corr < -0.02

    def test_nonstationary_default_initial_value_warns(self):
        # Linear persistence is 0.9 but alpha (1 + theta ** 2) + beta > 1
        params = np.array([0.1, 0.3, 1.5, 0.6])
        with pytest.warns(InitialValueWarning):
            _, sigma2 = NGARCH().simulate(
                params, 5, RandomState(4).standard_normal, burn=0
            )
        assert_allclose(sigma2[0], 0.1)

    def test_initial_value_array(self):
        process = NGARCH(2, 1, 3)
        params = make_parameters((2, 1, 3), seed=23)
        _, sigma2 = process.simulate(
            params, 10, RandomState(5).standard_normal, 0, initial_value=[1.0, 2.0, 3.0]
        )
        assert_allclose(sigma2[:3], [1.0, 2.0, 3.0])


class TestForecast:
    @staticmethod
    def setup_forecast(order, nobs=60, seed=31, theta_scale=0.7):
        params = make_parameters(order, seed=seed, theta_scale=theta_scale)
        resids, _ = simulate_data(order, params, nobs, seed)
        process = make_ngarch(*order)
        backcast, var_bounds = var_inputs(process, resids)
        return process, params, resids, backcast, var_bounds

    @staticmethod
    def analytic_reference(params, order, resids, backcast, origin, horizon):
        """
        An independent forecast recursion for one origin.

        Realized shocks are used where they are available. A shock that is not
        realized has the expected value (1 + theta ** 2) * variance.
        """
        p, o, q = order
        sigma2 = reference_variance(params, order, resids, backcast)
        theta = dict(enumerate(params[1 + p : 1 + p + o], 1))
        variance = dict(enumerate(sigma2[: origin + 1]))

        def lagged(idx):
            return backcast if idx < 0 else variance[idx]

        out = []
        for target in range(origin + 1, origin + 1 + horizon):
            value = params[0]
            for i in range(1, p + 1):
                th, idx = theta.get(i, 0.0), target - i
                if idx < 0:
                    value += params[i] * (1 + th**2) * backcast
                elif idx <= origin:
                    shock = resids[idx] - th * np.sqrt(sigma2[idx])
                    value += params[i] * shock**2
                else:
                    value += params[i] * (1 + th**2) * variance[idx]
            for j in range(1, q + 1):
                value += params[p + o + j] * lagged(target - j)
            variance[target] = value
            out.append(value)
        return np.array(out)

    @pytest.mark.parametrize("order", ORDERS)
    def test_analytic_matches_independent_recursion(self, order):
        process, params, resids, backcast, var_bounds = self.setup_forecast(order)
        horizon = 8
        # Includes origins at the start of the sample, where lags use the backcast
        for start in (0, 1, 2, 3, 30, resids.shape[0] - 1):
            fcast = process.forecast(
                params, resids, backcast, var_bounds, start=start, horizon=horizon
            )
            assert fcast.forecasts.shape == (resids.shape[0] - start, horizon)
            for k, origin in enumerate(range(start, resids.shape[0])):
                expected = self.analytic_reference(
                    params, order, resids, backcast, origin, horizon
                )
                assert_allclose(fcast.forecasts[k], expected, rtol=1e-10)
            assert fcast.forecast_paths is None

    @pytest.mark.parametrize("theta", [-0.9, 0.0, 0.5, 1.4])
    def test_closed_form_for_the_ngarch_11(self, theta):
        # f_h = omega + phi f_{h-1}, phi = alpha (1 + theta ** 2) + beta
        omega, alpha, beta = 0.1, 0.07, 0.8
        params = np.array([omega, alpha, theta, beta])
        phi = alpha * (1 + theta**2) + beta
        process, _, resids, backcast, var_bounds = self.setup_forecast((1, 1, 1))
        fcast = process.forecast(params, resids, backcast, var_bounds, horizon=25)
        sigma2 = np.empty_like(resids)
        process.compute_variance(params, resids, sigma2, backcast, var_bounds)
        f1 = omega + alpha * (resids[-1] - theta * np.sqrt(sigma2[-1])) ** 2
        f1 += beta * sigma2[-1]
        h = np.arange(25)
        # f_{h+1} - mean = phi ** h (f_1 - mean), where mean = omega / (1 - phi)
        mean = omega / (1 - phi)
        expected = mean + phi**h * (f1 - mean)
        assert_allclose(fcast.forecasts[0], expected, rtol=1e-12)
        assert fcast.forecasts.shape == (1, 25)

    def test_analytic_converges_to_unconditional_variance(self):
        omega, alpha, theta, beta = 0.1, 0.07, 0.9, 0.8
        phi = alpha * (1 + theta**2) + beta
        process, _, resids, backcast, var_bounds = self.setup_forecast((1, 1, 1))
        params = np.array([omega, alpha, theta, beta])
        fcast = process.forecast(params, resids, backcast, var_bounds, horizon=1000)
        assert_allclose(fcast.forecasts[0, -1], omega / (1 - phi), rtol=1e-8)

    def test_horizon_one_is_the_next_variance(self):
        order = (2, 1, 2)
        process, params, resids, backcast, var_bounds = self.setup_forecast(order)
        # The variance of the period after the sample, from the recursion with an
        # additional (zero) residual
        padded = np.r_[resids, 0.0]
        sigma2 = np.empty_like(padded)
        process.compute_variance(
            params, padded, sigma2, backcast, np.vstack([var_bounds, var_bounds[-1]])
        )
        fcast = process.forecast(params, resids, backcast, var_bounds, horizon=3)
        assert_allclose(fcast.forecasts[0, 0], sigma2[-1], rtol=1e-12)
        # A horizon of 1 from every origin, including the first observations
        one = process.forecast(params, resids, backcast, var_bounds, start=0, horizon=1)
        assert one.forecasts.shape == (resids.shape[0], 1)
        assert_allclose(one.forecasts[-1, 0], sigma2[-1], rtol=1e-12)
        for origin in (0, 1, 2, 30):
            expected = self.analytic_reference(
                params, order, resids, backcast, origin, 1
            )
            assert_allclose(one.forecasts[origin], expected, rtol=1e-10)

    @pytest.mark.parametrize("dist", ["normal", "skew-t"])
    @pytest.mark.parametrize(
        "order",
        [(1, 1, 1), (2, 1, 2), (3, 2, 1), (1, 0, 2)],
        ids=["111", "212", "321", "102"],
    )
    def test_simulation_matches_analytic(self, order, dist):
        process, params, resids, backcast, var_bounds = self.setup_forecast(
            order, nobs=40, seed=33, theta_scale=0.9
        )
        if dist == "normal":
            rng = RandomState(100).standard_normal
        else:
            rng = SkewStudent(seed=RandomState(100)).simulate([6.0, -0.4])
        horizon, sims, start = 6, 20000, 36
        analytic = process.forecast(
            params, resids, backcast, var_bounds, start=start, horizon=horizon
        )
        sim = process.forecast(
            params,
            resids,
            backcast,
            var_bounds,
            start=start,
            horizon=horizon,
            method="simulation",
            simulations=sims,
            rng=rng,
        )
        paths = sim.forecast_paths
        assert paths.shape == (40 - start, sims, horizon)
        assert sim.shocks.shape == paths.shape
        # The one-step variance is known
        assert_allclose(paths[:, :, 0], analytic.forecasts[:, [0]] * np.ones(sims))
        assert_allclose(sim.forecasts[:, 0], analytic.forecasts[:, 0], rtol=1e-12)
        assert_allclose(sim.forecasts, paths.mean(1), rtol=1e-12)
        se = paths.std(1, ddof=1) / np.sqrt(sims)
        z = (sim.forecasts[:, 1:] - analytic.forecasts[:, 1:]) / se[:, 1:]
        assert np.all(np.abs(z) < 4.0), z
        # The shocks are sqrt(variance) * z, so z has the variance 1
        assert_allclose(
            (sim.shocks[:, :, 1:] ** 2 / paths[:, :, 1:]).mean(), 1.0, atol=0.03
        )

    def test_simulation_from_the_start_of_the_sample(self):
        # The lags before the sample use the backcast in analytic and simulations
        order = (3, 2, 2)
        process, params, resids, backcast, var_bounds = self.setup_forecast(
            order, nobs=10, seed=34
        )
        sims = 30000
        analytic = process.forecast(
            params, resids, backcast, var_bounds, start=0, horizon=5
        )
        sim = process.forecast(
            params,
            resids,
            backcast,
            var_bounds,
            start=0,
            horizon=5,
            method="simulation",
            simulations=sims,
            rng=RandomState(101).standard_normal,
        )
        se = sim.forecast_paths.std(1, ddof=1) / np.sqrt(sims)
        z = (sim.forecasts[:, 1:] - analytic.forecasts[:, 1:]) / se[:, 1:]
        assert np.all(np.abs(z) < 4.0), z

    def test_simulation_is_reproducible_and_depends_on_the_draws(self):
        order = (1, 1, 1)
        process, params, resids, backcast, var_bounds = self.setup_forecast(order)

        def run(seed):
            return process.forecast(
                params,
                resids,
                backcast,
                var_bounds,
                horizon=4,
                method="simulation",
                simulations=50,
                rng=RandomState(seed).standard_normal,
            ).forecast_paths

        assert_allclose(run(1), run(1), rtol=1e-12)
        assert np.abs(run(1) - run(2)).max() > 1e-3

    def test_simulation_uses_the_shift(self):
        # A single large negative residual and a shift: with z = -1 for all draws,
        # the next shock is -sigma and the variance follows the recursion
        order = (1, 1, 1)
        process, _, resids, backcast, var_bounds = self.setup_forecast(order)
        params = np.array([0.1, 0.2, 0.7, 0.5])
        fcast = process.forecast(
            params,
            resids,
            backcast,
            var_bounds,
            horizon=3,
            method="simulation",
            simulations=4,
            rng=lambda size: -np.ones(size),
        )
        f1 = fcast.forecast_paths[0, 0, 0]
        f2 = 0.1 + 0.2 * f1 * (-1.0 - 0.7) ** 2 + 0.5 * f1
        f3 = 0.1 + 0.2 * f2 * (-1.0 - 0.7) ** 2 + 0.5 * f2
        assert_allclose(fcast.forecast_paths[0, :, 1], f2, rtol=1e-12)
        assert_allclose(fcast.forecast_paths[0, :, 2], f3, rtol=1e-12)
        assert_allclose(fcast.shocks[0, 0, :], -np.sqrt([f1, f2, f3]), rtol=1e-12)

    def test_bootstrap(self):
        order = (2, 1, 1)
        params = make_parameters(order, seed=35)
        resids, _ = simulate_data(order, params, 300, 35)
        process = make_ngarch(*order)
        backcast, var_bounds = var_inputs(process, resids)
        analytic = process.forecast(
            params, resids, backcast, var_bounds, start=250, horizon=5
        )
        boot = process.forecast(
            params,
            resids,
            backcast,
            var_bounds,
            start=250,
            horizon=5,
            method="bootstrap",
            simulations=500,
            random_state=RandomState(5),
        )
        assert boot.forecast_paths.shape == (50, 500, 5)
        assert np.all(np.isfinite(boot.forecasts))
        assert np.all(boot.forecasts > 0)
        assert_allclose(boot.forecasts[:, 0], analytic.forecasts[:, 0], rtol=1e-12)
        # The bootstrap is close to the analytic forecasts
        assert_allclose(boot.forecasts, analytic.forecasts, rtol=0.25)
        again = process.forecast(
            params,
            resids,
            backcast,
            var_bounds,
            start=250,
            horizon=5,
            method="bootstrap",
            simulations=500,
            random_state=RandomState(5),
        )
        assert_allclose(again.forecasts, boot.forecasts, rtol=1e-12)


ALL_CASES = ["s1", "s2", "s3", "s5"]
# The cases with q = 1, for which the first variance can be set to the initial value
# of rugarch
Q1_CASES = ["s1", "s2", "s3"]


class TestRugarch:
    @staticmethod
    def setup_case(name):
        case = RUGARCH_CASES[name]
        order = case.order
        params = np.array(case.params)
        resids = RUGARCH_Y - case.mu
        process = make_ngarch(*order)
        var_bounds = process.variance_bounds(resids)
        return case, order, params, resids, process, var_bounds

    @staticmethod
    def exact_backcast(case, params):
        """The backcast that makes the first variance the initial value of rugarch."""
        persistence = params[1] * (1 + params[2] ** 2) + params[3]
        return (case.init - params[0]) / persistence

    @pytest.mark.parametrize("name", ALL_CASES)
    def test_variance(self, name):
        case, order, params, resids, process, var_bounds = self.setup_case(name)
        p, o, q = order
        sigma2 = np.empty_like(resids)
        if q == 1:
            backcast = self.exact_backcast(case, params)
            process.compute_variance(params, resids, sigma2, backcast, var_bounds)
            # The 1e-6 in the fGARCH filter of rugarch
            assert_allclose(sigma2, case.sigma2, rtol=5e-6)
            shifted = params.copy()
            shifted[3] += params[1] * 1e-6
            backcast = self.exact_backcast(case, shifted)
            process.compute_variance(shifted, resids, sigma2, backcast, var_bounds)
            assert_allclose(sigma2, case.sigma2, rtol=1e-11)
        else:
            # The initial values differ, but the persistence is 0.5 and the
            # differences have vanished in the second half of the sample
            backcast = process.backcast(resids)
            shifted = params.copy()
            shifted[1 + p + o] += params[1] * 1e-6
            process.compute_variance(shifted, resids, sigma2, backcast, var_bounds)
            assert_allclose(sigma2[40:], case.sigma2[40:], rtol=1e-9)

    @pytest.mark.parametrize("name", Q1_CASES)
    def test_loglikelihood(self, name):
        case, _, params, resids, process, var_bounds = self.setup_case(name)
        shifted = params.copy()
        shifted[3] += params[1] * 1e-6
        backcast = self.exact_backcast(case, shifted)
        sigma2 = np.empty_like(resids)
        process.compute_variance(shifted, resids, sigma2, backcast, var_bounds)
        if case.dist == "norm":
            dist = Normal()
            dist_params: list[float] = []
        else:
            dist = StudentsT()
            dist_params = [case.shape]
        llf = dist.loglikelihood(dist_params, resids, sigma2)
        assert_allclose(llf, case.loglikelihood, rtol=1e-12, atol=1e-8)
        # Without the 1e-6 of rugarch
        backcast = self.exact_backcast(case, params)
        process.compute_variance(params, resids, sigma2, backcast, var_bounds)
        llf = dist.loglikelihood(dist_params, resids, sigma2)
        assert_allclose(llf, case.loglikelihood, rtol=1e-6)

    @pytest.mark.parametrize("name", ALL_CASES)
    def test_forecast(self, name):
        case, order, params, resids, process, var_bounds = self.setup_case(name)
        if order[2] == 1:
            backcast = self.exact_backcast(case, params)
        else:
            backcast = process.backcast(resids)
        fcast = process.forecast(params, resids, backcast, var_bounds, horizon=10)
        # rugarch uses the variance of its filter, which includes the 1e-6, at the
        # origin
        assert_allclose(fcast.forecasts[0], case.forecast, rtol=3e-6)


class TestArchModel:
    def test_arch_model(self):
        y, _ = simulate_data((1, 1, 1), [0.1, 0.1, 0.5, 0.75], 600, 41)
        am = arch_model(y, vol="ngarch", o=1)
        assert isinstance(am.volatility, NGARCH)
        assert (am.volatility.p, am.volatility.o, am.volatility.q) == (1, 1, 1)
        am = arch_model(y, vol="NGARCH", p=2, o=1, q=2)
        assert (am.volatility.p, am.volatility.o, am.volatility.q) == (2, 1, 2)
        with pytest.raises(ValueError, match=r"o must be an integer <= p \(1\)"):
            arch_model(y, vol="ngarch", p=1, o=2)
        with pytest.raises(ValueError, match="p must be an integer >= 1"):
            arch_model(y, vol="ngarch", p=0, o=0)

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"o": 0}, {"p": 2, "o": 0, "q": 1}, {"p": 1, "o": 0, "q": 0}],
        ids=["default", "o0", "p2-o0-q1", "p1-o0-q0"],
    )
    def test_arch_model_without_a_shift_is_garch(self, kwargs):
        y, _ = simulate_data((1, 1, 1), [0.1, 0.1, 0.5, 0.75], 600, 41)
        with pytest.warns(ValueWarning, match="vol='ngarch' with o=0 has no shift"):
            am = arch_model(y, vol="ngarch", **kwargs)
        assert type(am.volatility) is GARCH
        ref = arch_model(y, vol="garch", **kwargs)
        assert str(am.volatility) == str(ref.volatility)
        assert_allclose(
            am.fit(disp="off").params.to_numpy(), ref.fit(disp="off").params.to_numpy()
        )

    def test_fit_results(self):
        y, _ = simulate_data((1, 1, 1), [0.1, 0.1, 0.5, 0.75], 1500, 42)
        res = arch_model(y, vol="ngarch", o=1, mean="zero").fit(disp="off")
        assert list(res.params.index) == ["omega", "alpha[1]", "theta[1]", "beta[1]"]
        assert "Zero Mean - NGARCH Model Results" in str(res)
        assert str(res.model.volatility) == "NGARCH(p: 1, o: 1, q: 1)"
        # The conditional volatility is the square root of the recursion
        params = res.params.to_numpy()
        process = res.model.volatility
        sigma2 = np.empty(y.shape[0])
        backcast, var_bounds = var_inputs(process, y)
        process.compute_variance(params, y, sigma2, backcast, var_bounds)
        assert_allclose(res.conditional_volatility, np.sqrt(sigma2), rtol=1e-10)
        assert_allclose(
            res.loglikelihood, Normal().loglikelihood([], y, sigma2), rtol=1e-10
        )
        # The estimates satisfy the constraints
        assert res.params["alpha[1]"] + res.params["beta[1]"] < 1
        assert res.params["theta[1]"] > 0

    def test_fix_matches_reference(self):
        y, _ = simulate_data((2, 1, 1), [0.1, 0.07, 0.04, 0.6, 0.7], 400, 43)
        am = ConstantMean(y, volatility=NGARCH(2, 1, 1))
        params = np.array([0.05, 0.1, 0.07, 0.04, 0.6, 0.7])
        res = am.fix(params)
        resids = y - 0.05
        backcast = NGARCH(2, 1, 1).backcast(y - y.mean())
        sigma2 = reference_variance(params[1:], (2, 1, 1), resids, backcast)
        assert_allclose(res.conditional_volatility, np.sqrt(sigma2), rtol=1e-10)
        expected = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + resids**2 / sigma2)
        assert_allclose(res.loglikelihood, expected, rtol=1e-10)

    def test_forecast_through_the_results(self):
        y, _ = simulate_data((1, 1, 1), [0.1, 0.1, 0.5, 0.75], 800, 44)
        res = arch_model(y, vol="ngarch", o=1).fit(disp="off")
        fcast = res.forecast(horizon=6, start=790, reindex=False)
        assert fcast.variance.shape == (10, 6)
        params = res.params
        phi = params["alpha[1]"] * (1 + params["theta[1]"] ** 2) + params["beta[1]"]
        variance = fcast.variance.to_numpy()
        for h in range(1, 6):
            assert_allclose(
                variance[:, h], params["omega"] + phi * variance[:, h - 1], rtol=1e-10
            )
        sim = res.forecast(
            horizon=3,
            start=795,
            method="simulation",
            simulations=200,
            reindex=False,
            rng=RandomState(0).standard_normal,
        )
        assert sim.variance.shape == (5, 3)
        assert_allclose(sim.variance.iloc[:, 0], variance[5:, 0], rtol=1e-10)
        boot = res.forecast(
            horizon=3,
            start=500,
            method="bootstrap",
            simulations=100,
            reindex=False,
            random_state=RandomState(0),
        )
        assert boot.variance.shape == (300, 3)

    def test_pickle(self):
        y, _ = simulate_data((1, 1, 1), [0.1, 0.1, 0.5, 0.75], 500, 45)
        am = arch_model(y, vol="ngarch", o=1)
        res = am.fit(disp="off")
        restored = pickle.loads(pickle.dumps(am))
        assert isinstance(restored.volatility, NGARCH)
        res2 = restored.fit(disp="off")
        assert_allclose(res2.params, res.params, rtol=1e-6)


def estimate_within(res, truth, nse=4.0):
    """The estimates are within nse standard errors of the truth."""
    se = res.std_err.to_numpy()
    est = res.params.to_numpy()
    z = (est - np.asarray(truth)) / se
    assert np.all(np.abs(z) < nse), pd.DataFrame(
        {"est": est, "truth": truth, "se": se, "z": z}, index=res.params.index
    )


def value_warnings(record):
    return [w for w in record if issubclass(w.category, ValueWarning)]


class TestWarnings:
    def test_arch_model_without_a_shift_points_at_the_caller(self):
        y, _ = simulate_data((1, 1, 1), [0.1, 0.1, 0.5, 0.75], 200, 41)
        with pytest.warns(ValueWarning, match="vol='ngarch' with o=0") as record:
            arch_model(y, vol="ngarch")
        assert [w.filename for w in value_warnings(record)] == [__file__]

    def test_no_shift_points_at_the_caller(self):
        with pytest.warns(ValueWarning, match=NO_SHIFT) as record:
            process = NGARCH(1, 0, 1)
        assert [w.filename for w in value_warnings(record)] == [__file__]
        assert process.num_params == 3

    def test_shift_does_not_warn(self):
        y, _ = simulate_data((1, 1, 1), [0.1, 0.1, 0.5, 0.75], 200, 41)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            NGARCH()
            NGARCH(2, 1, 1)
            arch_model(y, vol="ngarch", o=1)
        assert value_warnings(record) == []

    @staticmethod
    def boundary_data(seed):
        # A persistence of exactly 1, so that some fits are not stationary
        params = np.array([0.02, 0.1, 1.0, 0.8])
        rng = RandomState(seed).standard_normal
        y, _ = NGARCH().simulate(params, 1500, rng, burn=500, initial_value=1.0)
        return y

    @staticmethod
    def persistence(res):
        _, alpha, theta, beta = res.params.to_numpy()
        return alpha * (1 + theta**2) + beta

    def test_persistence_of_one_or_more_warns(self):
        mod = ZeroMean(self.boundary_data(9), volatility=NGARCH())
        with pytest.warns(ValueWarning, match="persistence of the estimated") as record:
            res = mod.fit(disp="off")
        warned = value_warnings(record)
        assert [w.filename for w in warned] == [__file__]
        persistence = self.persistence(res)
        assert persistence >= 1
        assert f"= {persistence:0.4f}, is 1 or more" in str(warned[0].message)

    def test_persistence_warning_follows_show_warning(self):
        mod = ZeroMean(self.boundary_data(9), volatility=NGARCH())
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            res = mod.fit(disp="off", show_warning=False)
        assert value_warnings(record) == []
        assert self.persistence(res) >= 1

    def test_stationary_fit_does_not_warn(self):
        mod = ZeroMean(self.boundary_data(6), volatility=NGARCH())
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            res = mod.fit(disp="off")
        assert value_warnings(record) == []
        assert self.persistence(res) < 1

    def test_fit_warning(self):
        # phi = 0.1 * 1.25 + 0.05 + 0.3 and 0.2 * 5 + 0.05
        process = NGARCH(2, 1, 1)
        assert process._fit_warning(np.array([0.1, 0.1, 0.05, 0.5, 0.3])) is None
        message = process._fit_warning(np.array([0.1, 0.2, 0.05, 2.0, 0.0]))
        assert message is not None
        assert "sum(alpha * (1 + theta ** 2)) + sum(beta) = 1.0500" in message
        # A persistence of exactly 1 is not stationary
        assert NGARCH()._fit_warning(np.array([0.1, 0.25, 1.0, 0.5])) is not None
        # Other processes do not check their estimates
        assert GARCH()._fit_warning(np.array([0.1, 0.1, 0.9])) is None


class TestRecovery:
    def test_normal(self):
        truth = [0.1, 0.1, 0.5, 0.75]
        y, _ = simulate_data((1, 1, 1), truth, 4000, 51)
        res = ConstantMean(y, volatility=NGARCH()).fit(disp="off")
        estimate_within(res, [0.0, *truth])
        a, b = NGARCH().constraints()
        assert np.all(a @ res.params.to_numpy()[1:] - b >= -1e-8)

    def test_normal_two_lags(self):
        truth = [0.1, 0.08, 0.04, 0.6, 0.7]
        y, _ = simulate_data((2, 1, 1), truth, 8000, 52)
        res = ConstantMean(y, volatility=NGARCH(2, 1, 1)).fit(disp="off")
        estimate_within(res, [0.0, *truth])

    def test_two_shifts(self):
        truth = [0.1, 0.1, 0.05, 0.7, 0.3, 0.65]
        y, _ = simulate_data((2, 2, 1), truth, 12000, 56)
        res = ConstantMean(y, volatility=NGARCH(2, 2, 1)).fit(disp="off")
        estimate_within(res, [0.0, *truth])

    def test_skew_t(self):
        truth = [0.1, 0.1, 0.5, 0.75]
        dist = SkewStudent(seed=RandomState(54))
        y, _ = simulate_data((1, 1, 1), truth, 6000, 54, rng=dist.simulate([8.0, -0.2]))
        res = ConstantMean(y, volatility=NGARCH(), distribution=SkewStudent()).fit(
            disp="off"
        )
        assert list(res.params.index[-2:]) == ["eta", "lambda"]
        estimate_within(res, [0.0, *truth, 8.0, -0.2])

    def test_student_t(self):
        truth = [0.1, 0.1, 0.5, 0.75]
        dist = StudentsT(seed=RandomState(55))
        y, _ = simulate_data((1, 1, 1), truth, 5000, 55, rng=dist.simulate([7.0]))
        res = ConstantMean(y, volatility=NGARCH(), distribution=StudentsT()).fit(
            disp="off"
        )
        estimate_within(res, [0.0, *truth, 7.0])


IN_MEAN_FORMS = ["log", "vol", "var", 1.5]


class TestArchInMean:
    @pytest.mark.parametrize("form", IN_MEAN_FORMS)
    def test_fix_matches_reference(self, form):
        order = (2, 1, 1)
        y, _ = simulate_data(order, [0.1, 0.07, 0.04, 0.6, 0.7], 300, 61)
        y = y + 0.05
        process = make_ngarch(*order)
        model = ARCHInMean(y, volatility=process, form=form)
        # Const, kappa, then the parameters of the volatility process
        params = np.array([0.05, 0.1, 0.1, 0.07, 0.04, 0.6, 0.7])
        res = model.fix(params)
        # The recursion in the residuals and the variance, which depend on each other
        backcast = process.backcast(y - y.mean())
        mu, kappa = params[:2]
        omega, alpha, theta, beta = params[2], params[3:5], params[5], params[6]
        sigma2 = np.empty(y.shape[0])
        resids = np.empty(y.shape[0])
        for t in range(y.shape[0]):
            value = omega + beta * (sigma2[t - 1] if t >= 1 else backcast)
            for i in (1, 2):
                th = theta if i == 1 else 0.0
                if t - i >= 0:
                    value += (
                        alpha[i - 1]
                        * (resids[t - i] - th * np.sqrt(sigma2[t - i])) ** 2
                    )
                else:
                    value += alpha[i - 1] * (1 + th**2) * backcast
            sigma2[t] = value
            if form == "log":
                mean = mu + kappa * np.log(value)
            else:
                power = {"vol": 1.0, "var": 2.0}.get(form, form)
                mean = mu + kappa * value ** (power / 2.0)
            resids[t] = y[t] - mean
        assert_allclose(res.resid, resids, rtol=1e-10, atol=1e-12)
        assert_allclose(res.conditional_volatility, np.sqrt(sigma2), rtol=1e-10)
        expected = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + resids**2 / sigma2)
        assert_allclose(res.loglikelihood, expected, rtol=1e-10)

    def test_fit_recovers_parameters(self):
        truth = [0.1, 0.1, 0.5, 0.75]
        kappa, mu = 0.08, 0.02
        # The variance in the mean, form "var"
        process = NGARCH()
        errors = RandomState(62).standard_normal(5500)
        data = np.empty(5500)
        sigma2 = np.empty(5500)
        sigma2[0] = 0.1 / (1 - 0.1 * 1.25 - 0.75)
        eps = np.sqrt(sigma2[0]) * errors[0]
        data[0] = mu + kappa * sigma2[0] + eps
        for t in range(1, 5500):
            sigma2[t] = (
                truth[0]
                + truth[1] * (eps - truth[2] * np.sqrt(sigma2[t - 1])) ** 2
                + truth[3] * sigma2[t - 1]
            )
            eps = np.sqrt(sigma2[t]) * errors[t]
            data[t] = mu + kappa * sigma2[t] + eps
        res = ARCHInMean(data[500:], volatility=process, form="var").fit(disp="off")
        estimate_within(res, [mu, kappa, *truth])

    def test_matches_constant_mean_without_the_in_mean_term(self):
        y, _ = simulate_data((1, 1, 1), [0.1, 0.1, 0.5, 0.75], 600, 63)
        aim = ARCHInMean(y, volatility=NGARCH(), form="var")
        params = np.array([0.0, 0.0, 0.1, 0.1, 0.5, 0.75])
        cm = ConstantMean(y, volatility=NGARCH()).fix(params[[0, 2, 3, 4, 5]])
        res = aim.fix(params)
        assert_allclose(
            res.conditional_volatility, cm.conditional_volatility, rtol=1e-12
        )
        assert_allclose(res.loglikelihood, cm.loglikelihood, rtol=1e-10)

    def test_compiled_and_python_updaters_give_the_same_fit_likelihood(self):
        y, _ = simulate_data((1, 1, 1), [0.1, 0.1, 0.5, 0.75], 300, 64)
        params = np.array([0.0, 0.05, 0.1, 0.1, 0.5, 0.75])
        process = NGARCH()
        llf = []
        for module in (recpy,) if MISSING_EXTENSION else (recpy, rec_cython):
            model = ARCHInMean(y, volatility=process, form="log")
            model._volatility_updater = module.NGARCHUpdater(1, 1, 1)
            model._recursion = module.ARCHInMeanRecursion(model._volatility_updater)
            llf.append(model.fix(params).loglikelihood)
        assert_allclose(llf[0], llf[-1], rtol=1e-12)
