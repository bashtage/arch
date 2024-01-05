import types
from typing import NamedTuple
import warnings

import numpy as np
from numpy.random import RandomState
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)
import pytest
from scipy.special import gamma, gammaln

from arch.univariate import ZeroMean
from arch.univariate.distribution import Normal, SkewStudent, StudentsT
import arch.univariate.recursions_python as recpy
from arch.univariate.volatility import (
    APARCH,
    ARCH,
    EGARCH,
    FIGARCH,
    GARCH,
    HARCH,
    ConstantVariance,
    EWMAVariance,
    FixedVariance,
    MIDASHyperbolic,
    RiskMetrics2006,
)
from arch.utility.exceptions import InitialValueWarning

rec: types.ModuleType
try:
    from arch.univariate import recursions as rec
except ImportError:
    rec = recpy

pytestmark = pytest.mark.filterwarnings("ignore::arch.compat.numba.PerformanceWarning")

EPS_HALF = np.sqrt(np.finfo(float).eps)


class SetupInfo(NamedTuple):
    rng: RandomState
    t: int
    resids: np.ndarray
    resid_var: float
    sigma2: np.ndarray
    backcast: float


@pytest.fixture(scope="module")
def setup():
    rng = RandomState(1234)
    t = 1000
    resids = rng.standard_normal(t)
    return SetupInfo(
        rng=rng,
        t=t,
        resids=resids,
        resid_var=np.var(resids),
        sigma2=np.zeros_like(resids),
        backcast=1.0,
    )


@pytest.mark.parametrize("initial_value", [None, 1.0])
def test_garch(setup, initial_value):
    garch = GARCH()

    sv = garch.starting_values(setup.resids)
    assert_equal(sv.shape[0], garch.num_params)

    bounds = garch.bounds(setup.resids)
    v = np.mean(setup.resids**2.0)
    assert_allclose(bounds[0], (1.0e-8 * v, 10.0 * v))
    assert_equal(bounds[1], (0.0, 1.0))
    assert_equal(bounds[2], (0.0, 1.0))
    backcast = garch.backcast(setup.resids)
    w = 0.94 ** np.arange(75)
    assert_almost_equal(backcast, np.sum((setup.resids[:75] ** 2) * (w / w.sum())))
    var_bounds = garch.variance_bounds(setup.resids)
    parameters = np.array([0.1, 0.1, 0.8])
    garch.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)
    cond_var_direct = np.zeros_like(setup.sigma2)
    rec.garch_recursion(
        parameters,
        setup.resids**2.0,
        np.sign(setup.resids),
        cond_var_direct,
        1,
        0,
        1,
        setup.t,
        backcast,
        var_bounds,
    )
    assert_allclose(setup.sigma2, cond_var_direct)

    a, b = garch.constraints()
    a_target = np.vstack((np.eye(3), np.array([[0, -1.0, -1.0]])))
    b_target = np.array([0.0, 0.0, 0.0, -1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)
    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = garch.simulate(
        parameters, setup.t, rng.simulate([]), initial_value=initial_value
    )
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    initial_value = 1.0
    sigma2 = np.zeros(setup.t + 500)
    data = np.zeros(setup.t + 500)
    for t in range(setup.t + 500):
        sigma2[t] = parameters[0]
        shock = initial_value if t == 0 else data[t - 1] ** 2.0
        sigma2[t] += parameters[1] * shock
        lagged_value = initial_value if t == 0 else sigma2[t - 1]
        sigma2[t] += parameters[2] * lagged_value
        data[t] = e[t] * np.sqrt(sigma2[t])
    data = data[500:]
    sigma2 = sigma2[500:]
    assert_almost_equal(data / sim_data[0], np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))

    names = garch.parameter_names()
    names_target = ["omega", "alpha[1]", "beta[1]"]
    assert_equal(names, names_target)

    assert isinstance(garch.__str__(), str)
    txt = garch.__repr__()
    assert str(hex(id(garch))) in txt

    assert_equal(garch.name, "GARCH")
    assert_equal(garch.num_params, 3)
    assert_equal(garch.power, 2.0)
    assert_equal(garch.p, 1)
    assert_equal(garch.o, 0)
    assert_equal(garch.q, 1)


def test_garch_power(setup):
    garch = GARCH(power=1.0)
    assert_equal(garch.num_params, 3)
    assert_equal(garch.name, "AVGARCH")
    assert_equal(garch.power, 1.0)

    sv = garch.starting_values(setup.resids)
    assert_equal(sv.shape[0], garch.num_params)

    bounds = garch.bounds(setup.resids)
    v = np.mean(np.abs(setup.resids))
    assert_allclose(bounds[0], (1.0e-8 * v, 10.0 * v))
    assert_equal(bounds[1], (0.0, 1.0))
    assert_equal(bounds[2], (0.0, 1.0))
    var_bounds = garch.variance_bounds(setup.resids)
    backcast = garch.backcast(setup.resids)
    w = 0.94 ** np.arange(75)
    assert_almost_equal(backcast, np.sum(np.abs(setup.resids[:75]) * (w / w.sum())))

    parameters = np.array([0.1, 0.1, 0.8])
    garch.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)
    cond_var_direct = np.zeros_like(setup.sigma2)
    rec.garch_recursion(
        parameters,
        np.abs(setup.resids),
        np.sign(setup.resids),
        cond_var_direct,
        1,
        0,
        1,
        setup.t,
        backcast,
        var_bounds,
    )
    cond_var_direct **= 2.0  # Square since recursion does not apply power
    assert_allclose(setup.sigma2, cond_var_direct)

    a, b = garch.constraints()
    a_target = np.vstack((np.eye(3), np.array([[0, -1.0, -1.0]])))
    b_target = np.array([0.0, 0.0, 0.0, -1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)
    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = garch.simulate(parameters, setup.t, rng.simulate([]))
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    initial_value = 1.0
    sigma = np.zeros(setup.t + 500)
    data = np.zeros(setup.t + 500)
    for t in range(setup.t + 500):
        sigma[t] = parameters[0]
        shock = initial_value if t == 0 else np.abs(data[t - 1])
        sigma[t] += parameters[1] * shock
        lagged_value = initial_value if t == 0 else sigma[t - 1]
        sigma[t] += parameters[2] * lagged_value
        data[t] = e[t] * sigma[t]
    data = data[500:]
    sigma2 = sigma[500:] ** 2.0
    assert_almost_equal(data - sim_data[0] + 1.0, np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))


@pytest.mark.parametrize("initial_value", [None, 1.0])
def test_arch(setup, initial_value):
    arch = ARCH()

    sv = arch.starting_values(setup.resids)
    assert_equal(sv.shape[0], arch.num_params)

    bounds = arch.bounds(setup.resids)
    v = np.mean(setup.resids**2.0)
    assert_allclose(bounds[0], (1.0e-8 * v, 10.0 * v))
    assert_equal(bounds[1], (0.0, 1.0))

    backcast = arch.backcast(setup.resids)
    w = 0.94 ** np.arange(75)
    assert_almost_equal(backcast, np.sum((setup.resids[:75] ** 2) * (w / w.sum())))

    parameters = np.array([0.5, 0.7])
    var_bounds = arch.variance_bounds(setup.resids)
    arch.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)
    cond_var_direct = np.zeros_like(setup.sigma2)
    rec.arch_recursion(
        parameters, setup.resids, cond_var_direct, 1, setup.t, backcast, var_bounds
    )
    assert_allclose(setup.sigma2, cond_var_direct)

    a, b = arch.constraints()
    a_target = np.vstack((np.eye(2), np.array([[0, -1.0]])))
    b_target = np.array([0.0, 0.0, -1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)
    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = arch.simulate(
        parameters, setup.t, rng.simulate([]), initial_value=initial_value
    )
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    initial_value = 1.0
    sigma2 = np.zeros(setup.t + 500)
    data = np.zeros(setup.t + 500)
    for t in range(setup.t + 500):
        sigma2[t] = parameters[0]
        shock = initial_value if t == 0 else data[t - 1] ** 2.0
        sigma2[t] += parameters[1] * shock
        data[t] = e[t] * np.sqrt(sigma2[t])
    data = data[500:]
    sigma2 = sigma2[500:]
    assert_almost_equal(data - sim_data[0] + 1.0, np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))

    names = arch.parameter_names()
    names_target = ["omega", "alpha[1]"]
    assert_equal(names, names_target)

    assert_equal(arch.name, "ARCH")
    assert_equal(arch.num_params, 2)
    assert_equal(arch.p, 1)
    assert isinstance(arch.__str__(), str)
    txt = arch.__repr__()
    assert str(hex(id(arch))) in txt


def test_arch_harch(setup):
    arch = ARCH(p=1)
    harch = HARCH(lags=1)
    assert_equal(arch.num_params, harch.num_params)
    parameters = np.array([0.5, 0.5])

    backcast = arch.backcast(setup.resids)
    assert_equal(backcast, harch.backcast(setup.resids))
    sigma2_arch = np.zeros_like(setup.sigma2)
    sigma2_harch = np.zeros_like(setup.sigma2)
    var_bounds = arch.variance_bounds(setup.resids)
    arch.compute_variance(parameters, setup.resids, sigma2_arch, backcast, var_bounds)
    harch.compute_variance(parameters, setup.resids, sigma2_harch, backcast, var_bounds)
    assert_allclose(sigma2_arch, sigma2_harch)

    a, b = arch.constraints()
    ah, bh = harch.constraints()
    assert_equal(a, ah)
    assert_equal(b, bh)
    assert isinstance(arch.__str__(), str)
    txt = arch.__repr__()
    assert str(hex(id(arch))) in txt


@pytest.mark.parametrize("initial_value", [None, 1.0])
def test_harch(setup, initial_value):
    harch = HARCH(lags=[1, 5, 22])

    sv = harch.starting_values(setup.resids)
    assert_equal(sv.shape[0], harch.num_params)

    bounds = harch.bounds(setup.resids)
    assert_equal(bounds[0], (0.0, 10.0 * np.mean(setup.resids**2.0)))
    assert_equal(bounds[1], (0.0, 1.0))
    assert_equal(bounds[2], (0.0, 1.0))
    assert_equal(bounds[3], (0.0, 1.0))
    var_bounds = harch.variance_bounds(setup.resids)
    backcast = harch.backcast(setup.resids)
    w = 0.94 ** np.arange(75)
    assert_almost_equal(backcast, np.sum((setup.resids[:75] ** 2) * (w / w.sum())))

    parameters = np.array([0.1, 0.4, 0.3, 0.2])

    var_bounds = harch.variance_bounds(setup.resids)
    harch.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)
    cond_var_direct = np.zeros_like(setup.sigma2)
    lags = np.array([1, 5, 22], dtype=np.int32)
    rec.harch_recursion(
        parameters, setup.resids, cond_var_direct, lags, setup.t, backcast, var_bounds
    )

    names = harch.parameter_names()
    names_target = ["omega", "alpha[1]", "alpha[5]", "alpha[22]"]
    assert_equal(names, names_target)

    assert_allclose(setup.sigma2, cond_var_direct)

    a, b = harch.constraints()
    a_target = np.vstack((np.eye(4), np.array([[0, -1.0, -1.0, -1.0]])))
    b_target = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)
    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = harch.simulate(
        parameters, setup.t, rng.simulate([]), initial_value=initial_value
    )
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    sigma2 = np.zeros(setup.t + 500)
    data = np.zeros(setup.t + 500)
    lagged = np.zeros(22)
    for t in range(setup.t + 500):
        sigma2[t] = parameters[0]
        lagged[:] = backcast
        if t > 0:
            if t == 1:
                lagged[0] = data[0] ** 2.0
            elif t < 22:
                lagged[:t] = data[t - 1 :: -1] ** 2.0
            else:
                lagged = data[t - 1 : t - 22 : -1] ** 2.0

        shock1 = data[t - 1] ** 2.0 if t > 0 else backcast
        if t >= 5:
            shock5 = np.mean(data[t - 5 : t] ** 2.0)
        else:
            shock5 = 0.0
            for i in range(5):
                shock5 += data[t - i - 1] if t - i - 1 >= 0 else backcast
            shock5 = shock5 / 5.0

        if t >= 22:
            shock22 = np.mean(data[t - 22 : t] ** 2.0)
        else:
            shock22 = 0.0
            for i in range(22):
                shock22 += data[t - i - 1] if t - i - 1 >= 0 else backcast
            shock22 = shock22 / 22.0

        sigma2[t] += (
            parameters[1] * shock1 + parameters[2] * shock5 + parameters[3] * shock22
        )

        data[t] = e[t] * np.sqrt(sigma2[t])
    data = data[500:]
    sigma2 = sigma2[500:]
    assert_almost_equal(data - sim_data[0] + 1.0, np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))

    assert_equal(harch.name, "HARCH")
    assert_equal(harch.lags, [1, 5, 22])
    assert_equal(harch.num_params, 4)
    assert isinstance(harch.__str__(), str)
    txt = harch.__repr__()
    assert str(hex(id(harch))) in txt


def test_constant_variance(setup):
    cv = ConstantVariance()

    sv = cv.starting_values(setup.resids)
    assert_equal(sv.shape[0], cv.num_params)

    bounds = cv.bounds(setup.resids)
    mean_square = np.mean(setup.resids**2.0)
    assert_almost_equal(bounds[0], (setup.resid_var / 100000.0, 10.0 * mean_square))

    backcast = cv.backcast(setup.resids)
    var_bounds = cv.variance_bounds(setup.resids)
    assert_almost_equal(setup.resid_var, backcast)

    parameters = np.array([setup.resid_var])

    cv.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)
    assert_allclose(np.ones_like(setup.sigma2) * setup.resid_var, setup.sigma2)

    a, b = cv.constraints()
    a_target = np.eye(1)
    b_target = np.array([0.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)

    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = cv.simulate(parameters, setup.t, rng.simulate([]))
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    sigma2 = np.zeros(setup.t + 500)
    sigma2[:] = parameters[0]
    data = np.zeros(setup.t + 500)
    data[:] = np.sqrt(sigma2) * e
    data = data[500:]
    sigma2 = sigma2[500:]

    names = cv.parameter_names()
    names_target = ["sigma2"]
    assert_equal(names, names_target)

    assert_almost_equal(data - sim_data[0] + 1.0, np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))

    assert_equal(cv.num_params, 1)
    assert_equal(cv.name, "Constant Variance")
    assert isinstance(cv.__str__(), str)
    txt = cv.__repr__()
    assert str(hex(id(cv))) in txt


def test_garch_no_symmetric(setup):
    garch = GARCH(p=0, o=1, q=1)

    sv = garch.starting_values(setup.resids)
    assert_equal(sv.shape[0], garch.num_params)

    bounds = garch.bounds(setup.resids)
    v = np.mean(setup.resids**2)
    assert_allclose(bounds[0], (1.0e-8 * v, 10.0 * v))
    assert_equal(bounds[1], (0.0, 2.0))
    assert_equal(bounds[2], (0.0, 1.0))
    var_bounds = garch.variance_bounds(setup.resids)
    backcast = garch.backcast(setup.resids)
    parameters = np.array([0.1, 0.1, 0.8])

    names = garch.parameter_names()
    names_target = ["omega", "gamma[1]", "beta[1]"]
    assert_equal(names, names_target)

    garch.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)
    cond_var_direct = np.zeros_like(setup.sigma2)
    rec.garch_recursion(
        parameters,
        setup.resids**2.0,
        np.sign(setup.resids),
        cond_var_direct,
        0,
        1,
        1,
        setup.t,
        backcast,
        var_bounds,
    )
    assert_allclose(setup.sigma2, cond_var_direct)

    a, b = garch.constraints()
    a_target = np.vstack((np.eye(3), np.array([[0, -0.5, -1.0]])))
    b_target = np.array([0.0, 0.0, 0.0, -1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)
    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = garch.simulate(parameters, setup.t, rng.simulate([]))
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    initial_value = 1.0
    sigma2 = np.zeros(setup.t + 500)
    data = np.zeros(setup.t + 500)
    for t in range(setup.t + 500):
        sigma2[t] = parameters[0]
        shock = (
            0.5 * initial_value if t == 0 else data[t - 1] ** 2.0 * (data[t - 1] < 0)
        )
        sigma2[t] += parameters[1] * shock
        lagged_value = initial_value if t == 0 else sigma2[t - 1]
        sigma2[t] += parameters[2] * lagged_value
        data[t] = e[t] * np.sqrt(sigma2[t])
    data = data[500:]
    sigma2 = sigma2[500:]
    assert_almost_equal(data - sim_data[0] + 1.0, np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))

    assert_equal(garch.p, 0)
    assert_equal(garch.o, 1)
    assert_equal(garch.q, 1)
    assert_equal(garch.num_params, 3)
    assert_equal(garch.name, "GJR-GARCH")


def test_garch_no_lagged_vol(setup):
    garch = GARCH(p=1, o=1, q=0)
    sv = garch.starting_values(setup.resids)
    assert_equal(sv.shape[0], garch.num_params)

    bounds = garch.bounds(setup.resids)
    v = np.mean(setup.resids**2)
    assert_allclose(bounds[0], (1.0e-8 * v, 10.0 * v))
    assert_equal(bounds[1], (0.0, 1.0))
    assert_equal(bounds[2], (-1.0, 2.0))

    backcast = garch.backcast(setup.resids)
    parameters = np.array([0.5, 0.25, 0.5])
    var_bounds = garch.variance_bounds(setup.resids)

    garch.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)
    cond_var_direct = np.zeros_like(setup.sigma2)
    rec.garch_recursion(
        parameters,
        setup.resids**2.0,
        np.sign(setup.resids),
        cond_var_direct,
        1,
        1,
        0,
        setup.t,
        backcast,
        var_bounds,
    )
    assert_allclose(setup.sigma2, cond_var_direct)

    a, b = garch.constraints()
    a_target = np.vstack((np.eye(3), np.array([[0, -1.0, -0.5]])))
    a_target[2, 1] = 1.0
    b_target = np.array([0.0, 0.0, 0.0, -1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)
    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = garch.simulate(parameters, setup.t, rng.simulate([]))
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    initial_value = 1.0
    sigma2 = np.zeros(setup.t + 500)
    data = np.zeros(setup.t + 500)
    for t in range(setup.t + 500):
        sigma2[t] = parameters[0]
        shock = initial_value if t == 0 else data[t - 1] ** 2.0
        sigma2[t] += parameters[1] * shock
        shock = (
            0.5 * initial_value if t == 0 else (data[t - 1] ** 2.0) * (data[t - 1] < 0)
        )
        sigma2[t] += parameters[2] * shock
        data[t] = e[t] * np.sqrt(sigma2[t])
    data = data[500:]
    sigma2 = sigma2[500:]
    assert_almost_equal(data - sim_data[0] + 1.0, np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))

    assert_equal(garch.p, 1)
    assert_equal(garch.o, 1)
    assert_equal(garch.q, 0)
    assert_equal(garch.num_params, 3)
    assert_equal(garch.name, "GJR-GARCH")


def test_arch_multiple_lags(setup):
    arch = ARCH(p=5)

    sv = arch.starting_values(setup.resids)
    assert_equal(sv.shape[0], arch.num_params)

    bounds = arch.bounds(setup.resids)
    v = np.mean(setup.resids**2)
    assert_allclose(bounds[0], (1.0e-8 * v, 10.0 * v))
    for i in range(1, 6):
        assert_equal(bounds[i], (0.0, 1.0))
    var_bounds = arch.variance_bounds(setup.resids)
    backcast = arch.backcast(setup.resids)
    parameters = np.array([0.25, 0.17, 0.16, 0.15, 0.14, 0.13])
    arch.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)
    cond_var_direct = np.zeros_like(setup.sigma2)
    rec.arch_recursion(
        parameters, setup.resids, cond_var_direct, 5, setup.t, backcast, var_bounds
    )
    assert_allclose(setup.sigma2, cond_var_direct)

    a, b = arch.constraints()
    a_target = np.vstack((np.eye(6), np.array([[0, -1.0, -1.0, -1.0, -1.0, -1.0]])))
    b_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)
    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = arch.simulate(parameters, setup.t, rng.simulate([]))
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    initial_value = 1.0
    sigma2 = np.zeros(setup.t + 500)
    data = np.zeros(setup.t + 500)
    for t in range(setup.t + 500):
        sigma2[t] = parameters[0]
        for i in range(5):
            if t - i - 1 < 0:
                sigma2[t] += parameters[i + 1] * initial_value
            else:
                sigma2[t] += parameters[i + 1] * data[t - i - 1] ** 2.0
        data[t] = e[t] * np.sqrt(sigma2[t])
    data = data[500:]
    sigma2 = sigma2[500:]
    assert_almost_equal(data - sim_data[0] + 1.0, np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))

    names = arch.parameter_names()
    names_target = ["omega"]
    names_target.extend(["alpha[" + str(i + 1) + "]" for i in range(5)])
    assert_equal(names, names_target)

    assert_equal(arch.num_params, 6)
    assert_equal(arch.name, "ARCH")


def test_harch_scalar(setup):
    harch = HARCH(lags=2)
    assert_equal(harch.num_params, 3)
    assert_equal(harch.name, "HARCH")


def test_garch_many_lags(setup):
    garch = GARCH(p=1, o=2, q=3)
    assert_equal(garch.num_params, 7)
    assert_equal(garch.name, "GJR-GARCH")

    names = garch.parameter_names()
    names_target = [
        "omega",
        "alpha[1]",
        "gamma[1]",
        "gamma[2]",
        "beta[1]",
        "beta[2]",
        "beta[3]",
    ]
    assert_equal(names, names_target)


def test_errors(setup):
    with pytest.raises(ValueError):
        GARCH(p=-1)
    with pytest.raises(ValueError):
        GARCH(o=-1)
    with pytest.raises(ValueError):
        GARCH(q=-1)
    with pytest.raises(ValueError):
        GARCH(p=0, q=0)
    with pytest.raises(ValueError):
        GARCH(power=-0.5)
    with pytest.raises(ValueError):
        EWMAVariance(lam=-0.5)
    with pytest.raises(ValueError):
        RiskMetrics2006(tau0=1, tau1=10)
    with pytest.raises(ValueError):
        RiskMetrics2006(tau0=1, tau1=10)
    with pytest.raises(ValueError):
        RiskMetrics2006(tau1=-10)
    with pytest.raises(ValueError):
        RiskMetrics2006(tau0=10, tau1=8, rho=1.5)
    with pytest.raises(ValueError):
        RiskMetrics2006(kmax=0)
    with pytest.raises(ValueError):
        RiskMetrics2006(rho=0.5)


def test_warnings_nonstationary(setup):
    garch = GARCH()
    parameters = np.array([0.1, 0.2, 0.8, 4.0])
    studt = StudentsT()
    warnings.simplefilter("always", UserWarning)
    with pytest.warns(InitialValueWarning):
        garch.simulate(parameters, 1000, studt.simulate([4.0]))


def test_warnings_nonstationary_garch(setup):
    garch = GARCH()
    parameters = np.array([0.1, 0.2, 0.8, 4.0, 0.5])
    skewstud = SkewStudent()
    with pytest.warns(InitialValueWarning):
        garch.simulate(parameters, 1000, skewstud.simulate([4.0, 0.5]))


def test_warnings_nonstationary_harch(setup):
    studt = StudentsT()
    harch = HARCH(lags=[1, 5, 22])
    parameters = np.array([0.1, 0.2, 0.4, 0.5])
    with pytest.warns(InitialValueWarning):
        harch.simulate(parameters, 1000, studt.simulate([4.0]))


def test_model_names(setup):
    garch = GARCH(2, 0, 0)
    assert_equal(garch.name, "ARCH")
    garch = GARCH(2, 0, 2)
    assert_equal(garch.name, "GARCH")
    garch = GARCH(2, 2, 2)
    assert_equal(garch.name, "GJR-GARCH")
    garch = GARCH(1, 0, 0, power=1.0)
    assert_equal(garch.name, "AVARCH")
    garch = GARCH(1, 0, 1, power=1.0)
    assert_equal(garch.name, "AVGARCH")
    garch = GARCH(1, 1, 1, power=1.0)
    assert_equal(garch.name, "TARCH/ZARCH")
    garch = GARCH(3, 0, 0, power=1.5)
    assert_equal(garch.name, "Power ARCH (power: 1.5)")
    assert "Power" in garch.__str__()
    garch = GARCH(1, 2, 1, power=1.5)
    assert_equal(garch.name, "Asym. Power GARCH (power: 1.5)")
    garch = GARCH(2, 0, 2, power=1.5)
    assert_equal(garch.name, "Power GARCH (power: 1.5)")


def test_ewma(setup):
    ewma = EWMAVariance()

    sv = ewma.starting_values(setup.resids)
    assert_equal(sv.shape[0], ewma.num_params)

    bounds = ewma.bounds(setup.resids)
    assert_equal(len(bounds), 0)
    var_bounds = ewma.variance_bounds(setup.resids)
    backcast = ewma.backcast(setup.resids)
    parameters = np.array([])

    names = ewma.parameter_names()
    names_target = []
    assert_equal(names, names_target)

    ewma.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)
    cond_var_direct = np.zeros_like(setup.sigma2)
    parameters = np.array([0.0, 0.06, 0.94])
    rec.garch_recursion(
        parameters,
        setup.resids**2.0,
        np.sign(setup.resids),
        cond_var_direct,
        1,
        0,
        1,
        setup.t,
        backcast,
        var_bounds,
    )

    assert_allclose(setup.sigma2 / cond_var_direct, np.ones_like(setup.sigma2))

    a, b = ewma.constraints()
    a_target = np.empty((0, 0))
    b_target = np.empty((0,))
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)
    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = ewma.simulate(parameters, setup.t, rng.simulate([]))
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    initial_value = 1.0

    sigma2 = np.zeros(setup.t + 500)
    data = np.zeros(setup.t + 500)
    sigma2[0] = initial_value
    data[0] = np.sqrt(initial_value)
    for t in range(1, setup.t + 500):
        sigma2[t] = 0.94 * sigma2[t - 1] + 0.06 * data[t - 1] ** 2.0
        data[t] = e[t] * np.sqrt(sigma2[t])

    data = data[500:]
    sigma2 = sigma2[500:]
    assert_almost_equal(data - sim_data[0] + 1.0, np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))

    assert_equal(ewma.num_params, 0)
    assert_equal(ewma.name, "EWMA/RiskMetrics")
    assert isinstance(ewma.__str__(), str)
    txt = ewma.__repr__()
    assert str(hex(id(ewma))) in txt


@pytest.mark.parametrize("initial_value", [None, 1.0])
def test_ewma_estimated(setup, initial_value):
    ewma = EWMAVariance(lam=None)

    sv = ewma.starting_values(setup.resids)
    assert sv == 0.94
    assert sv.shape[0] == ewma.num_params

    bounds = ewma.bounds(setup.resids)
    assert len(bounds) == 1
    assert bounds == [(0, 1)]

    ewma.variance_bounds(setup.resids)

    backcast = ewma.backcast(setup.resids)
    w = 0.94 ** np.arange(75)
    assert_almost_equal(backcast, np.sum((setup.resids[:75] ** 2) * (w / w.sum())))

    parameters = np.array([0.9234])

    var_bounds = ewma.variance_bounds(setup.resids)
    ewma.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)
    cond_var_direct = np.zeros_like(setup.sigma2)
    cond_var_direct[0] = backcast
    parameters = np.array([0, 1 - parameters[0], parameters[0]])
    rec.garch_recursion(
        parameters,
        setup.resids**2.0,
        np.sign(setup.resids),
        cond_var_direct,
        1,
        0,
        1,
        setup.t,
        backcast,
        var_bounds,
    )
    assert_allclose(setup.sigma2, cond_var_direct)
    assert_allclose(setup.sigma2 / cond_var_direct, np.ones_like(setup.sigma2))

    names = ewma.parameter_names()
    names_target = ["lam"]
    assert_equal(names, names_target)

    a, b = ewma.constraints()
    a_target = np.ones((1, 1))
    b_target = np.zeros((1,))
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)

    assert_equal(ewma.num_params, 1)
    assert_equal(ewma.name, "EWMA/RiskMetrics")
    assert isinstance(ewma.__str__(), str)
    txt = ewma.__repr__()
    assert str(hex(id(ewma))) in txt

    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    lam = parameters[-1]
    sim_data = ewma.simulate(
        [lam], setup.t, rng.simulate([]), initial_value=initial_value
    )
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    initial_value = 1.0

    sigma2 = np.zeros(setup.t + 500)
    data = np.zeros(setup.t + 500)
    sigma2[0] = initial_value
    data[0] = np.sqrt(initial_value)
    for t in range(1, setup.t + 500):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * data[t - 1] ** 2.0
        data[t] = e[t] * np.sqrt(sigma2[t])

    data = data[500:]
    sigma2 = sigma2[500:]
    assert_almost_equal(data - sim_data[0] + 1.0, np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))


@pytest.mark.parametrize("initial_value", [None, 1.0])
def test_riskmetrics(setup, initial_value):
    rm06 = RiskMetrics2006()

    sv = rm06.starting_values(setup.resids)
    assert_equal(sv.shape[0], rm06.num_params)

    bounds = rm06.bounds(setup.resids)
    assert_equal(len(bounds), 0)
    var_bounds = rm06.variance_bounds(setup.resids)
    backcast = rm06.backcast(setup.resids)
    assert_equal(backcast.shape[0], 14)
    parameters = np.array([])

    names = rm06.parameter_names()
    names_target = []
    assert_equal(names, names_target)

    # TODO: Test variance fit by RM06
    rm06.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)

    a, b = rm06.constraints()
    a_target = np.empty((0, 0))
    b_target = np.empty((0,))
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)

    # TODO: Test RM06 Simulation
    state = setup.rng.get_state()
    assert isinstance(state, tuple)
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = rm06.simulate(
        parameters, setup.t, rng.simulate([]), initial_value=initial_value
    )
    assert isinstance(sim_data, tuple)
    assert len(sim_data) == 2
    assert isinstance(sim_data[0], np.ndarray)
    assert isinstance(sim_data[1], np.ndarray)

    assert_equal(rm06.num_params, 0)
    assert_equal(rm06.name, "RiskMetrics2006")


@pytest.mark.parametrize("initial_value", [None, 2.0])
def test_egarch(setup, initial_value):
    egarch = EGARCH(p=1, o=1, q=1)

    sv = egarch.starting_values(setup.resids)
    assert_equal(sv.shape[0], egarch.num_params)

    bounds = egarch.bounds(setup.resids)
    assert_equal(len(bounds), egarch.num_params)
    const = np.log(10000.0)
    lnv = np.log(np.mean(setup.resids**2.0))
    assert_equal(bounds[0], (lnv - const, lnv + const))
    assert_equal(bounds[1], (-np.inf, np.inf))
    assert_equal(bounds[2], (-np.inf, np.inf))
    assert_equal(bounds[3], (0.0, 1.0))
    backcast = egarch.backcast(setup.resids)

    w = 0.94 ** np.arange(75)
    backcast_test = np.sum((setup.resids[:75] ** 2) * (w / w.sum()))
    assert_almost_equal(backcast, np.log(backcast_test))

    var_bounds = egarch.variance_bounds(setup.resids)
    parameters = np.array([0.1, 0.1, -0.1, 0.95])
    egarch.compute_variance(
        parameters, setup.resids, setup.sigma2, backcast, var_bounds
    )
    cond_var_direct = np.zeros_like(setup.sigma2)
    lnsigma2 = np.empty(setup.t)
    std_resids = np.empty(setup.t)
    abs_std_resids = np.empty(setup.t)
    rec.egarch_recursion(
        parameters,
        setup.resids,
        cond_var_direct,
        1,
        1,
        1,
        setup.t,
        backcast,
        var_bounds,
        lnsigma2,
        std_resids,
        abs_std_resids,
    )
    assert_allclose(setup.sigma2, cond_var_direct)

    a, b = egarch.constraints()
    a_target = np.vstack(np.array([[0, 0, 0, -1.0]]))
    b_target = np.array([-1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)

    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = egarch.simulate(
        parameters, setup.t, rng.simulate([]), initial_value=initial_value
    )
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    initial_value = 0.1 / (1 - 0.95)
    lnsigma2 = np.zeros(setup.t + 500)
    lnsigma2[0] = initial_value
    sigma2 = np.zeros(setup.t + 500)
    sigma2[0] = np.exp(lnsigma2[0])
    data = np.zeros(setup.t + 500)
    data[0] = np.sqrt(sigma2[0]) * e[0]
    norm_const = np.sqrt(2 / np.pi)
    for t in range(1, setup.t + 500):
        lnsigma2[t] = parameters[0]
        lnsigma2[t] += parameters[1] * (np.abs(e[t - 1]) - norm_const)
        lnsigma2[t] += parameters[2] * e[t - 1]
        lnsigma2[t] += parameters[3] * lnsigma2[t - 1]

    sigma2 = np.exp(lnsigma2)
    data = e * np.sqrt(sigma2)

    data = data[500:]
    sigma2 = sigma2[500:]

    assert_almost_equal(data - sim_data[0] + 1.0, np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))

    names = egarch.parameter_names()
    names_target = ["omega", "alpha[1]", "gamma[1]", "beta[1]"]
    assert_equal(names, names_target)
    assert_equal(egarch.name, "EGARCH")
    assert_equal(egarch.num_params, 4)

    assert_equal(egarch.p, 1)
    assert_equal(egarch.o, 1)
    assert_equal(egarch.q, 1)
    assert isinstance(egarch.__str__(), str)
    txt = egarch.__repr__()
    assert str(hex(id(egarch))) in txt

    with pytest.raises(ValueError):
        EGARCH(p=0, o=0, q=1)
    with pytest.raises(ValueError):
        EGARCH(p=1, o=1, q=-1)
    parameters = np.array([0.1, 0.1, -0.1, 1.05])
    with pytest.warns(InitialValueWarning):
        egarch.simulate(parameters, setup.t, rng.simulate([]))


def test_egarch_100(setup):
    egarch = EGARCH(p=1, o=0, q=0)

    sv = egarch.starting_values(setup.resids)
    assert_equal(sv.shape[0], egarch.num_params)

    backcast = egarch.backcast(setup.resids)
    w = 0.94 ** np.arange(75)
    backcast_test = np.sum((setup.resids[:75] ** 2) * (w / w.sum()))
    assert_almost_equal(backcast, np.log(backcast_test))

    var_bounds = egarch.variance_bounds(setup.resids)
    parameters = np.array([0.1, 0.4])
    egarch.compute_variance(
        parameters, setup.resids, setup.sigma2, backcast, var_bounds
    )
    cond_var_direct = np.zeros_like(setup.sigma2)
    lnsigma2 = np.empty(setup.t)
    std_resids = np.empty(setup.t)
    abs_std_resids = np.empty(setup.t)
    rec.egarch_recursion(
        parameters,
        setup.resids,
        cond_var_direct,
        1,
        0,
        0,
        setup.t,
        backcast,
        var_bounds,
        lnsigma2,
        std_resids,
        abs_std_resids,
    )
    assert_allclose(setup.sigma2, cond_var_direct)

    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = egarch.simulate(parameters, setup.t, rng.simulate([]))
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    initial_value = 0.1 / (1 - 0.95)
    lnsigma2 = np.zeros(setup.t + 500)
    lnsigma2[0] = initial_value
    sigma2 = np.zeros(setup.t + 500)
    sigma2[0] = np.exp(lnsigma2[0])
    data = np.zeros(setup.t + 500)
    data[0] = np.sqrt(sigma2[0]) * e[0]
    norm_const = np.sqrt(2 / np.pi)
    for t in range(1, setup.t + 500):
        lnsigma2[t] = parameters[0]
        lnsigma2[t] += parameters[1] * (np.abs(e[t - 1]) - norm_const)

    sigma2 = np.exp(lnsigma2)
    data = e * np.sqrt(sigma2)

    data = data[500:]
    sigma2 = sigma2[500:]

    assert_almost_equal(data - sim_data[0] + 1.0, np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))


def test_fixed_variance(setup):
    variance = np.arange(1000.0) + 1.0
    fv = FixedVariance(variance)
    fv.start, fv.stop = 0, 1000
    parameters = np.array([2.0])
    resids = setup.resids
    sigma2 = np.empty_like(resids)
    backcast = fv.backcast(resids)
    var_bounds = fv.variance_bounds(resids, 2.0)
    fv.compute_variance(parameters, resids, sigma2, backcast, var_bounds)
    sv = fv.starting_values(resids)
    cons = fv.constraints()
    bounds = fv.bounds(resids)
    assert_allclose(sigma2, 2.0 * variance)
    assert_allclose(sv, (resids / np.sqrt(variance)).var())
    assert var_bounds.shape == (resids.shape[0], 2)
    assert fv.num_params == 1
    assert fv.parameter_names() == ["scale"]
    assert fv.name == "Fixed Variance"
    assert_equal(cons[0], np.ones((1, 1)))
    assert_equal(cons[1], np.zeros(1))
    assert_equal(bounds[0][0], sv[0] / 100000.0)

    sigma2 = np.empty(500)
    fv.start = 250
    fv.stop = 750
    fv.compute_variance(parameters, resids[250:750], sigma2, backcast, var_bounds)
    assert_allclose(sigma2, 2.0 * variance[250:750])

    fv = FixedVariance(variance, unit_scale=True)
    fv.start, fv.stop = 0, 1000
    sigma2 = np.empty_like(resids)
    parameters = np.empty(0)
    fv.compute_variance(parameters, resids, sigma2, backcast, var_bounds)
    sv = fv.starting_values(resids)
    cons = fv.constraints()
    bounds = fv.bounds(resids)

    assert_allclose(sigma2, variance)
    assert_allclose(sv, np.empty(0))
    assert fv.num_params == 0
    assert fv.parameter_names() == []
    assert fv.name == "Fixed Variance (Unit Scale)"
    assert_equal(cons[0], np.empty((0, 0)))
    assert_equal(cons[1], np.empty(0))
    assert bounds == []
    rng = Normal()
    with pytest.raises(NotImplementedError):
        fv.simulate(parameters, 1000, rng.simulate([]))

    fv = FixedVariance(variance, unit_scale=True)
    fv.start, fv.stop = 123, 731
    sigma2 = np.empty_like(resids)
    parameters = np.empty(0)
    assert fv.start == 123
    assert fv.stop == 731
    with pytest.raises(ValueError):
        fv.compute_variance(parameters, resids, sigma2, backcast, var_bounds)


@pytest.mark.parametrize("initial_value", [None, 1.0])
def test_midas_symmetric(setup, initial_value):
    midas = MIDASHyperbolic()

    sv = midas.starting_values(setup.resids)
    assert_equal(sv.shape[0], midas.num_params)

    bounds = midas.bounds(setup.resids)
    assert_equal(bounds[0], (0.0, 10.0 * np.mean(setup.resids**2.0)))
    assert_equal(bounds[1], (0.0, 1.0))
    assert_equal(bounds[2], (0.0, 1.0))
    backcast = midas.backcast(setup.resids)
    w = 0.94 ** np.arange(75)
    assert_almost_equal(backcast, np.sum((setup.resids[:75] ** 2) * (w / w.sum())))
    var_bounds = midas.variance_bounds(setup.resids)
    parameters = np.array([0.1, 0.9, 0.4])
    midas.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)
    cond_var_direct = np.zeros_like(setup.sigma2)
    weights = midas._weights(parameters)
    theta = parameters[-1]
    j = np.arange(1, 22 + 1)
    direct_weights = gamma(j + theta) / (gamma(j + 1) * gamma(theta))
    direct_weights = direct_weights / direct_weights.sum()
    assert_allclose(weights, direct_weights)
    resids = setup.resids
    direct_params = parameters.copy()
    direct_params[-1] = 0.0  # gamma, strip theta
    recpy.midas_recursion_python(
        direct_params,
        weights,
        resids,
        cond_var_direct,
        setup.t,
        backcast,
        var_bounds,
    )
    assert_allclose(setup.sigma2, cond_var_direct)

    a, b = midas.constraints()
    a_target = np.zeros((5, 3))
    a_target[0, 0] = 1
    a_target[1, 1] = 1
    a_target[2, 1] = -1
    a_target[3, 2] = 1
    a_target[4, 2] = -1
    b_target = np.array([0.0, 0.0, -1.0, 0.0, -1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)
    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = midas.simulate(
        parameters, setup.t, rng.simulate([]), initial_value=initial_value
    )
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    initial_value = 1.0
    sigma2 = np.zeros(setup.t + 500)
    data = np.zeros(setup.t + 500)
    sigma2[:22] = initial_value
    omega, alpha = parameters[:2]
    for t in range(22, setup.t + 500):
        sigma2[t] = omega
        for i in range(22):
            shock = initial_value if t - i - 1 < 22 else data[t - i - 1] ** 2.0
            sigma2[t] += alpha * weights[i] * shock
        data[t] = e[t] * np.sqrt(sigma2[t])
    data = data[500:]
    sigma2 = sigma2[500:]
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))
    assert_almost_equal(data / sim_data[0], np.ones_like(data))

    names = midas.parameter_names()
    names_target = ["omega", "alpha", "theta"]
    assert_equal(names, names_target)

    assert isinstance(midas.__str__(), str)
    txt = midas.__repr__()
    assert str(hex(id(midas))) in txt

    assert_equal(midas.name, "MIDAS Hyperbolic")
    assert_equal(midas.num_params, 3)
    assert_equal(midas.m, 22)

    with pytest.warns(InitialValueWarning):
        parameters = np.array([0.1, 1.1, 0.4])
        midas.simulate(parameters, setup.t, rng.simulate([]))


def test_midas_asymmetric(setup):
    midas = MIDASHyperbolic(33, asym=True)

    sv = midas.starting_values(setup.resids)
    assert_equal(sv.shape[0], midas.num_params)

    bounds = midas.bounds(setup.resids)
    assert_equal(bounds[0], (0.0, 10.0 * np.mean(setup.resids**2.0)))
    assert_equal(bounds[1], (0.0, 1.0))
    assert_equal(bounds[2], (-1.0, 2.0))
    assert_equal(bounds[3], (0.0, 1.0))
    backcast = midas.backcast(setup.resids)
    w = 0.94 ** np.arange(75)
    assert_almost_equal(backcast, np.sum((setup.resids[:75] ** 2) * (w / w.sum())))
    var_bounds = midas.variance_bounds(setup.resids)
    parameters = np.array([0.1, 0.3, 1.2, 0.4])
    midas.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)
    cond_var_direct = np.zeros_like(setup.sigma2)
    weights = midas._weights(parameters)
    wlen = len(weights)
    theta = parameters[-1]
    j = np.arange(1, wlen + 1)
    direct_weights = gammaln(j + theta) - gammaln(j + 1) - gammaln(theta)
    direct_weights = np.exp(direct_weights)
    direct_weights = direct_weights / direct_weights.sum()
    assert_allclose(direct_weights, weights)
    resids = setup.resids
    direct_params = parameters[:3].copy()
    recpy.midas_recursion_python(
        direct_params,
        weights,
        resids,
        cond_var_direct,
        setup.t,
        backcast,
        var_bounds,
    )
    assert_allclose(setup.sigma2, cond_var_direct)

    a, b = midas.constraints()
    a_target = np.zeros((5, 4))
    a_target[0, 0] = 1
    a_target[1, 1] = 1
    a_target[1, 2] = 1
    a_target[2, 1] = -1
    a_target[2, 2] = -0.5
    a_target[3, 3] = 1
    a_target[4, 3] = -1
    b_target = np.array([0.0, 0.0, -1.0, 0.0, -1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)
    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    burn = wlen
    sim_data = midas.simulate(parameters, setup.t, rng.simulate([]), burn=burn)
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + burn)
    initial_value = 1.0
    sigma2 = np.zeros(setup.t + burn)
    data = np.zeros(setup.t + burn)
    sigma2[:wlen] = initial_value
    omega, alpha, gamma = parameters[:3]
    for t in range(wlen, setup.t + burn):
        sigma2[t] = omega
        for i in range(wlen):
            if t - i - 1 < wlen:
                shock = initial_value
                coeff = (alpha + 0.5 * gamma) * weights[i]
            else:
                shock = data[t - i - 1] ** 2.0
                coeff = (alpha + gamma * (data[t - i - 1] < 0)) * weights[i]
            sigma2[t] += coeff * shock
        data[t] = e[t] * np.sqrt(sigma2[t])
    data = data[burn:]
    sigma2 = sigma2[burn:]
    assert_almost_equal(data / sim_data[0], np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))

    names = midas.parameter_names()
    names_target = ["omega", "alpha", "gamma", "theta"]
    assert_equal(names, names_target)

    assert isinstance(midas.__str__(), str)
    txt = midas.__repr__()
    assert str(hex(id(midas))) in txt

    assert_equal(midas.name, "MIDAS Hyperbolic")
    assert_equal(midas.num_params, 4)
    assert_equal(midas.m, 33)

    with pytest.warns(InitialValueWarning):
        parameters = np.array([0.1, 0.3, 1.6, 0.4])
        midas.simulate(parameters, setup.t, rng.simulate([]))


@pytest.mark.parametrize("initial_value", [None, 1.0])
def test_figarch(setup, initial_value):
    trunc_lag = 750
    figarch = FIGARCH(truncation=trunc_lag)

    sv = figarch.starting_values(setup.resids)
    assert_equal(sv.shape[0], figarch.num_params)

    bounds = figarch.bounds(setup.resids)
    assert_equal(bounds[0], (0.0, 10.0 * np.mean(setup.resids**2.0)))
    assert_equal(bounds[1], (0.0, 0.5))
    assert_allclose(np.array(bounds[2]), np.array([0.0, 1 - EPS_HALF]))
    assert_allclose(np.array(bounds[3]), np.array([0.0, 1 - EPS_HALF]))
    assert len(bounds) == figarch.num_params

    backcast = figarch.backcast(setup.resids)
    w = 0.94 ** np.arange(75)
    assert_almost_equal(backcast, np.sum((setup.resids[:75] ** 2) * (w / w.sum())))
    var_bounds = figarch.variance_bounds(setup.resids)
    parameters = np.array([1, 0.2, 0.4, 0.2])
    figarch.compute_variance(
        parameters, setup.resids, setup.sigma2, backcast, var_bounds
    )

    cond_var_direct = np.zeros_like(setup.sigma2)
    fresids = setup.resids**2
    p = q = 1
    nobs = setup.resids.shape[0]
    recpy.figarch_recursion_python(
        parameters,
        fresids,
        cond_var_direct,
        p,
        q,
        nobs,
        trunc_lag,
        backcast,
        var_bounds,
    )
    assert_allclose(setup.sigma2, cond_var_direct)

    a, b = figarch.constraints()
    a_target = np.zeros((7, 4))
    a_target[0, 0] = 1
    a_target[1, 1] = 1
    a_target[2, 1] = -2
    a_target[2, 2] = -1
    a_target[3, 2] = 1
    a_target[4, 2] = -1
    a_target[5, 3] = 1
    a_target[6, 1] = 1
    a_target[6, 2] = 1
    a_target[6, 3] = -1
    b_target = np.array([0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)

    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = figarch.simulate(parameters, setup.t, rng.simulate([]))
    setup.rng.set_state(state)
    lam = rec.figarch_weights(parameters[1:], p, q, trunc_lag)
    lam_rev = lam[::-1]
    omega_tilde = parameters[0] / (1 - parameters[-1])
    initial_value = omega_tilde / (1 - lam.sum())
    e = setup.rng.standard_normal(trunc_lag + setup.t + 500)
    sigma2 = np.zeros(trunc_lag + setup.t + 500)
    data = np.zeros(trunc_lag + setup.t + 500)
    sigma2[:trunc_lag] = initial_value
    data[:trunc_lag] = np.sqrt(sigma2[:trunc_lag]) * e[:trunc_lag]

    for t in range(trunc_lag, trunc_lag + setup.t + 500):
        sigma2[t] = omega_tilde + lam_rev.dot(data[t - trunc_lag : t] ** 2)
        data[t] = e[t] * np.sqrt(sigma2[t])
    data = data[trunc_lag + 500 :]
    sigma2 = sigma2[trunc_lag + 500 :]
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))
    assert_almost_equal(data / sim_data[0], np.ones_like(data))

    names = figarch.parameter_names()
    names_target = ["omega", "phi", "d", "beta"]
    assert_equal(names, names_target)

    assert isinstance(figarch.__str__(), str)
    txt = figarch.__repr__()
    assert str(hex(id(figarch))) in txt

    assert_equal(figarch.name, "FIGARCH")
    assert_equal(figarch.num_params, 4)
    assert_equal(figarch.truncation, trunc_lag)

    params = np.array([0.1, 1.01, 0.4, 1.005])
    with pytest.warns(InitialValueWarning):
        figarch.simulate(params, 20, rng.simulate([]))


def test_figarch_no_phi(setup):
    trunc_lag = 333
    figarch = FIGARCH(p=0, truncation=trunc_lag)

    sv = figarch.starting_values(setup.resids)
    assert_equal(sv.shape[0], figarch.num_params)

    bounds = figarch.bounds(setup.resids)
    assert_equal(bounds[0], (0.0, 10.0 * np.mean(setup.resids**2.0)))
    assert_allclose(np.array(bounds[1]), np.array([0.0, 1 - EPS_HALF]))
    assert_allclose(np.array(bounds[2]), np.array([0.0, 1 - EPS_HALF]))
    assert len(bounds) == figarch.num_params

    a, b = figarch.constraints()
    a_target = np.zeros((5, 3))
    a_target[0, 0] = 1
    a_target[1, 1] = 1
    a_target[2, 1] = -1
    a_target[3, 2] = 1
    a_target[4, 1] = 1
    a_target[4, 2] = -1
    b_target = np.array([0.0, 0.0, -1.0, 0.0, 0.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)


def test_figarch_no_beta(setup):
    figarch = FIGARCH(q=0)

    sv = figarch.starting_values(setup.resids)
    assert_equal(sv.shape[0], figarch.num_params)

    bounds = figarch.bounds(setup.resids)
    assert_equal(bounds[0], (0.0, 10.0 * np.mean(setup.resids**2.0)))
    assert_equal(bounds[1], (0.0, 0.5))
    assert_allclose(np.array(bounds[2]), np.array([0.0, 1 - EPS_HALF]))
    assert len(bounds) == figarch.num_params

    a, b = figarch.constraints()
    a_target = np.zeros((5, 3))
    a_target[0, 0] = 1
    a_target[1, 1] = 1
    a_target[2, 1] = -2
    a_target[2, 2] = -1
    a_target[3, 2] = 1
    a_target[4, 2] = -1
    b_target = np.array([0.0, 0.0, -1.0, 0.0, -1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)


def test_figarch_no_phi_beta(setup):
    figarch = FIGARCH(p=0, q=0, power=0.7)

    sv = figarch.starting_values(setup.resids)
    assert_equal(sv.shape[0], figarch.num_params)

    backcast = figarch.backcast(setup.resids)
    assert_allclose(np.sqrt(backcast) ** 0.7, figarch.backcast_transform(backcast))

    bounds = figarch.bounds(setup.resids)
    assert_equal(bounds[0], (0.0, 10.0 * np.mean(np.abs(setup.resids) ** 0.7)))
    assert_allclose(np.array(bounds[1]), np.array([0.0, 1 - EPS_HALF]))
    assert len(bounds) == figarch.num_params

    a, b = figarch.constraints()
    a_target = np.zeros((3, 2))
    a_target[0, 0] = 1
    a_target[1, 1] = 1
    a_target[2, 1] = -1
    b_target = np.array([0.0, 0.0, -1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)


def test_figarch_errors(setup):
    with pytest.raises(ValueError):
        FIGARCH(truncation=-1)
    with pytest.raises(ValueError):
        FIGARCH(truncation="apple")
    with pytest.raises(ValueError):
        FIGARCH(p=2)
    with pytest.raises(ValueError):
        FIGARCH(q=-1)
    with pytest.raises(ValueError):
        FIGARCH(power=0)
    with pytest.raises(ValueError):
        FIGARCH(power=-0.25)


def test_figarch_edge_cases(setup):
    figarch = FIGARCH(power=0.5)
    assert "power" in str(figarch)
    figarch = FIGARCH()
    assert "power" not in str(figarch)


@pytest.mark.parametrize("p", [0, 1])
@pytest.mark.parametrize("q", [0, 1])
@pytest.mark.parametrize("power", [0.5, 1.0, 2.0])
def test_figarch_str(setup, p, q, power):
    figarch = FIGARCH(p=p, q=q, power=power)
    s = str(figarch).lower()
    assert "arch" in s
    assert f"q: {q}" in s
    assert f"p: {p}" in s
    if power not in (1.0, 2.0):
        assert f"power: {power:0.1f}" in s


@pytest.mark.parametrize("initial_value", [None, 1.0])
def test_aparch(setup, initial_value):
    aparch = APARCH()
    sv = aparch.starting_values(setup.resids)
    assert_equal(sv.shape[0], aparch.num_params)

    bounds = aparch.bounds(setup.resids)
    upper = max(np.mean(setup.resids**2.0), np.mean(np.abs(setup.resids) ** 0.5))
    assert_equal(bounds[0], (0.0, 10.0 * upper))
    assert_equal(bounds[1], (0.0, 1.0))
    assert_equal(bounds[2], (-0.9997, 0.9997))
    assert_equal(bounds[3], (0.0, 1.0))
    assert_equal(bounds[4], (0.05, 4.0))
    backcast = aparch.backcast(setup.resids)
    w = 0.94 ** np.arange(75)
    assert_almost_equal(backcast, np.sum((setup.resids[:75] ** 2) * (w / w.sum())))
    var_bounds = aparch.variance_bounds(setup.resids)
    parameters = np.array([0.1, 0.1, -0.5, 0.8, 1.2])
    aparch.compute_variance(
        parameters, setup.resids, setup.sigma2, backcast, var_bounds
    )
    cond_var_direct = np.zeros_like(setup.sigma2)
    rec.aparch_recursion(
        parameters,
        setup.resids,
        np.abs(setup.resids),
        cond_var_direct,
        cond_var_direct.copy(),
        1,
        1,
        1,
        setup.t,
        backcast,
        var_bounds,
    )
    assert_allclose(setup.sigma2, cond_var_direct)

    a, b = aparch.constraints()
    a_target = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, -1.0],
            [0, -1, 0, -1, 0],
        ]
    )
    b_target = np.array([0.0, 0.0, -0.9997, -0.9997, 0.0, 0.05, -4.0, -1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)
    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = aparch.simulate(
        parameters, setup.t, rng.simulate([]), initial_value=initial_value
    )
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    initial_value = 1.0
    sigma_delta = np.zeros(setup.t + 500)
    sigma2 = np.zeros(setup.t + 500)
    data = np.zeros(setup.t + 500)
    delta = parameters[-1]
    for t in range(setup.t + 500):
        sigma_delta[t] = parameters[0]
        shock = initial_value if t == 0 else np.abs(data[t - 1])
        shock -= 0 if t == 0 else parameters[2] * data[t - 1]
        sigma_delta[t] += parameters[1] * shock**delta
        lagged_value = initial_value if t == 0 else sigma_delta[t - 1]
        sigma_delta[t] += parameters[3] * lagged_value
        sigma2[t] = sigma_delta[t] ** (2 / delta)
        data[t] = e[t] * np.sqrt(sigma2[t])
    data = data[500:]
    sigma2 = sigma2[500:]
    assert_almost_equal(data / sim_data[0], np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))

    names = aparch.parameter_names()
    names_target = ["omega", "alpha[1]", "gamma[1]", "beta[1]", "delta"]
    assert_equal(names, names_target)

    assert isinstance(aparch.__str__(), str)
    assert "p: 1, o: 1, q: 1" in str(aparch)
    txt = aparch.__repr__()
    assert str(hex(id(aparch))) in txt

    assert_equal(aparch.name, "APARCH")
    assert_equal(aparch.num_params, 5)
    assert_equal(aparch._delta, np.nan)
    assert aparch.p == aparch.o == aparch.q == 1

    parameters[1] = 0.9
    with pytest.warns(InitialValueWarning):
        aparch.simulate(parameters, setup.t, rng.simulate([]))


def test_aparch_delta(setup):
    delta = 0.7
    aparch = APARCH(delta=0.7)
    sv = aparch.starting_values(setup.resids)
    assert_equal(sv.shape[0], aparch.num_params)

    bounds = aparch.bounds(setup.resids)
    upper = max(np.mean(setup.resids**2.0), np.mean(np.abs(setup.resids) ** 0.5))
    assert_equal(bounds[0], (0.0, 10.0 * upper))
    assert_equal(bounds[1], (0.0, 1.0))
    assert_equal(bounds[2], (-0.9997, 0.9997))
    assert_equal(bounds[3], (0.0, 1.0))
    backcast = aparch.backcast(setup.resids)
    w = 0.94 ** np.arange(75)
    assert_almost_equal(backcast, np.sum((setup.resids[:75] ** 2) * (w / w.sum())))
    var_bounds = aparch.variance_bounds(setup.resids)
    parameters = np.array([0.1, 0.1, -0.5, 0.8])
    aparch.compute_variance(
        parameters, setup.resids, setup.sigma2, backcast, var_bounds
    )
    cond_var_direct = np.zeros_like(setup.sigma2)
    rec_parameters = np.array([0.1, 0.1, -0.5, 0.8, delta])
    rec.aparch_recursion(
        rec_parameters,
        setup.resids,
        np.abs(setup.resids),
        cond_var_direct,
        cond_var_direct.copy(),
        1,
        1,
        1,
        setup.t,
        backcast,
        var_bounds,
    )
    assert_allclose(setup.sigma2, cond_var_direct)

    a, b = aparch.constraints()
    a_target = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
            [0, -1, 0, -1],
        ]
    )
    b_target = np.array([0.0, 0.0, -0.9997, -0.9997, 0.0, -1.0])
    assert_array_equal(a, a_target)
    assert_array_equal(b, b_target)
    state = setup.rng.get_state()
    rng = Normal(seed=RandomState())
    rng.generator.set_state(state)
    sim_data = aparch.simulate(parameters, setup.t, rng.simulate([]))
    setup.rng.set_state(state)
    e = setup.rng.standard_normal(setup.t + 500)
    initial_value = 1.0
    sigma_delta = np.zeros(setup.t + 500)
    sigma2 = np.zeros(setup.t + 500)
    data = np.zeros(setup.t + 500)
    for t in range(setup.t + 500):
        sigma_delta[t] = parameters[0]
        shock = initial_value if t == 0 else np.abs(data[t - 1])
        shock -= 0 if t == 0 else parameters[2] * data[t - 1]
        sigma_delta[t] += parameters[1] * shock**delta
        lagged_value = initial_value if t == 0 else sigma_delta[t - 1]
        sigma_delta[t] += parameters[3] * lagged_value
        sigma2[t] = sigma_delta[t] ** (2 / delta)
        data[t] = e[t] * np.sqrt(sigma2[t])
    data = data[500:]
    sigma2 = sigma2[500:]
    assert_almost_equal(data / sim_data[0], np.ones_like(data))
    assert_almost_equal(sigma2 / sim_data[1], np.ones_like(sigma2))

    names = aparch.parameter_names()
    names_target = ["omega", "alpha[1]", "gamma[1]", "beta[1]"]
    assert_equal(names, names_target)

    assert isinstance(aparch.__str__(), str)
    assert "p: 1, o: 1, q: 1, delta: 0.7" in str(aparch)
    txt = aparch.__repr__()
    assert str(hex(id(aparch))) in txt

    assert_equal(aparch.name, "APARCH")
    assert not aparch._est_delta
    assert_equal(aparch.num_params, 4)
    assert_equal(aparch._delta, 0.7)
    assert aparch.p == aparch.o == aparch.q == 1


def test_aparch_against_garch(setup):
    garch = GARCH()
    aparch = APARCH(o=0, delta=2.0)
    assert "o:" not in str(aparch)
    assert "Common Asym :" not in str(aparch)

    sv = garch.starting_values(setup.resids)
    assert_equal(sv.shape[0], aparch.num_params)

    backcast = aparch.backcast(setup.resids)
    var_bounds = aparch.variance_bounds(setup.resids)
    parameters = np.array([0.1, 0.1, 0.8])
    garch.compute_variance(parameters, setup.resids, setup.sigma2, backcast, var_bounds)
    aparch_sigma2 = setup.sigma2.copy()
    aparch.compute_variance(
        parameters, setup.resids, aparch_sigma2, backcast, var_bounds
    )
    assert_allclose(setup.sigma2, aparch_sigma2)


def test_aparch_exceptions(setup):
    with pytest.raises(ValueError, match="delta must be between 0.05"):
        APARCH(delta=0.0)
    with pytest.raises(ValueError, match="delta must be between 0.05"):
        APARCH(delta=4.1)
    with pytest.raises(TypeError, match="delta must be convertible"):
        APARCH(delta="a")
    with pytest.raises(ValueError, match="All lags lengths must be"):
        APARCH(p=0)
    with pytest.raises(ValueError, match="o must be <= p"):
        APARCH(p=1, o=2)


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("o", [0, 1, 2])
@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("common_asym", [True, False])
@pytest.mark.parametrize("delta", [1.7, None])
def test_aparch_configs(p, o, q, delta, common_asym):
    if o > p or p == 0:
        pytest.skip("o < p or p is 0")
    rs = np.random.RandomState(19992131)
    rng = Normal(seed=rs)

    aparch = APARCH(p=p, o=o, q=q, delta=delta, common_asym=common_asym)
    omega = 1
    a = [0.15 / p] * p
    o = 1 if (common_asym and o > 0) else o
    g = [-0.6] * o
    params = np.r_[omega, a, g]
    if q > 0:
        params = np.r_[params, [0.8 / q] * q]
    if delta is None:
        params = np.r_[params, 1.7]
    data, _ = aparch.simulate(params, 2500, rng=rng.simulate([]))
    mod = ZeroMean(data, volatility=aparch)
    res = mod.fit(disp="off")
    assert res.params.shape[0] == aparch.num_params
    assert aparch.common_asym == (common_asym and o > 0)


def test_figarch_weights():
    params = np.array([0.2, 0.4, 0.2])
    lam_py = recpy.figarch_weights_python(params, 1, 1, 1000)
    lam_nb = recpy.figarch_weights(params, 1, 1, 1000)
    lam_cy = rec.figarch_weights(params, 1, 1, 1000)
    assert_allclose(lam_py, lam_nb)
    assert_allclose(lam_py, lam_cy)
