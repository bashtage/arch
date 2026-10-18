"""String options of the bootstraps and multiple comparisons ignore case"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.bootstrap import (
    MCS,
    SPA,
    CircularBlockBootstrap,
    IIDBootstrap,
    MovingBlockBootstrap,
    RealityCheck,
    StationaryBootstrap,
    StepM,
)

BAD_VALUES = [
    pytest.param("bogus", ValueError, id="bogus"),
    pytest.param("", ValueError, id="empty"),
    pytest.param(None, TypeError, id="none"),
    pytest.param(1, TypeError, id="int"),
]


def variants(value):
    return [value.upper(), value.capitalize(), value.title()]


@pytest.fixture(scope="module")
def data():
    return np.random.default_rng(7).standard_normal((120, 2))


def mean_func(x, params=None, state=None):
    if state is not None:
        x = x - x.mean(0) + params + 0.1 * state.standard_normal(x.shape)
    elif params is not None:
        x = x - x.mean(0) + params
    return x.mean(0)


def conf_int(data, **kwargs):
    bs = IIDBootstrap(data, seed=11)
    return bs.conf_int(mean_func, reps=50, studentize_reps=10, **kwargs)


CI_METHODS = ["basic", "percentile", "studentized", "norm", "bc", "bca"]
CI_ALIASES = {"var": "norm", "cov": "norm", "debiased": "bc", "bias-corrected": "bc"}


@pytest.mark.parametrize("method", CI_METHODS + list(CI_ALIASES))
def test_conf_int_method_case(data, method):
    expected = conf_int(data, method=method)
    for spelling in variants(method):
        assert_allclose(conf_int(data, method=spelling), expected)


@pytest.mark.parametrize("alias", list(CI_ALIASES))
def test_conf_int_method_aliases(data, alias):
    # The documented aliases give the same interval as the method they name
    assert_allclose(
        conf_int(data, method=alias), conf_int(data, method=CI_ALIASES[alias])
    )


@pytest.mark.parametrize("tail", ["two", "upper", "lower"])
def test_conf_int_tail_case(data, tail):
    expected = conf_int(data, tail=tail)
    for spelling in variants(tail):
        assert_allclose(conf_int(data, tail=spelling), expected)
    # A one-sided interval is unbounded on the other side
    if tail == "lower":
        assert np.all(np.isinf(expected[1]))
    elif tail == "upper":
        assert np.all(np.isneginf(expected[0]))


SAMPLING = ["nonparametric", "parametric", "semi", "semi-parametric", "semiparametric"]


@pytest.mark.parametrize("sampling", SAMPLING)
def test_conf_int_sampling_case(data, sampling):
    expected = conf_int(data, sampling=sampling)
    for spelling in variants(sampling):
        assert_allclose(conf_int(data, sampling=spelling), expected)


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
@pytest.mark.parametrize(
    ("param", "listing"),
    [
        ("method", "'basic', 'percentile', 'studentized', 'norm', 'var', 'cov'"),
        ("tail", "'two', 'lower' or 'upper'"),
        ("sampling", "'nonparametric', 'parametric', 'semi', 'semi-parametric'"),
    ],
)
def test_conf_int_invalid(data, param, listing, bad, exc):
    bs = IIDBootstrap(data, seed=11)
    state = bs.generator.bit_generator.state
    with pytest.raises(exc, match=rf"{param} must be one of {listing}.*; got "):
        bs.conf_int(mean_func, reps=50, **{param: bad})
    # Invalid options are rejected before the bootstrap is run
    assert bs.generator.bit_generator.state == state
    assert bs._results is None


BOOTSTRAPS = {
    "stationary": StationaryBootstrap,
    "sb": StationaryBootstrap,
    "circular": CircularBlockBootstrap,
    "cbb": CircularBlockBootstrap,
    "moving block": MovingBlockBootstrap,
    "mbb": MovingBlockBootstrap,
    "moving_block": MovingBlockBootstrap,
}


@pytest.fixture(scope="module")
def losses():
    values = np.random.default_rng(3).standard_normal((100, 4)) ** 2
    return pd.DataFrame(values, columns=list("abcd"))


@pytest.mark.parametrize("name", list(BOOTSTRAPS))
@pytest.mark.parametrize("procedure", [MCS, SPA, RealityCheck, StepM])
def test_multiple_comparison_bootstrap_case(losses, name, procedure):
    def build(bootstrap):
        if procedure is MCS:
            return MCS(losses, 0.1, reps=20, block_size=5, bootstrap=bootstrap, seed=0)
        return procedure(
            losses["a"],
            losses[["b", "c", "d"]],
            block_size=5,
            reps=20,
            bootstrap=bootstrap,
            seed=0,
        )

    expected = build(name)
    assert type(expected.bootstrap) is BOOTSTRAPS[name]
    expected.compute()
    for spelling in variants(name):
        other = build(spelling)
        assert type(other.bootstrap) is BOOTSTRAPS[name]
        other.compute()
        if procedure is MCS:
            assert_allclose(other.pvalues, expected.pvalues)
        elif procedure is StepM:
            assert other.superior_models == expected.superior_models
        else:
            assert_allclose(other.pvalues, expected.pvalues)


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
@pytest.mark.parametrize("procedure", [MCS, SPA, RealityCheck, StepM])
def test_multiple_comparison_bootstrap_invalid(losses, procedure, bad, exc):
    match = r"bootstrap must be one of 'stationary', 'sb', 'circular', 'cbb', "
    if procedure is MCS:
        args = (losses, 0.1)
    else:
        args = (losses["a"], losses[["b", "c", "d"]])
    with pytest.raises(exc, match=match + r"'moving block' or 'mbb'; got "):
        procedure(*args, bootstrap=bad)


@pytest.mark.parametrize(
    ("method", "canonical"),
    [("R", "r"), ("r", "r"), ("max", "max"), ("MAX", "max"), ("Max", "max")],
)
def test_mcs_method_canonical_value(losses, method, canonical):
    mcs = MCS(losses, 0.1, reps=20, block_size=5, method=method, seed=0)
    assert mcs.method == canonical
    mcs.compute()
    reference = MCS(losses, 0.1, reps=20, block_size=5, method=canonical, seed=0)
    reference.compute()
    assert_allclose(mcs.pvalues, reference.pvalues)


def test_mcs_method_set_after_construction(losses):
    reference = MCS(losses, 0.1, reps=20, block_size=5, method="max", seed=0)
    reference.compute()
    mcs = MCS(losses, 0.1, reps=20, block_size=5, method="r", seed=0)
    mcs.method = "MAX"
    mcs.compute()
    assert_allclose(mcs.pvalues, reference.pvalues)


@pytest.mark.parametrize("pvalue_type", ["lower", "consistent", "upper"])
def test_better_models_pvalue_type_case(losses, pvalue_type):
    spa = SPA(losses["a"], losses[["b", "c", "d"]], block_size=5, reps=50, seed=0)
    spa.compute()
    expected = spa.better_models(0.5, pvalue_type=pvalue_type)
    for spelling in variants(pvalue_type):
        assert_allclose(spa.better_models(0.5, pvalue_type=spelling), expected)


@pytest.mark.parametrize(("bad", "exc"), BAD_VALUES)
def test_better_models_pvalue_type_invalid(losses, bad, exc):
    spa = SPA(losses["a"], losses[["b", "c", "d"]], block_size=5, reps=50, seed=0)
    spa.compute()
    match = r"pvalue_type must be one of 'lower', 'consistent' or 'upper'; got "
    with pytest.raises(exc, match=match):
        spa.better_models(0.5, pvalue_type=bad)
