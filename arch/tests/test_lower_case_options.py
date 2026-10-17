"""
Lower case is the only spelling of a string option that arch shows to users

The options are not case sensitive, but the documented spelling, the annotations,
the defaults, the values in the error messages and the values that objects expose
are all lower case.
"""

from __future__ import annotations

import importlib
import inspect
import re
import typing
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.bootstrap import MCS, SPA, IIDBootstrap
from arch.covariance.kernel import get_kernel_estimator
from arch.covariance.var import PreWhitenedRecolored
from arch.unitroot import ADF, KPSS, PhillipsPerron
from arch.unitroot._phillips_ouliaris import (
    phillips_ouliaris_cv,
    phillips_ouliaris_pval,
)
from arch.unitroot.cointegration import DynamicOLS, engle_granger, phillips_ouliaris
from arch.unitroot.unitroot import mackinnoncrit, mackinnonp
from arch.univariate import ARCHInMean, arch_model
from arch.univariate.volatility import GARCH
from arch.utility.timeseries import add_trend

MODULES = (
    "arch.univariate",
    "arch.univariate.base",
    "arch.univariate.mean",
    "arch.univariate.volatility",
    "arch.bootstrap",
    "arch.bootstrap.base",
    "arch.bootstrap.multiple_comparison",
    "arch.unitroot",
    "arch.unitroot.unitroot",
    "arch.unitroot.cointegration",
    "arch.unitroot._engle_granger",
    "arch.unitroot._phillips_ouliaris",
    "arch.covariance.kernel",
    "arch.covariance.var",
    "arch.utility.timeseries",
)


def public_objects() -> list[tuple[str, object]]:
    """Public classes and functions of the arch modules, and their public methods"""
    found: dict[int, tuple[str, object]] = {}

    def add(name: str, obj: object) -> None:
        found.setdefault(id(obj), (name, obj))

    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name, obj in vars(module).items():
            if name.startswith("_"):
                continue
            if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            if not getattr(obj, "__module__", "").startswith("arch."):
                continue
            add(f"{obj.__module__}.{obj.__qualname__}", obj)
            if inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if attr.startswith("_") and attr != "__init__":
                        continue
                    func = member
                    if isinstance(member, staticmethod | classmethod):
                        func = member.__func__
                    if inspect.isfunction(func):
                        add(f"{obj.__module__}.{obj.__qualname__}.{attr}", func)
    return sorted(found.values(), key=lambda item: item[0])


OBJECTS = public_objects()
IDS = [name for name, _ in OBJECTS]


def literal_values(hint: object) -> list[str]:
    if typing.get_origin(hint) is typing.Literal:
        return [arg for arg in typing.get_args(hint) if isinstance(arg, str)]
    return [value for arg in typing.get_args(hint) for value in literal_values(arg)]


def test_objects_found():
    assert len(OBJECTS) > 300
    assert "arch.univariate.mean.arch_model" in IDS
    assert "arch.bootstrap.multiple_comparison.MCS" in IDS
    assert "arch.unitroot._phillips_ouliaris.phillips_ouliaris" in IDS


@pytest.mark.parametrize(("name", "obj"), OBJECTS, ids=IDS)
def test_literal_annotations_and_defaults_are_lower_case(name, obj):
    target = obj.__init__ if inspect.isclass(obj) else obj
    try:
        hints = typing.get_type_hints(target)
        signature = inspect.signature(target)
    except (NameError, TypeError, ValueError):  # pragma: no cover
        pytest.skip("The annotations cannot be resolved")
    else:
        for param, hint in hints.items():
            values = literal_values(hint)
            assert values == [value.lower() for value in values], (name, param, values)
            default = (
                signature.parameters[param].default
                if param in signature.parameters
                else 0
            )
            if values and isinstance(default, str):
                assert default == default.lower(), (name, param, default)


@pytest.mark.parametrize(("name", "obj"), OBJECTS, ids=IDS)
def test_documented_options_are_lower_case(name, obj):
    docscrape = pytest.importorskip("numpydoc.docscrape")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parameters = docscrape.NumpyDocString(inspect.getdoc(obj) or "")["Parameters"]
    quoted = re.compile(r"""["'`]{1,2}([^"'`\n]+)["'`]{1,2}""")
    for parameter in parameters:
        if not parameter.type.startswith("{"):
            continue
        options = quoted.findall(parameter.type)
        if not options:  # The values are numbers
            continue
        assert options == [option.lower() for option in options], (name, parameter)
        # The description does not show another spelling of an option
        lower = {option.lower() for option in options}
        for mention in quoted.findall(" ".join(parameter.desc)):
            if mention.lower() in lower:
                assert mention == mention.lower(), (name, parameter.name, mention)


# A callable that raises for a string that is not an option, with the name of the option
@pytest.fixture(scope="module")
def data():
    rs = np.random.default_rng(20261017)
    y = pd.Series(np.cumsum(rs.standard_normal(250)), name="y")
    x = pd.DataFrame(
        np.cumsum(rs.standard_normal((250, 2)), axis=0), columns=["a", "b"]
    )
    losses = pd.DataFrame(rs.standard_normal((60, 3)) ** 2, columns=list("abc"))
    returns = pd.Series(rs.standard_normal(400), name="ret")
    return {"y": y, "x": x, "losses": losses, "returns": returns}


def option_calls(data):
    y, x, losses, returns = data["y"], data["x"], data["losses"], data["returns"]
    res = arch_model(returns, mean="ar", lags=1).fit(disp="off")
    mod = arch_model(returns)

    def bootstrap_ci(**kw):
        return IIDBootstrap(returns.to_numpy(), seed=1).conf_int(np.mean, reps=20, **kw)

    spa = SPA(losses["a"], losses[["b", "c"]], reps=20, seed=1)
    spa.compute()
    return {
        "arch_model mean": lambda v: arch_model(returns, mean=v),
        "arch_model vol": lambda v: arch_model(returns, vol=v),
        "arch_model dist": lambda v: arch_model(returns, dist=v),
        "fit cov_type": lambda v: mod.fit(disp="off", cov_type=v),
        "fit disp": lambda v: mod.fit(disp=v),
        "forecast method": lambda v: res.forecast(method=v),
        "forecast align": lambda v: res.forecast(align=v),
        "plot annualize": lambda v: res.plot(annualize=v),
        "hedgehog plot_type": lambda v: res.hedgehog_plot(plot_type=v),
        "ARCHInMean form": lambda v: ARCHInMean(returns, volatility=GARCH(), form=v),
        "conf_int method": lambda v: bootstrap_ci(method=v),
        "conf_int tail": lambda v: bootstrap_ci(tail=v),
        "conf_int sampling": lambda v: bootstrap_ci(sampling=v),
        "MCS method": lambda v: MCS(losses, 0.1, method=v),
        "MCS bootstrap": lambda v: MCS(losses, 0.1, bootstrap=v),
        "SPA bootstrap": lambda v: SPA(losses["a"], losses[["b", "c"]], bootstrap=v),
        "better_models pvalue_type": lambda v: spa.better_models(0.5, pvalue_type=v),
        "ADF trend": lambda v: ADF(y, trend=v),
        "ADF method": lambda v: ADF(y, method=v),
        "KPSS trend": lambda v: KPSS(y, trend=v),
        "PhillipsPerron test_type": lambda v: PhillipsPerron(y, test_type=v),
        "mackinnonp dist_type": lambda v: mackinnonp(-3.0, dist_type=v),
        "mackinnoncrit dist_type": lambda v: mackinnoncrit(dist_type=v),
        "mackinnoncrit regression": lambda v: mackinnoncrit(regression=v),
        "engle_granger trend": lambda v: engle_granger(y, x, trend=v),
        "engle_granger method": lambda v: engle_granger(y, x, method=v),
        "phillips_ouliaris test_type": lambda v: phillips_ouliaris(y, x, test_type=v),
        "phillips_ouliaris kernel": lambda v: phillips_ouliaris(y, x, kernel=v),
        "phillips_ouliaris_cv test_type": lambda v: phillips_ouliaris_cv(
            v, "c", 2, 100
        ),
        "phillips_ouliaris_pval test_type": lambda v: phillips_ouliaris_pval(
            -3.0, v, "c", 2
        ),
        "DynamicOLS trend": lambda v: DynamicOLS(y, x, trend=v),
        "DynamicOLS method": lambda v: DynamicOLS(y, x, method=v),
        "DynamicOLS.fit cov_type": lambda v: DynamicOLS(y, x, lags=1, leads=1).fit(
            cov_type=v
        ),
        "get_kernel_estimator kernel": get_kernel_estimator,
        "PreWhitenedRecolored method": lambda v: PreWhitenedRecolored(
            x.to_numpy(), method=v
        ),
        "add_trend trend": lambda v: add_trend(x, trend=v),
        "add_trend has_constant": lambda v: add_trend(x, has_constant=v),
    }


def test_listed_options_are_lower_case(data):
    # The values in the message of an invalid option are all lower case
    calls = option_calls(data)
    assert len(calls) > 30
    for name, call in calls.items():
        with pytest.raises(ValueError, match="must be ") as exc:
            call("bogus")
        listed = str(exc.value).split("; got")[0]
        assert listed == listed.lower(), name


def test_mcs_method_is_lower_case(data):
    losses = data["losses"]
    expected = None
    for method in ("r", "R", "max", "MAX", "Max"):
        mcs = MCS(losses, 0.1, reps=20, block_size=5, method=method, seed=0)
        assert mcs.method == method.lower()
        mcs.compute()
        if method.lower() == "max":
            continue
        if expected is None:
            expected = mcs.pvalues["Pvalue"]
        assert_allclose(mcs.pvalues["Pvalue"], expected)
    assert MCS(losses, 0.1).method == "r"


@pytest.mark.parametrize("test_type", ["za", "zt", "pu", "pz"])
def test_phillips_ouliaris_test_type_spellings(data, test_type):
    y, x = data["y"], data["x"]
    expected = phillips_ouliaris(y, x, test_type=test_type, bandwidth=3)
    for spelling in (test_type.upper(), test_type.capitalize()):
        res = phillips_ouliaris(y, x, test_type=spelling, bandwidth=3)
        assert_allclose(res.stat, expected.stat)
        assert_allclose(res.pvalue, expected.pvalue)
        assert res.name == expected.name
    # The test is named by its statistic
    assert test_type.capitalize() in expected.name
    assert expected.trend == "c"


@pytest.mark.parametrize("annualize", ["d", "w", "m"])
def test_plot_annualize_spellings(data, annualize):
    plt = pytest.importorskip("matplotlib.pyplot")
    res = arch_model(data["returns"]).fit(disp="off")
    scale = {"d": 252, "w": 52, "m": 12}[annualize]
    for spelling in (annualize, annualize.upper()):
        fig = res.plot(annualize=spelling)
        line = fig.axes[1].lines[0]
        assert_allclose(line.get_ydata(), res.conditional_volatility * np.sqrt(scale))
        plt.close(fig)


def test_old_spellings_are_still_accepted(data):
    returns = data["returns"]
    expected = arch_model(returns, mean="ar", lags=1, vol="egarch", dist="skewt")
    old = arch_model(returns, mean="AR", lags=1, vol="EGARCH", dist="SkewT")
    assert old.name == expected.name
    assert type(old.volatility) is type(expected.volatility)
    assert type(old.distribution) is type(expected.distribution)
    assert_allclose(
        phillips_ouliaris_cv("Zt", "C", 2, 200), phillips_ouliaris_cv("zt", "c", 2, 200)
    )
    assert_allclose(
        phillips_ouliaris_pval(-3.0, "Za", "CT", 3),
        phillips_ouliaris_pval(-3.0, "za", "ct", 3),
    )
    assert get_kernel_estimator("QuadraticSpectral") is get_kernel_estimator(
        "quadratic-spectral"
    )
