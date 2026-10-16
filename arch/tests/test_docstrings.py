"""
Structural checks of the public docstrings.

The docstring rules are described in ``doc/source/contributing.rst``.  The style
rules (periods, capitalization, summary wording) are checked statically by
``python -m numpydoc lint`` using ``[tool.numpydoc_validation]`` in ``pyproject.toml``.
That linter reads the source and so cannot see docstrings that are inherited when a
class is created (``DocStringInheritor``) nor Cython objects.  This module runs
``numpydoc.validate.validate`` on the live objects instead and checks the structural
codes only.

The checked objects are

* the public names in the ``__all__`` of the ``arch`` packages and modules in
  ``ALL_MODULES``,
* the public classes and functions of the modules in ``PUBLIC_MODULES`` (which have no
  ``__all__``) and the docstrings of the ``arch.data`` modules, and
* every object named in an ``autosummary``, ``autoclass``, ``autofunction``,
  ``autoexception`` or ``autodata`` directive in ``doc/source`` (when the documentation
  is available).

Classes are expanded to their public methods and properties, as the autosummary class
template does.  Each object is validated once, under the name of the class that
defines it.

``KNOWN_FAILURES`` lists the failures that exist today.  A test fails when an object
has a failure that is not listed (a new problem) and when a listed failure no longer
occurs (remove it from the mapping).  The mapping can only shrink.  Regenerate it with
``python -m arch.tests.test_docstrings`` after fixing docstrings.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import re
import sys
import warnings

import pytest

import arch

# The codes reported by validate are those of numpydoc 1.10 and 1.11
pytest.importorskip("numpydoc", minversion="1.10")
numpydoc_validate = importlib.import_module("numpydoc.validate")
validate = numpydoc_validate.validate

# The structural codes.  GL08 (no docstring) is only reported when there is no
# docstring anywhere in the method resolution order (see ``structural_codes``).
#
# GL05 tabs, GL06 unknown section, GL07 sections out of order, GL08 no docstring,
# GL09 deprecation directive placement, GL10 unknown reST directive, SS01 no summary,
# PR01 parameters not documented, PR02 unknown parameters, PR03 wrong parameter order,
# PR04 parameter without a type, PR05 type ends with a period, PR06 "str" not "string"
# (and similar), PR07 parameter without a description, PR10 no space before the colon,
# RT01 no Returns section, RT03 return without a description, YD01 no Yields section
STRUCTURAL_CODES = frozenset(
    {
        "GL05",
        "GL06",
        "GL07",
        "GL08",
        "GL09",
        "GL10",
        "SS01",
        "PR01",
        "PR02",
        "PR03",
        "PR04",
        "PR05",
        "PR06",
        "PR07",
        "PR10",
        "RT01",
        "RT03",
        "YD01",
    }
)

ALL_MODULES = (
    "arch",
    "arch.univariate",
    "arch.univariate.base",
    "arch.univariate.diagnostics",
    "arch.univariate.volatility",
    "arch.univariate.distribution",
    "arch.univariate.mean",
    "arch.bootstrap",
    "arch.bootstrap.base",
    "arch.bootstrap.multiple_comparison",
    "arch.unitroot",
    "arch.unitroot.unitroot",
    "arch.unitroot.cointegration",
    "arch.covariance.kernel",
    "arch.covariance.var",
    "arch.utility",
    "arch.utility.testing",
    "arch.utility.array",
)
PUBLIC_MODULES = (
    "arch.utility.cov",
    "arch.utility.io",
    "arch.utility.timeseries",
    "arch.utility.exceptions",
    "arch.data.utility",
)
DATA_MODULES = (
    "arch.data.binary",
    "arch.data.core_cpi",
    "arch.data.crude",
    "arch.data.default",
    "arch.data.frenchdata",
    "arch.data.nasdaq",
    "arch.data.sp500",
    "arch.data.vix",
    "arch.data.wti",
)

DOC_SOURCE = Path(arch.__file__).parent.parent / "doc" / "source"
_DIRECTIVE = re.compile(r"\.\. auto(?:class|function|exception|data)::\s*(\S+)")
_MODULE = re.compile(r"\.\. (?:py:)?(?:current)?module::\s*(\S+)")
_NAME = re.compile(r"^~?[A-Za-z_][\w.]*$")


def resolve(full_name: str) -> object | None:
    parts = full_name.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        try:
            for part in parts[i:]:
                obj = getattr(obj, part)
        except AttributeError:
            return None
        return obj
    return None


def documented_names() -> list[str]:
    """Names used by the autodoc/autosummary directives in doc/source."""
    names: list[str] = []
    if not DOC_SOURCE.exists():
        return names
    for rst in sorted(DOC_SOURCE.rglob("*.rst")):
        rel = rst.relative_to(DOC_SOURCE).as_posix()
        if rel.startswith(("changes", "_templates")):
            continue
        module = None
        in_autosummary = False
        for line in rst.read_text(encoding="utf-8").splitlines():
            if match := _MODULE.match(line):
                module = match.group(1)
                in_autosummary = False
            elif match := _DIRECTIVE.match(line):
                name = match.group(1).lstrip("~")
                names.append(f"{module}.{name}" if module and "." not in name else name)
                in_autosummary = False
            elif line.startswith(".. autosummary::"):
                in_autosummary = True
            elif in_autosummary:
                entry = line.strip()
                if not entry or entry.startswith(":"):
                    continue
                if line.startswith("   ") and _NAME.match(entry):
                    entry = entry.lstrip("~")
                    if entry.startswith("arch.") or not module:
                        names.append(entry)
                    else:
                        names.append(f"{module}.{entry}")
                elif not line.startswith(" "):
                    in_autosummary = False
    return names


def defining_name(obj: object, fallback: str) -> str:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if module and qualname and "<" not in qualname:
        return f"{module}.{qualname}"
    return fallback


def collect() -> list[str]:
    """Qualified names of the objects to validate, sorted and unique."""
    found: set[str] = set()

    def add_class_members(cls: type) -> None:
        for attr in dir(cls):
            if attr.startswith("_"):
                continue
            owner = next((k for k in cls.__mro__ if attr in vars(k)), None)
            if owner is None or not owner.__module__.startswith("arch"):
                continue
            raw = vars(owner)[attr]
            if isinstance(raw, staticmethod | classmethod):
                raw = raw.__func__
            if isinstance(raw, property) or inspect.isroutine(raw):
                found.add(f"{owner.__module__}.{owner.__qualname__}.{attr}")

    def add(full_name: str) -> None:
        obj = resolve(full_name)
        if obj is None or inspect.ismodule(obj):
            return
        if inspect.isclass(obj):
            name = defining_name(obj, full_name)
            if name.startswith("arch."):
                found.add(name)
                add_class_members(obj)
        elif inspect.isroutine(obj):
            name = defining_name(obj, full_name)
            if name.startswith("arch."):
                found.add(name)
        # Anything else is a type alias or a constant, which has no docstring to check

    for module_name in ALL_MODULES:
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", ()):
            add(f"{module_name}.{name}")
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        for name, obj in vars(module).items():
            is_arch_object = inspect.isclass(obj) or inspect.isroutine(obj)
            if not name.startswith("_") and is_arch_object:
                if getattr(obj, "__module__", None) == module_name:
                    add(f"{module_name}.{name}")
    for name in documented_names():
        add(name)
    # The docstring of each data module
    found.update(DATA_MODULES)
    return sorted(found)


def structural_codes(name: str) -> frozenset[str]:
    """Structural numpydoc codes reported by ``validate`` for one object."""
    with warnings.catch_warnings():
        # numpydoc warns about malformed section underlines
        warnings.simplefilter("ignore")
        codes = {code for code, _ in validate(name)["errors"]}
        codes &= STRUCTURAL_CODES
        if "GL08" in codes:
            # Plain inheritance of a docstring (autodoc does the same) is not a
            # missing docstring
            obj = numpydoc_validate.Validator._load_obj(name)
            if inspect.getdoc(obj):
                codes.discard("GL08")
    return frozenset(codes)


OBJECTS = collect()

# BEGIN KNOWN_FAILURES (generated by ``python -m arch.tests.test_docstrings``)
KNOWN_FAILURES: dict[str, frozenset[str]] = {
    "arch.bootstrap.base.CircularBlockBootstrap": frozenset(
        {"GL07", "PR01", "PR02", "PR04"}
    ),
    "arch.bootstrap.base.CircularBlockBootstrap.clone": frozenset(
        {"PR01", "PR02", "PR04"}
    ),
    "arch.bootstrap.base.CircularBlockBootstrap.update_indices": frozenset({"RT01"}),
    "arch.bootstrap.base.IIDBootstrap": frozenset(
        {"GL06", "GL07", "PR01", "PR02", "PR04"}
    ),
    "arch.bootstrap.base.IIDBootstrap.bootstrap": frozenset({"GL07", "YD01"}),
    "arch.bootstrap.base.IIDBootstrap.clone": frozenset({"PR01", "PR02", "PR04"}),
    "arch.bootstrap.base.IIDBootstrap.conf_int": frozenset({"GL07", "PR03", "SS01"}),
    "arch.bootstrap.base.IIDBootstrap.update_indices": frozenset({"RT01"}),
    "arch.bootstrap.base.IIDBootstrap.var": frozenset({"PR01", "PR02", "PR04", "PR10"}),
    "arch.bootstrap.base.IndependentSamplesBootstrap": frozenset(
        {"GL06", "GL07", "PR01", "PR02", "PR04"}
    ),
    "arch.bootstrap.base.IndependentSamplesBootstrap.update_indices": frozenset(
        {"RT01"}
    ),
    "arch.bootstrap.base.MovingBlockBootstrap": frozenset(
        {"GL07", "PR01", "PR02", "PR04"}
    ),
    "arch.bootstrap.base.MovingBlockBootstrap.update_indices": frozenset({"RT01"}),
    "arch.bootstrap.base.StationaryBootstrap": frozenset(
        {"GL07", "PR01", "PR02", "PR04"}
    ),
    "arch.bootstrap.base.StationaryBootstrap.update_indices": frozenset({"RT01"}),
    "arch.bootstrap.multiple_comparison.RealityCheck": frozenset({"GL07"}),
    "arch.bootstrap.multiple_comparison.SPA": frozenset({"GL07"}),
    "arch.bootstrap.multiple_comparison.StepM": frozenset({"GL07"}),
    "arch.doc": frozenset({"GL08"}),
    "arch.unitroot._engle_granger.EngleGrangerTestResults.summary": frozenset({"RT01"}),
    "arch.unitroot._engle_granger.engle_granger": frozenset(
        {"PR01", "PR02", "PR04", "PR10"}
    ),
    "arch.unitroot._phillips_ouliaris.PhillipsOuliarisTestResults.summary": frozenset(
        {"RT01"}
    ),
    "arch.unitroot.cointegration.CanonicalCointegratingReg.fit": frozenset({"PR03"}),
    "arch.unitroot.cointegration.DynamicOLSResults": frozenset({"PR01"}),
    "arch.unitroot.cointegration.FullyModifiedOLS.fit": frozenset({"PR03"}),
    "arch.unitroot.unitroot.ADF": frozenset({"GL07"}),
    "arch.unitroot.unitroot.DFGLS": frozenset({"GL07", "PR01"}),
    "arch.unitroot.unitroot.KPSS": frozenset({"GL07"}),
    "arch.unitroot.unitroot.PhillipsPerron": frozenset({"GL07"}),
    "arch.unitroot.unitroot.UnitRootTest.summary": frozenset({"RT01"}),
    "arch.unitroot.unitroot.VarianceRatio": frozenset({"GL07", "PR03"}),
    "arch.unitroot.unitroot.mackinnonp": frozenset({"GL07"}),
    "arch.univariate.base.ARCHModel": frozenset({"PR01"}),
    "arch.univariate.base.ARCHModel.compute_param_cov": frozenset({"RT01"}),
    "arch.univariate.base.ARCHModel.forecast": frozenset({"GL07"}),
    "arch.univariate.base.ARCHModel.simulate": frozenset({"GL08"}),
    "arch.univariate.base.ARCHModelForecast": frozenset({"PR01", "PR07"}),
    "arch.univariate.base.ARCHModelForecastSimulation": frozenset({"PR04", "PR07"}),
    "arch.univariate.base.format_float_fixed": frozenset({"PR01", "RT01"}),
    "arch.univariate.distribution.Distribution": frozenset({"PR01"}),
    "arch.univariate.distribution.Distribution.partial_moment": frozenset({"GL07"}),
    "arch.univariate.distribution.GeneralizedError.loglikelihood": frozenset(
        {"PR01", "PR02"}
    ),
    "arch.univariate.distribution.GeneralizedError.partial_moment": frozenset({"GL07"}),
    "arch.univariate.distribution.Normal.loglikelihood": frozenset({"PR01", "PR02"}),
    "arch.univariate.distribution.Normal.partial_moment": frozenset({"GL07"}),
    "arch.univariate.distribution.SkewStudent.loglikelihood": frozenset(
        {"PR01", "PR02"}
    ),
    "arch.univariate.distribution.SkewStudent.partial_moment": frozenset({"GL07"}),
    "arch.univariate.distribution.StudentsT.loglikelihood": frozenset({"PR01", "PR02"}),
    "arch.univariate.distribution.StudentsT.partial_moment": frozenset({"GL07"}),
    "arch.univariate.mean.ARCHInMean": frozenset({"GL07"}),
    "arch.univariate.mean.ARCHInMean.forecast": frozenset({"GL07"}),
    "arch.univariate.mean.ARX": frozenset({"GL07"}),
    "arch.univariate.mean.ConstantMean": frozenset({"GL07"}),
    "arch.univariate.mean.HARX": frozenset({"GL07"}),
    "arch.univariate.mean.HARX.forecast": frozenset({"GL07"}),
    "arch.univariate.mean.LS": frozenset({"GL07"}),
    "arch.univariate.mean.MAMean.forecast": frozenset({"GL07"}),
    "arch.univariate.mean.ZeroMean": frozenset({"GL07"}),
    "arch.univariate.mean.arch_model": frozenset({"GL07"}),
    "arch.univariate.recursions.VolatilityUpdater.initialize_update": frozenset(
        {"GL08"}
    ),
    "arch.univariate.volatility.APARCH": frozenset({"GL07"}),
    "arch.univariate.volatility.APARCH.compute_variance": frozenset({"RT01"}),
    "arch.univariate.volatility.ARCH": frozenset({"GL07"}),
    "arch.univariate.volatility.BootstrapRng.random_state": frozenset({"GL08"}),
    "arch.univariate.volatility.BootstrapRng.rng": frozenset({"GL08"}),
    "arch.univariate.volatility.ConstantVariance.compute_variance": frozenset({"RT01"}),
    "arch.univariate.volatility.EGARCH": frozenset({"GL07"}),
    "arch.univariate.volatility.EGARCH.compute_variance": frozenset({"RT01"}),
    "arch.univariate.volatility.EWMAVariance": frozenset({"GL07"}),
    "arch.univariate.volatility.EWMAVariance.compute_variance": frozenset({"RT01"}),
    "arch.univariate.volatility.FIAPARCH": frozenset({"GL07"}),
    "arch.univariate.volatility.FIAPARCH.compute_variance": frozenset({"RT01"}),
    "arch.univariate.volatility.FIGARCH": frozenset({"GL07"}),
    "arch.univariate.volatility.FIGARCH.compute_variance": frozenset({"RT01"}),
    "arch.univariate.volatility.FixedVariance.compute_variance": frozenset({"RT01"}),
    "arch.univariate.volatility.GARCH": frozenset({"GL07"}),
    "arch.univariate.volatility.GARCH.compute_variance": frozenset({"RT01"}),
    "arch.univariate.volatility.HARCH": frozenset({"GL07"}),
    "arch.univariate.volatility.HARCH.compute_variance": frozenset({"RT01"}),
    "arch.univariate.volatility.MIDASHyperbolic": frozenset({"GL07"}),
    "arch.univariate.volatility.MIDASHyperbolic.compute_variance": frozenset({"RT01"}),
    "arch.univariate.volatility.MIDASHyperbolic.constraints": frozenset({"RT01"}),
    "arch.univariate.volatility.RiskMetrics2006": frozenset({"GL07"}),
    "arch.univariate.volatility.RiskMetrics2006.compute_variance": frozenset({"RT01"}),
    "arch.univariate.volatility.VolatilityProcess.update": frozenset({"PR04"}),
    "arch.utility.array.DocStringInheritor": frozenset({"PR01"}),
    "arch.utility.array.ensure1d": frozenset({"GL08"}),
    "arch.utility.array.ensure2d": frozenset({"GL08"}),
    "arch.utility.array.parse_dataframe": frozenset({"GL08"}),
    "arch.utility.io.pval_format": frozenset({"PR01", "RT01"}),
    "arch.utility.io.str_format": frozenset({"PR01", "RT01"}),
    "arch.utility.test": frozenset({"PR01", "RT01"}),
    "arch.utility.testing.WaldTestStatistic.alternative": frozenset({"GL08"}),
    "arch.utility.timeseries.add_trend": frozenset({"RT01"}),
}
# END KNOWN_FAILURES


@pytest.mark.parametrize("name", OBJECTS, ids=OBJECTS)
def test_docstring_structure(name):
    found = structural_codes(name)
    known = KNOWN_FAILURES.get(name, frozenset())
    new = sorted(found - known)
    fixed = sorted(known - found)
    messages = []
    if new:
        report = {c: m for c, m in validate(name)["errors"] if c in new}
        messages.append(f"New docstring failures: {report}")
    if fixed:
        messages.append(
            f"Fixed, remove from KNOWN_FAILURES in {Path(__file__).name}: {fixed}"
        )
    assert not messages, f"{name}: " + "; ".join(messages)


def test_known_failures_are_current_objects():
    if not DOC_SOURCE.exists():
        pytest.skip("The documentation source is not available")
    unknown = sorted(set(KNOWN_FAILURES) - set(OBJECTS))
    assert not unknown, f"Not a checked object, remove from KNOWN_FAILURES: {unknown}"


def test_objects_found():
    # Guard against the collection silently becoming empty or shrinking
    assert len(OBJECTS) > 400
    assert "arch.univariate.base.ARCHModel.fit" in OBJECTS
    assert "arch.bootstrap.base.IIDBootstrap.conf_int" in OBJECTS


def render_known_failures(failures: dict[str, frozenset[str]]) -> str:
    lines = ["KNOWN_FAILURES: dict[str, frozenset[str]] = {"]
    for name in sorted(failures):
        codes = ", ".join(f'"{code}"' for code in sorted(failures[name]))
        entry = f'    "{name}": frozenset({{{codes}}}),'
        if len(entry) > 88:
            entry = f'    "{name}": frozenset(\n        {{{codes}}}\n    ),'
        lines.append(entry)
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    current = {name: structural_codes(name) for name in OBJECTS}
    current = {name: codes for name, codes in current.items() if codes}
    sys.stdout.write(render_known_failures(current) + "\n")
