import types

from arch.univariate import recursions_python
from arch.univariate.diagnostics import excess_kurtosis, hill_estimator, var_ratio
from arch.univariate.distribution import (
    Distribution,
    GeneralizedError,
    Normal,
    SkewStudent,
    StudentsT,
)
from arch.univariate.mean import (
    ARX,
    HARX,
    LS,
    ARCHInMean,
    ConstantMean,
    ZeroMean,
    arch_model,
)
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

recursions: types.ModuleType
try:
    from arch.univariate import recursions
except ImportError:
    recursions = recursions_python

__all__ = [
    "APARCH",
    "ARCH",
    "ARX",
    "EGARCH",
    "FIGARCH",
    "GARCH",
    "HARCH",
    "HARX",
    "LS",
    "ARCHInMean",
    "ConstantMean",
    "ConstantVariance",
    "Distribution",
    "EWMAVariance",
    "FixedVariance",
    "GeneralizedError",
    "MIDASHyperbolic",
    "Normal",
    "RiskMetrics2006",
    "SkewStudent",
    "StudentsT",
    "ZeroMean",
    "arch_model",
    "excess_kurtosis",
    "hill_estimator",
    "recursions",
    "recursions_python",
    "var_ratio",
]
