import types

from arch.univariate import recursions_python
from arch.univariate.backtest import (
    TrafficLightResult,
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_pof,
    traffic_light,
    var_exceptions,
)
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
    MAMean,
    ZeroMean,
    arch_model,
)
from arch.univariate.volatility import (
    APARCH,
    ARCH,
    EGARCH,
    FIAPARCH,
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
    "FIAPARCH",
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
    "MAMean",
    "MIDASHyperbolic",
    "Normal",
    "RiskMetrics2006",
    "SkewStudent",
    "StudentsT",
    "TrafficLightResult",
    "ZeroMean",
    "arch_model",
    "christoffersen_conditional_coverage",
    "christoffersen_independence",
    "excess_kurtosis",
    "hill_estimator",
    "kupiec_pof",
    "recursions",
    "recursions_python",
    "traffic_light",
    "var_exceptions",
    "var_ratio",
]
