from typing import Optional, Union

from arch.typing import Float64Array, Int32Array

def harch_recursion(
    parameters: Float64Array,
    resids: Float64Array,
    sigma2: Float64Array,
    lags: Int32Array,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array: ...
def arch_recursion(
    parameters: Float64Array,
    resids: Float64Array,
    sigma2: Float64Array,
    p: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array: ...
def garch_recursion(
    parameters: Float64Array,
    fresids: Float64Array,
    sresids: Float64Array,
    sigma2: Float64Array,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array: ...
def egarch_recursion(
    parameters: Float64Array,
    resids: Float64Array,
    sigma2: Float64Array,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array,
    lnsigma2: Float64Array,
    std_resids: Float64Array,
    abs_std_resids: Float64Array,
) -> Float64Array: ...
def midas_recursion(
    parameters: Float64Array,
    weights: Float64Array,
    resids: Float64Array,
    sigma2: Float64Array,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array: ...
def figarch_weights(
    parameters: Float64Array, p: int, q: int, trunc_lag: int
) -> Float64Array: ...
def figarch_recursion(
    parameters: Float64Array,
    fresids: Float64Array,
    sigma2: Float64Array,
    p: int,
    q: int,
    nobs: int,
    trunc_lag: int,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array: ...
def aparch_recursion(
    parameters: Float64Array,
    resids: Float64Array,
    abs_resids: Float64Array,
    sigma2: Float64Array,
    sigma_delta: Float64Array,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array: ...
def harch_core(
    t: int,
    parameters: Float64Array,
    resids: Float64Array,
    sigma2: Float64Array,
    lags: Int32Array,
    backcast: float,
    var_bounds: Float64Array,
) -> Float64Array: ...
def garch_core(
    t: int,
    parameters: Float64Array,
    resids: Float64Array,
    sigma2: Float64Array,
    backcast: float,
    var_bounds: Float64Array,
    p: int,
    o: int,
    q: int,
    power: float,
) -> Float64Array: ...

class VolatilityUpdater:
    def initialize_update(
        self, parameters: Float64Array, backcast: Union[float, Float64Array], nobs: int
    ) -> None: ...
    def _update_tester(
        self,
        t: int,
        parameters: Float64Array,
        resids: Float64Array,
        sigma2: Float64Array,
        var_bounds: Float64Array,
    ) -> None: ...

class GARCHUpdater(VolatilityUpdater):
    def __init__(self, p: int, o: int, q: int, power: float) -> None: ...

class EWMAUpdater(VolatilityUpdater):
    def __init__(self, lam: Optional[float]) -> None: ...

class FIGARCHUpdater(VolatilityUpdater):
    def __init__(self, p: int, q: int, power: float, truncation: int) -> None: ...

class HARCHUpdater(VolatilityUpdater):
    def __init__(self, lags: Int32Array) -> None: ...

class MIDASUpdater(VolatilityUpdater):
    def __init__(self, m: int, asym: bool) -> None: ...

class RiskMetrics2006Updater(VolatilityUpdater):
    def __init__(
        self,
        kmax: int,
        combination_weights: Float64Array,
        smoothing_parameters: Float64Array,
    ) -> None: ...

class ARCHInMeanRecursion:
    def __init__(self, updater: VolatilityUpdater) -> None: ...
    def recursion(
        self,
        y: Float64Array,
        x: Float64Array,
        mean_parameters: Float64Array,
        variance_params: Float64Array,
        sigma2: Float64Array,
        var_bounds: Float64Array,
        power: float,
    ) -> Float64Array: ...
