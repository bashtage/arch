from typing import Optional, Union

from arch._typing import Float64Array, Float64Array1D, Float64Array2D, Int32Array

def harch_recursion(
    parameters: Float64Array1D,
    resids: Float64Array1D,
    sigma2: Float64Array1D,
    lags: Int32Array,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
) -> Float64Array: ...
def arch_recursion(
    parameters: Float64Array1D,
    resids: Float64Array1D,
    sigma2: Float64Array1D,
    p: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
) -> Float64Array: ...
def garch_recursion(
    parameters: Float64Array1D,
    fresids: Float64Array1D,
    sresids: Float64Array1D,
    sigma2: Float64Array1D,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
) -> Float64Array: ...
def egarch_recursion(
    parameters: Float64Array1D,
    resids: Float64Array1D,
    sigma2: Float64Array1D,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
    lnsigma2: Float64Array1D,
    std_resids: Float64Array1D,
    abs_std_resids: Float64Array1D,
) -> Float64Array: ...
def midas_recursion(
    parameters: Float64Array1D,
    weights: Float64Array1D,
    resids: Float64Array1D,
    sigma2: Float64Array1D,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
) -> Float64Array: ...
def figarch_weights(
    parameters: Float64Array1D, p: int, q: int, trunc_lag: int
) -> Float64Array: ...
def figarch_recursion(
    parameters: Float64Array1D,
    fresids: Float64Array1D,
    sigma2: Float64Array1D,
    p: int,
    q: int,
    nobs: int,
    trunc_lag: int,
    backcast: float,
    var_bounds: Float64Array2D,
) -> Float64Array: ...
def aparch_recursion(
    parameters: Float64Array1D,
    resids: Float64Array1D,
    abs_resids: Float64Array1D,
    sigma2: Float64Array1D,
    sigma_delta: Float64Array1D,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
) -> Float64Array: ...
def harch_core(
    t: int,
    parameters: Float64Array1D,
    resids: Float64Array1D,
    sigma2: Float64Array1D,
    lags: Int32Array,
    backcast: float,
    var_bounds: Float64Array2D,
) -> Float64Array: ...
def garch_core(
    t: int,
    parameters: Float64Array1D,
    resids: Float64Array1D,
    sigma2: Float64Array1D,
    backcast: float,
    var_bounds: Float64Array2D,
    p: int,
    o: int,
    q: int,
    power: float,
) -> Float64Array: ...

class VolatilityUpdater:
    def initialize_update(
        self,
        parameters: Float64Array1D,
        backcast: Union[float, Float64Array1D],
        nobs: int,
    ) -> None: ...
    def _update_tester(
        self,
        t: int,
        parameters: Float64Array1D,
        resids: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
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
        combination_weights: Float64Array1D,
        smoothing_parameters: Float64Array1D,
    ) -> None: ...

class ARCHInMeanRecursion:
    def __init__(self, updater: VolatilityUpdater) -> None: ...
    def recursion(
        self,
        y: Float64Array1D,
        x: Float64Array2D,
        mean_parameters: Float64Array1D,
        variance_params: Float64Array1D,
        sigma2: Float64Array1D,
        var_bounds: Float64Array2D,
        power: float,
    ) -> Float64Array: ...
