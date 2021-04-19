from typing import Optional, Union

import numpy as np

from arch.typing import NDArray

def harch_recursion(
    parameters: NDArray,
    resids: NDArray,
    sigma2: NDArray,
    lags: NDArray,
    nobs: int,
    backcast: float,
    var_bounds: NDArray,
) -> NDArray: ...
def arch_recursion(
    parameters: NDArray,
    resids: NDArray,
    sigma2: NDArray,
    p: int,
    nobs: int,
    backcast: float,
    var_bounds: NDArray,
) -> NDArray: ...
def garch_recursion(
    parameters: NDArray,
    fresids: NDArray,
    sresids: NDArray,
    sigma2: NDArray,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: NDArray,
) -> NDArray: ...
def egarch_recursion(
    parameters: NDArray,
    resids: NDArray,
    sigma2: NDArray,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: NDArray,
    lnsigma2: NDArray,
    std_resids: NDArray,
    abs_std_resids: NDArray,
) -> NDArray: ...
def midas_recursion(
    parameters: NDArray,
    weights: NDArray,
    resids: NDArray,
    sigma2: NDArray,
    nobs: int,
    backcast: float,
    var_bounds: NDArray,
) -> NDArray: ...
def figarch_weights(parameters: NDArray, p: int, q: int, trunc_lag: int) -> NDArray: ...
def figarch_recursion(
    parameters: NDArray,
    fresids: NDArray,
    sigma2: NDArray,
    p: int,
    q: int,
    nobs: int,
    trunc_lag: int,
    backcast: float,
    var_bounds: NDArray,
) -> NDArray: ...
def aparch_recursion(
    parameters: NDArray,
    resids: NDArray,
    abs_resids: NDArray,
    sigma2: NDArray,
    sigma_delta: NDArray,
    p: int,
    o: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: NDArray,
) -> NDArray: ...
def harch_core(
    t: int,
    parameters: NDArray,
    resids: NDArray,
    sigma2: NDArray,
    lags: NDArray,
    backcast: float,
    var_bounds: NDArray,
) -> NDArray: ...
def garch_core(
    t: int,
    parameters: NDArray,
    resids: NDArray,
    sigma2: NDArray,
    backcast: float,
    var_bounds: NDArray,
    p: int,
    o: int,
    q: int,
    power: float,
) -> NDArray: ...

class VolatiltyUpdater:
    def initialize_update(
        self, parameters: NDArray, backcast: Union[float, NDArray], nobs: int
    ) -> None: ...
    def _update_tester(
        self,
        t: int,
        parameters: NDArray,
        resids: NDArray,
        sigma2: NDArray,
        var_bounds: NDArray,
    ) -> None: ...

class GARCHUpdater(VolatiltyUpdater):
    def __init__(self, p: int, o: int, q: int, power: float) -> None: ...

class EWMAUpdater(VolatiltyUpdater):
    def __init__(self, lam: Optional[float]) -> None: ...

class FIGARCHUpdater(VolatiltyUpdater):
    def __init__(self, p: int, q: int, power: float, truncation: int) -> None: ...

class HARCHUpdater(VolatiltyUpdater):
    def __init__(self, lags: NDArray) -> None: ...

class MIDASUpdater(VolatiltyUpdater):
    def __init__(self, m: int, asym: bool) -> None: ...

class ARCHInMeanRecursion:
    def __init__(self, updater: VolatiltyUpdater) -> None: ...
    def recursion(
        self,
        y: NDArray,
        x: NDArray,
        mean_parameters: NDArray,
        variance_params: NDArray,
        sigma2: NDArray,
        backcast: float,
        var_bounds: NDArray,
        power: float,
    ) -> NDArray: ...
