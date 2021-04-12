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
