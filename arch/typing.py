from collections.abc import Hashable
import datetime as dt
from typing import Any, Callable, Literal, Optional, TypeVar, Union

import numpy as np
from pandas import DataFrame, Series, Timestamp
from numpy.typing import NDArray

__all__ = [
    "NDArray",
    "ArrayLike",
    "NDArrayOrFrame",
    "AnyPandas",
    "DateLike",
    "ArrayLike1D",
    "ArrayLike2D",
    "Label",
    "FloatOrArray",
    "RNGType",
    "Literal",
    "UnitRootTrend",
    "ForecastingMethod",
    "Float64Array",
    "Int64Array",
    "Int32Array",
    "BoolArray",
    "AnyArray",
    "IntArray",
    "RandomStateState",
    "Uint32Array",
    "BootstrapIndexT",
]

Float64Array = NDArray[np.float64]           # For float64 (or double)
Int64Array = NDArray[np.int64]               # For int64 (np.longlong)
Int32Array = NDArray[np.int32]               # For int32 (np.intc)
IntArray = NDArray[np.int_]                  # For platform-dependent int
BoolArray = NDArray[np.bool_]                # For boolean arrays
AnyArray = NDArray[Any]                      # For arrays with any dtype and shape
Uint32Array = NDArray[np.uint32]             # For unsigned 32-bit int (np.uintc)

BootstrapIndexT = Union[
    Int64Array, tuple[Int64Array, ...], tuple[list[Int64Array], dict[str, Int64Array]]
]
RandomStateState = tuple[str, Uint32Array, int, int, float]

RNGType = Callable[[Union[int, tuple[int, ...]]], Float64Array]
ArrayLike1D = Union[NDArray, Series]
ArrayLike2D = Union[NDArray, DataFrame]
ArrayLike = Union[NDArray, DataFrame, Series]
NDArrayOrFrame = TypeVar("NDArrayOrFrame", Float64Array, DataFrame)
AnyPandas = Union[Series, DataFrame]
DateLike = Union[str, dt.datetime, np.datetime64, Timestamp]
Label = Optional[Hashable]
FloatOrArray = TypeVar("FloatOrArray", float, np.ndarray)
UnitRootTrend = Literal["n", "c", "ct", "ctt"]
ForecastingMethod = Literal["analytic", "simulation", "bootstrap"]
